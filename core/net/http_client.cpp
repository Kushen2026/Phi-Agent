#define SECURITY_WIN32
#include "net/http_client.hpp"
#include "util/trace.hpp"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <security.h>
#include <schannel.h>
#include <wincrypt.h>
// SCH_IGNORE_REVOCATION is not defined in older mingw-w64 schannel.h
#ifndef SCH_IGNORE_REVOCATION
#define SCH_IGNORE_REVOCATION 0x00000004
#endif
// CERT_CHAIN_REVOCATION_CHECK_DISABLED is not defined in this mingw; use CHAIN instead
#ifndef CERT_CHAIN_REVOCATION_CHECK_DISABLED
#define CERT_CHAIN_REVOCATION_CHECK_DISABLED CERT_CHAIN_REVOCATION_CHECK_CHAIN
#endif

#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "util/base.hpp"

namespace phi {

namespace {

// ── connection pool for HTTP reuse ─────────────────────────────────────────
//
// The pool is disabled (see acquire_pooled / release_pooled): the previous
// implementation neither matched host/port nor honoured the `Connection: close`
// header, so a recycled socket almost always failed on reuse. Only the two
// entry points are kept and they do the right thing directly — there is no
// backing storage to age out, so no idle-sweeper either.

void ensure_wsa() {
	static std::once_flag once;
	std::call_once(once, [] {
		WSADATA wsa;
		WSAStartup(MAKEWORD(2, 2), &wsa);
	});
}

SOCKET acquire_pooled(const std::string& /*host*/, int /*port*/) {
	// Pool disabled: see release_pooled. The prior implementation neither matched
	// host/port nor respected the Connection: close header on every request, so
	// returning pooled sockets almost certainly failed on reuse. Better to close
	// explicitly than to hand back a dead socket.
	return INVALID_SOCKET;
}

void release_pooled(SOCKET sock) {
	// Pool disabled: close immediately instead of recycling a Connection: close
	// socket into a pool that was never functional.
	if (sock != INVALID_SOCKET) closesocket(sock);
}

struct UrlParts {
	std::string host;
	std::string path;
	bool https = true;
	int port = 443;
};

bool split_url(const std::string& url, UrlParts& out) {
	std::string rest = url;
	if (starts_with(rest, "https://")) {
		out.https = true;
		rest = rest.substr(8);
	} else if (starts_with(rest, "http://")) {
		out.https = false;
		out.port = 80;
		rest = rest.substr(7);
	} else {
		return false;
	}
	size_t slash = rest.find('/');
	std::string hostport = slash == std::string::npos ? rest : rest.substr(0, slash);
	out.path = slash == std::string::npos ? "/" : rest.substr(slash);

	if (!hostport.empty() && hostport.front() == '[') {
		// IPv6: [::1]:8080 or [::1]
		size_t bracket = hostport.find(']');
		if (bracket == std::string::npos) return false;
		out.host = hostport.substr(1, bracket - 1);   // strip brackets
		if (bracket + 1 < hostport.size() && hostport[bracket + 1] == ':') {
			out.port = atoi(hostport.c_str() + bracket + 2);
		} else if (!out.https) {
			out.port = 80;
		} else {
			out.port = 443;
		}
	} else {
		// IPv4 or hostname
		size_t colon = hostport.rfind(':');
		if (colon != std::string::npos && colon > 0) {
			out.port = atoi(hostport.c_str() + colon + 1);
			out.host = hostport.substr(0, colon);
		} else {
			out.host = hostport;
			if (out.https) out.port = 443;
			else out.port = 80;
		}
	}
	return !out.host.empty() && out.port > 0;
}

// non-blocking connect with a hard timeout (aligns with the old WinHTTP 30s);
// the wait is sliced so `cancelled` (agent abort) is honored mid-connect —
// a single 30s select() made the pause button look dead during black-holed
// connects
timeval connect_wait_slice() {
	timeval tv{0, 250000};  // 250ms
	return tv;
}

SOCKET tcp_connect(const std::string& host, int port, std::string& err,
	const std::function<bool()>& cancelled = {}) {
	addrinfo hints{};
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	addrinfo* list = nullptr;
	char port_str[16];
	snprintf(port_str, sizeof(port_str), "%d", port);
	if (getaddrinfo(host.c_str(), port_str, &hints, &list) != 0 || !list) {
		err = "resolve failed: " + host;
		return INVALID_SOCKET;
	}
	SOCKET sock = INVALID_SOCKET;
	for (addrinfo* ai = list; ai && sock == INVALID_SOCKET; ai = ai->ai_next) {
		sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
		if (sock == INVALID_SOCKET) continue;

		u_long nonblock = 1;
		ioctlsocket(sock, FIONBIO, &nonblock);
		if (connect(sock, ai->ai_addr, (int)ai->ai_addrlen) == 0) {
			nonblock = 0;
			ioctlsocket(sock, FIONBIO, &nonblock);
			break;
		}
		if (WSAGetLastError() != WSAEWOULDBLOCK) {
			closesocket(sock);
			sock = INVALID_SOCKET;
			continue;
		}
		// wait for the connect to complete in 250ms slices, polling `cancelled`
		// between slices (30s hard cap — same budget as the old single select)
		bool connected = false;
		bool cancelled_now = false;
		for (int slice = 0; slice < 120 && !connected && !cancelled_now; slice++) {
			if (cancelled && cancelled()) {
				cancelled_now = true;
				break;
			}
			fd_set w, e;
			FD_ZERO(&w);
			FD_SET(sock, &w);
			FD_ZERO(&e);
			FD_SET(sock, &e);
			timeval tv = connect_wait_slice();
			int sel = select(0, nullptr, &w, &e, &tv);
			if (sel > 0) {
				int so_err = 0, len = sizeof(so_err);
				getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&so_err, &len);
				if (so_err == 0) {
					nonblock = 0;
					ioctlsocket(sock, FIONBIO, &nonblock);
					connected = true;
					break;
				}
				break;  // connect failed on this address
			}
			// sel == 0: slice elapsed, keep waiting (unless aborted above)
		}
		if (connected) break;
		if (cancelled_now && err.empty()) err = "cancelled";
		closesocket(sock);
		sock = INVALID_SOCKET;
	}
	freeaddrinfo(list);
	if (sock == INVALID_SOCKET && err.empty()) err = "connection failed: " + host;
	return sock;
}

bool send_all(SOCKET s, const char* data, size_t len, std::string& err,
	const std::function<bool()>* cancelled = nullptr) {
	size_t off = 0;
	while (off < len) {
		// honor abort during large uploads (a big context body can take many
		// SO_SNDTIMEO windows to drain into a slow peer); poll between sends
		if (cancelled && (*cancelled)()) {
			err = "cancelled";
			return false;
		}
		int n = send(s, data + off, (int)std::min(len - off, (size_t)0x40000000), 0);
		if (n == SOCKET_ERROR) {
			err = "send failed";
			return false;
		}
		off += (size_t)n;
	}
	return true;
}

// one recv with the socket's timeout; >0 bytes, 0 = clean eof, -1 = timeout slice.
// fatal_err (optional): set when the socket died (reset/abort) — the old code
// mapped every fatal error to a CLEAN eof, so a connection reset mid-stream
// made truncated LLM responses look complete
int recv_slice(SOCKET s, char* buf, size_t cap, std::string* fatal_err = nullptr) {
	int n = recv(s, buf, (int)cap, 0);
	if (n > 0) return n;
	if (n == 0) return 0;
	int e = WSAGetLastError();
	if (e == WSAETIMEDOUT || e == WSAEWOULDBLOCK) return -1;
	if (fatal_err) {
		*fatal_err = std::string("connection lost (WSAError ") + std::to_string(e) + ")";
	}
	return 0;
}

// ── schannel TLS ───────────────────────────────────────────────────────────

struct Tls {
	CredHandle cred{};
	CtxtHandle ctx{};
	bool have_cred = false;
	bool have_ctx = false;
	SecPkgContext_StreamSizes sizes{};
	std::string raw;  // encrypted bytes pending decryption
	bool established = false;
};

std::string hex8(unsigned long v) {
	char b[16];
	snprintf(b, sizeof(b), "%lx", v);
	return std::string(b);
}

// perform the TLS handshake; validates the certificate chain + name like
// WinHTTP's defaults did. `cancelled` is polled each round: the socket must
// have SO_RCVTIMEO set BEFORE the handshake, otherwise a black-holed
// handshake (server accepts TCP, never answers) blocks in recv() forever and
// neither the idle timeout nor the pause button can reach it.
bool tls_handshake(Tls& t, SOCKET s, const std::string& host, std::string& err,
	bool allow_untrusted = false, const std::function<bool()>& cancelled = {}) {
	SCHANNEL_CRED sc{};
	sc.dwVersion = SCHANNEL_CRED_VERSION;
	sc.dwFlags = SCH_USE_STRONG_CRYPTO | SCH_IGNORE_REVOCATION;
	SECURITY_STATUS ss = AcquireCredentialsHandleW(nullptr, UNISP_NAME_W, SECPKG_CRED_OUTBOUND,
		nullptr, &sc, nullptr, nullptr, &t.cred, nullptr);
	if (ss != SEC_E_OK) {
		err = "tls: acquire credentials failed";
		return false;
	}
	t.have_cred = true;

	const ULONG req = ISC_REQ_CONFIDENTIALITY | ISC_REQ_SEQUENCE_DETECT |
		ISC_REQ_EXTENDED_ERROR | ISC_REQ_STREAM | ISC_REQ_ALLOCATE_MEMORY | ISC_REQ_MANUAL_CRED_VALIDATION;

	std::wstring whost = utf8_to_wide(host);
	std::vector<char> in_raw;  // handshake bytes received but not yet consumed
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);

	for (int round = 0; round < 240; round++) {
		if (cancelled && cancelled()) {
			err = "cancelled";
			return false;
		}
		if (std::chrono::steady_clock::now() > deadline) {
			err = "tls: handshake timed out";
			return false;
		}

		SecBuffer out_buf;
		out_buf.BufferType = SECBUFFER_TOKEN;
		out_buf.cbBuffer = 0;
		out_buf.pvBuffer = nullptr;
		SecBufferDesc out_desc;
		out_desc.ulVersion = SECBUFFER_VERSION;
		out_desc.cBuffers = 1;
		out_desc.pBuffers = &out_buf;

		SecBuffer in_bufs[2];
		in_bufs[0].BufferType = SECBUFFER_TOKEN;
		in_bufs[0].cbBuffer = (ULONG)in_raw.size();
		in_bufs[0].pvBuffer = in_raw.empty() ? nullptr : in_raw.data();
		in_bufs[1].BufferType = SECBUFFER_EMPTY;
		in_bufs[1].cbBuffer = 0;
		in_bufs[1].pvBuffer = nullptr;
		SecBufferDesc in_desc;
		in_desc.ulVersion = SECBUFFER_VERSION;
		in_desc.cBuffers = 2;
		in_desc.pBuffers = in_bufs;

		ULONG attrs = 0;
		ss = InitializeSecurityContextW(&t.cred, t.have_ctx ? &t.ctx : nullptr,
			(SEC_WCHAR*)whost.c_str(), req, 0, SECURITY_NATIVE_DREP,
			in_raw.empty() ? nullptr : &in_desc, 0,
			t.have_ctx ? nullptr : &t.ctx, &out_desc, &attrs, nullptr);
		if (!t.have_ctx && (ss == SEC_I_CONTINUE_NEEDED || ss == SEC_E_OK)) t.have_ctx = true;

		if (ss == SEC_E_INCOMPLETE_MESSAGE) {
			char buf[16384];
			int got = recv_slice(s, buf, sizeof(buf));
			if (got > 0) {
				in_raw.insert(in_raw.end(), buf, buf + got);
			} else if (got < 0) {
				continue;  // timeout slice: retry ISC to re-check for completeness
			} else {
				err = "tls: connection closed during handshake";
				return false;
			}
			continue;
		}

		if (ss == SEC_I_CONTINUE_NEEDED || ss == SEC_E_OK) {
			if (out_buf.cbBuffer > 0 && out_buf.pvBuffer) {
				bool ok = send_all(s, (const char*)out_buf.pvBuffer, out_buf.cbBuffer, err);
				FreeContextBuffer(out_buf.pvBuffer);
				if (!ok) {
					if (err.empty()) err = "tls: handshake send failed";
					return false;
				}
			}
			// consume what ISC used (SECBUFFER_EXTRA marks leftovers)
			size_t consumed = in_raw.size();
			for (int i = 0; i < 2; i++) {
				if (in_bufs[i].BufferType == SECBUFFER_EXTRA && in_bufs[i].cbBuffer > 0) {
					consumed = in_raw.size() - in_bufs[i].cbBuffer;
				}
			}
			in_raw.erase(in_raw.begin(), in_raw.begin() + (long)consumed);

			if (ss == SEC_E_OK) {
				// leftover bytes are the first encrypted application-data records
				t.raw.assign(in_raw.begin(), in_raw.end());
				t.established = true;

				// certificate validation: chain trust + name match (WinHTTP parity)
				// (skipped when allow_untrusted — test seam for loopback repro probes)
				if (allow_untrusted) {
					if (QueryContextAttributesW(&t.ctx, SECPKG_ATTR_STREAM_SIZES, &t.sizes) != SEC_E_OK ||
						t.sizes.cbMaximumMessage == 0) {
						t.sizes.cbHeader = 5;
						t.sizes.cbTrailer = 36;
						t.sizes.cbMaximumMessage = 16384;
					}
					return true;
				}
				PCCERT_CONTEXT remote = nullptr;
				if (QueryContextAttributesW(&t.ctx, SECPKG_ATTR_REMOTE_CERT_CONTEXT, &remote) != SEC_E_OK ||
					!remote) {
					err = "tls: no server certificate";
					return false;
				}
				CERT_CHAIN_POLICY_PARA policy_para;
				policy_para.cbSize = sizeof(policy_para);
				policy_para.dwFlags = 0;
				policy_para.pvExtraPolicyPara = nullptr;
				CERT_CHAIN_POLICY_STATUS policy_status;
				policy_status.cbSize = sizeof(policy_status);
				SSL_EXTRA_CERT_CHAIN_POLICY_PARA ssl_para;
				ssl_para.cbSize = sizeof(ssl_para);
				ssl_para.dwAuthType = AUTHTYPE_SERVER;
				ssl_para.fdwChecks = 0;
				ssl_para.pwszServerName = (LPWSTR)whost.c_str();
				policy_para.pvExtraPolicyPara = &ssl_para;
				CERT_CHAIN_PARA chain_para;
				chain_para.cbSize = sizeof(chain_para);
				chain_para.RequestedUsage.dwType = USAGE_MATCH_TYPE_AND;
				chain_para.RequestedUsage.Usage.cUsageIdentifier = 0;
				chain_para.RequestedUsage.Usage.rgpszUsageIdentifier = nullptr;
				PCCERT_CHAIN_CONTEXT chain = nullptr;
				bool trusted = false;
				if (CertGetCertificateChain(nullptr, remote, nullptr, nullptr, &chain_para,
						CERT_CHAIN_REVOCATION_CHECK_DISABLED, nullptr, &chain) && chain) {
					trusted = CertVerifyCertificateChainPolicy(CERT_CHAIN_POLICY_SSL, chain,
						&policy_para, &policy_status) && policy_status.dwError == 0;
					CertFreeCertificateChain(chain);
				}
				CertFreeCertificateContext(remote);
				if (!trusted) {
					err = "tls: server certificate not trusted for " + host;
					return false;
				}

				// stream sizes
				if (QueryContextAttributesW(&t.ctx, SECPKG_ATTR_STREAM_SIZES, &t.sizes) != SEC_E_OK ||
					t.sizes.cbMaximumMessage == 0) {
					t.sizes.cbHeader = 5;
					t.sizes.cbTrailer = 36;
					t.sizes.cbMaximumMessage = 16384;
				}
				return true;
			}

			// more handshake bytes from the peer
			char buf[16384];
			int got = recv_slice(s, buf, sizeof(buf));
			if (got > 0) {
				in_raw.insert(in_raw.end(), buf, buf + got);
			} else if (got == 0) {
				err = "tls: connection closed during handshake";
				return false;
			}
			continue;
		}

		err = "tls: handshake failed (0x" + hex8((unsigned long)ss) + ")";
		return false;
	}
	err = "tls: handshake did not converge";
	return false;
}

bool tls_send(Tls& t, SOCKET s, const char* data, size_t len, std::string& err) {
	std::vector<char> frame;
	size_t off = 0;
	while (off < len) {
		size_t chunk = std::min(len - off, (size_t)t.sizes.cbMaximumMessage);
		frame.resize((size_t)t.sizes.cbHeader + chunk + t.sizes.cbTrailer);
		SecBuffer bufs[3];
		bufs[0].BufferType = SECBUFFER_STREAM_HEADER;
		bufs[0].cbBuffer = t.sizes.cbHeader;
		bufs[0].pvBuffer = frame.data();
		bufs[1].BufferType = SECBUFFER_DATA;
		bufs[1].cbBuffer = (ULONG)chunk;
		bufs[1].pvBuffer = frame.data() + t.sizes.cbHeader;
		bufs[2].BufferType = SECBUFFER_STREAM_TRAILER;
		bufs[2].cbBuffer = t.sizes.cbTrailer;
		bufs[2].pvBuffer = frame.data() + t.sizes.cbHeader + chunk;
		SecBufferDesc desc;
		desc.ulVersion = SECBUFFER_VERSION;
		desc.cBuffers = 3;
		desc.pBuffers = bufs;
		memcpy(frame.data() + t.sizes.cbHeader, data + off, chunk);
		if (EncryptMessage(&t.ctx, 0, &desc, 0) != SEC_E_OK) {
			err = "tls: encrypt failed";
			return false;
		}
		size_t total = (size_t)bufs[0].cbBuffer + bufs[1].cbBuffer + bufs[2].cbBuffer;
		if (!send_all(s, frame.data(), total, err)) return false;
		off += chunk;
	}
	return true;
}

// decrypt pending raw bytes; appends plaintext to plain. Returns false on
// fatal errors; eof set when the peer closed the TLS connection cleanly.
bool tls_decrypt(Tls& t, std::string& plain, bool& eof, std::string& err) {
	while (!t.raw.empty()) {
		SecBuffer bufs[4];
		for (int i = 0; i < 4; i++) {
			bufs[i].BufferType = SECBUFFER_EMPTY;
			bufs[i].cbBuffer = 0;
			bufs[i].pvBuffer = nullptr;
		}
		bufs[0].BufferType = SECBUFFER_DATA;
		bufs[0].cbBuffer = (ULONG)t.raw.size();
		bufs[0].pvBuffer = t.raw.data();
		SecBufferDesc desc;
		desc.ulVersion = SECBUFFER_VERSION;
		desc.cBuffers = 4;
		desc.pBuffers = bufs;
		SECURITY_STATUS ss = DecryptMessage(&t.ctx, &desc, 0, nullptr);
		if (ss == SEC_E_INCOMPLETE_MESSAGE) return true;  // need more raw bytes
		if (ss == SEC_I_CONTEXT_EXPIRED || ss == SEC_E_CONTEXT_EXPIRED) {
			t.raw.clear();
			eof = true;
			return true;
		}
		if (ss != SEC_E_OK) {
			err = "tls: decrypt failed (0x" + hex8((unsigned long)ss) + ")";
			return false;
		}
		for (int i = 0; i < 4; i++) {
			if (bufs[i].BufferType == SECBUFFER_DATA && bufs[i].cbBuffer > 0) {
				plain.append((const char*)bufs[i].pvBuffer, bufs[i].cbBuffer);
			}
		}
		bool extra = false;
		for (int i = 0; i < 4; i++) {
			if (bufs[i].BufferType == SECBUFFER_EXTRA && bufs[i].cbBuffer > 0) {
				t.raw.erase(0, t.raw.size() - bufs[i].cbBuffer);
				extra = true;
			}
		}
		if (!extra) t.raw.clear();
		if (plain.empty() && !extra) return true;  // decrypted but no app data (e.g. alerts)
	}
	return true;
}

// ── chunked transfer decoding ──────────────────────────────────────────────

struct ChunkDecoder {
	enum State { SIZE_LINE, DATA, SKIP_CR, SKIP_LF };
	bool chunked = false;
	State state = SIZE_LINE;
	size_t remaining = 0;
	bool done = false;
	std::string size_line;

	void feed(const char* data, size_t len, const std::function<void(const char*, size_t)>& out) {
		if (!chunked) {
			out(data, len);
			return;
		}
		if (done) return;  // terminal chunk seen: discard the trailer section (it
		                   // used to leak into the SSE stream as garbage bytes)
		size_t i = 0;
		while (i < len) {
			char c = data[i++];
			switch (state) {
				case SIZE_LINE:
					if (c == '\n') {
						std::string hex = size_line;
						size_line.clear();
						size_t semi = hex.find(';');
						if (semi != std::string::npos) hex = hex.substr(0, semi);
						while (!hex.empty() && (hex.back() == '\r' || hex.back() == ' ')) hex.pop_back();
						unsigned long long sz = 0;
						sscanf(hex.c_str(), "%llx", &sz);
						if (sz == 0) {
							done = true;
						} else {
							remaining = (size_t)sz;
							state = DATA;
						}
					} else if (c != '\r') {
						size_line += c;
					}
					break;
				case DATA:
				{
					size_t take = std::min(len - i + 1, remaining);
					out(data + i - 1, take);
					i -= 1;  // re-advance below
					i += take;
					remaining -= take;
					if (remaining == 0) state = SKIP_CR;
					break;
				}
				case SKIP_CR:
					if (c == '\r') {
						state = SKIP_LF;
					} else if (c == '\n') {
						// bare LF without preceding CR: treat as line ending
						// (lenient parsing for malformed chunk terminators)
						state = SIZE_LINE;
					} else {
						// unexpected byte in CRLF sequence — skip it and
						// start reading the next size line
						state = SIZE_LINE;
						i--;  // reprocess
					}
					break;
				case SKIP_LF:
					if (c == '\n') {
						state = SIZE_LINE;
					} else {
						// missing LF after CR: malformed, but recover by
						// treating the current byte as the start of the
						// next size line
						state = SIZE_LINE;
						i--;  // reprocess
					}
					break;
			}
			if (done) {
				out(data + i, len - i);
				return;
			}
		}
	}
};

}  // namespace

// ── public API ─────────────────────────────────────────────────────────────

HttpStreamResult http_request_impl(
	const std::string& method,
	const std::string& url,
	const std::vector<std::pair<std::string, std::string>>& headers,
	const std::string& body,
	const std::function<bool(const char* data, size_t len)>* on_chunk,
	int idle_timeout_ms,
	const std::function<bool()>& cancelled,
	bool allow_untrusted) {

	ensure_wsa();
	HttpStreamResult result;
	UrlParts parts;
	if (!split_url(url, parts)) {
		result.error = "invalid url: " + url;
		return result;
	}

	std::string err;
	// try pooled connection first for non-TLS requests
	SOCKET sock = INVALID_SOCKET;
	if (!parts.https) {
		sock = acquire_pooled(parts.host, parts.port);
	}
	if (sock == INVALID_SOCKET) {
		sock = tcp_connect(parts.host, parts.port, err, cancelled);
		if (sock == INVALID_SOCKET) {
			result.error = err;
			return result;
		}
	}

	// short recv timeout slices so `cancelled` is honored even on silent
	// streams — including during the TLS handshake below (a black-holed
	// handshake used to block in recv() forever: no timeout, no abort)
	DWORD slice_ms = 250;
	setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&slice_ms, sizeof(slice_ms));

	// effective idle timeout: 0/negative means "no deadline" (callers that pass 0
	// want the historical no-timeout behavior); any positive value is enforced on
	// time since the last plaintext byte arriving. This makes the idle_timeout_ms
	// parameter actually do something instead of being silently ignored.
	const bool have_idle_timeout = idle_timeout_ms > 0;
	auto last_data_at = std::chrono::steady_clock::now();
	// bound blocking sends too: a stalled peer with a large request body (big
	// context) used to hang the worker indefinitely
	DWORD snd_ms = 30000;
	setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&snd_ms, sizeof(snd_ms));

	Tls tls;
	if (parts.https) {
		if (!tls_handshake(tls, sock, parts.host, err, allow_untrusted, cancelled)) {
			result.error = err;
			closesocket(sock);
			return result;
		}
	}

	// build request
	std::string req = method + " " + parts.path + " HTTP/1.1\r\n";
	if (parts.host.find(':') != std::string::npos) {
		// IPv6: RFC 7230 requires brackets in Host header
		req += "Host: [" + parts.host + "]";
	} else {
		req += "Host: " + parts.host;
	}
	if ((parts.https && parts.port != 443) || (!parts.https && parts.port != 80)) {
		req += ":" + std::to_string(parts.port);
	}
	req += "\r\n";
	for (const auto& [key, value] : headers) {
		req += key + ": " + value + "\r\n";
	}
	req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
	req += "Connection: close\r\n\r\n";
	req += body;

	if (parts.https) {
		if (!tls_send(tls, sock, req.data(), req.size(), err)) {
			result.error = err.empty() ? "tls send failed" : err;
			closesocket(sock);
			return result;
		}
	} else {
		if (!send_all(sock, req.data(), req.size(), err, &cancelled)) {
			result.error = err;
			closesocket(sock);
			return result;
		}
	}

	std::string head;           // response head bytes
	bool head_done = false;
	ChunkDecoder chunker;
	std::string pending;        // plaintext body bytes not yet processed
	bool eof = false;
	bool stop_requested = false;
	std::string error;
	std::string socket_error;   // fatal recv error (reset/abort), distinct from clean eof
	std::map<std::string, std::string> hdr;  // response headers (hoisted for tail checks)

	// pull more plaintext into `pending` (decrypts for TLS); returns false on eof
	auto pull = [&]() {
		if (parts.https) {
			std::string plain;
			if (!tls_decrypt(tls, plain, eof, error)) return false;
			pending += plain;
			if (!pending.empty() || eof) return true;
			// recv whenever no plaintext is available — including when tls.raw
			// still holds a PARTIAL record (DecryptMessage returned
			// SEC_E_INCOMPLETE_MESSAGE). Gating recv on raw being empty would
			// never fetch the record's remaining segments: the stream would
			// freeze mid-body while this loop spins on the incomplete record.
			{
				char buf[16384];
				auto t_rcv0 = phi::trace::now_ms();
				int got = recv_slice(sock, buf, sizeof(buf), &socket_error);
				PHI_TRACE("RECV-TLS got=%d wait_ms=%lld", got,
					(long long)(phi::trace::now_ms() - t_rcv0));
				if (got > 0) {
					tls.raw.append(buf, (size_t)got);
					if (have_idle_timeout) last_data_at = std::chrono::steady_clock::now();
				} else if (got == 0) {
					eof = true;
				}
			}
			return true;
		}
		char buf[65536];
		auto t_rcv0 = phi::trace::now_ms();
		int n = recv_slice(sock, buf, sizeof(buf), &socket_error);
		PHI_TRACE("RECV got=%d wait_ms=%lld", n,
			(long long)(phi::trace::now_ms() - t_rcv0));
		if (n > 0) {
			pending.append(buf, (size_t)n);
			if (have_idle_timeout) last_data_at = std::chrono::steady_clock::now();
		} else if (n == 0) {
			eof = true;
		}
		return true;
	};

	auto process_body = [&](const char* data, size_t len) {
		chunker.feed(data, len, [&](const char* p, size_t n) {
			if (stop_requested) return;
			if (!on_chunk) {
				result.body.append(p, n);
				return;
			}
			if (!(*on_chunk)(p, n)) stop_requested = true;
		});
	};

	while (!stop_requested) {
		if (cancelled && cancelled()) break;
		if (have_idle_timeout &&
			std::chrono::steady_clock::now() - last_data_at >
				std::chrono::milliseconds(idle_timeout_ms)) {
			error = "idle timeout: no response bytes for " +
			        std::to_string(idle_timeout_ms) + " ms";
			break;
		}
		if (!pending.empty()) {
			// split off the response head once we have it
			if (!head_done) {
				head += pending;
				pending.clear();
				size_t sep = head.find("\r\n\r\n");
				if (sep == std::string::npos) {
					// keep accumulating the head
				} else {
					std::string body_part = head.substr(sep + 4);
					head = head.substr(0, sep);
					pending = body_part;

					result.ok = true;
					int major = 1, minor = 1, code = 0;
					if (sscanf(head.c_str(), "HTTP/%d.%d %d", &major, &minor, &code) >= 3) {
						result.status = code;
					}

					size_t pos = head.find("\r\n");
					while (pos != std::string::npos) {
						size_t next = head.find("\r\n", pos + 2);
						std::string line = head.substr(pos + 2,
							next == std::string::npos ? std::string::npos : next - pos - 2);
						size_t colon = line.find(':');
						if (colon != std::string::npos) {
							std::string k = line.substr(0, colon);
							for (auto& c : k) c = (char)tolower((unsigned char)c);
							size_t v = colon + 1;
							while (v < line.size() && (line[v] == ' ' || line[v] == '\t')) v++;
							hdr[k] = line.substr(v);
						}
						if (next == std::string::npos) break;
						pos = next;
					}
					auto it = hdr.find("transfer-encoding");
					if (it != hdr.end() && it->second.find("chunked") != std::string::npos) {
						chunker.chunked = true;
					}
					head_done = true;
				}
			}
			if (!pending.empty()) {
				std::string take = std::move(pending);
				pending.clear();
				process_body(take.data(), take.size());
				continue;  // process again (in case data remains) — loop re-checks cancel
			}
			continue;
		}

		if (eof) break;
		if (cancelled && cancelled()) break;

		if (!pull()) break;
		if (!pending.empty()) {
			continue;
		}
	}

	// a stream that ended mid-body (reset, or clean close before the terminal
	// chunk / before content-length was met) is a transport failure — the old
	// code accepted truncated responses as complete
	if (result.ok && error.empty()) {
		auto cl = hdr.find("content-length");
		bool incomplete = (chunker.chunked && !chunker.done) ||
			(!on_chunk && !chunker.chunked && cl != hdr.end() &&
				result.body.size() < (size_t)strtoull(cl->second.c_str(), nullptr, 10));
		if (incomplete && !stop_requested && !(cancelled && cancelled())) {
			error = socket_error.empty() ? "response truncated: connection closed mid-body" : socket_error;
		}
	}
	if (!result.ok && error.empty()) {
		result.error = eof ? "no response from server" : "connection lost";
	} else if (!error.empty()) {
		result.error = error;
		result.ok = false;
	}
	result.headers = std::move(hdr);
	// connection pooling: for non-TLS successful connections, try to reuse the socket
	if (result.ok && !parts.https && socket_error.empty()) {
		release_pooled(sock);
	} else {
		if (sock != INVALID_SOCKET) closesocket(sock);
	}
	return result;
}

HttpStreamResult http_post_stream(
	const std::string& url,
	const std::vector<std::pair<std::string, std::string>>& headers,
	const std::string& body,
	const std::function<bool(const char* data, size_t len)>& on_chunk,
	int idle_timeout_ms,
	const std::function<bool()>& cancelled,
	bool allow_untrusted) {
	// the POST path keeps streaming through on_chunk; the impl accumulates the
	// body when on_chunk is null, which is exactly the old behavior
	return http_request_impl("POST", url, headers, body, &on_chunk, idle_timeout_ms, cancelled, allow_untrusted);
}

HttpStreamResult http_request(
	const std::string& method,
	const std::string& url,
	const std::vector<std::pair<std::string, std::string>>& headers,
	const std::string& body,
	int idle_timeout_ms,
	const std::function<bool()>& cancelled,
	bool allow_untrusted) {
	return http_request_impl(method, url, headers, body, nullptr, idle_timeout_ms, cancelled, allow_untrusted);
}

HttpStreamResult http_get(const std::string& url, int idle_timeout_ms) {
	return http_request_impl("GET", url, {}, "", nullptr, idle_timeout_ms, {}, false);
}

// ── parallel request helper ────────────────────────────────────────────────

std::vector<ParallelResult> http_parallel_requests(
	const std::vector<ParallelRequest>& requests,
	int concurrent_limit) {
	std::vector<ParallelResult> results(requests.size());
	if (requests.empty()) return results;

	std::atomic<size_t> idx{0};
	size_t n_workers = std::min<size_t>(
		std::max<size_t>(1, (size_t)concurrent_limit), requests.size());
	std::vector<std::thread> threads;
	threads.reserve(n_workers);
	auto worker = [&]() {
		for (;;) {
			size_t i = idx.fetch_add(1);
			if (i >= requests.size()) return;
			const auto& req = requests[i];
			results[i] = {true, (int)i,
				http_request(req.method, req.url, req.headers, req.body,
					req.idle_timeout_ms, req.cancelled, req.allow_untrusted)};
		}
	};
	for (size_t k = 0; k < n_workers; k++) threads.emplace_back(worker);
	for (auto& t : threads) t.join();
	return results;
}

}  // namespace phi
