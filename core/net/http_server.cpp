#include "net/http_server.hpp"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "util/base.hpp"

namespace phi {


static bool sock_send_all(int sock, const char* data, size_t len) {
	size_t sent = 0;
	while (sent < len) {
		int n = ::send(sock, data + sent, (int)std::min(len - sent, (size_t)0x40000000), 0);
		if (n <= 0) return false;
		sent += (size_t)n;
	}
	return true;
}

static void sock_close(int sock) {
	if (sock >= 0) {
		::shutdown(sock, SD_BOTH);
		::closesocket(sock);
	}
}

// RAII：client_loop 的任何返回路径（早退、异常、正常结束）都要把 done 置 true，
// 否则 accept_loop 的 remove_if 看不到它完成——尤其 plain-http 路径（每个 UI 资产
// 请求都走它）和 7 个早退路径都不设置，会让僵尸 std::thread 无界累积。
struct ClientDoneGuard {
	std::shared_ptr<std::atomic<bool>> done;
	~ClientDoneGuard() {
		if (done) done->store(true, std::memory_order_relaxed);
	}
};

// ── lifecycle ──────────────────────────────────────────────────────────────

bool HttpServer::start(int port) {
	WSADATA wsa;
	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;

	listen_sock_ = (int)::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (listen_sock_ < 0) return false;
	BOOL reuse = TRUE;
	::setsockopt(listen_sock_, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));

	sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	addr.sin_port = htons((uint16_t)port);
	if (::bind(listen_sock_, (sockaddr*)&addr, sizeof(addr)) != 0) return false;
	if (::listen(listen_sock_, 16) != 0) return false;

	sockaddr_in bound{};
	int len = sizeof(bound);
	::getsockname(listen_sock_, (sockaddr*)&bound, &len);
	port_ = ntohs(bound.sin_port);
	running_ = true;

	{
		std::lock_guard<std::mutex> lock(threads_mutex_);
		ThreadEntry entry;
		entry.t = std::thread(&HttpServer::accept_loop, this);
		threads_.push_back(std::move(entry));
	}
	return true;
}

void HttpServer::stop() {
	if (!running_.exchange(false)) return;
	sock_close(listen_sock_);
	listen_sock_ = -1;
	// wake the accept loop so it exits its blocking accept() call
	std::vector<ThreadEntry> stolen_threads;
	{
		std::lock_guard<std::mutex> lock(threads_mutex_);
		stolen_threads.swap(threads_);
	}
	std::vector<std::shared_ptr<WebSocketConnection>> snapshot;
	{
		std::lock_guard<std::mutex> lock(clients_mutex_);
		for (auto& [id, client] : clients_) snapshot.push_back(client);
		clients_.clear();
	}
	for (auto& c : snapshot) {
		c->closed.store(true);
		c->stopping.store(true);
	}
	for (auto& c : snapshot) {
		c->close_sock();  // aborts in-flight sends/recvs (closes exactly once)
		c->q_cv.notify_all();
	}
	// join all background threads before returning — prevents use-after-free
	// if the server object is destroyed while a detached thread is still running
	for (auto& e : stolen_threads) {
		if (e.t.joinable()) e.t.join();
	}
}

// ── http ───────────────────────────────────────────────────────────────────

static const char* status_text(int s) {
	switch (s) {
		case 200: return "OK";
		case 400: return "Bad Request";
		case 403: return "Forbidden";
		case 404: return "Not Found";
		case 405: return "Method Not Allowed";
		case 500: return "Internal Server Error";
		default: return "Unknown";
	}
}

static bool read_line(int sock, std::string& line) {
	line.clear();
	char c;
	while (true) {
		int n = ::recv(sock, &c, 1, 0);
		if (n <= 0) return false;
		if (c == '\n') break;
		if (c != '\r') line.push_back(c);
		if (line.size() > 65536) return false;
	}
	return true;
}

// read exactly len bytes; returns false on error/disconnect. A recv timeout
// (WSAETIMEDOUT) inside a frame is fatal for that frame but must not be
// mistaken for idle silence, so it just fails the read.
static bool recv_full(int sock, void* buf, size_t len) {
	char* p = (char*)buf;
	size_t got = 0;
	while (got < len) {
		int n = ::recv(sock, p + got, (int)(len - got), 0);
		if (n <= 0) return false;
		got += (size_t)n;
	}
	return true;
}

static void url_decode(std::string& s) {
	std::string out;
	out.reserve(s.size());
	auto hex = [](char c) -> int {
		if (c >= '0' && c <= '9') return c - '0';
		if (c >= 'a' && c <= 'f') return c - 'a' + 10;
		if (c >= 'A' && c <= 'F') return c - 'A' + 10;
		return -1;
	};
	for (size_t i = 0; i < s.size(); i++) {
		if (s[i] == '%') {
			// malformed %xx (truncated or non-hex) is rejected: silently keeping
			// the raw bytes would let a %2f../%00 smuggle past path checks
			if (i + 2 >= s.size() || hex(s[i + 1]) < 0 || hex(s[i + 2]) < 0) {
				throw std::runtime_error("invalid percent-encoding");
			}
			out.push_back((char)((hex(s[i + 1]) << 4) | hex(s[i + 2])));
			i += 2;
			continue;
		}
		if (s[i] == '+') out.push_back(' ');
		else out.push_back(s[i]);
	}
	s = out;
}

void HttpServer::accept_loop() {
	while (running_) {
		sockaddr_in peer{};
		int len = sizeof(peer);
		int sock = (int)::accept(listen_sock_, (sockaddr*)&peer, &len);
		if (sock < 0) {
			if (!running_) break;
			continue;
		}
		BOOL nodelay = TRUE;
		::setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof(nodelay));
		// bound blocking sends: a client that stops reading (busy renderer
		// parsing the previous multi-MB snapshot) must not wedge the broadcasting
		// thread forever — otherwise snapshots stop and the UI freezes mid-stream.
		// 30s (not 5s): a legitimate renderer can be busy for seconds on a huge
		// snapshot render; killing the connection that early surfaced to the user
		// as the spurious "未连接" error mid-agent-run (restart was the only fix).
		DWORD send_timeout_ms = 30000;
		::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&send_timeout_ms, sizeof(send_timeout_ms));
		// bound blocking receives for plain HTTP too: without a recv timeout a
		// client that connects and sends a partial request head (or nothing at
		// all) blocks its client_loop thread in read_line()/recv() FOREVER — each
		// such "slowloris" client permanently leaked one thread slot. The WS
		// path re-sets a shorter window below (20s heartbeat probes).
		DWORD recv_timeout_ms = 30000;
		::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&recv_timeout_ms, sizeof(recv_timeout_ms));
		{
		std::lock_guard<std::mutex> lock(threads_mutex_);
		// Recycle completed threads before adding a new one to prevent unbounded
		// growth of threads_ (each UI reconnect adds a thread that would otherwise
		// linger joinable until stop()).
		auto it = std::remove_if(threads_.begin(), threads_.end(),
			[](ThreadEntry& e) {
				if (!e.done->load(std::memory_order_relaxed)) return false;
				if (e.t.joinable()) e.t.join();
				return true;
			});
		threads_.erase(it, threads_.end());
	auto done = std::make_shared<std::atomic<bool>>(false);
	ThreadEntry entry;
	entry.t = std::thread(&HttpServer::client_loop, this, sock, done);
	entry.done = done;
	threads_.push_back(std::move(entry));
	}
	}
}

struct WebSocketConnection;
static void ws_sender_loop(std::shared_ptr<WebSocketConnection> conn);

bool HttpServer::try_upgrade(int sock, HttpRequest& req, std::shared_ptr<WebSocketConnection>* out) {
	std::string upgrade;
	auto it = req.headers.find("upgrade");
	if (it != req.headers.end()) upgrade = to_lower(it->second);
	if (upgrade != "websocket") return false;

	// accept key: base64(raw sha1(key + GUID))
	std::string key = req.headers["sec-websocket-key"];
	if (key.empty()) return false;
	static const char* GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
	std::string hex_digest = sha1_hex(key + GUID);
	std::string raw_bytes;
	raw_bytes.reserve(20);
	static auto hexval = [](char c) -> int {
		if (c >= '0' && c <= '9') return c - '0';
		if (c >= 'a' && c <= 'f') return c - 'a' + 10;
		return c - 'A' + 10;
	};
	for (size_t i = 0; i + 1 < hex_digest.size(); i += 2) {
		raw_bytes.push_back((char)((hexval(hex_digest[i]) << 4) | hexval(hex_digest[i + 1])));
	}
	std::string accept = base64_encode(raw_bytes);

	std::string response =
		"HTTP/1.1 101 Switching Protocols\r\n"
		"Upgrade: websocket\r\n"
		"Connection: Upgrade\r\n"
		"Sec-WebSocket-Accept: " + accept + "\r\n\r\n";
	if (!sock_send_all(sock, response.data(), response.size())) return false;

	// heartbeat read window: a blocking recv returns WSAETIMEDOUT after this
	// much inbound silence, which the ws read loop turns into ping probes
	DWORD recv_timeout_ms = 20000;
	::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&recv_timeout_ms, sizeof(recv_timeout_ms));

	// register client
	auto conn = std::make_shared<WebSocketConnection>();
	conn->sock = sock;
	uint64_t id;
	{
		std::lock_guard<std::mutex> lock(clients_mutex_);
		id = next_client_id_++;
		conn->id = id;
		clients_[id] = conn;
	}
	// start the sender BEFORE ws_open_: the open handler may already emit a
	// snapshot through conn->send_text() (enqueue only — the sender drains it).
	// If the thread cannot be created, unregister + close so the connection
	// never leaks as a registered zombie with a socket nobody owns.
	try {
		conn->sender = std::thread(ws_sender_loop, conn);
	} catch (...) {
		{
			std::lock_guard<std::mutex> lock(clients_mutex_);
			clients_.erase(id);
		}
		conn->closed.store(true);
		conn->close_sock();
		return false;
	}
	// exceptions from the open handler no longer leak the registration: the
	// client_loop cleanup (unregister + close + sender join) runs on return.
	// They must also NOT escape to the bare thread entry (std::terminate).
	try {
		if (ws_open_) ws_open_(*conn);
	} catch (const std::exception& e) {
		fprintf(stderr, "[ws] open handler error: %s\n", e.what());
	} catch (...) {
		fprintf(stderr, "[ws] open handler unknown error\n");
	}
	if (out) *out = conn;
	return true;
}

// ── ws frames ──────────────────────────────────────────────────────────────

// write a pre-encoded frame (no payload framing); same close-on-failure
// semantics as ws_send_frame
static bool ws_send_frame_raw(WebSocketConnection& conn, const std::string& wire) {
	std::lock_guard<std::mutex> lock(conn.send_mutex);
	if (conn.closed.load()) return false;
	int fd = conn.sock.load();
	if (!sock_send_all(fd, wire.data(), wire.size())) {
		conn.closed.store(true);
		conn.close_sock();
		return false;
	}
	return true;
}

static bool ws_send_frame(WebSocketConnection& conn, int opcode, const uint8_t* data, size_t len) {
	std::vector<uint8_t> frame;
	frame.reserve(len + 10);
	frame.push_back((uint8_t)(0x80 | opcode));
	if (len < 126) {
		frame.push_back((uint8_t)len);
	} else if (len < 65536) {
		frame.push_back(126);
		frame.push_back((uint8_t)(len >> 8));
		frame.push_back((uint8_t)(len & 0xff));
	} else {
		frame.push_back(127);
		for (int i = 7; i >= 0; i--) frame.push_back((uint8_t)(((uint64_t)len >> (i * 8)) & 0xff));
	}
	if (len > 0) frame.insert(frame.end(), data, data + len);
	std::lock_guard<std::mutex> lock(conn.send_mutex);
	if (conn.closed.load()) return false;
	int fd = conn.sock.load();
	if (!sock_send_all(fd, (const char*)frame.data(), frame.size())) {
		// peer stalled (send timeout hit) or connection gone: stop using the
		// socket. Closing it wakes the client's reader thread and lets later
		// broadcasts skip this client fast instead of waiting another timeout.
		conn.closed.store(true);
		conn.close_sock();
		return false;
	}
	return true;
}

void WebSocketConnection::enqueue(std::string text, bool coalesce) {
	enqueue_msg(OutMessage{std::move(text), coalesce, 0x1});
}

void WebSocketConnection::enqueue_frame(int opcode, const uint8_t* data, size_t len) {
	// pre-encode the ws frame and queue it for the sender thread — heartbeat
	// pings must never block the reader thread on a stalled peer send
	std::vector<uint8_t> frame;
	frame.reserve(len + 10);
	frame.push_back((uint8_t)(0x80 | opcode));
	if (len < 126) {
		frame.push_back((uint8_t)len);
	} else if (len < 65536) {
		frame.push_back(126);
		frame.push_back((uint8_t)(len >> 8));
		frame.push_back((uint8_t)(len & 0xff));
	} else {
		frame.push_back(127);
		for (int i = 7; i >= 0; i--) frame.push_back((uint8_t)(((uint64_t)len >> (i * 8)) & 0xff));
	}
	if (len > 0) frame.insert(frame.end(), data, data + len);
	std::string wire(frame.size(), '\0');
	std::memcpy(wire.data(), frame.data(), frame.size());
	enqueue_msg(OutMessage{std::move(wire), false, -1});
}

void WebSocketConnection::enqueue_msg(OutMessage&& m) {
	{
		std::lock_guard<std::mutex> lock(q_mutex);
		if (closed.load()) return;
		if (m.coalescable) {
			// patches carry full streaming state: an older queued patch is
			// subsumed by the newest (the client drops stale generations)
			for (auto it = out_queue.begin(); it != out_queue.end();) {
				if (it->coalescable) it = out_queue.erase(it);
				else ++it;
			}
		}
		if (out_queue.size() >= 256) out_queue.pop_front();  // bounded backlog
		out_queue.push_back(std::move(m));
	}
	q_cv.notify_one();
}

void WebSocketConnection::send_text(const std::string& text) {
	enqueue(std::string(text), false);
}

void WebSocketConnection::close_sock() {
	// closesocket exactly once even with concurrent callers: the sender thread
	// (send failure) and the cleanup paths (client_loop / server stop) can race
	// here. Windows recycles fd numbers aggressively, so a second close on a
	// stale fd can hit a freshly reused socket owned by an unrelated connection
	// — corrupting another client or crashing the process.
	int fd = sock.exchange(-1);
	if (fd >= 0) {
		::shutdown(fd, SD_BOTH);
		::closesocket(fd);
	}
}

// per-connection flush thread: the ONLY place that does blocking sends, so a
// stalled renderer delays just this client — never the broadcasting/agent
// threads (which would freeze the stream for the SO_SNDTIMEO duration)
static void ws_sender_loop(std::shared_ptr<WebSocketConnection> conn) {
	std::unique_lock<std::mutex> lock(conn->q_mutex);
	while (true) {
		conn->q_cv.wait(lock, [&] {
			return conn->stopping.load() || conn->closed.load() || !conn->out_queue.empty();
		});
		if (conn->out_queue.empty()) break;  // stopping/closed and fully drained
		WebSocketConnection::OutMessage m = std::move(conn->out_queue.front());
		conn->out_queue.pop_front();
		lock.unlock();
		bool ok;
		if (m.opcode < 0) {
			// pre-encoded frame (ping/pong): write verbatim
			ok = ws_send_frame_raw(*conn, m.text);
		} else {
			ok = ws_send_frame(*conn, m.opcode, (const uint8_t*)m.text.data(), m.text.size());
		}
		lock.lock();
		if (!ok) break;
	}
}

void HttpServer::broadcast(const std::string& text) {
	// shared_ptr snapshot: connections stay alive even if a client thread
	// unregisters mid-broadcast (no use-after-free on send_mutex/closed)
	std::vector<std::shared_ptr<WebSocketConnection>> snapshot;
	{
		std::lock_guard<std::mutex> lock(clients_mutex_);
		for (auto& [id, c] : clients_) snapshot.push_back(c);
	}
	for (auto& c : snapshot) c->enqueue(text, false);
}

void HttpServer::broadcast_latest(const std::string& text) {
	std::vector<std::shared_ptr<WebSocketConnection>> snapshot;
	{
		std::lock_guard<std::mutex> lock(clients_mutex_);
		for (auto& [id, c] : clients_) snapshot.push_back(c);
	}
	for (auto& c : snapshot) c->enqueue(text, true);
}

static const size_t MAX_WS_PAYLOAD = 64ull * 1024 * 1024;

void HttpServer::client_loop(int sock, std::shared_ptr<std::atomic<bool>> done) {
	ClientDoneGuard done_guard{std::move(done)};  // RAII: ensures done is set on ANY return path
	// read request head
	std::string line;
	if (!read_line(sock, line)) { sock_close(sock); return; }
	HttpRequest req;
	{
		// "GET /path?query HTTP/1.1"
		size_t sp1 = line.find(' ');
		size_t sp2 = line.rfind(' ');
		if (sp1 == std::string::npos || sp2 == std::string::npos || sp2 <= sp1) {
			sock_close(sock);
			return;
		}
		req.method = line.substr(0, sp1);
		std::string target = line.substr(sp1 + 1, sp2 - sp1 - 1);
		size_t q = target.find('?');
		if (q != std::string::npos) {
			req.query = target.substr(q + 1);
			req.path = target.substr(0, q);
		} else {
			req.path = target;
		}
		// malformed percent-encoding → 400 (not silently-decoded raw bytes)
		try {
			url_decode(req.path);
		} catch (const std::exception&) {
			static const char resp[] =
				"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
			sock_send_all(sock, resp, sizeof(resp) - 1);
			sock_close(sock);
			return;
		}
	}
	size_t content_length = 0;
	size_t header_count = 0;
	bool header_ok = false;
	while (read_line(sock, line)) {
		if (line.empty()) { header_ok = true; break; }
		if (++header_count > 256) { sock_close(sock); return; }  // reject header explosion
		size_t colon = line.find(':');
		if (colon == std::string::npos) continue;
		std::string key = to_lower(trim(line.substr(0, colon)));
		std::string value = trim(line.substr(colon + 1));
		req.headers[key] = value;
		if (key == "content-length") content_length = (size_t)strtoull(value.c_str(), nullptr, 10);
	}
	if (!header_ok) { sock_close(sock); return; }
	if (content_length > 0) {
		if (content_length > MAX_WS_PAYLOAD) { sock_close(sock); return; }
		req.body.resize(content_length);
		size_t got = 0;
		while (got < content_length) {
			int n = ::recv(sock, req.body.data() + got, (int)(content_length - got), 0);
			if (n <= 0) { sock_close(sock); return; }
			got += (size_t)n;
		}
	}

	// websocket session loop (runs on this thread until close). conn comes
	// straight from try_upgrade — a lookup by sock here raced with a concurrent
	// client registering between this connection's register and its lookup,
	// which could hand back the WRONG connection (or none) and corrupt state.
	std::shared_ptr<WebSocketConnection> conn;
	if (try_upgrade(sock, req, &conn)) {
		if (!conn) { sock_close(sock); return; }

		std::string fragmented;
		int frag_op = 0;
		bool alive = true;
		// heartbeat: server pings the client every 20s of inbound silence and
		// drops the link after 3 unanswered probes (60s). Without this a dead or
		// half-open socket (client killed without a close frame) was only
		// discovered by a send failure — which for a quiet session could take
		// minutes, leaving a zombie connection and a UI stuck "正在重连…".
		int silent_probes = 0;
		constexpr int kMaxSilentProbes = 3;
		uint8_t hdr[2] = {0, 0};
		size_t have_hdr_bytes = 0;  // persist partial header across iterations
		while (alive) {
			if (silent_probes > 0) {
				if (silent_probes >= kMaxSilentProbes) { alive = false; break; }
				// send the ping through the sender thread: a blocking send here
				// could hold the reader hostage for the whole SO_SNDTIMEO on a
				// stalled peer (and skew the next probe window)
				conn->enqueue_frame(0x9, nullptr, 0);
			}
			// Continue reading partial header from previous iteration
			while (have_hdr_bytes < 2) {
				int n = ::recv(sock, (char*)hdr + have_hdr_bytes, (int)(2 - have_hdr_bytes), 0);
				if (n < 0) {
					int err = WSAGetLastError();
					if (err == WSAETIMEDOUT || err == WSAEWOULDBLOCK) {
						// recv timeout = inbound silence: probe the client
						silent_probes++;
						break;
					}
					alive = false;
					break;
				}
				if (n == 0) { alive = false; break; }
				silent_probes = 0;  // any inbound traffic proves liveness
				have_hdr_bytes += (size_t)n;
			}
			if (!alive) break;
			// A probe timeout means hdr holds stale/never-received bytes: parsing
			// it produced random opcodes/lengths (garbage frames, bogus giant
			// allocations, spurious disconnects after exactly ~60s idle)
			if (have_hdr_bytes < 2) continue;
			bool fin = (hdr[0] & 0x80) != 0;
			int opcode = hdr[0] & 0x0f;
			bool masked = (hdr[1] & 0x80) != 0;
			uint64_t len = hdr[1] & 0x7f;
			if (len == 126) {
				uint8_t ext[2];
				if (!recv_full(sock, ext, 2)) break;
				len = ((uint64_t)ext[0] << 8) | ext[1];
			} else if (len == 127) {
				uint8_t ext[8];
				if (!recv_full(sock, ext, 8)) break;
				len = 0;
				for (int i = 0; i < 8; i++) len = (len << 8) | ext[i];
			}
			if (len > MAX_WS_PAYLOAD) break;
			uint8_t mask[4] = {0, 0, 0, 0};
			if (masked && !recv_full(sock, (char*)mask, 4)) break;
			silent_probes = 0;  // mid-frame bytes count as liveness too
			have_hdr_bytes = 0;  // reset after successful header parse

			std::vector<uint8_t> payload((size_t)len);
			size_t got = 0;
			while (got < len) {
				int n = ::recv(sock, (char*)payload.data() + got, (int)std::min(len - got, (uint64_t)0x40000000), 0);
				if (n <= 0) { alive = false; break; }
				got += (size_t)n;
			}
			if (!alive) break;
			if (masked) {
				for (size_t i = 0; i < payload.size(); i++) payload[i] ^= mask[i % 4];
			}

			switch (opcode) {
				case 0x0:  // continuation
					fragmented.append((char*)payload.data(), payload.size());
					if (fin) {
						std::string msg;
						msg.swap(fragmented);
						if (ws_message_ && frag_op == 0x1) {
							try {
								ws_message_(*conn, msg);
							} catch (const std::exception& e) {
								// a throw on this bare thread = std::terminate = app crash
								fprintf(stderr, "[ws] message handler error: %s\n", e.what());
							}
						}
						frag_op = 0;
					}
					break;
				case 0x1:
				case 0x2: {
					std::string msg((char*)payload.data(), payload.size());
					if (!fin) {
						fragmented = msg;
						frag_op = opcode;
					} else if (ws_message_ && opcode == 0x1) {
						try {
							ws_message_(*conn, msg);
						} catch (const std::exception& e) {
							fprintf(stderr, "[ws] message handler error: %s\n", e.what());
						}
					}
					break;
				}
				case 0x8:  // close
					conn->enqueue_frame(0x8, nullptr, 0);
					alive = false;
					break;
				case 0x9:  // ping
					conn->enqueue_frame(0xA, payload.data(), payload.size());
					break;
				case 0xA:  // pong
					break;
				default:
					break;
			}
		}

		{
			std::lock_guard<std::mutex> lock(clients_mutex_);
			clients_.erase(conn->id);
		}
		conn->closed.store(true);
		if (ws_close_) ws_close_(conn->id);
		// close the socket FIRST (exactly once — the sender thread may have
		// already closed it after a failed send, and a second closesocket on a
		// recycled fd could kill an unrelated connection): it aborts an
		// in-flight blocking send immediately, then the flush thread drains-exits
		// and the join is bounded — no detached thread can outlive the connection
		conn->close_sock();
		conn->stopping.store(true);
		conn->q_cv.notify_all();
		if (conn->sender.joinable()) conn->sender.join();
		// conn (shared_ptr) is released when the last broadcast holding it
		// finishes — no fixed grace sleep, no manual delete
		// done is set by ClientDoneGuard on any return path
		return;
	}

	// plain http
	HttpResponse res;
	if (http_handler_) {
		res = http_handler_(req);
	} else {
		res.status = 404;
		res.body = "no handler";
	}
	// conditional GET: an unchanged asset answers 304 with an empty body, so the
	// renderer reuses its cached copy (including its parsed/compiled script cache)
	// instead of re-downloading and re-parsing ~220KB of js/css on every launch
	if (res.status == 200 && !res.etag.empty()) {
		auto inm = req.headers.find("if-none-match");
		if (inm != req.headers.end() && inm->second == res.etag) {
			res.status = 304;
			res.body.clear();
		}
	}
	std::string head = "HTTP/1.1 " + std::to_string(res.status) + " " + status_text(res.status) + "\r\n";
	head += "Content-Type: " + res.content_type + "\r\n";
	head += "Content-Length: " + std::to_string(res.body.size()) + "\r\n";
	if (res.cache_no_store) {
		head += "Cache-Control: no-store\r\n";
	} else {
		// revalidate each load, but let the client reuse the stored bytes
		head += "Cache-Control: no-cache\r\n";
		if (!res.etag.empty()) head += "ETag: " + res.etag + "\r\n";
	}
	head += "Connection: close\r\n\r\n";
	std::string out = head + res.body;
	sock_send_all(sock, out.data(), out.size());
	sock_close(sock);
}

}  // namespace phi
