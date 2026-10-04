// phi base utilities implementation (win32).
#include "util/base.hpp"

#include <windows.h>
#include <shellapi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>

namespace phi {

// ── strings ────────────────────────────────────────────────────────────────

bool starts_with(std::string_view s, std::string_view prefix) {
	return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}
bool ends_with(std::string_view s, std::string_view suffix) {
	return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}
bool contains(std::string_view s, std::string_view needle) {
	return s.find(needle) != std::string_view::npos;
}
std::string to_lower(std::string_view s) {
	std::string out(s);
	std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return (char)std::tolower(c); });
	return out;
}
std::string trim(std::string_view s) {
	size_t b = 0, e = s.size();
	while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) b++;
	while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) e--;
	return std::string(s.substr(b, e - b));
}
std::string replace_all(std::string_view s, std::string_view from, std::string_view to) {
	if (from.empty()) return std::string(s);
	std::string out;
	out.reserve(s.size());
	size_t pos = 0;
	while (true) {
		size_t hit = s.find(from, pos);
		if (hit == std::string_view::npos) {
			out.append(s.substr(pos));
			break;
		}
		out.append(s.substr(pos, hit - pos));
		out.append(to);
		pos = hit + from.size();
	}
	return out;
}
std::vector<std::string> split(std::string_view s, char sep) {
	std::vector<std::string> parts;
	size_t pos = 0;
	while (true) {
		size_t hit = s.find(sep, pos);
		if (hit == std::string_view::npos) {
			parts.emplace_back(s.substr(pos));
			break;
		}
		parts.emplace_back(s.substr(pos, hit - pos));
		pos = hit + 1;
	}
	return parts;
}
std::string join(const std::vector<std::string>& parts, std::string_view sep) {
	std::string out;
	for (size_t i = 0; i < parts.size(); i++) {
		if (i > 0) out.append(sep);
		out.append(parts[i]);
	}
	return out;
}

// ── paths ──────────────────────────────────────────────────────────────────

static char path_sep() { return '/'; }

std::string path_join(std::string_view a, std::string_view b) {
	std::string out(a);
	// remember whether `a` was nothing but a root, because stripping the
	// trailing separator would otherwise empty it out and lose the root
	bool a_is_bare_root =
		(out.size() == 1 && (out[0] == '/' || out[0] == '\\')) ||
		(out.size() == 2 && out[1] == ':' && (out[0] == '/' || std::isalpha((unsigned char)out[0]))) ||
		(out.size() == 3 && out[1] == ':' && (out[2] == '/' || out[2] == '\\'));
	while (!out.empty() && (out.back() == '/' || out.back() == '\\')) out.pop_back();
	if (out.empty() && !a_is_bare_root) return std::string(b);
	if (a_is_bare_root && out.empty()) {
		// "/" -> "/", "C:" stays "C:", "C:/" -> "C:"
		out = std::string(a.substr(0, 1));
		if (a.size() >= 2 && a[1] == ':') {
			out = std::string(a.substr(0, 2));  // "C:"
		}
	}
	// if out is already a root (e.g. "/"), don't add another separator
	if (!out.empty() && (out.back() == '/' || out.back() == '\\')) {
		if (!b.empty() && (b.front() == '/' || b.front() == '\\')) {
			return out + std::string(b);
		}
		return out + std::string(b);  // b starts with separator: just concatenate
	}
	if (!b.empty() && (b.front() == '/' || b.front() == '\\')) return out + std::string(b);
	out.push_back(path_sep());
	out.append(b);
	return out;
}

std::string path_dirname(std::string_view p) {
	size_t pos = p.find_last_of("/\\");
	if (pos == std::string_view::npos) return "";
	if (pos == 0) return std::string(p.substr(0, 1));
	return std::string(p.substr(0, pos));
}

std::string path_basename(std::string_view p) {
	size_t pos = p.find_last_of("/\\");
	return pos == std::string_view::npos ? std::string(p) : std::string(p.substr(pos + 1));
}

std::string path_normalize(std::string_view p) {
	// lexical: collapse //, resolve . and .. (keeps leading \\?\ and drive letters)
	std::string s(p);
	std::replace(s.begin(), s.end(), '\\', '/');
	bool unc = starts_with(s, "//");
	std::vector<std::string> stack;
	for (auto& part : split(s, '/')) {
		if (part.empty() || part == ".") continue;
		// a drive root ("C:") is a boundary that ".." must not cross
		bool is_drive = (part.size() == 2 && part[1] == ':');
		if (part == ".." && !stack.empty() && stack.back() != ".." &&
			!(stack.back().size() == 2 && stack.back()[1] == ':')) {
			stack.pop_back();
		} else {
			stack.push_back(part);
		}
		(void)is_drive;
	}
	std::string out = join(stack, "/");
	if (unc) out = "//" + out;
	if ((p.size() >= 2 && p[1] == ':') || starts_with(p, "/") || starts_with(p, "\\")) {
		if (starts_with(p, "//")) {
			// keep as-is
		} else if (!p.empty() && (p[0] == '/' || p[0] == '\\') && out.compare(0, 1, "/") != 0) {
			out = "/" + out;
		}
	}
	if (out.empty()) out = ".";
	return out;
}

std::string path_absolute(std::string_view p) {
	// p.size() guard: p[1] on a 1-char input reads out of bounds (UB)
	if (p.size() >= 2 && p[0] == '/' && p[1] == '/') return path_normalize(p);  // //server/share
	// real drive-absolute paths (C:/x, C:\x); a drive-RELATIVE "C:foo" falls
	// through so it resolves against the process cwd instead of the per-drive one
	if (p.size() >= 3 && p[1] == ':' && (p[2] == '/' || p[2] == '\\')) return path_normalize(p);
	std::wstring full = utf8_to_wide(p);
	std::replace(full.begin(), full.end(), L'/', L'\\');
	wchar_t buf[MAX_PATH * 2] = {0};
	DWORD n = GetFullPathNameW(full.c_str(), MAX_PATH * 2, buf, nullptr);
	if (n == 0 || n >= MAX_PATH * 2) return path_normalize(p);
	return wide_to_utf8(buf);
}

std::string path_backslash(std::string_view p) {
	std::string s(p);
	std::replace(s.begin(), s.end(), '/', '\\');
	return s;
}

bool path_exists(std::string_view p) {
	std::wstring full = utf8_to_wide(path_backslash(p));
	DWORD attrs = GetFileAttributesW(full.c_str());
	return attrs != INVALID_FILE_ATTRIBUTES;
}
bool path_is_dir(std::string_view p) {
	std::wstring full = utf8_to_wide(path_backslash(p));
	DWORD attrs = GetFileAttributesW(full.c_str());
	return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY);
}

// ── files ──────────────────────────────────────────────────────────────────

static HANDLE open_read(std::string_view path) {
	std::wstring full = utf8_to_wide(path_backslash(path));
	return CreateFileW(full.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
		nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
}

bool read_file(std::string_view path, std::string& out) {
	HANDLE h = open_read(path);
	if (h == INVALID_HANDLE_VALUE) return false;
	char buf[65536];
	DWORD got = 0;
	out.clear();
	bool ok = true;
	for (;;) {
		if (!ReadFile(h, buf, sizeof(buf), &got, nullptr)) { ok = false; break; }
		if (got == 0) break;
		out.append(buf, got);
	}
	CloseHandle(h);
	if (!ok) { out.clear(); return false; }
	return true;
}
std::optional<std::string> read_file(std::string_view path) {
	std::string out;
	if (!read_file(path, out)) return std::nullopt;
	return out;
}
bool read_file_bytes(std::string_view path, std::vector<uint8_t>& out) {
	std::string tmp;
	if (!read_file(path, tmp)) return false;
	out.assign(tmp.begin(), tmp.end());
	return true;
}

bool write_file(std::string_view path, std::string_view data) {
	std::wstring full = utf8_to_wide(path_backslash(path));
	HANDLE h = CreateFileW(full.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL, nullptr);
	if (h == INVALID_HANDLE_VALUE) return false;
	DWORD wrote = 0;
	BOOL ok = TRUE;
	while (!data.empty()) {
		ok = WriteFile(h, data.data(), (DWORD)data.size(), &wrote, nullptr);
		if (!ok || wrote == 0) {
			ok = FALSE;
			break;
		}
		data.remove_prefix(wrote);
	}
	FlushFileBuffers(h);
	CloseHandle(h);
	return ok == TRUE;
}

// true append at EOF (session JSONL): no truncate, no read-modify-write —
// concurrent appends from worker + WS threads cannot lose each other's data
// and a crash mid-write destroys at most the last partial line
bool append_file(std::string_view path, std::string_view data) {
	std::wstring full = utf8_to_wide(path_backslash(path));
	HANDLE h = CreateFileW(full.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr, OPEN_ALWAYS,
		FILE_ATTRIBUTE_NORMAL, nullptr);
	if (h == INVALID_HANDLE_VALUE) return false;
	DWORD wrote = 0;
	BOOL ok = TRUE;
	while (!data.empty()) {
		ok = WriteFile(h, data.data(), (DWORD)data.size(), &wrote, nullptr);
		if (!ok) break;
		data.remove_prefix(wrote);
	}
	// flush to disk: session JSONL is crash-recovery data — without this the
	// OS could lose already-"written" lines (e.g. a full assistant turn) when
	// the process or machine dies, leaving a session that ends mid-turn
	if (ok == TRUE) FlushFileBuffers(h);
	CloseHandle(h);
	return ok == TRUE;
}

bool write_file_atomic(std::string_view path, std::string_view data) {
	// unique tmp name: two WS threads saving different stores concurrently used
	// to truncate each other's temp file (fixed ".tmp" name)
	static std::atomic<uint64_t> tmp_counter{0};
	std::string tmp = std::string(path) + ".tmp" + std::to_string(++tmp_counter) + "-" +
		std::to_string(GetCurrentProcessId());
	if (!write_file(tmp, data)) return false;
	std::wstring from = utf8_to_wide(path_backslash(tmp));
	std::wstring to = utf8_to_wide(path_backslash(path));
	if (!MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING)) {
		DeleteFileW(from.c_str());
		return false;
	}
	return true;
}

bool mkdirs(std::string_view path) {
	std::wstring full = utf8_to_wide(path_backslash(path_normalize(path)));
	if (full.empty() || path_exists(wide_to_utf8(full))) return true;
	// walk the path creating each segment
	for (size_t i = 1; i <= full.size(); i++) {
		if (i == full.size() || full[i] == L'\\') {
			std::wstring part = full.substr(0, i);
			if (part.size() == 2 && part[1] == L':') continue;  // drive root
			if (part.empty() || path_exists(wide_to_utf8(part))) continue;
			if (!CreateDirectoryW(part.c_str(), nullptr) &&
				GetLastError() != ERROR_ALREADY_EXISTS) {
				// real failure (bad name, access denied, path too long): report it
				// instead of silently returning whatever the final probe says
				return false;
			}
		}
	}
	return path_is_dir(wide_to_utf8(full));
}

bool remove_file(std::string_view path) {
	std::wstring full = utf8_to_wide(path_backslash(path));
	if (path_is_dir(wide_to_utf8(full))) {
		return RemoveDirectoryW(full.c_str()) != 0;
	}
	return DeleteFileW(full.c_str()) != 0;
}

int recycle_path(std::string_view path, std::string* detail) {
	std::string full = path_backslash(path);
	std::wstring wide = utf8_to_wide(full);
	std::vector<wchar_t> buf(wide.begin(), wide.end());
	buf.push_back(L'\0');
	buf.push_back(L'\0');  // double-null terminated list
	SHFILEOPSTRUCTW op = {};
	op.hwnd = nullptr;
	op.wFunc = FO_DELETE;
	op.pFrom = buf.data();
	op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT;
	int rc = SHFileOperationW(&op);
	// SHFileOperation returns 0 on success BUT fAnyOperationsAborted may be set
	// when some files could not be moved to the recycle bin (e.g. open files).
	// In that case we must report failure so callers like delete_session() know
	// the operation did not fully succeed.
	if (rc == 0) {
		if (op.fAnyOperationsAborted) {
			if (detail) *detail = "some files could not be moved to the recycle bin";
			return -1;
		}
		return 0;
	}
	if (detail) {
		char buf2[96];
		snprintf(buf2, sizeof(buf2), "SHFileOperationW rc=0x%x aborted=%d", rc, (int)op.fAnyOperationsAborted);
		*detail = buf2;
	}
	return rc;
}

std::vector<std::string> list_dir(std::string_view dir) {
	// FindFirstFileA would interpret the pattern in the ANSI codepage and fail
	// on non-ASCII paths (e.g. a session dir containing CJK characters),
	// silently returning an empty listing; use the wide API like every other
	// path helper here.
	std::vector<std::string> out;
	std::wstring pattern = utf8_to_wide(path_backslash(path_join(dir, "*")));
	WIN32_FIND_DATAW fd;
	HANDLE h = FindFirstFileW(pattern.c_str(), &fd);
	if (h == INVALID_HANDLE_VALUE) return out;
	do {
		std::string name = wide_to_utf8(fd.cFileName);
		if (name == "." || name == "..") continue;
		out.push_back(std::move(name));
	} while (FindNextFileW(h, &fd));
	// FindNextFileW returns FALSE both at the normal end of the enumeration AND
	// on a real failure (directory removed / access revoked mid-walk). Only
	// ERROR_NO_MORE_FILES means "done"; anything else means the listing above
	// is INCOMPLETE — previously that was silently treated as a full listing.
	bool complete = GetLastError() == ERROR_NO_MORE_FILES;
	FindClose(h);
	(void)complete;  // signature cannot signal partialness; best-effort view returned
	return out;
}

int64_t file_mtime_ms(std::string_view path) {
	HANDLE h = open_read(path);
	if (h == INVALID_HANDLE_VALUE) return 0;
	FILETIME ft;
	int64_t ms = 0;
	if (GetFileTime(h, nullptr, nullptr, &ft)) {
		ULARGE_INTEGER li;
		li.HighPart = ft.dwHighDateTime;
		li.LowPart = ft.dwLowDateTime;
		ms = (int64_t)((li.QuadPart - 116444736000000000ULL) / 10000);
	}
	CloseHandle(h);
	return ms;
}

// ── time ───────────────────────────────────────────────────────────────────

int64_t now_ms() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
}

// ── ids / encoding ─────────────────────────────────────────────────────────

static std::mt19937_64& rng() {
	static std::mt19937_64 engine([] {
		std::random_device rd;
		return ((uint64_t)rd() << 32) | rd();
	}());
	return engine;
}

std::string uuid4() {
	std::uniform_int_distribution<uint32_t> dist(0, 0xffffffffu);
	uint32_t a = dist(rng()), b = dist(rng()), c = dist(rng()), d = dist(rng());
	char buf[40];
	 snprintf(buf, sizeof(buf), "%08x-%04x-4%03x-%04x-%08x%04x", a, (b >> 16) & 0xffff, b & 0x0fffu,
		((c >> 16) & 0x3fff) | 0x8000, c & 0xffff, d >> 16);
	return buf;
}

static const char* B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(std::string_view data) {
	return base64_encode_bytes(std::vector<uint8_t>(data.begin(), data.end()));
}

std::string base64_encode_bytes(const std::vector<uint8_t>& data) {
	std::string out;
	out.reserve((data.size() + 2) / 3 * 4);
	for (size_t i = 0; i < data.size(); i += 3) {
		uint32_t v = (uint32_t)data[i] << 16;
		if (i + 1 < data.size()) v |= (uint32_t)data[i + 1] << 8;
		if (i + 2 < data.size()) v |= (uint32_t)data[i + 2];
		out.push_back(B64[(v >> 18) & 63]);
		out.push_back(B64[(v >> 12) & 63]);
		out.push_back(i + 1 < data.size() ? B64[(v >> 6) & 63] : '=');
		out.push_back(i + 2 < data.size() ? B64[v & 63] : '=');
	}
	return out;
}

std::optional<std::string> base64_decode(std::string_view data) {
	static int8_t rev[256];
	static bool init = [] {
		memset(rev, -1, sizeof(rev));
		for (int i = 0; i < 64; i++) rev[(uint8_t)B64[i]] = (int8_t)i;
		return true;
	}();
	(void)init;
	std::string out;
	uint32_t acc = 0;
	int bits = 0;
	int pad_count = 0;
	for (size_t i = 0; i < data.size(); i++) {
		char c = data[i];
		if (c == '=') {
			// '=' is padding: it consumes the current (partial) 6-bit group and
			// ends the payload — anything except further '=' after it is invalid.
			// This rejects "AA==AA" (decoded as "i" before the fix).
			if (pad_count >= 2) return std::nullopt;
			// first '=' is only legal when the unflushed bits are 2 (3 data chars
			// + 1 pad) or 4 (2 data chars + 2 pads). bits==6 means only one data
			// char preceded, bits==0 means a whole 4-char group preceded — both
			// are illegal, and were silently accepted before this check.
			if (pad_count == 0 && bits != 2 && bits != 4) return std::nullopt;
			pad_count++;
			bits -= 6;  // the pad group contributes no data bits
			continue;
		}
		if (c == '\n' || c == '\r' || c == ' ') continue;
		if (pad_count > 0) return std::nullopt;  // data after padding
		int8_t v = rev[(uint8_t)c];
		if (v < 0) return std::nullopt;
		acc = (acc << 6) | (uint32_t)v;
		bits += 6;
		if (bits >= 8) {
			bits -= 8;
			out.push_back((char)((acc >> bits) & 0xff));
		}
	}
	if (pad_count > 2) return std::nullopt;  // more than 2 padding chars
	return out;
}

std::string sha1_hex(std::string_view data) {
	// minimal SHA-1 (websocket handshake only)
	uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
	std::vector<uint8_t> msg(data.begin(), data.end());
	uint64_t bitlen = (uint64_t)msg.size() * 8;
	msg.push_back(0x80);
	while (msg.size() % 64 != 56) msg.push_back(0);
	for (int i = 7; i >= 0; i--) msg.push_back((uint8_t)(bitlen >> (i * 8)));

	auto rol = [](uint32_t v, int n) { return (v << n) | (v >> (32 - n)); };
	for (size_t off = 0; off < msg.size(); off += 64) {
		uint32_t w[80];
		for (int i = 0; i < 16; i++) {
			w[i] = ((uint32_t)msg[off + i * 4] << 24) | ((uint32_t)msg[off + i * 4 + 1] << 16) |
				((uint32_t)msg[off + i * 4 + 2] << 8) | (uint32_t)msg[off + i * 4 + 3];
		}
		for (int i = 16; i < 80; i++) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
		uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
		for (int i = 0; i < 80; i++) {
			uint32_t f, k;
			if (i < 20) { f = (b & c) | ((~b) & d); k = 0x5A827999; }
			else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
			else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
			else { f = b ^ c ^ d; k = 0xCA62C1D6; }
			uint32_t tmp = rol(a, 5) + f + e + k + w[i];
			e = d; d = c; c = rol(b, 30); b = a; a = tmp;
		}
		h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
	}
	char out[41];
	for (int i = 0; i < 5; i++) snprintf(out + i * 8, 9, "%08x", h[i]);
	return out;
}

// ── wide/utf8 ──────────────────────────────────────────────────────────────

std::wstring utf8_to_wide(std::string_view s) {
	if (s.empty()) return L"";
	// MB_ERR_INVALID_CHARS: malformed utf-8 used to be silently replaced, so a
	// corrupted path "succeeded" with a wrong name. Fail loudly (empty result)
	// so the caller's file operation errors out instead of touching a mangled path.
	int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int)s.size(), nullptr, 0);
	if (n <= 0) return L"";
	std::wstring w((size_t)n, L'\0');
	if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int)s.size(), w.data(), n) <= 0)
		return L"";
	return w;
}

#ifndef WC_ERR_INVALID_CHARS
#define WC_ERR_INVALID_CHARS 0x0080
#endif

std::string wide_to_utf8(const std::wstring& w) {
	if (w.empty()) return "";
	// strict conversion first (WC_ERR_INVALID_CHARS flags lone surrogates etc.);
	// fall back to substitution only when strict fails, so real filesystem
	// names still round-trip instead of vanishing
	int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
	DWORD flags = WC_ERR_INVALID_CHARS;
	if (n <= 0) {
		n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
		flags = 0;
		if (n <= 0) return "";
	}
	std::string s((size_t)n, '\0');
	if (WideCharToMultiByte(CP_UTF8, flags, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr) <= 0)
		return "";
	return s;
}

// ── environment ────────────────────────────────────────────────────────────

std::optional<std::string> get_env(std::string_view name) {
	// Query required buffer size first (GetEnvironmentVariable returns 0 on failure,
	// or the size needed when the provided buffer is too small)
	DWORD required = GetEnvironmentVariableA(name.data(), nullptr, 0);
	if (required == 0) return std::nullopt;
	// required includes the null terminator, so allocate (required - 1) chars
	std::string result(required - 1, '\0');
	DWORD got = GetEnvironmentVariableA(name.data(), result.data(), required);
	if (got == 0 || got >= required) return std::nullopt;
	result.resize(got);
	return result;
}

std::string exe_dir_path() {
	wchar_t buf[MAX_PATH];
	DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
	if (n == 0 || n >= MAX_PATH) return ".";
	std::wstring w(buf, n);
	std::string exe = wide_to_utf8(w);
	return path_dirname(exe);
}

// ── processes ──────────────────────────────────────────────────────────────

ExecResult exec_capture(const std::string& cmdline, std::string_view cwd, int timeout_ms) {
	ExecResult result;
	ChildProcess* child = spawn_pipe(cmdline, cwd);
	if (!child) {
		result.exit_code = -1;
		result.err = "failed to spawn process";
		return result;
	}
	int64_t deadline = now_ms() + timeout_ms;
	while (true) {
		std::string chunk;
		bool more = child_poll(child, chunk);
		if (!chunk.empty()) {
			// merge stderr markers are not distinguished by the pipe; treat as output
			result.out += chunk;
		}
		if (!child_alive(child) && !more) break;
		if (now_ms() > deadline) {
			child_kill(child);
			result.timed_out = true;
			break;
		}
		Sleep(5);
	}
	result.exit_code = child_wait(child);
	child_free(child);
	return result;
}

struct ChildProcess {
	HANDLE job = nullptr;
	HANDLE proc = nullptr;
	HANDLE thread = nullptr;
	HANDLE pipe_read = nullptr;
	HANDLE pipe_write = nullptr;
	bool exited = false;
	std::string pending;
	DWORD exit_code = 0;
};

static const size_t PIPE_BUF = 64 * 1024;

// quote one argument per Windows command-line rules (for "bash" -c "<arg>")
static std::string win_quote_arg(const std::string& arg) {
	std::string out = "\"";
	size_t backslashes = 0;
	for (char c : arg) {
		if (c == '\\') {
			backslashes++;
			continue;
		}
		if (c == '"') {
			out.append(backslashes * 2 + 1, '\\');
			out.push_back('"');
		} else {
			out.append(backslashes, '\\');
			out.push_back(c);
		}
		backslashes = 0;
	}
	out.append(backslashes * 2, '\\');
	out.push_back('"');
	return out;
}

// resolve bash.exe (Git for Windows) once; empty when not installed
static const std::wstring& bash_exe_path() {
	static std::wstring cached = [] {
		const wchar_t* candidates[] = {
			L"C:\\Program Files\\Git\\bin\\bash.exe",
			L"C:\\Program Files\\Git\\usr\\bin\\bash.exe",
			L"C:\\Program Files (x86)\\Git\\bin\\bash.exe",
		};
		for (auto* c : candidates) {
			DWORD attrs = GetFileAttributesW(c);
			if (attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY)) return std::wstring(c);
		}
		wchar_t localappdata[MAX_PATH];
		if (GetEnvironmentVariableW(L"LOCALAPPDATA", localappdata, MAX_PATH) > 0) {
			std::wstring p = std::wstring(localappdata) + L"\\Programs\\Git\\bin\\bash.exe";
			DWORD attrs = GetFileAttributesW(p.c_str());
			if (attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY)) return p;
		}
		wchar_t found[MAX_PATH];
		if (SearchPathW(nullptr, L"bash.exe", nullptr, MAX_PATH, found, nullptr) > 0) return std::wstring(found);
		return std::wstring();
	}();
	return cached;
}

static ChildProcess* spawn_process(const std::wstring& wcmdline, const std::string& cwd) {
	auto* child = new ChildProcess();
	SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
	if (!CreatePipe(&child->pipe_read, &child->pipe_write, &sa, 0)) {
		delete child;
		return nullptr;
	}
	// make read end non-inheritable
	SetHandleInformation(child->pipe_read, HANDLE_FLAG_INHERIT, 0);

	// stdin from NUL: a command that reads stdin must see EOF, not the app's
	// (possibly absent) console — otherwise it blocks forever waiting for input
	HANDLE stdin_nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
		&sa, OPEN_EXISTING, 0, nullptr);
	if (stdin_nul == INVALID_HANDLE_VALUE) stdin_nul = nullptr;

	STARTUPINFOW si{};
	si.cb = sizeof(si);
	si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
	si.wShowWindow = SW_HIDE;
	si.hStdInput = stdin_nul;
	si.hStdOutput = child->pipe_write;
	si.hStdError = child->pipe_write;

	// kill-on-close job: child_kill/free terminate the WHOLE tree (npm -> node
	// chains would otherwise keep the pipe write end open and wedge the reader)
	child->job = CreateJobObjectW(nullptr, nullptr);
	if (child->job) {
		JOBOBJECT_EXTENDED_LIMIT_INFORMATION lim{};
		lim.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
		SetInformationJobObject(child->job, JobObjectExtendedLimitInformation, &lim, sizeof(lim));
	}

	PROCESS_INFORMATION pi{};
	std::wstring wdir = cwd.empty() ? std::wstring() : utf8_to_wide(path_backslash(std::string(cwd)));
	BOOL ok = CreateProcessW(nullptr, const_cast<LPWSTR>(wcmdline.c_str()), nullptr, nullptr, TRUE,
		CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT, nullptr,
		wdir.empty() ? nullptr : wdir.c_str(), &si, &pi);
	if (stdin_nul) CloseHandle(stdin_nul);
	CloseHandle(child->pipe_write);  // parent must not hold the write end (success AND failure paths)
	child->pipe_write = nullptr;     // failure path deletes the struct: never leave a stale handle behind
	if (!ok) {
		if (child->job) CloseHandle(child->job);
		CloseHandle(child->pipe_read);
		delete child;
		return nullptr;
	}
	child->proc = pi.hProcess;
	child->thread = pi.hThread;
	if (child->job) AssignProcessToJobObject(child->job, pi.hProcess);
	return child;
}

ChildProcess* spawn_pipe(const std::string& cmdline, std::string_view cwd) {
	// raw CreateProcessW: utf-8 command line and cwd survive non-ASCII paths
	// (the old CreateProcessA interpreted utf-8 as ANSI codepage and failed or
	// mangled every path containing Chinese characters)
	return spawn_process(utf8_to_wide(cmdline), std::string(cwd));
}

ChildProcess* spawn_shell(const std::string& cmdline, std::string_view cwd) {
	// bash tool entry: run through a real shell so pipes/redirections/&&/builtins
	// and Unix tools (ls, grep, rm, ...) work — the raw CreateProcess path only
	// accepts a single executable. Prefer Git Bash; fall back to cmd.exe.
	const std::wstring& bash = bash_exe_path();
	if (!bash.empty()) {
		// bash -c without a login shell keeps the inherited Windows PATH; add the
		// msys /usr/bin so Unix coreutils (ls, grep, sort, ...) resolve
		std::string script = "export PATH=\"/usr/bin:$PATH\"; " + cmdline;
		std::wstring wcmd = L"\"" + bash + L"\" -c " + utf8_to_wide(win_quote_arg(script));
		return spawn_process(wcmd, std::string(cwd));
	}
	std::string cmd_line = "cmd.exe /d /s /c " + cmdline;
	return spawn_process(utf8_to_wide(cmd_line), std::string(cwd));
}

bool child_poll(ChildProcess* p, std::string& chunk) {
	chunk.clear();
	if (!p->pending.empty()) {
		chunk.swap(p->pending);
		return true;
	}
	if (p->exited) return false;
	// PeekNamedPipe first: a plain ReadFile on an anonymous pipe BLOCKS until
	// data arrives or the write end closes — a silent child (ffmpeg, a server,
	// a long build) would wedge the caller past every deadline check and the
	// bash timeout could never fire ("stuck after one tool call")
	DWORD avail = 0;
	if (PeekNamedPipe(p->pipe_read, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) {
		char buf[PIPE_BUF];
		DWORD got = 0;
		if (ReadFile(p->pipe_read, buf, (DWORD)std::min<DWORD>(sizeof(buf), avail), &got, nullptr) && got > 0) {
			chunk.assign(buf, got);
			return true;
		}
	}
	// no data this tick: fold in exit state (child_alive drains leftovers into
	// pending) and hand them out on this or the next call
	if (!child_alive(p) && !p->pending.empty()) {
		chunk.swap(p->pending);
		return true;
	}
	return false;
}

bool child_alive(ChildProcess* p) {
	if (p->exited) return false;
	DWORD code = 0;
	if (GetExitCodeProcess(p->proc, &code) && code != STILL_ACTIVE) {
		// drain remaining pipe output before reporting exit
		char buf[PIPE_BUF];
		DWORD got = 0;
		while (PeekNamedPipe(p->pipe_read, nullptr, 0, nullptr, &got, nullptr) && got > 0) {
			DWORD read = 0;
			if (!ReadFile(p->pipe_read, buf, std::min(sizeof(buf), (size_t)got), &read, nullptr) || read == 0) break;
			p->pending.append(buf, read);
		}
		p->exit_code = code;
		p->exited = true;
		return false;
	}
	return true;
}

int child_wait(ChildProcess* p) {
	if (!p->exited) {
		WaitForSingleObject(p->proc, 15000);
		child_alive(p);
	}
	return (int)(int32_t)p->exit_code;
}

void child_kill(ChildProcess* p) {
	if (!p) return;
	// terminate the whole process tree: killing only the direct child leaves
	// grandchildren holding the pipe write end open (the reader would block
	// forever even after the kill)
	if (p->job && !p->exited) TerminateJobObject(p->job, 1);
	if (p->proc && !p->exited) TerminateProcess(p->proc, 1);
}

void child_free(ChildProcess* p) {
	if (!p) return;
	if (p->proc) CloseHandle(p->proc);
	if (p->thread) CloseHandle(p->thread);
	if (p->pipe_read) CloseHandle(p->pipe_read);
	// KILL_ON_JOB_CLOSE: any survivors in the tree die with the job handle
	if (p->job) CloseHandle(p->job);
	delete p;
}

}  // namespace phi
