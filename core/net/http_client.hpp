// HTTP client: raw sockets + schannel TLS (no external dependency).
// Used for LLM provider calls (SSE streaming responses).
//
// NOTE ON IMPLEMENTATION: this used to be a WinHTTP client. On Windows hosts
// with security-software network filtering, WinHttpReadData never delivered
// response body bytes incrementally — everything was buffered until the
// connection closed (verified with a loopback SSE server: raw sockets stream
// per-chunk, WinHTTP delivers all at once at the end). That buffering made
// token-by-token streaming impossible AND made abort hang until the LLM
// response finished. Raw sockets bypass it and give deterministic cancel:
// closing the socket tears down the stream immediately.
#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>

namespace phi {

struct HttpStreamResult {
	bool ok = false;          // transport-level success (got a response)
	int status = 0;           // http status
	std::string error;        // transport error message
	std::string body;         // full body when not streaming callback (error bodies)
	std::map<std::string, std::string> headers;  // response headers (lowercase keys)
};

// POST with streaming response. on_chunk receives decoded response body bytes
// as they arrive (SSE events). Return false from on_chunk to stop reading and
// tear down the connection (used for abort). `cancelled` is polled between
// network reads; return true to stop the same way (covers silent streams).
// headers: key -> value (no CRLF).
HttpStreamResult http_post_stream(
	const std::string& url,
	const std::vector<std::pair<std::string, std::string>>& headers,
	const std::string& body,
	const std::function<bool(const char* data, size_t len)>& on_chunk,
	int idle_timeout_ms = 300000,
	const std::function<bool()>& cancelled = {},
	bool allow_untrusted = false);

// method-generic request (GET for the web tools; body usually empty).
// Accumulates the full body and exposes response headers (redirect Location).
HttpStreamResult http_request(
	const std::string& method,
	const std::string& url,
	const std::vector<std::pair<std::string, std::string>>& headers,
	const std::string& body,
	int idle_timeout_ms = 300000,
	const std::function<bool()>& cancelled = {},
	bool allow_untrusted = false);

// Simple blocking GET (unused today, kept for diagnostics endpoints).
HttpStreamResult http_get(const std::string& url, int idle_timeout_ms = 30000);

struct ParallelRequest {
	std::string url;
	std::string method;
	std::vector<std::pair<std::string, std::string>> headers;
	std::string body;
	int idle_timeout_ms = 300000;
	std::function<bool()> cancelled;
	bool allow_untrusted = false;
};

struct ParallelResult {
	bool success;
	int index; // original index in requests vector
	HttpStreamResult result;
};

// Run multiple HTTP requests in parallel using threads.
// Returns results in the same order as input requests.
std::vector<ParallelResult> http_parallel_requests(
	const std::vector<ParallelRequest>& requests,
	int concurrent_limit = 4);

}  // namespace phi
