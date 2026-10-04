// Loopback HTTP/1.1 + WebSocket server (the desktop app's UI bridge).
// Binds 127.0.0.1 only. Threads: one accept thread, one thread per client.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace phi {

struct HttpRequest {
	std::string method;
	std::string path;      // decoded, without query
	std::string query;     // raw query string
	std::map<std::string, std::string> headers;  // lower-cased keys
	std::string body;
};

struct HttpResponse {
	int status = 200;
	std::string content_type = "text/plain; charset=utf-8";
	std::string body;
	bool cache_no_store = true;
	// static-asset validator: non-empty means it is sent as ETag and compared
	// against If-None-Match (304 + empty body on a hit). The UI ships ~220KB of
	// js/css; without a validator every launch re-parses all of it from scratch.
	std::string etag;
};

struct WebSocketConnection {
	uint64_t id = 0;
	// implemented by the server; thread-safe, ASYNC: messages are queued and
	// flushed by a per-connection sender thread, so a slow/stalled renderer
	// never blocks the broadcasting thread (stream stalls of seconds = the old
	// blocking send hitting SO_SNDTIMEO while the renderer was busy parsing)
	void send_text(const std::string& text);
	// latest-wins enqueue: drops older queued coalescable messages (stream
	// patches carry full state, so older ones are redundant)
	void enqueue(std::string text, bool coalesce);
	// queue a pre-encoded ws frame (ping/pong) through the sender thread —
	// heartbeat pings must never block the reader thread on a stalled peer
	void enqueue_frame(int opcode, const uint8_t* data, size_t len);
	// close the underlying socket exactly once; safe from any thread
	void close_sock();
	// atomic + close-once: the sender thread (send failure) and the cleanup
	// paths (client_loop / server stop) could both closesocket() the same fd.
	// Windows recycles fd numbers aggressively, so the second close could hit a
	// freshly reused fd belonging to an unrelated socket — corrupting another
	// connection or crashing the process.
	std::atomic<int> sock{-1};
	// embedded (not pointers): a broadcast holds a shared_ptr, so the
	// connection — and these members — stay alive until the send finishes,
	// even if the client thread has already cleaned up its registration.
	std::mutex send_mutex;
	std::atomic<bool> closed{false};
	// outbound queue + sender thread
	struct OutMessage {
		std::string text;
		bool coalescable = false;
		int opcode = 0x1;  // 0x1 = text payload, -1 = pre-encoded raw frame
	};
	std::deque<OutMessage> out_queue;
	std::mutex q_mutex;
	std::condition_variable q_cv;
	std::atomic<bool> stopping{false};
	std::thread sender;

private:
	void enqueue_msg(OutMessage&& m);
};

class HttpServer {
public:
	using HttpHandler = std::function<HttpResponse(const HttpRequest&)>;
	using WsOpenHandler = std::function<void(WebSocketConnection&)>;
	using WsMessageHandler = std::function<void(WebSocketConnection&, const std::string&)>;
	using WsCloseHandler = std::function<void(uint64_t)>;

	HttpServer() = default;
	~HttpServer() { stop(); }

	// Bind 127.0.0.1; port 0 picks a free port. Returns false on failure.
	bool start(int port);
	void stop();
	int port() const { return port_; }

	void set_http_handler(HttpHandler h) { http_handler_ = std::move(h); }
	void set_ws_handlers(WsOpenHandler open, WsMessageHandler message, WsCloseHandler close) {
		ws_open_ = std::move(open);
		ws_message_ = std::move(message);
		ws_close_ = std::move(close);
	}
	// Send to every connected websocket client; thread-safe. broadcast()
	// preserves order (FIFO); broadcast_latest() coalesces per client — older
	// queued latest-wins messages are dropped in favor of the newest.
	void broadcast(const std::string& text);
	void broadcast_latest(const std::string& text);

private:
	void accept_loop();
	void client_loop(int sock, std::shared_ptr<std::atomic<bool>> done);
	// returns true when the connection was upgraded to websocket; on success
	// *out receives the registered connection (the caller MUST use this pointer
	// instead of looking it up by sock — two clients can race between register
	// and lookup)
	bool try_upgrade(int sock, HttpRequest& req, std::shared_ptr<WebSocketConnection>* out);

	int listen_sock_ = -1;
	std::atomic<int> port_{0};
	std::atomic<bool> running_{false};

	HttpHandler http_handler_;
	WsOpenHandler ws_open_;
	WsMessageHandler ws_message_;
	WsCloseHandler ws_close_;

	std::mutex clients_mutex_;
	std::map<uint64_t, std::shared_ptr<WebSocketConnection>> clients_;
	uint64_t next_client_id_ = 1;
	// track accept + client threads so stop() can join them before destruction.
	// Each entry owns the thread + a completion flag so accept_loop can clean
	// up finished clients without waiting for stop().
	struct ThreadEntry {
		std::thread t;
		std::shared_ptr<std::atomic<bool>> done = std::make_shared<std::atomic<bool>>(false);
	};
	std::mutex threads_mutex_;
	std::vector<ThreadEntry> threads_;
};

}  // namespace phi
