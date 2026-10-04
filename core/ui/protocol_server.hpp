// ProtocolServer: binds the loopback HttpServer and dispatches the web-ui
// JSON protocol (requests in protocol.ts) to the agent session services.
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "agent/agent.hpp"

namespace phi {

class ProtocolServer {
public:
	ProtocolServer();
	~ProtocolServer();

	// base_dir contains ui/, assets/, data/
	bool start(const std::string& base_dir);
	void shutdown();
	int port() const;

	// native folder picker injected by the desktop shell; returns nullopt on cancel
	void set_folder_picker(std::function<std::optional<std::string>(const std::string& initial)> picker);

	// dev-only: inject a fake provider stream for repro probes (must be called
	// after start(), before any client sends create_session)
	void set_stream_fn(ProviderStreamFn fn);

	// dev-only: run without a window (reads requests over the same loopback server)
	void wait_idle() const;

	struct Impl;  // defined in protocol_server.cpp
private:
	std::unique_ptr<Impl> impl_;
};

}  // namespace phi
