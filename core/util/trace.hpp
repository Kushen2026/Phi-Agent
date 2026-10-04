// Temporary diagnostics (systematic-debugging phase 1): cheap timestamped
// stderr tracing, enabled only when PHI_TRACE is set. Zero cost otherwise.
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace phi::trace {
inline bool enabled() {
	static const bool on = std::getenv("PHI_TRACE") != nullptr;
	return on;
}
inline long long now_ms() {
	using namespace std::chrono;
	return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}
}  // namespace phi::trace

#define PHI_TRACE(...)                                                        \
	do {                                                                      \
		if (phi::trace::enabled()) {                                          \
			char buf_[1024];                                                  \
			int n_ = snprintf(buf_, sizeof buf_, "[t%lld] ",                  \
				(long long)phi::trace::now_ms());                             \
			if (n_ > 0 && n_ < (int)sizeof buf_) {                            \
				int m_ = snprintf(buf_ + n_, sizeof buf_ - (size_t)n_,        \
					__VA_ARGS__);                                             \
				if (m_ > 0) {                                                 \
					size_t cap_ = sizeof buf_ - (size_t)n_ - 1;               \
					/* truncated user-format portion: overwrite the tail      \
					   with a marker so the log reader can tell silent        \
					   drops from real output */                              \
					if ((size_t)m_ >= cap_) {                                 \
						static const char mark_[] = " ... (truncated)";       \
						size_t mk_ = sizeof mark_ - 1;                        \
						if (mk_ < cap_) {                                     \
							memcpy(buf_ + n_ + cap_ - mk_, mark_, mk_);       \
							buf_[n_ + cap_] = '\0';                           \
						}                                                     \
					}                                                         \
					fputs(buf_, stderr);                                      \
					fputc('\n', stderr);                                      \
				}                                                             \
			}                                                                 \
		}                                                                     \
	} while (0)
