// phi base utilities: strings, paths, files, time, ids, encoding.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace phi {

// ── strings ────────────────────────────────────────────────────────────────
bool starts_with(std::string_view s, std::string_view prefix);
bool ends_with(std::string_view s, std::string_view suffix);
bool contains(std::string_view s, std::string_view needle);
std::string to_lower(std::string_view s);
std::string trim(std::string_view s);
std::string replace_all(std::string_view s, std::string_view from, std::string_view to);
std::vector<std::string> split(std::string_view s, char sep);
std::string join(const std::vector<std::string>& parts, std::string_view sep);

// ── paths (windows-oriented, accepts / and \) ──────────────────────────────
std::string path_join(std::string_view a, std::string_view b);
std::string path_dirname(std::string_view p);
std::string path_basename(std::string_view p);
std::string path_normalize(std::string_view p);   // lexical clean, keeps case
std::string path_absolute(std::string_view p);    // resolves via cwd, no symlink resolution
std::string path_backslash(std::string_view p);   // forward slashes -> backslashes
bool path_exists(std::string_view p);
bool path_is_dir(std::string_view p);

// ── files ──────────────────────────────────────────────────────────────────
bool read_file(std::string_view path, std::string& out);
std::optional<std::string> read_file(std::string_view path);
bool read_file_bytes(std::string_view path, std::vector<uint8_t>& out);
bool write_file(std::string_view path, std::string_view data);
bool append_file(std::string_view path, std::string_view data);
// write via tmp file + rename so a crash cannot corrupt the target
bool write_file_atomic(std::string_view path, std::string_view data);
bool mkdirs(std::string_view path);
bool remove_file(std::string_view path);
// move to Windows recycle bin (recoverable); returns 0 on success, otherwise an
// SHFileOperation error code (or -1 when the operation was aborted/partially failed
// without an error code); detail receives a human-readable hint when non-null.
int recycle_path(std::string_view path, std::string* detail = nullptr);
std::vector<std::string> list_dir(std::string_view dir);        // names, entries only
int64_t file_mtime_ms(std::string_view path);

// ── time ───────────────────────────────────────────────────────────────────
int64_t now_ms();

// ── ids / encoding ─────────────────────────────────────────────────────────
std::string uuid4();
std::string base64_encode(std::string_view data);
std::optional<std::string> base64_decode(std::string_view data);
std::string base64_encode_bytes(const std::vector<uint8_t>& data);
std::string sha1_hex(std::string_view data);   // used by the websocket handshake

// ── wide/utf8 conversion ───────────────────────────────────────────────────
std::wstring utf8_to_wide(std::string_view s);
std::string wide_to_utf8(const std::wstring& w);

// ── environment ────────────────────────────────────────────────────────────
std::optional<std::string> get_env(std::string_view name);
// directory containing the running exe (no trailing slash)
std::string exe_dir_path();

// ── processes ──────────────────────────────────────────────────────────────
struct ExecResult {
	int exit_code = 0;
	std::string out;
	std::string err;
	bool timed_out = false;
};
// run a command (utf-8 argv0/args, raw CreateProcess, no shell)
ExecResult exec_capture(const std::string& cmdline, std::string_view cwd, int timeout_ms);
// async process handle for streaming tool output (bash tool)
struct ChildProcess;
// spawn with piped stdout+stderr (merged). Returns nullptr on failure.
ChildProcess* spawn_pipe(const std::string& cmdline, std::string_view cwd);
// spawn through a shell (Git Bash if installed, else cmd.exe /c) so shell
// syntax (pipes, &&, redirects, Unix coreutils) works; utf-8 command + cwd
ChildProcess* spawn_shell(const std::string& cmdline, std::string_view cwd);
// non-blocking read of new output; false when child exited and buffer drained
bool child_poll(ChildProcess* p, std::string& chunk);
bool child_alive(ChildProcess* p);
int child_wait(ChildProcess* p);
void child_kill(ChildProcess* p);
void child_free(ChildProcess* p);

// ── json forward decl (avoid including json.hpp everywhere) ────────────────
class JsonValue;

}  // namespace phi
