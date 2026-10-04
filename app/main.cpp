#include "shell.hpp"
#include "../core/util/base.hpp"
#include "../core/bench/media_bench.hpp"
#include "../core/tools/tools_media.hpp"

#include <windows.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace phi;

// phi.exe is a GUI-subsystem binary (see makefile: -mwindows), so it has no
// console of its own. A redirected stdout (a pipe or a file from the calling
// shell) is inherited and already usable — reuse it. Only when there is neither
// a usable stream nor a parent console do we allocate one.
void attach_console() {
	HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
	bool have_stream = out && out != INVALID_HANDLE_VALUE &&
	                   GetFileType(out) != FILE_TYPE_UNKNOWN && GetConsoleWindow() == nullptr;
	if (have_stream) return;  // running `phi.exe --media-bench | ...` from a shell
	if (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole()) {
		freopen("CONOUT$", "w", stdout);
		freopen("CONOUT$", "w", stderr);
		freopen("CONIN$", "r", stdin);
		SetConsoleOutputCP(CP_UTF8);
	}
}

std::vector<std::string> command_line_args() {
	std::vector<std::string> args;
	int argc = 0;
	LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
	if (!wargv) return args;
	for (int i = 1; i < argc; i++) args.push_back(wide_to_utf8(wargv[i]));
	LocalFree(wargv);
	return args;
}

}  // namespace

extern "C" int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
	// base_dir = parent directory of exe_dir_path() (exe lives in <base>/bin)
	std::string base_dir = path_dirname(exe_dir_path());

	std::vector<std::string> args = command_line_args();
	if (!args.empty() && args[0] == "--media-bench") {
		attach_console();
		media::BenchOptions opts;
		std::string err;
		std::vector<std::string> rest(args.begin() + 1, args.end());
		if (!media::parse_media_bench_args(rest, &opts, &err)) {
			if (!err.empty()) fprintf(stderr, "error: %s\n", err.c_str());
			return 2;
		}
		int rc = media::media_bench_main(opts);
		fflush(stdout);
		return rc;
	}
	if (!args.empty() && args[0] == "--media-tool") {
		// Invoke one of the two media tools directly (no model in the loop), so the
		// tool layer - schema, argument checks, the graph it builds - can be run and
		// inspected from a shell. See core/tools/tools_media_cli.cpp.
		attach_console();
		std::vector<std::string> rest(args.begin() + 1, args.end());
		int rc = phi::media_tool_main(rest, base_dir);
		fflush(stdout);
		return rc;
	}

	return app_main(base_dir);
}
