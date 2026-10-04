#pragma once

#include <functional>
#include <optional>
#include <string>

namespace phi {

// Starts the protocol server, opens the main window hosting WebView2, runs the
// message loop until the window closes. Returns process exit code.
int app_main(const std::string& base_dir);

// Native folder picker (IFileDialog, FOS_PICKFOLDERS). Returns nullopt on cancel.
std::optional<std::string> pick_folder_dialog(const std::string& initial_dir);

}  // namespace phi
