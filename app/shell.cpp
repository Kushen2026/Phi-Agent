#include <cstdio>

#include "shell.hpp"

#include "../core/ui/protocol_server.hpp"
#include "../core/util/base.hpp"
#include "../core/util/trace.hpp"
#include "../core/ui/ui_theme.hpp"

#include <windows.h>
#include <shobjidl.h>
#include <objbase.h>
#include <shlwapi.h>
#include <dwmapi.h>
#include "WebView2.h"

#include <cstring>
#include <cstdlib>
#include <cwchar>
#include <optional>
#include <string>

namespace phi {

// ── window plumbing ────────────────────────────────────────────────────────

static constexpr wchar_t kWndClassW[] = L"PhiMainWindow";
// safety net for the placeholder: if NavigationCompleted never arrives (blocked
// or failed navigation) the window must not keep showing the splash forever
static constexpr UINT_PTR kRevealTimerId = 1;
static constexpr UINT kRevealTimeoutMs = 2500;

static HINSTANCE g_hinst = nullptr;
static HBRUSH g_brush = nullptr;
// the user's theme colours the splash and the webview canvas from the first
// frame — with a light theme a dark splash would flash for the whole startup
static UiCanvasTheme g_theme;
static int g_port = 0;
static ICoreWebView2Controller* g_controller = nullptr;
static ICoreWebView2* g_webview = nullptr;
// false until the webview has produced its first frame: while false the window
// paints its own placeholder, so a double-click gives immediate feedback and the
// hand-off to the webview has no blank/white gap
static bool g_webview_shown = false;

static long long g_start_ms = 0;

// launch stage timings — emitted only when PHI_TRACE is set (zero cost otherwise)
static void stage(const char* name) {
	if (!trace::enabled()) return;
	fprintf(stderr, "[startup] %-24s +%lldms\n", name, trace::now_ms() - g_start_ms);
}

static void update_webview_bounds(HWND hwnd) {
	if (!g_controller) return;
	RECT rc{};
	if (!GetClientRect(hwnd, &rc)) return;
	g_controller->put_Bounds(rc);
}

// hand the window over to the webview: called on the first navigation completion
// (or by the fallback timer). Idempotent, safe from any thread that can be here.
static void reveal_webview() {
	if (g_webview_shown) return;
	g_webview_shown = true;
	HWND hwnd = FindWindowW(kWndClassW, nullptr);
	if (hwnd) KillTimer(hwnd, kRevealTimerId);
	if (g_controller) g_controller->put_IsVisible(TRUE);
	if (hwnd) {
		update_webview_bounds(hwnd);
		InvalidateRect(hwnd, nullptr, TRUE);
	}
	stage("webview shown");
}

// Splash painted between "window visible" and "webview first frame": the window
// is on screen within milliseconds of the double-click, so the several hundred
// ms spent starting the WebView2 runtime are no longer a blank window.
static void paint_placeholder(HWND hwnd) {
	PAINTSTRUCT ps{};
	HDC dc = BeginPaint(hwnd, &ps);
	if (!g_webview_shown) {
		RECT rc{};
		GetClientRect(hwnd, &rc);
		SetBkMode(dc, TRANSPARENT);

		// Φ mark, vertically centered; colours follow the active theme so the
		// splash looks like the UI it is about to be replaced by
		const COLORREF mark_color = g_theme.light ? RGB(76, 117, 111) : RGB(138, 190, 183);
		const COLORREF tip_color = g_theme.light ? RGB(122, 132, 128) : RGB(120, 126, 148);

		HFONT mark_font = CreateFontW(-104, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
			DEFAULT_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
			DEFAULT_PITCH, L"Segoe UI");
		HGDIOBJ old_font = mark_font ? SelectObject(dc, mark_font) : nullptr;
		SetTextColor(dc, mark_color);
		RECT mark = rc;
		mark.bottom = (rc.top + rc.bottom) / 2 + 36;
		DrawTextW(dc, L"\u03a6", -1, &mark, DT_CENTER | DT_BOTTOM | DT_SINGLELINE | DT_NOPREFIX);
		if (old_font) SelectObject(dc, old_font);
		if (mark_font) DeleteObject(mark_font);

		// hint line right below it
		HFONT tip_font = CreateFontW(-16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
			DEFAULT_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
			DEFAULT_PITCH, L"Microsoft YaHei UI");
		old_font = tip_font ? SelectObject(dc, tip_font) : nullptr;
		SetTextColor(dc, tip_color);
		RECT tip = rc;
		tip.top = (rc.top + rc.bottom) / 2 + 50;
		DrawTextW(dc, L"\u6b63\u5728\u542f\u52a8\u2026", -1, &tip,
			DT_CENTER | DT_TOP | DT_SINGLELINE | DT_NOPREFIX);
		if (old_font) SelectObject(dc, old_font);
		if (tip_font) DeleteObject(tip_font);
	}
	EndPaint(hwnd, &ps);
}

// force the native title bar into dark mode (DWMWA_USE_IMMERSIVE_DARK_MODE)
static void apply_dark_title_bar(HWND hwnd) {
	BOOL use_dark = TRUE;
	DwmSetWindowAttribute(hwnd, 20, &use_dark, sizeof(use_dark));
}

static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
	switch (msg) {
		case WM_PAINT:
			paint_placeholder(hwnd);
			return 0;
		case WM_TIMER:
			if (wp == kRevealTimerId) reveal_webview();
			return 0;
		case WM_SIZE:
			if (wp != SIZE_MINIMIZED) update_webview_bounds(hwnd);
			return 0;
		case WM_GETMINMAXINFO: {
			auto* mmi = reinterpret_cast<MINMAXINFO*>(lp);
			mmi->ptMinTrackSize = {640, 480};
			return 0;
		}
		case WM_NCACTIVATE: {
			LRESULT result = DefWindowProcW(hwnd, msg, wp, lp);
			apply_dark_title_bar(hwnd);  // title-bar repaint can drop the attribute
			return result;
		}
		case WM_CLOSE:
			DestroyWindow(hwnd);
			return 0;
		case WM_DESTROY:
			PostQuitMessage(0);
			return 0;
		default:
			return DefWindowProcW(hwnd, msg, wp, lp);
	}
}

// ── webview2 bootstrap (completion handlers with static lifetime) ──────────

static constexpr IID kIID_ControllerHandler = {
	0x6c4819f3, 0xc9b7, 0x4260, {0x81, 0x27, 0xc9, 0xf5, 0xbd, 0xe7, 0xf6, 0x8c}};
static constexpr IID kIID_EnvironmentHandler = {
	0x4e8a3389, 0xc9d8, 0x4bd2, {0xb6, 0xb5, 0x12, 0x4f, 0xee, 0x6c, 0xc1, 0x4d}};

// Chromium switches that remove work which is irrelevant to showing the UI but
// is paid on every launch — most of it is network activity (component/extension
// updates, sync, domain reliability) that a desktop shell never needs and that
// makes the first frame wait on a request timeout when the machine is offline.
static constexpr wchar_t kBrowserArgs[] =
	L"--no-first-run "
	L"--no-default-browser-check "
	L"--disable-background-networking "
	L"--disable-component-update "
	L"--disable-domain-reliability "
	L"--disable-sync "
	L"--disable-features=CalculateNativeWinOcclusion,Translate,OptimizationHints,"
	L"OptimizationGuideModelDownloading,MediaRouter,AutofillServerCommunication";

class NavigationCompletedHandler final : public ICoreWebView2NavigationCompletedEventHandler {
public:
	HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
		if (!ppv) return E_POINTER;
		if (riid == IID_IUnknown || riid == IID_ICoreWebView2NavigationCompletedEventHandler) {
			*ppv = static_cast<ICoreWebView2NavigationCompletedEventHandler*>(this);
			AddRef();
			return S_OK;
		}
		*ppv = nullptr;
		return E_NOINTERFACE;
	}
	ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&ref_); }
	ULONG STDMETHODCALLTYPE Release() override {
		ULONG c = InterlockedDecrement(&ref_);
		if (c == 0) delete this;
		return c;
	}
	HRESULT STDMETHODCALLTYPE Invoke(
		ICoreWebView2*, ICoreWebView2NavigationCompletedEventArgs*) override {
		// first frame is ready (or the navigation failed): show the webview
		reveal_webview();
		return S_OK;
	}

private:
	LONG ref_ = 1;
};

class ControllerCompletedHandler final
	: public ICoreWebView2CreateCoreWebView2ControllerCompletedHandler {
public:
	HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
		if (!ppv) return E_POINTER;
		if (riid == IID_IUnknown || riid == kIID_ControllerHandler) {
			*ppv = static_cast<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler*>(this);
			AddRef();
			return S_OK;
		}
		*ppv = nullptr;
		return E_NOINTERFACE;
	}
	ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&ref_); }
	ULONG STDMETHODCALLTYPE Release() override {
		ULONG c = InterlockedDecrement(&ref_);
		if (c == 0) delete this;
		return c;
	}
	HRESULT STDMETHODCALLTYPE Invoke(HRESULT result, ICoreWebView2Controller* controller) override {
		if (FAILED(result) || !controller) {
			fprintf(stderr, "[shell] controller creation failed hr=0x%08lx\n", (unsigned long)result);
			MessageBoxW(nullptr, L"WebView2 控制器创建失败", L"Phi", MB_ICONERROR | MB_OK);
			PostQuitMessage(1);
			return result;
		}
		g_controller = controller;
		g_controller->AddRef();
		stage("controller ready");

		// the webview paints white until its first frame lands — on a dark shell
		// that is a full-window flash, so set the default background to match
		ICoreWebView2Controller2* controller2 = nullptr;
		if (SUCCEEDED(controller->QueryInterface(IID_ICoreWebView2Controller2,
				reinterpret_cast<void**>(&controller2))) &&
			controller2) {
			COREWEBVIEW2_COLOR bg{};
			bg.A = 255;
			bg.R = (BYTE)g_theme.r;
			bg.G = (BYTE)g_theme.g;
			bg.B = (BYTE)g_theme.b;
			controller2->put_DefaultBackgroundColor(bg);
			controller2->Release();
		}

		// keep the splash on screen until the first frame: the webview stays
		// hidden until NavigationCompleted fires (or the fallback timer expires)
		controller->put_IsVisible(FALSE);

		if (SUCCEEDED(controller->get_CoreWebView2(&g_webview)) && g_webview) {
			ICoreWebView2Settings* settings = nullptr;
			if (SUCCEEDED(g_webview->get_Settings(&settings)) && settings) {
				settings->put_AreDefaultContextMenusEnabled(FALSE);
				settings->Release();
			}
			auto* nav_handler = new NavigationCompletedHandler();
			EventRegistrationToken token{};
			g_webview->add_NavigationCompleted(nav_handler, &token);
			nav_handler->Release();

			wchar_t url[96];
			const char* perf = std::getenv("PHI_PERF");
			if (perf && perf[0] == '1') {
				swprintf(url, 96, L"http://127.0.0.1:%d/?perf=1", g_port);
			} else {
				swprintf(url, 96, L"http://127.0.0.1:%d/", g_port);
			}
			g_webview->Navigate(url);
		} else {
			// nothing to wait for — show the window content as-is
			reveal_webview();
		}

		// size the webview to the current window rect
		HWND hwnd = FindWindowW(kWndClassW, nullptr);
		if (hwnd) update_webview_bounds(hwnd);
		fprintf(stderr, "[shell] webview ready\n");
		return S_OK;
	}

private:
	LONG ref_ = 1;
};

class EnvironmentCompletedHandler final
	: public ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler {
public:
	HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
		if (!ppv) return E_POINTER;
		if (riid == IID_IUnknown || riid == kIID_EnvironmentHandler) {
			*ppv = static_cast<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler*>(this);
			AddRef();
			return S_OK;
		}
		*ppv = nullptr;
		return E_NOINTERFACE;
	}
	ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&ref_); }
	ULONG STDMETHODCALLTYPE Release() override {
		ULONG c = InterlockedDecrement(&ref_);
		if (c == 0) delete this;
		return c;
	}
	HRESULT STDMETHODCALLTYPE Invoke(HRESULT result, ICoreWebView2Environment* environment) override {
		if (FAILED(result) || !environment) {
			fprintf(stderr, "[shell] environment creation failed hr=0x%08lx\n", (unsigned long)result);
			MessageBoxW(nullptr, L"WebView2 环境创建失败（请确认已安装 WebView2 运行时）", L"Phi",
				MB_ICONERROR | MB_OK);
			PostQuitMessage(1);
			return result;
		}
		stage("environment ready");
		HWND hwnd = FindWindowW(kWndClassW, nullptr);
		auto* handler = new ControllerCompletedHandler();
		HRESULT hr = environment->CreateCoreWebView2Controller(hwnd, handler);
		if (FAILED(hr)) {
			fprintf(stderr, "[shell] CreateCoreWebView2Controller failed hr=0x%08lx\n", (unsigned long)hr);
			MessageBoxW(nullptr, L"WebView2 控制器创建失败", L"Phi", MB_ICONERROR | MB_OK);
			PostQuitMessage(1);
		}
		handler->Release();
		return S_OK;
	}

private:
	LONG ref_ = 1;
};

// ── app_main ───────────────────────────────────────────────────────────────

int app_main(const std::string& base_dir) {
	g_start_ms = trace::now_ms();
	g_hinst = GetModuleHandleW(nullptr);

	// resolve the persisted theme first: the splash, the window background and the
	// webview canvas all have to be the right colour on the very first frame
	g_theme = resolve_ui_canvas_theme(base_dir);

	// 1. window first: it is on screen (in the user's theme, with the Φ splash)
	//    within a few ms, so everything below overlaps with "the user already sees
	//    the app"
	WNDCLASSEXW wx{};
	wx.cbSize = sizeof(wx);
	wx.lpfnWndProc = wnd_proc;
	wx.hInstance = g_hinst;
	wx.lpszClassName = kWndClassW;
	wx.hIcon = LoadIconW(g_hinst, MAKEINTRESOURCEW(1));
	wx.hIconSm = LoadIconW(g_hinst, MAKEINTRESOURCEW(1));
	g_brush = CreateSolidBrush(RGB(g_theme.r, g_theme.g, g_theme.b));  // theme bg, avoids a flash
	wx.hbrBackground = g_brush;
	if (!RegisterClassExW(&wx)) {
		fprintf(stderr, "[shell] RegisterClassExW failed err=%lu\n", GetLastError());
		return 1;
	}

	int sw = GetSystemMetrics(SM_CXSCREEN);
	int sh = GetSystemMetrics(SM_CYSCREEN);
	int ww = 1280, wh = 860;
	HWND hwnd = CreateWindowExW(0, kWndClassW, L"Φ · Phi",
		WS_OVERLAPPEDWINDOW, (sw - ww) / 2, (sh - wh) / 2, ww, wh,
		nullptr, nullptr, g_hinst, nullptr);
	if (!hwnd) {
		fprintf(stderr, "[shell] CreateWindowExW failed err=%lu\n", GetLastError());
		return 1;
	}
	apply_dark_title_bar(hwnd);
	ShowWindow(hwnd, SW_SHOW);
	UpdateWindow(hwnd);
	stage("window shown");

	// 2. COM + WebView2 runtime: starting the browser process is the single most
	//    expensive step, so kick it off as early as possible and let it run in the
	//    background while the protocol server below comes up
	HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
	if (hr != S_OK && hr != RPC_E_CHANGED_MODE) {
		MessageBoxW(nullptr, L"CoInitializeEx 失败", L"Phi", MB_ICONERROR | MB_OK);
		return 1;
	}

	std::string data_path = path_join(base_dir, "data/webview");
	mkdirs(data_path);

	std::wstring loader_path = utf8_to_wide(path_join(exe_dir_path(), "webview2loader.dll"));
	HMODULE loader = LoadLibraryW(loader_path.c_str());
	if (!loader) {
		DWORD err = GetLastError();
		fprintf(stderr, "[shell] LoadLibrary(webview2loader.dll) failed err=%lu\n", err);
		MessageBoxW(nullptr, L"缺少 webview2loader.dll（bin 目录下）", L"Phi", MB_ICONERROR | MB_OK);
		return 1;
	}

	using CreateEnvFn = HRESULT(STDAPICALLTYPE*)(
		PCWSTR browser_folder, PCWSTR user_data_folder, ICoreWebView2EnvironmentOptions* options,
		ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler* handler);
	// GetProcAddress returns FARPROC, whose type is not compatible with
	// CreateEnvFn; the cast goes through void* — the same idiom cuda_api.cpp and
	// the Media Foundation loaders use — so -Wcast-function-type stays quiet.
	auto create_env = reinterpret_cast<CreateEnvFn>(
		(void*)GetProcAddress(loader, "CreateCoreWebView2EnvironmentWithOptions"));
	if (!create_env) {
		MessageBoxW(nullptr, L"WebView2Loader.dll 缺少入口", L"Phi", MB_ICONERROR | MB_OK);
		return 1;
	}

	// keep the handler alive until Invoke fires (static storage is fine:
	// the app is single-window, single-lifetime)
	static EnvironmentCompletedHandler env_handler;
	// the vendored WebView2Loader.dll is 1.0.4191.47 (2020) and answers E_INVALIDARG
	// when handed a custom ICoreWebView2EnvironmentOptions implementation, so the
	// switches ride on the documented environment variable instead: the runtime
	// reads it (and it is ignored whenever options carry their own arguments)
	SetEnvironmentVariableW(L"WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", kBrowserArgs);
	std::wstring user_data = utf8_to_wide(data_path);
	HRESULT hr2 = create_env(nullptr, user_data.c_str(), nullptr, &env_handler);
	fprintf(stderr, "[shell] CreateEnvironment hr=0x%08lx\n", (unsigned long)hr2);
	if (FAILED(hr2)) {
		MessageBoxW(nullptr, L"WebView2 环境创建失败（请确认已安装 WebView2 运行时）", L"Phi",
			MB_ICONERROR | MB_OK);
		return 1;
	}
	stage("env requested");

	// 3. protocol server (binds the loopback UI bridge): runs concurrently with
	//    the browser process start above. Its navigation callback cannot fire
	//    before the message loop below, so g_port is always set in time.
	ProtocolServer server;
	server.set_folder_picker(&pick_folder_dialog);
	if (!server.start(base_dir)) {
		MessageBoxW(nullptr, L"协议服务启动失败", L"Phi", MB_ICONERROR | MB_OK);
		return 1;
	}
	g_port = server.port();
	fprintf(stderr, "[shell] listening on http://127.0.0.1:%d/\n", g_port);
	stage("protocol server up");

	// fallback: never leave the splash up if the navigation event goes missing
	SetTimer(hwnd, kRevealTimerId, kRevealTimeoutMs, nullptr);

	// 4. message loop
	MSG msg;
	while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
		TranslateMessage(&msg);
		DispatchMessageW(&msg);
	}

	// 5. cleanup
	server.shutdown();
	if (g_webview) {
		g_webview->Release();
		g_webview = nullptr;
	}
	if (g_controller) {
		g_controller->Close();
		g_controller->Release();
		g_controller = nullptr;
	}
	if (g_brush) DeleteObject(g_brush);
	CoUninitialize();
	return static_cast<int>(msg.wParam);
}

// ── native folder picker ───────────────────────────────────────────────────

std::optional<std::string> pick_folder_dialog(const std::string& initial_dir) {
	HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
	if (hr != S_OK && hr != RPC_E_CHANGED_MODE) return std::nullopt;
	bool need_uninit = (hr == S_OK);

	std::optional<std::string> result;

	IFileDialog* dialog = nullptr;
	hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
		IID_IFileDialog, reinterpret_cast<void**>(&dialog));
	if (SUCCEEDED(hr) && dialog) {
		dialog->SetOptions(FOS_PICKFOLDERS);
		if (!initial_dir.empty()) {
			std::wstring init_w = utf8_to_wide(initial_dir);
			IShellItem* item = nullptr;
			if (SUCCEEDED(SHCreateItemFromParsingName(init_w.c_str(), nullptr, IID_IShellItem,
				reinterpret_cast<void**>(&item)))) {
				dialog->SetFolder(item);
				item->Release();
			}
		}
		if (SUCCEEDED(dialog->Show(nullptr))) {
			IShellItem* item = nullptr;
			if (SUCCEEDED(dialog->GetResult(&item)) && item) {
				PWSTR wide_path = nullptr;
				if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &wide_path)) && wide_path) {
					result = wide_to_utf8(wide_path);
					CoTaskMemFree(wide_path);
				}
				item->Release();
			}
		}
		dialog->Release();
	}

	if (need_uninit) CoUninitialize();
	return result;
}

}  // namespace phi
