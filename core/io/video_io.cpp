// video_io — video decode through the Media Foundation SourceReader.
//
// The file is opened with MFCreateSourceReaderFromURL and the stream is asked
// for the decoder's native output (NV12), which every H.264/HEVC/VP9 decoder in
// the box can produce.  The chroma is then converted to tightly packed RGB on
// the host with fixed-point studio-swing maths, so the same file always yields
// the same bytes on every machine.
//
// Asking the SourceReader for RGB32 instead would hand the conversion to the
// Video Processor MFT, which is a D3D11 component: in a headless session it
// cannot be instantiated and SetCurrentMediaType fails with MF_E_INVALIDMEDIATYPE
// (verified on the build machine).  RGB32 is still tried as a second choice, for
// containers whose decoder only offers packed RGB.
//
// Linkage note: this TU is part of core*.o and bin/phi.exe is linked against a
// fixed library list that predates this layer (no mfplat/mfreadwrite).  Every
// Media Foundation entry point is therefore resolved with LoadLibrary +
// GetProcAddress — the same treatment the CUDA driver API and NVRTC get in this
// engine — and
// the attribute/format GUIDs are spelled out locally instead of being imported
// from mfuuid.lib (values verified against the SDK).
#include "io/video_io.hpp"

#include <windows.h>

#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfreadwrite.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace phi::media {
namespace {

constexpr i64 kTicksPerSecond = 10000000;  // 100 ns units

// ── Media Foundation entry points, resolved at load time ───────────────────

void* mf_symbol(const char* name) {
	static HMODULE mods[4] = {nullptr, nullptr, nullptr, nullptr};
	static const char* dlls[4] = {"mfplat.dll", "mfreadwrite.dll", "mf.dll", "mfcore.dll"};
	for (int i = 0; i < 4; i++) {
		if (!mods[i]) mods[i] = LoadLibraryA(dlls[i]);
		if (mods[i]) {
			void* p = (void*)GetProcAddress(mods[i], name);
			if (p) return p;
		}
	}
	return nullptr;
}

template <class Fn>
Fn mf_import(const char* name) {
	void* p = mf_symbol(name);
	if (!p) throw MediaError(std::string("Media Foundation is missing entry point ") + name);
	return reinterpret_cast<Fn>(p);
}

// Thin wrappers so that not a single MF API is called through an import library
// (there is none on the link line).
HRESULT mf_create_media_type(IMFMediaType** out) {
	using Fn = HRESULT(WINAPI*)(IMFMediaType**);
	static Fn fn = nullptr;
	if (!fn) fn = mf_import<Fn>("MFCreateMediaType");
	return fn(out);
}

// Idempotent MFStartup; Media Foundation is refcounted so this is safe even if
// the host process already brought it up.
void mf_startup() {
	static bool done = false;
	if (done) return;
	using StartupFn = HRESULT(WINAPI*)(ULONG, DWORD);
	StartupFn startup = mf_import<StartupFn>("MFStartup");
	HRESULT hr = startup(MF_VERSION, 0 /* MFSTARTUP_FULL */);
	if (FAILED(hr)) {
		char buf[64];
		snprintf(buf, sizeof buf, "0x%08lX", (unsigned long)hr);
		throw MediaError(std::string("MFStartup failed (") + buf +
		                 ") — Media Foundation needs CoInitializeEx on this thread");
	}
	done = true;
}

std::wstring utf8_to_wide(const std::string& s) {
	int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
	std::wstring w(n > 0 ? (size_t)n : 0, L'\0');
	if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
	return w;
}

void mf_check(HRESULT hr, const char* what) {
	if (SUCCEEDED(hr)) return;
	char buf[64];
	snprintf(buf, sizeof buf, " (0x%08lX)", (unsigned long)hr);
	throw MediaError(std::string(what) + buf);
}

template <class T>
struct ComPtr {
	T* p = nullptr;
	ComPtr() = default;
	~ComPtr() { reset(); }
	ComPtr(const ComPtr&) = delete;
	ComPtr& operator=(const ComPtr&) = delete;
	ComPtr(ComPtr&& o) noexcept : p(o.p) { o.p = nullptr; }
	ComPtr& operator=(ComPtr&& o) noexcept {
		if (this != &o) {
			reset();
			p = o.p;
			o.p = nullptr;
		}
		return *this;
	}
	T* operator->() const { return p; }
	T* get() const { return p; }
	explicit operator bool() const { return p != nullptr; }
	void reset() {
		if (p) {
			p->Release();
			p = nullptr;
		}
	}
	// Out-parameter slot; the holder must be empty (open() closes first).
	T** put() {
		reset();
		return &p;
	}
};

// ── attribute / format GUIDs (identical to the MF SDK's mfuuid values) ─────

const GUID kMtMajorType = {0x48EBA18E, 0xF8C9, 0x4687, {0xBF, 0x11, 0x0A, 0x74, 0xC9, 0xF9, 0x6A, 0x8F}};
const GUID kMtSubtype = {0xF7E34C9A, 0x42E8, 0x4714, {0xB7, 0x4B, 0xCB, 0x29, 0xD7, 0x2C, 0x35, 0xE5}};
const GUID kMtFrameSize = {0x1652C33D, 0xD6B2, 0x4012, {0xB8, 0x34, 0x72, 0x03, 0x08, 0x49, 0xA3, 0x7D}};
const GUID kMtFrameRate = {0xC459A2E8, 0x3D2C, 0x4E44, {0xB1, 0x32, 0xFE, 0xE5, 0x15, 0x6C, 0x7B, 0xB0}};
const GUID kMtPixelAspectRatio = {0xC6376A1E, 0x8D0A, 0x4027, {0xBE, 0x45, 0x6D, 0x9A, 0x0A, 0xD3, 0x9B, 0xB6}};
const GUID kMtInterlaceMode = {0xE2724BB8, 0xE676, 0x4806, {0xB4, 0xB2, 0xA8, 0xD6, 0xEF, 0xB4, 0x4C, 0xCD}};
const GUID kMtAllSamplesIndependent = {0xC9173739, 0x5E56, 0x461C, {0xB7, 0x13, 0x46, 0xFB, 0x99, 0x5C, 0xB9, 0x5F}};
const GUID kMtDefaultStride = {0x644B4E48, 0x1E02, 0x4516, {0xB0, 0xEB, 0xC0, 0x1C, 0xA9, 0xD4, 0x9A, 0xC6}};
const GUID kPdDuration = {0x6C990D33, 0xBB8E, 0x477A, {0x85, 0x98, 0x0D, 0x5D, 0x96, 0xFC, 0xD8, 0x8A}};
const GUID kMediaTypeVideo = {0x73646976, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kVideoFormatRgb32 = {0x00000016, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kVideoFormatNv12 = {0x3231564E, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kNullGuid = {0x00000000, 0x0000, 0x0000, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}};

constexpr UINT32 kProgressive = 2;  // MFVideoInterlace_Progressive
constexpr DWORD kFirstVideoStream = (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM;
constexpr DWORD kMediaSource = (DWORD)MF_SOURCE_READER_MEDIASOURCE;

// ── pixel plumbing ─────────────────────────────────────────────────────────

enum class Layout {
	None,
	Nv12,     // 4:2:0, Y plane then interleaved U/V, converted on the host
	Packed,   // 32- or 24-bit B,G,R(,X) rows straight from MF
};

inline unsigned char clamp8(int v) { return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

// NV12 -> tightly packed top-down RGB.  Fixed point, studio swing (16..235 /
// 16..240); BT.709 for tall frames and BT.601 otherwise, which is the same
// default MPEG uses when a bitstream carries no matrix hint.
void nv12_to_rgb(const unsigned char* y_plane, const unsigned char* uv_plane, i64 stride, i64 w,
                 i64 h, bool bt709, std::vector<unsigned char>& rgb) {
	const int c_y = 1192;  // 1.164 * 1024
	const int c_rv = bt709 ? 1836 : 1634;
	const int c_gu = bt709 ? -218 : -401;
	const int c_gv = bt709 ? -546 : -833;
	const int c_bu = bt709 ? 2163 : 2065;

	rgb.assign((size_t)(w * h * 3), 0);
	for (i64 j = 0; j < h; j++) {
		const unsigned char* yrow = y_plane + (size_t)(j * stride);
		const unsigned char* uvrow = uv_plane + (size_t)((j / 2) * stride);
		unsigned char* dst = &rgb[(size_t)(j * w * 3)];
		for (i64 i = 0; i < w; i++) {
			const int yy = (int)yrow[i] - 16;
			const int uu = (int)uvrow[(i & ~(i64)1)] - 128;  // U is the even byte
			const int vv = (int)uvrow[(i & ~(i64)1) + 1] - 128;
			const int base = yy * c_y;
			dst[i * 3 + 0] = clamp8((base + c_rv * vv) >> 10);
			dst[i * 3 + 1] = clamp8((base + c_gu * uu + c_gv * vv) >> 10);
			dst[i * 3 + 2] = clamp8((base + c_bu * uu) >> 10);
		}
	}
}

}  // namespace

// ── VideoReader ────────────────────────────────────────────────────────────

struct VideoReader::Impl {
	ComPtr<IMFSourceReader> reader;
	DWORD stream = kFirstVideoStream;
	Layout layout = Layout::None;
	i64 index = 0;   // frame counter of the current read sequence
	i64 stride = 0;  // bytes per row (luma) from MF_MT_DEFAULT_STRIDE
	bool bt709 = false;
	bool eof = false;
	// seek() only positions the reader and records the target time; read()
	// drops every frame that still precedes it.  That is what makes seeking
	// correct on MF builds whose SetCurrentPosition is a no-op, at the price of
	// decoding forward from the head.
	double target_seconds = 0.0;
	bool has_target = false;
};

VideoReader::~VideoReader() { close(); }

void VideoReader::close() {
	if (impl_) {
		impl_->reader.reset();
		delete impl_;
		impl_ = nullptr;
	}
	w_ = h_ = 0;
	fps_ = 0.0;
	duration_ = 0.0;
}

void VideoReader::open(const std::string& path) {
	close();
	mf_startup();

	using CreateReaderFn = HRESULT(WINAPI*)(LPCWSTR, IMFAttributes*, IMFSourceReader**);
	CreateReaderFn create_reader = mf_import<CreateReaderFn>("MFCreateSourceReaderFromURL");

	const std::wstring wpath = utf8_to_wide(path);
	if (wpath.empty()) throw MediaError("VideoReader::open: empty path");

	ComPtr<IMFSourceReader> reader;
	mf_check(create_reader(wpath.c_str(), nullptr, reader.put()),
	         ("VideoReader::open: cannot open " + path).c_str());

	// A stream must exist and must be video before anything else is configured.
	ComPtr<IMFMediaType> native;
	HRESULT hr = reader->GetNativeMediaType(kFirstVideoStream, 0, native.put());
	if (FAILED(hr) || !native) throw MediaError("VideoReader::open: no video stream in " + path);

	// Native NV12 first (works with every decoder, converted here); RGB32 only
	// as a fallback, because it needs the D3D11 Video Processor MFT.
	const GUID* wanted[2] = {&kVideoFormatNv12, &kVideoFormatRgb32};
	Layout chosen = Layout::None;
	HRESULT last_error = E_FAIL;
	for (int attempt = 0; attempt < 2 && chosen == Layout::None; attempt++) {
		ComPtr<IMFMediaType> out;
		mf_check(mf_create_media_type(out.put()), "MFCreateMediaType");
		mf_check(out->SetGUID(kMtMajorType, kMediaTypeVideo), "SetGUID(major)");
		mf_check(out->SetGUID(kMtSubtype, *wanted[attempt]), "SetGUID(subtype)");
		mf_check(out->SetUINT32(kMtInterlaceMode, kProgressive), "SetUINT32(interlace)");
		mf_check(out->SetUINT32(kMtAllSamplesIndependent, TRUE), "SetUINT32(all samples)");
		// Carrying the native geometry over keeps picky decoders happy.
		UINT32 nw = 0, nh = 0;
		if (SUCCEEDED(MFGetAttributeSize(native.get(), kMtFrameSize, &nw, &nh)) && nw && nh)
			mf_check(MFSetAttributeSize(out.get(), kMtFrameSize, nw, nh), "SetAttributeSize");
		UINT32 num = 0, den = 0;
		if (SUCCEEDED(MFGetAttributeRatio(native.get(), kMtFrameRate, &num, &den)) && num && den)
			mf_check(MFSetAttributeRatio(out.get(), kMtFrameRate, num, den), "SetAttributeRatio");
		mf_check(MFSetAttributeRatio(out.get(), kMtPixelAspectRatio, 1, 1), "SetAttributeRatio(par)");

		hr = reader->SetCurrentMediaType(kFirstVideoStream, nullptr, out.get());
		if (SUCCEEDED(hr)) {
			chosen = attempt == 0 ? Layout::Nv12 : Layout::Packed;
		} else {
			last_error = hr;
		}
	}
	if (chosen == Layout::None) {
		char buf[128];
		snprintf(buf, sizeof buf, " (0x%08lX)", (unsigned long)last_error);
		throw MediaError(std::string("VideoReader::open: no supported decoder output for ") + path +
		                 buf);
	}

	ComPtr<IMFMediaType> current;
	mf_check(reader->GetCurrentMediaType(kFirstVideoStream, current.put()),
	         "VideoReader::open: GetCurrentMediaType");

	UINT32 w = 0, h = 0;
	if (FAILED(MFGetAttributeSize(current.get(), kMtFrameSize, &w, &h)) || w == 0 || h == 0)
		throw MediaError("VideoReader::open: the decoder reported no frame size");
	w_ = (i64)w;
	h_ = (i64)h;

	UINT32 num = 0, den = 0;
	if (SUCCEEDED(MFGetAttributeRatio(current.get(), kMtFrameRate, &num, &den)) && num && den)
		fps_ = (double)num / (double)den;
	else
		fps_ = 30.0;  // a container without a rate attribute; only used for sampling

	impl_ = new Impl();
	impl_->reader = std::move(reader);
	impl_->stream = kFirstVideoStream;
	impl_->layout = chosen;
	// MF_MT_DEFAULT_STRIDE is stored as UINT32 but means a signed row pitch; a
	// negative value would be a bottom-up surface (never seen from a decoder).
	UINT32 stride_attr = 0;
	impl_->stride = SUCCEEDED(current->GetUINT32(kMtDefaultStride, &stride_attr))
	                    ? (i64)(INT32)stride_attr
	                    : 0;
	// SD-ish content in this engine is 601; tall frames are 709.
	impl_->bt709 = h_ >= 720;

	// Duration is a presentation attribute of the whole file.
	PROPVARIANT var;
	PropVariantInit(&var);
	if (SUCCEEDED(impl_->reader->GetPresentationAttribute(kMediaSource, kPdDuration, &var))) {
		if (var.vt == VT_UI8 || var.vt == VT_I8)
			duration_ = (double)var.uhVal.QuadPart / (double)kTicksPerSecond;
		PropVariantClear(&var);
	}
}

bool VideoReader::read(VideoFrame& out) {
	if (!impl_ || !impl_->reader) throw MediaError("VideoReader::read: the reader is not open");

	// Frames that still precede a pending seek target are dropped, not returned
	// (see seek()).  The tolerance is half a frame, so a seek lands on the frame
	// at the requested time or the first one after it.
	const double tolerance = std::max(0.5 / (fps_ > 0.0 ? fps_ : 30.0), 0.002);

	for (;;) {
		if (impl_->eof) return false;

		DWORD actual_stream = 0, flags = 0;
		LONGLONG timestamp = 0;
		ComPtr<IMFSample> sample;
		HRESULT hr = impl_->reader->ReadSample(impl_->stream, 0, &actual_stream, &flags, &timestamp,
		                                       sample.put());
		if (FAILED(hr)) {
			char buf[96];
			snprintf(buf, sizeof buf, " (0x%08lX)", (unsigned long)hr);
			throw MediaError(std::string("VideoReader::read: ReadSample failed") + buf);
		}
		if (flags & MF_SOURCE_READERF_ENDOFSTREAM) impl_->eof = true;
		if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) {
			// The decoder renegotiated (resolution/stride); pick the new geometry up.
			ComPtr<IMFMediaType> current;
			if (SUCCEEDED(impl_->reader->GetCurrentMediaType(kFirstVideoStream, current.put())) &&
			    current) {
				UINT32 w = 0, h = 0;
				if (SUCCEEDED(MFGetAttributeSize(current.get(), kMtFrameSize, &w, &h)) && w && h) {
					w_ = (i64)w;
					h_ = (i64)h;
					impl_->bt709 = h_ >= 720;
				}
				UINT32 stride_attr = 0;
				if (SUCCEEDED(current->GetUINT32(kMtDefaultStride, &stride_attr)))
					impl_->stride = (i64)(INT32)stride_attr;
			}
		}
		if (!sample) {
			if (impl_->eof) return false;
			continue;  // a gap or a stream tick
		}

		ComPtr<IMFMediaBuffer> buffer;
		mf_check(sample->ConvertToContiguousBuffer(buffer.put()),
		         "VideoReader::read: ConvertToContiguousBuffer");

		BYTE* data = nullptr;
		DWORD max_len = 0, cur_len = 0;
		mf_check(buffer->Lock(&data, &max_len, &cur_len), "VideoReader::read: Lock");

		VideoFrame frame;
		frame.w = w_;
		frame.h = h_;
		frame.index = impl_->index++;
		frame.seconds = (double)timestamp / (double)kTicksPerSecond;
		frame.rgb.assign((size_t)(w_ * h_ * 3), 0);

		if (impl_->layout == Layout::Nv12) {
			// A decoder may pad the luma pitch; when the attribute is missing,
			// the buffer length gives it away (NV12 is stride*h*3/2 bytes).
			i64 stride = impl_->stride;
			if (stride < w_) {
				const i64 derived = h_ > 0 ? (i64)cur_len * 2 / (3 * h_) : w_;
				stride = derived >= w_ ? derived : w_;
			}
			const i64 need = stride * h_ + stride * ((h_ + 1) / 2);
			if (stride > 0 && need <= (i64)cur_len)
				nv12_to_rgb(data, data + stride * h_, stride, w_, h_, impl_->bt709, frame.rgb);
		} else if (impl_->layout == Layout::Packed) {
			// MF's RGB32/RGB24 surfaces are B,G,R[,X] in memory.
			const i64 bpp = ((i64)cur_len >= w_ * h_ * 4) ? 4 : 3;
			i64 stride = impl_->stride != 0 ? impl_->stride : w_ * bpp;
			const i64 abs_stride = stride < 0 ? -stride : stride;
			if ((h_ - 1) * abs_stride + w_ * bpp <= (i64)cur_len) {
				for (i64 y = 0; y < h_; y++) {
					const i64 src_y = stride < 0 ? (h_ - 1 - y) : y;  // bottom-up surface
					const unsigned char* row = data + (size_t)(src_y * abs_stride);
					unsigned char* dst = &frame.rgb[(size_t)(y * w_ * 3)];
					for (i64 x = 0; x < w_; x++) {
						dst[x * 3 + 0] = row[x * bpp + 2];
						dst[x * 3 + 1] = row[x * bpp + 1];
						dst[x * 3 + 2] = row[x * bpp + 0];
					}
				}
			}
		}

		buffer->Unlock();

		if (impl_->has_target) {
			if (frame.seconds + tolerance < impl_->target_seconds) continue;  // before the seek target
			impl_->has_target = false;
		}
		out = std::move(frame);
		return true;
	}
}

void VideoReader::seek(double seconds) {
	if (!impl_ || !impl_->reader) throw MediaError("VideoReader::seek: the reader is not open");
	if (seconds < 0.0) seconds = 0.0;
	if (duration_ > 0.0 && seconds > duration_) seconds = duration_;

	PROPVARIANT var;
	PropVariantInit(&var);
	var.vt = VT_I8;
	var.hVal.QuadPart = (LONGLONG)llround(seconds * (double)kTicksPerSecond);
	HRESULT hr = impl_->reader->SetCurrentPosition(kNullGuid, var);
	PropVariantClear(&var);
	mf_check(hr, "VideoReader::seek: SetCurrentPosition");

	impl_->eof = false;
	impl_->index = fps_ > 0.0 ? (i64)llround(seconds * fps_) : 0;
	impl_->target_seconds = seconds;
	impl_->has_target = true;  // read() skips whatever still precedes the target
}

std::vector<VideoFrame> VideoReader::sample_frames(i64 max_frames) {
	std::vector<VideoFrame> out;
	if (!impl_ || !impl_->reader || max_frames <= 0) return out;
	out.reserve((size_t)max_frames);

	if (max_frames == 1 || duration_ <= 0.0) {
		// Either a single sample (the head of the file) or a container with no
		// duration to sample over: decode forward.
		seek(0.0);
		VideoFrame f;
		if (read(f)) out.push_back(std::move(f));
		return out;
	}

	for (i64 i = 0; i < max_frames; i++) {
		// Uniform grid over the whole file, first and last sample included.
		double t = duration_ * (double)i / (double)(max_frames - 1);
		if (t > duration_ - 0.05) t = std::max(0.0, duration_ - 0.05);  // skip the very end
		seek(t);
		VideoFrame f;
		if (!read(f)) continue;
		f.index = i;
		out.push_back(std::move(f));
	}
	return out;
}

}  // namespace phi::media
