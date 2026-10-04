// 8-bit PNG/JPEG I/O (WIC) + the canvas/crop/resize primitives of the image path.
//
// The reference implementation (ComfyUI on PIL/numpy) resizes with
// common_upscale(..., "lanczos", "center"), i.e. "scale preserving aspect, then
// centre-crop"; here the two halves are exposed separately (resize_lanczos3,
// crop_centre) so the caller can compose exactly that and, when it needs to, only
// one of them.
#include "io/image_io.hpp"

#include <windows.h>
#include <objbase.h>
#include <oleauto.h>
#include <wincodec.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace phi::media {
namespace {

// ── COM ────────────────────────────────────────────────────────────────────
//
// COM is brought up lazily, on the first call that actually touches WIC: the
// module is used both from the app (whose threads may already have picked an
// apartment) and from host tests that never initialise COM at all.
//
// RPC_E_CHANGED_MODE is *not* an error here. It means the calling thread is
// already in another apartment (in practice, the MTA), and WIC works fine from
// an MTA — failing on it would break every image call made from a worker thread.
// In that case we also own nothing, so CoUninitialize must not be called later.
HRESULT com_ready() {
	static std::once_flag once;
	static HRESULT hr = S_OK;
	std::call_once(once, [] {
		HRESULT r = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
		if (r == RPC_E_CHANGED_MODE || r == S_FALSE) {
			hr = S_OK;  // already in an apartment we do not own
			return;
		}
		hr = r;
		// Deliberately no CoUninitialize: the apartment lives as long as the
		// process, so a WIC object handed to another thread can never be torn out
		// from under its owner at thread exit.
	});
	return hr;
}

void check_hr(HRESULT hr, const char* what) {
	if (FAILED(hr)) {
		char buf[160];
		snprintf(buf, sizeof buf, "image_io: %s failed (hr=0x%08lX)", what, (unsigned long)hr);
		throw MediaError(buf);
	}
}

void ensure_com(const char* what) {
	check_hr(com_ready(), what);
}

// WIC takes UTF-16 paths; the rest of the engine speaks UTF-8.
std::wstring widen(const std::string& s) {
	if (s.empty()) throw MediaError("image_io: empty path");
	int need = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
	if (need <= 0) throw MediaError("image_io: path is not valid UTF-8");
	std::wstring w((size_t)need, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), need);
	return w;
}

// WIC's 24bpp formats are stored as B,G,R in memory (the GUID name describes the
// logical channel order, the byte order is BGR). The engine's buffers are RGB,
// so every crossing of that boundary swaps the first and last byte — this is the
// classic channel-swap bug, and it is why the round-trip test uses an image that
// is neither symmetric nor grey.
void swap_rb_inplace(unsigned char* p, size_t pixels) {
	for (size_t i = 0; i < pixels; i++, p += 3) std::swap(p[0], p[2]);
}

// ── resize maths ───────────────────────────────────────────────────────────

// One output coordinate's source taps. Samples left of the image fold onto
// sample 0 (`lo`) and samples past the end onto sample src-1 (`hi`), which is
// the same result as the reference dropping the out-of-range taps and
// renormalising, because both put exactly that weight on the edge pixel.
struct TapRow {
	int first = 0;
	std::vector<float> w;  // taps for first .. first + w.size() - 1
	float lo = 0.0f;       // weight onto sample 0
	float hi = 0.0f;       // weight onto sample src-1
};

// kPi by hand: <cmath> only defines M_PI outside of strict -std=c++NN mode.
constexpr double kPi = 3.14159265358979323846;

double sinc(double x) {
	if (x == 0.0) return 1.0;
	double px = kPi * x;
	return std::sin(px) / px;
}

double lanczos3(double t) {
	const double a = 3.0;
	if (t <= -a || t >= a) return 0.0;
	return sinc(t) * sinc(t / a);
}

// Separable weights: exactly one table per axis, reused for every row/column.
//
// The window is widened by 1/ratio when downscaling (`fs`), which is the
// antialias part: below 1:1 an output pixel averages several source pixels, and
// a fixed 3-sample window would just alias.
std::vector<TapRow> build_taps(int src, int dst) {
	const double a = 3.0;
	const double scale = (double)src / (double)dst;
	const double fs = scale > 1.0 ? scale : 1.0;  // filterscale
	const double support = a * fs;

	std::vector<TapRow> rows((size_t)dst);
	for (int o = 0; o < dst; o++) {
		// centre in "sample index + 0.5" space, matching the reference: with
		// scale == 1 this puts all the weight on sample o and nothing elsewhere.
		const double c = ((double)o + 0.5) * scale;
		TapRow& r = rows[(size_t)o];
		const int i0 = (int)std::floor(c - support - 0.5);
		const int i1 = (int)std::ceil(c + support - 0.5);
		int start = i1 + 1, end = i0 - 1;  // in-range span actually touched
		for (int i = i0; i <= i1; i++) {
			const double t = ((double)i + 0.5 - c) / fs;
			const double w = lanczos3(t);
			if (w == 0.0) continue;
			if (i < 0) {
				r.lo += (float)w;
			} else if (i >= src) {
				r.hi += (float)w;
			} else {
				if (i < start) start = i;
				if (i > end) end = i;
				r.w.push_back((float)w);
			}
		}
		if (end >= start) r.first = start;

		float sum = r.lo + r.hi;
		for (float v : r.w) sum += v;
		if (!(sum > 0.0f)) {
			// Degenerate (extreme downscale on a 1-pixel axis, or a rounding
			// hole): fall back to the nearest sample instead of dividing by zero.
			int nearest = (int)std::floor(c);
			nearest = std::min(std::max(nearest, 0), src - 1);
			r.first = nearest;
			r.w.assign(1, 1.0f);
			r.lo = r.hi = 0.0f;
			continue;
		}
		const float inv = 1.0f / sum;
		r.lo *= inv;
		r.hi *= inv;
		for (float& v : r.w) v *= inv;
	}
	return rows;
}

}  // namespace

// ── PNG / JPEG via WIC ─────────────────────────────────────────────────────

void write_png(const std::string& path, const std::vector<unsigned char>& rgb, int w, int h) {
	if (w <= 0 || h <= 0) throw MediaError("write_png: empty image");
	const size_t need = (size_t)w * (size_t)h * 3;
	if (rgb.size() < need) throw MediaError("write_png: pixel buffer smaller than w*h*3");

	ensure_com("write_png");
	const std::wstring wpath = widen(path);

	IWICImagingFactory* factory = nullptr;
	IWICStream* stream = nullptr;
	IWICBitmapEncoder* encoder = nullptr;
	IWICBitmapFrameEncode* frame = nullptr;
	IPropertyBag2* props = nullptr;
	IWICBitmap* bmp = nullptr;
	IWICFormatConverter* conv = nullptr;
	try {
		check_hr(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
		                          IID_IWICImagingFactory, (void**)&factory),
		         "CoCreateInstance(WICImagingFactory)");
		check_hr(factory->CreateStream(&stream), "CreateStream");
		check_hr(stream->InitializeFromFilename(wpath.c_str(), GENERIC_WRITE),
		         "IWICStream::InitializeFromFilename");
		check_hr(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder),
		         "CreateEncoder(Png)");
		check_hr(encoder->Initialize(static_cast<IStream*>(stream), WICBitmapEncoderNoCache),
		         "IWICBitmapEncoder::Initialize");
		check_hr(encoder->CreateNewFrame(&frame, &props), "CreateNewFrame");
		check_hr(frame->Initialize(props), "IWICBitmapFrameEncode::Initialize");
		check_hr(frame->SetSize((UINT)w, (UINT)h), "SetSize");

		WICPixelFormatGUID fmt = GUID_WICPixelFormat24bppBGR;
		check_hr(frame->SetPixelFormat(&fmt), "SetPixelFormat");

		std::vector<unsigned char> bgr(rgb.begin(), rgb.begin() + (ptrdiff_t)need);
		swap_rb_inplace(bgr.data(), (size_t)w * (size_t)h);

		if (IsEqualGUID(fmt, GUID_WICPixelFormat24bppBGR)) {
			check_hr(frame->WritePixels((UINT)h, (UINT)w * 3, (UINT)bgr.size(), bgr.data()),
			         "WritePixels");
		} else {
			// SetPixelFormat may hand back a *different* format than the one asked
			// for. Writing the BGR bytes anyway would produce garbage, so go through
			// a converter for whatever the encoder negotiated.
			check_hr(factory->CreateBitmapFromMemory((UINT)w, (UINT)h, GUID_WICPixelFormat24bppBGR,
			                                         (UINT)w * 3, (UINT)bgr.size(), bgr.data(),
			                                         &bmp),
			         "CreateBitmapFromMemory");
			check_hr(factory->CreateFormatConverter(&conv), "CreateFormatConverter");
			check_hr(conv->Initialize(static_cast<IWICBitmapSource*>(bmp), fmt,
			                          WICBitmapDitherTypeNone, nullptr, 0.0,
			                          WICBitmapPaletteTypeCustom),
			         "IWICFormatConverter::Initialize(png)");
			check_hr(frame->WriteSource(static_cast<IWICBitmapSource*>(conv), nullptr),
			         "WriteSource");
		}
		check_hr(frame->Commit(), "frame->Commit");
		check_hr(encoder->Commit(), "encoder->Commit");
	} catch (...) {
		if (conv) conv->Release();
		if (bmp) bmp->Release();
		if (props) props->Release();
		if (frame) frame->Release();
		if (encoder) encoder->Release();
		if (stream) stream->Release();
		if (factory) factory->Release();
		throw;
	}
	if (conv) conv->Release();
	if (bmp) bmp->Release();
	if (props) props->Release();
	if (frame) frame->Release();
	if (encoder) encoder->Release();
	if (stream) stream->Release();
	if (factory) factory->Release();
}

std::vector<unsigned char> read_image(const std::string& path, int* out_w, int* out_h) {
	ensure_com("read_image");
	const std::wstring wpath = widen(path);

	IWICImagingFactory* factory = nullptr;
	IWICBitmapDecoder* decoder = nullptr;
	IWICBitmapFrameDecode* frame = nullptr;
	IWICFormatConverter* conv = nullptr;
	std::vector<unsigned char> buf;
	try {
		check_hr(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
		                          IID_IWICImagingFactory, (void**)&factory),
		         "CoCreateInstance(WICImagingFactory)");
		// The decoder is picked from the file, so this covers PNG *and* JPEG (and
		// anything else WIC has a codec for) without the caller knowing which.
		check_hr(factory->CreateDecoderFromFilename(wpath.c_str(), nullptr, GENERIC_READ,
		                                            WICDecodeMetadataCacheOnDemand, &decoder),
		         "CreateDecoderFromFilename");
		check_hr(decoder->GetFrame(0, &frame), "IWICBitmapDecoder::GetFrame");

		UINT uw = 0, uh = 0;
		check_hr(frame->GetSize(&uw, &uh), "GetSize");
		if (uw == 0 || uh == 0) throw MediaError("read_image: zero-sized image");

		// Convert first, so every source format (palette, 16bpp, CMYK, alpha, ...)
		// lands in one known layout instead of being special-cased per codec.
		check_hr(factory->CreateFormatConverter(&conv), "CreateFormatConverter");
		check_hr(conv->Initialize(static_cast<IWICBitmapSource*>(frame),
		                          GUID_WICPixelFormat24bppBGR, WICBitmapDitherTypeNone, nullptr,
		                          0.0, WICBitmapPaletteTypeCustom),
		         "IWICFormatConverter::Initialize(24bppBGR)");

		const size_t stride = (size_t)uw * 3;
		buf.resize(stride * (size_t)uh);
		check_hr(conv->CopyPixels(nullptr, (UINT)stride, (UINT)buf.size(), buf.data()),
		         "CopyPixels");
		swap_rb_inplace(buf.data(), (size_t)uw * (size_t)uh);

		if (out_w) *out_w = (int)uw;
		if (out_h) *out_h = (int)uh;
	} catch (...) {
		if (conv) conv->Release();
		if (frame) frame->Release();
		if (decoder) decoder->Release();
		if (factory) factory->Release();
		throw;
	}
	if (conv) conv->Release();
	if (frame) frame->Release();
	if (decoder) decoder->Release();
	if (factory) factory->Release();
	return buf;
}

// ── in-memory codec crossings ──────────────────────────────────────────────

namespace {

// Seekable read/write IStream over a *private copy* of `data`. The copy is not
// laziness: WIC wants an IStream it can seek, and the caller's buffer is either
// const (a session's stored bytes) or has to stay usable afterwards. The
// HGLOBAL is handed to the stream with fDeleteOnRelease = TRUE, so releasing the
// stream releases it too and no path has to free it twice.
IStream* stream_over_memory(const unsigned char* data, size_t size, const char* what) {
	HGLOBAL hg = GlobalAlloc(GMEM_MOVEABLE, size);
	if (!hg) throw MediaError(std::string(what) + ": GlobalAlloc failed");
	void* p = GlobalLock(hg);
	if (!p) {
		GlobalFree(hg);
		throw MediaError(std::string(what) + ": GlobalLock failed");
	}
	memcpy(p, data, size);
	GlobalUnlock(hg);
	IStream* stream = nullptr;
	HRESULT hr = CreateStreamOnHGlobal(hg, TRUE, &stream);
	if (FAILED(hr) || !stream) {
		GlobalFree(hg);
		char buf[160];
		snprintf(buf, sizeof buf, "image_io: CreateStreamOnHGlobal failed (hr=0x%08lX)",
		         (unsigned long)hr);
		throw MediaError(buf);
	}
	return stream;
}

}  // namespace

std::vector<unsigned char> decode_image_memory(const unsigned char* data, size_t size,
                                               int* out_w, int* out_h) {
	if (!data || size == 0) throw MediaError("decode_image_memory: empty buffer");
	if (size > (size_t)UINT_MAX) throw MediaError("decode_image_memory: buffer too large for WIC");
	ensure_com("decode_image_memory");

	IWICImagingFactory* factory = nullptr;
	IStream* stream = nullptr;
	IWICBitmapDecoder* decoder = nullptr;
	IWICBitmapFrameDecode* frame = nullptr;
	IWICFormatConverter* conv = nullptr;
	std::vector<unsigned char> buf;
	try {
		check_hr(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
		                          IID_IWICImagingFactory, (void**)&factory),
		         "CoCreateInstance(WICImagingFactory)");
		stream = stream_over_memory(data, size, "decode_image_memory");
		// The decoder is picked from the container bytes, exactly like
		// CreateDecoderFromFilename picks it from the extension.
		check_hr(factory->CreateDecoderFromStream(stream, nullptr, WICDecodeMetadataCacheOnDemand,
		                                          &decoder),
		         "CreateDecoderFromStream");
		check_hr(decoder->GetFrame(0, &frame), "IWICBitmapDecoder::GetFrame");

		UINT uw = 0, uh = 0;
		check_hr(frame->GetSize(&uw, &uh), "GetSize");
		if (uw == 0 || uh == 0) throw MediaError("decode_image_memory: zero-sized image");

		check_hr(factory->CreateFormatConverter(&conv), "CreateFormatConverter");
		check_hr(conv->Initialize(static_cast<IWICBitmapSource*>(frame),
		                          GUID_WICPixelFormat24bppBGR, WICBitmapDitherTypeNone, nullptr,
		                          0.0, WICBitmapPaletteTypeCustom),
		         "IWICFormatConverter::Initialize(24bppBGR)");

		const size_t stride = (size_t)uw * 3;
		buf.resize(stride * (size_t)uh);
		check_hr(conv->CopyPixels(nullptr, (UINT)stride, (UINT)buf.size(), buf.data()),
		         "CopyPixels");
		swap_rb_inplace(buf.data(), (size_t)uw * (size_t)uh);

		if (out_w) *out_w = (int)uw;
		if (out_h) *out_h = (int)uh;
	} catch (...) {
		if (conv) conv->Release();
		if (frame) frame->Release();
		if (decoder) decoder->Release();
		if (stream) stream->Release();
		if (factory) factory->Release();
		throw;
	}
	if (conv) conv->Release();
	if (frame) frame->Release();
	if (decoder) decoder->Release();
	if (stream) stream->Release();
	if (factory) factory->Release();
	return buf;
}

std::vector<unsigned char> encode_jpeg_memory(const unsigned char* rgb, int w, int h, int quality) {
	if (!rgb || w <= 0 || h <= 0) throw MediaError("encode_jpeg_memory: empty image");
	ensure_com("encode_jpeg_memory");
	const int q = std::clamp(quality, 1, 100);

	IWICImagingFactory* factory = nullptr;
	IStream* stream = nullptr;
	IWICBitmapEncoder* encoder = nullptr;
	IWICBitmapFrameEncode* frame = nullptr;
	IPropertyBag2* props = nullptr;
	IWICBitmap* bmp = nullptr;
	IWICFormatConverter* conv = nullptr;
	std::vector<unsigned char> out;
	try {
		check_hr(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
		                          IID_IWICImagingFactory, (void**)&factory),
		         "CoCreateInstance(WICImagingFactory)");
		// NULL init + fDeleteOnRelease: a fresh, growable HGLOBAL-backed stream.
		check_hr(CreateStreamOnHGlobal(nullptr, TRUE, &stream), "CreateStreamOnHGlobal");
		check_hr(factory->CreateEncoder(GUID_ContainerFormatJpeg, nullptr, &encoder),
		         "CreateEncoder(Jpeg)");
		check_hr(encoder->Initialize(static_cast<IStream*>(stream), WICBitmapEncoderNoCache),
		         "IWICBitmapEncoder::Initialize");
		check_hr(encoder->CreateNewFrame(&frame, &props), "CreateNewFrame");
		// ImageQuality is JPEG's own knob (1.0 == q100) and it is consumed by the
		// frame's Initialize below. It has to be written to the bag *before* that
		// call: afterwards the write still returns S_OK and changes nothing (the
		// encoder keeps its default), which is a silent way to ship every picture
		// at the wrong quality. A rejected write is not fatal either — the
		// encoder's own default is a fine fallback.
		if (props) {
			PROPBAG2 opt = {};
			opt.pstrName = const_cast<LPOLESTR>(L"ImageQuality");
			VARIANT var;
			VariantInit(&var);
			var.vt = VT_R4;
			var.fltVal = (float)q / 100.0f;
			props->Write(1, &opt, &var);
			VariantClear(&var);
		}
		check_hr(frame->Initialize(props), "IWICBitmapFrameEncode::Initialize");
		check_hr(frame->SetSize((UINT)w, (UINT)h), "SetSize");

		WICPixelFormatGUID fmt = GUID_WICPixelFormat24bppBGR;
		check_hr(frame->SetPixelFormat(&fmt), "SetPixelFormat");

		std::vector<unsigned char> bgr(rgb, rgb + (size_t)w * (size_t)h * 3);
		swap_rb_inplace(bgr.data(), (size_t)w * (size_t)h);

		if (IsEqualGUID(fmt, GUID_WICPixelFormat24bppBGR)) {
			check_hr(frame->WritePixels((UINT)h, (UINT)w * 3, (UINT)bgr.size(), bgr.data()),
			         "WritePixels");
		} else {
			// SetPixelFormat may hand back a *different* format than the one asked
			// for; writing the BGR bytes anyway would produce garbage (see write_png).
			check_hr(factory->CreateBitmapFromMemory((UINT)w, (UINT)h, GUID_WICPixelFormat24bppBGR,
			                                         (UINT)w * 3, (UINT)bgr.size(), bgr.data(), &bmp),
			         "CreateBitmapFromMemory");
			check_hr(factory->CreateFormatConverter(&conv), "CreateFormatConverter");
			check_hr(conv->Initialize(static_cast<IWICBitmapSource*>(bmp), fmt,
			                          WICBitmapDitherTypeNone, nullptr, 0.0,
			                          WICBitmapPaletteTypeCustom),
			         "IWICFormatConverter::Initialize(jpeg)");
			check_hr(frame->WriteSource(static_cast<IWICBitmapSource*>(conv), nullptr),
			         "WriteSource");
		}
		check_hr(frame->Commit(), "frame->Commit");
		check_hr(encoder->Commit(), "encoder->Commit");

		// The committed bytes live in the stream's HGLOBAL. Stat() is the
		// authoritative length: the block itself is allocation-granular (bigger).
		STATSTG st = {};
		check_hr(stream->Stat(&st, STATFLAG_NONAME), "IStream::Stat");
		HGLOBAL hg = nullptr;
		check_hr(GetHGlobalFromStream(stream, &hg), "GetHGlobalFromStream");
		const size_t have = (size_t)GlobalSize(hg);
		size_t n = (size_t)st.cbSize.QuadPart;
		if (n == 0 || n > have) n = have;
		if (n == 0) throw MediaError("encode_jpeg_memory: encoder produced no bytes");
		const void* p = GlobalLock(hg);
		if (!p) throw MediaError("encode_jpeg_memory: GlobalLock failed");
		out.assign((const unsigned char*)p, (const unsigned char*)p + n);
		GlobalUnlock(hg);
	} catch (...) {
		if (conv) conv->Release();
		if (bmp) bmp->Release();
		if (props) props->Release();
		if (frame) frame->Release();
		if (encoder) encoder->Release();
		if (stream) stream->Release();
		if (factory) factory->Release();
		throw;
	}
	if (conv) conv->Release();
	if (bmp) bmp->Release();
	if (props) props->Release();
	if (frame) frame->Release();
	if (encoder) encoder->Release();
	if (stream) stream->Release();
	if (factory) factory->Release();
	return out;
}

// ── canvas + centre crop ───────────────────────────────────────────────────

void crop_centre(const unsigned char* src, int src_w, int src_h, int dst_w, int dst_h,
                 std::vector<unsigned char>& out) {
	if (!src || src_w <= 0 || src_h <= 0) throw MediaError("crop_centre: empty source");
	if (dst_w <= 0 || dst_h <= 0) throw MediaError("crop_centre: empty destination");
	if (dst_w > src_w || dst_h > src_h) {
		throw MediaError("crop_centre: destination is larger than the source");
	}
	// Integer division floors, so an odd delta leaves the extra pixel on the
	// right/bottom — the canvas is only there to give the DiT a multiple of 32,
	// and the reference crop steals that pixel from the far side.
	const int x0 = (src_w - dst_w) / 2;
	const int y0 = (src_h - dst_h) / 2;
	const size_t row_bytes = (size_t)dst_w * 3;
	out.resize(row_bytes * (size_t)dst_h);
	for (int y = 0; y < dst_h; y++) {
		memcpy(out.data() + (size_t)y * row_bytes,
		       src + (((size_t)(y + y0) * (size_t)src_w) + (size_t)x0) * 3, row_bytes);
	}
}

// ── separable Lanczos-3 resize ─────────────────────────────────────────────

void resize_lanczos3(const unsigned char* src, int src_w, int src_h, int dst_w, int dst_h,
                     std::vector<unsigned char>& out) {
	if (!src || src_w <= 0 || src_h <= 0) throw MediaError("resize_lanczos3: empty source");
	if (dst_w <= 0 || dst_h <= 0) throw MediaError("resize_lanczos3: empty destination");
	const size_t out_bytes = (size_t)dst_w * (size_t)dst_h * 3;
	if (dst_w == src_w && dst_h == src_h) {  // identity must be bit-exact
		out.assign(src, src + (((size_t)src_w * (size_t)src_h) * 3));
		return;
	}
	out.assign(out_bytes, 0);

	const std::vector<TapRow> hx = build_taps(src_w, dst_w);
	const std::vector<TapRow> vy = build_taps(src_h, dst_h);

	// Horizontal pass into float: the vertical pass then reads a fully filtered
	// image, and quantising only once at the end avoids a double rounding.
	std::vector<float> tmp((size_t)dst_w * (size_t)src_h * 3);
	const size_t src_row = (size_t)src_w * 3;
	const size_t dst_row = (size_t)dst_w * 3;
	for (int y = 0; y < src_h; y++) {
		const unsigned char* srow = src + (size_t)y * src_row;
		float* trow = tmp.data() + (size_t)y * dst_row;
		for (int o = 0; o < dst_w; o++) {
			const TapRow& r = hx[(size_t)o];
			float a0 = r.lo * srow[0] + r.hi * srow[src_row - 3];
			float a1 = r.lo * srow[1] + r.hi * srow[src_row - 2];
			float a2 = r.lo * srow[2] + r.hi * srow[src_row - 1];
			const unsigned char* p = srow + (size_t)r.first * 3;
			for (size_t j = 0; j < r.w.size(); j++, p += 3) {
				const float w = r.w[j];
				a0 += w * p[0];
				a1 += w * p[1];
				a2 += w * p[2];
			}
			float* d = trow + (size_t)o * 3;
			d[0] = a0;
			d[1] = a1;
			d[2] = a2;
		}
	}

	const float* first_row = tmp.data();
	const float* last_row = tmp.data() + (size_t)(src_h - 1) * dst_row;

	// Vertical pass, accumulated row-wise so the scratch line stays in cache.
	std::vector<float> acc(dst_row, 0.0f);
	std::vector<unsigned char> staging(out_bytes);
	for (int oy = 0; oy < dst_h; oy++) {
		const TapRow& r = vy[(size_t)oy];
		std::fill(acc.begin(), acc.end(), 0.0f);
		if (r.lo != 0.0f) {
			for (size_t i = 0; i < dst_row; i++) acc[i] += r.lo * first_row[i];
		}
		if (r.hi != 0.0f) {
			for (size_t i = 0; i < dst_row; i++) acc[i] += r.hi * last_row[i];
		}
		for (size_t j = 0; j < r.w.size(); j++) {
			const float* srow = tmp.data() + (size_t)(r.first + (int)j) * dst_row;
			const float w = r.w[j];
			for (size_t i = 0; i < dst_row; i++) acc[i] += w * srow[i];
		}
		unsigned char* drow = staging.data() + (size_t)oy * dst_row;
		for (size_t i = 0; i < dst_row; i++) {
			// Clamp then round half up; the reference also clamps after filtering
			// (a sinc kernel overshoots on hard edges, so clamping is required).
			float v = std::max(0.0f, acc[i]);  // max(0, NaN) == 0
			v = std::min(255.0f, v);
			drow[i] = (unsigned char)(v + 0.5f);
		}
	}
	out.swap(staging);
}

// ── float HWC -> 8-bit ─────────────────────────────────────────────────────

void float_to_srgb8(const float* src, int w, int h, int channels,
                    std::vector<unsigned char>& out) {
	if (!src || w <= 0 || h <= 0 || channels <= 0) {
		throw MediaError("float_to_srgb8: bad shape");
	}
	const size_t n = (size_t)w * (size_t)h * (size_t)channels;
	out.resize(n);
	for (size_t i = 0; i < n; i++) {
		// The image path feeds a decoded VAE sample here, which lives in [-1,1];
		// the clamp range is therefore baked in as that interval. Rounding is
		// half away from zero *after* the [-1,1] -> [0,255] mapping, so 0.0 maps
		// to 128 and not 127 (127.5 rounds up).
		float v = (src[i] + 1.0f) * 0.5f;
		if (!(v > 0.0f)) v = 0.0f;  // also catches NaN
		else if (v > 1.0f) v = 1.0f;
		out[i] = (unsigned char)(v * 255.0f + 0.5f);
	}
}

void float01_to_srgb8(const float* src, int w, int h, int channels,
                      std::vector<unsigned char>& out) {
	if (!src || w <= 0 || h <= 0 || channels <= 0) {
		throw MediaError("float01_to_srgb8: bad shape");
	}
	const size_t n = (size_t)w * (size_t)h * (size_t)channels;
	out.resize(n);
	for (size_t i = 0; i < n; i++) {
		// The video VAE's decode already ends in clamp(v, 0, 1) (vae.py), so this
		// is the reference's `(image * 255)` with its own clamp kept for safety —
		// including for NaN, which `!(v > 0)` catches.
		float v = src[i];
		if (!(v > 0.0f)) v = 0.0f;
		else if (v > 1.0f) v = 1.0f;
		out[i] = (unsigned char)(v * 255.0f + 0.5f);
	}
}

void video_frame_to_hwc(const float* planes, int channels, int frames, int f, int h, int w,
                        std::vector<float>& out) {
	if (!planes || channels <= 0 || frames <= 0 || h <= 0 || w <= 0)
		throw MediaError("video_frame_to_hwc: bad shape");
	if (f < 0 || f >= frames) throw MediaError("video_frame_to_hwc: frame out of range");
	const size_t plane = (size_t)h * (size_t)w;
	out.resize(plane * (size_t)channels);
	// Source is [C, T, H, W]; the destination is one frame's HWC.
	for (int c = 0; c < channels; c++) {
		const float* p = planes + ((size_t)c * (size_t)frames + (size_t)f) * plane;
		for (size_t i = 0; i < plane; i++) out[i * (size_t)channels + (size_t)c] = p[i];
	}
}

void chw_to_hwc(const float* src, int channels, int h, int w, std::vector<float>& out) {
	if (!src || channels <= 0 || h <= 0 || w <= 0) throw MediaError("chw_to_hwc: bad shape");
	const size_t plane = (size_t)h * (size_t)w;
	out.resize(plane * (size_t)channels);
	for (int c = 0; c < channels; c++) {
		const float* p = src + plane * (size_t)c;
		for (size_t i = 0; i < plane; i++) out[i * (size_t)channels + (size_t)c] = p[i];
	}
}

void hwc_to_chw(const float* src, int h, int w, int channels, std::vector<float>& out) {
	if (!src || channels <= 0 || h <= 0 || w <= 0) throw MediaError("hwc_to_chw: bad shape");
	const size_t plane = (size_t)h * (size_t)w;
	out.resize(plane * (size_t)channels);
	for (int c = 0; c < channels; c++) {
		float* p = out.data() + plane * (size_t)c;
		for (size_t i = 0; i < plane; i++) p[i] = src[i * (size_t)channels + (size_t)c];
	}
}

}  // namespace phi::media
