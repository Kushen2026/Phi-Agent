// mf_mux — MP4 muxing through the Media Foundation Sink Writer.
//
// open() creates an IMFSinkWriter for the file and gives it a video stream and
// an audio stream.  Which encoder drives each stream is a property of the
// machine, so the candidates are enumerated (MFTEnumEx) and tried in a fixed
// priority order:
//
//   video   hardware H.264 > software H.264 > Motion JPEG
//   audio   AAC > FLAC > MP3 > no audio track
//
// Hardware first because it is what the picture quality is judged by here: a
// hardware H.264 encoder (NVIDIA NVENC, Intel Quick Sync, AMD AMF) is reached
// only when the writer is created with MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS,
// and it buys a better picture per bit than the software encoder beside a much
// lower host cost.  H.264 itself is preferred to HEVC for compatibility: every
// player, browser and editor reads it.  The sink writer owns the encoders and
// the container: the caller only hands over RGB frames and float PCM with their
// presentation times, and close() flushes and finalises.
//
// Linkage note: see video_io.cpp — Media Foundation is resolved at runtime
// (GetProcAddress) because bin/phi.exe's link line has no mfplat/mfreadwrite,
// and the attribute GUIDs are spelled out locally.
#include "io/mf_mux.hpp"

#include <windows.h>

#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfreadwrite.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>
#include <string>
#include <vector>

namespace phi::media {
namespace {

constexpr i64 kTicksPerSecond = 10000000;  // 100 ns units, MF's time base
constexpr i64 kMaxBitrate = 20 * 1000 * 1000;

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

// Thin wrappers so that not a single MF API is called through the import
// library (there is none on the link line).
HRESULT mf_create_memory_buffer(DWORD size, IMFMediaBuffer** out) {
	using Fn = HRESULT(WINAPI*)(DWORD, IMFMediaBuffer**);
	static Fn fn = nullptr;
	if (!fn) fn = mf_import<Fn>("MFCreateMemoryBuffer");
	return fn(size, out);
}

HRESULT mf_create_sample(IMFSample** out) {
	using Fn = HRESULT(WINAPI*)(IMFSample**);
	static Fn fn = nullptr;
	if (!fn) fn = mf_import<Fn>("MFCreateSample");
	return fn(out);
}

HRESULT mf_create_attributes(DWORD size, IMFAttributes** out) {
	using Fn = HRESULT(WINAPI*)(IMFAttributes**, UINT32);
	static Fn fn = nullptr;
	if (!fn) fn = mf_import<Fn>("MFCreateAttributes");
	return fn(out, size);
}

void mf_startup() {
	static bool done = false;
	if (done) return;
	using StartupFn = HRESULT(WINAPI*)(ULONG, DWORD);
	StartupFn startup = mf_import<StartupFn>("MFStartup");
	HRESULT hr = startup(MF_VERSION, 0);
	if (FAILED(hr)) {
		char buf[64];
		snprintf(buf, sizeof buf, "0x%08lX", (unsigned long)hr);
		throw MediaError(std::string("MFStartup failed (") + buf + ")");
	}
	done = true;
}

std::wstring utf8_to_wide(const std::string& s) {
	int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
	std::wstring w(n > 0 ? (size_t)n : 0, L'\0');
	if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
	return w;
}



std::string hr_text(HRESULT hr) {
	char buf[32];
	snprintf(buf, sizeof buf, "0x%08lX", (unsigned long)hr);
	return buf;
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
const GUID kMtAvgBitrate = {0x20332624, 0xFB0D, 0x4D9E, {0xBD, 0x0D, 0xCB, 0xF6, 0x78, 0x6C, 0x10, 0x2E}};
const GUID kMtAudioNumChannels = {0x37E48BF5, 0x645E, 0x4C5B, {0x89, 0xDE, 0xAD, 0xA9, 0xE2, 0x9B, 0x69, 0x6A}};
const GUID kMtAudioSamplesPerSecond = {0x5FAEEAE7, 0x0290, 0x4C31, {0x9E, 0x8A, 0xC5, 0x34, 0xF6, 0x8D, 0x9D, 0xBA}};
const GUID kMtAudioBitsPerSample = {0xF2DEB57F, 0x40FA, 0x4764, {0xAA, 0x33, 0xED, 0x4F, 0x2D, 0x1F, 0xF6, 0x69}};
const GUID kMtAudioBlockAlignment = {0x322DE230, 0x9EEB, 0x43BD, {0xAB, 0x7A, 0xFF, 0x41, 0x22, 0x51, 0x54, 0x1D}};
const GUID kMtAudioAvgBytesPerSecond = {0x1AAB75C8, 0xCFEF, 0x451C, {0xAB, 0x95, 0xAC, 0x03, 0x4B, 0x8E, 0x17, 0x31}};
const GUID kMediaTypeVideo = {0x73646976, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kMediaTypeAudio = {0x73647561, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kVideoFormatRgb32 = {0x00000016, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kVideoFormatH264 = {0x34363248, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kVideoFormatMjpg = {0x47504A4D, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kAudioFormatPcm = {0x00000001, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kAudioFormatAac = {0x00001610, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kAudioFormatFlac = {0x0000F1AC, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kAudioFormatMp3 = {0x00000055, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kMtMpeg2Profile = {0xAD76A80B, 0x2D5C, 0x4E0B, {0xB3, 0x75, 0x64, 0xE5, 0x20, 0x13, 0x70, 0x36}};

// ── encoder discovery ─────────────────────────────────────────────────────

const GUID kMftCategoryVideoEncoder = {0xF79EAC7D, 0xE545, 0x4387, {0xBD, 0xEE, 0xD6, 0x47, 0xD7, 0xBD, 0xE4, 0x2A}};
const GUID kMftCategoryAudioEncoder = {0x91C64BD0, 0xF91E, 0x4D8C, {0x92, 0x76, 0xDB, 0x24, 0x82, 0x79, 0xD9, 0x75}};
const GUID kMftFriendlyName = {0x314FFBAE, 0x5B41, 0x4C95, {0x9C, 0x19, 0x4E, 0x7D, 0x58, 0x6F, 0xAC, 0xE3}};
// MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS. Without it on the writer's attribute
// store the sink writer never looks at a hardware MFT at all, whatever the box
// has installed; with it, hardware candidates are preferred and the software
// ones remain the fallback.
const GUID kEnableHardwareTransforms = {0xA634A91C, 0x822B, 0x41B9, {0xA4, 0x94, 0x4D, 0xE4, 0x64, 0x36, 0x12, 0xB0}};

constexpr UINT32 kEnumFlagAll = 0x3F;       // MFT_ENUM_FLAG_ALL
constexpr UINT32 kEnumFlagHardware = 0x04;  // MFT_ENUM_FLAG_HARDWARE
constexpr UINT32 kEnumFlagSoftware = kEnumFlagAll & ~kEnumFlagHardware;
constexpr UINT32 kH264ProfileHigh = 100;  // eAVEncH264VProfile_High

constexpr UINT32 kProgressive = 2;  // MFVideoInterlace_Progressive

short to_i16(float v) {
	if (v >= 1.0f) return 32767;
	if (v <= -1.0f) return -32768;
	return (short)lrintf(v * 32767.0f);
}

// A sane constant bitrate for the requested frame size/rate; the encoder uses
// it for rate control and the container stores it in the sample description.
i64 video_bitrate(i64 w, i64 h, i64 fps) {
	i64 br = w * h * fps / 8;  // ~1 bit per pixel per frame, generous but bounded
	if (br < 250000) br = 250000;
	if (br > kMaxBitrate) br = kMaxBitrate;
	return br;
}


std::string wide_to_utf8(const std::wstring& w) {
	if (w.empty()) return {};
	int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
	std::string s(n > 0 ? (size_t)n : 0, '\0');
	if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
	return s;
}

// Is an encoder for this subtype registered on this machine? Fills `name` with
// the first matching MFT's friendly name, so the log and the tool's report can
// say which encoder actually ran rather than which one was hoped for.  A box
// whose mfplat has no MFTEnumEx (or none of the codec) simply reports false and
// the caller falls through to the next candidate.
bool encoder_registered(const GUID& category, const GUID& major, const GUID& subtype, UINT32 flags,
                        std::string* name) {
	using EnumFn = HRESULT(WINAPI*)(GUID, UINT32, const MFT_REGISTER_TYPE_INFO*,
	                                const MFT_REGISTER_TYPE_INFO*, IMFActivate***, UINT32*);
	EnumFn enumerate = nullptr;
	try {
		enumerate = mf_import<EnumFn>("MFTEnumEx");
	} catch (const MediaError&) {
		return false;
	}
	const MFT_REGISTER_TYPE_INFO out = {major, subtype};
	IMFActivate** acts = nullptr;
	UINT32 count = 0;
	if (FAILED(enumerate(category, flags, nullptr, &out, &acts, &count))) {
		if (acts) CoTaskMemFree(acts);
		return false;
	}
	if (count > 0 && name) {
		wchar_t* wide = nullptr;
		UINT32 wide_len = 0;
		if (SUCCEEDED(acts[0]->GetAllocatedString(kMftFriendlyName, &wide, &wide_len)) && wide) {
			*name = wide_to_utf8(std::wstring(wide, wide_len));
			CoTaskMemFree(wide);
		}
	}
	for (UINT32 i = 0; i < count; i++) acts[i]->Release();
	CoTaskMemFree(acts);
	return count > 0;
}

// One encoder to try: the subtype to configure the stream for, whether the
// sink writer may use a hardware MFT for it, the tag it is named by in the log
// (and by the override below), and the name to report.
struct VideoCandidate {
	GUID subtype;
	bool hardware;
	const char* tag;
	std::string name;
};

struct AudioCandidate {
	GUID subtype;
	const char* tag;
	std::string name;
};

// PHI_VIDEO_ENCODER / PHI_AUDIO_ENCODER narrow the list to one encoder. The
// search picks for the machine, not for the job, and a driver that registers a
// hardware encoder it cannot actually run is the one case where the caller
// knows better: `PHI_VIDEO_ENCODER=software` and `PHI_AUDIO_ENCODER=none` are
// the escape hatches.  Unset or "auto" keeps the built-in order.
const char* encoder_override(const char* env) {
	const char* v = getenv(env);
	if (!v || !*v || strcmp(v, "auto") == 0) return nullptr;
	return v;
}

// The machine's encoders in priority order: the hardware H.264 encoder (NVENC,
// Quick Sync, AMF - whatever the driver registered), the software H.264 one the
// OS ships, and Motion JPEG as the last resort, since a container that carries
// something is worth more than an error.  An empty video list means this machine
// has no video encoder at all, which open() reports rather than papering over.
// An empty audio list means the file is written without an audio track.
std::vector<VideoCandidate> video_candidates(bool want_video) {
	std::vector<VideoCandidate> out;
	if (!want_video) return out;
	std::string name;
	if (encoder_registered(kMftCategoryVideoEncoder, kMediaTypeVideo, kVideoFormatH264,
	                       kEnumFlagHardware, &name))
		out.push_back({kVideoFormatH264, true, "hardware", name});
	if (encoder_registered(kMftCategoryVideoEncoder, kMediaTypeVideo, kVideoFormatH264,
	                       kEnumFlagSoftware, &name))
		out.push_back({kVideoFormatH264, false, "software", name});
	if (encoder_registered(kMftCategoryVideoEncoder, kMediaTypeVideo, kVideoFormatMjpg,
	                       kEnumFlagAll, &name))
		out.push_back({kVideoFormatMjpg, false, "mjpeg", name});

	if (const char* want = encoder_override("PHI_VIDEO_ENCODER")) {
		std::vector<VideoCandidate> keep;
		for (const VideoCandidate& c : out)
			if (strcmp(c.tag, want) == 0) keep.push_back(c);
		if (keep.empty()) {
			fprintf(stderr, "[mf_mux] PHI_VIDEO_ENCODER=%s matches no encoder here; keeping the default order\n",
			        want);
			return out;
		}
		return keep;
	}
	return out;
}

std::vector<AudioCandidate> audio_candidates(bool want_audio) {
	std::vector<AudioCandidate> out;
	if (!want_audio) return out;
	std::string name;
	// AAC first: it is what an MP4 is expected to carry and what every player
	// decodes.  FLAC is the next best quality (lossless) but a rarer thing to
	// meet in the wild; MP3 is the universally readable fallback.
	if (encoder_registered(kMftCategoryAudioEncoder, kMediaTypeAudio, kAudioFormatAac, kEnumFlagAll, &name))
		out.push_back({kAudioFormatAac, "aac", name});
	if (encoder_registered(kMftCategoryAudioEncoder, kMediaTypeAudio, kAudioFormatFlac, kEnumFlagAll, &name))
		out.push_back({kAudioFormatFlac, "flac", name});
	if (encoder_registered(kMftCategoryAudioEncoder, kMediaTypeAudio, kAudioFormatMp3, kEnumFlagAll, &name))
		out.push_back({kAudioFormatMp3, "mp3", name});

	if (const char* want = encoder_override("PHI_AUDIO_ENCODER")) {
		if (strcmp(want, "none") == 0) return {};
		std::vector<AudioCandidate> keep;
		for (const AudioCandidate& c : out)
			if (strcmp(c.tag, want) == 0) keep.push_back(c);
		if (keep.empty()) {
			fprintf(stderr, "[mf_mux] PHI_AUDIO_ENCODER=%s matches no encoder here; keeping the default order\n",
			        want);
			return out;
		}
		return keep;
	}
	return out;
}

}  // namespace

struct Mp4Muxer::Impl {
	ComPtr<IMFSinkWriter> writer;
	DWORD video_stream = 0;
	DWORD audio_stream = 0;
	bool has_video = false;
	bool has_audio = false;
	bool finalized = false;

	// The encoders the sink writer took, as reported by open().
	std::string video_encoder;
	std::string audio_encoder;

	i64 width = 0, height = 0;
	i64 fps = 0;
	i64 sample_rate = 0, channels = 0;

	i64 frame_ticks = 0;
	i64 video_frames = 0;
	LONGLONG last_video_ts = -1;
	i64 audio_frames = 0;  // audio frames (not samples) written so far

	// Builds the sink writer and its streams for one pair of encoders. Returns
	// false (with nothing left open) when the encoder or the container rejects
	// the configuration, so open() can try the next candidate. `hardware` asks
	// the writer for a hardware MFT; `audio_subtype` may be null for a file with
	// no audio track.
	bool configure(const std::string& path, const GUID& video_subtype, bool hardware,
	               const GUID* audio_subtype);
};

Mp4Muxer::~Mp4Muxer() { close(); }

bool Mp4Muxer::Impl::configure(const std::string& path, const GUID& video_subtype, bool hardware,
                               const GUID* audio_subtype) {
	using CreateWriterFn = HRESULT(WINAPI*)(LPCWSTR, IMFByteStream*, IMFAttributes*, IMFSinkWriter**);
	using CreateTypeFn = HRESULT(WINAPI*)(IMFMediaType**);
	CreateWriterFn create_writer = mf_import<CreateWriterFn>("MFCreateSinkWriterFromURL");
	CreateTypeFn create_type = mf_import<CreateTypeFn>("MFCreateMediaType");

	// The writer's attribute store is what tells Media Foundation whether it may
	// reach for a hardware MFT at all (see kEnableHardwareTransforms above).
	ComPtr<IMFAttributes> attrs;
	if (FAILED(mf_create_attributes(4, attrs.put()))) return false;
	if (hardware) attrs->SetUINT32(kEnableHardwareTransforms, TRUE);

	const std::wstring wpath = utf8_to_wide(path);
	ComPtr<IMFSinkWriter> w;
	if (FAILED(create_writer(wpath.c_str(), nullptr, attrs.get(), w.put())) || !w) return false;

	// ── video ──────────────────────────────────────────────────────────────
	if (width > 0 && height > 0) {
		ComPtr<IMFMediaType> out;
		if (FAILED(create_type(out.put()))) return false;
		out->SetGUID(kMtMajorType, kMediaTypeVideo);
		out->SetGUID(kMtSubtype, video_subtype);
		out->SetUINT32(kMtAvgBitrate, (UINT32)video_bitrate(width, height, fps));
		if (video_subtype == kVideoFormatH264) {
			// High profile is where H.264's real compression lives (CABAC, 8x8
			// transforms, B frames). Every player of this century reads it; the
			// alternative a hardware encoder picks on its own is Constrained
			// Baseline, which is a visibly worse picture at the same bitrate.
			out->SetUINT32(kMtMpeg2Profile, kH264ProfileHigh);
		}
		if (FAILED(MFSetAttributeSize(out.get(), kMtFrameSize, (UINT32)width, (UINT32)height)))
			return false;
		if (FAILED(MFSetAttributeRatio(out.get(), kMtFrameRate, (UINT32)fps, 1))) return false;
		if (FAILED(MFSetAttributeRatio(out.get(), kMtPixelAspectRatio, 1, 1))) return false;
		out->SetUINT32(kMtInterlaceMode, kProgressive);

		DWORD stream = 0;
		if (FAILED(w->AddStream(out.get(), &stream))) return false;

		ComPtr<IMFMediaType> in;
		if (FAILED(create_type(in.put()))) return false;
		in->SetGUID(kMtMajorType, kMediaTypeVideo);
		in->SetGUID(kMtSubtype, kVideoFormatRgb32);
		if (FAILED(MFSetAttributeSize(in.get(), kMtFrameSize, (UINT32)width, (UINT32)height)))
			return false;
		if (FAILED(MFSetAttributeRatio(in.get(), kMtFrameRate, (UINT32)fps, 1))) return false;
		if (FAILED(MFSetAttributeRatio(in.get(), kMtPixelAspectRatio, 1, 1))) return false;
		in->SetUINT32(kMtInterlaceMode, kProgressive);
		// Positive stride: top-down 32-bit pixels, one row per w*4 bytes.
		in->SetUINT32(kMtDefaultStride, (UINT32)(width * 4));
		if (FAILED(w->SetInputMediaType(stream, in.get(), nullptr))) return false;

		video_stream = stream;
		has_video = true;
	}

	// ── audio (optional: a machine without an audio encoder still gets video) ─
	if (audio_subtype && sample_rate > 0 && channels > 0) {
		ComPtr<IMFMediaType> out;
		bool configured = false;
		if (SUCCEEDED(create_type(out.put())) && out) {
			out->SetGUID(kMtMajorType, kMediaTypeAudio);
			out->SetGUID(kMtSubtype, *audio_subtype);
			out->SetUINT32(kMtAudioNumChannels, (UINT32)channels);
			out->SetUINT32(kMtAudioSamplesPerSecond, (UINT32)sample_rate);
			out->SetUINT32(kMtAudioBitsPerSample, 16);
			// A lossless encoder sets its own byte rate; asking it for one is
			// what makes it reject the type.
			if (*audio_subtype != kAudioFormatFlac)
				out->SetUINT32(kMtAudioAvgBytesPerSecond, (UINT32)(16000 * channels / 2));
			DWORD stream = 0;
			if (SUCCEEDED(w->AddStream(out.get(), &stream))) {
				ComPtr<IMFMediaType> in;
				if (SUCCEEDED(create_type(in.put())) && in) {
					in->SetGUID(kMtMajorType, kMediaTypeAudio);
					in->SetGUID(kMtSubtype, kAudioFormatPcm);
					in->SetUINT32(kMtAudioNumChannels, (UINT32)channels);
					in->SetUINT32(kMtAudioSamplesPerSecond, (UINT32)sample_rate);
					in->SetUINT32(kMtAudioBitsPerSample, 16);
					in->SetUINT32(kMtAudioBlockAlignment, (UINT32)(channels * 2));
					in->SetUINT32(kMtAudioAvgBytesPerSecond, (UINT32)(sample_rate * channels * 2));
					if (SUCCEEDED(w->SetInputMediaType(stream, in.get(), nullptr))) {
						audio_stream = stream;
						has_audio = true;
						configured = true;
					}
				}
			}
		}
		if (!configured && !has_video) return false;  // nothing could carry the file
		if (!configured)
			fprintf(stderr, "[mf_mux] warning: this encoder refused the audio stream\n");
	}

	if (!has_video && !has_audio) return false;

	// BeginWriting() is documented as optional (the first WriteSample is meant
	// to trigger it), but on this build of Media Foundation the lazy path fails
	// with MF_E_INVALIDREQUEST for every sample.  Starting the pipeline here
	// surfaces the real error while the media types are still in scope.
	HRESULT begin_hr = w->BeginWriting();
	if (FAILED(begin_hr)) return false;

	writer = std::move(w);
	return true;
}

void Mp4Muxer::open(const std::string& path, i64 width, i64 height, i64 fps, i64 sample_rate,
                    i64 channels) {
	close();
	if ((width <= 0 || height <= 0) && (sample_rate <= 0 || channels <= 0))
		throw MediaError("Mp4Muxer::open: no video and no audio requested");
	if (fps < 0 || sample_rate < 0 || channels < 0)
		throw MediaError("Mp4Muxer::open: negative fps/rate/channels");
	mf_startup();

	Impl* io = new Impl();
	io->width = width > 0 ? width : 0;
	io->height = height > 0 ? height : 0;
	io->fps = fps > 0 ? fps : 24;
	io->sample_rate = sample_rate > 0 ? sample_rate : 0;
	io->channels = channels > 0 ? channels : 0;
	io->frame_ticks = kTicksPerSecond / io->fps;

	const bool want_video = io->width > 0 && io->height > 0;
	const bool want_audio = io->sample_rate > 0 && io->channels > 0;
	const std::vector<VideoCandidate> videos = video_candidates(want_video);
	const std::vector<AudioCandidate> audios = audio_candidates(want_audio);
	if (want_audio && audios.empty())
		fprintf(stderr,
		        "[mf_mux] warning: no audio encoder is registered on this machine; writing video only\n");

	auto reset_state = [&]() {
		io->has_video = false;
		io->has_audio = false;
		io->video_stream = 0;
		io->audio_stream = 0;
		io->writer.reset();
	};

	bool ok = false;
	std::string first_error;
	if (want_video && videos.empty()) {
		delete io;
		throw MediaError("Mp4Muxer::open: no video encoder is registered on this machine");
	}
	if (want_video) {
		// Both lists are walked from the top. The video encoder changes more
		// slowly (a box has a hardware one or it does not) while the audio side
		// has a longer tail of near-equivalent encoders, so the audio list is the
		// outer loop: every audio candidate is paired with the best video
		// candidate that accepts the frame, and `ai == audios.size()` is the last
		// resort of a video-only file.
		for (size_t ai = 0; ai <= audios.size() && !ok; ai++) {
			const GUID* audio_subtype = ai < audios.size() ? &audios[ai].subtype : nullptr;
			bool any_video = false;
			for (size_t vi = 0; vi < videos.size() && !ok; vi++) {
				reset_state();
				if (io->configure(path, videos[vi].subtype, videos[vi].hardware, audio_subtype)) {
					ok = true;
					any_video = true;
					io->video_encoder = videos[vi].name;
					if (io->has_audio && audio_subtype) io->audio_encoder = audios[ai].name;
				} else {
					char buf[160];
					snprintf(buf, sizeof buf,
					         "%s rejected the %lldx%lld @ %lld fps frame",
					         videos[vi].name.c_str(), (long long)io->width,
					         (long long)io->height, (long long)io->fps);
					first_error = buf;
					fprintf(stderr, "[mf_mux] %s; trying the next encoder\n", first_error.c_str());
				}
			}
			// Not one video encoder took the frame: the audio list cannot fix a
			// picture that has no encoder, so report it now instead of trying the
			// same three candidates once per audio codec.
			if (!any_video) break;
			// The picture came up but the audio encoder would not take the PCM:
			// keep the picture and try the next audio candidate from the top of
			// the video list (both streams are configured together).
			if (ok && audio_subtype && !io->has_audio) {
				fprintf(stderr, "[mf_mux] %s would not take the audio stream; trying the next audio encoder\n",
				        audios[ai].name.c_str());
				ok = false;
				reset_state();
			}
		}
	} else {
		// Audio-only file: there is no video encoder to pick, so the audio list is
		// walked on its own.
		for (size_t ai = 0; ai < audios.size() && !ok; ai++) {
			reset_state();
			if (io->configure(path, kVideoFormatH264, false, &audios[ai].subtype) && io->has_audio) {
				ok = true;
				io->audio_encoder = audios[ai].name;
			}
		}
	}

	if (!ok) {
		delete io;
		throw MediaError("Mp4Muxer::open: cannot create " + path +
		                 (first_error.empty() ? " (no usable encoder)" : " (" + first_error + ")"));
	}

	fprintf(stderr, "[mf_mux] %s -> video %s%s%s\n", path.c_str(),
	        io->video_encoder.empty() ? "(none)" : io->video_encoder.c_str(),
	        io->audio_encoder.empty() ? "" : ", audio ",
	        io->audio_encoder.empty() ? "" : io->audio_encoder.c_str());

	video_encoder_ = io->video_encoder;
	audio_encoder_ = io->audio_encoder;
	impl_ = io;
	open_ = true;
}

const std::string& Mp4Muxer::video_encoder() const { return video_encoder_; }

const std::string& Mp4Muxer::audio_encoder() const { return audio_encoder_; }

void Mp4Muxer::add_video(const unsigned char* rgb, i64 w, i64 h, double seconds) {
	Impl* io = impl_;
	if (!io || !open_) throw MediaError("Mp4Muxer::add_video: the muxer is not open");
	if (!io->has_video) throw MediaError("Mp4Muxer::add_video: this file has no video stream");
	if (!rgb) throw MediaError("Mp4Muxer::add_video: null frame");
	if (w != io->width || h != io->height) {
		char buf[128];
		snprintf(buf, sizeof buf, ": the frame is %lldx%lld, the file is %lldx%lld", (long long)w,
		         (long long)h, (long long)io->width, (long long)io->height);
		throw MediaError(std::string("Mp4Muxer::add_video") + buf);
	}

	LONGLONG ts = seconds >= 0.0
	                  ? (LONGLONG)llround(seconds * (double)kTicksPerSecond)
	                  : (LONGLONG)(io->video_frames * io->frame_ticks);
	// The sink writer wants strictly increasing timestamps.
	if (ts <= io->last_video_ts) ts = io->last_video_ts + io->frame_ticks;

	const DWORD bytes = (DWORD)(io->width * io->height * 4);
	ComPtr<IMFMediaBuffer> buffer;
	ComPtr<IMFSample> sample;
	if (FAILED(mf_create_memory_buffer(bytes, buffer.put()))) throw MediaError("MFCreateMemoryBuffer");
	BYTE* dst = nullptr;
	if (FAILED(buffer->Lock(&dst, nullptr, nullptr))) throw MediaError("IMFMediaBuffer::Lock");
	// Tight RGB -> top-down B,G,R,X, which is MFVideoFormat_RGB32's layout.
	for (i64 y = 0; y < io->height; y++) {
		const unsigned char* src = rgb + (size_t)(y * io->width * 3);
		unsigned char* row = dst + (size_t)(y * io->width * 4);
		for (i64 x = 0; x < io->width; x++) {
			row[x * 4 + 0] = src[x * 3 + 2];
			row[x * 4 + 1] = src[x * 3 + 1];
			row[x * 4 + 2] = src[x * 3 + 0];
			row[x * 4 + 3] = 0xFF;
		}
	}
	buffer->Unlock();
	buffer->SetCurrentLength(bytes);

	if (FAILED(mf_create_sample(sample.put())) || FAILED(sample->AddBuffer(buffer.get())))
		throw MediaError("MFCreateSample");
	sample->SetSampleTime(ts);
	sample->SetSampleDuration(io->frame_ticks);

	HRESULT hr = io->writer->WriteSample(io->video_stream, sample.get());
	if (FAILED(hr)) throw MediaError("Mp4Muxer::add_video: WriteSample failed " + hr_text(hr));
	io->last_video_ts = ts;
	io->video_frames++;
}

void Mp4Muxer::add_audio(const float* pcm, i64 n_samples) {
	Impl* io = impl_;
	if (!io || !open_) throw MediaError("Mp4Muxer::add_audio: the muxer is not open");
	if (!io->has_audio) return;  // the file was opened without an audio stream
	if (!pcm || n_samples <= 0) return;
	if (n_samples % io->channels != 0) n_samples -= n_samples % io->channels;  // whole frames only

	const i64 frames = n_samples / io->channels;
	const DWORD bytes = (DWORD)(n_samples * 2);

	ComPtr<IMFMediaBuffer> buffer;
	ComPtr<IMFSample> sample;
	if (FAILED(mf_create_memory_buffer(bytes, buffer.put()))) throw MediaError("MFCreateMemoryBuffer");
	BYTE* dst = nullptr;
	if (FAILED(buffer->Lock(&dst, nullptr, nullptr))) throw MediaError("IMFMediaBuffer::Lock");
	for (i64 i = 0; i < n_samples; i++) {
		const short s = to_i16(pcm[i]);
		dst[2 * i] = (BYTE)(s & 0xFF);
		dst[2 * i + 1] = (BYTE)((s >> 8) & 0xFF);
	}
	buffer->Unlock();
	buffer->SetCurrentLength(bytes);

	if (FAILED(mf_create_sample(sample.put())) || FAILED(sample->AddBuffer(buffer.get())))
		throw MediaError("MFCreateSample");
	const LONGLONG ts = (LONGLONG)(io->audio_frames * kTicksPerSecond / io->sample_rate);
	const LONGLONG dur = (LONGLONG)(frames * kTicksPerSecond / io->sample_rate);
	sample->SetSampleTime(ts);
	sample->SetSampleDuration(dur);

	HRESULT hr = io->writer->WriteSample(io->audio_stream, sample.get());
	if (FAILED(hr)) throw MediaError("Mp4Muxer::add_audio: WriteSample failed " + hr_text(hr));
	io->audio_frames += frames;
}

void Mp4Muxer::close() {
	if (!impl_) return;
	Impl* io = impl_;
	impl_ = nullptr;
	open_ = false;

	if (io->writer && !io->finalized) {
		// Finalize flushes the encoders and writes the container index.
		io->finalized = true;
		HRESULT hr = io->writer->Finalize();
		if (FAILED(hr)) fprintf(stderr, "[mf_mux] warning: Finalize failed %s\n", hr_text(hr).c_str());
	}
	io->writer.reset();
	delete io;
}

}  // namespace phi::media
