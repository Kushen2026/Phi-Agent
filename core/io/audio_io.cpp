// audio_io — audio decode/encode and resampling.
//
// Decoding goes through the Media Foundation SourceReader (WAV, MP3, AAC, ...),
// requested as 32-bit float so the caller gets normalised PCM in [-1, 1].  WAV
// writing is done by hand: a 16-bit RIFF header is 44 deterministic bytes and
// does not depend on a system MFT being registered, which is exactly what the
// conformance tests want.  Resampling prefers the MF resampler MFT
// (CLSID_CResamplerMediaObject) and falls back to a windowed-sinc (Lanczos-3)
// implemented here when the MFT is unavailable or misbehaves.
//
// Linkage note: see video_io.cpp — Media Foundation is resolved at runtime
// (GetProcAddress) because bin/phi.exe's link line has no mfplat/mfreadwrite,
// and the attribute GUIDs are spelled out locally.
#include "io/audio_io.hpp"

#include <windows.h>

#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfreadwrite.h>
#include <mftransform.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace phi::media {
namespace {

constexpr i64 kTicksPerSecond = 10000000;

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
const GUID kMtAudioNumChannels = {0x37E48BF5, 0x645E, 0x4C5B, {0x89, 0xDE, 0xAD, 0xA9, 0xE2, 0x9B, 0x69, 0x6A}};
const GUID kMtAudioSamplesPerSecond = {0x5FAEEAE7, 0x0290, 0x4C31, {0x9E, 0x8A, 0xC5, 0x34, 0xF6, 0x8D, 0x9D, 0xBA}};
const GUID kMtAudioBitsPerSample = {0xF2DEB57F, 0x40FA, 0x4764, {0xAA, 0x33, 0xED, 0x4F, 0x2D, 0x1F, 0xF6, 0x69}};
const GUID kMtAudioBlockAlignment = {0x322DE230, 0x9EEB, 0x43BD, {0xAB, 0x7A, 0xFF, 0x41, 0x22, 0x51, 0x54, 0x1D}};
const GUID kMtAudioAvgBytesPerSecond = {0x1AAB75C8, 0xCFEF, 0x451C, {0xAB, 0x95, 0xAC, 0x03, 0x4B, 0x8E, 0x17, 0x31}};
const GUID kMediaTypeAudio = {0x73647561, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kAudioFormatFloat = {0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
const GUID kClsidResamplerMediaObject = {0xF447B69E, 0x1884, 0x4A7E, {0x80, 0x55, 0x34, 0x6F, 0x74, 0xD6, 0xED, 0xB3}};
const GUID kIidImfTransform = {0xBF94C121, 0x5B05, 0x4E6F, {0x80, 0x00, 0xBA, 0x59, 0x89, 0x61, 0x41, 0x4D}};

constexpr DWORD kFirstAudioStream = (DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM;

// ── little-endian byte helpers (WAV is little-endian on every host) ────────

void put_u16(std::vector<unsigned char>& v, unsigned x) {
	v.push_back((unsigned char)(x & 0xFF));
	v.push_back((unsigned char)((x >> 8) & 0xFF));
}
void put_u32(std::vector<unsigned char>& v, unsigned x) {
	put_u16(v, x & 0xFFFF);
	put_u16(v, (x >> 16) & 0xFFFF);
}
unsigned rd_u16(const unsigned char* p) { return (unsigned)p[0] | ((unsigned)p[1] << 8); }
unsigned rd_u32(const unsigned char* p) {
	return (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

// Paths in this engine are UTF-8 (a reference audio very often lives under a
// user's own, non-ASCII folder), so every file this module opens goes through
// the wide conversion rather than the C runtime's ANSI `fopen`: with the latter
// the WAVE fallback below would silently fail to *find* a file that `open()` the
// MF reader had already refused, and the user would be told "cannot decode" for
// a file that is fine.
std::vector<unsigned char> read_file(const std::string& path) {
	std::vector<unsigned char> v;
	FILE* f = _wfopen(utf8_to_wide(path).c_str(), L"rb");
	if (!f) return v;
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (n > 0) {
		v.resize((size_t)n);
		if (fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
	}
	fclose(f);
	return v;
}

// Deterministic float -> int16 (round to nearest, clamped).
short to_i16(float v) {
	if (v >= 1.0f) return 32767;
	if (v <= -1.0f) return -32768;
	return (short)lrintf(v * 32767.0f);
}

// ── decode ─────────────────────────────────────────────────────────────────

AudioClip read_audio_mf(const std::string& path) {
	mf_startup();
	using CreateReaderFn = HRESULT(WINAPI*)(LPCWSTR, IMFAttributes*, IMFSourceReader**);
	CreateReaderFn create_reader = mf_import<CreateReaderFn>("MFCreateSourceReaderFromURL");
	using CreateTypeFn = HRESULT(WINAPI*)(IMFMediaType**);
	CreateTypeFn create_type = mf_import<CreateTypeFn>("MFCreateMediaType");

	const std::wstring wpath = utf8_to_wide(path);
	if (wpath.empty()) throw MediaError("read_audio: empty path");

	ComPtr<IMFSourceReader> reader;
	mf_check(create_reader(wpath.c_str(), nullptr, reader.put()),
	         ("read_audio: cannot open " + path).c_str());

	ComPtr<IMFMediaType> native;
	if (FAILED(reader->GetNativeMediaType(kFirstAudioStream, 0, native.put())) || !native)
		throw MediaError("read_audio: no audio stream in " + path);

	// Ask for float32; the reader inserts whatever decoder + converter it needs.
	ComPtr<IMFMediaType> out;
	mf_check(create_type(out.put()), "MFCreateMediaType");
	mf_check(out->SetGUID(kMtMajorType, kMediaTypeAudio), "read_audio: SetGUID(major)");
	mf_check(out->SetGUID(kMtSubtype, kAudioFormatFloat), "read_audio: SetGUID(subtype)");
	HRESULT hr = reader->SetCurrentMediaType(kFirstAudioStream, nullptr, out.get());
	if (FAILED(hr)) {
		// Retry carrying the native type's attributes over.
		mf_check(native->SetGUID(kMtSubtype, kAudioFormatFloat), "read_audio: SetGUID(subtype)");
		hr = reader->SetCurrentMediaType(kFirstAudioStream, nullptr, native.get());
	}
	mf_check(hr, "read_audio: cannot decode to float32");

	ComPtr<IMFMediaType> current;
	mf_check(reader->GetCurrentMediaType(kFirstAudioStream, current.put()),
	         "read_audio: GetCurrentMediaType");
	UINT32 channels = 0, rate = 0;
	if (FAILED(current->GetUINT32(kMtAudioNumChannels, &channels)) || channels == 0)
		throw MediaError("read_audio: the decoder reported no channel count");
	if (FAILED(current->GetUINT32(kMtAudioSamplesPerSecond, &rate)) || rate == 0)
		throw MediaError("read_audio: the decoder reported no sample rate");

	AudioClip clip;
	clip.channels = (i64)channels;
	clip.sample_rate = (i64)rate;

	for (;;) {
		DWORD actual = 0, flags = 0;
		LONGLONG ts = 0;
		ComPtr<IMFSample> sample;
		hr = reader->ReadSample(kFirstAudioStream, 0, &actual, &flags, &ts, sample.put());
		mf_check(hr, "read_audio: ReadSample failed");
		if (flags & MF_SOURCE_READERF_ENDOFSTREAM) break;
		if (!sample) continue;

		ComPtr<IMFMediaBuffer> buffer;
		mf_check(sample->ConvertToContiguousBuffer(buffer.put()), "read_audio: contiguous buffer");
		BYTE* data = nullptr;
		DWORD max_len = 0, cur_len = 0;
		mf_check(buffer->Lock(&data, &max_len, &cur_len), "read_audio: Lock");
		const size_t n = (size_t)cur_len / sizeof(float);
		const size_t old = clip.samples.size();
		clip.samples.resize(old + n);
		if (n) memcpy(&clip.samples[old], data, n * sizeof(float));
		buffer->Unlock();
	}
	return clip;
}

// Deterministic RIFF/WAVE reader used when MF cannot open the file (and as the
// reference for what write_wav produced).  Handles PCM16 and IEEE float32.
bool read_wav_riff(const std::string& path, AudioClip& out) {
	const std::vector<unsigned char> f = read_file(path);
	if (f.size() < 44) return false;
	if (memcmp(f.data(), "RIFF", 4) != 0 || memcmp(f.data() + 8, "WAVE", 4) != 0) return false;

	unsigned channels = 0, rate = 0, bits = 0, format = 0;
	const unsigned char* data = nullptr;
	size_t data_bytes = 0;

	size_t pos = 12;
	while (pos + 8 <= f.size()) {
		const unsigned char* id = f.data() + pos;
		const unsigned size = rd_u32(f.data() + pos + 4);
		const size_t body = pos + 8;
		if (body + size > f.size()) break;
		if (memcmp(id, "fmt ", 4) == 0 && size >= 16) {
			format = rd_u16(f.data() + body);
			channels = rd_u16(f.data() + body + 2);
			rate = rd_u32(f.data() + body + 4);
			bits = rd_u16(f.data() + body + 14);
		} else if (memcmp(id, "data", 4) == 0) {
			data = f.data() + body;
			data_bytes = size;
		}
		pos = body + size + (size & 1);  // chunks are word aligned
	}
	if (!data || channels == 0 || rate == 0) return false;

	const size_t frame_bytes = (size_t)channels * (bits / 8);
	if (frame_bytes == 0) return false;
	const size_t frames = data_bytes / frame_bytes;
	out.channels = (i64)channels;
	out.sample_rate = (i64)rate;
	out.samples.resize(frames * channels);

	if (format == 1 && bits == 16) {
		for (size_t i = 0; i < out.samples.size(); i++)
			out.samples[i] = (float)(short)rd_u16(data + i * 2) / 32768.0f;
	} else if (format == 3 && bits == 32) {
		memcpy(out.samples.data(), data, out.samples.size() * sizeof(float));
	} else if (format == 1 && bits == 8) {
		for (size_t i = 0; i < out.samples.size(); i++)
			out.samples[i] = ((float)data[i] - 128.0f) / 128.0f;
	} else {
		return false;
	}
	return true;
}

// ── resampling ─────────────────────────────────────────────────────────────

double sinc(double x) {
	if (std::fabs(x) < 1e-12) return 1.0;
	const double p = x * 3.14159265358979323846;
	return std::sin(p) / p;
}

double lanczos3(double x) {
	const double a = 3.0;
	if (x <= -a || x >= a) return 0.0;
	return sinc(x) * sinc(x / a);
}

// Windowed-sinc resampler.  Deterministic: fixed kernel, fixed summation order,
// no accumulation across calls, so the same input always produces the same
// bytes (the conformance tests compare peaks and lengths).
std::vector<float> resample_host(const std::vector<float>& in, i64 channels, i64 in_sr,
                                 i64 out_sr) {
	const i64 frames_in = (i64)(in.size() / (size_t)channels);
	if (frames_in <= 0) return {};
	const i64 frames_out = std::max<i64>(1, (i64)llround((double)frames_in * (double)out_sr /
	                                                     (double)in_sr));

	const double ratio = (double)out_sr / (double)in_sr;
	// Downsampling lowers the kernel's cut-off so the decimation still filters.
	const double cutoff = std::min(1.0, ratio);
	const double half_width = 3.0 / cutoff;

	std::vector<float> out((size_t)(frames_out * channels), 0.0f);
	for (i64 j = 0; j < frames_out; j++) {
		const double center = ((double)j + 0.5) / ratio - 0.5;  // centre in input samples
		const i64 i0 = (i64)std::ceil(center - half_width);
		const i64 i1 = (i64)std::floor(center + half_width);
		for (i64 c = 0; c < channels; c++) {
			double acc = 0.0, wsum = 0.0;
			for (i64 i = i0; i <= i1; i++) {
				if (i < 0 || i >= frames_in) continue;
				const double w = lanczos3(((double)i - center) * cutoff);
				if (w == 0.0) continue;
				acc += w * (double)in[(size_t)(i * channels + c)];
				wsum += w;
			}
			out[(size_t)(j * channels + c)] = wsum != 0.0 ? (float)(acc / wsum) : 0.0f;
		}
	}
	return out;
}

// MF resampler MFT. Returns true only when the MFT produced a structurally sane
// result; the caller falls back to resample_host() otherwise.
bool resample_mft(const std::vector<float>& in, i64 channels, i64 in_sr, i64 out_sr,
                  std::vector<float>& out) {
	IMFTransform* xf = nullptr;
	if (FAILED(CoCreateInstance(kClsidResamplerMediaObject, nullptr, CLSCTX_INPROC_SERVER,
	                            kIidImfTransform, (void**)&xf)) ||
	    !xf)
		return false;

	using CreateTypeFn = HRESULT(WINAPI*)(IMFMediaType**);
	CreateTypeFn create_type = nullptr;
	void* sym = mf_symbol("MFCreateMediaType");
	if (!sym) {
		xf->Release();
		return false;
	}
	create_type = reinterpret_cast<CreateTypeFn>(sym);

	const UINT32 ch = (UINT32)channels;
	auto make_type = [&](i64 rate) -> IMFMediaType* {
		IMFMediaType* t = nullptr;
		if (FAILED(create_type(&t))) return nullptr;
		t->SetGUID(kMtMajorType, kMediaTypeAudio);
		t->SetGUID(kMtSubtype, kAudioFormatFloat);
		t->SetUINT32(kMtAudioNumChannels, ch);
		t->SetUINT32(kMtAudioSamplesPerSecond, (UINT32)rate);
		t->SetUINT32(kMtAudioBitsPerSample, 32);
		t->SetUINT32(kMtAudioBlockAlignment, ch * 4);
		t->SetUINT32(kMtAudioAvgBytesPerSecond, ch * 4 * (UINT32)rate);
		return t;
	};

	bool ok = false;
	std::vector<float> acc;
	IMFMediaType* in_type = make_type(in_sr);
	IMFMediaType* out_type = make_type(out_sr);
	MFT_OUTPUT_STREAM_INFO stream_info = {};
	const i64 frames_in = (i64)(in.size() / (size_t)channels);
	const i64 chunk_frames = 4096;

	if (in_type && out_type &&
	    SUCCEEDED(xf->SetInputType(0, in_type, 0))) {
		HRESULT hr = xf->SetOutputType(0, out_type, 0);
		if (FAILED(hr)) {
			// Negotiate: the resampler only exposes output types once the input
			// type is set, and it may insist on a particular block alignment.
			for (DWORD i = 0; i < 64; i++) {
				IMFMediaType* avail = nullptr;
				if (FAILED(xf->GetOutputAvailableType(0, i, &avail)) || !avail) break;
				UINT32 rate = 0, acc_ch = 0;
				avail->GetUINT32(kMtAudioSamplesPerSecond, &rate);
				avail->GetUINT32(kMtAudioNumChannels, &acc_ch);
				if (rate == (UINT32)out_sr && acc_ch == ch &&
				    SUCCEEDED(xf->SetOutputType(0, avail, 0))) {
					hr = S_OK;
					avail->Release();
					break;
				}
				avail->Release();
			}
		}
		if (SUCCEEDED(hr) && SUCCEEDED(xf->GetOutputStreamInfo(0, &stream_info))) {
			ok = true;
		}
	}
	if (in_type) in_type->Release();
	if (out_type) out_type->Release();
	if (!ok) {
		xf->Release();
		return false;
	}

	const DWORD out_buf_bytes =
	    (DWORD)std::max<i64>(stream_info.cbSize ? stream_info.cbSize : 0,
	                         (i64)ch * 4 * (chunk_frames * out_sr / std::max<i64>(1, in_sr) + 64));

	// Drain whatever the MFT has ready. Returns false on a hard failure.
	bool hard_failure = false;
	auto drain = [&]() {
		for (int guard = 0; guard < 4096 && !hard_failure; guard++) {
			IMFSample* sample = nullptr;
			IMFMediaBuffer* buffer = nullptr;
			MFT_OUTPUT_DATA_BUFFER db;
			memset(&db, 0, sizeof db);
			db.dwStreamID = 0;
			if (!(stream_info.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES)) {
				if (FAILED(mf_create_memory_buffer(out_buf_bytes, &buffer)) ||
				    FAILED(mf_create_sample(&sample))) {
					hard_failure = true;
					break;
				}
				sample->AddBuffer(buffer);
			}
			db.pSample = sample;
			DWORD status = 0;
			HRESULT hr = xf->ProcessOutput(0, 1, &db, &status);
			if (hr == MF_E_TRANSFORM_NEED_MORE_INPUT) {
				if (sample) sample->Release();
				if (buffer) buffer->Release();
				return;
			}
			if (hr == MF_E_TRANSFORM_STREAM_CHANGE) {
				IMFMediaType* avail = nullptr;
				if (SUCCEEDED(xf->GetOutputAvailableType(0, 0, &avail)) && avail) {
					xf->SetOutputType(0, avail, 0);
					avail->Release();
				}
				if (sample) sample->Release();
				if (buffer) buffer->Release();
				continue;
			}
			if (FAILED(hr)) {
				if (sample) sample->Release();
				if (buffer) buffer->Release();
				hard_failure = true;
				return;
			}
			IMFMediaBuffer* got = nullptr;
			if (db.pSample && SUCCEEDED(db.pSample->ConvertToContiguousBuffer(&got)) && got) {
				BYTE* d = nullptr;
				DWORD cur = 0;
				if (SUCCEEDED(got->Lock(&d, nullptr, &cur))) {
					const size_t n = (size_t)cur / sizeof(float);
					const size_t old = acc.size();
					acc.resize(old + n);
					if (n) memcpy(&acc[old], d, n * sizeof(float));
					got->Unlock();
				}
				got->Release();
			}
			if (db.pEvents) db.pEvents->Release();
			if (db.pSample && db.pSample != sample) db.pSample->Release();
			if (sample) sample->Release();
			if (buffer) buffer->Release();
		}
	};

	for (i64 fed = 0; fed < frames_in && !hard_failure; fed += chunk_frames) {
		const i64 n = std::min<i64>(chunk_frames, frames_in - fed);
		IMFMediaBuffer* buffer = nullptr;
		IMFSample* sample = nullptr;
		if (FAILED(mf_create_memory_buffer((DWORD)(n * ch * 4), &buffer)) ||
		    FAILED(mf_create_sample(&sample))) {
			hard_failure = true;
			break;
		}
		BYTE* d = nullptr;
		if (FAILED(buffer->Lock(&d, nullptr, nullptr))) {
			hard_failure = true;
		} else {
			memcpy(d, &in[(size_t)(fed * channels)], (size_t)(n * channels * 4));
			buffer->Unlock();
			buffer->SetCurrentLength((DWORD)(n * ch * 4));
			sample->AddBuffer(buffer);
			sample->SetSampleTime((LONGLONG)(fed * kTicksPerSecond / in_sr));
			sample->SetSampleDuration((LONGLONG)(n * kTicksPerSecond / in_sr));
			if (FAILED(xf->ProcessInput(0, sample, 0))) hard_failure = true;
		}
		if (sample) sample->Release();
		if (buffer) buffer->Release();
		if (!hard_failure) drain();
	}
	if (!hard_failure) {
		xf->ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0);
		drain();
		// One more pass: the drain message makes the MFT emit its tail.
		drain();
	}
	xf->ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0);
	xf->Release();

	if (hard_failure) return false;
	out.swap(acc);
	return true;
}

}  // namespace

// ── public API ─────────────────────────────────────────────────────────────

AudioClip read_audio(const std::string& path) {
	std::string mf_error;
	try {
		return read_audio_mf(path);
	} catch (const MediaError& e) {
		mf_error = e.what();
	}

	// MF could not decode it: if it is a plain RIFF/WAVE we can read it here,
	// which keeps the WAV path working on a machine without the WAVE MFT.
	AudioClip clip;
	if (read_wav_riff(path, clip)) return clip;
	throw MediaError("read_audio: " + mf_error);
}

void write_wav(const std::string& path, const AudioClip& clip) {
	if (clip.channels <= 0) throw MediaError("write_wav: channel count must be > 0");
	if (clip.sample_rate <= 0) throw MediaError("write_wav: sample rate must be > 0");
	if (clip.samples.size() % (size_t)clip.channels != 0)
		throw MediaError("write_wav: sample count is not a multiple of the channel count");

	const unsigned channels = (unsigned)clip.channels;
	const unsigned rate = (unsigned)clip.sample_rate;
	const unsigned data_bytes = (unsigned)(clip.samples.size() * 2);
	const unsigned block_align = channels * 2;

	std::vector<unsigned char> hdr;
	hdr.reserve(44);
	hdr.insert(hdr.end(), {'R', 'I', 'F', 'F'});
	put_u32(hdr, 36 + data_bytes);
	hdr.insert(hdr.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
	put_u32(hdr, 16);            // PCM fmt chunk size
	put_u16(hdr, 1);             // WAVE_FORMAT_PCM
	put_u16(hdr, channels);
	put_u32(hdr, rate);
	put_u32(hdr, rate * block_align);  // byte rate
	put_u16(hdr, block_align);
	put_u16(hdr, 16);            // bits per sample
	hdr.insert(hdr.end(), {'d', 'a', 't', 'a'});
	put_u32(hdr, data_bytes);

	std::vector<unsigned char> body(44 + (size_t)data_bytes);
	memcpy(body.data(), hdr.data(), 44);
	unsigned char* p = body.data() + 44;
	for (size_t i = 0; i < clip.samples.size(); i++) {
		const short s = to_i16(clip.samples[i]);
		p[2 * i] = (unsigned char)(s & 0xFF);
		p[2 * i + 1] = (unsigned char)((s >> 8) & 0xFF);
	}

	FILE* f = _wfopen(utf8_to_wide(path).c_str(), L"wb");
	if (!f) throw MediaError("write_wav: cannot create " + path);
	const size_t wrote = fwrite(body.data(), 1, body.size(), f);
	fclose(f);
	if (wrote != body.size()) throw MediaError("write_wav: short write to " + path);
}

std::vector<float> resample(const std::vector<float>& in, i64 channels, i64 in_sr, i64 out_sr) {
	if (channels <= 0) throw MediaError("resample: channel count must be > 0");
	if (in_sr <= 0 || out_sr <= 0) throw MediaError("resample: sample rates must be > 0");
	if (in.empty()) return in;
	if (in_sr == out_sr) return in;

	const i64 frames_in = (i64)(in.size() / (size_t)channels);
	const i64 expected = (i64)llround((double)frames_in * (double)out_sr / (double)in_sr);

	std::vector<float> mft;
	if (resample_mft(in, channels, in_sr, out_sr, mft) && !mft.empty()) {
		const i64 frames_out = (i64)(mft.size() / (size_t)channels);
		const i64 tolerance = std::max<i64>(2, expected / 100);  // 1 %, at least 2 samples
		if (mft.size() % (size_t)channels == 0 && frames_out > 0 &&
		    std::llabs(frames_out - expected) <= tolerance) {
			return mft;  // the MFT produced the expected amount of audio
		}
	}
	// Deterministic host fallback (Lanczos-3 windowed sinc).
	return resample_host(in, channels, in_sr, out_sr);
}

std::vector<float> to_channels(const std::vector<float>& in, i64 in_ch, i64 channels) {
	if (in_ch <= 0 || channels <= 0) throw MediaError("to_channels: channel count must be > 0");
	if (in.empty()) return in;
	if (in_ch == channels) return in;

	const i64 frames = (i64)(in.size() / (size_t)in_ch);
	std::vector<float> out((size_t)(frames * channels), 0.0f);
	if (frames == 0) return out;

	for (i64 f = 0; f < frames; f++) {
		const float* src = &in[(size_t)(f * in_ch)];
		float* dst = &out[(size_t)(f * channels)];
		if (channels == 1) {
			double acc = 0.0;  // 1 = average
			for (i64 c = 0; c < in_ch; c++) acc += (double)src[c];
			dst[0] = (float)(acc / (double)in_ch);
		} else if (in_ch == 1) {
			dst[0] = src[0];  // mono -> first channel, 2 = duplicate mono
			if (channels >= 2) dst[1] = src[0];
		} else {
			const i64 copy = std::min<i64>(in_ch, channels);
			for (i64 c = 0; c < copy; c++) dst[c] = src[c];
			// anything past the source channel count stays silent (zero fill)
		}
	}
	return out;
}

}  // namespace phi::media
