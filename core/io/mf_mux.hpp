// MP4 muxing: IMFSinkWriter wrapper for one video + one audio stream.
//
// The encoders are not fixed: open() searches the machine for the ones it has
// and drives them in a priority order — hardware H.264 (NVENC / Quick Sync /
// AMF) first, then the software H.264 encoder, then Motion JPEG; audio is AAC,
// then FLAC, then MP3 — see mf_mux.cpp for why.  The names it settled on are
// readable through video_encoder()/audio_encoder() and are the ones a caller
// should report, not the ones it hoped for.
//
// A frozen interface: the video_generate tool is
// the only consumers. Frames are submitted in presentation order; the muxer owns
// the encoder and the interleaving, so the caller only needs frame timing.
#pragma once

#include <string>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

class Mp4Muxer {
public:
	Mp4Muxer() = default;
	~Mp4Muxer();
	Mp4Muxer(const Mp4Muxer&) = delete;
	Mp4Muxer& operator=(const Mp4Muxer&) = delete;

	// `video` and `audio` may each be disabled by passing 0 for their size/rate.
	// At least one must be enabled. Throws MediaError on any MF failure.
	void open(const std::string& path, i64 width, i64 height, i64 fps, i64 sample_rate,
	          i64 channels);

	// One RGB frame (tightly packed, w*h*3). `seconds` is the presentation time;
	// pass a negative value to let the muxer derive it from the frame index/fps.
	void add_video(const unsigned char* rgb, i64 w, i64 h, double seconds);

	// Interleaved float PCM in [-1, 1] at the rate/channels given to open().
	// `n_samples` counts *values* (frames x channels), not frames: the muxer
	// divides by the channel count itself, and a caller that hands it frame counts
	// writes half a buffer and half a duration (see video_gen's drain loop).
	void add_audio(const float* pcm, i64 n_samples);

	// Flushes the encoders and finalises the file. Idempotent.
	void close();

	// The encoders this file is actually being written with, as found on this
	// machine while open() ran ("" when the stream is absent). These are MFT
	// friendly names, e.g. "NVIDIA H.264 Encoder MFT".
	const std::string& video_encoder() const;
	const std::string& audio_encoder() const;

private:
	struct Impl;
	Impl* impl_ = nullptr;
	bool open_ = false;
	// Kept on the object rather than in Impl: the encoders a *finished* file was
	// written with stay readable after close(), which is when a caller reports
	// them.
	std::string video_encoder_;
	std::string audio_encoder_;
};

}  // namespace phi::media
