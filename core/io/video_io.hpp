// Video decode: Media Foundation SourceReader -> tightly packed 8-bit RGB.
//
// A frozen interface: vision_prep and the muxer are the only
// consumers. System components only (MF is part of Windows), no third-party
// codec, matching the engine's rule.
#pragma once

#include <string>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

struct VideoFrame {
	std::vector<unsigned char> rgb;   // w * h * 3, tightly packed, row major
	i64 w = 0, h = 0;
	i64 index = 0;                    // 0-based frame counter
	double seconds = 0.0;             // presentation time
};

class VideoReader {
public:
	VideoReader() = default;
	~VideoReader();
	VideoReader(const VideoReader&) = delete;
	VideoReader& operator=(const VideoReader&) = delete;

	// Opens the file and configures the reader to output RGB32 (converted by
	// MF's video processor). Throws MediaError on failure.
	void open(const std::string& path);
	void close();

	i64 width() const { return w_; }
	i64 height() const { return h_; }
	double fps() const { return fps_; }
	double duration_seconds() const { return duration_; }

	// Reads the next frame in presentation order. Returns false at EOF.
	bool read(VideoFrame& out);

	// Seeks to `seconds` and resumes sequential reads from there.
	void seek(double seconds);

	// Decode at most `max_frames`, sampled uniformly over the whole file (a
	// single frame if max_frames == 1). Used by vision_prep for video refs.
	std::vector<VideoFrame> sample_frames(i64 max_frames);

private:
	struct Impl;
	Impl* impl_ = nullptr;
	i64 w_ = 0, h_ = 0;
	double fps_ = 0.0, duration_ = 0.0;
};

}  // namespace phi::media
