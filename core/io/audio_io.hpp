// Audio decode/encode: WAV via Media Foundation, resampling via an MF AudioResampler
// MFT (with a deterministic linear fallback when the MFT is unavailable).
//
// A frozen interface: the audio VAE and the muxer consume exactly this, and
// nothing here reaches into either.
#pragma once

#include <string>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

struct AudioClip {
	std::vector<float> samples;   // interleaved if channels > 1
	i64 sample_rate = 0;
	i64 channels = 1;
	double seconds() const {
		return sample_rate > 0 ? (double)(samples.size() / (size_t)channels) / (double)sample_rate
		                       : 0.0;
	}
};

// Decodes any file MF can read (WAV, MP3, AAC, ...) to float32, interleaved.
AudioClip read_audio(const std::string& path);

// Writes 16-bit PCM WAV.
void write_wav(const std::string& path, const AudioClip& clip);

// Resamples `in` (interleaved, `channels`) from `in_sr` to `out_sr`. Uses the MF
// resampler MFT; falls back to a windowed-sinc (Lanczos-3) on the host. The
// fallback is deterministic and bit-reproducible, which the tests rely on.
std::vector<float> resample(const std::vector<float>& in, i64 channels, i64 in_sr, i64 out_sr);

// Downmix/upmix to `channels` (1 = average, 2 = duplicate mono, >2 = zero-fill).
std::vector<float> to_channels(const std::vector<float>& in, i64 in_ch, i64 channels);

}  // namespace phi::media
