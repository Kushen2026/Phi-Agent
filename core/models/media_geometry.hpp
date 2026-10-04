// Frame / canvas geometry shared by image_generate and video_generate.
//
// Both media tools use one size philosophy (plan §5.3, §19, §20):
//
//   request -> (video only: yuv420 even rounding) -> canvas = ceil(v/32)*32
//           -> generate -> symmetric centre crop -> effective output
//
// For still images the effective output is exactly what the caller asked for.
// For video it is the even-rounded request, because the H.264 + AAC muxer needs
// yuv420 (both dimensions even). The canvas is always a multiple of 32: the
// H3 video VAE compresses 16x spatially and the DiT patchifies another 2x, so
// 32 is the coarsest factor the whole chain can agree on; the Qwen-Image VAE
// compresses 16x too and its DiT patchifies nothing, so the same alignment
// covers both.
//
// This header exists so the two tools cannot drift apart: the rule lives in one
// place, is unit-tested against the plan's worked examples, and is what
// `video_generate` reports even when the H3 engine is not built yet.
#pragma once

#include "util/media_common.hpp"

namespace phi::media {

// The canvas alignment every model in this engine runs at.
constexpr i64 kCanvasAlign = 32;

// The reference workflows' step counts. They are also the shipped defaults of the
// settings panel's `采样步数` (tools.media.image_steps / video_steps / music_steps):
// the image chain is Qwen-Image-2.1 (KSampler, 25), the video chain is MiniMax H3
// (BasicScheduler beta, 20) and the music chain is ACE-Step 1.5 (KSampler euler,
// 50 - the step count `ACE-Step-音乐生成.json` ships). They live here, beside the
// geometry both chains share, because the scheduler node and the media tools are
// now the only places that read them (the monolithic pipelines are gone).
inline constexpr i64 kImageSteps = 25;
inline constexpr i64 kVideoSteps = 20;
inline constexpr i64 kMusicSteps = 50;

// The reference limits on the H3 ref2va reference blocks, enforced by the tool
// schema and again in the tool body. These are the released node's Autogrow
// maxima (`MiniMaxH3ReferenceToVideo`: 9 reference images, 3 reference videos,
// 3 standalone audios) - an older build of this chain capped images at 3 and
// audios at 2, which silently refused a request the model supports.
constexpr i64 kMaxRefImages = 9;
constexpr i64 kMaxRefVideos = 3;
constexpr i64 kMaxRefAudios = 3;

// Where the canvas is trimmed to get back to the effective output. When the
// delta is odd the extra pixel comes off the right / bottom side (the plan's
// rule; `crop_centre` implements exactly this).
struct FrameCrop {
	i64 left = 0, top = 0, right = 0, bottom = 0;
	i64 width() const { return left + right; }
	i64 height() const { return top + bottom; }
};

struct FramePlan {
	i64 req_w = 0, req_h = 0;      // what the caller asked for
	i64 out_w = 0, out_h = 0;      // effective output (= req for images)
	i64 canvas_w = 0, canvas_h = 0;
	FrameCrop crop;
	// True when the effective output differs from the request: video only, and
	// only for an odd width and/or height. Callers must report this back.
	bool adjusted = false;
};

// Plans one frame. `video` applies the yuv420 even-size rule first.
FramePlan plan_frame(i64 req_w, i64 req_h, bool video);

// ── MiniMax H3 frame/latent planning (plan §5.2) ────────────────────────────
//
// The H3 temporal VAE compresses time by 17 per (5+17k) chunk with a 5-frame
// priming chunk, so only frame counts of the form 17k+5 are legal. The plan
// snaps the requested count up to the next legal value.

bool h3_frame_count_ok(i64 n);          // n == 17k+5 for some k >= 0
i64 h3_align_frame_count(i64 n);        // smallest legal count >= n (min 5)
i64 h3_video_latent_t(i64 frames);      // 2 if frames <= 5 else ((n-5)/17)*5+2
i64 h3_audio_latent_t(i64 frames, i64 fps = 24);   // round(frames / fps * 40)

// The whole planning result for one video request.
struct VideoPlan {
	FramePlan frame;
	i64 fps = 24;
	i64 frames = 0;        // aligned 17k+5
	i64 video_t = 0;       // latent frames
	i64 audio_t = 0;       // 40 Hz audio latent frames
	i64 latent_h = 0, latent_w = 0;   // video VAE latent (canvas / 16)
	i64 tokens_h = 0, tokens_w = 0;   // DiT tokens per frame (canvas / 32)
	i64 video_tokens = 0;  // video_t * tokens_h * tokens_w
	i64 audio_channels = 32;
	i64 audio_groups = 2;  // the audio latent is [B, 32, 2, T]
};

// Plans one video request: yuv420 + 32-align + crop + H3 frame/latent maths.
VideoPlan plan_video(i64 req_w, i64 req_h, i64 duration_frames, i64 fps = 24);

}  // namespace phi::media
