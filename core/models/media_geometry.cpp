#include "models/media_geometry.hpp"

#include <cmath>

namespace phi::media {

FramePlan plan_frame(i64 req_w, i64 req_h, bool video) {
	if (req_w <= 0 || req_h <= 0) throw MediaError("plan_frame: width/height must be positive");
	FramePlan p;
	p.req_w = req_w;
	p.req_h = req_h;

	i64 ow = req_w, oh = req_h;
	if (video) {
		// yuv420 wants both dimensions even; the plan rounds *down* one pixel.
		ow = req_w - (req_w % 2);
		oh = req_h - (req_h % 2);
		// A 1-pixel request still needs a legal frame; clamping to 2 keeps the
		// plan honest (the caller is told the effective size) instead of
		// producing a zero-sized canvas.
		if (ow < 2) ow = 2;
		if (oh < 2) oh = 2;
	}
	p.out_w = ow;
	p.out_h = oh;
	p.adjusted = (ow != req_w || oh != req_h);

	p.canvas_w = round_up(ow, kCanvasAlign);
	p.canvas_h = round_up(oh, kCanvasAlign);

	// Symmetric crop; the leftover pixel when the delta is odd goes to the
	// right / bottom, which is what crop_centre() takes off too.
	const i64 dw = p.canvas_w - p.out_w;
	const i64 dh = p.canvas_h - p.out_h;
	p.crop.left = dw / 2;
	p.crop.right = dw - dw / 2;
	p.crop.top = dh / 2;
	p.crop.bottom = dh - dh / 2;
	return p;
}

// ── H3 frame/latent planning ────────────────────────────────────────────────

bool h3_frame_count_ok(i64 n) { return n >= 5 && (n - 5) % 17 == 0; }

i64 h3_align_frame_count(i64 n) {
	if (n <= 5) return 5;
	// ceil((n - 5) / 17) * 17 + 5
	const i64 k = (n - 5 + 16) / 17;
	return 17 * k + 5;
}

i64 h3_video_latent_t(i64 frames) {
	if (frames <= 5) return 2;
	return ((frames - 5) / 17) * 5 + 2;
}

i64 h3_audio_latent_t(i64 frames, i64 fps) {
	if (fps <= 0) fps = 24;
	if (frames <= 0) return 0;
	return (i64)std::llround((double)frames / (double)fps * 40.0);
}

VideoPlan plan_video(i64 req_w, i64 req_h, i64 duration_frames, i64 fps) {
	if (fps <= 0) fps = 24;
	VideoPlan v;
	v.fps = fps;
	v.frame = plan_frame(req_w, req_h, /*video=*/true);

	v.frames = h3_align_frame_count(duration_frames > 0 ? duration_frames : 5);
	v.video_t = h3_video_latent_t(v.frames);
	v.audio_t = h3_audio_latent_t(v.frames, fps);

	// 16x spatial VAE, then the DiT's (1,2,2) patchify -> 32x in total.
	v.latent_h = v.frame.canvas_h / 16;
	v.latent_w = v.frame.canvas_w / 16;
	v.tokens_h = v.frame.canvas_h / 32;
	v.tokens_w = v.frame.canvas_w / 32;
	v.video_tokens = v.video_t * v.tokens_h * v.tokens_w;
	return v;
}

}  // namespace phi::media
