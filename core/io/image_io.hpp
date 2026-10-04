// 8-bit image I/O and the resampling primitives the image path needs.
//
// Still images go through WIC (Windows Imaging Component): the OS ships the PNG
// and JPEG codecs, so the engine keeps its "system components only" rule and
// carries no libpng / stb_image / PIL. MediaError comes in through
// media_common.hpp because it is part of this module's contract (crop_centre and
// every WIC failure throw it), not just an implementation detail.
#pragma once
#include <string>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

// ── PNG / JPEG via WIC ────────────────────────────────────────────────────
// 8-bit sRGB. `rgb` is tightly packed HWC, 3 bytes/pixel, row major.
void write_png(const std::string& path, const std::vector<unsigned char>& rgb, int w, int h);
// Reads 8-bit PNG or JPEG and converts to tightly packed 8-bit sRGB HWC.
std::vector<unsigned char> read_image(const std::string& path, int* out_w, int* out_h);

// ── in-memory codec crossings (the agent's image path) ────────────────────
// A picture is not always a file on disk: an attachment arrives as bytes and a
// re-encode wants to read those bytes back, so the two codec entries above have
// memory twins. Both go through the same WIC decoders/encoders.

// Decode anything WIC ships a decoder for (PNG, JPEG, BMP, GIF, TIFF, ...) from
// memory into tightly packed 8-bit sRGB HWC. Throws MediaError on a buffer no
// decoder accepts, which is how callers tell "this is not a picture I can touch"
// from "this is a picture I must not touch".
std::vector<unsigned char> decode_image_memory(const unsigned char* data, size_t size,
                                               int* out_w, int* out_h);

// Baseline JPEG in memory, `quality` in 1..100 (clamped). JPEG has no alpha
// channel, so the caller hands in 3-channel sRGB — which is what
// decode_image_memory (and every other producer here) already gives.
std::vector<unsigned char> encode_jpeg_memory(const unsigned char* rgb, int w, int h, int quality);

// ── canvas + centre crop ──────────────────────────────────────────────────
// The canvas is ceil(v/32)*32 for each axis; the result is centre-cropped back
// to exactly (dst_w, dst_h). When the difference is odd the extra pixel is
// removed from the RIGHT / BOTTOM side (left/top get floor(delta/2)).
void crop_centre(const unsigned char* src, int src_w, int src_h, int dst_w, int dst_h,
                 std::vector<unsigned char>& out);

// ── separable Lanczos-3 resize ────────────────────────────────────────────
// Matches ComfyUI's comfy.utils.common_upscale(..., "lanczos", "center"):
// antialias-filtered, support scaled by 1/ratio when downscaling, edge pixels
// clamped, normalised weights, working in float and clamped to [0,255] at the end.
void resize_lanczos3(const unsigned char* src, int src_w, int src_h, int dst_w, int dst_h,
                     std::vector<unsigned char>& out);

// float HWC -> 8-bit. The two VAEs in this engine disagree about their output
// range and the reference is explicit about each, so the conversion is two named
// entry points rather than one function with a flag:
//   * `float_to_srgb8`   maps [-1,1] -> [0,255]; the *image* path's Flux AE
//     decodes an image centred on zero, which is what ComfyUI's image VAE does.
//   * `float01_to_srgb8` maps [ 0,1] -> [0,255]; the *video* path's H3 video VAE
//     already applies `pixel * std + mean` and clamps to [0,1] (vae.py), which is
//     also what the reference converts to bytes (`(image * 255)`).
// Feeding an H3 decode through the [-1,1] entry point compresses the whole range
// into [128,255]: a washed-out grey picture with no real black. Both round
// half-away-from-zero after the mapping.
void float_to_srgb8(const float* src, int w, int h, int channels,
                    std::vector<unsigned char>& out);
void float01_to_srgb8(const float* src, int w, int h, int channels,
                      std::vector<unsigned char>& out);

// Planar [3, T, H, W] -> the packed HWC bytes of frame `f`.
// The video VAE hands back one planar tensor for the whole clip, while every
// consumer of a *frame* (the crop, the encoder) wants interleaved RGB; getting
// this wrong is not subtle (the three planes land in three horizontal bands),
// so the per-frame extraction is a named helper with a test rather than three
// lines inside the mux loop.
void video_frame_to_hwc(const float* planes, int channels, int frames, int f, int h, int w,
                        std::vector<float>& out);

// Planar [C,H,W] fp32 -> tightly packed interleaved [H,W,C] fp32.
//
// The GPU kernels work in NCHW because that is what a convolution wants, but
// everything that talks to a *file* (WIC, the encoder) wants HWC. Getting this
// conversion wrong is not subtle in a good way: a HWC consumer handed a CHW
// buffer reads the three channels at 3x stride in each axis, which for a smooth
// image means R=G=B (a grey picture) wrapped into a 3x3 tile. That is precisely
// what shipped before this helper existed, so it is a named, tested function
// rather than an inline loop at each call site.
void chw_to_hwc(const float* src, int channels, int h, int w, std::vector<float>& out);
// The inverse, for symmetry and for the encoders.
void hwc_to_chw(const float* src, int h, int w, int channels, std::vector<float>& out);

}  // namespace phi::media
