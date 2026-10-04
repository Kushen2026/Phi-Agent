#pragma once

// Shared plumbing for every media node: the typed payload accessors, the graph
// state bag, and the small helpers a node needs (path / model resolution, the
// tokenizer cache, the two normal-fills, the reference-block builders). Each
// node lives in its own translation unit and registers through a function
// declared at the bottom; registry.cpp calls them in catalogue order.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "graph/graph.hpp"
#include "graph/graph_media.hpp"
#include "io/audio_io.hpp"
#include "io/image_io.hpp"
#include "io/mf_mux.hpp"
#include "io/video_io.hpp"
#include "util/media_common.hpp"
#include "models/lora.hpp"
#include "models/media_models.hpp"
#include "models/model_common.hpp"
#include "models/media_geometry.hpp"
#include "runtime/vram_budget.hpp"
#include "sampling/h3_joint_av.hpp"
#include "sampling/sampling.hpp"
#include "sampling/sampling_h3.hpp"
#include "text/tokenizer.hpp"
#include "text/vision_prep.hpp"
#include "util/base.hpp"

namespace phi::media {

// ── payload accessors (graph_media.hpp) ────────────────────────────────────
std::shared_ptr<GraphModelCache> graph_state(GraphContext& ctx);
void release_media_cache(GraphModelCache& cache);
GpuCtx* graph_gpu(GraphContext& ctx);

// ── small helpers ──────────────────────────────────────────────────────────
Port port(const char* name, SocketType t, bool optional = false, const char* doc = "");

// Hands the shared streaming / activation arenas back to the device: the phase
// boundary every streaming node draws (text encode, a sampler, a VAE decode).
void release_shared_arenas(GpuCtx* g);

bool looks_absolute(const std::string& p);
// A caller-supplied path: absolute is taken as-is, relative is joined with the
// session cwd (for reference images / audio / video).
std::string resolve_user_path(GraphContext& ctx, const std::string& p);
// A model file: an absolute path, a path relative to the models directory, or a
// bare filename matched anywhere under it. Empty => the shipped default for the
// role.
std::string resolve_model(GraphContext& ctx, const std::string& name, const char* role);
// The three files a text encoder reads, as one struct: the token -> id table, the
// BPE merge ranks and the special-token ids. Each is an independent selection
// (three roles per chain), and each falls back - when the node named nothing - to
// the chain's own role, so a hand-wired graph gets the same three files the tool
// would. `arch` picks the chain: "qwen3vl_8b" (image), "qwen3vl_32b" (video),
// "ace15" (music).
TokenizerFiles tokenizer_files_of(GraphContext& ctx, const std::string& arch,
                                  const std::string& override_vocab = std::string(),
                                  const std::string& override_merges = std::string(),
                                  const std::string& override_config = std::string());
// An optional JSON config a loader reads (Breeze-TTS-2's `config.json`, the
// codec's `config.json`, the tokenizer's `tokenizer_config.json`): a selection
// resolves to a file under the models directory; empty stays empty, which means
// "read it beside the weights", exactly as every shipped layout is laid out.
std::string resolve_config_file(GraphContext& ctx, const std::string& name);
std::shared_ptr<Qwen2Tokenizer> load_tokenizer(GraphContext& ctx, const TokenizerFiles& f);

// Box-Muller over a mixed seed: the image chain draws a 32-bit field, the video
// chain a 64-bit one, one spare at a time so an odd size does not shift it.
void fill_normal32(float* dst, size_t n, u64 seed);
void fill_normal64(float* dst, size_t n, u64 seed);

// The seed an initial-noise latent starts from. A seed that was given - on the
// node's `seed` wire, or as its `seed` parameter - is used exactly as it is,
// 0 included: 0 is a seed like any other, and being used verbatim is the whole
// point of it (the same number has to give the same noise back). Only a node
// that carries no seed at all draws a fresh one. Shared by the two latent nodes
// so one rule covers both chains.
u64 latent_seed(const JsonValue& params, const Value& wired);

// ── ref2va reference blocks (shared by the text-encode node) ───────────────
//
// The release-order rules these implement are ComfyUI's
// `MiniMaxH3ReferenceToVideo`: images first (each sized by `ref_image_size`),
// then videos with their soundtracks, then standalone audio. Each reference
// contributes one `H3Ref` (the DiT's packed row block) and, for the visual ones,
// the tokens the text tower will see (`append_ref_video_blocks`, `prep_refs`).
void ref_image_size(i64 w, i64 h, i64 canvas_w, i64 canvas_h, i64* tw, i64* th);
void hwc_bytes_to_planar(const std::vector<unsigned char>& rgb, i64 w, i64 h, i64 t, i64 f,
                         std::vector<float>& out);
// A reference image, decoded and resized per `ref2va_image_size` ("match": the
// generation's pixel area; "max": the reference pipeline's 2048 px short edge).
// Both consumers - the text tower's vision run and the video VAE - are given
// these pixels, so the two can never disagree about the grid.
RefImagePixels load_ref_image(const std::string& path, i64 canvas_w, i64 canvas_h, bool use_max);
// One reference image as an H3Ref block: VAE-encoded onto its own latent grid.
void build_image_ref(ConditioningData& c, const RefImagePixels& px, VideoVae& vvae);
void adapted_ref_video_canvas(i64 vw, i64 vh, i64* cw, i64* ch);
// The frames of a reference clip, at the DiT's 24 fps, resized to the canvas both
// consumers share, snapped to the legal 17k+5 frame count. `target_frames` is the
// generation's own (already aligned) length.
RefVideoFrames load_ref_clip(const std::string& path, i64 target_frames);
// One reference video (+ its soundtrack when `audio_path` is not empty and an
// audio VAE is wired) as one H3Ref block: video rows from the video VAE, audio
// rows from the audio VAE, so the DiT packs them as the reference's
// `video_audio` block. `frames` must be `load_ref_clip`'s output.
void build_video_ref(ConditioningData& c, const RefVideoFrames& frames,
                     const std::string& audio_path, AudioVae* avae, VideoVae& vvae);
void build_audio_ref(ConditioningData& c, const std::string& path, AudioVae& avae);

// ── one registration entry point per node ──────────────────────────────────
void register_model_loader(NodeRegistry& reg);
void register_clip_loader(NodeRegistry& reg);
void register_vae_loader(NodeRegistry& reg);
void register_lora_loader(NodeRegistry& reg);
void register_qwen_image_edit_encode(NodeRegistry& reg);
void register_h3_ref2va_encode(NodeRegistry& reg);
void register_h3_sigma_shift(NodeRegistry& reg);
void register_qwen_image_cache(NodeRegistry& reg);
void register_latent(NodeRegistry& reg);
void register_scheduler(NodeRegistry& reg);
void register_sampler(NodeRegistry& reg);
void register_vae_decode(NodeRegistry& reg);
void register_vae_encode_image(NodeRegistry& reg);
void register_load_image(NodeRegistry& reg);
void register_save_image(NodeRegistry& reg);
void register_save_audio(NodeRegistry& reg);
void register_save_video(NodeRegistry& reg);
void register_seed(NodeRegistry& reg);
void register_const_int(NodeRegistry& reg);

// ── music (ACE-Step 1.5) ───────────────────────────────────────────────
void register_ace_text_encode(NodeRegistry& reg);
void register_ace_model_sampling(NodeRegistry& reg);

// ── tts (Breeze-TTS-2) ────────────────────────────────────────────────
void register_breeze_text_encode(NodeRegistry& reg);
void register_breeze_sampler(NodeRegistry& reg);
void register_breeze_decode(NodeRegistry& reg);

}  // namespace phi::media
