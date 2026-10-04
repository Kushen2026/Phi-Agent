// Qwen3-TTS 12 Hz codec decoder (Mimi-family RVQ + conv decoder), host fp32.
//
// Line-by-line port of vendor/codec_model.py's Qwen3TTSTokenizerV2Decoder (the
// reference pack's own copy of qwen-tts 0.1.1). Every module below names the
// python it mirrors; the evaluation order that matters for fp32 agreement is
// spelled out at the few sites where the reference does something other than
// the obvious thing (the rank-1 `1e-9` guard inside SnakeBeta, the right-edge
// trim of the transposed convs, the layer scales, the exact GELU).
//
//   correspondence
//   SnakeBeta                                 this file (kernels::snake_beta_f32)
//   Qwen3TTSTokenizerV2CausalConvNet          MimiCausalConv + Conv1dRef
//   Qwen3TTSTokenizerV2CausalTransConvNet     MimiCausalTConv + ConvT1dRef
//   ...DecoderResidualUnit / ...DecoderBlock  MimiResidualUnit / MimiDecoderBlock
//   Qwen3TTSTokenizerV2ConvNeXtBlock          MimiConvNeXt
//   ...DecoderTransformerModel                MimiCodec::pre_transformer
//   EuclideanCodebook / VectorQuantization /
//   ResidualVectorQuantization /
//   ResidualVectorQuantizer /
//   SplitResidualVectorQuantizer              MimiCodec::quantize_frames
//   Qwen3TTSTokenizerV2Decoder.forward        MimiCodec::decode_chunk
//   Qwen3TTSTokenizerV2Decoder.chunked_decode MimiCodec::decode
//
// Helper note: `linear_bias` / `layer_norm_rows` / `gelu_tanh` live in an
// anonymous namespace inside core/models/audio_vocoder.cpp, so they cannot be
// reused from here even though this file is their sibling; the ones below are
// the same maths (the ConvNeXt block wants the *erf* GELU, not the tanh one, so
// that one is genuinely different anyway).
#include "models/mimi_codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "kernels/parallel_for.hpp"
#include <cstdio>
#include "kernels/snake.hpp"
#include "util/base.hpp"
#include "util/json.hpp"

namespace phi::media {

using kernels::parallel_for;

namespace {

// y[m, n] = sum_k x[m, k] * w[n, k] (+ b[n]);  x [M, K], w [N, K] row major.
void linear_bias(const float* x, i64 m, i64 k, const float* w, const float* b, float* y, i64 n) {
	parallel_for(m, [&](i64 r) {
		const float* xr = x + r * k;
		float* yr = y + r * n;
		for (i64 o = 0; o < n; o++) {
			const float* wr = w + o * k;
			float acc = b ? b[o] : 0.0f;
			for (i64 i = 0; i < k; i++) acc += wr[i] * xr[i];
			yr[o] = acc;
		}
	});
}

// torch.nn.functional.layer_norm over the last dimension (biased variance).
void layer_norm_rows(const float* x, i64 rows, i64 cols, const float* w, const float* b, float eps,
                     float* y) {
	parallel_for(rows, [&](i64 r) {
		const float* xr = x + r * cols;
		float* yr = y + r * cols;
		float sum = 0.0f;
		for (i64 i = 0; i < cols; i++) sum += xr[i];
		const float mean = sum / (float)cols;
		float sq = 0.0f;
		for (i64 i = 0; i < cols; i++) {
			const float d = xr[i] - mean;
			sq += d * d;
		}
		const float inv = 1.0f / std::sqrt(sq / (float)cols + eps);
		for (i64 i = 0; i < cols; i++) {
			const float v = (xr[i] - mean) * inv;
			yr[i] = v * w[i] + (b ? b[i] : 0.0f);
		}
	});
}

// nn.GELU() - the exact erf form, *not* approximate="tanh" (the ConvNeXt block
// uses the module default).
float gelu_erf(float x) {
	return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

// Qwen3TTSTokenizerV2DecoderRMSNorm: plain RMSNorm (weight, not 1 + weight).
void rms_norm_rows(const float* x, i64 rows, i64 cols, const float* w, float eps, float* y) {
	parallel_for(rows, [&](i64 r) {
		const float* xr = x + r * cols;
		float* yr = y + r * cols;
		float sq = 0.0f;
		for (i64 i = 0; i < cols; i++) sq += xr[i] * xr[i];
		const float inv = 1.0f / std::sqrt(sq / (float)cols + eps);
		for (i64 i = 0; i < cols; i++) yr[i] = xr[i] * inv * w[i];
	});
}

// [C, S] -> [S, C], for the two places where the reference transposes between
// the conv stack (channel major) and the transformer (token major).
void transpose_cs(const float* x, float* y, i64 c, i64 s) {
	parallel_for(s, [&](i64 t) {
		for (i64 ch = 0; ch < c; ch++) y[t * c + ch] = x[ch * s + t];
	});
}

void transpose_sc(const float* x, float* y, i64 s, i64 c) {
	parallel_for(s, [&](i64 t) {
		for (i64 ch = 0; ch < c; ch++) y[ch * s + t] = x[t * c + ch];
	});
}

// A [OC, IC, K] conv whose `pad` is the reference's left-only causal padding
// (CausalConvNet: kernel_size = (k-1)*dilation + 1, padding = kernel_size - stride,
// stride 1 here, so `pad` is the whole of it).
Conv1dRef load_causal_conv(const SafeTensors& st, const std::string& base, i64 dilation) {
	const StTensor& w = st.require(base + ".conv.weight");
	if (w.shape.size() != 3) throw MediaError("mimi: '" + base + ".conv.weight' is not 3-D");
	Conv1dRef c;
	c.oc = w.shape[0];
	c.ic = w.shape[1];
	c.k = w.shape[2];
	c.dilation = dilation;
	c.stride = 1;
	c.pad = (c.k - 1) * dilation;
	c.w = borrow_f32(st, base + ".conv.weight", w.numel);
	c.b = borrow_f32(st, base + ".conv.bias", c.oc);
	return c;
}

// A [IC, OC, K] ConvTranspose1d with an explicit stride (torch's layout).
ConvT1dRef load_tconv(const SafeTensors& st, const std::string& base, i64 stride) {
	const StTensor& w = st.require(base + ".conv.weight");
	if (w.shape.size() != 3) throw MediaError("mimi: '" + base + ".conv.weight' is not 3-D");
	ConvT1dRef c;
	c.ic = w.shape[0];
	c.oc = w.shape[1];
	c.k = w.shape[2];
	c.stride = stride;
	c.pad = 0;
	c.w = borrow_f32(st, base + ".conv.weight", w.numel);
	c.b = borrow_f32(st, base + ".conv.bias", c.oc);
	return c;
}

}  // namespace

// ── module forwards ────────────────────────────────────────────────────────

void MimiSnake::apply(const float* x, float* y, i64 L) const {
	kernels::snake_beta_f32(x, y, channels, L, alpha, beta);
}

void MimiCausalTConv::forward(const float* x, i64 L, float* y) const {
	// ConvTranspose1d(k, stride) then the reference's `hidden_state[..., : -right_pad]`.
	// The trim is expressed as a *shorter output row* rather than a copy of the
	// kept prefix: with `Lout = out_len(L)` the accumulate loop's own bound
	// (`hi = ceil((Lout - kk) / stride)`) drops exactly the columns the reference
	// slices away, row by row, so no compaction pass (and none of the
	// read-after-write it would create between neighbouring rows) is needed.
	c.forward(x, L, y, out_len(L));
}

void MimiResidualUnit::forward(const float* x, i64 L, float* y, float* tmp) const {
	// act1 -> conv1 -> act2 -> conv2 -> + x
	act1.apply(x, tmp, L);
	conv1.forward(tmp, L, y);
	act2.apply(y, tmp, L);
	conv2.forward(tmp, L, y);
	const i64 n = act1.channels * L;
	parallel_for(n, [&](i64 i) { y[i] += x[i]; });
}

const float* MimiDecoderBlock::forward(const float* x, i64 L, float* y, float* a, float* b) const {
	// [SnakeBeta, CausalTransConvNet, 3 x ResidualUnit]
	const i64 outL = L * rate;
	act.apply(x, a, L);
	up.forward(a, L, y);
	const float* cur = y;
	float* dst = b;
	for (const MimiResidualUnit& u : units) {
		u.forward(cur, outL, dst, a);
		cur = dst;
		dst = (dst == b) ? y : b;
	}
	return cur;
}

void MimiConvNeXt::forward(const float* x, i64 L, float* y, float* dw_out, float* tc,
                           float* hc) const {
	// depthwise causal conv (groups = dim): out[c][j] = b[c] + sum_k w[c][k] * x[c][j + k - pad]
	const i64 k = dw.k;
	const i64 pad = dw.pad;
	parallel_for(dim, [&](i64 c) {
		const float* xr = x + c * L;
		float* yr = dw_out + c * L;
		const float* w = dw.w + c * k;   // [dim, 1, k]: one tap set per channel
		const float bv = dw_bias ? dw_bias[c] : 0.0f;
		for (i64 j = 0; j < L; j++) {
			float acc = bv;
			for (i64 kk = 0; kk < k; kk++) {
				const i64 idx = j + kk - pad;
				if (idx >= 0 && idx < L) acc += w[kk] * xr[idx];
			}
			yr[j] = acc;
		}
	});
	// [C, L] -> [L, C]: LayerNorm is over the channels, then the pointwise pair.
	transpose_cs(dw_out, hc, dim, L);
	layer_norm_rows(hc, L, dim, norm_w, norm_b, 1e-6f, hc);
	const i64 c4 = 4 * dim;
	linear_bias(hc, L, dim, pw1_w, pw1_b, tc, c4);
	parallel_for(L * c4, [&](i64 i) { tc[i] = gelu_erf(tc[i]); });
	linear_bias(tc, L, c4, pw2_w, pw2_b, hc, dim);
	// gamma * result, then the residual add (the reference multiplies before it)
	parallel_for(dim, [&](i64 c) {
		const float g = gamma[c];
		const float* hr = hc + c;   // [L, dim], column c
		const float* xr = x + c * L;
		float* yr = y + c * L;
		for (i64 j = 0; j < L; j++) yr[j] = xr[j] + g * hr[j * dim];
	});
}

// ── loading ────────────────────────────────────────────────────────────────

void MimiCodec::open(const std::string& path, GpuCtx* gpu, const std::string& config_json) {
	(void)gpu;   // the decode is host fp32: the codec needs no device buffers
	st_.open(path);

	// The shipped tokenizer file has the decoder at `decoder.*` (that file's root
	// is the tokenizer model); the Breeze checkpoint's own copy nests the same tree
	// under `codec_model.`. Either file is accepted.
	std::string p;
	if (st_.find("decoder.pre_conv.conv.weight")) p = "";
	else if (st_.find("codec_model.decoder.pre_conv.conv.weight")) p = "codec_model.";
	else throw MediaError("mimi: no decoder tree in " + path);

	// ── widths, all read off the checkpoint ──
	const StTensor& pre = st_.require(p + "decoder.pre_conv.conv.weight");
	codebook_dim_ = pre.shape[1];
	latent_ = pre.shape[0];
	const StTensor& head = st_.require(p + "decoder.decoder.0.conv.weight");
	decoder_dim_ = head.shape[0];
	if (head.shape[1] != latent_)
		throw MediaError("mimi: the decoder head's input width is not latent_dim");
	const StTensor& inproj = st_.require(p + "decoder.pre_transformer.input_proj.weight");
	hidden_ = inproj.shape[0];
	if (inproj.shape[1] != latent_)
		throw MediaError("mimi: pre_transformer input width does not match latent_dim");
	emb_dim_ = codebook_dim_ / 2;   // SplitResidualVectorQuantizer: dimension = codebook_dim // 2

	const StTensor& rf =
	    st_.require(p + "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum");
	codebook_size_ = rf.shape[0];
	if (rf.shape[1] != emb_dim_) throw MediaError("mimi: a codebook entry is not emb_dim wide");
	i64 n_rest = 0;
	while (st_.find(p + "decoder.quantizer.rvq_rest.vq.layers." + std::to_string(n_rest) +
	                "._codebook.embedding_sum"))
		n_rest++;
	if (n_rest <= 0) throw MediaError("mimi: the acoustic RVQ has no codebooks");
	codebooks_ = 1 + n_rest;
	outp_first_ = borrow_f32(st_, p + "decoder.quantizer.rvq_first.output_proj.weight",
	                         codebook_dim_ * emb_dim_);
	outp_rest_ = borrow_f32(st_, p + "decoder.quantizer.rvq_rest.output_proj.weight",
	                        codebook_dim_ * emb_dim_);

	// The codebooks are derived: embedding = embedding_sum / clamp(usage, 1e-5).
	emb_first_.assign((size_t)(codebook_size_ * emb_dim_), 0.0f);
	emb_rest_.assign((size_t)(n_rest * codebook_size_ * emb_dim_), 0.0f);
	// `data_f32` is a *borrowed* view of the checkpoint's own (fp32) values, so
	// deriving the 16 codebooks costs one pass and no copy of the file.
	auto fill_book = [&](const std::string& base, float* dst) {
		const StTensor& st_sum = st_.require(base + ".embedding_sum");
		const StTensor& st_use = st_.require(base + ".cluster_usage");
		if (st_use.numel != codebook_size_)
			throw MediaError("mimi: cluster_usage length mismatch at " + base);
		if (st_sum.numel != codebook_size_ * emb_dim_)
			throw MediaError("mimi: embedding_sum shape mismatch at " + base);
		const float* sum = st_.data_f32(st_sum);
		const float* use = st_.data_f32(st_use);
		for (i64 c = 0; c < codebook_size_; c++) {
			const float inv = 1.0f / std::max(use[(size_t)c], 1e-5f);
			for (i64 i = 0; i < emb_dim_; i++)
				dst[c * emb_dim_ + i] = sum[(size_t)(c * emb_dim_ + i)] * inv;
		}
	};
	fill_book(p + "decoder.quantizer.rvq_first.vq.layers.0._codebook", emb_first_.data());
	for (i64 l = 0; l < n_rest; l++)
		fill_book(p + "decoder.quantizer.rvq_rest.vq.layers." + std::to_string(l) + "._codebook",
		          emb_rest_.data() + (size_t)(l * codebook_size_ * emb_dim_));

	// ── the bottleneck transformer ──
	num_layers_ =
	    [&] {
		    i64 n = 0;
		    while (st_.find(p + "decoder.pre_transformer.layers." + std::to_string(n) +
		                    ".input_layernorm.weight"))
			    n++;
		    return n;
	    }();
	if (num_layers_ <= 0) throw MediaError("mimi: the bottleneck transformer has no layers");
	const std::string l0 = p + "decoder.pre_transformer.layers.0.";
	const StTensor& qw = st_.require(l0 + "self_attn.q_proj.weight");
	if (qw.shape[1] != hidden_) throw MediaError("mimi: attention input width mismatch");
	// head_dim cannot be read off a shape (16 heads of 64 and 8 of 128 look the
	// same), so it comes from the sibling config.json - the one the reference
	// builds the codec from - and the shapes are then checked against it.
	heads_ = 16;
	head_dim_ = 64;
	sliding_window_ = 72;
	rms_eps_ = 1e-5f;
	rope_theta_ = 10000.0f;
	{
		const size_t slash = path.find_last_of("/\\");
		// The tokenizer's own config.json, unless the caller selected one.
		const std::string cfg =
		    !config_json.empty()
		        ? config_json
		        : (slash == std::string::npos ? std::string()
		                                      : path.substr(0, slash + 1)) + "config.json";
		std::vector<u8> raw;
		if (read_file_bytes(cfg, raw)) {
			std::string err;
			auto j = json_parse(std::string((const char*)raw.data(), raw.size()), &err);
			const JsonValue* dc = j ? j->find("decoder_config") : nullptr;
			if (dc && dc->is_object()) {
				if (const JsonValue* v = dc->find("num_attention_heads")) heads_ = v->as_int(heads_);
				if (const JsonValue* v = dc->find("head_dim")) head_dim_ = v->as_int(head_dim_);
				if (const JsonValue* v = dc->find("sliding_window"))
					sliding_window_ = v->as_int(sliding_window_);
				if (const JsonValue* v = dc->find("rms_norm_eps"))
					rms_eps_ = (float)v->as_number(rms_eps_);
				if (const JsonValue* v = dc->find("rope_theta"))
					rope_theta_ = (float)v->as_number(rope_theta_);
			}
		}
	}
	const i64 q_dim = heads_ * head_dim_;
	if (qw.shape[0] != q_dim || st_.require(l0 + "self_attn.k_proj.weight").shape[0] != q_dim)
		throw MediaError("mimi: attention width is not num_attention_heads * head_dim");

	in_proj_w_ = borrow_f32(st_, p + "decoder.pre_transformer.input_proj.weight", hidden_ * latent_);
	in_proj_b_ = borrow_f32(st_, p + "decoder.pre_transformer.input_proj.bias", hidden_);
	out_proj_w_ = borrow_f32(st_, p + "decoder.pre_transformer.output_proj.weight", latent_ * hidden_);
	out_proj_b_ = borrow_f32(st_, p + "decoder.pre_transformer.output_proj.bias", latent_);
	final_norm_ = borrow_f32(st_, p + "decoder.pre_transformer.norm.weight", hidden_);

	layers_.assign((size_t)num_layers_, MimiPreLayer{});
	for (i64 l = 0; l < num_layers_; l++) {
		const std::string b = p + "decoder.pre_transformer.layers." + std::to_string(l) + ".";
		MimiPreLayer& L = layers_[(size_t)l];
		auto ln = [&](const std::string& n) -> const float* {
			const StTensor& t = st_.require(b + n);
			if (t.numel != hidden_) throw MediaError("mimi: '" + b + n + "' is not hidden-wide");
			return st_.data_f32(t);
		};
		L.in_ln = ln("input_layernorm.weight");
		L.post_ln = ln("post_attention_layernorm.weight");
		L.q_w = borrow_f32(st_, b + "self_attn.q_proj.weight", q_dim * hidden_);
		L.k_w = borrow_f32(st_, b + "self_attn.k_proj.weight", q_dim * hidden_);
		L.v_w = borrow_f32(st_, b + "self_attn.v_proj.weight", q_dim * hidden_);
		L.o_w = borrow_f32(st_, b + "self_attn.o_proj.weight", hidden_ * q_dim);
		const StTensor& g = st_.require(b + "mlp.gate_proj.weight");
		if (g.shape[1] != hidden_) throw MediaError("mimi: mlp input width mismatch");
		inter_ = g.shape[0];
		L.gate_w = borrow_f32(st_, b + "mlp.gate_proj.weight", inter_ * hidden_);
		L.up_w = borrow_f32(st_, b + "mlp.up_proj.weight", inter_ * hidden_);
		L.down_w = borrow_f32(st_, b + "mlp.down_proj.weight", hidden_ * inter_);
		L.attn_scale = ln("self_attn_layer_scale.scale");
		L.mlp_scale = ln("mlp_layer_scale.scale");
	}

	// ── conv stack ──
	pre_conv_.c = load_causal_conv(st_, p + "decoder.pre_conv", 1);

	while (st_.find(p + "decoder.upsample." + std::to_string(n_up_) + ".0.conv.weight")) n_up_++;
	if (n_up_ <= 0) throw MediaError("mimi: no upsample stage");
	if (n_up_ > 2) throw MediaError("mimi: more than two upsample stages (the stride product is fixed)");
	for (i64 i = 0; i < n_up_; i++) {
		const std::string b = p + "decoder.upsample." + std::to_string(i) + ".";
		const StTensor& tw = st_.require(b + "0.conv.weight");
		const i64 factor = tw.shape[2];   // ConvTranspose1d(k=factor, stride=factor)
		up_t_[i] = load_tconv(st_, b + "0", factor);
		if (up_t_[i].oc != latent_ || up_t_[i].ic != latent_)
			throw MediaError("mimi: an upsample stage is not latent_dim -> latent_dim");
		MimiConvNeXt& cn = convnext_[i];
		cn.dim = latent_;
		cn.dw = load_causal_conv(st_, b + "1.dwconv", 1);
		if (cn.dw.ic != 1 || cn.dw.oc != latent_)
			throw MediaError("mimi: the ConvNeXt depthwise conv is not per-channel");
		cn.dw_bias = cn.dw.b;
		cn.gamma = borrow_f32(st_, b + "1.gamma", latent_);
		cn.norm_w = borrow_f32(st_, b + "1.norm.weight", latent_);
		cn.norm_b = borrow_f32(st_, b + "1.norm.bias", latent_);
		const StTensor& pw1 = st_.require(b + "1.pwconv1.weight");
		const i64 four = pw1.shape[0];
		if (pw1.shape[1] != latent_ || four != 4 * latent_)
			throw MediaError("mimi: ConvNeXt is not dim -> 4*dim -> dim");
		cn.pw1_w = borrow_f32(st_, b + "1.pwconv1.weight", four * latent_);
		cn.pw1_b = borrow_f32(st_, b + "1.pwconv1.bias", four);
		cn.pw2_w = borrow_f32(st_, b + "1.pwconv2.weight", latent_ * four);
		cn.pw2_b = borrow_f32(st_, b + "1.pwconv2.bias", latent_);
	}

	head_conv_.c = load_causal_conv(st_, p + "decoder.decoder.0", 1);
	if (head_conv_.c.ic != latent_ || head_conv_.c.oc != decoder_dim_)
		throw MediaError("mimi: decoder.0 is not latent_dim -> decoder_dim");

	// The DecoderBlocks: `in_dim = decoder_dim >> i`, `out_dim = decoder_dim >> (i+1)`,
	// `rate = upsample_rates[i]` (the transposed conv's kernel is 2 * rate).
	const i64 dilations[3] = {1, 3, 9};
	for (i64 l = 1;; l++) {
		const std::string b = p + "decoder.decoder." + std::to_string(l) + ".";
		if (!st_.find(b + "block.0.alpha")) break;
		MimiDecoderBlock blk;
		blk.in_ch = decoder_dim_ >> (l - 1);
		blk.out_ch = decoder_dim_ >> l;
		if (blk.in_ch <= 0 || blk.out_ch <= 0) throw MediaError("mimi: too many decoder blocks");
		const StTensor& tw = st_.require(b + "block.1.conv.weight");
		const i64 k = tw.shape[2];
		blk.rate = k / 2;
		if (blk.rate <= 0 || 2 * blk.rate != k)
			throw MediaError("mimi: decoder block " + std::to_string(l) + " has an odd kernel");
		if (tw.shape[0] != blk.in_ch || tw.shape[1] != blk.out_ch)
			throw MediaError("mimi: decoder block " + std::to_string(l) + " channel mismatch");
		blk.up.c = load_tconv(st_, b + "block.1", blk.rate);
		blk.up.right_pad = blk.up.c.k - blk.up.c.stride;
		{
			const StTensor& a0 = st_.require(b + "block.0.alpha");
			blk.act.channels = a0.shape[0];
			if (blk.act.channels != blk.in_ch)
				throw MediaError("mimi: a decoder block's SnakeBeta width mismatches");
			blk.act.alpha = st_.data_f32(a0);
			blk.act.beta = st_.data_f32(st_.require(b + "block.0.beta"));
		}
		for (i64 r = 0; r < 3; r++) {
			MimiResidualUnit u;
			const std::string ub = b + "block." + std::to_string(r + 2) + ".";
			auto snake = [&](const std::string& n, MimiSnake& s) {
				const StTensor& a0 = st_.require(ub + n + ".alpha");
				s.channels = a0.shape[0];
				if (s.channels != blk.out_ch)
					throw MediaError("mimi: a residual unit's SnakeBeta width mismatches");
				s.alpha = st_.data_f32(a0);
				s.beta = st_.data_f32(st_.require(ub + n + ".beta"));
			};
			snake("act1", u.act1);
			snake("act2", u.act2);
			u.conv1.c = load_causal_conv(st_, ub + "conv1", dilations[r]);
			if (u.conv1.c.ic != blk.out_ch || u.conv1.c.oc != blk.out_ch || u.conv1.c.k != 7)
				throw MediaError("mimi: a residual unit's conv1 is not a 7-tap square conv");
			u.conv2.c = load_causal_conv(st_, ub + "conv2", 1);
			if (u.conv2.c.k != 1) throw MediaError("mimi: a residual unit's conv2 is not 1-tap");
			blk.units.push_back(u);
		}
		blocks_.push_back(blk);
		up_rates_.push_back(blk.rate);
	}
	if (blocks_.empty()) throw MediaError("mimi: no decoder blocks");

	{
		const StTensor& a0 = st_.require(p + "decoder.decoder.5.alpha");
		final_snake_.channels = a0.shape[0];
		final_snake_.alpha = st_.data_f32(a0);
		final_snake_.beta = st_.data_f32(st_.require(p + "decoder.decoder.5.beta"));
		if (final_snake_.channels != (decoder_dim_ >> (i64)blocks_.size()))
			throw MediaError("mimi: the final SnakeBeta's width is not decoder_dim >> stages");
	}
	final_conv_.c = load_causal_conv(st_, p + "decoder.decoder.6", 1);
	if (final_conv_.c.oc != 1 || final_conv_.c.ic != final_snake_.channels)
		throw MediaError("mimi: the final conv is not output_dim -> 1");

	total_upsample_ = 1;
	for (i64 r : up_rates_) total_upsample_ *= r;
	for (i64 i = 0; i < n_up_; i++) total_upsample_ *= up_t_[i].stride;
	sample_rate_ = 24000;
	loaded_ = true;
}

// ── quantizer ──────────────────────────────────────────────────────────────

void MimiCodec::quantize_frames(const i32* codes, i64 frames, float* out) const {
	const i64 K = codebooks_;
	auto clamp_code = [&](i32 c) {
		if (c < 0) return (i32)0;
		if (c >= codebook_size_) return (i32)codebook_size_ - 1;
		return c;
	};
	parallel_for(frames, [&](i64 t) {
		// The acoustic half: the 15 codebooks' entries summed into one [emb_dim] row.
		float s[512];
		for (i64 i = 0; i < emb_dim_; i++) s[i] = 0.0f;
		for (i64 cb = 1; cb < K; cb++) {
			const i32 code = clamp_code(codes[t * K + cb]);
			const float* e = &emb_rest_[(size_t)(((cb - 1) * codebook_size_ + code) * emb_dim_)];
			for (i64 i = 0; i < emb_dim_; i++) s[i] += e[i];
		}
		const float* e0 = &emb_first_[(size_t)(clamp_code(codes[t * K]) * emb_dim_)];
		// Both halves end in a 1x1 Conv1d (emb_dim -> codebook_dim, no bias) and
		// the reference sums the two results.
		for (i64 o = 0; o < codebook_dim_; o++) {
			const float* wf = outp_first_ + o * emb_dim_;
			const float* wr = outp_rest_ + o * emb_dim_;
			float acc = 0.0f;
			for (i64 i = 0; i < emb_dim_; i++) acc += wf[i] * e0[i] + wr[i] * s[i];
			out[o * frames + t] = acc;
		}
	});
}

// ── bottleneck transformer ─────────────────────────────────────────────────

void MimiCodec::pre_transformer(float* x, i64 frames) const {
	const i64 H = hidden_;
	const i64 qd = heads_ * head_dim_;
	const float scale = 1.0f / std::sqrt((float)head_dim_);
	std::vector<float> h((size_t)(frames * H));
	std::vector<float> hn((size_t)(frames * H));
	std::vector<float> q((size_t)(frames * qd)), k((size_t)(frames * qd)), v((size_t)(frames * qd));
	std::vector<float> attn((size_t)(frames * qd));
	std::vector<float> o((size_t)(frames * H));
	std::vector<float> gate((size_t)(frames * inter_)), up((size_t)(frames * inter_));
	std::vector<float> hh((size_t)(frames * inter_)), down((size_t)(frames * H));

	linear_bias(x, frames, latent_, in_proj_w_, in_proj_b_, h.data(), H);

	const i64 half = head_dim_ / 2;
	std::vector<float> inv((size_t)half);
	for (i64 i = 0; i < half; i++)
		inv[(size_t)i] = 1.0f / std::pow(rope_theta_, (float)(2 * i) / (float)head_dim_);

	for (i64 li = 0; li < num_layers_; li++) {
		const MimiPreLayer& L = layers_[(size_t)li];
		// ── attention ──
		rms_norm_rows(h.data(), frames, H, L.in_ln, rms_eps_, hn.data());
		linear_bias(hn.data(), frames, H, L.q_w, nullptr, q.data(), qd);
		linear_bias(hn.data(), frames, H, L.k_w, nullptr, k.data(), qd);
		linear_bias(hn.data(), frames, H, L.v_w, nullptr, v.data(), qd);
		// RoPE, per head, half-split (GPT-J) pairing, position = the frame index.
		parallel_for(frames, [&](i64 t) {
			for (i64 hd = 0; hd < heads_; hd++) {
				float* qh = q.data() + t * qd + hd * head_dim_;
				float* kh = k.data() + t * qd + hd * head_dim_;
				for (i64 i = 0; i < half; i++) {
					const float ang = (float)t * inv[(size_t)i];
					const float c = std::cos(ang), s = std::sin(ang);
					const float q0 = qh[i], q1 = qh[i + half];
					qh[i] = q0 * c - q1 * s;
					qh[i + half] = q1 * c + q0 * s;
					const float k0 = kh[i], k1 = kh[i + half];
					kh[i] = k0 * c - k1 * s;
					kh[i + half] = k1 * c + k0 * s;
				}
			}
		});
		// Causal sliding-window attention: query t sees keys
		// [max(0, t - sliding_window + 1), t].
		parallel_for(frames, [&](i64 t) {
			const i64 lo = std::max<i64>(0, t - sliding_window_ + 1);
			const i64 nk = t - lo + 1;
			std::vector<float> score((size_t)nk);
			for (i64 hd = 0; hd < heads_; hd++) {
				const float* qh = q.data() + t * qd + hd * head_dim_;
				float mx = -std::numeric_limits<float>::infinity();
				for (i64 kk = 0; kk < nk; kk++) {
					const float* kh = k.data() + (lo + kk) * qd + hd * head_dim_;
					float acc = 0.0f;
					for (i64 d = 0; d < head_dim_; d++) acc += qh[d] * kh[d];
					acc *= scale;
					score[(size_t)kk] = acc;
					mx = std::max(mx, acc);
				}
				float sum = 0.0f;
				for (i64 kk = 0; kk < nk; kk++) {
					const float e = std::exp(score[(size_t)kk] - mx);
					score[(size_t)kk] = e;
					sum += e;
				}
				const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
				float* oh = attn.data() + t * qd + hd * head_dim_;
				for (i64 d = 0; d < head_dim_; d++) oh[d] = 0.0f;
				for (i64 kk = 0; kk < nk; kk++) {
					const float p = score[(size_t)kk] * inv_sum;
					const float* vh = v.data() + (lo + kk) * qd + hd * head_dim_;
					for (i64 d = 0; d < head_dim_; d++) oh[d] += p * vh[d];
				}
			}
		});
		linear_bias(attn.data(), frames, qd, L.o_w, nullptr, o.data(), H);
		for (i64 i = 0; i < frames * H; i++)
			h[(size_t)i] += L.attn_scale[i % H] * o[(size_t)i];
		// ── mlp ──
		rms_norm_rows(h.data(), frames, H, L.post_ln, rms_eps_, hn.data());
		linear_bias(hn.data(), frames, H, L.gate_w, nullptr, gate.data(), inter_);
		linear_bias(hn.data(), frames, H, L.up_w, nullptr, up.data(), inter_);
		for (i64 i = 0; i < frames * inter_; i++) {
			const float g = gate[(size_t)i];
			hh[(size_t)i] = (g / (1.0f + std::exp(-g))) * up[(size_t)i];
		}
		linear_bias(hh.data(), frames, inter_, L.down_w, nullptr, down.data(), H);
		for (i64 i = 0; i < frames * H; i++)
			h[(size_t)i] += L.mlp_scale[i % H] * down[(size_t)i];
	}

	rms_norm_rows(h.data(), frames, H, final_norm_, rms_eps_, hn.data());
	linear_bias(hn.data(), frames, H, out_proj_w_, out_proj_b_, x, latent_);
}

// ── decode ─────────────────────────────────────────────────────────────────

void MimiCodec::decode_chunk(const i32* codes, i64 frames, std::vector<float>& out) const {
	const i64 L0 = frames;
	std::vector<float> z((size_t)(codebook_dim_ * L0));
	quantize_frames(codes, L0, z.data());

	std::vector<float> h((size_t)(latent_ * L0));
	pre_conv_.forward(z.data(), L0, h.data());

	std::vector<float> tok((size_t)(latent_ * L0));
	transpose_cs(h.data(), tok.data(), latent_, L0);
	pre_transformer(tok.data(), L0);
	transpose_sc(tok.data(), h.data(), L0, latent_);

	// ── decoder.upsample ──
	i64 L = L0;
	std::vector<float> cur = std::move(h);
	for (i64 i = 0; i < n_up_; i++) {
		const i64 stride = up_t_[i].stride;
		const i64 newL = L * stride;   // ConvTranspose1d(k = stride, stride) -> exactly stride * L
		std::vector<float> up((size_t)(latent_ * newL));
		up_t_[i].forward(cur.data(), L, up.data(), newL);
		std::vector<float> cn((size_t)(latent_ * newL));
		std::vector<float> dw((size_t)(latent_ * newL));
		std::vector<float> tc((size_t)(newL * 4 * latent_));
		std::vector<float> hc((size_t)(newL * latent_));
		convnext_[i].forward(up.data(), newL, cn.data(), dw.data(), tc.data(), hc.data());
		cur = std::move(cn);
		L = newL;
	}

	// ── decoder.decoder ──
	std::vector<float> a((size_t)(decoder_dim_ * L));
	head_conv_.forward(cur.data(), L, a.data());
	cur.clear();
	cur.shrink_to_fit();

	for (size_t bi = 0; bi < blocks_.size(); bi++) {
		const MimiDecoderBlock& blk = blocks_[bi];
		const i64 outL = L * blk.rate;
		std::vector<float> y((size_t)(blk.out_ch * outL));
		std::vector<float> s1((size_t)std::max<i64>(blk.in_ch * L, blk.out_ch * outL));
		std::vector<float> s2((size_t)(blk.out_ch * outL));
		const float* res = blk.forward(a.data(), L, y.data(), s1.data(), s2.data());
		std::vector<float> next((size_t)(blk.out_ch * outL));
		std::memcpy(next.data(), res, sizeof(float) * next.size());
		a = std::move(next);
		L = outL;
	}

	std::vector<float> fs((size_t)(final_snake_.channels * L));
	final_snake_.apply(a.data(), fs.data(), L);
	a.clear();
	a.shrink_to_fit();
	std::vector<float> pcm((size_t)L);
	final_conv_.forward(fs.data(), L, pcm.data());

	out.resize((size_t)L);
	for (i64 i = 0; i < L; i++) {
		float v = pcm[(size_t)i];
		if (v > 1.0f) v = 1.0f;
		else if (v < -1.0f) v = -1.0f;
		out[(size_t)i] = v;
	}
}

std::vector<float> MimiCodec::decode(const std::vector<i32>& codes, i64 frames) {
	if (!loaded_) throw MediaError("mimi: decode before open");
	if (frames <= 0) return {};
	if ((i64)codes.size() < frames * codebooks_)
		throw MediaError("mimi: codes hold " + std::to_string(codes.size()) + " values, expected " +
		                 std::to_string(frames * codebooks_));
	// Qwen3TTSTokenizerV2Decoder.chunked_decode: 300-frame chunks with 25 frames of
	// left context, the context part of each chunk's waveform dropped again.
	const i64 chunk = 300, context_max = 25;
	std::vector<float> out;
	out.reserve((size_t)(frames * total_upsample_));
	std::vector<float> part;
	i64 start = 0;
	while (start < frames) {
		const i64 end = std::min(start + chunk, frames);
		const i64 ctx = (start - context_max > 0) ? context_max : start;
		const i64 len = end - (start - ctx);
		part.clear();
		decode_chunk(codes.data() + (start - ctx) * codebooks_, len, part);
		const i64 drop = ctx * total_upsample_;
		if ((i64)part.size() > drop)
			out.insert(out.end(), part.begin() + (size_t)drop, part.end());
		start = end;
	}
	// The reference trims to `frames * decode_upsample_rate` samples.
	out.resize((size_t)(frames * total_upsample_), 0.0f);
	return out;
}

}  // namespace phi::media
