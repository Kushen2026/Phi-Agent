// See the header for what this is and why it is a separate pass. This file is the
// device half, and its whole job is to agree with the host decoder bit for bit.
//
// ── the exact arithmetic, and where each part comes from ───────────────────
//
// Decode (SafeTensors::dequant_rows -> quant.cpp):
//   asym_w4a8_int8:  v = codebook[i] * s_rel[g] * s_channel     (dequant_codebook_int4_row)
//   w6a8_int8:       v = (float)i   * s_rel[g] * s_channel      (dequant_packed6_row)
// The operand order and the two multiplies are reproduced exactly; both are
// single-rounded fp32 multiplies (no FMA contraction is possible in a product of
// two factors, and the compiler cannot fuse them into an add), so the fp32 value
// the absmax and the rounding see is the same one the host sees.
//
// Quantise (quantize_impl_into):
//   amax  = max_c |v|                    over the whole row
//   scale = amax > 1e-30 ? amax / 127 : 1
//   q     = clamp(trunc((double)(v / scale) +- 0.5), -127, 127)
// The rounding is done in double on the host, so it is done in double here; the
// ties-away-from-zero form is `d >= 0 ? d + 0.5 : d - 0.5` truncated toward zero,
// i.e. `(long)` on a value already pushed off the tie.
//
// The scale tables are small but per tensor, and they are read once per row per
// group - not once per element, which is what made the host decoder slow (it
// called a dtype-switching accessor twice for every byte).
//
// ── layout ────────────────────────────────────────────────────────────────
//
// v[0] rows        v[1] k            v[2] group        v[3] rel dtype
// v[4] codes off   v[5] srel off     v[6] sch off      v[7] codebook off
// v[8] q off       v[9] s off        v[10] flags        v[11] codes row bytes
// v[12] srel row bytes                v[13] layout
// flags bit0 = has_channel_scale, bit1 = has_codebook
// srv[0] = pool    uav[0] = q (int8 codes)   uav[1] = s (fp32 scales)

#include "kernels/w4a8_expand.hpp"

#include "kernels/kernels.hpp"

namespace phi::media {

static const char* kW4a8ExpandCuda = R"CUDA(
#include <cuda_fp16.h>

struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

typedef unsigned int uint;
typedef unsigned char uchar;

// One element of the relative table, at the dtype it is stored in. Every
// conversion below is exact (fp16, bf16 and fp8 all have fewer significand bits
// than fp32), so this agrees with the host's `scale_at` for every dtype the
// reader can hand over.
__device__ __forceinline__ float rel_one(const uchar* p, uint rdt) {
    if (rdt == 0u) return *(const float*)p;
    if (rdt == 1u) return __half2float(*(const __half*)p);
    if (rdt == 2u) return __uint_as_float(((unsigned)*(const unsigned short*)p) << 16u);
    const unsigned b = (unsigned)*(const uchar*)p;
    if (rdt == 3u) {
        // E4M3: bias 7, 3 mantissa bits; 0x7F/0xFF are NaN and every other
        // all-ones exponent is inf. Subnormals are mantissa * 2^-9.
        const unsigned sign = (b >> 7u) & 1u, e = (b >> 3u) & 15u, m = b & 7u;
        float v;
        if (e == 15u) v = (m == 7u) ? __uint_as_float(0x7fc00000u) : __uint_as_float(0x7f800000u);
        else if (e == 0u) v = (float)m * ldexpf(1.0f, -9);
        else v = (float)(8u + m) * ldexpf(1.0f, (int)e - 10);
        return sign ? -v : v;
    }
    // E5M2: bias 15, 2 mantissa bits. The host's table maps *every* all-ones
    // exponent to inf here (`f8_lut` only builds the NaN case for E4M3), so this
    // does the same rather than "fixing" it - the two have to agree.
    const unsigned sign = (b >> 7u) & 1u, e = (b >> 2u) & 31u, m = b & 3u;
    float v;
    if (e == 31u) v = __uint_as_float(0x7f800000u);
    else if (e == 0u) v = (float)m * ldexpf(1.0f, -16);
    else v = (float)(4u + m) * ldexpf(1.0f, (int)e - 17);
    return sign ? -v : v;
}

__device__ __forceinline__ unsigned rel_bytes(uint rdt) {
    return rdt == 0u ? 4u : (rdt <= 2u ? 2u : 1u);
}

// The host quantiser's rounding, in double, exactly (see the file header).
__device__ __forceinline__ signed char q8(float v, float inv) {
    const double d = (double)(v * inv);
    long r = (long)(d >= 0 ? d + 0.5 : d - 0.5);
    r = r < -127 ? -127 : (r > 127 ? 127 : r);
    return (signed char)r;
}

// The four 6-bit codes of one three-byte group (quant.cpp::unpack_6bit_triple).
__device__ __forceinline__ void six(const uchar* t, unsigned out[4]) {
    out[0] = (unsigned)(t[0] & 0x3F);
    out[1] = (unsigned)(((t[0] >> 6) | (t[1] << 2)) & 0x3F);
    out[2] = (unsigned)(((t[1] >> 4) | (t[2] << 4)) & 0x3F);
    out[3] = (unsigned)(t[2] >> 2);
}

// numthreads(256, 1, 1) - grid = one block per output row
extern "C" __global__ void w4a8_to_i8(Args a) {
    const uint rows = a.v[0];
    const uint k = a.v[1];
    const uint group = a.v[2];
    const uint rdt = a.v[3];
    const uint flags = a.v[10];
    const uint codes_row = a.v[11];
    const uint srel_row = a.v[12];
    const uint layout = a.v[13];
    const uint has_sch = flags & 1u;
    const uint has_cb = (flags >> 1u) & 1u;
    const uint t = threadIdx.x;

    const uint row = blockIdx.x;
    if (row >= rows || k == 0u) return;

    __shared__ float cb[16];
    __shared__ float red[32];
    // The codebook is 16 fp32 entries beside the weight; only the 4-bit family
    // has one, and reading it when the file does not is a read past the staged
    // region. `cb[0] = 0` for the six-bit family keeps the unused path defined.
    if (t < 16u) cb[t] = has_cb ? ((const float*)(a.s[0] + a.v[7]))[t] : 0.0f;
    __syncthreads();

    const uchar* codes = (const uchar*)(a.s[0] + a.v[4]) + (size_t)row * codes_row;
    const uchar* srel = (const uchar*)(a.s[0] + a.v[5]) + (size_t)row * srel_row;
    const float s_ch = has_sch ? *(const float*)(a.s[0] + a.v[6] + (size_t)row * 4u) : 1.0f;
    const uint rsz = rel_bytes(rdt);
    const uint ngroups = k / group;

    // ── pass 1: the row's absmax, group by group ──────────────────────────
    // Threads take groups round-robin, so a group's codes are read by one thread
    // (coalescing is not the point here: the row is at most `k` bytes and pass 2
    // reads it back out of L2).
    float loc = 0.0f;
    for (uint gi = t; gi < ngroups; gi += 256u) {
        const float g = rel_one(srel + (size_t)gi * rsz, rdt);
        const uchar* cg = codes + (size_t)gi * (layout == 0u ? (group >> 1u) : (group * 3u / 4u));
        float m = 0.0f;
        if (layout == 0u) {
            for (uint j = 0; j < group >> 1u; j++) {
                const unsigned b = cg[j];
                const float v0 = (cb[b & 15u] * g) * s_ch;
                const float v1 = (cb[b >> 4u] * g) * s_ch;
                m = fmaxf(m, fmaxf(fabsf(v0), fabsf(v1)));
            }
        } else {
            for (uint j = 0; j < group / 4u; j++) {
                unsigned c[4];
                six(cg + (size_t)j * 3u, c);
                for (uint i = 0; i < 4u; i++) {
                    const float v = ((float)c[i] * g) * s_ch;
                    m = fmaxf(m, fabsf(v));
                }
            }
        }
        loc = fmaxf(loc, m);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) loc = fmaxf(loc, __shfl_xor_sync(0xffffffffu, loc, off));
    const uint nwarp = blockDim.x >> 5u;
    if ((t & 31u) == 0u) red[t >> 5u] = loc;
    __syncthreads();
    if (t == 0u) {
        float m = 0.0f;
        for (uint w = 0; w < nwarp; w++) m = fmaxf(m, red[w]);
        red[0] = m;
    }
    __syncthreads();
    const float amax = red[0];
    const float scale = amax > 1e-30f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
    if (t == 0u) ((float*)(a.u[1] + a.v[9]))[row] = scale;

    // ── pass 2: quantise ──────────────────────────────────────────────────
    signed char* qrow = (signed char*)(a.u[0] + a.v[8]) + (size_t)row * k;
    for (uint gi = t; gi < ngroups; gi += 256u) {
        const float g = rel_one(srel + (size_t)gi * rsz, rdt);
        uchar* qg = (uchar*)qrow + (size_t)gi * group;
        if (layout == 0u) {
            const uchar* cg = codes + (size_t)gi * (group >> 1u);
            for (uint j = 0; j < group >> 1u; j++) {
                const unsigned b = cg[j];
                qg[2u * j] = (uchar)q8((cb[b & 15u] * g) * s_ch, inv);
                qg[2u * j + 1u] = (uchar)q8((cb[b >> 4u] * g) * s_ch, inv);
            }
        } else {
            const uchar* cg = codes + (size_t)gi * (group * 3u / 4u);
            for (uint j = 0; j < group / 4u; j++) {
                unsigned c[4];
                six(cg + (size_t)j * 3u, c);
                for (uint i = 0; i < 4u; i++)
                    qg[j * 4u + i] = (uchar)q8(((float)c[i] * g) * s_ch, inv);
            }
        }
    }
}
)CUDA";

const char* w4a8_expand_hlsl() { return kW4a8ExpandCuda; }

void dispatch_w4a8_expand(ComputeContext& ctx, const W4a8ExpandArgs& a) {
	if (a.rows <= 0 || a.k <= 0) return;
	if (a.group <= 0 || a.k % a.group != 0)
		throw MediaError("w4a8_expand: K=" + std::to_string(a.k) + " is not a multiple of group=" +
		                 std::to_string(a.group));
	const i64 cpg = a.group / (a.layout == CodeLayout::Nibble4 ? 2 : 4);
	if (a.group % (a.layout == CodeLayout::Nibble4 ? 2 : 4) != 0)
		throw MediaError("w4a8_expand: group=" + std::to_string(a.group) +
		                 " is not a whole number of code groups for this layout");
	if (a.codes_row_bytes < cpg * (a.k / a.group))
		throw MediaError("w4a8_expand: a " + std::to_string(a.k) + "-column row needs " +
		                 std::to_string(cpg * (a.k / a.group)) + " packed bytes, got " +
		                 std::to_string(a.codes_row_bytes));
	if (!a.pool || !a.q || !a.s) throw MediaError("w4a8_expand: missing binding");
	if (a.has_codebook && a.layout != CodeLayout::Nibble4)
		throw MediaError("w4a8_expand: only the 4-bit family indexes a codebook");

	GpuKernel* k = ctx.pipeline("w4a8_to_i8", w4a8_expand_hlsl(), "w4a8_to_i8",
	                            ShaderModel::SM6_0);
	KernelParams p{};
	p.values[0] = (u32)a.rows;
	p.values[1] = (u32)a.k;
	p.values[2] = (u32)a.group;
	p.values[3] = (u32)a.rel_dtype;
	p.values[4] = (u32)a.codes_off;
	p.values[5] = (u32)a.srel_off;
	p.values[6] = (u32)a.sch_off;
	p.values[7] = (u32)a.cb_off;
	p.values[8] = (u32)a.q_off;
	p.values[9] = (u32)a.s_off;
	p.values[10] = (a.has_channel_scale ? 1u : 0u) | (a.has_codebook ? 2u : 0u);
	p.values[11] = (u32)a.codes_row_bytes;
	p.values[12] = (u32)a.srel_row_bytes;
	p.values[13] = (u32)a.layout;
	p.srv[0] = a.pool;
	p.uav[0] = a.q;
	p.uav[1] = a.s;
	ctx.dispatch(k, p, (u32)a.rows, 1, 1);
}

}  // namespace phi::media
