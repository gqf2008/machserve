//! HIP kernels for the P1 decode slice, compiled at runtime via hiprtc.
//!
//! Keep them minimal and correct; performance tuning (flash attention,
//! vectorized GEMM epilogues) is a later phase.

use crate::Error;
use mach_engine::hip::hip_arch;
use mach_kernel_sys::hip::{self, Hip, HipKernelModule, HipStream};

/// Embedding gather: `x = emb[*tok]`.
const EMBED_GATHER: &str = r#"
extern "C" __global__ void embed_gather(const int* tok, const float* emb, float* x, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) {
        x[i] = emb[(long long)(*tok) * cols + i];
    }
}
"#;

/// RMSNorm over `rows` rows of `cols` columns, one block per row.
const RMS_NORM: &str = r#"
extern "C" __global__ void rms_norm(const float* x, const float* w, float* y, int cols, float eps) {
    int row = blockIdx.x;
    const float* xr = x + (long long)row * cols;
    float* yr = y + (long long)row * cols;
    __shared__ float red[256];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) yr[i] = xr[i] * inv * w[i];
}
"#;

/// SwiGLU: `out = a * silu(b)`.
const SILU_MUL: &str = r#"
extern "C" __global__ void silu_mul(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = b[i];
        float s = v / (1.0f + __expf(-v));
        out[i] = a[i] * s;
    }
}
"#;

/// Residual add: `x[i] = x[i] + y[i]` (in place on `x`).
const ADD: &str = r#"
extern "C" __global__ void add(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] + y[i];
}
"#;

/// Broadcast-add a bias vector to every row: `x[row][col] += bias[col]`.
const ADD_BIAS: &str = r#"
extern "C" __global__ void add_bias(float* x, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int c = idx % cols;
        x[idx] += bias[c];
    }
}
"#;

/// Store a K/V row into the cache at position `*pos`.
const KV_STORE: &str = r#" 
extern "C" __global__ void kv_store(const float* kv, float* cache, const int* pos_buf,
                                    int kv_heads, int head_dim, int max_seq) {
    int p = *pos_buf;
    int total = kv_heads * head_dim;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        cache[((long long)p * kv_heads * head_dim) + i] = kv[i];
    }
}
"#;

/// Rotary position embeddings applied in place to q and k.
const ROPE: &str = r#"
extern "C" __global__ void rope(float* q, float* k, const int* pos_buf,
                                int n_heads, int n_kv_heads, int head_dim, float theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = *pos_buf;
    int half = head_dim / 2;
    int total_q = n_heads * head_dim;
    int total_k = n_kv_heads * head_dim;
    // GPT-NeoX rotary: pairs (d, d + half), matching HF rotate_half.
    if (i < total_q) {
        int h = i / head_dim;
        int d = i % head_dim;
        if (d < half) {
            float freq = 1.0f / powf(theta, (2.0f * (float)d) / (float)head_dim);
            float ang = (float)pos * freq;
            float c = cosf(ang), sn = sinf(ang);
            float* p = q + h * head_dim;
            float a = p[d], b = p[d + half];
            p[d] = a * c - b * sn;
            p[d + half] = a * sn + b * c;
        }
    }
    if (i < total_k) {
        int h = i / head_dim;
        int d = i % head_dim;
        if (d < half) {
            float freq = 1.0f / powf(theta, (2.0f * (float)d) / (float)head_dim);
            float ang = (float)pos * freq;
            float c = cosf(ang), sn = sinf(ang);
            float* p = k + h * head_dim;
            float a = p[d], b = p[d + half];
            p[d] = a * c - b * sn;
            p[d + half] = a * sn + b * c;
        }
    }
}
"#;

/// Decode attention (GQA) over positions `0..=*pos`, two-pass softmax.
const ATTN_DECODE: &str = r#"
extern "C" __global__ void attn_decode(
    const float* __restrict__ q,
    const float* __restrict__ kc,
    const float* __restrict__ vc,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads, int n_kv_heads, int head_dim, float scale) {

    extern __shared__ float smem[];
    float* scores = smem;          // [max_positions]
    float* red = smem + 1024;      // [blockDim.x]

    int h = blockIdx.x;
    int groups = n_heads / n_kv_heads;
    int kv = h / groups;
    const float* qh = q + h * head_dim;
    int pos = *pos_buf;

    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        const float* kp = kc + ((long long)p * n_kv_heads + kv) * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) s += qh[d] * kp[d];
        scores[p] = s * scale;
    }
    __syncthreads();

    float maxv = -1e30f;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) maxv = fmaxf(maxv, scores[p]);
    red[threadIdx.x] = maxv;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    float m = red[0];

    float sumv = 0.0f;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) sumv += __expf(scores[p] - m);
    red[threadIdx.x] = sumv;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float ssum = red[0];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p <= pos; p++) {
            const float* vp = vc + ((long long)p * n_kv_heads + kv) * head_dim + d;
            acc += __expf(scores[p] - m) * (*vp);
        }
        out[h * head_dim + d] = acc / ssum;
    }
}
"#;

/// Batched embedding: `x[s, :] = emb[tok[s], :]`.
const EMBED_BATCHED: &str = r#"
extern "C" __global__ void embed_batched(const int* tok, const float* emb, float* x,
                                         int cols, int batch) {
    int s = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) x[(long long)s * cols + i] = emb[(long long)tok[s] * cols + i];
}
"#;

/// Batched rotary embeddings with per-sequence positions.
const ROPE_BATCHED: &str = r#"
extern "C" __global__ void rope_batched(float* q, float* k, const int* pos_buf,
                                        int batch, int n_heads, int n_kv_heads,
                                        int head_dim, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total_q = batch * n_heads * head_dim;
    int total_k = batch * n_kv_heads * head_dim;
    if (idx < total_q) {
        int s = idx / (n_heads * head_dim);
        int rem = idx % (n_heads * head_dim);
        int h = rem / head_dim;
        int d = rem % head_dim;
        if (d < half) {
            int pos = pos_buf[s];
            float freq = 1.0f / powf(theta, (2.0f * (float)d) / (float)head_dim);
            float ang = (float)pos * freq;
            float c = cosf(ang), sn = sinf(ang);
            float* p = q + (long long)s * n_heads * head_dim + h * head_dim;
            float a = p[d], b = p[d + half];
            p[d] = a * c - b * sn;
            p[d + half] = a * sn + b * c;
        }
    }
    if (idx < total_k) {
        int s = idx / (n_kv_heads * head_dim);
        int rem = idx % (n_kv_heads * head_dim);
        int h = rem / head_dim;
        int d = rem % head_dim;
        if (d < half) {
            int pos = pos_buf[s];
            float freq = 1.0f / powf(theta, (2.0f * (float)d) / (float)head_dim);
            float ang = (float)pos * freq;
            float c = cosf(ang), sn = sinf(ang);
            float* p = k + (long long)s * n_kv_heads * head_dim + h * head_dim;
            float a = p[d], b = p[d + half];
            p[d] = a * c - b * sn;
            p[d + half] = a * sn + b * c;
        }
    }
}
"#;

/// Batched KV store: each sequence writes its k/v row at its own position.
const KV_STORE_BATCHED: &str = r#"
extern "C" __global__ void kv_store_batched(const float* kv, float* cache,
                                            const int* pos_buf, const int* slots,
                                            int batch, int kv_heads, int head_dim,
                                            int max_seq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_heads * head_dim;
    if (idx < total) {
        int s = idx / (kv_heads * head_dim);
        int i = idx % (kv_heads * head_dim);
        int p = pos_buf[s];
        cache[((long long)slots[s] * max_seq + p) * kv_heads * head_dim + i] =
            kv[(long long)s * kv_heads * head_dim + i];
    }
}
"#;

/// Batched decode attention (GQA), one block per (sequence, head).
const ATTN_DECODE_BATCHED: &str = r#"
extern "C" __global__ void attn_decode_batched(
    const float* __restrict__ q,
    const float* __restrict__ kc,
    const float* __restrict__ vc,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    const int* __restrict__ slots,
    int batch, int n_heads, int n_kv_heads, int head_dim, float scale, int max_seq) {
    extern __shared__ float smem[];
    float* scores = smem;
    float* red = smem + max_seq;

    int s = blockIdx.x / n_heads;
    int h = blockIdx.x % n_heads;
    int groups = n_heads / n_kv_heads;
    int kv = h / groups;
    int slot = slots[s];
    int pos = pos_buf[s];
    const float* qh = q + ((long long)s * n_heads + h) * head_dim;

    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        const float* kp = kc + ((long long)slot * max_seq + p) * n_kv_heads * head_dim + kv * head_dim;
        float sc = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) sc += qh[dd] * kp[dd];
        scores[p] = sc * scale;
    }
    __syncthreads();

    float maxv = -1e30f;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) maxv = fmaxf(maxv, scores[p]);
    red[threadIdx.x] = maxv;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
        __syncthreads();
    }
    float m = red[0];

    float sumv = 0.0f;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) sumv += __expf(scores[p] - m);
    red[threadIdx.x] = sumv;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    float ssum = red[0];

    for (int dd = threadIdx.x; dd < head_dim; dd += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p <= pos; p++) {
            const float* vp = vc + ((long long)slot * max_seq + p) * n_kv_heads * head_dim + kv * head_dim + dd;
            acc += __expf(scores[p] - m) * (*vp);
        }
        out[((long long)s * n_heads + h) * head_dim + dd] = acc / ssum;
    }
}
"#;

/// Per-row argmax over a [batch, vocab] logits matrix.
const ARGMAX_BATCHED: &str = r#"
extern "C" __global__ void argmax_batched(const float* logits, int* out_tok,
                                          int vocab, int batch) {
    int s = blockIdx.x;
    const float* row = logits + (long long)s * vocab;
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    float v = -1e30f;
    int idx = -1;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if (row[i] > v || (row[i] == v && i < idx)) {
            v = row[i];
            idx = i;
        }
    }
    s_val[threadIdx.x] = v;
    s_idx[threadIdx.x] = idx;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            float a = s_val[threadIdx.x + st];
            float b = s_val[threadIdx.x];
            int ai = s_idx[threadIdx.x + st];
            int bi = s_idx[threadIdx.x];
            if (a > b || (a == b && ai < bi)) {
                s_val[threadIdx.x] = a;
                s_idx[threadIdx.x] = ai;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out_tok[s] = s_idx[0];
}
"#;

/// f32 -> fp16 cast (device conversion matches `crate::fp16::f32_to_f16`).
const CAST_F32_F16: &str = r#"
__device__ inline unsigned short f32_to_f16(float x) {
    union { _Float16 h; unsigned short u; } c;
    c.h = (_Float16)x;
    return c.u;
}

extern "C" __global__ void cast_f32_f16(const float* x, unsigned short* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f32_to_f16(x[i]);
}
"#;

/// fp16 -> f32 cast (device conversion matches `crate::fp16::f16_to_f32`).
const CAST_F16_F32: &str = r#"
__device__ inline float f16_to_f32(unsigned short h) {
    union { _Float16 h; unsigned short u; } c;
    c.u = h;
    return (float)c.h;
}

extern "C" __global__ void cast_f16_f32(const unsigned short* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f16_to_f32(x[i]);
}
"#;

/// fp16 KV cache: store f32 K/V rows as fp16, attention reads fp16 K/V
/// (half the cache memory and bandwidth of the f32 path).
const KV_F16: &str = r#"
__device__ inline unsigned short f32_to_f16_bits(float x) {
    _Float16 h = (_Float16)x;
    union { _Float16 h; unsigned short u; } c;
    c.h = h;
    return c.u;
}

__device__ inline float f16_bits_to_f32(unsigned short u) {
    union { _Float16 h; unsigned short u; } c;
    c.u = u;
    return (float)c.h;
}

extern "C" __global__ void kv_store_batched_f16(const float* kv,
                                                unsigned short* cache,
                                                const int* pos_buf, const int* slots,
                                                int batch, int kv_heads, int head_dim,
                                                int max_seq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_heads * head_dim;
    if (idx < total) {
        int s = idx / (kv_heads * head_dim);
        int i = idx % (kv_heads * head_dim);
        int p = pos_buf[s];
        cache[((long long)slots[s] * max_seq + p) * kv_heads * head_dim + i] =
            f32_to_f16_bits(kv[(long long)s * kv_heads * head_dim + i]);
    }
}


"#;

/// GQA-reuse decode attention (f16 KV), fused tiled two-phase.
///
/// One block per (sequence, KV head). Phase A (tid = position lane) computes
/// the FULL head_dim dot `q[g] . k[p]` for every group head, reading each
/// K[p] row once and reusing it across the group; scores go to shared. Phase B
/// (tid = (dim, lane)) runs online softmax per (group, dim), reading each
/// V[p][dim] once and reusing it across the group, then merges the per-dim
/// lanes. K/V global traffic drops by `groups`x vs block-per-head decode.
const ATTN_DECODE_BATCHED_F16_GQA: &str = r#"
__device__ inline float f16_bits_to_f32(unsigned short u) {
    union { _Float16 h; unsigned short u; } c;
    c.u = u;
    return (float)c.h;
}

extern "C" __global__ void attn_decode_batched_f16_gqa(
    const float* __restrict__ q,
    const unsigned short* __restrict__ kc,
    const unsigned short* __restrict__ vc,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    const int* __restrict__ slots,
    const int* __restrict__ run_mask,
    int batch, int n_heads, int n_kv_heads, int head_dim, float scale, int max_seq) {
    const int s = blockIdx.x / n_kv_heads;
    const int kv = blockIdx.x % n_kv_heads;
    const int slot = slots[s];
    const int pos = pos_buf[s];
    if (run_mask[s] != 0) {
        return;
    }
    const int T = blockDim.x;
    const int tid = threadIdx.x;
    const int groups = n_heads / n_kv_heads;
    const int per = T / head_dim;   // lanes per output dim (key split)
    const int tile = T;             // positions per phase-A round
    const int np = pos + 1;

    extern __shared__ float sm[];
    float* scores = sm;                          // [groups][tile]
    float* sm_m = sm + (long long)groups * tile; // [T][groups] lane partials
    float* sm_l = sm_m + (long long)T * groups;
    float* sm_a = sm_l + (long long)T * groups;

    // Phase-B online-softmax state per (dim, lane) per group.
    float m[16], l[16], acc[16];
    for (int g = 0; g < groups; g++) {
        m[g] = -1e30f;
        l[g] = 0.0f;
        acc[g] = 0.0f;
    }

    for (int tile0 = 0; tile0 < np; tile0 += tile) {
        const int n_t = (np - tile0 < tile) ? (np - tile0) : tile;
        // Phase A: full head_dim dot per group, K[p] read once per position.
        if (tid < n_t) {
            const int p = tile0 + tid;
            const unsigned short* krow =
                kc + (((long long)slot * max_seq + p) * n_kv_heads + kv) * head_dim;
            float dot[16];
            for (int g = 0; g < groups; g++) dot[g] = 0.0f;
            // Vectorized K row reads (8 fp16 per uint4) cut L1 load count.
            for (int dd8 = 0; dd8 < head_dim; dd8 += 8) {
                const uint4 v = *((const uint4*)(krow + dd8));
                float f0 = f16_bits_to_f32(v.x & 0xffffu);
                float f1 = f16_bits_to_f32(v.x >> 16);
                float f2 = f16_bits_to_f32(v.y & 0xffffu);
                float f3 = f16_bits_to_f32(v.y >> 16);
                float f4 = f16_bits_to_f32(v.z & 0xffffu);
                float f5 = f16_bits_to_f32(v.z >> 16);
                float f6 = f16_bits_to_f32(v.w & 0xffffu);
                float f7 = f16_bits_to_f32(v.w >> 16);
                for (int g = 0; g < groups; g++) {
                    const float* qg = q + ((long long)s * n_heads + kv * groups + g) * head_dim + dd8;
                    dot[g] += qg[0] * f0 + qg[1] * f1 + qg[2] * f2 + qg[3] * f3
                            + qg[4] * f4 + qg[5] * f5 + qg[6] * f6 + qg[7] * f7;
                }
            }
            for (int g = 0; g < groups; g++) {
                scores[(long long)g * tile + tid] = dot[g] * scale;
            }
        }
        __syncthreads();
        // Phase B: V reused across the group; online softmax per (g, dim).
        const int dd = tid % head_dim;
        const int c = tid / head_dim;
        const unsigned short* vrow =
            vc + (((long long)slot * max_seq + tile0) * n_kv_heads + kv) * head_dim + dd;
        for (int pp = c; pp < n_t; pp += per) {
            const float vvv = f16_bits_to_f32(vrow[(long long)pp * n_kv_heads * head_dim]);
            for (int g = 0; g < groups; g++) {
                const float sc = scores[(long long)g * tile + pp];
                const float mnew = fmaxf(m[g], sc);
                const float alpha = __expf(m[g] - mnew);
                const float beta = __expf(sc - mnew);
                l[g] = l[g] * alpha + beta;
                acc[g] = acc[g] * alpha + beta * vvv;
                m[g] = mnew;
            }
        }
        __syncthreads(); // protect shared scores before the next tile overwrites
    }

    // Merge the per-lane partials and write the output.
    const int dd = tid % head_dim;
    const int c = tid / head_dim;
    for (int g = 0; g < groups; g++) {
        const long long idx = ((long long)c * groups + g) * head_dim + dd;
        sm_m[idx] = m[g];
        sm_l[idx] = l[g];
        sm_a[idx] = acc[g];
    }
    __syncthreads();
    if (c == 0) {
        for (int g = 0; g < groups; g++) {
            float m2 = -1e30f, l2 = 0.0f, a2 = 0.0f;
            for (int cc = 0; cc < per; cc++) {
                const long long idx = ((long long)cc * groups + g) * head_dim + dd;
                const float mi = sm_m[idx];
                const float li = sm_l[idx];
                const float ai = sm_a[idx];
                const float mnew = fmaxf(m2, mi);
                const float alpha = __expf(m2 - mnew);
                const float beta = __expf(mi - mnew);
                l2 = l2 * alpha + li * beta;
                a2 = a2 * alpha + ai * beta;
                m2 = mnew;
            }
            const int h = kv * groups + g;
            out[((long long)s * n_heads + h) * head_dim + dd] = a2 / l2;
        }
    }
}
"#;

/// Prefill attention with shared K/V reads (one block per (run, head)).
///
/// A "run" is C consecutive query rows of one sequence at positions
/// [base, base+C). Each key position is read once and reused for every row
/// that causally attends to it, so attention K/V traffic drops from O(C * pos)
/// (decode-style per-row attention) to O(pos).
///
/// Layout: 256 threads = 64 rows x 4 dim-groups (head_dim/4 each). A row's
/// score dot is reduced across its quad with warp shuffles; the causal mask
/// (`r >= p - base`) is value-based so warps stay uniform. Two passes: max
/// scores (pass A) then exp-weighted V accumulation (pass B).
const ATTN_PREFILL_F16: &str = r#"
__device__ float f16_to_f32(unsigned short h) {
    unsigned int sign = ((unsigned int)h & 0x8000u) << 16;
    unsigned int e = (h >> 10) & 0x1fu;
    unsigned int m = h & 0x3ffu;
    unsigned int b;
    if (e == 0u) {
        if (m == 0u) b = sign;
        else {
            unsigned int m2 = m;
            unsigned int e2 = 127u + 1u - 15u;
            while ((m2 & 0x400u) == 0u) { m2 <<= 1; e2--; }
            b = sign | (e2 << 23) | ((m2 & 0x3ffu) << 13);
        }
    } else if (e == 0x1fu) {
        b = sign | 0x7f800000u | (m << 13);
    } else {
        b = sign | ((e + 127u - 15u) << 23) | (m << 13);
    }
    return __uint_as_float(b);
}

extern "C" __global__ void attn_prefill_f16(
    const float* __restrict__ q,          // [total_rows, n_heads, head_dim]
    const unsigned short* __restrict__ kc, // [slot][max_seq][n_kv_heads][head_dim]
    const unsigned short* __restrict__ vc,
    float* __restrict__ out,              // [total_rows, n_heads, head_dim]
    int qoff, int C, int base, int n_heads, int n_kv_heads, int head_dim,
    float scale, int max_seq, int slot)
{
    const int h = blockIdx.x;
    const int kv = h / (n_heads / n_kv_heads);
    const int tid = threadIdx.x;
    const int r = tid / 4;        // 0..63 query row within the run
    const int g = tid % 4;        // dim group (head_dim/4 each)
    const int D = head_dim / 4;   // <= 16 for head_dim <= 64
    const int P = base + C;       // key positions 0..P-1

    __shared__ float s_max[64];
    __shared__ float s_sum[64];
    if (tid < 64) {
        s_max[tid] = -1e30f;
        s_sum[tid] = 0.0f;
    }
    __syncthreads();

    const float* qh = q + ((long long)(qoff + r) * n_heads + h) * head_dim;
    float qreg[16];
    #pragma unroll
    for (int d = 0; d < D; d++) qreg[d] = qh[g * D + d];

    const unsigned short* kbase = kc + ((long long)slot * max_seq) * n_kv_heads * head_dim + kv * head_dim;
    const unsigned short* vbase = vc + ((long long)slot * max_seq) * n_kv_heads * head_dim + kv * head_dim;

    // Pass A: per-row max score over causally-attended keys.
    for (int p = 0; p < P; p++) {
        float s = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            s += qreg[d] * f16_to_f32(kbase[(long long)p * n_kv_heads * head_dim + g * D + d]);
        }
        s += __shfl_down(s, 1);
        s += __shfl_down(s, 2);
        s = __shfl(s, (tid & ~3) & 31);
        s *= scale;
        float sm = (r >= p - base) ? s : -1e30f;
        if (g == 0) {
            s_max[r] = fmaxf(s_max[r], sm);
        }
    }
    __syncthreads();

    // Pass B: exp-weighted V accumulation (K read again, V read once).
    float sumv = 0.0f;
    float acc[16];
    #pragma unroll
    for (int d = 0; d < D; d++) acc[d] = 0.0f;
    for (int p = 0; p < P; p++) {
        float s = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            s += qreg[d] * f16_to_f32(kbase[(long long)p * n_kv_heads * head_dim + g * D + d]);
        }
        s += __shfl_down(s, 1);
        s += __shfl_down(s, 2);
        s = __shfl(s, (tid & ~3) & 31);
        s *= scale;
        float e = (r >= p - base) ? __expf(s - s_max[r]) : 0.0f;
        sumv += e;
        const unsigned short* vp = vbase + (long long)p * n_kv_heads * head_dim + g * D;
        #pragma unroll
        for (int d = 0; d < D; d++) acc[d] += e * f16_to_f32(vp[d]);
    }
    if (g == 0) {
        s_sum[r] = sumv;
    }
    __syncthreads();

    float* oh = out + ((long long)(qoff + r) * n_heads + h) * head_dim;
    if (r < C) {
        float inv = 1.0f / s_sum[r];
        #pragma unroll
        for (int d = 0; d < D; d++) oh[g * D + d] = acc[d] * inv;
    }
}
"#;

/// fp16 embedding gather -> f32 activations (matches `crate::fp16::f16_to_f32`).
const EMBED_GATHER_F16: &str = r#" 
__device__ inline float f16_to_f32(unsigned short h) {
    union { _Float16 h; unsigned short u; } c;
    c.u = h;
    return (float)c.h;
}

extern "C" __global__ void embed_gather_f16(const int* tok, const unsigned short* emb,
                                            float* x, int cols, int batch) {
    int s = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) x[(long long)s * cols + i] = f16_to_f32(emb[(long long)tok[s] * cols + i]);
}
"#;

/// MoE routing (single token): softmax over `ne` expert logits, select the
/// `topk` highest-probability experts (ties: lower index, matching the CPU
/// reference), and emit per-slot normalized weights (prob / top-k sum).
const MOE_ROUTER: &str = r#"
extern "C" __global__ void moe_router(
    const float* __restrict__ logits,
    int* __restrict__ out_ids,
    float* __restrict__ out_w,
    int ne, int topk) {
    extern __shared__ float sm[];
    float* probs = sm;
    __shared__ float red[256];
    float maxv = -1e30f;
    for (int i = threadIdx.x; i < ne; i += blockDim.x) maxv = fmaxf(maxv, logits[i]);
    red[threadIdx.x] = maxv;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    float m = red[0];
    float sumv = 0.0f;
    for (int i = threadIdx.x; i < ne; i += blockDim.x) {
        probs[i] = __expf(logits[i] - m);
        sumv += probs[i];
    }
    red[threadIdx.x] = sumv;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    for (int i = threadIdx.x; i < ne; i += blockDim.x) probs[i] *= inv;
    __syncthreads();
    if (threadIdx.x == 0) {
        float norm = 0.0f;
        for (int k = 0; k < topk; k++) {
            int best = -1;
            float bestv = -1e30f;
            for (int i = 0; i < ne; i++) {
                int used = 0;
                for (int j = 0; j < k; j++) {
                    if (out_ids[j] == i) { used = 1; break; }
                }
                if (!used) {
                    if (probs[i] > bestv || (probs[i] == bestv && (best < 0 || i < best))) {
                        bestv = probs[i];
                        best = i;
                    }
                }
            }
            out_ids[k] = best;
            norm += probs[best];
        }
        for (int k = 0; k < topk; k++) out_w[k] = probs[out_ids[k]] / norm;
    }
}
"#;

/// MoE weight gather: pack the selected experts' gate/up/down matrices into
/// contiguous per-slot scratch (slot k holds expert `ids[k]`).
const MOE_GATHER_WEIGHTS: &str = r#"
extern "C" __global__ void moe_gather_weights(
    const float* __restrict__ wg, const float* __restrict__ wu, const float* __restrict__ wd,
    const int* __restrict__ ids,
    float* __restrict__ wg_pack, float* __restrict__ wu_pack, float* __restrict__ wd_pack,
    int ne, int inter, int d, int topk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    long long gi = (long long)inter * d;
    long long di = (long long)d * inter;
    if (i < topk * gi) {
        int slot = (int)(i / gi);
        long long rem = i % gi;
        long long src = (long long)ids[slot] * gi + rem;
        wg_pack[i] = wg[src];
        wu_pack[i] = wu[src];
    }
    if (i < topk * di) {
        int slot = (int)(i / di);
        long long rem = i % di;
        long long src = (long long)ids[slot] * di + rem;
        wd_pack[i] = wd[src];
    }
}
"#;

/// MoE residual accumulate: `x[i] += sum_k w[k] * down_all[k*d + i]`.
const MOE_ACCUMULATE: &str = r#"
extern "C" __global__ void moe_accumulate(
    float* __restrict__ x,
    const float* __restrict__ down_all,
    const float* __restrict__ w,
    int d, int topk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        float acc = 0.0f;
        for (int k = 0; k < topk; k++) acc += w[k] * down_all[(long long)k * d + i];
        x[i] += acc;
    }
}
"#;

/// Compiled kernels plus the hipBLAS handle, bound to one stream.
pub struct HipKernels {
    hip: std::sync::Arc<Hip>,
    /// Shared execution stream (also the capture stream).
    pub stream: HipStream,
    /// hipBLAS handle bound to `stream`.
    pub blas: mach_kernel_sys::hipblas::HipBlas,
    embed: HipKernelModule,
    rms_norm: HipKernelModule,
    silu_mul: HipKernelModule,
    add: HipKernelModule,
    add_bias: HipKernelModule,
    kv_store: HipKernelModule,
    attn_decode: HipKernelModule,
    rope: HipKernelModule,
    embed_batched: HipKernelModule,
    rope_batched: HipKernelModule,
    kv_store_batched: HipKernelModule,
    attn_decode_batched: HipKernelModule,
    argmax_batched: HipKernelModule,
    cast_f32_f16: HipKernelModule,
    cast_f16_f32: HipKernelModule,
    embed_f16: HipKernelModule,
    kv_store_f16: HipKernelModule,
    attn_f16_gqa: HipKernelModule,
    attn_prefill_f16: HipKernelModule,
    moe_router: HipKernelModule,
    moe_gather: HipKernelModule,
    moe_accumulate: HipKernelModule,
}

// SAFETY: a HipKernels instance is used by one model on one thread; the raw
// stream handle is only touched there, and the loaded runtimes are Send+Sync.
unsafe impl Send for HipKernels {}
unsafe impl Sync for HipKernels {}

impl HipKernels {
    /// Compiles all kernels and initializes hipBLAS on a fresh stream.
    pub fn new(hip: std::sync::Arc<Hip>) -> Result<Self, Error> {
        let arch = hip_arch();
        let mut stream = std::ptr::null_mut();
        unsafe { hip::check(&hip, (hip.api.hip_stream_create)(&mut stream))? };

        let blas = mach_kernel_sys::hipblas::HipBlas::new(std::sync::Arc::clone(&hip))?;
        blas.set_stream(stream)?;

        Ok(Self {
            hip,
            stream,
            blas,
            embed: HipKernelModule::compile(&arch, EMBED_GATHER, "embed_gather")?,
            rms_norm: HipKernelModule::compile(&arch, RMS_NORM, "rms_norm")?,
            silu_mul: HipKernelModule::compile(&arch, SILU_MUL, "silu_mul")?,
            add: HipKernelModule::compile(&arch, ADD, "add")?,
            add_bias: HipKernelModule::compile(&arch, ADD_BIAS, "add_bias")?,
            kv_store: HipKernelModule::compile(&arch, KV_STORE, "kv_store")?,
            attn_decode: HipKernelModule::compile(&arch, ATTN_DECODE, "attn_decode")?,
            rope: HipKernelModule::compile(&arch, ROPE, "rope")?,
            embed_batched: HipKernelModule::compile(&arch, EMBED_BATCHED, "embed_batched")?,
            rope_batched: HipKernelModule::compile(&arch, ROPE_BATCHED, "rope_batched")?,
            kv_store_batched: HipKernelModule::compile(
                &arch,
                KV_STORE_BATCHED,
                "kv_store_batched",
            )?,
            attn_decode_batched: HipKernelModule::compile(
                &arch,
                ATTN_DECODE_BATCHED,
                "attn_decode_batched",
            )?,
            argmax_batched: HipKernelModule::compile(&arch, ARGMAX_BATCHED, "argmax_batched")?,
            cast_f32_f16: HipKernelModule::compile(&arch, CAST_F32_F16, "cast_f32_f16")?,
            cast_f16_f32: HipKernelModule::compile(&arch, CAST_F16_F32, "cast_f16_f32")?,
            embed_f16: HipKernelModule::compile(&arch, EMBED_GATHER_F16, "embed_gather_f16")?,
            kv_store_f16: HipKernelModule::compile(&arch, KV_F16, "kv_store_batched_f16")?,
            attn_f16_gqa: HipKernelModule::compile(
                &arch,
                ATTN_DECODE_BATCHED_F16_GQA,
                "attn_decode_batched_f16_gqa",
            )?,
            attn_prefill_f16: HipKernelModule::compile(
                &arch,
                ATTN_PREFILL_F16,
                "attn_prefill_f16",
            )?,
            moe_router: HipKernelModule::compile(&arch, MOE_ROUTER, "moe_router")?,
            moe_gather: HipKernelModule::compile(&arch, MOE_GATHER_WEIGHTS, "moe_gather_weights")?,
            moe_accumulate: HipKernelModule::compile(&arch, MOE_ACCUMULATE, "moe_accumulate")?,
        })
    }

    /// The raw HIP runtime.
    pub fn hip(&self) -> &std::sync::Arc<Hip> {
        &self.hip
    }
}

impl HipKernels {
    pub fn launch_embed(
        &self,
        tok: *const i32,
        emb: *const f32,
        x: *mut f32,
        cols: i32,
    ) -> Result<(), Error> {
        let t = tok;
        let e = emb;
        let xp = x;
        let mut p = vec![
            &t as *const *const i32 as *mut core::ffi::c_void,
            &e as *const *const f32 as *mut core::ffi::c_void,
            &xp as *const *mut f32 as *mut core::ffi::c_void,
            &cols as *const i32 as *mut core::ffi::c_void,
        ];
        // cols can exceed one block (256 threads), so grid over blocks.
        let blocks = (cols as u32).div_ceil(256);
        Ok(self
            .embed
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    pub fn launch_rms_norm(
        &self,
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        rows: i32,
        cols: i32,
        eps: f32,
    ) -> Result<(), Error> {
        let xp = x;
        let wp = w;
        let yp = y;
        let mut p = vec![
            &xp as *const *const f32 as *mut core::ffi::c_void,
            &wp as *const *const f32 as *mut core::ffi::c_void,
            &yp as *const *mut f32 as *mut core::ffi::c_void,
            &cols as *const i32 as *mut core::ffi::c_void,
            &eps as *const f32 as *mut core::ffi::c_void,
        ];
        Ok(self
            .rms_norm
            .launch([rows as u32, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    pub fn launch_silu_mul(
        &self,
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        n: i32,
    ) -> Result<(), Error> {
        let ap = a;
        let bp = b;
        let op = out;
        let mut p = vec![
            &ap as *const *const f32 as *mut core::ffi::c_void,
            &bp as *const *const f32 as *mut core::ffi::c_void,
            &op as *const *mut f32 as *mut core::ffi::c_void,
            &n as *const i32 as *mut core::ffi::c_void,
        ];
        let blocks = (n as u32).div_ceil(256);
        Ok(self
            .silu_mul
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    pub fn launch_add(&self, x: *mut f32, y: *const f32, n: i32) -> Result<(), Error> {
        let xp = x;
        let yp = y;
        let mut p = vec![
            &xp as *const *mut f32 as *mut core::ffi::c_void,
            &yp as *const *const f32 as *mut core::ffi::c_void,
            &n as *const i32 as *mut core::ffi::c_void,
        ];
        let blocks = (n as u32).div_ceil(256);
        Ok(self
            .add
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// Broadcast-adds `bias` (length `cols`) to every row of `x`.
    pub fn launch_add_bias(
        &self,
        x: *mut f32,
        bias: *const f32,
        rows: i32,
        cols: i32,
    ) -> Result<(), Error> {
        let xp = x;
        let bp = bias;
        let mut p = vec![
            &xp as *const *mut f32 as *mut core::ffi::c_void,
            &bp as *const *const f32 as *mut core::ffi::c_void,
            &rows as *const i32 as *mut core::ffi::c_void,
            &cols as *const i32 as *mut core::ffi::c_void,
        ];
        let total = (rows * cols) as u32;
        let blocks = total.div_ceil(256);
        Ok(self
            .add_bias
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    pub fn launch_kv_store(
        &self,
        kv: *const f32,
        cache: *mut f32,
        pos: *const i32,
        kv_heads: i32,
        head_dim: i32,
        max_seq: i32,
    ) -> Result<(), Error> {
        let kvp = kv;
        let cp = cache;
        let pp = pos;
        let mut p = vec![
            &kvp as *const *const f32 as *mut core::ffi::c_void,
            &cp as *const *mut f32 as *mut core::ffi::c_void,
            &pp as *const *const i32 as *mut core::ffi::c_void,
            &kv_heads as *const i32 as *mut core::ffi::c_void,
            &head_dim as *const i32 as *mut core::ffi::c_void,
            &max_seq as *const i32 as *mut core::ffi::c_void,
        ];
        Ok(self
            .kv_store
            .launch([1, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_attn_decode(
        &self,
        q: *const f32,
        kc: *const f32,
        vc: *const f32,
        out: *mut f32,
        pos: *const i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
    ) -> Result<(), Error> {
        let qp = q;
        let kp = kc;
        let vp = vc;
        let op = out;
        let pp = pos;
        let mut p = vec![
            &qp as *const *const f32 as *mut core::ffi::c_void,
            &kp as *const *const f32 as *mut core::ffi::c_void,
            &vp as *const *const f32 as *mut core::ffi::c_void,
            &op as *const *mut f32 as *mut core::ffi::c_void,
            &pp as *const *const i32 as *mut core::ffi::c_void,
            &n_heads as *const i32 as *mut core::ffi::c_void,
            &n_kv_heads as *const i32 as *mut core::ffi::c_void,
            &head_dim as *const i32 as *mut core::ffi::c_void,
            &scale as *const f32 as *mut core::ffi::c_void,
        ];
        // Dynamic shared memory: scores[1024] + red[256] floats.
        const SHARED_FLOATS: u32 = 1024 + 256;
        Ok(self.attn_decode.launch_shmem(
            [n_heads as u32, 1, 1],
            [256, 1, 1],
            &mut p,
            self.stream,
            SHARED_FLOATS * 4,
        )?)
    }

    /// Applies rotary position embeddings to q and k in place.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_rope(
        &self,
        q: *mut f32,
        k: *mut f32,
        pos: *const i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        theta: f32,
    ) -> Result<(), Error> {
        let qp = q;
        let kp = k;
        let pp = pos;
        let mut p = vec![
            &qp as *const *mut f32 as *mut core::ffi::c_void,
            &kp as *const *mut f32 as *mut core::ffi::c_void,
            &pp as *const *const i32 as *mut core::ffi::c_void,
            &n_heads as *const i32 as *mut core::ffi::c_void,
            &n_kv_heads as *const i32 as *mut core::ffi::c_void,
            &head_dim as *const i32 as *mut core::ffi::c_void,
            &theta as *const f32 as *mut core::ffi::c_void,
        ];
        let total = (n_heads.max(n_kv_heads) * head_dim) as u32;
        let blocks = total.div_ceil(256);
        Ok(self
            .rope
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// `out[1, n] = x[1, k] @ W^T` where `w` is `[n, k]` row-major.
    pub fn gemm(
        &self,
        out: *mut f32,
        x: *const f32,
        w: *const f32,
        n: i32,
        k: i32,
    ) -> Result<(), Error> {
        // Column-major hipBLAS. A row-major [n, k] weight matrix IS the
        // column-major storage of B (k x n) with ldb = k, so opB = N, ldb = k.
        // For m = 1 the A/C leading dims are 1.
        self.blas
            .sgemm(
                mach_kernel_sys::hipblas::HIPBLAS_OP_N,
                mach_kernel_sys::hipblas::HIPBLAS_OP_N,
                1,
                n,
                k,
                1.0,
                x,
                1,
                w,
                k,
                0.0,
                out,
                1,
            )
            .map_err(|e| Error::Model(format!("hipblas sgemm m=1 n={n} k={k}: {e}")))
    }

    /// Batched GEMM: `out[B, n] = x[B, k] @ W^T`, `w` is `[n, k]` row-major.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batched(
        &self,
        out: *mut f32,
        x: *const f32,
        w: *const f32,
        batch: i32,
        n: i32,
        k: i32,
    ) -> Result<(), Error> {
        self.blas
            .sgemm(
                mach_kernel_sys::hipblas::HIPBLAS_OP_T,
                mach_kernel_sys::hipblas::HIPBLAS_OP_N,
                n,
                batch,
                k,
                1.0,
                w,
                k,
                x,
                k,
                0.0,
                out,
                n,
            )
            .map_err(|e| Error::Model(format!("hipblas batched sgemm m={n} n={batch} k={k}: {e}")))
    }

    pub fn launch_embed_batched(
        &self,
        tok: *const i32,
        emb: *const f32,
        x: *mut f32,
        cols: i32,
        batch: i32,
    ) -> Result<(), Error> {
        let tp = tok;
        let ep = emb;
        let xp = x;
        let mut p = vec![
            &tp as *const *const i32 as *mut core::ffi::c_void,
            &ep as *const *const f32 as *mut core::ffi::c_void,
            &xp as *const *mut f32 as *mut core::ffi::c_void,
            &cols as *const i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
        ];
        let blocks = (cols as u32).div_ceil(256);
        Ok(self.embed_batched.launch(
            [blocks, batch as u32, 1],
            [256, 1, 1],
            &mut p,
            self.stream,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_rope_batched(
        &self,
        q: *mut f32,
        k: *mut f32,
        pos: *const i32,
        batch: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        theta: f32,
    ) -> Result<(), Error> {
        let qp = q;
        let kp = k;
        let pp = pos;
        let mut p = vec![
            &qp as *const *mut f32 as *mut core::ffi::c_void,
            &kp as *const *mut f32 as *mut core::ffi::c_void,
            &pp as *const *const i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &n_heads as *const i32 as *mut core::ffi::c_void,
            &n_kv_heads as *const i32 as *mut core::ffi::c_void,
            &head_dim as *const i32 as *mut core::ffi::c_void,
            &theta as *const f32 as *mut core::ffi::c_void,
        ];
        let total = (batch * n_heads.max(n_kv_heads) * head_dim) as u32;
        let blocks = total.div_ceil(256);
        Ok(self
            .rope_batched
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_kv_store_batched(
        &self,
        kv: *const f32,
        cache: *mut f32,
        pos: *const i32,
        slots: *const i32,
        batch: i32,
        kv_heads: i32,
        head_dim: i32,
        max_seq: i32,
    ) -> Result<(), Error> {
        let kvp = kv;
        let cp = cache;
        let pp = pos;
        let sp = slots;
        let mut p = vec![
            &kvp as *const *const f32 as *mut core::ffi::c_void,
            &cp as *const *mut f32 as *mut core::ffi::c_void,
            &pp as *const *const i32 as *mut core::ffi::c_void,
            &sp as *const *const i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &kv_heads as *const i32 as *mut core::ffi::c_void,
            &head_dim as *const i32 as *mut core::ffi::c_void,
            &max_seq as *const i32 as *mut core::ffi::c_void,
        ];
        let total = (batch * kv_heads * head_dim) as u32;
        let blocks = total.div_ceil(256);
        Ok(self
            .kv_store_batched
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_attn_decode_batched(
        &self,
        q: *const f32,
        kc: *const f32,
        vc: *const f32,
        out: *mut f32,
        pos: *const i32,
        slots: *const i32,
        batch: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        max_seq: i32,
    ) -> Result<(), Error> {
        let qp = q;
        let kp = kc;
        let vp = vc;
        let op = out;
        let pp = pos;
        let sp = slots;
        let mut p = vec![
            &qp as *const *const f32 as *mut core::ffi::c_void,
            &kp as *const *const f32 as *mut core::ffi::c_void,
            &vp as *const *const f32 as *mut core::ffi::c_void,
            &op as *const *mut f32 as *mut core::ffi::c_void,
            &pp as *const *const i32 as *mut core::ffi::c_void,
            &sp as *const *const i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &n_heads as *const i32 as *mut core::ffi::c_void,
            &n_kv_heads as *const i32 as *mut core::ffi::c_void,
            &head_dim as *const i32 as *mut core::ffi::c_void,
            &scale as *const f32 as *mut core::ffi::c_void,
            &max_seq as *const i32 as *mut core::ffi::c_void,
        ];
        let grid = (batch * n_heads) as u32;
        let shared = (max_seq as u32 + 256) * 4;
        Ok(self.attn_decode_batched.launch_shmem(
            [grid, 1, 1],
            [256, 1, 1],
            &mut p,
            self.stream,
            shared,
        )?)
    }

    pub fn launch_argmax_batched(
        &self,
        logits: *const f32,
        out_tok: *mut i32,
        vocab: i32,
        batch: i32,
    ) -> Result<(), Error> {
        let lp = logits;
        let op = out_tok;
        let mut p = vec![
            &lp as *const *const f32 as *mut core::ffi::c_void,
            &op as *const *mut i32 as *mut core::ffi::c_void,
            &vocab as *const i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
        ];
        Ok(self
            .argmax_batched
            .launch([batch as u32, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// Casts `n` f32 values to fp16 (device-side, matches `crate::fp16`).
    pub fn launch_cast_f32_f16(&self, x: *const f32, y: *mut u16, n: usize) -> Result<(), Error> {
        let xp = x;
        let yp = y;
        let ni = n as i32;
        let mut p = vec![
            &xp as *const *const f32 as *mut core::ffi::c_void,
            &yp as *const *mut u16 as *mut core::ffi::c_void,
            &ni as *const i32 as *mut core::ffi::c_void,
        ];
        let blocks = (n as u32).div_ceil(256);
        Ok(self
            .cast_f32_f16
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// Casts `n` fp16 values to f32 (device-side, matches `crate::fp16`).
    pub fn launch_cast_f16_f32(&self, x: *const u16, y: *mut f32, n: usize) -> Result<(), Error> {
        let xp = x;
        let yp = y;
        let ni = n as i32;
        let mut p = vec![
            &xp as *const *const u16 as *mut core::ffi::c_void,
            &yp as *const *mut f32 as *mut core::ffi::c_void,
            &ni as *const i32 as *mut core::ffi::c_void,
        ];
        let blocks = (n as u32).div_ceil(256);
        Ok(self
            .cast_f16_f32
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// Embedding gather from an fp16 table into f32 activations (2D grid:
    /// `blockIdx.y` = sequence, `blockIdx.x` covers columns).
    pub fn launch_embed_f16(
        &self,
        tok: *const i32,
        emb: *const u16,
        x: *mut f32,
        cols: i32,
        batch: i32,
    ) -> Result<(), Error> {
        let tp = tok;
        let ep = emb;
        let xp = x;
        let mut p = vec![
            &tp as *const *const i32 as *mut core::ffi::c_void,
            &ep as *const *const u16 as *mut core::ffi::c_void,
            &xp as *const *mut f32 as *mut core::ffi::c_void,
            &cols as *const i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
        ];
        let blocks = (cols as u32).div_ceil(256);
        Ok(self
            .embed_f16
            .launch([blocks, batch as u32, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// fp16 GEMM for m = 1: `out[1, n] = x[1, k] @ w16^T`. Hidden layers output
    /// fp16 (`yh` scratch) then cast to f32: rocBLAS fp16 C is far faster than
    /// fp32 C for the tall-skinny shapes. `xh`/`yh` are fp16 scratch of `k`/`n`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
        &self,
        out: *mut f32,
        x: *const f32,
        w16: *const u16,
        n: i32,
        k: i32,
        xh: *mut u16,
        yh: *mut u16,
    ) -> Result<(), Error> {
        use mach_kernel_sys::hipblas::{HIPBLAS_COMPUTE_32F, HIPBLAS_OP_N, HIPBLAS_R_16F};
        self.launch_cast_f32_f16(x, xh, k as usize)?;
        self.blas
            .gemm_ex(
                HIPBLAS_OP_N,
                HIPBLAS_OP_N,
                1,
                n,
                k,
                HIPBLAS_R_16F,
                xh as *const core::ffi::c_void,
                1,
                HIPBLAS_R_16F,
                w16 as *const core::ffi::c_void,
                k,
                HIPBLAS_R_16F,
                yh as *mut core::ffi::c_void,
                1,
                HIPBLAS_COMPUTE_32F,
            )
            .map_err(|e| Error::Model(format!("hipblas gemm_ex m=1 n={n} k={k}: {e}")))?;
        self.launch_cast_f16_f32(yh, out, n as usize)
    }

    /// Batched fp16 GEMM: `out[B, n] = x[B, k] @ w16^T`. Hidden layers output
    /// fp16 (`yh` scratch) then cast to f32 (rocBLAS fp16 C is much faster for
    /// the tall-skinny MLP shapes). `w16` is row-major `[n, k]`; `xh`/`yh` are
    /// fp16 scratch of `batch*k` / `batch*n`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batched_f16(
        &self,
        out: *mut f32,
        x: *const f32,
        w16: *const u16,
        batch: i32,
        n: i32,
        k: i32,
        xh: *mut u16,
        yh: *mut u16,
    ) -> Result<(), Error> {
        use mach_kernel_sys::hipblas::{
            HIPBLAS_COMPUTE_32F, HIPBLAS_OP_N, HIPBLAS_OP_T, HIPBLAS_R_16F,
        };
        self.launch_cast_f32_f16(x, xh, (batch * k) as usize)?;
        self.blas
            .gemm_ex(
                HIPBLAS_OP_T,
                HIPBLAS_OP_N,
                n,
                batch,
                k,
                HIPBLAS_R_16F,
                w16 as *const core::ffi::c_void,
                k,
                HIPBLAS_R_16F,
                xh as *const core::ffi::c_void,
                k,
                HIPBLAS_R_16F,
                yh as *mut core::ffi::c_void,
                n,
                HIPBLAS_COMPUTE_32F,
            )
            .map_err(|e| {
                Error::Model(format!(
                    "hipblas gemm_ex batched m={n} n={batch} k={k}: {e}"
                ))
            })?;
        self.launch_cast_f16_f32(yh, out, (batch * n) as usize)
    }

    /// Batched fp16 GEMM with fp32 output (lm_head: keeps fp32 logits for the
    /// sampler). `xh` is fp16 scratch of `batch*k`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batched_f16_logits(
        &self,
        out: *mut f32,
        x: *const f32,
        w16: *const u16,
        batch: i32,
        n: i32,
        k: i32,
        xh: *mut u16,
    ) -> Result<(), Error> {
        use mach_kernel_sys::hipblas::{
            HIPBLAS_COMPUTE_32F, HIPBLAS_OP_N, HIPBLAS_OP_T, HIPBLAS_R_16F, HIPBLAS_R_32F,
        };
        self.launch_cast_f32_f16(x, xh, (batch * k) as usize)?;
        self.blas
            .gemm_ex(
                HIPBLAS_OP_T,
                HIPBLAS_OP_N,
                n,
                batch,
                k,
                HIPBLAS_R_16F,
                w16 as *const core::ffi::c_void,
                k,
                HIPBLAS_R_16F,
                xh as *const core::ffi::c_void,
                k,
                HIPBLAS_R_32F,
                out as *mut core::ffi::c_void,
                n,
                HIPBLAS_COMPUTE_32F,
            )
            .map_err(|e| {
                Error::Model(format!(
                    "hipblas gemm_ex batched m={n} n={batch} k={k}: {e}"
                ))
            })
    }

    /// Prefill attention for a run of `C` consecutive rows (same slot,
    /// positions `[base, base+C)`) with shared K/V reads. One block per head.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attn_prefill_f16(
        &self,
        q: *const f32,
        kc: *const u16,
        vc: *const u16,
        out: *mut f32,
        qoff: i32,
        c: i32,
        base: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        max_seq: i32,
        slot: i32,
    ) -> Result<(), Error> {
        let qp = q;
        let kp = kc;
        let vp = vc;
        let op = out;
        let mut p = vec![
            &qp as *const *const f32 as *mut core::ffi::c_void,
            &kp as *const *const u16 as *mut core::ffi::c_void,
            &vp as *const *const u16 as *mut core::ffi::c_void,
            &op as *const *mut f32 as *mut core::ffi::c_void,
            &qoff as *const i32 as *mut core::ffi::c_void,
            &c as *const i32 as *mut core::ffi::c_void,
            &base as *const i32 as *mut core::ffi::c_void,
            &n_heads as *const i32 as *mut core::ffi::c_void,
            &n_kv_heads as *const i32 as *mut core::ffi::c_void,
            &head_dim as *const i32 as *mut core::ffi::c_void,
            &scale as *const f32 as *mut core::ffi::c_void,
            &max_seq as *const i32 as *mut core::ffi::c_void,
            &slot as *const i32 as *mut core::ffi::c_void,
        ];
        Ok(self.attn_prefill_f16.launch(
            [n_heads as u32, 1, 1],
            [256, 1, 1],
            &mut p,
            self.stream,
        )?)
    }

    /// Stores f32 K/V rows into an fp16 cache at per-row positions (slots).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_kv_store_batched_f16(
        &self,
        kv: *const f32,
        cache: *mut u16,
        pos: *const i32,
        slots: *const i32,
        batch: i32,
        kv_heads: i32,
        head_dim: i32,
        max_seq: i32,
    ) -> Result<(), Error> {
        let kvp = kv;
        let cp = cache;
        let pp = pos;
        let sp = slots;
        let mut p = vec![
            &kvp as *const *const f32 as *mut core::ffi::c_void,
            &cp as *const *mut u16 as *mut core::ffi::c_void,
            &pp as *const *const i32 as *mut core::ffi::c_void,
            &sp as *const *const i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &kv_heads as *const i32 as *mut core::ffi::c_void,
            &head_dim as *const i32 as *mut core::ffi::c_void,
            &max_seq as *const i32 as *mut core::ffi::c_void,
        ];
        let total = (batch * kv_heads * head_dim) as u32;
        let blocks = total.div_ceil(256);
        Ok(self
            .kv_store_f16
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// Decode attention over an fp16 KV cache (f32 q, fp32 output).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attn_decode_batched_f16_gqa(
        &self,
        q: *const f32,
        kc: *const u16,
        vc: *const u16,
        out: *mut f32,
        pos: *const i32,
        slots: *const i32,
        run_mask: *const i32,
        batch: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        max_seq: i32,
    ) -> Result<(), Error> {
        let qp = q;
        let kp = kc;
        let vp = vc;
        let op = out;
        let pp = pos;
        let sp = slots;
        let mp = run_mask;
        let mut p = vec![
            &qp as *const *const f32 as *mut core::ffi::c_void,
            &kp as *const *const u16 as *mut core::ffi::c_void,
            &vp as *const *const u16 as *mut core::ffi::c_void,
            &op as *const *mut f32 as *mut core::ffi::c_void,
            &pp as *const *const i32 as *mut core::ffi::c_void,
            &sp as *const *const i32 as *mut core::ffi::c_void,
            &mp as *const *const i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &n_heads as *const i32 as *mut core::ffi::c_void,
            &n_kv_heads as *const i32 as *mut core::ffi::c_void,
            &head_dim as *const i32 as *mut core::ffi::c_void,
            &scale as *const f32 as *mut core::ffi::c_void,
            &max_seq as *const i32 as *mut core::ffi::c_void,
        ];
        let grid = (batch * n_kv_heads) as u32;
        let groups = n_heads / n_kv_heads;
        // scores [groups][256] + 3 merge arrays [256][groups], all floats.
        let shared = (groups * 256 + 3 * 256 * groups) as u32 * 4;
        Ok(self.attn_f16_gqa.launch_shmem(
            [grid, 1, 1],
            [256, 1, 1],
            &mut p,
            self.stream,
            shared,
        )?)
    }

    /// Synchronizes the execution stream.
    /// MoE router for a single token: softmax over `ne` logits + top-k
    /// selection; writes `out_ids[topk]` and normalized `out_w[topk]`.
    pub fn launch_moe_router(
        &self,
        logits: *const f32,
        out_ids: *mut i32,
        out_w: *mut f32,
        ne: i32,
        topk: i32,
    ) -> Result<(), Error> {
        let lp = logits;
        let ip = out_ids;
        let wp = out_w;
        let mut p = vec![
            &lp as *const *const f32 as *mut core::ffi::c_void,
            &ip as *const *mut i32 as *mut core::ffi::c_void,
            &wp as *const *mut f32 as *mut core::ffi::c_void,
            &ne as *const i32 as *mut core::ffi::c_void,
            &topk as *const i32 as *mut core::ffi::c_void,
        ];
        Ok(self.moe_router.launch_shmem(
            [1, 1, 1],
            [256, 1, 1],
            &mut p,
            self.stream,
            (ne as u32) * 4,
        )?)
    }

    /// Packs the selected experts' gate/up/down weights into per-slot scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_moe_gather_weights(
        &self,
        wg: *const f32,
        wu: *const f32,
        wd: *const f32,
        ids: *const i32,
        wg_pack: *mut f32,
        wu_pack: *mut f32,
        wd_pack: *mut f32,
        ne: i32,
        inter: i32,
        d: i32,
        topk: i32,
    ) -> Result<(), Error> {
        let wgp = wg;
        let wup = wu;
        let wdp = wd;
        let idp = ids;
        let wgpp = wg_pack;
        let wupp = wu_pack;
        let wdpp = wd_pack;
        let mut p = vec![
            &wgp as *const *const f32 as *mut core::ffi::c_void,
            &wup as *const *const f32 as *mut core::ffi::c_void,
            &wdp as *const *const f32 as *mut core::ffi::c_void,
            &idp as *const *const i32 as *mut core::ffi::c_void,
            &wgpp as *const *mut f32 as *mut core::ffi::c_void,
            &wupp as *const *mut f32 as *mut core::ffi::c_void,
            &wdpp as *const *mut f32 as *mut core::ffi::c_void,
            &ne as *const i32 as *mut core::ffi::c_void,
            &inter as *const i32 as *mut core::ffi::c_void,
            &d as *const i32 as *mut core::ffi::c_void,
            &topk as *const i32 as *mut core::ffi::c_void,
        ];
        let total = ((topk as i64) * (inter as i64) * (d as i64))
            .max((topk as i64) * (d as i64) * (inter as i64));
        let blocks = (total as u32).div_ceil(256);
        Ok(self
            .moe_gather
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// MoE residual accumulate: `x[i] += sum_k w[k] * down_all[k*d + i]`.
    pub fn launch_moe_accumulate(
        &self,
        x: *mut f32,
        down_all: *const f32,
        w: *const f32,
        d: i32,
        topk: i32,
    ) -> Result<(), Error> {
        let xp = x;
        let dp = down_all;
        let wp = w;
        let mut p = vec![
            &xp as *const *mut f32 as *mut core::ffi::c_void,
            &dp as *const *const f32 as *mut core::ffi::c_void,
            &wp as *const *const f32 as *mut core::ffi::c_void,
            &d as *const i32 as *mut core::ffi::c_void,
            &topk as *const i32 as *mut core::ffi::c_void,
        ];
        let blocks = (d as u32).div_ceil(256);
        Ok(self
            .moe_accumulate
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    pub fn sync(&self) -> Result<(), Error> {
        unsafe {
            hip::check(
                &self.hip,
                (self.hip.api.hip_stream_synchronize)(self.stream),
            )?
        };
        Ok(())
    }
}

impl Drop for HipKernels {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                let _ = (self.hip.api.hip_stream_destroy)(self.stream);
            }
        }
    }
}
