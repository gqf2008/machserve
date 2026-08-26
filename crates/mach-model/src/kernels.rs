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

/// Per-head RMSNorm (Qwen3 QK-norm), one block per (row, head); applied to
/// q and k in place after the projections, before RoPE.
const QK_NORM: &str = r#"
extern "C" __global__ void qk_norm(float* q, float* k,
                                   const float* qw, const float* kw,
                                   int rows, int n_heads, int n_kv_heads,
                                   int head_dim, float eps) {
    int row = blockIdx.x;
    int head = blockIdx.y;
    // q: [rows, n_heads * head_dim]
    {
        float* xr = q + ((long long)row * n_heads + head) * head_dim;
        const float* wr = qw + (long long)head * head_dim;
        __shared__ float redq[256];
        float ss = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) ss += xr[i] * xr[i];
        redq[threadIdx.x] = ss;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) redq[threadIdx.x] += redq[threadIdx.x + s];
            __syncthreads();
        }
        float inv = rsqrtf(redq[0] / (float)head_dim + eps);
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            xr[i] = xr[i] * inv * wr[i];
        }
    }
    // k: [rows, n_kv_heads * head_dim] (skip when head >= n_kv_heads).
    if (head < n_kv_heads) {
        float* xr = k + ((long long)row * n_kv_heads + head) * head_dim;
        const float* wr = kw + (long long)head * head_dim;
        __shared__ float redk[256];
        float ss = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) ss += xr[i] * xr[i];
        redk[threadIdx.x] = ss;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) redk[threadIdx.x] += redk[threadIdx.x + s];
            __syncthreads();
        }
        float inv = rsqrtf(redk[0] / (float)head_dim + eps);
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            xr[i] = xr[i] * inv * wr[i];
        }
    }
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

/// MLA: merge q_nope `[heads*nope]` and q_rope `[heads*rope]` into
/// q `[heads*(nope+rope)]`.
const MLA_ASSEMBLE_Q: &str = r#"
extern "C" __global__ void mla_assemble_q(const float* __restrict__ q_nope,
                                          const float* __restrict__ q_rope,
                                          float* __restrict__ q,
                                          int n_heads, int nope, int rope) {
    int hd = nope + rope;
    int total = n_heads * hd;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int h = i / hd;
        int dd = i % hd;
        q[i] = (dd < nope) ? q_nope[h * nope + dd] : q_rope[h * rope + (dd - nope)];
    }
}
"#;

/// MLA: expand `kv [heads*(nope+v_hd)]` + shared `k_rope [rope]` into per-head
/// k `[heads*(nope+rope)]` and v `[heads*v_hd]`, stored at `*pos` in caches.
const MLA_ASSEMBLE_KV: &str = r#"
extern "C" __global__ void mla_assemble_kv(const float* __restrict__ kv,
                                           const float* __restrict__ k_rope,
                                           float* __restrict__ kc,
                                           float* __restrict__ vc,
                                           const int* __restrict__ pos_buf,
                                           int n_heads, int nope, int rope, int v_hd) {
    int pos = *pos_buf;
    int hd = nope + rope;
    int total = n_heads * hd;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int h = i / hd;
        int dd = i % hd;
        float val = (dd < nope) ? kv[h * (nope + v_hd) + dd] : k_rope[dd - nope];
        kc[(long long)pos * n_heads * hd + i] = val;
    }
    int total_v = n_heads * v_hd;
    if (i < total_v) {
        int h = i / v_hd;
        int dd = i % v_hd;
        vc[(long long)pos * n_heads * v_hd + i] = kv[h * (nope + v_hd) + nope + dd];
    }
}
"#;

/// MLA decode attention over the expanded per-head k/v caches (k head_dim =
/// qk_nope+qk_rope, v head_dim = v_head_dim; one block per head).
const MLA_ATTN_DECODE: &str = r#"
extern "C" __global__ void mla_attn_decode(
    const float* __restrict__ q,
    const float* __restrict__ kc,
    const float* __restrict__ vc,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads, int k_hd, int v_hd, float scale) {
    extern __shared__ float smem[];
    float* scores = smem;
    float* red = smem + 1024;
    int h = blockIdx.x;
    const float* qh = q + h * k_hd;
    int pos = *pos_buf;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        const float* kp = kc + (long long)p * n_heads * k_hd + h * k_hd;
        float s = 0.0f;
        for (int d = 0; d < k_hd; d++) s += qh[d] * kp[d];
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
    for (int d = threadIdx.x; d < v_hd; d += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p <= pos; p++) {
            float vp = vc[(long long)p * n_heads * v_hd + h * v_hd + d];
            acc += __expf(scores[p] - m) * vp;
        }
        out[h * v_hd + d] = acc / ssum;
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

/// Paged KV store: writes one token's `[kv_heads, head_dim]` K/V row into a
/// page pool at `(physical page, in-page offset)` resolved via the block table
/// — the store counterpart of `attn_decode_paged`. Wired into `batched.rs` in
/// the paged-KV integration batch.
#[allow(dead_code)]
const KV_STORE_PAGED: &str = r#"
extern "C" __global__ void kv_store_paged(const float* kv, float* pool,
                                          const int* pos_buf,
                                          const int* table_offsets,
                                          const int* block_tables,
                                          int batch, int kv_heads, int head_dim,
                                          int tokens_per_page) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_heads * head_dim;
    if (idx < total) {
        int s = idx / (kv_heads * head_dim);
        int i = idx % (kv_heads * head_dim);
        int p = pos_buf[s];
        int logical = p / tokens_per_page;
        int off = p % tokens_per_page;
        int page = block_tables[table_offsets[s] + logical];
        pool[((long long)page * tokens_per_page + off) * kv_heads * head_dim + i] =
            kv[(long long)s * kv_heads * head_dim + i];
    }
}
"#;

/// Paged decode attention: reads the KV prefix `0..=pos` through per-request
/// block tables from a page pool (`[num_pages, tokens_per_page, kv_heads,
/// head_dim]`), enabling cross-request prefix sharing. Mirrors the CPU
/// reference in `paged_kv.rs`; covered by the offline hiprtc compile gate.
/// Not yet wired into `batched.rs` (the paged-KV integration batch), so it is
#[allow(dead_code)] // wired into batched.rs in the paged-KV integration batch
const ATTN_DECODE_PAGED: &str = r#"
extern "C" __global__ void attn_decode_paged(
    const float* __restrict__ q,
    const float* __restrict__ k_pool,
    const float* __restrict__ v_pool,
    const int* __restrict__ block_tables,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    const int* __restrict__ table_offsets,
    int batch, int n_heads, int n_kv_heads, int head_dim,
    float scale, int tokens_per_page, int max_pages) {
    extern __shared__ float smem[];
    float* scores = smem;
    float* red = smem + max_pages * tokens_per_page;

    int s = blockIdx.x / n_heads;
    int h = blockIdx.x % n_heads;
    int groups = n_heads / n_kv_heads;
    int kv = h / groups;
    int pos = pos_buf[s];
    const int* table = block_tables + table_offsets[s];
    const float* qh = q + ((long long)s * n_heads + h) * head_dim;

    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        int logical = p / tokens_per_page;
        int off = p % tokens_per_page;
        int page = table[logical];
        const float* kp = k_pool + ((long long)page * tokens_per_page + off) * n_kv_heads * head_dim + kv * head_dim;
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
            int logical = p / tokens_per_page;
            int off = p % tokens_per_page;
            int page = table[logical];
            const float* vp = v_pool + ((long long)page * tokens_per_page + off) * n_kv_heads * head_dim + kv * head_dim + dd;
            acc += __expf(scores[p] - m) * (*vp);
        }
        out[((long long)s * n_heads + h) * head_dim + dd] = acc / ssum;
    }
}
"#;

/// Paged KV store (f16): same signature/addressing as `kv_store_paged`, but
/// the page pool holds f16 bit patterns (`unsigned short`) instead of f32 —
/// the store counterpart of `attn_decode_paged_f16_gqa`. Converts each f32 K/V
/// element to f16 before writing, following `kv_store_batched_f16`.
#[allow(dead_code)] // wired into batched.rs in the paged-KV integration batch
const KV_STORE_PAGED_F16: &str = r#"
__device__ inline unsigned short f32_to_f16_bits(float x) {
    _Float16 h = (_Float16)x;
    union { _Float16 h; unsigned short u; } c;
    c.h = h;
    return c.u;
}

extern "C" __global__ void kv_store_paged_f16(const float* kv,
                                              unsigned short* pool,
                                              const int* pos_buf,
                                              const int* table_offsets,
                                              const int* block_tables,
                                              int batch, int kv_heads, int head_dim,
                                              int tokens_per_page) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_heads * head_dim;
    if (idx < total) {
        int s = idx / (kv_heads * head_dim);
        int i = idx % (kv_heads * head_dim);
        int p = pos_buf[s];
        int logical = p / tokens_per_page;
        int off = p % tokens_per_page;
        int page = block_tables[table_offsets[s] + logical];
        pool[((long long)page * tokens_per_page + off) * kv_heads * head_dim + i] =
            f32_to_f16_bits(kv[(long long)s * kv_heads * head_dim + i]);
    }
}
"#;

/// Paged decode attention (f16 KV): same signature/addressing as
/// `attn_decode_paged`, but k_pool/v_pool hold f16 bit patterns
/// (`unsigned short`, `[num_pages, tokens_per_page, kv_heads, head_dim]`) that
/// are read back to f32 for the softmax math, following the f16 reads in
/// `attn_decode_batched_f16_gqa`. Covered by the offline hiprtc compile gate.
/// Not yet wired into `batched.rs` (the paged-KV integration batch), so it is
#[allow(dead_code)] // wired into batched.rs in the paged-KV integration batch
const ATTN_DECODE_PAGED_F16_GQA: &str = r#"
__device__ inline float f16_bits_to_f32(unsigned short u) {
    union { _Float16 h; unsigned short u; } c;
    c.u = u;
    return (float)c.h;
}

extern "C" __global__ void attn_decode_paged_f16_gqa(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_pool,
    const unsigned short* __restrict__ v_pool,
    const int* __restrict__ block_tables,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    const int* __restrict__ table_offsets,
    int batch, int n_heads, int n_kv_heads, int head_dim,
    float scale, int tokens_per_page, int max_pages) {
    extern __shared__ float smem[];
    float* scores = smem;
    float* red = smem + max_pages * tokens_per_page;

    int s = blockIdx.x / n_heads;
    int h = blockIdx.x % n_heads;
    int groups = n_heads / n_kv_heads;
    int kv = h / groups;
    int pos = pos_buf[s];
    const int* table = block_tables + table_offsets[s];
    const float* qh = q + ((long long)s * n_heads + h) * head_dim;

    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        int logical = p / tokens_per_page;
        int off = p % tokens_per_page;
        int page = table[logical];
        const unsigned short* kp = k_pool + ((long long)page * tokens_per_page + off) * n_kv_heads * head_dim + kv * head_dim;
        float sc = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) sc += qh[dd] * f16_bits_to_f32(kp[dd]);
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
            int logical = p / tokens_per_page;
            int off = p % tokens_per_page;
            int page = table[logical];
            const unsigned short* vp = v_pool + ((long long)page * tokens_per_page + off) * n_kv_heads * head_dim + kv * head_dim + dd;
            acc += __expf(scores[p] - m) * f16_bits_to_f32(*vp);
        }
        out[((long long)s * n_heads + h) * head_dim + dd] = acc / ssum;
    }
}
"#;

/// MLA batched: merge q_nope / q_rope into q across `batch` rows.
const MLA_ASSEMBLE_Q_BATCHED: &str = r#"
extern "C" __global__ void mla_assemble_q_batched(
    const float* __restrict__ q_nope, const float* __restrict__ q_rope,
    float* __restrict__ q, int batch, int n_heads, int nope, int rope) {
    int hd = nope + rope;
    int total = batch * n_heads * hd;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int s = i / (n_heads * hd);
        int rem = i % (n_heads * hd);
        int h = rem / hd;
        int dd = rem % hd;
        q[i] = (dd < nope)
            ? q_nope[(long long)s * n_heads * nope + h * nope + dd]
            : q_rope[(long long)s * n_heads * rope + h * rope + (dd - nope)];
    }
}
"#;

/// MLA batched: extract the latent columns from kv_a `[batch, kv_lora+rope]`
/// into a contiguous `[batch, kv_lora]` buffer (rms_norm needs contiguous rows;
/// the latent is followed by the k_rope columns in kv_a).
const MLA_EXTRACT_KV_LORA: &str = r#"
extern "C" __global__ void mla_extract_kv_lora(const float* __restrict__ kv_a,
                                               float* __restrict__ out,
                                               int batch, int kv_lora_rank, int rope) {
    int total = batch * kv_lora_rank;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int s = i / kv_lora_rank;
        int dd = i % kv_lora_rank;
        out[i] = kv_a[(long long)s * (kv_lora_rank + rope) + dd];
    }
}
"#;

/// MLA batched: extract the shared k_rope columns from kv_a
/// `[batch, kv_lora + rope]` into a contiguous `[batch, rope]` buffer.
const MLA_EXTRACT_K_ROPE: &str = r#"
extern "C" __global__ void mla_extract_k_rope(const float* __restrict__ kv_a,
                                              float* __restrict__ k_rope,
                                              int batch, int kv_lora, int rope) {
    int total = batch * rope;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int s = i / rope;
        int dd = i % rope;
        k_rope[i] = kv_a[(long long)s * (kv_lora + rope) + kv_lora + dd];
    }
}
"#;

/// MLA batched: expand kv + shared k_rope into per-head k/v caches at the
/// per-row (slot, pos) position.
const MLA_ASSEMBLE_KV_BATCHED: &str = r#"
extern "C" __global__ void mla_assemble_kv_batched(
    const float* __restrict__ kv, const float* __restrict__ k_rope,
    float* __restrict__ kc, float* __restrict__ vc,
    const int* __restrict__ pos_buf, const int* __restrict__ slots,
    int batch, int max_seq, int n_heads, int nope, int rope, int v_hd) {
    int hd = nope + rope;
    int total = batch * n_heads * hd;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int s = i / (n_heads * hd);
        int rem = i % (n_heads * hd);
        int h = rem / hd;
        int dd = rem % hd;
        int slot = slots[s];
        int pos = pos_buf[s];
        float val = (dd < nope)
            ? kv[(long long)s * n_heads * (nope + v_hd) + h * (nope + v_hd) + dd]
            : k_rope[(long long)s * rope + (dd - nope)];
        kc[((long long)slot * max_seq + pos) * n_heads * hd + h * hd + dd] = val;
    }
    int total_v = batch * n_heads * v_hd;
    if (i < total_v) {
        int s = i / (n_heads * v_hd);
        int rem = i % (n_heads * v_hd);
        int h = rem / v_hd;
        int dd = rem % v_hd;
        int slot = slots[s];
        int pos = pos_buf[s];
        vc[((long long)slot * max_seq + pos) * n_heads * v_hd + h * v_hd + dd] =
            kv[(long long)s * n_heads * (nope + v_hd) + h * (nope + v_hd) + nope + dd];
    }
}
"#;

/// MLA batched decode attention over the expanded per-head k/v caches.
const MLA_ATTN_DECODE_BATCHED: &str = r#"
extern "C" __global__ void mla_attn_decode_batched(
    const float* __restrict__ q, const float* __restrict__ kc,
    const float* __restrict__ vc, float* __restrict__ out,
    const int* __restrict__ pos_buf, const int* __restrict__ slots,
    int batch, int n_heads, int k_hd, int v_hd, float scale, int max_seq) {
    extern __shared__ float smem[];
    float* scores = smem;
    float* red = smem + max_seq;
    int s = blockIdx.x / n_heads;
    int h = blockIdx.x % n_heads;
    int slot = slots[s];
    int pos = pos_buf[s];
    const float* qh = q + ((long long)s * n_heads + h) * k_hd;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        const float* kp = kc + ((long long)slot * max_seq + p) * n_heads * k_hd + h * k_hd;
        float sc = 0.0f;
        for (int dd = 0; dd < k_hd; dd++) sc += qh[dd] * kp[dd];
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
    for (int dd = threadIdx.x; dd < v_hd; dd += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p <= pos; p++) {
            float vp = vc[((long long)slot * max_seq + p) * n_heads * v_hd + h * v_hd + dd];
            acc += __expf(scores[p] - m) * vp;
        }
        out[((long long)s * n_heads + h) * v_hd + dd] = acc / ssum;
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

/// Batched MoE routing: one block per token, softmax + top-k with the same
/// tie-breaking (lower index) as the CPU reference. Outputs `out_ids[B, topk]`
/// and normalized `out_w[B, topk]`.
const MOE_ROUTER_BATCHED: &str = r#"
extern "C" __global__ void moe_router_batched(
    const float* __restrict__ logits,
    int* __restrict__ out_ids,
    float* __restrict__ out_w,
    int ne, int topk, int batch) {
    int s = blockIdx.x;
    const float* lg = logits + (long long)s * ne;
    int* ids = out_ids + (long long)s * topk;
    float* w = out_w + (long long)s * topk;
    extern __shared__ float sm[];
    float* probs = sm;
    __shared__ float red[256];
    float maxv = -1e30f;
    for (int i = threadIdx.x; i < ne; i += blockDim.x) maxv = fmaxf(maxv, lg[i]);
    red[threadIdx.x] = maxv;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
        __syncthreads();
    }
    float m = red[0];
    float sumv = 0.0f;
    for (int i = threadIdx.x; i < ne; i += blockDim.x) {
        probs[i] = __expf(lg[i] - m);
        sumv += probs[i];
    }
    red[threadIdx.x] = sumv;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
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
                    if (ids[j] == i) { used = 1; break; }
                }
                if (!used) {
                    if (probs[i] > bestv || (probs[i] == bestv && (best < 0 || i < best))) {
                        bestv = probs[i];
                        best = i;
                    }
                }
            }
            ids[k] = best;
            norm += probs[best];
        }
        for (int k = 0; k < topk; k++) w[k] = probs[ids[k]] / norm;
    }
}
"#;

/// Batched MoE: count how many (token, slot) pairs route to each expert via
/// atomicAdd on `counts[ne]` (must be zeroed before launch).
const MOE_COUNT_EXPERTS: &str = r#"
extern "C" __global__ void moe_count_experts(
    const int* __restrict__ ids,
    int* __restrict__ counts,
    int batch, int topk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * topk;
    if (i < total) {
        int e = ids[i];
        atomicAdd(&counts[e], 1);
    }
}
"#;

/// Batched MoE: gather routed rows into a packed per-expert layout. For each
/// (token, slot) pair routed to expert e, atomically claims position `j` in
/// expert e's segment and copies the token row from `x[B, d]` into
/// `xg[B*topk, d]` at `offsets[e] + j`, recording the source row and weight.
const MOE_GATHER_ROWS: &str = r#"
extern "C" __global__ void moe_gather_rows(
    const float* __restrict__ x,
    const int* __restrict__ ids,
    const float* __restrict__ w,
    const int* __restrict__ offsets,
    int* __restrict__ pos,
    float* __restrict__ xg,
    float* __restrict__ gw,
    int* __restrict__ row_idx,
    int batch, int topk, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * topk;
    if (i < total) {
        int t = i / topk;
        int k = i % topk;
        int e = ids[i];
        int j = atomicAdd(&pos[e], 1);
        int dst = offsets[e] + j;
        row_idx[dst] = t;
        gw[dst] = w[i];
        const float* src = x + (long long)t * d;
        float* dstp = xg + (long long)dst * d;
        for (int c = 0; c < d; c++) dstp[c] = src[c];
    }
}
"#;

/// Batched MoE: `h_acc[src] += gw[j] * down[j]` for a packed expert segment.
/// `row_idx`/`gw`/`down` are already offset to the segment base.
const MOE_SCATTER_ADD: &str = r#"
extern "C" __global__ void moe_scatter_add(
    float* __restrict__ h_acc,
    const int* __restrict__ row_idx,
    const float* __restrict__ gw,
    const float* __restrict__ down,
    int cnt, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = cnt * d;
    if (i < total) {
        int j = i / d;
        int c = i % d;
        int t = row_idx[j];
        h_acc[(long long)t * d + c] += gw[j] * down[(long long)j * d + c];
    }
}
"#;

/// Exclusive prefix sum over `counts[ne]` -> `offsets[ne]` (single block;
/// `ne <= 256` covers all real MoE expert counts). GPU-side replacement for
/// the host round-trip that computed gather offsets.
const MOE_PREFIX_SUM: &str = r#"
extern "C" __global__ void moe_prefix_sum(const int* counts, int* offsets, int ne) {
    __shared__ int sm[256];
    int i = threadIdx.x;
    int v = (i < ne) ? counts[i] : 0;
    sm[i] = v;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
        int t = 0;
        if (i >= off) t = sm[i - off];
        __syncthreads();
        sm[i] = sm[i] + t;
        __syncthreads();
    }
    if (i < ne) {
        offsets[i] = sm[i] - v; // exclusive prefix
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
    qk_norm: HipKernelModule,
    silu_mul: HipKernelModule,
    add: HipKernelModule,
    add_bias: HipKernelModule,
    kv_store: HipKernelModule,
    attn_decode: HipKernelModule,
    mla_assemble_q: HipKernelModule,
    mla_assemble_kv: HipKernelModule,
    mla_attn_decode: HipKernelModule,
    mla_assemble_q_batched: HipKernelModule,
    mla_extract_kv_lora: HipKernelModule,
    mla_extract_k_rope: HipKernelModule,
    mla_assemble_kv_batched: HipKernelModule,
    mla_attn_decode_batched: HipKernelModule,
    rope: HipKernelModule,
    embed_batched: HipKernelModule,
    rope_batched: HipKernelModule,
    kv_store_batched: HipKernelModule,
    attn_decode_batched: HipKernelModule,
    kv_store_paged: HipKernelModule,
    attn_decode_paged: HipKernelModule,
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
    moe_router_batched: HipKernelModule,
    moe_count: HipKernelModule,
    moe_gather_rows: HipKernelModule,
    moe_scatter_add: HipKernelModule,
    moe_prefix_sum: HipKernelModule,
}

// SAFETY: a HipKernels instance is used by one model on one thread; the raw
// stream handle is only touched there, and the loaded runtimes are Send+Sync.
unsafe impl Send for HipKernels {}
unsafe impl Sync for HipKernels {}

/// A compiled kernel module shared across model loads. `HipKernelModule` is
/// refcounted: the module unloads only when the last reference drops. The
/// cache holds one reference for the process lifetime, so cached modules stay
/// loaded; launches are per-stream, so sharing handles across threads is safe
/// (same rationale as the `unsafe impl Send/Sync for HipKernels` below).
#[derive(Clone)]
struct CachedModule(HipKernelModule);

// SAFETY: the wrapped module is refcounted via `HipKernelModule`'s internal
// `Arc`; the cache keeps one reference alive and launches are per-stream.
unsafe impl Send for CachedModule {}
unsafe impl Sync for CachedModule {}

/// In-process hiprtc compile cache, keyed by `(arch, source)`. Loading several
/// models in one process (spec-decode draft+target, server, tests) previously
/// recompiled every kernel per model (~36 serial hiprtc compiles each time);
/// the cache makes the second and later model loads reuse the compiled modules.
static KERNEL_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(String, &'static str), CachedModule>>,
> = std::sync::OnceLock::new();

/// Compiles `source` for `arch`, or returns a cached clone when this process
/// already compiled the same source. Prints per-kernel progress to stderr when
/// `MACH_COMPILE_PROGRESS` is set.
fn compile_cached(arch: &str, source: &'static str, name: &str) -> Result<HipKernelModule, Error> {
    let verbose = std::env::var_os("MACH_COMPILE_PROGRESS").is_some();
    let cache =
        KERNEL_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let key = (arch.to_string(), source);
    if let Some(m) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        if verbose {
            eprintln!("[hiprtc] {name}: cache hit");
        }
        return Ok(m.0.clone());
    }
    if verbose {
        eprintln!("[hiprtc] compiling {name} for {arch} ...");
    }
    let m = HipKernelModule::compile(arch, source, name)?;
    if verbose {
        eprintln!("[hiprtc] {name}: compiled");
    }
    cache
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key, CachedModule(m.clone()));
    Ok(m)
}

impl HipKernels {
    /// Compiles all kernels and initializes hipBLAS on a fresh stream.
    pub fn new(hip: std::sync::Arc<Hip>) -> Result<Self, Error> {
        let arch = hip_arch();
        let t0 = std::time::Instant::now();
        let mut stream = std::ptr::null_mut();
        unsafe { hip::check(&hip, (hip.api.hip_stream_create)(&mut stream))? };

        let blas = mach_kernel_sys::hipblas::HipBlas::new(std::sync::Arc::clone(&hip))?;
        blas.set_stream(stream)?;

        let self_ = Self {
            hip,
            stream,
            blas,
            embed: compile_cached(&arch, EMBED_GATHER, "embed_gather")?,
            rms_norm: compile_cached(&arch, RMS_NORM, "rms_norm")?,
            qk_norm: compile_cached(&arch, QK_NORM, "qk_norm")?,
            silu_mul: compile_cached(&arch, SILU_MUL, "silu_mul")?,
            add: compile_cached(&arch, ADD, "add")?,
            add_bias: compile_cached(&arch, ADD_BIAS, "add_bias")?,
            kv_store: compile_cached(&arch, KV_STORE, "kv_store")?,
            attn_decode: compile_cached(&arch, ATTN_DECODE, "attn_decode")?,
            mla_assemble_q: compile_cached(&arch, MLA_ASSEMBLE_Q, "mla_assemble_q")?,
            mla_assemble_kv: compile_cached(&arch, MLA_ASSEMBLE_KV, "mla_assemble_kv")?,
            mla_attn_decode: compile_cached(&arch, MLA_ATTN_DECODE, "mla_attn_decode")?,
            mla_assemble_q_batched: compile_cached(
                &arch,
                MLA_ASSEMBLE_Q_BATCHED,
                "mla_assemble_q_batched",
            )?,
            mla_extract_kv_lora: compile_cached(&arch, MLA_EXTRACT_KV_LORA, "mla_extract_kv_lora")?,
            mla_extract_k_rope: compile_cached(&arch, MLA_EXTRACT_K_ROPE, "mla_extract_k_rope")?,
            mla_assemble_kv_batched: compile_cached(
                &arch,
                MLA_ASSEMBLE_KV_BATCHED,
                "mla_assemble_kv_batched",
            )?,
            mla_attn_decode_batched: compile_cached(
                &arch,
                MLA_ATTN_DECODE_BATCHED,
                "mla_attn_decode_batched",
            )?,
            rope: compile_cached(&arch, ROPE, "rope")?,
            embed_batched: compile_cached(&arch, EMBED_BATCHED, "embed_batched")?,
            rope_batched: compile_cached(&arch, ROPE_BATCHED, "rope_batched")?,
            kv_store_batched: compile_cached(&arch, KV_STORE_BATCHED, "kv_store_batched")?,
            attn_decode_batched: compile_cached(&arch, ATTN_DECODE_BATCHED, "attn_decode_batched")?,
            kv_store_paged: compile_cached(&arch, KV_STORE_PAGED, "kv_store_paged")?,
            attn_decode_paged: compile_cached(&arch, ATTN_DECODE_PAGED, "attn_decode_paged")?,
            argmax_batched: compile_cached(&arch, ARGMAX_BATCHED, "argmax_batched")?,
            cast_f32_f16: compile_cached(&arch, CAST_F32_F16, "cast_f32_f16")?,
            cast_f16_f32: compile_cached(&arch, CAST_F16_F32, "cast_f16_f32")?,
            embed_f16: compile_cached(&arch, EMBED_GATHER_F16, "embed_gather_f16")?,
            kv_store_f16: compile_cached(&arch, KV_F16, "kv_store_batched_f16")?,
            attn_f16_gqa: compile_cached(
                &arch,
                ATTN_DECODE_BATCHED_F16_GQA,
                "attn_decode_batched_f16_gqa",
            )?,
            attn_prefill_f16: compile_cached(&arch, ATTN_PREFILL_F16, "attn_prefill_f16")?,
            moe_router: compile_cached(&arch, MOE_ROUTER, "moe_router")?,
            moe_gather: compile_cached(&arch, MOE_GATHER_WEIGHTS, "moe_gather_weights")?,
            moe_accumulate: compile_cached(&arch, MOE_ACCUMULATE, "moe_accumulate")?,
            moe_router_batched: compile_cached(&arch, MOE_ROUTER_BATCHED, "moe_router_batched")?,
            moe_count: compile_cached(&arch, MOE_COUNT_EXPERTS, "moe_count_experts")?,
            moe_gather_rows: compile_cached(&arch, MOE_GATHER_ROWS, "moe_gather_rows")?,
            moe_scatter_add: compile_cached(&arch, MOE_SCATTER_ADD, "moe_scatter_add")?,
            moe_prefix_sum: compile_cached(&arch, MOE_PREFIX_SUM, "moe_prefix_sum")?,
        };
        if std::env::var_os("MACH_COMPILE_PROGRESS").is_some() {
            let n = KERNEL_CACHE
                .get()
                .map(|c| c.lock().unwrap_or_else(|p| p.into_inner()).len())
                .unwrap_or(0);
            eprintln!(
                "[hiprtc] {n} kernels cached, ready in {:.2}s",
                t0.elapsed().as_secs_f64()
            );
        }
        Ok(self_)
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

    /// Per-head RMSNorm on q and k (Qwen3 QK-norm), in place.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_qk_norm(
        &self,
        q: *mut f32,
        k: *mut f32,
        qw: *const f32,
        kw: *const f32,
        rows: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<(), Error> {
        let qp = q;
        let kp = k;
        let qwp = qw;
        let kwp = kw;
        let mut p = vec![
            &qp as *const *mut f32 as *mut core::ffi::c_void,
            &kp as *const *mut f32 as *mut core::ffi::c_void,
            &qwp as *const *const f32 as *mut core::ffi::c_void,
            &kwp as *const *const f32 as *mut core::ffi::c_void,
            &rows as *const i32 as *mut core::ffi::c_void,
            &n_heads as *const i32 as *mut core::ffi::c_void,
            &n_kv_heads as *const i32 as *mut core::ffi::c_void,
            &head_dim as *const i32 as *mut core::ffi::c_void,
            &eps as *const f32 as *mut core::ffi::c_void,
        ];
        Ok(self.qk_norm.launch(
            [rows as u32, n_heads as u32, 1],
            [256, 1, 1],
            &mut p,
            self.stream,
        )?)
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

    /// MLA: merge q_nope + q_rope into q `[heads*(nope+rope)]`.
    pub fn launch_mla_assemble_q(
        &self,
        q_nope: *const f32,
        q_rope: *const f32,
        q: *mut f32,
        n_heads: i32,
        nope: i32,
        rope: i32,
    ) -> Result<(), Error> {
        let qnp = q_nope;
        let qrp = q_rope;
        let qp = q;
        let mut p = vec![
            &qnp as *const *const f32 as *mut core::ffi::c_void,
            &qrp as *const *const f32 as *mut core::ffi::c_void,
            &qp as *const *mut f32 as *mut core::ffi::c_void,
            &n_heads as *const i32 as *mut core::ffi::c_void,
            &nope as *const i32 as *mut core::ffi::c_void,
            &rope as *const i32 as *mut core::ffi::c_void,
        ];
        let total = (n_heads * (nope + rope)) as u32;
        let blocks = total.div_ceil(256);
        Ok(self
            .mla_assemble_q
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// MLA: expand kv + shared k_rope into per-head k/v caches at `*pos`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_mla_assemble_kv(
        &self,
        kv: *const f32,
        k_rope: *const f32,
        kc: *mut f32,
        vc: *mut f32,
        pos: *const i32,
        n_heads: i32,
        nope: i32,
        rope: i32,
        v_hd: i32,
    ) -> Result<(), Error> {
        let kvp = kv;
        let krp = k_rope;
        let kcp = kc;
        let vcp = vc;
        let pp = pos;
        let mut p = vec![
            &kvp as *const *const f32 as *mut core::ffi::c_void,
            &krp as *const *const f32 as *mut core::ffi::c_void,
            &kcp as *const *mut f32 as *mut core::ffi::c_void,
            &vcp as *const *mut f32 as *mut core::ffi::c_void,
            &pp as *const *const i32 as *mut core::ffi::c_void,
            &n_heads as *const i32 as *mut core::ffi::c_void,
            &nope as *const i32 as *mut core::ffi::c_void,
            &rope as *const i32 as *mut core::ffi::c_void,
            &v_hd as *const i32 as *mut core::ffi::c_void,
        ];
        let total = (n_heads * (nope + rope)).max(n_heads * v_hd) as u32;
        let blocks = total.div_ceil(256);
        Ok(self
            .mla_assemble_kv
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// MLA decode attention over expanded per-head k/v caches.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_mla_attn_decode(
        &self,
        q: *const f32,
        kc: *const f32,
        vc: *const f32,
        out: *mut f32,
        pos: *const i32,
        n_heads: i32,
        k_hd: i32,
        v_hd: i32,
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
            &k_hd as *const i32 as *mut core::ffi::c_void,
            &v_hd as *const i32 as *mut core::ffi::c_void,
            &scale as *const f32 as *mut core::ffi::c_void,
        ];
        const SHARED_FLOATS: u32 = 1024 + 256;
        Ok(self.mla_attn_decode.launch_shmem(
            [n_heads as u32, 1, 1],
            [256, 1, 1],
            &mut p,
            self.stream,
            SHARED_FLOATS * 4,
        )?)
    }

    /// MLA batched: merge q_nope / q_rope into q across rows.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_mla_assemble_q_batched(
        &self,
        q_nope: *const f32,
        q_rope: *const f32,
        q: *mut f32,
        batch: i32,
        n_heads: i32,
        nope: i32,
        rope: i32,
    ) -> Result<(), Error> {
        let qnp = q_nope;
        let qrp = q_rope;
        let qp = q;
        let mut p = vec![
            &qnp as *const *const f32 as *mut core::ffi::c_void,
            &qrp as *const *const f32 as *mut core::ffi::c_void,
            &qp as *const *mut f32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &n_heads as *const i32 as *mut core::ffi::c_void,
            &nope as *const i32 as *mut core::ffi::c_void,
            &rope as *const i32 as *mut core::ffi::c_void,
        ];
        let total = (batch * n_heads * (nope + rope)) as u32;
        let blocks = total.div_ceil(256);
        Ok(self
            .mla_assemble_q_batched
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// MLA batched: extract the latent columns into a contiguous buffer.
    pub fn launch_mla_extract_kv_lora(
        &self,
        kv_a: *const f32,
        kv_lora: *mut f32,
        batch: i32,
        kv_lora_rank: i32,
        rope: i32,
    ) -> Result<(), Error> {
        let kvap = kv_a;
        let kvlp = kv_lora;
        let mut p = vec![
            &kvap as *const *const f32 as *mut core::ffi::c_void,
            &kvlp as *const *mut f32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &kv_lora_rank as *const i32 as *mut core::ffi::c_void,
            &rope as *const i32 as *mut core::ffi::c_void,
        ];
        let total = (batch * kv_lora_rank) as u32;
        let blocks = total.div_ceil(256);
        Ok(self
            .mla_extract_kv_lora
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// MLA batched: extract shared k_rope columns into a contiguous buffer.
    pub fn launch_mla_extract_k_rope(
        &self,
        kv_a: *const f32,
        k_rope: *mut f32,
        batch: i32,
        kv_lora: i32,
        rope: i32,
    ) -> Result<(), Error> {
        let kvap = kv_a;
        let krp = k_rope;
        let mut p = vec![
            &kvap as *const *const f32 as *mut core::ffi::c_void,
            &krp as *const *mut f32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &kv_lora as *const i32 as *mut core::ffi::c_void,
            &rope as *const i32 as *mut core::ffi::c_void,
        ];
        let total = (batch * rope) as u32;
        let blocks = total.div_ceil(256);
        Ok(self
            .mla_extract_k_rope
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// MLA batched: expand kv + k_rope into per-head k/v caches.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_mla_assemble_kv_batched(
        &self,
        kv: *const f32,
        k_rope: *const f32,
        kc: *mut f32,
        vc: *mut f32,
        pos: *const i32,
        slots: *const i32,
        batch: i32,
        max_seq: i32,
        n_heads: i32,
        nope: i32,
        rope: i32,
        v_hd: i32,
    ) -> Result<(), Error> {
        let kvp = kv;
        let krp = k_rope;
        let kcp = kc;
        let vcp = vc;
        let pp = pos;
        let sp = slots;
        let mut p = vec![
            &kvp as *const *const f32 as *mut core::ffi::c_void,
            &krp as *const *const f32 as *mut core::ffi::c_void,
            &kcp as *const *mut f32 as *mut core::ffi::c_void,
            &vcp as *const *mut f32 as *mut core::ffi::c_void,
            &pp as *const *const i32 as *mut core::ffi::c_void,
            &sp as *const *const i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &max_seq as *const i32 as *mut core::ffi::c_void,
            &n_heads as *const i32 as *mut core::ffi::c_void,
            &nope as *const i32 as *mut core::ffi::c_void,
            &rope as *const i32 as *mut core::ffi::c_void,
            &v_hd as *const i32 as *mut core::ffi::c_void,
        ];
        let total = (batch * n_heads * (nope + rope)).max(batch * n_heads * v_hd) as u32;
        let blocks = total.div_ceil(256);
        Ok(self
            .mla_assemble_kv_batched
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// MLA batched decode attention over expanded per-head k/v caches.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_mla_attn_decode_batched(
        &self,
        q: *const f32,
        kc: *const f32,
        vc: *const f32,
        out: *mut f32,
        pos: *const i32,
        slots: *const i32,
        batch: i32,
        n_heads: i32,
        k_hd: i32,
        v_hd: i32,
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
            &k_hd as *const i32 as *mut core::ffi::c_void,
            &v_hd as *const i32 as *mut core::ffi::c_void,
            &scale as *const f32 as *mut core::ffi::c_void,
            &max_seq as *const i32 as *mut core::ffi::c_void,
        ];
        let grid = (batch * n_heads) as u32;
        let shared = (max_seq as u32 + 256) * 4;
        Ok(self.mla_attn_decode_batched.launch_shmem(
            [grid, 1, 1],
            [256, 1, 1],
            &mut p,
            self.stream,
            shared,
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

    #[allow(clippy::too_many_arguments)]
    /// Paged KV store (batched): writes per-row K/V into a page pool via the
    /// block table, for the paged decode path.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_kv_store_paged(
        &self,
        kv: *const f32,
        pool: *mut f32,
        pos: *const i32,
        table_offsets: *const i32,
        block_tables: *const i32,
        batch: i32,
        kv_heads: i32,
        head_dim: i32,
        tokens_per_page: i32,
    ) -> Result<(), Error> {
        let kvp = kv;
        let pp = pool;
        let posp = pos;
        let toff = table_offsets;
        let bt = block_tables;
        let mut p = vec![
            &kvp as *const *const f32 as *mut core::ffi::c_void,
            &pp as *const *mut f32 as *mut core::ffi::c_void,
            &posp as *const *const i32 as *mut core::ffi::c_void,
            &toff as *const *const i32 as *mut core::ffi::c_void,
            &bt as *const *const i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &kv_heads as *const i32 as *mut core::ffi::c_void,
            &head_dim as *const i32 as *mut core::ffi::c_void,
            &tokens_per_page as *const i32 as *mut core::ffi::c_void,
        ];
        let total = (batch * kv_heads * head_dim) as u32;
        let blocks = total.div_ceil(256);
        Ok(self
            .kv_store_paged
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// Paged decode attention (batched): reads the KV prefix through per-row
    /// block tables from the page pool.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attn_decode_paged(
        &self,
        q: *const f32,
        k_pool: *const f32,
        v_pool: *const f32,
        block_tables: *const i32,
        out: *mut f32,
        pos: *const i32,
        table_offsets: *const i32,
        batch: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        tokens_per_page: i32,
        max_pages: i32,
    ) -> Result<(), Error> {
        let qp = q;
        let kp = k_pool;
        let vp = v_pool;
        let bt = block_tables;
        let op = out;
        let pp = pos;
        let toff = table_offsets;
        let mut p = vec![
            &qp as *const *const f32 as *mut core::ffi::c_void,
            &kp as *const *const f32 as *mut core::ffi::c_void,
            &vp as *const *const f32 as *mut core::ffi::c_void,
            &bt as *const *const i32 as *mut core::ffi::c_void,
            &op as *const *mut f32 as *mut core::ffi::c_void,
            &pp as *const *const i32 as *mut core::ffi::c_void,
            &toff as *const *const i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &n_heads as *const i32 as *mut core::ffi::c_void,
            &n_kv_heads as *const i32 as *mut core::ffi::c_void,
            &head_dim as *const i32 as *mut core::ffi::c_void,
            &scale as *const f32 as *mut core::ffi::c_void,
            &tokens_per_page as *const i32 as *mut core::ffi::c_void,
            &max_pages as *const i32 as *mut core::ffi::c_void,
        ];
        let shared = ((max_pages * tokens_per_page + 256) * 4) as u32;
        Ok(self.attn_decode_paged.launch_shmem(
            [(batch * n_heads) as u32, 1, 1],
            [256, 1, 1],
            &mut p,
            self.stream,
            shared,
        )?)
    }

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

    /// Batched MoE router: one block per token, softmax + top-k selection;
    /// writes `out_ids[B, topk]` and normalized `out_w[B, topk]`.
    pub fn launch_moe_router_batched(
        &self,
        logits: *const f32,
        out_ids: *mut i32,
        out_w: *mut f32,
        ne: i32,
        topk: i32,
        batch: i32,
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
            &batch as *const i32 as *mut core::ffi::c_void,
        ];
        Ok(self.moe_router_batched.launch_shmem(
            [batch as u32, 1, 1],
            [256, 1, 1],
            &mut p,
            self.stream,
            (ne as u32) * 4,
        )?)
    }

    /// Batched MoE expert histogram: `counts[e] += 1` per routed (token, slot)
    /// pair (counts must be zeroed before launch).
    pub fn launch_moe_count_experts(
        &self,
        ids: *const i32,
        counts: *mut i32,
        batch: i32,
        topk: i32,
    ) -> Result<(), Error> {
        let ip = ids;
        let cp = counts;
        let mut p = vec![
            &ip as *const *const i32 as *mut core::ffi::c_void,
            &cp as *const *mut i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &topk as *const i32 as *mut core::ffi::c_void,
        ];
        let total = batch * topk;
        let blocks = (total as u32).div_ceil(256);
        Ok(self
            .moe_count
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// Batched MoE row gather: packs routed token rows into a per-expert
    /// contiguous layout using prefix-sum `offsets[ne]` and `pos[ne]` counters
    /// (zeroed before launch).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_moe_gather_rows(
        &self,
        x: *const f32,
        ids: *const i32,
        w: *const f32,
        offsets: *const i32,
        pos: *mut i32,
        xg: *mut f32,
        gw: *mut f32,
        row_idx: *mut i32,
        batch: i32,
        topk: i32,
        d: i32,
    ) -> Result<(), Error> {
        let xp = x;
        let idp = ids;
        let wp = w;
        let op = offsets;
        let pp = pos;
        let xgp = xg;
        let gwp = gw;
        let rip = row_idx;
        let mut p = vec![
            &xp as *const *const f32 as *mut core::ffi::c_void,
            &idp as *const *const i32 as *mut core::ffi::c_void,
            &wp as *const *const f32 as *mut core::ffi::c_void,
            &op as *const *const i32 as *mut core::ffi::c_void,
            &pp as *const *mut i32 as *mut core::ffi::c_void,
            &xgp as *const *mut f32 as *mut core::ffi::c_void,
            &gwp as *const *mut f32 as *mut core::ffi::c_void,
            &rip as *const *mut i32 as *mut core::ffi::c_void,
            &batch as *const i32 as *mut core::ffi::c_void,
            &topk as *const i32 as *mut core::ffi::c_void,
            &d as *const i32 as *mut core::ffi::c_void,
        ];
        let total = batch * topk;
        let blocks = (total as u32).div_ceil(256);
        Ok(self
            .moe_gather_rows
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// Batched MoE scatter-add: `h_acc[src] += gw[j] * down[j]` for a packed
    /// expert segment (`row_idx`/`gw`/`down` pre-offset to the segment base).
    pub fn launch_moe_scatter_add(
        &self,
        h_acc: *mut f32,
        row_idx: *const i32,
        gw: *const f32,
        down: *const f32,
        cnt: i32,
        d: i32,
    ) -> Result<(), Error> {
        let hp = h_acc;
        let rip = row_idx;
        let gwp = gw;
        let dp = down;
        let mut p = vec![
            &hp as *const *mut f32 as *mut core::ffi::c_void,
            &rip as *const *const i32 as *mut core::ffi::c_void,
            &gwp as *const *const f32 as *mut core::ffi::c_void,
            &dp as *const *const f32 as *mut core::ffi::c_void,
            &cnt as *const i32 as *mut core::ffi::c_void,
            &d as *const i32 as *mut core::ffi::c_void,
        ];
        let total = cnt * d;
        let blocks = (total as u32).div_ceil(256);
        Ok(self
            .moe_scatter_add
            .launch([blocks, 1, 1], [256, 1, 1], &mut p, self.stream)?)
    }

    /// Exclusive prefix sum over `counts[ne]` into `offsets[ne]` (single
    /// block, `ne <= 256`).
    pub fn launch_moe_prefix_sum(
        &self,
        counts: *const i32,
        offsets: *mut i32,
        ne: i32,
    ) -> Result<(), Error> {
        let cp = counts;
        let op = offsets;
        let mut p = vec![
            &cp as *const *const i32 as *mut core::ffi::c_void,
            &op as *const *mut i32 as *mut core::ffi::c_void,
            &ne as *const i32 as *mut core::ffi::c_void,
        ];
        Ok(self
            .moe_prefix_sum
            .launch([1, 1, 1], [256, 1, 1], &mut p, self.stream)?)
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

/// Offline hiprtc compile gate: every kernel source must compile for the
/// target arch **without a device** (hiprtc is a device-independent compiler).
/// Needs only the ROCm DLLs present; skips when they are absent (plain CI).
#[cfg(all(test, feature = "hip"))]
mod offline_tests {
    use super::*;
    use mach_kernel_sys::hip::HipKernelModule;

    macro_rules! all_kernels {
        ($($name:ident),* $(,)?) => {
            &[ $( (stringify!($name), $name), )* ]
        };
    }

    const ALL_KERNELS: &[(&str, &str)] = all_kernels![
        EMBED_GATHER,
        RMS_NORM,
        QK_NORM,
        SILU_MUL,
        ADD,
        ADD_BIAS,
        KV_STORE,
        ROPE,
        ATTN_DECODE,
        MLA_ASSEMBLE_Q,
        MLA_ASSEMBLE_KV,
        MLA_ATTN_DECODE,
        EMBED_BATCHED,
        ROPE_BATCHED,
        KV_STORE_BATCHED,
        ATTN_DECODE_BATCHED,
        MLA_ASSEMBLE_Q_BATCHED,
        MLA_EXTRACT_KV_LORA,
        MLA_EXTRACT_K_ROPE,
        MLA_ASSEMBLE_KV_BATCHED,
        MLA_ATTN_DECODE_BATCHED,
        ARGMAX_BATCHED,
        CAST_F32_F16,
        CAST_F16_F32,
        KV_F16,
        ATTN_DECODE_BATCHED_F16_GQA,
        ATTN_PREFILL_F16,
        EMBED_GATHER_F16,
        MOE_ROUTER,
        MOE_GATHER_WEIGHTS,
        MOE_ACCUMULATE,
        MOE_ROUTER_BATCHED,
        MOE_COUNT_EXPERTS,
        MOE_GATHER_ROWS,
        MOE_SCATTER_ADD,
        MOE_PREFIX_SUM,
        KV_STORE_PAGED,
        ATTN_DECODE_PAGED,
        KV_STORE_PAGED_F16,
        ATTN_DECODE_PAGED_F16_GQA,
    ];

    #[test]
    fn all_kernel_sources_compile_offline() {
        if mach_kernel_sys::hip::hip().is_err() {
            eprintln!("skipping: ROCm runtime not available");
            return;
        }
        for (name, src) in ALL_KERNELS {
            let size = HipKernelModule::compile_only("gfx1100", src)
                .unwrap_or_else(|e| panic!("kernel {name} failed offline hiprtc compile: {e}"));
            assert!(size > 0, "kernel {name} produced an empty code object");
        }
    }
}
/// GPU parity: the paged kernels (`kv_store_paged` / `attn_decode_paged`) must
/// produce bit-identical results to the contiguous kernels on the device.
#[cfg(all(test, feature = "hip"))]
mod gpu_tests {
    use super::*;
    use mach_kernel_sys::hip;

    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5
        }
    }

    #[test]
    fn paged_attn_and_store_match_contiguous_gpu() {
        let Ok(h) = hip::hip() else {
            eprintln!("skipping: ROCm runtime not available");
            return;
        };
        if hip::device_count().map(|n| n <= 0).unwrap_or(true) {
            eprintln!("skipping: no HIP device");
            return;
        }
        let arch = "gfx1100";
        let attn_c =
            hip::HipKernelModule::compile(arch, ATTN_DECODE_BATCHED, "attn_decode_batched")
                .expect("compile attn contig");
        let attn_p = hip::HipKernelModule::compile(arch, ATTN_DECODE_PAGED, "attn_decode_paged")
            .expect("compile attn paged");
        let store_c = hip::HipKernelModule::compile(arch, KV_STORE_BATCHED, "kv_store_batched")
            .expect("compile store contig");
        let store_p = hip::HipKernelModule::compile(arch, KV_STORE_PAGED, "kv_store_paged")
            .expect("compile store paged");

        let batch = 1usize;
        let n_heads = 4usize;
        let n_kv_heads = 2usize;
        let head_dim = 8usize;
        let max_seq = 12usize;
        let tpp = 4usize;
        let max_pages = 3usize;
        let pos: i32 = 11; // last position (spans all 3 pages)
        let mut rng = lcg(7);

        // q, a K/V row to store, and the contiguous prefix KV.
        let q: Vec<f32> = (0..batch * n_heads * head_dim).map(|_| rng()).collect();
        let kv_row: Vec<f32> = (0..batch * n_kv_heads * head_dim).map(|_| rng()).collect();
        let kvn = max_seq * n_kv_heads * head_dim;
        let mut kc = vec![0.0f32; kvn];
        let mut vc = vec![0.0f32; kvn];
        for p in 0..max_seq {
            for kvh in 0..n_kv_heads {
                for dd in 0..head_dim {
                    kc[(p * n_kv_heads + kvh) * head_dim + dd] = rng();
                    vc[(p * n_kv_heads + kvh) * head_dim + dd] = rng();
                }
            }
        }
        // Page pools hold the same KV arranged by page.
        let pooln = max_pages * tpp * n_kv_heads * head_dim;
        let mut k_pool = vec![0.0f32; pooln];
        let mut v_pool = vec![0.0f32; pooln];
        for p in 0..max_seq {
            let (page, off) = (p / tpp, p % tpp);
            for kvh in 0..n_kv_heads {
                for dd in 0..head_dim {
                    k_pool[((page * tpp + off) * n_kv_heads + kvh) * head_dim + dd] =
                        kc[(p * n_kv_heads + kvh) * head_dim + dd];
                    v_pool[((page * tpp + off) * n_kv_heads + kvh) * head_dim + dd] =
                        vc[(p * n_kv_heads + kvh) * head_dim + dd];
                }
            }
        }
        let block_tables: Vec<i32> = vec![0, 1, 2];
        let table_offsets: Vec<i32> = vec![0];
        let pos_buf: Vec<i32> = vec![pos];
        let slots: Vec<i32> = vec![0];

        let bytes = |n: usize| n * std::mem::size_of::<f32>();
        let ibytes = |n: usize| n * std::mem::size_of::<i32>();
        let dq = hip::malloc(&h, bytes(q.len())).unwrap();
        let dkc = hip::malloc(&h, bytes(kc.len())).unwrap();
        let dvc = hip::malloc(&h, bytes(vc.len())).unwrap();
        let dkrow = hip::malloc(&h, bytes(kv_row.len())).unwrap();
        let dk_pool = hip::malloc(&h, bytes(k_pool.len())).unwrap();
        let dv_pool = hip::malloc(&h, bytes(v_pool.len())).unwrap();
        let dout_c = hip::malloc(&h, bytes(q.len())).unwrap();
        let dout_p = hip::malloc(&h, bytes(q.len())).unwrap();
        let dpos = hip::malloc(&h, ibytes(pos_buf.len())).unwrap();
        let dslots = hip::malloc(&h, ibytes(slots.len())).unwrap();
        let dtables = hip::malloc(&h, ibytes(block_tables.len())).unwrap();
        let doffs = hip::malloc(&h, ibytes(table_offsets.len())).unwrap();
        // The contiguous store writes into a scratch cache; page store into pools.
        let dstore_c = hip::malloc(&h, bytes(kc.len())).unwrap();

        let cp = |dst: *mut std::ffi::c_void, src: &[f32]| {
            hip::memcpy(
                &h,
                dst,
                src.as_ptr() as *const std::ffi::c_void,
                bytes(src.len()),
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )
            .unwrap()
        };
        let cpi = |dst: *mut std::ffi::c_void, src: &[i32]| {
            hip::memcpy(
                &h,
                dst,
                src.as_ptr() as *const std::ffi::c_void,
                ibytes(src.len()),
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )
            .unwrap()
        };
        cp(dq, &q);
        cp(dkc, &kc);
        cp(dvc, &vc);
        cp(dkrow, &kv_row);
        cp(dk_pool, &k_pool);
        cp(dv_pool, &v_pool);
        cpi(dpos, &pos_buf);
        cpi(dslots, &slots);
        cpi(dtables, &block_tables);
        cpi(doffs, &table_offsets);

        let qp = dq;
        let kcp = dkc;
        let vcp = dvc;
        let kpoolp = dk_pool;
        let vpoolp = dv_pool;
        let dtp = dtables;
        let doffp = doffs;
        let posp = dpos;
        let slotp = dslots;
        let stream = std::ptr::null_mut();

        // Contiguous attention launch.
        let mut p1: Vec<*mut std::ffi::c_void> = vec![
            &qp as *const *mut std::ffi::c_void as *mut _,
            &kcp as *const *mut std::ffi::c_void as *mut _,
            &vcp as *const *mut std::ffi::c_void as *mut _,
            &dout_c as *const *mut std::ffi::c_void as *mut _,
            &posp as *const *mut std::ffi::c_void as *mut _,
            &slotp as *const *mut std::ffi::c_void as *mut _,
            &(batch as i32) as *const i32 as *mut _,
            &(n_heads as i32) as *const i32 as *mut _,
            &(n_kv_heads as i32) as *const i32 as *mut _,
            &(head_dim as i32) as *const i32 as *mut _,
            &(1.0f32 / (head_dim as f32).sqrt()) as *const f32 as *mut _,
            &(max_seq as i32) as *const i32 as *mut _,
        ];
        let shared_c = ((max_seq + 256) * 4) as u32;
        attn_c
            .launch_shmem(
                [(batch * n_heads) as u32, 1, 1],
                [256, 1, 1],
                &mut p1,
                stream,
                shared_c,
            )
            .expect("launch attn contig");

        // Paged attention launch.
        let mut p2: Vec<*mut std::ffi::c_void> = vec![
            &qp as *const *mut std::ffi::c_void as *mut _,
            &kpoolp as *const *mut std::ffi::c_void as *mut _,
            &vpoolp as *const *mut std::ffi::c_void as *mut _,
            &dtp as *const *mut std::ffi::c_void as *mut _,
            &dout_p as *const *mut std::ffi::c_void as *mut _,
            &posp as *const *mut std::ffi::c_void as *mut _,
            &doffp as *const *mut std::ffi::c_void as *mut _,
            &(batch as i32) as *const i32 as *mut _,
            &(n_heads as i32) as *const i32 as *mut _,
            &(n_kv_heads as i32) as *const i32 as *mut _,
            &(head_dim as i32) as *const i32 as *mut _,
            &(1.0f32 / (head_dim as f32).sqrt()) as *const f32 as *mut _,
            &(tpp as i32) as *const i32 as *mut _,
            &(max_pages as i32) as *const i32 as *mut _,
        ];
        let shared_p = ((max_pages * tpp + 256) * 4) as u32;
        attn_p
            .launch_shmem(
                [(batch * n_heads) as u32, 1, 1],
                [256, 1, 1],
                &mut p2,
                stream,
                shared_p,
            )
            .expect("launch attn paged");

        // Contiguous store: writes kv_row into dstore_c at slot 0, pos.
        let mut p3: Vec<*mut std::ffi::c_void> = vec![
            &dkrow as *const *mut std::ffi::c_void as *mut _,
            &dstore_c as *const *mut std::ffi::c_void as *mut _,
            &posp as *const *mut std::ffi::c_void as *mut _,
            &slotp as *const *mut std::ffi::c_void as *mut _,
            &(batch as i32) as *const i32 as *mut _,
            &(n_kv_heads as i32) as *const i32 as *mut _,
            &(head_dim as i32) as *const i32 as *mut _,
            &(max_seq as i32) as *const i32 as *mut _,
        ];
        store_c
            .launch(
                [((batch * n_kv_heads * head_dim) as u32).div_ceil(256), 1, 1],
                [256, 1, 1],
                &mut p3,
                stream,
            )
            .expect("launch store contig");

        // Paged store: writes kv_row into dk_pool at (page, off) for pos.
        let mut p4: Vec<*mut std::ffi::c_void> = vec![
            &dkrow as *const *mut std::ffi::c_void as *mut _,
            &dk_pool as *const *mut std::ffi::c_void as *mut _,
            &posp as *const *mut std::ffi::c_void as *mut _,
            &doffp as *const *mut std::ffi::c_void as *mut _,
            &dtp as *const *mut std::ffi::c_void as *mut _,
            &(batch as i32) as *const i32 as *mut _,
            &(n_kv_heads as i32) as *const i32 as *mut _,
            &(head_dim as i32) as *const i32 as *mut _,
            &(tpp as i32) as *const i32 as *mut _,
        ];
        store_p
            .launch(
                [((batch * n_kv_heads * head_dim) as u32).div_ceil(256), 1, 1],
                [256, 1, 1],
                &mut p4,
                stream,
            )
            .expect("launch store paged");

        unsafe {
            hip::check(&h, (h.api.hip_device_synchronize)()).unwrap();
        }

        let mut out_c = vec![0.0f32; q.len()];
        let mut out_p = vec![0.0f32; q.len()];
        hip::memcpy(
            &h,
            out_c.as_mut_ptr() as *mut _,
            dout_c as *const _,
            bytes(q.len()),
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        hip::memcpy(
            &h,
            out_p.as_mut_ptr() as *mut _,
            dout_p as *const _,
            bytes(q.len()),
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        assert_eq!(
            out_c, out_p,
            "paged attention must be bit-identical to contiguous on GPU"
        );

        // Store parity: the contiguous slot-(0,pos) row == paged (page,off) row.
        let mut store_c = vec![0.0f32; kv_row.len()];
        let mut store_p = vec![0.0f32; kv_row.len()];
        hip::memcpy(
            &h,
            store_c.as_mut_ptr() as *mut _,
            unsafe { dstore_c.add((pos as usize * n_kv_heads * head_dim) * 4) } as *const _,
            bytes(kv_row.len()),
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        let (page, off) = ((pos as usize) / tpp, (pos as usize) % tpp);
        hip::memcpy(
            &h,
            store_p.as_mut_ptr() as *mut _,
            // SAFETY: `dk_pool` is the page pool buffer; the (page, off)
            // offset for position `pos` stays within the allocated
            // `[max_pages, tpp, kv_heads, head_dim]` extent.
            unsafe { dk_pool.add(((page * tpp + off) * n_kv_heads * head_dim) * 4) } as *const _,
            bytes(kv_row.len()),
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        assert_eq!(
            store_c, store_p,
            "paged store must write the same row as contiguous store"
        );

        for p in [
            dq, dkc, dvc, dkrow, dk_pool, dv_pool, dout_c, dout_p, dpos, dslots, dtables, doffs,
            dstore_c,
        ] {
            hip::free(&h, p).unwrap();
        }
    }
}
