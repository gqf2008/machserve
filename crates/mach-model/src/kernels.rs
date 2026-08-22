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
    if (i < total_q) {
        int h = i / head_dim;
        int d = i % head_dim;
        if (d < half) {
            float freq = 1.0f / powf(theta, (2.0f * (float)d) / (float)head_dim);
            float ang = (float)pos * freq;
            float c = cosf(ang), sn = sinf(ang);
            float* p = q + h * head_dim + 2 * d;
            float a = p[0], b = p[1];
            p[0] = a * c - b * sn;
            p[1] = a * sn + b * c;
        }
    }
    if (i < total_k) {
        int h = i / head_dim;
        int d = i % head_dim;
        if (d < half) {
            float freq = 1.0f / powf(theta, (2.0f * (float)d) / (float)head_dim);
            float ang = (float)pos * freq;
            float c = cosf(ang), sn = sinf(ang);
            float* p = k + h * head_dim + 2 * d;
            float a = p[0], b = p[1];
            p[0] = a * c - b * sn;
            p[1] = a * sn + b * c;
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
            float* p = q + (long long)s * n_heads * head_dim + h * head_dim + 2 * d;
            float a = p[0], b = p[1];
            p[0] = a * c - b * sn;
            p[1] = a * sn + b * c;
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
            float* p = k + (long long)s * n_kv_heads * head_dim + h * head_dim + 2 * d;
            float a = p[0], b = p[1];
            p[0] = a * c - b * sn;
            p[1] = a * sn + b * c;
        }
    }
}
"#;

/// Batched KV store: each sequence writes its k/v row at its own position.
const KV_STORE_BATCHED: &str = r#"
extern "C" __global__ void kv_store_batched(const float* kv, float* cache,
                                            const int* pos_buf, int batch,
                                            int kv_heads, int head_dim, int max_seq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_heads * head_dim;
    if (idx < total) {
        int s = idx / (kv_heads * head_dim);
        int i = idx % (kv_heads * head_dim);
        int p = pos_buf[s];
        cache[((long long)s * max_seq + p) * kv_heads * head_dim + i] =
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
    int batch, int n_heads, int n_kv_heads, int head_dim, float scale, int max_seq) {
    extern __shared__ float smem[];
    float* scores = smem;
    float* red = smem + max_seq;

    int s = blockIdx.x / n_heads;
    int h = blockIdx.x % n_heads;
    int groups = n_heads / n_kv_heads;
    int kv = h / groups;
    int pos = pos_buf[s];
    const float* qh = q + ((long long)s * n_heads + h) * head_dim;

    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        const float* kp = kc + ((long long)s * max_seq + p) * n_kv_heads * head_dim + kv * head_dim;
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
            const float* vp = vc + ((long long)s * max_seq + p) * n_kv_heads * head_dim + kv * head_dim + dd;
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
__device__ unsigned short f32_to_f16(float x) {
    unsigned int b = __float_as_uint(x);
    unsigned int sign = (b >> 16) & 0x8000u;
    int e = (int)((b >> 23) & 0xffu);
    unsigned int m = b & 0x7fffffu;
    if (e == 255) return (unsigned short)(sign | 0x7c00u | (m ? 0x200u : 0u));
    int ne = e - 127 + 15;
    if (ne >= 31) return (unsigned short)(sign | 0x7c00u);
    if (ne <= 0) {
        if (ne < -10) return (unsigned short)sign;
        unsigned int mm = m | 0x800000u;
        int shift = 14 - ne;
        unsigned int half = 1u << (shift - 1);
        unsigned int m16 = mm >> shift;
        unsigned int rem = mm & ((1u << shift) - 1u);
        if (rem > half || (rem == half && (m16 & 1u))) m16++;
        return (unsigned short)(sign | m16);
    }
    unsigned int m16 = m >> 13;
    unsigned int rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (m16 & 1u))) {
        m16++;
        if (m16 == 0x400u) { m16 = 0u; ne++; if (ne >= 31) return (unsigned short)(sign | 0x7c00u); }
    }
    return (unsigned short)(sign | ((unsigned int)ne << 10) | m16);
}

extern "C" __global__ void cast_f32_f16(const float* x, unsigned short* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f32_to_f16(x[i]);
}
"#;

/// fp16 -> f32 cast (device conversion matches `crate::fp16::f16_to_f32`).
const CAST_F16_F32: &str = r#"
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

extern "C" __global__ void cast_f16_f32(const unsigned short* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f16_to_f32(x[i]);
}
"#;

/// fp16 embedding gather -> f32 activations (matches `crate::fp16::f16_to_f32`).
const EMBED_GATHER_F16: &str = r#"
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

extern "C" __global__ void embed_gather_f16(const int* tok, const unsigned short* emb,
                                            float* x, int cols, int batch) {
    int s = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) x[(long long)s * cols + i] = f16_to_f32(emb[(long long)tok[s] * cols + i]);
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
        batch: i32,
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
        let mut p = vec![
            &qp as *const *const f32 as *mut core::ffi::c_void,
            &kp as *const *const f32 as *mut core::ffi::c_void,
            &vp as *const *const f32 as *mut core::ffi::c_void,
            &op as *const *mut f32 as *mut core::ffi::c_void,
            &pp as *const *const i32 as *mut core::ffi::c_void,
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

    /// Synchronizes the execution stream.
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
