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
    int i = threadIdx.x;
    int p = *pos_buf;
    int total = kv_heads * head_dim;
    if (i < total) {
        cache[((long long)p * kv_heads * head_dim) + i] = kv[i];
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
        Ok(self
            .embed
            .launch([1, 1, 1], [256, 1, 1], &mut p, self.stream)?)
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
