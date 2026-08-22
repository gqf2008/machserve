import io
p = "E:/Users/gxh/Documents/GitHub/machserve/crates/mach-model/src/sampling.rs"
s = '''//! GPU-side sampling.
//!
//! Fixes the P1 end-to-end bottleneck: instead of reading all `vocab` logits
//! (e.g. 607KB for a 151936-vocab model) back to the host every token, an
//! argmax kernel reduces on-device and only a 4-byte token id is read back.
//! Greedy (argmax) first; top-k/top-p build on the same scratch design.

use crate::Error;
use mach_engine::hip::hip_arch;
use mach_kernel_sys::hip::{self, Hip, HipKernelModule, HipStream};
use std::sync::Arc;

/// Block-local argmax: each block reduces a contiguous 256-element slice.
const ARGMAX_SLICE: &str = r#"
extern "C" __global__ void argmax_slice(const float* logits, float* out_val,
                                        int* out_idx, int vocab) {
    int base = blockIdx.x * blockDim.x;
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    float v = -1e30f;
    int idx = -1;
    int i = base + threadIdx.x;
    if (i < vocab) {
        v = logits[i];
        idx = i;
    }
    s_val[threadIdx.x] = v;
    s_idx[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float a = s_val[threadIdx.x + s];
            float b = s_val[threadIdx.x];
            int ai = s_idx[threadIdx.x + s];
            int bi = s_idx[threadIdx.x];
            if (a > b || (a == b && ai < bi)) {
                s_val[threadIdx.x] = a;
                s_idx[threadIdx.x] = ai;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_val[blockIdx.x] = s_val[0];
        out_idx[blockIdx.x] = s_idx[0];
    }
}
"#;

/// Final reduce over block maxima (n may exceed blockDim; threads fold).
const ARGMAX_REDUCE: &str = r#"
extern "C" __global__ void argmax_reduce(const float* in_val, const int* in_idx,
                                         float* out_val, int* out_idx, int n) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    float v = -1e30f;
    int idx = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (in_val[j] > v || (in_val[j] == v && in_idx[j] < idx)) {
            v = in_val[j];
            idx = in_idx[j];
        }
    }
    s_val[threadIdx.x] = v;
    s_idx[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float a = s_val[threadIdx.x + s];
            float b = s_val[threadIdx.x];
            int ai = s_idx[threadIdx.x + s];
            int bi = s_idx[threadIdx.x];
            if (a > b || (a == b && ai < bi)) {
                s_val[threadIdx.x] = a;
                s_idx[threadIdx.x] = ai;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_val[0] = s_val[0];
        out_idx[0] = s_idx[0];
    }
}
"#;

/// On-device greedy sampler bound to a stream.
pub struct HipSampler {
    hip: Arc<Hip>,
    stream: HipStream,
    slice: HipKernelModule,
    reduce: HipKernelModule,
    block_val: *mut f32,
    block_idx: *mut i32,
    out_val: *mut f32,
    out_idx: *mut i32,
    host_idx: *mut i32,
}

impl HipSampler {
    /// Compiles the kernels and allocates scratch. `stream` must be the stream
    /// the model launches on.
    pub fn new(hip: Arc<Hip>, stream: HipStream) -> Result<Self, Error> {
        let arch = hip_arch();
        let max_blocks = 4096usize; // covers vocab up to ~1M
        let block_val = hip::malloc(&hip, max_blocks * 4)? as *mut f32;
        let block_idx = hip::malloc(&hip, max_blocks * 4)? as *mut i32;
        let out_val = hip::malloc(&hip, 4)? as *mut f32;
        let out_idx = hip::malloc(&hip, 4)? as *mut i32;
        let host_idx = hip::host_malloc(&hip, 4)? as *mut i32;
        Ok(Self {
            hip,
            stream,
            slice: HipKernelModule::compile(&arch, ARGMAX_SLICE, "argmax_slice")?,
            reduce: HipKernelModule::compile(&arch, ARGMAX_REDUCE, "argmax_reduce")?,
            block_val,
            block_idx,
            out_val,
            out_idx,
            host_idx,
        })
    }

    /// Returns the index of the maximum element of `logits` (greedy sample).
    /// Deterministic tie-break: smallest index wins.
    pub fn argmax(&self, logits: *const f32, vocab: usize) -> Result<u32, Error> {
        let nblocks = vocab.div_ceil(256);
        assert!(nblocks <= 4096, "vocab too large for sampler scratch");

        let lp = logits;
        let bv = self.block_val;
        let bi = self.block_idx;
        let vocab_i = vocab as i32;
        let mut p1 = vec![
            &lp as *const *const f32 as *mut core::ffi::c_void,
            &bv as *const *mut f32 as *mut core::ffi::c_void,
            &bi as *const *mut i32 as *mut core::ffi::c_void,
            &vocab_i as *const i32 as *mut core::ffi::c_void,
        ];
        self.slice
            .launch([nblocks as u32, 1, 1], [256, 1, 1], &mut p1, self.stream)?;

        let iv = self.block_val;
        let ii = self.block_idx;
        let ov = self.out_val;
        let oi = self.out_idx;
        let n = nblocks as i32;
        let mut p2 = vec![
            &iv as *const *const f32 as *mut core::ffi::c_void,
            &ii as *const *const i32 as *mut core::ffi::c_void,
            &ov as *const *mut f32 as *mut core::ffi::c_void,
            &oi as *const *mut i32 as *mut core::ffi::c_void,
            &n as *const i32 as *mut core::ffi::c_void,
        ];
        self.reduce
            .launch([1, 1, 1], [256, 1, 1], &mut p2, self.stream)?;

        unsafe {
            hip::check(&self.hip, (self.hip.api.hip_stream_synchronize)(self.stream))?;
            hip::memcpy(
                &self.hip,
                self.host_idx as *mut core::ffi::c_void,
                self.out_idx as *const core::ffi::c_void,
                4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )?;
        }
        Ok(unsafe { *self.host_idx } as u32)
    }
}

impl Drop for HipSampler {
    fn drop(&mut self) {
        unsafe {
            let _ = hip::free(&self.hip, self.block_val as *mut _);
            let _ = hip::free(&self.hip, self.block_idx as *mut _);
            let _ = hip::free(&self.hip, self.out_val as *mut _);
            let _ = hip::free(&self.hip, self.out_idx as *mut _);
            let _ = hip::host_free(&self.hip, self.host_idx as *mut _);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn have_gpu() -> Option<Arc<Hip>> {
        match hip::hip() {
            Ok(h) => match hip::device_count() {
                Ok(n) if n > 0 => Some(h),
                _ => {
                    eprintln!("skipping HIP test: no device");
                    None
                }
            },
            Err(e) => {
                eprintln!("skipping HIP test: {e}");
                None
            }
        }
    }

    fn cpu_argmax_first(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        best as u32
    }

    fn run(vocab: usize, seed: u64) {
        let h = have_gpu().expect("gpu");
        let mut rng = seed;
        let logits: Vec<f32> = (0..vocab)
            .map(|_| {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rng >> 40) as f32) / (1u64 << 24) as f32
            })
            .collect();
        let dlogits = hip::malloc(&h, vocab * 4).unwrap() as *mut f32;
        hip::memcpy(
            &h,
            dlogits as *mut _,
            logits.as_ptr() as *const _,
            vocab * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        let mut stream = std::ptr::null_mut();
        unsafe { hip::check(&h, (h.api.hip_stream_create)(&mut stream)).unwrap() };
        let s = HipSampler::new(Arc::clone(&h), stream).unwrap();
        let got = s.argmax(dlogits, vocab).unwrap();
        let want = cpu_argmax_first(&logits);
        assert_eq!(got, want, "vocab={vocab} seed={seed}");
        hip::free(&h, dlogits as *mut _).unwrap();
        unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };
    }

    #[test]
    fn argmax_matches_cpu() {
        run(1000, 1);
        run(256, 2);
        run(257, 3);
        run(151936, 4);
        run(4096, 5);
    }

    #[test]
    fn argmax_tie_breaks_to_first_index() {
        let h = have_gpu().expect("gpu");
        let vocab = 1024usize;
        let mut logits = vec![0.1f32; vocab];
        logits[700] = 0.5; // unique max
        let dlogits = hip::malloc(&h, vocab * 4).unwrap() as *mut f32;
        hip::memcpy(
            &h,
            dlogits as *mut _,
            logits.as_ptr() as *const _,
            vocab * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        let mut stream = std::ptr::null_mut();
        unsafe { hip::check(&h, (h.api.hip_stream_create)(&mut stream)).unwrap() };
        let s = HipSampler::new(Arc::clone(&h), stream).unwrap();
        assert_eq!(s.argmax(dlogits, vocab).unwrap(), 700);
        // All-equal: must return the smallest index (0).
        let eq = vec![0.5f32; vocab];
        hip::memcpy(
            &h,
            dlogits as *mut _,
            eq.as_ptr() as *const _,
            vocab * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        assert_eq!(s.argmax(dlogits, vocab).unwrap(), 0);
        hip::free(&h, dlogits as *mut _).unwrap();
        unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };
    }
}
'''
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sampling.rs written")
