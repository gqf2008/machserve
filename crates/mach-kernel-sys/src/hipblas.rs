//! hipBLAS FFI (GEMM). Loads `hipblas.dll` dynamically.
//!
//! ABI verified against `C:\Program Files\AMD\ROCm\6.2\include\hipblas\hipblas.h`:
//! `hipblasHandle_t = void*`, `HIPBLAS_OP_N = 111`.

use crate::hip::{Hip, HipError, HipStream, check};
use std::sync::{Arc, OnceLock};

/// hipblas operation enums (hipblas.h).
pub const HIPBLAS_OP_N: i32 = 111;
pub const HIPBLAS_OP_T: i32 = 112;
pub const HIPBLAS_OP_C: i32 = 113;

/// Opaque hipBLAS handle.
pub type HipBlasHandle = *mut core::ffi::c_void;

/// Raw hipBLAS function pointers.
#[derive(Clone, Copy)]
pub struct HipBlasApi {
    pub hipblas_create: unsafe extern "C" fn(*mut HipBlasHandle) -> i32,
    pub hipblas_destroy: unsafe extern "C" fn(HipBlasHandle) -> i32,
    pub hipblas_set_stream: unsafe extern "C" fn(HipBlasHandle, HipStream) -> i32,
    pub hipblas_sgemm: unsafe extern "C" fn(
        HipBlasHandle,
        i32,
        i32,
        i32,
        i32,
        i32,
        *const f32,
        *const core::ffi::c_void,
        i32,
        *const core::ffi::c_void,
        i32,
        *const f32,
        *mut core::ffi::c_void,
        i32,
    ) -> i32,
}

/// Loaded hipBLAS runtime.
pub struct HipBlasLib {
    _lib: libloading::Library,
    api: HipBlasApi,
}

// SAFETY: the library handle is kept alive for the struct's lifetime and the
// hipBLAS calls we issue are thread-safe.
unsafe impl Send for HipBlasLib {}
unsafe impl Sync for HipBlasLib {}

static HIPBLAS: OnceLock<Result<Arc<HipBlasLib>, HipError>> = OnceLock::new();

fn load() -> Result<Arc<HipBlasLib>, HipError> {
    crate::hip::ensure_rocm_on_path();
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(bin) = crate::hip::rocm_bin() {
        candidates.push(bin.join("hipblas.dll"));
    }
    let mut lib = None;
    let mut last_err = None;
    for c in &candidates {
        match unsafe { libloading::Library::new(c.to_str().unwrap_or("")) } {
            Ok(l) => {
                lib = Some(l);
                break;
            }
            Err(e) => last_err = Some(format!("{}: {e}", c.display())),
        }
    }
    let lib = match lib {
        Some(l) => l,
        None => match unsafe { libloading::Library::new("hipblas.dll") } {
            Ok(l) => l,
            Err(e) => {
                return Err(HipError::Library(format!(
                    "hipblas.dll not found ({}; fallback: {e})",
                    last_err.unwrap_or_default()
                )));
            }
        },
    };
    let api = HipBlasApi {
        hipblas_create: *(unsafe { lib.get(b"hipblasCreate\0") })
            .map_err(|e| HipError::Symbol(format!("hipblasCreate({e})")))?,
        hipblas_destroy: *(unsafe { lib.get(b"hipblasDestroy\0") })
            .map_err(|e| HipError::Symbol(format!("hipblasDestroy({e})")))?,
        hipblas_set_stream: *(unsafe { lib.get(b"hipblasSetStream\0") })
            .map_err(|e| HipError::Symbol(format!("hipblasSetStream({e})")))?,
        hipblas_sgemm: *(unsafe { lib.get(b"hipblasSgemm\0") })
            .map_err(|e| HipError::Symbol(format!("hipblasSgemm({e})")))?,
    };
    Ok(Arc::new(HipBlasLib { _lib: lib, api }))
}

/// Returns the process-wide hipBLAS library (loaded once).
pub fn hipblas() -> Result<Arc<HipBlasLib>, HipError> {
    HIPBLAS.get_or_init(load).clone()
}

/// A hipBLAS handle bound to a HIP runtime; safe wrapper around the raw API.
pub struct HipBlas {
    lib: Arc<HipBlasLib>,
    handle: HipBlasHandle,
    _hip: Arc<Hip>,
}

impl HipBlas {
    /// Creates a hipBLAS handle.
    pub fn new(hip: Arc<Hip>) -> Result<Self, HipError> {
        let lib = hipblas()?;
        let mut handle = std::ptr::null_mut();
        unsafe { check(&hip, (lib.api.hipblas_create)(&mut handle))? };
        Ok(Self {
            lib,
            handle,
            _hip: hip,
        })
    }

    /// Binds the handle to a stream so calls are ordered on it (required for
    /// CUDA/HIP graph capture).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn set_stream(&self, stream: HipStream) -> Result<(), HipError> {
        unsafe {
            check(
                &self._hip,
                (self.lib.api.hipblas_set_stream)(self.handle, stream),
            )
        }
    }

    /// `C = alpha * op(A) * op(B) + beta * C` in fp32.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn sgemm(
        &self,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    ) -> Result<(), HipError> {
        unsafe {
            check(
                &self._hip,
                (self.lib.api.hipblas_sgemm)(
                    self.handle,
                    trans_a,
                    trans_b,
                    m,
                    n,
                    k,
                    &alpha,
                    a as *const core::ffi::c_void,
                    lda,
                    b as *const core::ffi::c_void,
                    ldb,
                    &beta,
                    c as *mut core::ffi::c_void,
                    ldc,
                ),
            )
        }
    }
}

impl Drop for HipBlas {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = (self.lib.api.hipblas_destroy)(self.handle);
            }
        }
    }
}

// SAFETY: the hipBLAS handle is process-global and only used by this struct;
// models are not intended to be shared across threads (one model per thread).
unsafe impl Send for HipBlas {}
unsafe impl Sync for HipBlas {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip;

    fn have_gpu() -> bool {
        matches!(hip::device_count(), Ok(n) if n > 0)
    }

    #[test]
    fn sgemm_probe() {
        if !have_gpu() {
            eprintln!("skipping: no device");
            return;
        }
        let h = hip::hip().unwrap();
        let blas = HipBlas::new(h.clone()).unwrap();
        let n = 4i32;
        // C = A @ B, all 4x4 identity-ish
        let a = vec![1.0f32; (n * n) as usize];
        let b = vec![2.0f32; (n * n) as usize];
        let mut c = vec![0.0f32; (n * n) as usize];
        let da = hip::malloc(&h, (n * n * 4) as usize).unwrap();
        let db = hip::malloc(&h, (n * n * 4) as usize).unwrap();
        let dc = hip::malloc(&h, (n * n * 4) as usize).unwrap();
        hip::memcpy(
            &h,
            da,
            a.as_ptr() as *const _,
            (n * n * 4) as usize,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        hip::memcpy(
            &h,
            db,
            b.as_ptr() as *const _,
            (n * n * 4) as usize,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        blas.sgemm(
            HIPBLAS_OP_N,
            HIPBLAS_OP_N,
            n,
            n,
            n,
            1.0,
            da as *const f32,
            n,
            db as *const f32,
            n,
            0.0,
            dc as *mut f32,
            n,
        )
        .expect("4x4 sgemm");

        // Random-data check against CPU: out[j] = sum_i x[i] * W[j,i]
        {
            let kk = 16i32;
            let nn2 = 32i32;
            let mut rng = 12345u64;
            let mut frand = move || {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rng >> 40) as f32) / (1u64 << 24) as f32 - 0.5
            };
            let wx2: Vec<f32> = (0..(nn2 * kk) as usize).map(|_| frand()).collect();
            let xin2: Vec<f32> = (0..kk as usize).map(|_| frand()).collect();
            let dw2 = hip::malloc(&h, (nn2 * kk * 4) as usize).unwrap();
            let dx2 = hip::malloc(&h, (kk * 4) as usize).unwrap();
            let dco2 = hip::malloc(&h, (nn2 * 4) as usize).unwrap();
            hip::memcpy(
                &h,
                dw2,
                wx2.as_ptr() as *const _,
                (nn2 * kk * 4) as usize,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )
            .unwrap();
            hip::memcpy(
                &h,
                dx2,
                xin2.as_ptr() as *const _,
                (kk * 4) as usize,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )
            .unwrap();
            blas.sgemm(
                HIPBLAS_OP_N,
                HIPBLAS_OP_N,
                1,
                nn2,
                kk,
                1.0,
                dx2 as *const f32,
                1,
                dw2 as *const f32,
                kk,
                0.0,
                dco2 as *mut f32,
                1,
            )
            .expect("random gemm");
            let mut cout2 = vec![0.0f32; nn2 as usize];
            hip::memcpy(
                &h,
                cout2.as_mut_ptr() as *mut _,
                dco2 as *const _,
                (nn2 * 4) as usize,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )
            .unwrap();
            let mut maxerr = 0.0f32;
            for j in 0..nn2 as usize {
                let mut want = 0.0f32;
                for i in 0..kk as usize {
                    want += xin2[i] * wx2[j * kk as usize + i];
                }
                maxerr = maxerr.max((cout2[j] - want).abs());
            }
            eprintln!("random gemm maxerr={}", maxerr);
            assert!(maxerr < 1e-3, "random gemm mismatch {maxerr}");
            hip::free(&h, dw2).unwrap();
            hip::free(&h, dx2).unwrap();
            hip::free(&h, dco2).unwrap();
        }
        hip::memcpy(
            &h,
            c.as_mut_ptr() as *mut _,
            dc as *const _,
            (n * n * 4) as usize,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        // All ones @ all twos = 8 per element
        assert!(
            c.iter().all(|&v| (v - 8.0).abs() < 1e-4),
            "got {:?}",
            &c[..4]
        );
        hip::free(&h, da).unwrap();
        hip::free(&h, db).unwrap();
        hip::free(&h, dc).unwrap();
        eprintln!("4x4 sgemm OK");
    }
}
