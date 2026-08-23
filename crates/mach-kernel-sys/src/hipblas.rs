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

/// hipblasDatatype_t values (ROCm 6.2: `hipblasDatatype_t` is a typedef of
/// `hipDataType`, so these are the hipDataType values).
pub const HIPBLAS_R_32F: i32 = 0;
pub const HIPBLAS_R_16F: i32 = 2;
pub const HIPBLAS_R_16B: i32 = 14;
/// fp8 E4M3 / E5M2 (`hipDataType` values; support on gfx1100 is probe-tested).
pub const HIPBLAS_R_8F_E4M3: i32 = 30;
pub const HIPBLAS_R_8F_E5M2: i32 = 31;
/// hipblasComputeType_t: at least 32-bit precision.
pub const HIPBLAS_COMPUTE_32F: i32 = 2;
/// hipblasGemmAlgo_t: default.
pub const HIPBLAS_GEMM_DEFAULT: i32 = 160;

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
    pub hipblas_gemm_ex: unsafe extern "C" fn(
        HipBlasHandle,
        i32,
        i32,
        i32,
        i32,
        i32,
        *const core::ffi::c_void,
        *const core::ffi::c_void,
        i32,
        i32,
        *const core::ffi::c_void,
        i32,
        i32,
        *const core::ffi::c_void,
        *mut core::ffi::c_void,
        i32,
        i32,
        i32,
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
        // ROCm 6.2: the plain `hipblasGemmEx` export is the legacy ABI and
        // rejects every enum; `hipblasGemmEx_v2` (hipDataType values) works.
        hipblas_gemm_ex: *(unsafe { lib.get(b"hipblasGemmEx_v2\0") })
            .map_err(|e| HipError::Symbol(format!("hipblasGemmEx_v2({e})")))?,
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
    /// `C = alpha * op(A) * op(B) + beta * C` with mixed dtypes via
    /// `hipblasGemmEx` (e.g. fp16 inputs / fp32 output, fp32 accumulate).
    /// `alpha`/`beta` are fp32 scalars matching `HIPBLAS_COMPUTE_32F`.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn gemm_ex(
        &self,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        a_type: i32,
        a: *const core::ffi::c_void,
        lda: i32,
        b_type: i32,
        b: *const core::ffi::c_void,
        ldb: i32,
        c_type: i32,
        c: *mut core::ffi::c_void,
        ldc: i32,
        compute: i32,
    ) -> Result<(), HipError> {
        self.gemm_ex_algo(
            trans_a,
            trans_b,
            m,
            n,
            k,
            a_type,
            a,
            lda,
            b_type,
            b,
            ldb,
            c_type,
            c,
            ldc,
            compute,
            HIPBLAS_GEMM_DEFAULT,
        )
    }

    /// [`Self::gemm_ex`] with an explicit rocBLAS algorithm/solution index
    /// (probe/tuning only; production uses the default).
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn gemm_ex_algo(
        &self,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        a_type: i32,
        a: *const core::ffi::c_void,
        lda: i32,
        b_type: i32,
        b: *const core::ffi::c_void,
        ldb: i32,
        c_type: i32,
        c: *mut core::ffi::c_void,
        ldc: i32,
        compute: i32,
        algo: i32,
    ) -> Result<(), HipError> {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            check(
                &self._hip,
                (self.lib.api.hipblas_gemm_ex)(
                    self.handle,
                    trans_a,
                    trans_b,
                    m,
                    n,
                    k,
                    &alpha as *const f32 as *const core::ffi::c_void,
                    a,
                    a_type,
                    lda,
                    b,
                    b_type,
                    ldb,
                    &beta as *const f32 as *const core::ffi::c_void,
                    c,
                    c_type,
                    ldc,
                    compute,
                    algo,
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
    #[allow(clippy::unnecessary_cast)]
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

            // Qwen-scale shapes: m=1, k=896, n up to 151936.
            {
                let kk = 896i32;
                for &nn in &[896i32, 4864i32, 151936i32] {
                    let wx2: Vec<f32> = (0..(nn as usize * kk as usize))
                        .map(|i| ((i % 7919) as f32 - 3959.0) / 100000.0)
                        .collect();
                    let xin2: Vec<f32> = (0..kk as usize)
                        .map(|i| ((i % 104729) as f32 - 50000.0) / 1000000.0)
                        .collect();
                    let dw2 = hip::malloc(&h, (nn as usize * kk as usize * 4) as usize).unwrap();
                    let dx2 = hip::malloc(&h, (kk as usize * 4) as usize).unwrap();
                    let dco2 = hip::malloc(&h, (nn as usize * 4) as usize).unwrap();
                    hip::memcpy(
                        &h,
                        dw2,
                        wx2.as_ptr() as *const _,
                        (nn as usize * kk as usize * 4) as usize,
                        hip::HIP_MEMCPY_HOST_TO_DEVICE,
                    )
                    .unwrap();
                    hip::memcpy(
                        &h,
                        dx2,
                        xin2.as_ptr() as *const _,
                        (kk as usize * 4) as usize,
                        hip::HIP_MEMCPY_HOST_TO_DEVICE,
                    )
                    .unwrap();
                    blas.sgemm(
                        HIPBLAS_OP_N,
                        HIPBLAS_OP_N,
                        1,
                        nn,
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
                    .expect("qwen gemm");
                    let mut cout2 = vec![0.0f32; nn as usize];
                    hip::memcpy(
                        &h,
                        cout2.as_mut_ptr() as *mut _,
                        dco2 as *const _,
                        (nn as usize * 4) as usize,
                        hip::HIP_MEMCPY_DEVICE_TO_HOST,
                    )
                    .unwrap();
                    // sample 64 outputs
                    let mut maxerr = 0.0f32;
                    for &j in &[0usize, 1, 17, 100, 999, 5000, 9999, 12345] {
                        if j >= nn as usize {
                            continue;
                        }
                        let mut want = 0.0f32;
                        for i in 0..kk as usize {
                            want += xin2[i] * wx2[j * kk as usize + i];
                        }
                        maxerr = maxerr.max((cout2[j] - want).abs());
                    }
                    eprintln!("qwen-shape gemm m=1 k={kk} n={nn} maxerr={maxerr}");
                    assert!(maxerr < 1e-2, "qwen-shape gemm n={nn} mismatch {maxerr}");
                    hip::free(&h, dw2).unwrap();
                    hip::free(&h, dx2).unwrap();
                    hip::free(&h, dco2).unwrap();
                }
            }

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

    /// fp16 inputs / fp32 output GEMM via `hipblasGemmEx` vs a CPU fp32 dot,
    /// using the model's layout: weight row-major [n, k] = column-major B(k x n),
    /// activation row-major [batch, k] = column-major (k x batch), transA=T.
    #[test]
    fn gemm_ex_fp16_probe() {
        if !have_gpu() {
            eprintln!("skipping: no device");
            return;
        }
        let h = hip::hip().expect("hip runtime");
        let blas = HipBlas::new(h.clone()).expect("hipblas handle");
        let mut stream = std::ptr::null_mut();
        unsafe { hip::check(&h, (h.api.hip_stream_create)(&mut stream)).unwrap() };
        blas.set_stream(stream).unwrap();

        let (n, batch, k) = (6i32, 4i32, 8i32);
        let w: Vec<f32> = (0..(n * k) as usize)
            .map(|i| ((i * 7) % 100) as f32 / 100.0 - 0.4)
            .collect();
        let x: Vec<f32> = (0..(batch * k) as usize)
            .map(|i| ((i * 13) % 100) as f32 / 100.0 - 0.4)
            .collect();
        let w16: Vec<u16> = w.iter().map(|&v| crate::hip::fp32_to_f16_host(v)).collect();
        let x16: Vec<u16> = x.iter().map(|&v| crate::hip::fp32_to_f16_host(v)).collect();

        let dw = hip::malloc(&h, w16.len() * 2).unwrap();
        let dx = hip::malloc(&h, x16.len() * 2).unwrap();
        let dc = hip::malloc(&h, (batch * n) as usize * 4).unwrap();
        hip::memcpy(
            &h,
            dw,
            w16.as_ptr() as *const _,
            w16.len() * 2,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        hip::memcpy(
            &h,
            dx,
            x16.as_ptr() as *const _,
            x16.len() * 2,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();

        blas.gemm_ex(
            HIPBLAS_OP_T,
            HIPBLAS_OP_N,
            n,
            batch,
            k,
            HIPBLAS_R_16F,
            dw as *const core::ffi::c_void,
            k,
            HIPBLAS_R_16F,
            dx as *const core::ffi::c_void,
            k,
            HIPBLAS_R_32F,
            dc,
            n,
            HIPBLAS_COMPUTE_32F,
        )
        .expect("gemm_ex");

        let mut c = vec![0.0f32; (batch * n) as usize];
        hip::memcpy(
            &h,
            c.as_mut_ptr() as *mut _,
            dc as *const _,
            c.len() * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        let wf: Vec<f32> = w16
            .iter()
            .map(|&u| crate::hip::fp16_to_f32_host(u))
            .collect();
        let xf: Vec<f32> = x16
            .iter()
            .map(|&u| crate::hip::fp16_to_f32_host(u))
            .collect();
        let mut maxerr = 0.0f32;
        for b in 0..batch as usize {
            for j in 0..n as usize {
                let mut want = 0.0f32;
                for t in 0..k as usize {
                    want += wf[j * k as usize + t] * xf[b * k as usize + t];
                }
                maxerr = maxerr.max((c[b * n as usize + j] - want).abs());
            }
        }
        eprintln!("gemm_ex fp16 maxerr={maxerr}");
        assert!(maxerr < 1e-3, "gemm_ex fp16 mismatch {maxerr}");
        hip::free(&h, dw).unwrap();
        hip::free(&h, dx).unwrap();
        hip::free(&h, dc).unwrap();
        unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };
        eprintln!("gemm_ex fp16 probe OK");
    }

    /// fp8 (E4M3) GEMM probe: quantize weights/activations to fp8 with a
    /// per-tensor scale and run `hipblasGemmEx`. The KEY signal is whether
    /// hipBLAS accepts fp8 on this GPU: an error here means fp8 GEMM is not
    /// available via hipBLAS on gfx1100 (ROCm 6.2), which blocks a native fp8
    /// path.
    #[test]
    fn gemm_ex_fp8_probe() {
        if !have_gpu() {
            eprintln!("skipping: no device");
            return;
        }
        let h = hip::hip().expect("hip runtime");
        let blas = HipBlas::new(h.clone()).expect("hipblas handle");
        let mut stream = std::ptr::null_mut();
        unsafe { hip::check(&h, (h.api.hip_stream_create)(&mut stream)).unwrap() };
        blas.set_stream(stream).unwrap();

        let f32_to_e4m3 = |x: f32| -> u8 {
            let sign: u8 = if x < 0.0 { 0x80 } else { 0 };
            let a = x.abs();
            if a > 448.0 {
                return sign | 0x7F; // saturate
            }
            if a < 2f32.powi(-6) * 0.5 {
                return sign; // ~zero
            }
            let bits = a.to_bits();
            let e = ((bits >> 23) & 0xFF) as i32 - 127;
            let m = bits & 0x7F_FFFF;
            let biased = (e + 7).clamp(1, 14);
            let m3 = ((m + (1 << 20)) >> 20) as u8; // top 3 bits, round-nearest
            sign | ((biased as u8) << 3) | (m3 & 0x7)
        };
        let e4m3_to_f32 = |u: u8| -> f32 {
            let sign = if u & 0x80 != 0 { -1.0f32 } else { 1.0 };
            let e = ((u >> 3) & 0x0F) as i32;
            let m = (u & 0x07) as f32;
            if e == 0 {
                sign * (m / 8.0) * 2f32.powi(-6)
            } else if e == 15 {
                f32::NAN
            } else {
                sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
            }
        };

        let (n, batch, k) = (6i32, 4i32, 8i32);
        let w: Vec<f32> = (0..(n * k) as usize)
            .map(|i| ((i * 7) % 100) as f32 / 100.0 - 0.4)
            .collect();
        let x: Vec<f32> = (0..(batch * k) as usize)
            .map(|i| ((i * 13) % 100) as f32 / 100.0 - 0.4)
            .collect();
        // Per-tensor scales.
        let w_scale = w.iter().map(|v| v.abs()).fold(1e-9f32, f32::max) / 448.0;
        let x_scale = x.iter().map(|v| v.abs()).fold(1e-9f32, f32::max) / 448.0;
        let w8: Vec<u8> = w.iter().map(|&v| f32_to_e4m3(v / w_scale)).collect();
        let x8: Vec<u8> = x.iter().map(|&v| f32_to_e4m3(v / x_scale)).collect();

        let dw = hip::malloc(&h, w8.len()).unwrap();
        let dx = hip::malloc(&h, x8.len()).unwrap();
        let dc = hip::malloc(&h, (batch * n) as usize * 4).unwrap();
        hip::memcpy(
            &h,
            dw,
            w8.as_ptr() as *const _,
            w8.len(),
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        hip::memcpy(
            &h,
            dx,
            x8.as_ptr() as *const _,
            x8.len(),
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();

        let err = blas.gemm_ex(
            HIPBLAS_OP_T,
            HIPBLAS_OP_N,
            n,
            batch,
            k,
            HIPBLAS_R_8F_E4M3,
            dw as *const core::ffi::c_void,
            k,
            HIPBLAS_R_8F_E4M3,
            dx as *const core::ffi::c_void,
            k,
            HIPBLAS_R_32F,
            dc,
            n,
            HIPBLAS_COMPUTE_32F,
        );
        match err {
            Err(e) => {
                eprintln!("gemm_ex fp8: hipBLAS REJECTED fp8 on this GPU: {e}");
                eprintln!("=> native fp8 GEMM via hipBLAS unavailable on gfx1100/ROCm 6.2");
                hip::free(&h, dw).unwrap();
                hip::free(&h, dx).unwrap();
                hip::free(&h, dc).unwrap();
                unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };
                return;
            }
            Ok(()) => {
                eprintln!("gemm_ex fp8: hipBLAS ACCEPTED fp8 on this GPU");
            }
        }
        // Compare dequantized vs CPU fp32 reference (fp8 error ~1e-2).
        let mut c = vec![0.0f32; (batch * n) as usize];
        hip::memcpy(
            &h,
            c.as_mut_ptr() as *mut _,
            dc as *const _,
            c.len() * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        let wf: Vec<f32> = w8.iter().map(|&u| e4m3_to_f32(u) * w_scale).collect();
        let xf: Vec<f32> = x8.iter().map(|&u| e4m3_to_f32(u) * x_scale).collect();
        let mut maxerr = 0.0f32;
        let mut maxabs = 0.0f32;
        for b in 0..batch as usize {
            for j in 0..n as usize {
                let mut want = 0.0f32;
                for t in 0..k as usize {
                    want += wf[j * k as usize + t] * xf[b * k as usize + t];
                }
                maxabs = maxabs.max(want.abs());
                maxerr = maxerr.max((c[b * n as usize + j] - want).abs());
            }
        }
        eprintln!("gemm_ex fp8 maxerr={maxerr} (ref max {maxabs})");
        assert!(
            c.iter().all(|v| v.is_finite()),
            "fp8 GEMM produced non-finite output"
        );
        hip::free(&h, dw).unwrap();
        hip::free(&h, dx).unwrap();
        hip::free(&h, dc).unwrap();
        unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };
    }
}
