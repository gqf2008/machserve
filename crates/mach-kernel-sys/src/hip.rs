//! HIP runtime FFI for the ROCm backend (amdhip64_6.dll + hiprtc0602.dll).
//!
//! DLLs are loaded **dynamically** at runtime (no link-time dependency). The
//! HIP runtime is resolved from the ROCm install (`MACH_HIP_PATH` or
//! `C:\Program Files\AMD\ROCm\<ver>\bin`) and falls back to bare DLL names
//! (e.g. copies in System32). This is the only place the runtime talks to the
//! HIP driver/runtime.
//!
//! Symbol names are the exported **camelCase** forms (verified via
//! GetProcAddress against ROCm 6.2): `hipGetDeviceCount`, `hipMalloc`, ...
//! ABI constants were verified against `hip_runtime_api.h` / `driver_types.h`.

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// HIP success code.
pub const HIP_SUCCESS: c_int = 0;
/// Capture mode: global (the only mode usable from multiple threads).
pub const HIP_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;
/// Memcpy kinds (driver_types.h).
pub const HIP_MEMCPY_HOST_TO_HOST: c_int = 0;
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
pub const HIP_MEMCPY_DEFAULT: c_int = 4;

/// Opaque HIP handles.
pub type HipStream = *mut c_void;
pub type HipEvent = *mut c_void;
pub type HipGraph = *mut c_void;
pub type HipGraphExec = *mut c_void;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipRtcProgram = *mut c_void;

/// Errors from the HIP layer.
#[derive(Debug, Clone, thiserror::Error)]
pub enum HipError {
    #[error("HIP call failed ({code}): {msg}")]
    Call { code: c_int, msg: String },
    #[error("HIP runtime library could not be loaded: {0}")]
    Library(String),
    #[error("symbol not found in HIP library: {0}")]
    Symbol(String),
    #[error("hiprtc failed ({code}): {msg}")]
    Rtc { code: c_int, msg: String },
    #[error("HIP device unavailable: {0}")]
    Device(String),
}

/// Raw function pointers loaded from `amdhip64_6.dll` / `hiprtc0602.dll`.
#[derive(Clone, Copy)]
pub struct HipApi {
    pub hip_get_device_count: unsafe extern "C" fn(*mut c_int) -> c_int,
    pub hip_set_device: unsafe extern "C" fn(c_int) -> c_int,
    pub hip_get_device_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int,
    pub hip_device_synchronize: unsafe extern "C" fn() -> c_int,
    pub hip_get_last_error: unsafe extern "C" fn() -> c_int,
    pub hip_get_error_string: unsafe extern "C" fn(c_int) -> *const c_char,
    pub hip_malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int,
    pub hip_free: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub hip_memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
    pub hip_memcpy_async:
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, HipStream) -> c_int,
    pub hip_memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int,
    pub hip_host_malloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> c_int,
    pub hip_host_free: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub hip_stream_create: unsafe extern "C" fn(*mut HipStream) -> c_int,
    pub hip_stream_destroy: unsafe extern "C" fn(HipStream) -> c_int,
    pub hip_stream_synchronize: unsafe extern "C" fn(HipStream) -> c_int,
    pub hip_event_create: unsafe extern "C" fn(*mut HipEvent) -> c_int,
    pub hip_event_destroy: unsafe extern "C" fn(HipEvent) -> c_int,
    pub hip_event_record: unsafe extern "C" fn(HipEvent, HipStream) -> c_int,
    pub hip_event_synchronize: unsafe extern "C" fn(HipEvent) -> c_int,
    pub hip_stream_begin_capture: unsafe extern "C" fn(HipStream, c_int) -> c_int,
    pub hip_stream_end_capture: unsafe extern "C" fn(HipStream, *mut HipGraph) -> c_int,
    pub hip_graph_instantiate:
        unsafe extern "C" fn(*mut HipGraphExec, HipGraph, *mut c_void, *mut c_char, usize) -> c_int,
    pub hip_graph_launch: unsafe extern "C" fn(HipGraphExec, HipStream) -> c_int,
    pub hip_graph_exec_destroy: unsafe extern "C" fn(HipGraphExec) -> c_int,
    pub hip_graph_destroy: unsafe extern "C" fn(HipGraph) -> c_int,
    pub hip_module_load_data: unsafe extern "C" fn(*mut HipModule, *const c_void) -> c_int,
    pub hip_module_get_function:
        unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> c_int,
    pub hip_module_unload: unsafe extern "C" fn(HipModule) -> c_int,
    pub hip_module_launch_kernel: unsafe extern "C" fn(
        HipFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        HipStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> c_int,
    pub hip_rtc_create_program: unsafe extern "C" fn(
        *mut HipRtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> c_int,
    pub hip_rtc_compile_program:
        unsafe extern "C" fn(HipRtcProgram, c_int, *const *const c_char) -> c_int,
    pub hip_rtc_get_code_size: unsafe extern "C" fn(HipRtcProgram, *mut usize) -> c_int,
    pub hip_rtc_get_code: unsafe extern "C" fn(HipRtcProgram, *mut c_char) -> c_int,
    pub hip_rtc_get_program_log_size: unsafe extern "C" fn(HipRtcProgram, *mut usize) -> c_int,
    pub hip_rtc_get_program_log: unsafe extern "C" fn(HipRtcProgram, *mut c_char) -> c_int,
    pub hip_rtc_destroy_program: unsafe extern "C" fn(*mut HipRtcProgram) -> c_int,
    pub hip_rtc_get_error_string: unsafe extern "C" fn(c_int) -> *const c_char,
}

/// Loaded HIP runtime (keeps the DLLs alive).
pub struct Hip {
    _hip_lib: libloading::Library,
    _rtc_lib: libloading::Library,
    /// Raw API pointers.
    pub api: HipApi,
}

// SAFETY: the DLL handles are kept alive by this struct and never unloaded
// while in use; all HIP driver calls we issue are thread-safe, so sharing the
// loaded runtime across threads is sound.
unsafe impl Send for Hip {}
unsafe impl Sync for Hip {}

/// Resolves the ROCm bin directory: `MACH_HIP_PATH`, else the newest
/// `C:\Program Files\AMD\ROCm\<ver>\bin` found on disk.
pub(crate) fn rocm_bin() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MACH_HIP_PATH") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Some(p);
        }
    }
    let base = PathBuf::from(r"C:\Program Files\AMD\ROCm");
    let Ok(entries) = std::fs::read_dir(&base) else {
        return None;
    };
    let mut vers: Vec<(u64, PathBuf)> = entries
        .filter_map(|e| {
            let e = e.ok()?;
            let name = e.file_name().to_string_lossy().into_owned();
            let v: u64 = name.trim_start_matches("6.").parse().ok()?;
            Some((v, e.path()))
        })
        .collect();
    vers.sort();
    vers.last().map(|(_, p)| p.join("bin"))
}

/// Loads the first library that succeeds from a list of names/paths.
fn load_first(candidates: &[String]) -> Result<libloading::Library, HipError> {
    let mut last_err = None;
    for c in candidates {
        match unsafe { libloading::Library::new(c.as_str()) } {
            Ok(l) => return Ok(l),
            Err(e) => last_err = Some(format!("{c}: {e}")),
        }
    }
    Err(HipError::Library(
        last_err.unwrap_or_else(|| "no candidates".into()),
    ))
}

/// Loads a symbol `$sym` (exported name) into field `$field` of `$api`.
/// Loads a symbol by its exported name; `T` is inferred from the field type.
fn sym<T: Copy>(lib: &libloading::Library, name: &str) -> Result<T, HipError> {
    unsafe { lib.get::<T>(name.as_bytes()) }
        .map(|s| *s)
        .map_err(|e| HipError::Symbol(format!("{name}({e})")))
}

/// Prepends the ROCm bin directory to `PATH` so `LoadLibrary` can resolve
/// the HIP/hipBLAS DLLs and their dependencies (rocblas, etc.).
pub(crate) fn ensure_rocm_on_path() {
    if let Some(bin) = rocm_bin() {
        let bin_s = bin.to_string_lossy().into_owned();
        let path = std::env::var("PATH").unwrap_or_default();
        if !path.split(';').any(|p| p.eq_ignore_ascii_case(&bin_s)) {
            // SAFETY: single-threaded at first DLL load; no other env reads race.
            unsafe { std::env::set_var("PATH", format!("{bin_s};{path}")) };
        }
    }
}

fn load() -> Result<Arc<Hip>, HipError> {
    ensure_rocm_on_path();
    let bin = rocm_bin();
    let mut hip_candidates: Vec<String> = Vec::new();
    if let Some(b) = &bin {
        hip_candidates.push(b.join("amdhip64_6.dll").to_string_lossy().into_owned());
        hip_candidates.push(b.join("amdhip64.dll").to_string_lossy().into_owned());
    }
    hip_candidates.push("amdhip64_6.dll".into());
    hip_candidates.push("amdhip64.dll".into());
    let hip_lib = load_first(&hip_candidates)?;

    let mut rtc_candidates: Vec<String> = Vec::new();
    if let Some(b) = &bin {
        rtc_candidates.push(b.join("hiprtc0602.dll").to_string_lossy().into_owned());
    }
    rtc_candidates.push("hiprtc0602.dll".into());
    let rtc_lib = load_first(&rtc_candidates)?;

    let api = HipApi {
        hip_get_device_count: sym(&hip_lib, "hipGetDeviceCount")?,
        hip_set_device: sym(&hip_lib, "hipSetDevice")?,
        hip_get_device_name: sym(&hip_lib, "hipDeviceGetName")?,
        hip_device_synchronize: sym(&hip_lib, "hipDeviceSynchronize")?,
        hip_get_last_error: sym(&hip_lib, "hipGetLastError")?,
        hip_get_error_string: sym(&hip_lib, "hipGetErrorString")?,
        hip_malloc: sym(&hip_lib, "hipMalloc")?,
        hip_free: sym(&hip_lib, "hipFree")?,
        hip_memcpy: sym(&hip_lib, "hipMemcpy")?,
        hip_memcpy_async: sym(&hip_lib, "hipMemcpyAsync")?,
        hip_memset: sym(&hip_lib, "hipMemset")?,
        hip_host_malloc: sym(&hip_lib, "hipHostMalloc")?,
        hip_host_free: sym(&hip_lib, "hipHostFree")?,
        hip_stream_create: sym(&hip_lib, "hipStreamCreate")?,
        hip_stream_destroy: sym(&hip_lib, "hipStreamDestroy")?,
        hip_stream_synchronize: sym(&hip_lib, "hipStreamSynchronize")?,
        hip_event_create: sym(&hip_lib, "hipEventCreate")?,
        hip_event_destroy: sym(&hip_lib, "hipEventDestroy")?,
        hip_event_record: sym(&hip_lib, "hipEventRecord")?,
        hip_event_synchronize: sym(&hip_lib, "hipEventSynchronize")?,
        hip_stream_begin_capture: sym(&hip_lib, "hipStreamBeginCapture")?,
        hip_stream_end_capture: sym(&hip_lib, "hipStreamEndCapture")?,
        hip_graph_instantiate: sym(&hip_lib, "hipGraphInstantiate")?,
        hip_graph_launch: sym(&hip_lib, "hipGraphLaunch")?,
        hip_graph_exec_destroy: sym(&hip_lib, "hipGraphExecDestroy")?,
        hip_graph_destroy: sym(&hip_lib, "hipGraphDestroy")?,
        hip_module_load_data: sym(&hip_lib, "hipModuleLoadData")?,
        hip_module_get_function: sym(&hip_lib, "hipModuleGetFunction")?,
        hip_module_unload: sym(&hip_lib, "hipModuleUnload")?,
        hip_module_launch_kernel: sym(&hip_lib, "hipModuleLaunchKernel")?,
        hip_rtc_create_program: sym(&rtc_lib, "hiprtcCreateProgram")?,
        hip_rtc_compile_program: sym(&rtc_lib, "hiprtcCompileProgram")?,
        hip_rtc_get_code_size: sym(&rtc_lib, "hiprtcGetCodeSize")?,
        hip_rtc_get_code: sym(&rtc_lib, "hiprtcGetCode")?,
        hip_rtc_get_program_log_size: sym(&rtc_lib, "hiprtcGetProgramLogSize")?,
        hip_rtc_get_program_log: sym(&rtc_lib, "hiprtcGetProgramLog")?,
        hip_rtc_destroy_program: sym(&rtc_lib, "hiprtcDestroyProgram")?,
        hip_rtc_get_error_string: sym(&rtc_lib, "hiprtcGetErrorString")?,
    };

    Ok(Arc::new(Hip {
        _hip_lib: hip_lib,
        _rtc_lib: rtc_lib,
        api,
    }))
}

static HIP: OnceLock<Result<Arc<Hip>, HipError>> = OnceLock::new();

/// Returns the process-wide HIP runtime handle (loaded once).
pub fn hip() -> Result<Arc<Hip>, HipError> {
    HIP.get_or_init(load).clone()
}

/// Maps a HIP error code to a message.
pub fn error_string(h: &Hip, code: c_int) -> String {
    unsafe {
        let p = (h.api.hip_get_error_string)(code);
        if p.is_null() {
            format!("unknown error {code}")
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

/// Converts an f32 to fp16 bits (round-to-nearest-even). Host-side mirror of
/// the device cast; kept here so kernel-sys tests can prepare fp16 operands.
#[must_use]
pub fn fp32_to_f16_host(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let mut e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = 14 - e;
        let half = 1u32 << (shift - 1);
        let mut m16 = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && (m16 & 1) == 1) {
            m16 += 1;
        }
        return sign | m16 as u16;
    }
    let mut m16 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (m16 & 1) == 1) {
        m16 += 1;
        if m16 == 0x400 {
            m16 = 0;
            e += 1;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | (((e as u16) & 0x1f) << 10) | (m16 as u16)
}

/// Expands fp16 bits to f32 (exact).
#[must_use]
pub fn fp16_to_f32_host(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x03ff) as u32;
    let b = if e == 0 {
        if m == 0 {
            sign
        } else {
            let mut m2 = m;
            let mut e2 = 127u32 - 15 + 1;
            while (m2 & 0x0400) == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            sign | (e2 << 23) | ((m2 & 0x03ff) << 13)
        }
    } else if e == 0x1f {
        sign | 0x7f80_0000 | (m << 13)
    } else {
        sign | (((e as i32 - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(b)
}

/// Converts a HIP/hiprtc return code into `Ok`/`Err` with a message.
pub fn check(h: &Hip, code: c_int) -> Result<(), HipError> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(HipError::Call {
            code,
            msg: error_string(h, code),
        })
    }
}

/// Process-wide HIP device count (0 when no device / library missing).
pub fn device_count() -> Result<i32, HipError> {
    let h = hip()?;
    let mut count = 0;
    unsafe { check(&h, (h.api.hip_get_device_count)(&mut count))? };
    Ok(count)
}

/// Returns the name of `device`.
pub fn device_name(device: i32) -> Result<String, HipError> {
    let h = hip()?;
    let mut buf = [0i8; 256];
    unsafe {
        check(
            &h,
            (h.api.hip_get_device_name)(buf.as_mut_ptr(), buf.len() as c_int, device),
        )?
    };
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let bytes: Vec<u8> = buf[..end].iter().map(|&b| b as u8).collect();
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

// ---------------------------------------------------------------------------
// Safe helpers
// ---------------------------------------------------------------------------

/// Allocates device memory of `bytes` bytes.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn malloc(h: &Hip, bytes: usize) -> Result<*mut c_void, HipError> {
    let mut ptr = std::ptr::null_mut();
    unsafe { check(h, (h.api.hip_malloc)(&mut ptr, bytes))? };
    Ok(ptr)
}

/// Frees device memory allocated by [`malloc`].
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn free(h: &Hip, ptr: *mut c_void) -> Result<(), HipError> {
    unsafe { check(h, (h.api.hip_free)(ptr)) }
}

/// Copies `bytes` between host/device according to `kind`, enqueued on `stream`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn memcpy_async(
    h: &Hip,
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    kind: c_int,
    stream: HipStream,
) -> Result<(), HipError> {
    unsafe { check(h, (h.api.hip_memcpy_async)(dst, src, bytes, kind, stream)) }
}

/// Allocates pinned host memory (for async H2D/D2H).
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn host_malloc(h: &Hip, bytes: usize) -> Result<*mut c_void, HipError> {
    let mut ptr = std::ptr::null_mut();
    unsafe { check(h, (h.api.hip_host_malloc)(&mut ptr, bytes, 0))? };
    Ok(ptr)
}

/// Frees pinned host memory allocated by [`host_malloc`].
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn host_free(h: &Hip, ptr: *mut c_void) -> Result<(), HipError> {
    unsafe { check(h, (h.api.hip_host_free)(ptr)) }
}

/// Copies `bytes` between host/device according to `kind`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn memcpy(
    h: &Hip,
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    kind: c_int,
) -> Result<(), HipError> {
    unsafe { check(h, (h.api.hip_memcpy)(dst, src, bytes, kind)) }
}

/// A runtime-compiled HIP kernel (hiprtc -> hipModule), with the launchable
/// function resolved at compile time.
pub struct HipKernelModule {
    handle: HipModule,
    func: HipFunction,
    _hip: Arc<Hip>,
}

impl HipKernelModule {
    /// Compiles `source` for `arch` (e.g. `"gfx1100"`), loads it, and resolves
    /// `kernel_name` as the launchable function.
    pub fn compile(arch: &str, source: &str, kernel_name: &str) -> Result<Self, HipError> {
        let h = hip()?;
        let mut prog: HipRtcProgram = std::ptr::null_mut();
        let src = std::ffi::CString::new(source).map_err(|_| HipError::Rtc {
            code: -1,
            msg: "kernel source contains NUL byte".into(),
        })?;
        let name = std::ffi::CString::new("mach_kernel.cpp").map_err(|_| HipError::Rtc {
            code: -1,
            msg: "bad name".into(),
        })?;
        let r = unsafe {
            (h.api.hip_rtc_create_program)(
                &mut prog,
                src.as_ptr(),
                name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if r != 0 {
            return Err(HipError::Rtc {
                code: r,
                msg: rtc_msg(&h, r),
            });
        }

        let opt = std::ffi::CString::new(format!("--offload-arch={arch}")).map_err(|_| {
            HipError::Rtc {
                code: -1,
                msg: "bad arch".into(),
            }
        })?;
        let opts = [opt.as_ptr()];
        let r =
            unsafe { (h.api.hip_rtc_compile_program)(prog, opts.len() as c_int, opts.as_ptr()) };
        if r != 0 {
            let log = rtc_log(&h, prog);
            unsafe { (h.api.hip_rtc_destroy_program)(&mut prog) };
            return Err(HipError::Rtc {
                code: r,
                msg: format!("{}; log: {}", rtc_msg(&h, r), log),
            });
        }

        let mut size = 0usize;
        unsafe { (h.api.hip_rtc_get_code_size)(prog, &mut size) };
        let mut code = vec![0u8; size];
        let r = unsafe { (h.api.hip_rtc_get_code)(prog, code.as_mut_ptr() as *mut c_char) };
        unsafe { (h.api.hip_rtc_destroy_program)(&mut prog) };
        if r != 0 {
            return Err(HipError::Rtc {
                code: r,
                msg: rtc_msg(&h, r),
            });
        }

        let mut module: HipModule = std::ptr::null_mut();
        unsafe {
            check(
                &h,
                (h.api.hip_module_load_data)(&mut module, code.as_ptr() as *const c_void),
            )?
        };

        let kname = std::ffi::CString::new(kernel_name).map_err(|_| HipError::Rtc {
            code: -1,
            msg: "bad kernel name".into(),
        })?;
        let mut func: HipFunction = std::ptr::null_mut();
        unsafe {
            check(
                &h,
                (h.api.hip_module_get_function)(&mut func, module, kname.as_ptr()),
            )?
        };

        Ok(Self {
            handle: module,
            func,
            _hip: h,
        })
    }

    /// Launches the kernel. `params` holds one pointer per kernel argument,
    /// each pointing at the argument value (HIP convention).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn launch_shmem(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        params: &mut [*mut c_void],
        stream: HipStream,
        shared_bytes: u32,
    ) -> Result<(), HipError> {
        let h = &self._hip;
        unsafe {
            check(
                h,
                (h.api.hip_module_launch_kernel)(
                    self.func,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_bytes,
                    stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )
        }
    }

    /// Launches the kernel with no dynamic shared memory.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn launch(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        params: &mut [*mut c_void],
        stream: HipStream,
    ) -> Result<(), HipError> {
        self.launch_shmem(grid, block, params, stream, 0)
    }
}

impl Drop for HipKernelModule {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = (self._hip.api.hip_module_unload)(self.handle);
            }
        }
    }
}

fn rtc_msg(h: &Hip, code: c_int) -> String {
    unsafe {
        let p = (h.api.hip_rtc_get_error_string)(code);
        if p.is_null() {
            format!("hiprtc error {code}")
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn rtc_log(h: &Hip, prog: HipRtcProgram) -> String {
    unsafe {
        let mut size = 0usize;
        if (h.api.hip_rtc_get_program_log_size)(prog, &mut size) != 0 || size == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; size + 1];
        if (h.api.hip_rtc_get_program_log)(prog, buf.as_mut_ptr() as *mut c_char) != 0 {
            return String::new();
        }
        buf.pop();
        String::from_utf8_lossy(&buf).into_owned()
    }
}
