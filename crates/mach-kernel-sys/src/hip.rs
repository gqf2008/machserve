//! HIP runtime FFI for the ROCm backend (Windows amdhip64_6.dll + hiprtc0602.dll).
//!
//! The DLLs are loaded **dynamically** at runtime (no link-time dependency), so
//! the crate builds with any toolchain (including GNU) and only fails when the
//! `hip` feature is used on a machine without ROCm. This is the only place the
//! runtime talks to the HIP driver/runtime.
//!
//! ABI constants were verified against `C:\Program Files\AMD\ROCm\6.2\include\hip\`
//! (hip_runtime_api.h / driver_types.h / hiprtc.h).

use std::ffi::{c_char, c_int, c_uint, c_void};
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
    pub hip_memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int,
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

/// Loads a symbol from `lib`, returning it as the fn-pointer type `$ty`.
macro_rules! get {
    ($lib:expr, $name:ident, $ty:ty) => {
        *(unsafe { $lib.get::<$ty>(concat!(stringify!($name), "\0").as_bytes()) }
            .map_err(|e| HipError::Symbol(format!("{}({e})", stringify!($name))))?)
    };
}

fn load() -> Result<Arc<Hip>, HipError> {
    let mut hip_lib = None;
    for n in ["amdhip64_6.dll", "amdhip64.dll"] {
        if let Ok(l) = unsafe { libloading::Library::new(n) } {
            hip_lib = Some(l);
            break;
        }
    }
    let hip_lib = hip_lib.ok_or_else(|| {
        HipError::Library("amdhip64_6.dll / amdhip64.dll not found (ROCm not installed?)".into())
    })?;

    let api = HipApi {
        hip_get_device_count: get!(
            hip_lib,
            hip_get_device_count,
            unsafe extern "C" fn(*mut c_int) -> c_int
        ),
        hip_set_device: get!(
            hip_lib,
            hip_set_device,
            unsafe extern "C" fn(c_int) -> c_int
        ),
        hip_get_device_name: get!(
            hip_lib,
            hip_get_device_name,
            unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int
        ),
        hip_device_synchronize: get!(
            hip_lib,
            hip_device_synchronize,
            unsafe extern "C" fn() -> c_int
        ),
        hip_get_last_error: get!(hip_lib, hip_get_last_error, unsafe extern "C" fn() -> c_int),
        hip_get_error_string: get!(
            hip_lib,
            hip_get_error_string,
            unsafe extern "C" fn(c_int) -> *const c_char
        ),
        hip_malloc: get!(
            hip_lib,
            hip_malloc,
            unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int
        ),
        hip_free: get!(
            hip_lib,
            hip_free,
            unsafe extern "C" fn(*mut c_void) -> c_int
        ),
        hip_memcpy: get!(
            hip_lib,
            hip_memcpy,
            unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int
        ),
        hip_memset: get!(
            hip_lib,
            hip_memset,
            unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int
        ),
        hip_stream_create: get!(
            hip_lib,
            hip_stream_create,
            unsafe extern "C" fn(*mut HipStream) -> c_int
        ),
        hip_stream_destroy: get!(
            hip_lib,
            hip_stream_destroy,
            unsafe extern "C" fn(HipStream) -> c_int
        ),
        hip_stream_synchronize: get!(
            hip_lib,
            hip_stream_synchronize,
            unsafe extern "C" fn(HipStream) -> c_int
        ),
        hip_event_create: get!(
            hip_lib,
            hip_event_create,
            unsafe extern "C" fn(*mut HipEvent) -> c_int
        ),
        hip_event_destroy: get!(
            hip_lib,
            hip_event_destroy,
            unsafe extern "C" fn(HipEvent) -> c_int
        ),
        hip_event_record: get!(
            hip_lib,
            hip_event_record,
            unsafe extern "C" fn(HipEvent, HipStream) -> c_int
        ),
        hip_event_synchronize: get!(
            hip_lib,
            hip_event_synchronize,
            unsafe extern "C" fn(HipEvent) -> c_int
        ),
        hip_stream_begin_capture: get!(
            hip_lib,
            hip_stream_begin_capture,
            unsafe extern "C" fn(HipStream, c_int) -> c_int
        ),
        hip_stream_end_capture: get!(
            hip_lib,
            hip_stream_end_capture,
            unsafe extern "C" fn(HipStream, *mut HipGraph) -> c_int
        ),
        hip_graph_instantiate: get!(
            hip_lib,
            hip_graph_instantiate,
            unsafe extern "C" fn(
                *mut HipGraphExec,
                HipGraph,
                *mut c_void,
                *mut c_char,
                usize,
            ) -> c_int
        ),
        hip_graph_launch: get!(
            hip_lib,
            hip_graph_launch,
            unsafe extern "C" fn(HipGraphExec, HipStream) -> c_int
        ),
        hip_graph_exec_destroy: get!(
            hip_lib,
            hip_graph_exec_destroy,
            unsafe extern "C" fn(HipGraphExec) -> c_int
        ),
        hip_graph_destroy: get!(
            hip_lib,
            hip_graph_destroy,
            unsafe extern "C" fn(HipGraph) -> c_int
        ),
        hip_module_load_data: get!(
            hip_lib,
            hip_module_load_data,
            unsafe extern "C" fn(*mut HipModule, *const c_void) -> c_int
        ),
        hip_module_get_function: get!(
            hip_lib,
            hip_module_get_function,
            unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> c_int
        ),
        hip_module_unload: get!(
            hip_lib,
            hip_module_unload,
            unsafe extern "C" fn(HipModule) -> c_int
        ),
        hip_module_launch_kernel: get!(
            hip_lib,
            hip_module_launch_kernel,
            unsafe extern "C" fn(
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
            ) -> c_int
        ),
        hip_rtc_create_program: get!(
            hip_lib,
            hip_rtc_create_program,
            unsafe extern "C" fn(
                *mut HipRtcProgram,
                *const c_char,
                *const c_char,
                c_int,
                *const *const c_char,
                *const *const c_char,
            ) -> c_int
        ),
        hip_rtc_compile_program: get!(
            hip_lib,
            hip_rtc_compile_program,
            unsafe extern "C" fn(HipRtcProgram, c_int, *const *const c_char) -> c_int
        ),
        hip_rtc_get_code_size: get!(
            hip_lib,
            hip_rtc_get_code_size,
            unsafe extern "C" fn(HipRtcProgram, *mut usize) -> c_int
        ),
        hip_rtc_get_code: get!(
            hip_lib,
            hip_rtc_get_code,
            unsafe extern "C" fn(HipRtcProgram, *mut c_char) -> c_int
        ),
        hip_rtc_get_program_log_size: get!(
            hip_lib,
            hip_rtc_get_program_log_size,
            unsafe extern "C" fn(HipRtcProgram, *mut usize) -> c_int
        ),
        hip_rtc_get_program_log: get!(
            hip_lib,
            hip_rtc_get_program_log,
            unsafe extern "C" fn(HipRtcProgram, *mut c_char) -> c_int
        ),
        hip_rtc_destroy_program: get!(
            hip_lib,
            hip_rtc_destroy_program,
            unsafe extern "C" fn(*mut HipRtcProgram) -> c_int
        ),
        hip_rtc_get_error_string: get!(
            hip_lib,
            hip_rtc_get_error_string,
            unsafe extern "C" fn(c_int) -> *const c_char
        ),
    };

    let rtc_lib = unsafe {
        libloading::Library::new("hiprtc0602.dll")
            .map_err(|e| HipError::Library(format!("hiprtc0602.dll not found ({e})")))?
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
    pub fn launch(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        params: &mut [*mut c_void],
        stream: HipStream,
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
                    0,
                    stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )
        }
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
