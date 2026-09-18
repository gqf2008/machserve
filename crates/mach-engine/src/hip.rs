//! ROCm/HIP backend: HIP graph capture on AMD GPUs.
//!
//! Implemented on top of `mach_kernel_sys::hip` (the single FFI boundary).
//! The capture lifecycle reuses the strict `NoCapture → Prepare → Capture`
//! state machine from [`crate::graph`]; this module only supplies the driver
//! work each transition brackets (`hipStreamBeginCapture` / `EndCapture`,
//! `hipGraphInstantiate` / `hipGraphLaunch`).

use crate::Error;
use crate::graph::{CaptureState, GraphCapture, GraphError, GraphHandle};
use mach_kernel_sys::hip::{self, Hip, HipGraphExec, HipStream};
use std::sync::Mutex;

/// Send+Sync wrapper for raw HIP handles. HIP driver handles are thread-safe
/// for the operations we issue; access to a capture stream is serialized by
/// the engine's per-stream locking discipline.
#[derive(Clone, Copy)]
struct HipHandle(*mut core::ffi::c_void);
unsafe impl Send for HipHandle {}
unsafe impl Sync for HipHandle {}

/// Default HIP offload architecture for the P1 target (RX 7900 XTX, RDNA3).
/// Override with the `MACH_HIP_ARCH` environment variable.
pub const DEFAULT_HIP_ARCH: &str = "gfx1100";

/// Returns the HIP offload arch to compile kernels for.
#[must_use]
pub fn hip_arch() -> String {
    std::env::var("MACH_HIP_ARCH").unwrap_or_else(|_| DEFAULT_HIP_ARCH.to_string())
}

pub struct HipGraphCapture {
    hip: std::sync::Arc<Hip>,
    stream: HipHandle,
    state: Mutex<CaptureState>,
    /// Whether this instance created the stream (and must destroy it).
    owns_stream: bool,
}

impl core::fmt::Debug for HipGraphCapture {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HipGraphCapture").finish_non_exhaustive()
    }
}

impl HipGraphCapture {
    /// Creates a capture backend with a fresh stream.
    pub fn new(hip: std::sync::Arc<Hip>) -> Result<Self, Error> {
        let mut stream = std::ptr::null_mut();
        unsafe { hip::check(&hip, (hip.api.hip_stream_create)(&mut stream))? };
        Self::with_owned_stream(hip, stream)
    }

    /// Creates a capture backend over an existing stream. The stream must be
    /// the one kernels are launched on; otherwise capture records nothing.
    /// The caller owns the stream's lifetime.
    pub fn with_stream(hip: std::sync::Arc<Hip>, stream: HipStream) -> Result<Self, Error> {
        Ok(Self {
            hip,
            stream: HipHandle(stream),
            state: Mutex::new(CaptureState::NoCapture),
            owns_stream: false,
        })
    }

    /// Internal constructor for the owning case.
    fn with_owned_stream(hip: std::sync::Arc<Hip>, stream: HipStream) -> Result<Self, Error> {
        Ok(Self {
            hip,
            stream: HipHandle(stream),
            state: Mutex::new(CaptureState::NoCapture),
            owns_stream: true,
        })
    }

    /// The capture stream (also used for replay).
    #[must_use]
    pub fn stream(&self) -> HipStream {
        self.stream.0
    }
}

impl Drop for HipGraphCapture {
    fn drop(&mut self) {
        if self.owns_stream && !self.stream.0.is_null() {
            unsafe {
                let _ = (self.hip.api.hip_stream_destroy)(self.stream.0);
            }
        }
    }
}

impl GraphCapture for HipGraphCapture {
    fn supported(&self) -> bool {
        true
    }

    fn prepare(&self) -> Result<(), GraphError> {
        let mut state = self.state.lock().unwrap();
        *state = state.prepare()?;
        Ok(())
    }

    fn begin(&self) -> Result<(), GraphError> {
        let mut state = self.state.lock().unwrap();
        let next = state.begin()?;
        *state = next;
        let r = unsafe {
            (self.hip.api.hip_stream_begin_capture)(
                self.stream.0,
                hip::HIP_STREAM_CAPTURE_MODE_GLOBAL,
            )
        };
        if r != hip::HIP_SUCCESS {
            *state = state.abort();
            return Err(GraphError::Driver(format!(
                "hipStreamBeginCapture: {r} {}",
                hip::error_string(&self.hip, r)
            )));
        }
        Ok(())
    }

    fn end(&self) -> Result<Box<dyn GraphHandle>, GraphError> {
        let mut state = self.state.lock().unwrap();
        let next = state.end()?;
        *state = next;
        let mut graph = std::ptr::null_mut();
        let r = unsafe { (self.hip.api.hip_stream_end_capture)(self.stream.0, &mut graph) };
        if r != hip::HIP_SUCCESS {
            // A failed end_capture leaves the stream in capture mode with no
            // public abort; the instance's stream is unusable afterwards and
            // the caller should drop this capture backend.
            return Err(GraphError::Driver(format!(
                "hipStreamEndCapture: {r} {}",
                hip::error_string(&self.hip, r)
            )));
        }
        let mut exec: HipGraphExec = std::ptr::null_mut();
        let r = unsafe {
            (self.hip.api.hip_graph_instantiate)(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if !graph.is_null() {
            unsafe {
                let _ = (self.hip.api.hip_graph_destroy)(graph);
            }
        }
        if r != hip::HIP_SUCCESS {
            return Err(GraphError::Driver(format!(
                "hipGraphInstantiate: {r} {}",
                hip::error_string(&self.hip, r)
            )));
        }
        // Upload the exec to the device up front rather than lazily inside
        // the first `hipGraphLaunch`: the device image is deterministic
        // before any replay and the first launch does not pay upload cost.
        // (Measured on ROCm 6.2 / Windows this does NOT prevent the large-
        // graph replay degeneration seen in issue #103 — kept because the
        // explicit upload is the correct, deterministic pattern.)
        let r = unsafe { (self.hip.api.hip_graph_upload)(exec, self.stream.0) };
        if r != hip::HIP_SUCCESS {
            unsafe {
                let _ = (self.hip.api.hip_graph_exec_destroy)(exec);
            }
            return Err(GraphError::Driver(format!(
                "hipGraphUpload: {r} {}",
                hip::error_string(&self.hip, r)
            )));
        }
        Ok(Box::new(HipGraph {
            hip: std::sync::Arc::clone(&self.hip),
            exec: HipHandle(exec),
            stream: self.stream,
        }))
    }

    fn abort(&self) -> Result<(), GraphError> {
        let mut state = self.state.lock().unwrap();
        *state = state.abort();
        Ok(())
    }
}

/// An instantiated HIP executable graph.
pub struct HipGraph {
    hip: std::sync::Arc<Hip>,
    exec: HipHandle,
    stream: HipHandle,
}

impl core::fmt::Debug for HipGraph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HipGraph").finish_non_exhaustive()
    }
}

impl GraphHandle for HipGraph {
    unsafe fn replay(&self) -> Result<(), GraphError> {
        let r = unsafe { (self.hip.api.hip_graph_launch)(self.exec.0, self.stream.0) };
        if r != hip::HIP_SUCCESS {
            let msg = hip::error_string(&self.hip, r);
            return Err(GraphError::Driver(format!("hipGraphLaunch: {r} {msg}")));
        }
        Ok(())
    }
}

impl Drop for HipGraph {
    fn drop(&mut self) {
        if !self.exec.0.is_null() {
            unsafe {
                let _ = (self.hip.api.hip_graph_exec_destroy)(self.exec.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mach_kernel_sys::hip::{HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE};
    use std::sync::Arc;

    const SAXPY_SRC: &str = r#"
extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"#;

    /// Returns the loaded HIP runtime with device 0 selected, else `None` (skip).
    fn dev() -> Option<Arc<Hip>> {
        let h = match hip::hip() {
            Ok(h) => h,
            Err(e) => {
                eprintln!("skipping HIP test: {e}");
                return None;
            }
        };
        match hip::device_count() {
            Ok(n) if n > 0 => {}
            Ok(n) => {
                eprintln!("skipping HIP test: device_count={n}");
                return None;
            }
            Err(e) => {
                eprintln!("skipping HIP test: {e}");
                return None;
            }
        }
        unsafe {
            if let Err(e) = hip::check(&h, (h.api.hip_set_device)(0)) {
                eprintln!("skipping HIP test: {e}");
                return None;
            }
        }
        Some(h)
    }

    struct DevBuf {
        hip: Arc<Hip>,
        ptr: *mut core::ffi::c_void,
    }

    impl DevBuf {
        fn alloc(h: &Arc<Hip>, bytes: usize) -> Self {
            let ptr = hip::malloc(h, bytes).unwrap();
            Self {
                hip: Arc::clone(h),
                ptr,
            }
        }
    }

    impl Drop for DevBuf {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                let _ = hip::free(&self.hip, self.ptr);
            }
        }
    }

    #[test]
    fn hip_device_is_visible() {
        let Some(_h) = dev() else { return };
        let name = hip::device_name(0).unwrap();
        assert!(!name.is_empty());
        eprintln!("HIP device 0: {name}");
    }

    #[test]
    fn hiprtc_saxpy_runs_on_gpu() {
        let Some(h) = dev() else { return };
        let h = &h;
        let n: i32 = 1 << 20;
        let x = DevBuf::alloc(h, n as usize * 4);
        let y = DevBuf::alloc(h, n as usize * 4);

        let mut hx = vec![1.0f32; n as usize];
        let mut hy = vec![1.0f32; n as usize];
        hip::memcpy(
            h,
            x.ptr,
            hx.as_mut_ptr() as *const _,
            (n * 4) as usize,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        hip::memcpy(
            h,
            y.ptr,
            hy.as_mut_ptr() as *const _,
            (n * 4) as usize,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();

        let module =
            hip::HipKernelModule::compile(&hip_arch(), SAXPY_SRC, "saxpy").expect("hiprtc compile");
        let a: f32 = 2.0;
        let xp = x.ptr;
        let yp = y.ptr;
        let mut params: Vec<*mut core::ffi::c_void> = vec![
            &a as *const f32 as *mut core::ffi::c_void,
            &xp as *const *mut core::ffi::c_void as *mut core::ffi::c_void,
            &yp as *const *mut core::ffi::c_void as *mut core::ffi::c_void,
            &n as *const i32 as *mut core::ffi::c_void,
        ];
        module
            .launch(
                [n as u32 / 256, 1, 1],
                [256, 1, 1],
                &mut params,
                std::ptr::null_mut(),
            )
            .expect("launch");
        unsafe {
            hip::check(h, (h.api.hip_device_synchronize)()).unwrap();
        }
        hip::memcpy(
            h,
            hy.as_mut_ptr() as *mut _,
            y.ptr as *const _,
            (n * 4) as usize,
            HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        assert!(
            hy.iter().all(|&v| v == 3.0),
            "saxpy expected 3.0, got {:?}",
            &hy[..4]
        );
    }

    #[test]
    fn hip_graph_capture_records_and_replays() {
        let Some(h) = dev() else { return };
        let h = &h;
        let n: i32 = 1 << 20;
        let x = DevBuf::alloc(h, n as usize * 4);
        let y = DevBuf::alloc(h, n as usize * 4);

        let mut hx = vec![1.0f32; n as usize];
        let mut hy = vec![0.0f32; n as usize];
        hip::memcpy(
            h,
            x.ptr,
            hx.as_mut_ptr() as *const _,
            (n * 4) as usize,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        hip::memcpy(
            h,
            y.ptr,
            hy.as_mut_ptr() as *const _,
            (n * 4) as usize,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();

        let module =
            hip::HipKernelModule::compile(&hip_arch(), SAXPY_SRC, "saxpy").expect("hiprtc compile");
        let a: f32 = 2.0;
        let xp = x.ptr;
        let yp = y.ptr;

        let cap = HipGraphCapture::new(Arc::clone(h)).unwrap();
        cap.prepare().unwrap();
        cap.begin().unwrap();
        {
            let mut params: Vec<*mut core::ffi::c_void> = vec![
                &a as *const f32 as *mut core::ffi::c_void,
                &xp as *const *mut core::ffi::c_void as *mut core::ffi::c_void,
                &yp as *const *mut core::ffi::c_void as *mut core::ffi::c_void,
                &n as *const i32 as *mut core::ffi::c_void,
            ];
            // Recorded into the graph instead of executed.
            module
                .launch(
                    [n as u32 / 256, 1, 1],
                    [256, 1, 1],
                    &mut params,
                    cap.stream(),
                )
                .expect("launch during capture");
        }
        let graph = cap.end().unwrap();

        // SAFETY: x/y are alive; replays are serialized on this thread and a
        // device sync orders the final read.
        unsafe { graph.replay().unwrap() };
        unsafe { graph.replay().unwrap() };
        unsafe {
            hip::check(h, (h.api.hip_device_synchronize)()).unwrap();
        }
        hip::memcpy(
            h,
            hy.as_mut_ptr() as *mut _,
            y.ptr as *const _,
            (n * 4) as usize,
            HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        // y starts at 0; two replays: 0 -> 2 -> 4.
        assert!(
            hy.iter().all(|&v| v == 4.0),
            "graph replay expected 4.0 after two replays, got {:?}",
            &hy[..4]
        );
    }

    #[test]
    fn hip_graph_lifecycle_is_strict() {
        let Some(h) = dev() else { return };
        let cap = HipGraphCapture::new(h).unwrap();
        assert!(cap.begin().is_err(), "begin before prepare must fail");
        cap.prepare().unwrap();
        cap.begin().unwrap();
        let graph = cap.end().unwrap();
        drop(graph);
        cap.prepare().unwrap();
    }
}
