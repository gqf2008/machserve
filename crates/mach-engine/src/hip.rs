//! ROCm/HIP backend: memory pool, streams and graph capture on AMD GPUs.
//!
//! Implemented on top of `mach_kernel_sys::hip` (the single FFI boundary).
//! The capture lifecycle reuses the strict `NoCapture → Prepare → Capture`
//! state machine from [`crate::graph`]; this module only supplies the driver
//! work each transition brackets (`hipStreamBeginCapture` / `EndCapture`,
//! `hipGraphInstantiate` / `hipGraphLaunch`).

use crate::Error;
use crate::graph::{CaptureState, GraphCapture, GraphError, GraphHandle};
use crate::memory::{Allocation, MemoryPool};
use mach_kernel_sys::hip::{self, Hip, HipGraphExec, HipStream};
use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Initialized HIP context for one device.
pub struct HipContext {
    /// Loaded HIP runtime.
    pub hip: std::sync::Arc<Hip>,
    /// Device ordinal.
    pub device: i32,
    /// Device name.
    pub name: String,
}

impl HipContext {
    /// Initializes HIP on `device`. Fails when no ROCm device is present.
    pub fn init(device: i32) -> Result<Self, Error> {
        let hip = hip::hip()?;
        let count = hip::device_count()?;
        if count <= 0 || device >= count {
            return Err(Error::BackendUnavailable(format!(
                "no HIP device (count={count}, requested={device})"
            )));
        }
        unsafe {
            hip::check(&hip, (hip.api.hip_set_device)(device))?;
        }
        let name = hip::device_name(device)?;
        Ok(Self { hip, device, name })
    }
}

/// HIP device memory pool: malloc/free + pin tracking (graph capture keeps
/// captured buffers pinned so the pool never hands them to a later alloc).
pub struct HipMemoryPool {
    hip: std::sync::Arc<Hip>,
    inner: Mutex<Inner>,
    pool_id: u64,
}

#[derive(Debug, Default)]
struct Inner {
    in_use: usize,
    pinned: HashSet<usize>,
}

impl core::fmt::Debug for HipMemoryPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HipMemoryPool")
            .field("pool_id", &self.pool_id)
            .field("bytes_in_use", &self.bytes_in_use())
            .finish()
    }
}

impl HipMemoryPool {
    /// Creates a pool over the given HIP runtime.
    #[must_use]
    pub fn new(hip: std::sync::Arc<Hip>) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            hip,
            inner: Mutex::new(Inner::default()),
            pool_id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl MemoryPool for HipMemoryPool {
    fn allocate(&self, bytes: usize, _align: usize) -> Result<Allocation, Error> {
        if bytes == 0 {
            return Err(Error::InvalidArgument("zero-size allocation".into()));
        }
        let ptr = hip::malloc(&self.hip, bytes)? as usize;
        self.inner.lock().unwrap().in_use += bytes;
        Ok(Allocation {
            pool_id: self.pool_id,
            offset: 0,
            bytes,
            ptr,
        })
    }

    fn free(&self, alloc: Allocation) -> Result<(), Error> {
        if alloc.pool_id != self.pool_id {
            return Err(Error::InvalidArgument(
                "allocation from another pool".into(),
            ));
        }
        hip::free(&self.hip, alloc.ptr as *mut _)?;
        let mut inner = self.inner.lock().unwrap();
        inner.in_use = inner.in_use.saturating_sub(alloc.bytes);
        Ok(())
    }

    fn pin(&self, alloc: Allocation) -> Result<(), Error> {
        if alloc.pool_id != self.pool_id {
            return Err(Error::InvalidArgument(
                "allocation from another pool".into(),
            ));
        }
        self.inner.lock().unwrap().pinned.insert(alloc.ptr);
        Ok(())
    }

    fn unpin(&self, alloc: Allocation) -> Result<(), Error> {
        if alloc.pool_id != self.pool_id {
            return Err(Error::InvalidArgument(
                "allocation from another pool".into(),
            ));
        }
        self.inner.lock().unwrap().pinned.remove(&alloc.ptr);
        Ok(())
    }

    fn bytes_in_use(&self) -> usize {
        self.inner.lock().unwrap().in_use
    }
}

/// HIP graph capture over a dedicated capture stream.
pub struct HipGraphCapture {
    hip: std::sync::Arc<Hip>,
    stream: HipHandle,
    state: Mutex<CaptureState>,
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
        Ok(Self {
            hip,
            stream: HipHandle(stream),
            state: Mutex::new(CaptureState::NoCapture),
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
        if !self.stream.0.is_null() {
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
            return Err(GraphError::Unsupported);
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
            return Err(GraphError::Unsupported);
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
            return Err(GraphError::Unsupported);
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
            return Err(GraphError::Unsupported);
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
    use crate::graph::GraphCapture;
    use mach_kernel_sys::hip::{HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE};

    const SAXPY_SRC: &str = r#"
extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"#;

    /// Returns a context if a HIP device exists, else `None` (skip).
    fn ctx() -> Option<HipContext> {
        match HipContext::init(0) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("skipping HIP test: {e}");
                None
            }
        }
    }

    #[test]
    fn hip_device_is_visible() {
        let Some(c) = ctx() else { return };
        assert_eq!(c.device, 0);
        assert!(!c.name.is_empty());
        eprintln!("HIP device 0: {}", c.name);
    }

    #[test]
    fn hiprtc_saxpy_runs_on_gpu() {
        let Some(c) = ctx() else { return };
        let h = &c.hip;
        let n: i32 = 1 << 20;
        let pool = HipMemoryPool::new(std::sync::Arc::clone(h));
        let x = pool.allocate(n as usize * 4, 256).unwrap();
        let y = pool.allocate(n as usize * 4, 256).unwrap();

        let mut hx = vec![1.0f32; n as usize];
        let mut hy = vec![1.0f32; n as usize];
        hip::memcpy(
            h,
            x.ptr as *mut _,
            hx.as_mut_ptr() as *const _,
            (n * 4) as usize,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        hip::memcpy(
            h,
            y.ptr as *mut _,
            hy.as_mut_ptr() as *const _,
            (n * 4) as usize,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();

        let module =
            hip::HipKernelModule::compile(&hip_arch(), SAXPY_SRC, "saxpy").expect("hiprtc compile");
        let a: f32 = 2.0;
        let mut params: Vec<*mut core::ffi::c_void> = vec![
            &a as *const f32 as *mut core::ffi::c_void,
            x.ptr as *mut core::ffi::c_void,
            y.ptr as *mut core::ffi::c_void,
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
        let Some(c) = ctx() else { return };
        let h = &c.hip;
        let n: i32 = 1 << 20;
        let pool = HipMemoryPool::new(std::sync::Arc::clone(h));
        let x = pool.allocate(n as usize * 4, 256).unwrap();
        let y = pool.allocate(n as usize * 4, 256).unwrap();
        pool.pin(x).unwrap();
        pool.pin(y).unwrap();

        let mut hx = vec![1.0f32; n as usize];
        let mut hy = vec![0.0f32; n as usize];
        hip::memcpy(
            h,
            x.ptr as *mut _,
            hx.as_mut_ptr() as *const _,
            (n * 4) as usize,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        hip::memcpy(
            h,
            y.ptr as *mut _,
            hy.as_mut_ptr() as *const _,
            (n * 4) as usize,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();

        let module =
            hip::HipKernelModule::compile(&hip_arch(), SAXPY_SRC, "saxpy").expect("hiprtc compile");
        let a: f32 = 2.0;

        let cap = HipGraphCapture::new(std::sync::Arc::clone(h)).unwrap();
        cap.prepare().unwrap();
        cap.begin().unwrap();
        {
            let mut params: Vec<*mut core::ffi::c_void> = vec![
                &a as *const f32 as *mut core::ffi::c_void,
                x.ptr as *mut core::ffi::c_void,
                y.ptr as *mut core::ffi::c_void,
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

        // SAFETY: x/y are pinned and alive; replays are serialized on this
        // thread and a device sync orders the final read.
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
        assert!(
            hy.iter().all(|&v| v == 3.0),
            "graph replay expected 3.0, got {:?}",
            &hy[..4]
        );
    }

    #[test]
    fn hip_graph_lifecycle_is_strict() {
        let Some(c) = ctx() else { return };
        let cap = HipGraphCapture::new(std::sync::Arc::clone(&c.hip)).unwrap();
        assert!(cap.begin().is_err(), "begin before prepare must fail");
        cap.prepare().unwrap();
        cap.begin().unwrap();
        let graph = cap.end().unwrap();
        drop(graph);
        cap.prepare().unwrap();
    }
}
