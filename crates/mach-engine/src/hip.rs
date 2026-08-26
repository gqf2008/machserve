//! ROCm/HIP backend: memory pool, streams and graph capture on AMD GPUs.
//!
//! Implemented on top of `mach_kernel_sys::hip` (the single FFI boundary).
//! The capture lifecycle reuses the strict `NoCapture → Prepare → Capture`
//! state machine from [`crate::graph`]; this module only supplies the driver
//! work each transition brackets (`hipStreamBeginCapture` / `EndCapture`,
//! `hipGraphInstantiate` / `hipGraphLaunch`).

use crate::Error;
use crate::graph::{CaptureState, GraphCapture, GraphError, GraphHandle};
use crate::memory::{Allocation, MemoryPool, Region, Tag, TaggedPool};
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
    /// Live tagged regions (each its own hipMalloc block).
    regions: Vec<RegionEntry>,
}

/// A live tagged region: an independent device block so resizing never
/// disturbs sibling regions (malloc new -> D2D copy -> free old).
#[derive(Debug)]
struct RegionEntry {
    tag: Tag,
    ptr: usize,
    bytes: usize,
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

impl TaggedPool for HipMemoryPool {
    fn allocate_region(&self, tag: Tag, bytes: usize, _align: usize) -> Result<Region, Error> {
        if bytes == 0 {
            return Err(Error::InvalidArgument("zero-size region".into()));
        }
        // hipMalloc returns 256-byte aligned blocks; the alignment argument is
        // accepted for interface parity with the CPU pool.
        let ptr = hip::malloc(&self.hip, bytes)? as usize;
        let mut inner = self.inner.lock().unwrap();
        inner.in_use += bytes;
        inner.regions.push(RegionEntry {
            tag: tag.clone(),
            ptr,
            bytes,
        });
        Ok(Region {
            pool_id: self.pool_id,
            tag,
            offset: 0,
            bytes,
            ptr,
        })
    }

    fn resize_region(&self, region: Region, new_bytes: usize) -> Result<Region, Error> {
        if region.pool_id != self.pool_id {
            return Err(Error::InvalidArgument("region from another pool".into()));
        }
        if new_bytes == 0 {
            return Err(Error::InvalidArgument("zero-size region".into()));
        }
        let mut inner = self.inner.lock().unwrap();
        let idx = inner
            .regions
            .iter()
            .position(|r| r.ptr == region.ptr)
            .ok_or_else(|| Error::Memory("region is not live".into()))?;
        let old = inner.regions[idx].bytes;
        if new_bytes == old {
            return Ok(region);
        }
        let tag = inner.regions[idx].tag.clone();
        // Fresh block + D2D copy of the preserved prefix + free the old block:
        // shrinking really releases VRAM (no caching-allocator slack).
        let new_ptr = hip::malloc(&self.hip, new_bytes)? as usize;
        let copy = old.min(new_bytes);
        if let Err(e) = hip::memcpy(
            &self.hip,
            new_ptr as *mut core::ffi::c_void,
            region.ptr as *const core::ffi::c_void,
            copy,
            hip::HIP_MEMCPY_DEVICE_TO_DEVICE,
        ) {
            // Do not leak the fresh block on the failure path; the old region
            // is untouched and stays live.
            let _ = hip::free(&self.hip, new_ptr as *mut core::ffi::c_void);
            return Err(e.into());
        }
        hip::free(&self.hip, region.ptr as *mut core::ffi::c_void)?;
        inner.regions[idx].ptr = new_ptr;
        inner.regions[idx].bytes = new_bytes;
        inner.in_use = inner.in_use - old + new_bytes;
        Ok(Region {
            pool_id: self.pool_id,
            tag,
            offset: 0,
            bytes: new_bytes,
            ptr: new_ptr,
        })
    }

    fn free_region(&self, region: Region) -> Result<(), Error> {
        if region.pool_id != self.pool_id {
            return Err(Error::InvalidArgument("region from another pool".into()));
        }
        let mut inner = self.inner.lock().unwrap();
        let idx = inner
            .regions
            .iter()
            .position(|r| r.ptr == region.ptr)
            .ok_or_else(|| Error::Memory("region is not live".into()))?;
        let entry = inner.regions.remove(idx);
        hip::free(&self.hip, entry.ptr as *mut core::ffi::c_void)?;
        inner.in_use = inner.in_use.saturating_sub(entry.bytes);
        Ok(())
    }

    fn shrink_to(&self, budget: usize) -> Result<usize, Error> {
        let inner = self.inner.lock().unwrap();
        let committed = inner.in_use;
        if committed <= budget {
            return Ok(0);
        }
        Err(Error::Memory(format!(
            "pool uses {committed} live bytes > budget {budget} (shrink/free regions first)"
        )))
    }

    fn committed_bytes(&self) -> usize {
        self.inner.lock().unwrap().in_use
    }

    fn region_by_tag(&self, tag: &Tag) -> Option<Region> {
        let inner = self.inner.lock().unwrap();
        inner
            .regions
            .iter()
            .find(|r| &r.tag == tag)
            .map(|r| Region {
                pool_id: self.pool_id,
                tag: r.tag.clone(),
                offset: 0,
                bytes: r.bytes,
                ptr: r.ptr,
            })
    }
}

/// HIP graph capture over a dedicated capture stream.
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
        let xp = x.ptr as *mut core::ffi::c_void;
        let yp = y.ptr as *mut core::ffi::c_void;
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
        let xp = x.ptr as *mut core::ffi::c_void;
        let yp = y.ptr as *mut core::ffi::c_void;

        let cap = HipGraphCapture::new(std::sync::Arc::clone(h)).unwrap();
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
        // y starts at 0; two replays: 0 -> 2 -> 4.
        assert!(
            hy.iter().all(|&v| v == 4.0),
            "graph replay expected 4.0 after two replays, got {:?}",
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

    fn write_region(h: &hip::Hip, r: Region, data: &[u8]) {
        assert!(data.len() <= r.bytes);
        hip::memcpy(
            h,
            r.ptr as *mut core::ffi::c_void,
            data.as_ptr() as *const core::ffi::c_void,
            data.len(),
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
    }

    fn read_region(h: &hip::Hip, r: Region) -> Vec<u8> {
        let mut out = vec![0u8; r.bytes];
        hip::memcpy(
            h,
            out.as_mut_ptr() as *mut core::ffi::c_void,
            r.ptr as *const core::ffi::c_void,
            out.len(),
            HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        out
    }

    fn pattern(seed: u8, len: usize) -> Vec<u8> {
        (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
    }

    /// HIP elastic memory: simulated VRAM crash (显存骤降). A background
    /// renderer/game eats VRAM, the expert cache and KV regions shrink, and the
    /// pool must never OOM — the service degrades smoothly and keeps serving.
    #[ignore = "GPU (opt-in: -- --ignored --test-threads=1)"]
    #[test]
    fn hip_region_shrink_to_under_pressure_no_oom() {
        let Some(c) = ctx() else { return };
        let h = &c.hip;
        let pool = HipMemoryPool::new(std::sync::Arc::clone(h));
        let exp = pool
            .allocate_region(Tag::expert_cache(), 1 << 20, 64)
            .unwrap();
        let kv = pool.allocate_region(Tag::kv(), 1 << 20, 64).unwrap();
        let pe = pattern(7, 1 << 20);
        let pk = pattern(9, 1 << 20);
        write_region(h, exp.clone(), &pe);
        write_region(h, kv.clone(), &pk);

        // External pressure: a hard budget below the current 2 MiB footprint.
        let budget = 512 << 10;
        let err = pool.shrink_to(budget).unwrap_err();
        assert!(
            err.to_string().contains("live bytes"),
            "shrink above live bytes must fail with a degradable error: {err}"
        );

        // Smooth degradation: evict experts to host + trim KV, then re-shrink.
        let exp_s = pool.resize_region(exp.clone(), 256 << 10).unwrap();
        let kv_s = pool.resize_region(kv.clone(), 256 << 10).unwrap();
        let d = read_region(h, exp_s.clone());
        assert_eq!(
            &d[..256 << 10],
            &pe[..256 << 10],
            "evicted expert region keeps its resident prefix"
        );
        let d = read_region(h, kv_s.clone());
        assert_eq!(
            &d[..256 << 10],
            &pk[..256 << 10],
            "trimmed KV region keeps its resident prefix"
        );

        // HIP regions are independent blocks: the resize already released VRAM
        // (fresh block + free old), so shrink_to is a budget check under budget.
        let freed = pool.shrink_to(budget).unwrap();
        assert!(
            pool.committed_bytes() <= budget,
            "committed must fit the budget"
        );
        assert_eq!(
            freed, 0,
            "HIP shrink_to under budget releases nothing further"
        );
        assert_eq!(pool.bytes_in_use(), 512 << 10);

        // Regions survive via tag re-fetch; data prefix intact.
        let exp_r = pool
            .region_by_tag(&Tag::expert_cache())
            .expect("expert live");
        let kv_r = pool.region_by_tag(&Tag::kv()).expect("kv live");
        let d = read_region(h, exp_r);
        assert_eq!(&d[..256 << 10], &pe[..256 << 10]);
        let d = read_region(h, kv_r);
        assert_eq!(&d[..256 << 10], &pk[..256 << 10]);

        // Service continues after the squeeze.
        let scratch = pool.allocate(4096, 8).unwrap();
        let extra = pool
            .allocate_region(Tag::new("scratch"), 64 << 10, 8)
            .unwrap();
        assert_eq!(pool.bytes_in_use(), (512 << 10) + 4096 + (64 << 10));
        let _ = (scratch, extra);
    }

    /// HIP elastic rebalancing: expert cache shrinks, KV grows; the pool hands
    /// VRAM between the two regions without a restart.
    #[ignore = "GPU (opt-in: -- --ignored --test-threads=1)"]
    #[test]
    fn hip_region_rebalance_hands_memory_to_kv() {
        let Some(c) = ctx() else { return };
        let h = &c.hip;
        let pool = HipMemoryPool::new(std::sync::Arc::clone(h));
        let exp = pool
            .allocate_region(Tag::expert_cache(), 512 << 10, 64)
            .unwrap();
        let kv = pool.allocate_region(Tag::kv(), 256 << 10, 64).unwrap();
        let pe = pattern(3, 512 << 10);
        let pk = pattern(4, 256 << 10);
        write_region(h, exp.clone(), &pe);
        write_region(h, kv.clone(), &pk);

        let budget = 768 << 10;
        let exp2 = pool.resize_region(exp.clone(), 384 << 10).unwrap();
        let kv2 = pool.resize_region(kv.clone(), 384 << 10).unwrap();
        let d = read_region(h, exp2.clone());
        assert_eq!(&d[..384 << 10], &pe[..384 << 10]);
        let d = read_region(h, kv2.clone());
        assert_eq!(&d[..256 << 10], &pk[..], "kv keeps its original prefix");

        assert_eq!(pool.bytes_in_use(), budget);
        pool.shrink_to(budget).unwrap();
        assert!(pool.committed_bytes() <= budget);
        let _ = (exp2, kv2);
    }
}
