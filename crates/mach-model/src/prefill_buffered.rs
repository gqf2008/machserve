//! Full-layer double-buffered prefill (FreeToken-style).
//!
//! While layer `l` is being computed, layer `l + 1`'s weights are prefetched.
//! On the GPU path that is an async host→device copy of the next MoE layer's
//! expert weights on a **separate stream**, so the copy overlaps the GEMMs of
//! the current layer and never enters a graph capture of the compute stream.
//!
//! The scheduling core is [`run_double_buffered`]: a generic layer pipeline
//! whose ordering contract makes the output **bitwise identical** to a
//! sequential per-layer loop (double buffering only changes *when* weights
//! arrive, never the math). The CPU tests verify that contract against an
//! independent sequential reference; the GPU integration in `batched.rs`
//! drives the same schedule through [`PrefetchEngine`].

/// Runs `n` layers through the double-buffered schedule.
///
/// `prepare(i)` produces layer `i`'s prepared (prefetched) weights;
/// `compute(i, prepared, state)` runs layer `i` from them and updates `state`.
///
/// Schedule (two layers in flight):
///   1. `prepare(0)` runs first;
///   2. for each `i`, `prepare(i + 1)` is *issued* before `compute(i)` runs, so
///      an asynchronous `prepare` overlaps the previous layer's compute;
///   3. `compute(n - 1)` runs last.
///
/// Guarantees for *synchronous* `prepare` (the CPU verification model):
///   - `compute(i)` only runs after `prepare(i)` finished and `compute(i - 1)`
///     finished (sequential loop over `i`);
///   - `prepare(i + 1)` fills the ping-pong slot opposite to the one
///     `compute(i)` reads, so there is no aliasing.
///
/// With an *asynchronous* `prepare` the caller additionally owns slot-reuse
/// safety: `prepare(i + 1)` must not overwrite a slot an in-flight `compute`
/// still reads. The GPU path does this with device-side stream events (see
/// [`PrefetchEngine`]); the schedule itself only fixes the issue order.
///
/// Outputs and `state` are identical to a sequential `prepare` + `compute`
/// loop — double buffering is a scheduling no-op for the math.
pub fn run_double_buffered<P, S, O, E>(
    n: usize,
    state: &mut S,
    mut prepare: impl FnMut(usize) -> Result<P, E>,
    mut compute: impl FnMut(usize, &P, &mut S) -> Result<O, E>,
) -> Result<Vec<O>, E> {
    let mut slots: [Option<P>; 2] = [None, None];
    let mut outputs = Vec::with_capacity(n);
    if n == 0 {
        return Ok(outputs);
    }
    slots[0] = Some(prepare(0)?);
    for i in 0..n {
        if i + 1 < n {
            slots[(i + 1) % 2] = Some(prepare(i + 1)?);
        }
        let prepared = slots[i % 2].take().expect("layer prepared before compute");
        outputs.push(compute(i, &prepared, state)?);
    }
    Ok(outputs)
}

#[cfg(feature = "hip")]
mod hip_impl {
    use crate::Error;
    use crate::config::{Config, ModelDType};
    use crate::weights::Weights;
    use mach_kernel_sys::hip::{self, Hip, HipEvent, HipStream};
    use std::sync::Arc;

    /// Pinned host staging for one MoE layer's expert weights (all experts).
    ///
    /// The staging is page-locked so the prefetch `hipMemcpyAsync` runs as a
    /// true asynchronous DMA that overlaps the compute stream's GEMMs.
    struct StagedExpert {
        /// `[num_experts, expert_size, d_model]` (gate / up).
        wg: *mut f32,
        wu: *mut f32,
        /// `[num_experts, d_model, expert_size]` (down).
        wd: *mut f32,
    }

    /// GPU prefetch orchestration for full-layer double-buffered prefill.
    ///
    /// Follows the [`run_double_buffered`] schedule: while MoE layer `k` is
    /// computed on the compute stream, MoE layer `k + 1`'s expert weights are
    /// copied host→device on a dedicated prefetch stream (so the copy never
    /// enters a graph capture of the compute stream). Two ping-pong device
    /// buffers hold one layer's full expert pool each, so the device holds only
    /// ~2 layers of experts instead of the whole checkpoint.
    ///
    /// Correctness is enforced with device-side events:
    /// - [`Self::weights_ready`] makes the compute stream wait for the prefetch
    ///   event of the current layer before the grouped expert GEMMs read it;
    /// - [`Self::layer_begin`] makes the prefetch stream wait for the previous
    ///   MoE layer's compute event before overwriting the ping-pong slot that
    ///   compute last read;
    /// - [`Self::begin`] makes the prefetch stream wait for the PREVIOUS
    ///   forward's last MoE layer compute event before the first prefetch
    ///   overwrites a slot that compute may still read (cross-step reuse).
    pub struct PrefetchEngine {
        hip: Arc<Hip>,
        stream: HipStream,
        cfg: Config,
        /// Layer indices that are routed MoE (ascending).
        moe_layers: Vec<usize>,
        /// Pinned host staging, indexed by MoE-layer rank `k`.
        staged: Vec<StagedExpert>,
        /// Ping-pong device slots, indexed by `k % 2`.
        wg_slots: [*mut f32; 2],
        wu_slots: [*mut f32; 2],
        wd_slots: [*mut f32; 2],
        /// `prefetch_ev[k]`: weights of MoE layer `k` resident on device.
        prefetch_ev: Vec<HipEvent>,
        /// `compute_ev[k]`: compute of MoE layer `k` finished.
        compute_ev: Vec<HipEvent>,
        pins: Vec<*mut core::ffi::c_void>,
        allocs: Vec<*mut core::ffi::c_void>,
    }

    // SAFETY: the engine is confined to one model on one thread; raw handles
    // are only touched there, and the loaded HIP runtime is Send + Sync.
    unsafe impl Send for PrefetchEngine {}
    unsafe impl Sync for PrefetchEngine {}

    impl PrefetchEngine {
        /// Builds the prefetch engine: pins every MoE layer's expert weights in
        /// host RAM, allocates two ping-pong device expert pools, and creates
        /// the prefetch stream plus per-layer events. F32 only for now (fp16
        /// prefetch would need a device-side f32→f16 cast on the prefetch
        /// stream, a follow-up).
        pub fn new(hip: Arc<Hip>, cfg: Config, w: &Weights) -> Result<Self, Error> {
            if cfg.dtype == ModelDType::F16 {
                return Err(Error::InvalidArgument(
                    "prefill buffering requires F32 (fp16 prefetch not implemented)".into(),
                ));
            }
            let moe_layers: Vec<usize> = w
                .layers
                .iter()
                .enumerate()
                .filter(|(_, l)| !l.moe_router.is_empty())
                .map(|(i, _)| i)
                .collect();
            if moe_layers.is_empty() {
                return Err(Error::InvalidArgument(
                    "prefill buffering requires an MoE checkpoint".into(),
                ));
            }
            let ne = cfg.num_experts;
            let einter = cfg.expert_size();
            let d = cfg.d_model;

            let mut pins = Vec::new();
            let mut staged = Vec::with_capacity(moe_layers.len());
            for &li in &moe_layers {
                let lw = &w.layers[li];
                staged.push(StagedExpert {
                    wg: pin_copy(&hip, &lw.moe_wg, &mut pins)?,
                    wu: pin_copy(&hip, &lw.moe_wu, &mut pins)?,
                    wd: pin_copy(&hip, &lw.moe_wd, &mut pins)?,
                });
            }

            let mut allocs = Vec::new();
            let wg_bytes = ne * einter * d * 4;
            let wd_bytes = ne * d * einter * 4;
            let mut dalloc = |bytes: usize| -> Result<*mut f32, Error> {
                let p = hip::malloc(&hip, bytes)?;
                allocs.push(p);
                Ok(p as *mut f32)
            };
            let wg_slots = [dalloc(wg_bytes)?, dalloc(wg_bytes)?];
            let wu_slots = [dalloc(wg_bytes)?, dalloc(wg_bytes)?];
            let wd_slots = [dalloc(wd_bytes)?, dalloc(wd_bytes)?];

            let mut stream = std::ptr::null_mut();
            unsafe { hip::check(&hip, (hip.api.hip_stream_create)(&mut stream))? };
            let n_moe = moe_layers.len();
            let prefetch_ev = create_events(&hip, n_moe)?;
            let compute_ev = create_events(&hip, n_moe)?;
            // NOTE: compute events are NOT pre-recorded here. The FIRST
            // `begin()` waits on `compute_ev[last]`, which no forward has
            // recorded yet — per HIP, waiting on a never-recorded event acts
            // as already-completed, so the first call returns immediately
            // without blocking.

            Ok(Self {
                hip,
                stream,
                cfg,
                moe_layers,
                staged,
                wg_slots,
                wu_slots,
                wd_slots,
                prefetch_ev,
                compute_ev,
                pins,
                allocs,
            })
        }

        /// MoE-layer rank `k` for a layer index, or `None` for a dense layer.
        fn moe_k(&self, li: usize) -> Option<usize> {
            self.moe_layers.binary_search(&li).ok()
        }

        /// Issues the prefetch of the first MoE layer. Called once per
        /// forward, before the layer loop.
        ///
        /// CROSS-STEP SLOT SAFETY: `prefetch(0)` writes ping-pong slot 0,
        /// which the previous forward's LAST MoE layer read whenever the MoE
        /// layer count is odd (slot `(n-1) % 2 == 0`). The per-layer waits
        /// inside a forward only chain to `compute_ev[n-2]` (`prefetch(k+1)`
        /// waits on `compute_ev[k-1]`), so without this wait the new copy can
        /// overtake the previous forward's last read — the window widens with
        /// the fast grouped-GEMV decode path. Waiting on the last layer's
        /// event covers every count (a superset of the even-count need). On
        /// the first call `compute_ev[last]` has never been recorded, which
        /// per HIP acts as already-completed, so it returns immediately (no
        /// pre-recording is needed or done).
        pub fn begin(&self) -> Result<(), Error> {
            let last = self.moe_layers.len() - 1;
            let ev = self.compute_ev[last];
            unsafe {
                hip::check(
                    &self.hip,
                    (self.hip.api.hip_stream_wait_event)(self.stream, ev, 0),
                )?;
            }
            self.prefetch(0)
        }

        /// Called at the start of layer `li`'s iteration: issues the prefetch
        /// of the next MoE layer so the H2D overlaps this layer's compute.
        pub fn layer_begin(&self, li: usize) -> Result<(), Error> {
            let Some(k) = self.moe_k(li) else {
                return Ok(());
            };
            // prefetch(k+1) writes the ping-pong slot that compute(k-1) last
            // read; wait for that compute to finish before overwriting.
            if k >= 1 {
                let ev = self.compute_ev[k - 1];
                unsafe {
                    hip::check(
                        &self.hip,
                        (self.hip.api.hip_stream_wait_event)(self.stream, ev, 0),
                    )?;
                }
            }
            self.prefetch(k + 1)
        }

        /// Called right before MoE layer `li`'s grouped GEMMs: makes the
        /// compute stream wait until that layer's weights are resident.
        #[allow(clippy::not_unsafe_ptr_arg_deref)] // safe FFI wrapper (see hip.rs)
        pub fn weights_ready(&self, li: usize, compute_stream: HipStream) -> Result<(), Error> {
            let Some(k) = self.moe_k(li) else {
                return Ok(());
            };
            unsafe {
                hip::check(
                    &self.hip,
                    (self.hip.api.hip_stream_wait_event)(compute_stream, self.prefetch_ev[k], 0),
                )?;
            }
            Ok(())
        }

        /// Called at the end of layer `li`'s iteration: records that the layer's
        /// compute finished (the prefetch of `k + 2` waits on it before reusing
        /// the ping-pong slot).
        #[allow(clippy::not_unsafe_ptr_arg_deref)] // safe FFI wrapper (see hip.rs)
        pub fn layer_end(&self, li: usize, compute_stream: HipStream) -> Result<(), Error> {
            let Some(k) = self.moe_k(li) else {
                return Ok(());
            };
            unsafe {
                hip::check(
                    &self.hip,
                    (self.hip.api.hip_event_record)(self.compute_ev[k], compute_stream),
                )?;
            }
            Ok(())
        }

        /// Device pointers of MoE layer `li`'s expert weights (ping-pong slot
        /// `k % 2`). `None` for dense layers.
        pub fn weights(&self, li: usize) -> Option<(*mut f32, *mut f32, *mut f32)> {
            let k = self.moe_k(li)?;
            let slot = k % 2;
            Some((
                self.wg_slots[slot],
                self.wu_slots[slot],
                self.wd_slots[slot],
            ))
        }

        /// Issues the H2D prefetch of MoE layer rank `k` into slot `k % 2` and
        /// records its prefetch event. No-op when `k` is out of range.
        fn prefetch(&self, k: usize) -> Result<(), Error> {
            if k >= self.moe_layers.len() {
                return Ok(());
            }
            let slot = k % 2;
            let s = &self.staged[k];
            let wg_bytes = self.cfg.num_experts * self.cfg.expert_size() * self.cfg.d_model * 4;
            let wd_bytes = self.cfg.num_experts * self.cfg.d_model * self.cfg.expert_size() * 4;
            unsafe {
                hip::memcpy_async(
                    &self.hip,
                    self.wg_slots[slot] as *mut core::ffi::c_void,
                    s.wg as *const core::ffi::c_void,
                    wg_bytes,
                    hip::HIP_MEMCPY_HOST_TO_DEVICE,
                    self.stream,
                )?;
                hip::memcpy_async(
                    &self.hip,
                    self.wu_slots[slot] as *mut core::ffi::c_void,
                    s.wu as *const core::ffi::c_void,
                    wg_bytes,
                    hip::HIP_MEMCPY_HOST_TO_DEVICE,
                    self.stream,
                )?;
                hip::memcpy_async(
                    &self.hip,
                    self.wd_slots[slot] as *mut core::ffi::c_void,
                    s.wd as *const core::ffi::c_void,
                    wd_bytes,
                    hip::HIP_MEMCPY_HOST_TO_DEVICE,
                    self.stream,
                )?;
                hip::check(
                    &self.hip,
                    (self.hip.api.hip_event_record)(self.prefetch_ev[k], self.stream),
                )?;
            }
            Ok(())
        }
    }

    /// Creates `n` HIP events.
    fn create_events(hip: &Hip, n: usize) -> Result<Vec<HipEvent>, Error> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let mut e = std::ptr::null_mut();
            unsafe { hip::check(hip, (hip.api.hip_event_create)(&mut e))? };
            v.push(e);
        }
        Ok(v)
    }

    /// Copies `src` into fresh page-locked host memory and returns the pointer.
    fn pin_copy(
        hip: &Arc<Hip>,
        src: &[f32],
        pins: &mut Vec<*mut core::ffi::c_void>,
    ) -> Result<*mut f32, Error> {
        let p = hip::host_malloc(hip, src.len() * 4)?;
        pins.push(p);
        // SAFETY: `p` is pinned host memory of `src.len()` f32s; it is fully
        // written here before any async copy reads it.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), p as *mut f32, src.len()) };
        Ok(p as *mut f32)
    }

    impl Drop for PrefetchEngine {
        fn drop(&mut self) {
            // Drain any in-flight prefetch copies before freeing their sources.
            unsafe {
                let _ = (self.hip.api.hip_stream_synchronize)(self.stream);
            }
            for &e in &self.prefetch_ev {
                unsafe {
                    let _ = (self.hip.api.hip_event_destroy)(e);
                }
            }
            for &e in &self.compute_ev {
                unsafe {
                    let _ = (self.hip.api.hip_event_destroy)(e);
                }
            }
            if !self.stream.is_null() {
                unsafe {
                    let _ = (self.hip.api.hip_stream_destroy)(self.stream);
                }
            }
            for &p in &self.allocs {
                let _ = hip::free(&self.hip, p);
            }
            for &p in &self.pins {
                let _ = hip::host_free(&self.hip, p);
            }
        }
    }
}

#[cfg(feature = "hip")]
pub use hip_impl::PrefetchEngine;

#[cfg(test)]
mod tests {
    use super::run_double_buffered;

    #[test]
    fn empty_pipeline_runs_no_layers() {
        let mut state = 0usize;
        let out = run_double_buffered(
            0,
            &mut state,
            |i| -> Result<usize, ()> { panic!("prepare({i}) must not run") },
            |i, _p, _s| -> Result<usize, ()> { panic!("compute({i}) must not run") },
        )
        .unwrap();
        assert!(out.is_empty());
        assert_eq!(state, 0);
    }

    #[test]
    fn single_layer_prepares_then_computes() {
        let mut state = 1usize;
        let mut calls = Vec::new();
        let out = run_double_buffered(
            1,
            &mut state,
            |i| {
                calls.push(i);
                Ok::<_, ()>(i + 10)
            },
            |_i, p, s| {
                *s += *p;
                Ok::<_, ()>(*s)
            },
        )
        .unwrap();
        assert_eq!(calls, vec![0]);
        assert_eq!(out, vec![11]);
    }

    #[test]
    fn compute_sees_its_own_prepared_layer() {
        // Distinct prepared values per layer: compute(i) must see prepare(i),
        // not a neighbour's (a ping-pong mix-up would surface immediately).
        let n = 6usize;
        let mut state = 0usize;
        let out = run_double_buffered(n, &mut state, Ok::<_, ()>, |i, p, s| {
            assert_eq!(*p, i, "compute({i}) must see prepare({i})");
            *s += i;
            Ok::<_, ()>(*s)
        })
        .unwrap();
        let expected: Vec<usize> = (0..n)
            .scan(0, |acc, i| {
                *acc += i;
                Some(*acc)
            })
            .collect();
        assert_eq!(out, expected);
    }
}
