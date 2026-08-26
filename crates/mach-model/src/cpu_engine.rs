//! CPU continuous-batching engine with cross-request prefix reuse.
//!
//! Mirrors the serving semantics of [`crate::continuous`] (interleaved
//! prefill + decode, request lifecycle, slot reuse) on the CPU reference path,
//! layered on the TokenSpeed-alignment stack: scheduler FSM + reuse planner +
//! prefix KV cache. It is the hardware-agnostic reference for the GPU wiring
//! (`batched.rs`): the scheduling logic here is exactly what the GPU engine
//! should implement, with the reference model swapped for the HIP kernels.
//!
//! Requests queue on `add`, are admitted as slots free up, prefilled one per
//! step via the prefix cache (delta-only when a shared prefix is cached), then
//! decoded one token per step until their `max_new` budget is exhausted.
//!
//! **Serving notes for the GPU wiring**: `capacity` bounds only the number of
//! concurrent slots; the admission queue is bounded by `max_pending` (backpressure)
//! and completed records by `max_finished` (oldest dropped) when built with
//! [`Self::new_with_limits`] — the HTTP layer should surface `add`'s error as
//! 429-style backpressure. `max_new = 0` is accepted: the prompt is prefilled
//! (and cached) and the request finishes with no decoded tokens.

use std::collections::VecDeque;

use crate::prefix_kv::PrefixKvCache;
use crate::ref_model::RefModel;
use crate::scheduler_fsm::{
    Bootstrapping, FsmEvent, RequestState, ScheduleDecode, SchedulePrefillFirstChunk,
};
use crate::{Config, Error, Weights};

/// A request waiting for a free slot.
struct PendingRequest {
    id: u64,
    prompt: Vec<u32>,
    max_new: usize,
}

/// An admitted, in-flight request.
struct Slot {
    id: u64,
    state: RequestState,
    prompt: Vec<u32>,
    max_new: usize,
    generated: Vec<u32>,
    model: RefModel,
    /// Logits after the prompt (first decoded token uses them).
    next_logits: Vec<f32>,
    reused_tokens: usize,
    computed_tokens: usize,
}

/// A completed request's output and prefix-reuse accounting.
#[derive(Debug, Clone)]
pub struct FinishedRequest {
    pub id: u64,
    pub generated: Vec<u32>,
    pub reused_tokens: usize,
    pub computed_tokens: usize,
    pub total_prompt_tokens: usize,
}

/// Aggregate engine accounting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuEngineStats {
    pub requests: usize,
    pub steps: u64,
    pub prompt_tokens_total: usize,
    pub prompt_tokens_reused: usize,
    pub prompt_tokens_computed: usize,
    pub decoded_tokens: usize,
}

impl CpuEngineStats {
    /// Fraction of prompt tokens served from the cache (0..1).
    #[must_use]
    pub fn savings(&self) -> f64 {
        if self.prompt_tokens_total == 0 {
            0.0
        } else {
            self.prompt_tokens_reused as f64 / self.prompt_tokens_total as f64
        }
    }
}

/// CPU continuous-batching engine over the reference model.
pub struct CpuEngine {
    cache: PrefixKvCache,
    cfg: Config,
    w: Weights,
    capacity: usize,
    slots: Vec<Option<Slot>>,
    pending: VecDeque<PendingRequest>,
    /// Admission-queue bound (backpressure); `add` rejects when full.
    max_pending: usize,
    next_id: u64,
    finished: Vec<FinishedRequest>,
    /// Completed-record bound; oldest records are dropped when exceeded.
    max_finished: usize,
    stats: CpuEngineStats,
}

impl CpuEngine {
    /// Builds an engine with `capacity` concurrent slots and a KV pool of
    /// `pool_blocks` blocks (`tokens_per_page` tokens per page).
    #[must_use]
    pub fn new(
        cfg: Config,
        w: Weights,
        capacity: usize,
        pool_blocks: i32,
        tokens_per_page: usize,
    ) -> Self {
        Self::new_with_limits(
            cfg,
            w,
            capacity,
            pool_blocks,
            tokens_per_page,
            usize::MAX,
            usize::MAX,
        )
    }

    /// Builds an engine with bounded admission queue and completed records:
    /// `max_pending` caps queued-but-not-admitted requests (backpressure for an
    /// HTTP layer), `max_finished` caps retained records (oldest dropped).
    #[must_use]
    pub fn new_with_limits(
        cfg: Config,
        w: Weights,
        capacity: usize,
        pool_blocks: i32,
        tokens_per_page: usize,
        max_pending: usize,
        max_finished: usize,
    ) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        Self {
            cache: PrefixKvCache::new(pool_blocks, tokens_per_page),
            cfg,
            w,
            capacity,
            slots: (0..capacity).map(|_| None).collect(),
            pending: VecDeque::new(),
            max_pending,
            next_id: 1,
            finished: Vec::new(),
            max_finished,
            stats: CpuEngineStats::default(),
        }
    }

    /// Maximum number of concurrent in-flight requests.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Queues a request; returns its id. Empty prompts are rejected.
    pub fn add(&mut self, prompt: &[u32], max_new: usize) -> Result<u64, Error> {
        if prompt.is_empty() {
            return Err(Error::InvalidArgument(
                "cpu engine rejects an empty prompt".into(),
            ));
        }
        if self.pending.len() >= self.max_pending {
            return Err(Error::InvalidArgument(
                "cpu engine admission queue is full (max_pending)".into(),
            ));
        }
        let id = self.next_id;
        self.next_id += 1;
        self.pending.push_back(PendingRequest {
            id,
            prompt: prompt.to_vec(),
            max_new,
        });
        Ok(id)
    }

    /// Completed requests, in finish order.
    #[must_use]
    pub fn finished(&self) -> &[FinishedRequest] {
        &self.finished
    }

    /// Takes and clears the completed records (bounds the engine's memory over
    /// a long run; the GPU engine should do the same periodically).
    pub fn drain_finished(&mut self) -> Vec<FinishedRequest> {
        std::mem::take(&mut self.finished)
    }

    /// Aggregate engine stats.
    #[must_use]
    pub fn stats(&self) -> &CpuEngineStats {
        &self.stats
    }

    /// True when no requests are queued or in flight.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.pending.is_empty() && self.slots.iter().all(Option::is_none)
    }

    /// Runs one engine step: admit queued requests into free slots, prefill
    /// one not-yet-prefilled slot, decode one token for every active slot.
    /// Returns false when the engine is idle (nothing more to do).
    pub fn step(&mut self) -> Result<bool, Error> {
        self.admit_pending();
        if self.is_idle() {
            return Ok(false);
        }
        self.stats.steps += 1;

        // Prefill the oldest not-yet-prefilled slot (delta-only via the cache).
        let cache = &mut self.cache;
        for slot in self.slots.iter_mut().flatten() {
            if !matches!(
                slot.state,
                RequestState::PrefillDone(_) | RequestState::Decoding(_)
            ) {
                prefill_slot(cache, slot)?;
                break;
            }
        }

        // Decode one token for every active slot; collect finishes.
        let mut finished_now: Vec<usize> = Vec::new();
        for (idx, slot) in self.slots.iter_mut().enumerate() {
            let Some(slot) = slot else { continue };
            if !matches!(slot.state, RequestState::Decoding(_)) {
                continue;
            }
            let tok = argmax(&slot.next_logits);
            slot.generated.push(tok);
            slot.next_logits = slot.model.decode_step(tok);
            slot.state = FsmEvent::ExtendResult(vec![tok as i32]).apply(slot.state.clone());
            if slot.generated.len() >= slot.max_new {
                slot.state = FsmEvent::Succeeded.apply(slot.state.clone());
                debug_assert!(matches!(slot.state, RequestState::Finished));
                finished_now.push(idx);
            }
        }
        for idx in finished_now {
            self.finish_slot(idx);
        }
        Ok(true)
    }

    /// Runs steps until the engine is idle.
    pub fn step_until_done(&mut self) -> Result<(), Error> {
        while self.step()? {}
        Ok(())
    }

    /// Admit queued requests into any free slot.
    fn admit_pending(&mut self) {
        while let Some(free) = self.slots.iter().position(Option::is_none) {
            let Some(req) = self.pending.pop_front() else {
                break;
            };
            let model = RefModel::new(self.cfg, self.w.clone());
            let mut state =
                FsmEvent::Bootstrapped.apply(RequestState::Bootstrapping(Bootstrapping));
            if let RequestState::Submitted(s) = &mut state {
                s.tokens = req.prompt.iter().map(|&t| t as i32).collect();
                s.max_new_tokens = req.max_new as i32;
            }
            self.slots[free] = Some(Slot {
                id: req.id,
                state,
                prompt: req.prompt,
                max_new: req.max_new,
                generated: Vec::new(),
                model,
                next_logits: Vec::new(),
                reused_tokens: 0,
                computed_tokens: 0,
            });
        }
    }

    /// Moves a finished slot into the completed record and frees the slot.
    fn finish_slot(&mut self, idx: usize) {
        let slot = self.slots[idx].take().expect("slot occupied");
        self.stats.requests += 1;
        self.stats.prompt_tokens_total += slot.prompt.len();
        self.stats.prompt_tokens_reused += slot.reused_tokens;
        self.stats.prompt_tokens_computed += slot.computed_tokens;
        self.stats.decoded_tokens += slot.generated.len();
        self.finished.push(FinishedRequest {
            id: slot.id,
            generated: slot.generated,
            reused_tokens: slot.reused_tokens,
            computed_tokens: slot.computed_tokens,
            total_prompt_tokens: slot.prompt.len(),
        });
        while self.finished.len() > self.max_finished {
            self.finished.remove(0);
        }
    }
}

/// Prefills a slot's full prompt through the prefix cache (delta-only).
fn prefill_slot(cache: &mut PrefixKvCache, slot: &mut Slot) -> Result<(), Error> {
    let (logits, stats) = cache.serve(&mut slot.model, &slot.prompt)?;
    slot.next_logits = logits;
    slot.reused_tokens = stats.reused_tokens;
    slot.computed_tokens = stats.computed_tokens;
    slot.state = FsmEvent::SchedulePrefillFirstChunk(SchedulePrefillFirstChunk {
        chunk_size: slot.prompt.len() as i32,
        reserve_tokens: 0,
    })
    .apply(slot.state.clone());
    slot.state =
        FsmEvent::ScheduleDecode(ScheduleDecode { reserve_tokens: 0 }).apply(slot.state.clone());
    Ok(())
}

/// Index of the maximum element (the greedy next token).
fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greedy(cfg: Config, w: &Weights, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut m = RefModel::new(cfg, w.clone());
        let mut logits = m.forward(prompt);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let t = argmax(&logits);
            out.push(t);
            logits = m.decode_step(t);
        }
        out
    }

    #[test]
    fn shared_prefix_requests_interleave_and_match_full_recompute() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 31).expect("weights");
        let system = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let tails = [100u32, 101, 102, 103, 104];
        let mut eng = CpuEngine::new(cfg, w.clone(), 5, 32, 4);
        let ids: Vec<u64> = tails
            .iter()
            .map(|&tail| {
                let mut p = system.to_vec();
                p.push(tail);
                eng.add(&p, 4).expect("add")
            })
            .collect();
        eng.step_until_done().expect("run");
        assert!(eng.is_idle());

        let finished = eng.finished();
        assert_eq!(finished.len(), 5);
        for (i, f) in finished.iter().enumerate() {
            assert_eq!(f.id, ids[i], "finish order = add order");
            let mut p = system.to_vec();
            p.push(tails[i]);
            assert_eq!(f.generated, greedy(cfg, &w, &p, 4), "request {i} decode");
            if i == 0 {
                assert_eq!(f.reused_tokens, 0);
                assert_eq!(f.computed_tokens, 9);
            } else {
                assert_eq!(f.reused_tokens, 8, "request {i} reuses the system prompt");
                assert_eq!(f.computed_tokens, 1);
            }
        }
        let st = *eng.stats();
        assert_eq!(st.requests, 5);
        assert_eq!(st.prompt_tokens_total, 45);
        assert_eq!(st.prompt_tokens_reused, 32);
        assert_eq!(st.prompt_tokens_computed, 13);
        assert!(st.savings() > 0.7);
        assert_eq!(st.decoded_tokens, 20);
    }

    #[test]
    fn capacity_slots_reuse_after_finish() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 17).expect("weights");
        // Capacity 2 but 5 requests: slots must be reused as requests finish.
        let mut eng = CpuEngine::new(cfg, w.clone(), 2, 32, 4);
        for i in 0..5u32 {
            eng.add(&[10 + i, 20 + i, 30 + i, 40 + i], 2).expect("add");
        }
        eng.step_until_done().expect("run");
        assert!(eng.is_idle());
        assert_eq!(
            eng.finished().len(),
            5,
            "all queued requests complete via slot reuse"
        );
        assert_eq!(eng.stats().requests, 5);
    }

    #[test]
    fn too_small_pool_errors_through_step() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 19).expect("weights");
        // Pool holds 1 block but a request needs 2 fresh pages -> error.
        let mut eng = CpuEngine::new(cfg, w, 2, 1, 4);
        eng.add(&[1, 2, 3, 4, 5, 6, 7, 8], 1).expect("add");
        assert!(
            eng.step().is_err(),
            "too-small pool surfaces as a step error"
        );
    }

    #[test]
    fn drain_finished_clears_records() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 23).expect("weights");
        let mut eng = CpuEngine::new(cfg, w, 2, 32, 4);
        eng.add(&[1, 2, 3, 4], 1).expect("add");
        eng.step_until_done().expect("run");
        assert_eq!(eng.finished().len(), 1);
        let drained = eng.drain_finished();
        assert_eq!(drained.len(), 1);
        assert!(eng.finished().is_empty(), "records cleared after drain");
    }

    #[test]
    fn empty_engine_is_idle() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 1).expect("weights");
        let mut eng = CpuEngine::new(cfg, w, 2, 16, 4);
        assert!(eng.is_idle());
        assert!(!eng.step().expect("step"), "idle engine reports no work");
    }

    #[test]
    fn admission_queue_is_bounded() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 71).expect("weights");
        // `add` always queues (slots fill on step); max_pending caps the queue.
        let mut eng = CpuEngine::new_with_limits(cfg, w, 1, 16, 4, 1, usize::MAX);
        eng.add(&[1, 2, 3, 4], 1).expect("queued (pending 1)");
        assert!(
            eng.add(&[5, 6, 7, 8], 1).is_err(),
            "queue full -> backpressure"
        );
        // Run drains the queue; a new request can queue again.
        eng.step_until_done().expect("run");
        eng.add(&[9, 10, 11, 12], 1).expect("space freed");
        eng.step_until_done().expect("run");
        eng.step_until_done().expect("run");
    }

    #[test]
    fn finished_records_are_bounded() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 72).expect("weights");
        let mut eng = CpuEngine::new_with_limits(cfg, w, 2, 32, 4, usize::MAX, 2);
        for i in 0..3u32 {
            eng.add(&[1 + i, 2 + i, 3 + i, 4 + i], 1).expect("add");
        }
        eng.step_until_done().expect("run");
        assert_eq!(
            eng.finished().len(),
            2,
            "oldest record dropped beyond max_finished"
        );
    }

    #[test]
    fn empty_prompt_is_rejected() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 1).expect("weights");
        let mut eng = CpuEngine::new(cfg, w, 2, 16, 4);
        assert!(eng.add(&[], 1).is_err());
    }
}
