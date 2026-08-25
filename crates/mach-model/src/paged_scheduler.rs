//! CPU-side paged scheduler: scheduler FSM + prefix-reuse planner + prefix KV
//! cache driving the reference model for many requests.
//!
//! This is the CPU proof of the TokenSpeed scheduler alignment: requests go
//! through the lifecycle FSM (`Submitted -> PrefillDone -> Decoding ->
//! Finished`), the prefill is served by [`crate::prefix_kv::PrefixKvCache`]
//! (which reuses any cached prefix and computes only the delta), and decoding
//! is greedy next-token. Aggregate stats report the prefix-sharing savings —
//! the FreeToken "multi-turn TTFT -65..-80%" goal, measured in tokens computed
//! vs reused. The GPU (`batched.rs`) wiring replaces the reference model with
//! the real kernels; the scheduler contract is identical.

use crate::prefix_kv::{PrefixKvCache, PrefixReuseStats};
use crate::ref_model::RefModel;
use crate::scheduler_fsm::{
    Bootstrapping, FsmEvent, RequestState, ScheduleDecode, SchedulePrefillFirstChunk,
};
use crate::{Config, Error, Weights};

/// Outcome of serving one request.
pub struct ServedRequest {
    /// Next-token logits right after the full prompt (prefill result).
    pub prompt_logits: Vec<f32>,
    /// Greedily decoded tokens (argmax), length `decode_len`.
    pub decoded: Vec<u32>,
    /// Prefix-reuse stats of this request's prompt prefill.
    pub stats: PrefixReuseStats,
}

/// Aggregate prefix-sharing savings across all served requests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedulerStats {
    pub requests: usize,
    pub prompt_tokens_total: usize,
    pub prompt_tokens_reused: usize,
    pub prompt_tokens_computed: usize,
    pub decoded_tokens: usize,
}

impl SchedulerStats {
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

/// CPU-side paged scheduler over the reference model.
pub struct PagedScheduler {
    cache: PrefixKvCache,
    cfg: Config,
    w: Weights,
    aggregate: SchedulerStats,
}

impl PagedScheduler {
    /// Builds a scheduler with a `pool_blocks`-block KV pool and
    /// `tokens_per_page`-token pages.
    #[must_use]
    pub fn new(cfg: Config, w: Weights, pool_blocks: i32, tokens_per_page: usize) -> Self {
        Self {
            cache: PrefixKvCache::new(pool_blocks, tokens_per_page),
            cfg,
            w,
            aggregate: SchedulerStats::default(),
        }
    }

    /// Aggregate prefix-sharing stats.
    #[must_use]
    pub fn stats(&self) -> &SchedulerStats {
        &self.aggregate
    }

    /// Serves one request: FSM `Bootstrapping -> ... -> Finished`, prefill via
    /// the prefix cache (delta-only), then greedy decode. Errors when the KV
    /// pool cannot satisfy the request's fresh-page demand.
    pub fn serve(&mut self, prompt: &[u32], decode_len: usize) -> Result<ServedRequest, Error> {
        if prompt.is_empty() {
            return Err(Error::InvalidArgument(
                "paged scheduler rejects an empty prompt".into(),
            ));
        }
        // FSM: Bootstrapping -> Submitted, carrying the prompt + budget.
        let mut state = FsmEvent::Bootstrapped.apply(RequestState::Bootstrapping(Bootstrapping));
        if let RequestState::Submitted(s) = &mut state {
            s.tokens = prompt.iter().map(|&t| t as i32).collect();
            s.max_new_tokens = decode_len as i32;
        }

        let mut model = RefModel::new(self.cfg, self.w.clone());
        let (prompt_logits, stats) = self.cache.serve(&mut model, prompt)?;

        // FSM: Submitted -> PrefillDone (one chunk covers the whole prompt).
        state = FsmEvent::SchedulePrefillFirstChunk(SchedulePrefillFirstChunk {
            chunk_size: prompt.len() as i32,
            reserve_tokens: 0,
        })
        .apply(state);
        debug_assert!(
            matches!(state, RequestState::PrefillDone(_)),
            "single-chunk prefill must complete the prompt"
        );

        // FSM: PrefillDone -> Decoding.
        state = FsmEvent::ScheduleDecode(ScheduleDecode { reserve_tokens: 0 }).apply(state);

        // Greedy decode: argmax each step, extending the FSM + the model.
        let mut decoded = Vec::with_capacity(decode_len);
        let mut logits = prompt_logits.clone();
        for _ in 0..decode_len {
            let tok = argmax(&logits);
            decoded.push(tok);
            logits = model.decode_step(tok);
            state = FsmEvent::ExtendResult(vec![tok as i32]).apply(state);
        }
        state = FsmEvent::Succeeded.apply(state);
        debug_assert!(
            matches!(state, RequestState::Finished),
            "decode must terminate in Finished"
        );

        self.aggregate.requests += 1;
        self.aggregate.prompt_tokens_total += stats.total_tokens;
        self.aggregate.prompt_tokens_reused += stats.reused_tokens;
        self.aggregate.prompt_tokens_computed += stats.computed_tokens;
        self.aggregate.decoded_tokens += decode_len;

        Ok(ServedRequest {
            prompt_logits,
            decoded,
            stats,
        })
    }
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
    fn shared_system_prompt_saved_across_requests() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 21).expect("weights");
        let system = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let tails = [100u32, 101, 102, 103, 104];
        let mut sched = PagedScheduler::new(cfg, w.clone(), 32, 4);

        for (i, &tail) in tails.iter().enumerate() {
            let mut prompt = system.to_vec();
            prompt.push(tail);
            let out = sched.serve(&prompt, 3).expect("serve");
            if i == 0 {
                assert_eq!(out.stats.reused_tokens, 0);
                assert_eq!(out.stats.computed_tokens, 9);
            } else {
                assert_eq!(
                    out.stats.reused_tokens, 8,
                    "request {i} reuses the system prompt"
                );
                assert_eq!(
                    out.stats.computed_tokens, 1,
                    "request {i} computes only the tail"
                );
            }
            // Greedy decode must match a fresh-model full recompute.
            assert_eq!(
                out.decoded,
                greedy(cfg, &w, &prompt, 3),
                "request {i} decode"
            );
        }

        let st = *sched.stats();
        assert_eq!(st.requests, 5);
        assert_eq!(st.prompt_tokens_total, 45);
        assert_eq!(st.prompt_tokens_reused, 32);
        assert_eq!(st.prompt_tokens_computed, 13);
        let savings = st.savings();
        assert!(
            savings > 0.7,
            "5 shared-prefix requests should reuse >70% of prompt tokens (got {savings:.2})"
        );
    }

    #[test]
    fn empty_prompt_is_rejected() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 8).expect("weights");
        let mut sched = PagedScheduler::new(cfg, w, 16, 4);
        assert!(sched.serve(&[], 1).is_err());
        assert_eq!(sched.stats().requests, 0);
    }
    #[test]
    fn zero_decode_len_still_finishes() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 8).expect("weights");
        let mut sched = PagedScheduler::new(cfg, w.clone(), 16, 4);
        let out = sched.serve(&[1, 2, 3, 4], 0).expect("serve");
        assert!(out.decoded.is_empty());
        assert_eq!(sched.stats().decoded_tokens, 0);
        assert_eq!(sched.stats().requests, 1);
    }
}
