//! Continuous-batching engine.
//!
//! Wraps [`BatchedModel`] with sequence lifecycle management: sequences are
//! added with a prompt, advance one token per engine step (prefill tokens are
//! consumed first, then greedy decode), and finish on EOS or `max_new`.
//! Finished sequences free their KV slot (compaction moves higher slots down),
//! so new sequences can join at any step. Prefill and decode are mixed in the
//! same batched step, exactly like a production continuous-batching server.
//!
//! Each sequence has a **stable** [`SeqId`] independent of its (changing) KV
//! slot, so callers can track outputs across compaction.

use crate::batched::BatchedModel;
use crate::sampling::SamplingParams;
use crate::state_reuse::{ReuseStats, StateReuse};
use crate::{Config, Error, Weights};
use mach_kernel_sys::hip::Hip;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Stable sequence identifier.
pub type SeqId = u64;

/// True when `generated` ends with any of `seqs` (OpenAI stop sequences).
#[must_use]
fn ends_with_stop(generated: &[u32], seqs: &[Vec<u32>]) -> bool {
    seqs.iter().any(|s| {
        !s.is_empty() && generated.len() >= s.len() && &generated[generated.len() - s.len()..] == s
    })
}

/// Per-sequence state in the engine.
#[derive(Debug)]
struct SeqState {
    id: SeqId,
    /// Remaining prompt tokens (prefill queue).
    prompt: VecDeque<u32>,
    /// Generated tokens so far.
    generated: Vec<u32>,
    /// Full token stream (prompt + generated) for anchor saving.
    all_tokens: Vec<u32>,
    /// Maximum generated tokens.
    max_new: usize,
    /// Per-sequence sampling configuration (seed advances every step).
    params: SamplingParams,
    /// EOS token id (None disables early stopping).
    eos: Option<u32>,
    /// Stop sequences: generation finishes as soon as `generated` ends with
    /// any of these token sequences (OpenAI `stop`).
    stop_seqs: Vec<Vec<u32>>,
    /// Per-token log-probabilities of `generated` (OpenAI `logprobs`).
    logprobs: Vec<f32>,
    /// Per-token top-`k` (token, logprob) lists (OpenAI `top_logprobs`).
    top_logprobs: Vec<Vec<(u32, f32)>>,
    /// Token occurrence counts of `generated` (presence/frequency penalties).
    counts: HashMap<u32, u32>,
    /// Static logit_bias (token, value) pairs applied to every sampled step.
    logit_bias: Vec<(u32, f32)>,
    /// Number of KV positions consumed (prompt + generated).
    len: usize,
    /// The token to feed next (first generated token after prefill, then each
    /// subsequent sampled token).
    first_decode: Option<u32>,
}

/// Finished sequence output retained until `ack` (keyed by stable id).
#[derive(Debug)]
struct FinishedSeq {
    id: SeqId,
    generated: Vec<u32>,
    logprobs: Vec<f32>,
    top_logprobs: Vec<Vec<(u32, f32)>>,
    stopped: bool,
}

/// Continuous-batching engine over a fixed-capacity batched model.
pub struct ContinuousModel {
    model: BatchedModel,
    /// Rows consumed per prefill step (>= capacity for chunked prefill).
    prefill_rows: usize,
    /// Active slot state; active slots always occupy `[0, active)`.
    seqs: Vec<Option<SeqState>>,
    active: usize,
    /// Finished sequences' outputs, keyed by stable id.
    finished: Vec<FinishedSeq>,
    next_id: SeqId,
    /// Agentic state reuse (opt-in): auto-saves anchors on finish and restores
    /// matching prefixes on add (incremental prefill).
    state_reuse: Option<StateReuse>,
}

unsafe impl Send for ContinuousModel {}

impl ContinuousModel {
    /// Builds an engine with `capacity` concurrent sequence slots.
    pub fn new(hip: Arc<Hip>, cfg: Config, w: &Weights, capacity: usize) -> Result<Self, Error> {
        Self::with_prefill_rows(hip, cfg, w, capacity, capacity)
    }

    /// Builds an engine with `capacity` slots and `prefill_rows` rows per
    /// prefill step (`>= capacity`): longer prompts prefill in fewer, larger
    /// steps).
    pub fn with_prefill_rows(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        capacity: usize,
        prefill_rows: usize,
    ) -> Result<Self, Error> {
        let model = BatchedModel::with_rows(hip, cfg, w, capacity, prefill_rows.max(capacity))?;
        Ok(Self {
            model,
            prefill_rows: prefill_rows.max(capacity),
            seqs: (0..capacity).map(|_| None).collect(),
            active: 0,
            finished: Vec::new(),
            next_id: 1,
            state_reuse: None,
        })
    }

    /// Builds a continuous-batching engine in MoE offload mode: experts stay in host
    /// RAM and the MoE layer is computed on the CPU (FreeToken `cpu` backend), so GPU
    /// memory is bounded by `expert_slots` regardless of the total expert count.
    pub fn with_prefill_rows_offload(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        capacity: usize,
        prefill_rows: usize,
        expert_slots: usize,
    ) -> Result<Self, Error> {
        let model = BatchedModel::with_expert_slots(
            hip,
            cfg,
            w,
            capacity,
            prefill_rows.max(capacity),
            expert_slots,
        )?;
        Ok(Self {
            model,
            prefill_rows: prefill_rows.max(capacity),
            seqs: (0..capacity).map(|_| None).collect(),
            active: 0,
            finished: Vec::new(),
            next_id: 1,
            state_reuse: None,
        })
    }

    /// Builds a continuous-batching engine in agentic state-reuse mode: every
    /// finished sequence leaves a token-boundary anchor, and a later sequence
    /// sharing that prefix restores it and prefills only the delta (multi-turn
    /// TTFT reduction). Existing constructors keep their exact behavior.
    pub fn with_state_reuse(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        capacity: usize,
        state_reuse: StateReuse,
    ) -> Result<Self, Error> {
        let mut m = Self::with_prefill_rows(hip, cfg, w, capacity, capacity)?;
        m.state_reuse = Some(state_reuse);
        Ok(m)
    }

    /// Maximum concurrent sequences.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.seqs.len()
    }

    /// Sequences still being processed (not yet finished).
    #[must_use]
    pub const fn active(&self) -> usize {
        self.active
    }

    /// State-reuse statistics when reuse mode is enabled, else `None`.
    #[must_use]
    pub fn reuse_stats(&self) -> Option<ReuseStats> {
        self.state_reuse.as_ref().map(StateReuse::stats)
    }

    /// Number of anchors currently held (reuse mode only).
    #[must_use]
    pub fn anchors_len(&self) -> Option<usize> {
        self.state_reuse.as_ref().map(|sr| sr.store().len())
    }

    /// Adds a sequence; returns its stable id.
    pub fn add(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        stop_seqs: Vec<Vec<u32>>,
        logit_bias: Vec<(u32, f32)>,
        mut params: SamplingParams,
    ) -> Result<SeqId, Error> {
        if prompt.is_empty() {
            return Err(Error::Model("prompt must not be empty".into()));
        }
        if self.active >= self.capacity() {
            return Err(Error::Model("engine at capacity".into()));
        }
        // A zero seed means 'unspecified': derive a unique per-sequence seed
        // so independent sequences do not share a draw schedule.
        if params.seed == 0 {
            params.seed = self.next_id;
        }
        let id = self.next_id;
        self.next_id += 1;
        let slot = self.active;
        // State reuse: restore the longest matching prefix anchor and prefill
        // only the delta (incremental prefill).
        let mut len = 0usize;
        let mut pending: VecDeque<u32> = prompt.iter().copied().collect();
        if let Some(sr) = &mut self.state_reuse
            && let Some(reused) = sr.find_reusable(prompt)
        {
            let anchor = sr
                .store()
                .get(reused.anchor_id)
                .expect("matched anchor must be present");
            self.model
                .restore_anchor(slot, anchor)
                .map_err(|e| Error::Model(format!("state-reuse restore failed: {e}")))?;
            len = reused.prefix_len;
            pending = prompt[reused.prefix_len..].iter().copied().collect();
        }
        self.seqs[slot] = Some(SeqState {
            id,
            prompt: pending,
            generated: Vec::new(),
            all_tokens: prompt.to_vec(),
            max_new,
            eos,
            stop_seqs,
            logprobs: Vec::new(),
            top_logprobs: Vec::new(),
            counts: HashMap::new(),
            logit_bias,
            params,
            len,
            first_decode: None,
        });
        self.active += 1;
        Ok(id)
    }

    /// Advances the engine by one batched step (chunked prefill + decode).
    ///
    /// Each step consumes up to `capacity` rows: a sequence still prefilling
    /// contributes up to the remaining budget of its pending prompt tokens
    /// (chunked prefill — one forward position per prompt token), a sequence
    /// already decoding contributes its next token (one row). Prefill and
    /// decode mix in the same batched forward.
    ///
    /// Returns `(seq_id, token)` for each sequence that produced a *real*
    /// token this step: the first generated token when prefill completes, or a
    /// decode token. Sequences still prefilling produce no entry (their
    /// per-position predictions are internal).
    pub fn step(&mut self) -> Result<Vec<(SeqId, u32)>, Error> {
        if self.active == 0 {
            return Ok(Vec::new());
        }
        let mut tokens = Vec::new();
        let mut lens = Vec::new();
        let mut slots = Vec::new();
        let mut params = Vec::new();
        let mut row_counts: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut row_bias: Vec<Vec<(u32, f32)>> = Vec::new();
        // (row_start, row_count, was_prefill) per active slot.
        let mut rows: Vec<(usize, usize, bool)> = Vec::with_capacity(self.active);
        let mut budget = self.prefill_rows;
        for i in 0..self.active {
            let s = self.seqs[i].as_ref().expect("active slot");
            if budget == 0 {
                rows.push((tokens.len(), 0, false));
                continue;
            }
            if !s.prompt.is_empty() {
                let take = s.prompt.len().min(budget);
                for j in 0..take {
                    tokens.push(s.prompt[j]);
                    lens.push((s.len + j) as u32);
                    slots.push(i as u32); // all rows of seq i live in slot i
                    params.push(s.params);
                    row_counts.push(Vec::new()); // no generated history during prefill
                    row_bias.push(s.logit_bias.clone());
                }
                rows.push((tokens.len() - take, take, true));
                budget -= take;
            } else {
                tokens.push(s.first_decode.expect("decode requires a prior token"));
                lens.push(s.len as u32);
                slots.push(i as u32);
                params.push(s.params);
                row_counts.push(s.counts.iter().map(|(&t, &c)| (t, c)).collect());
                row_bias.push(s.logit_bias.clone());
                rows.push((tokens.len() - 1, 1, false));
                budget -= 1;
            }
        }

        let (sampled, logprobs, topk) = self.model.decode_step_explicit(
            &tokens,
            &lens,
            &slots,
            &mut params,
            &row_counts,
            &row_bias,
        )?;
        // The sampler advanced each row's seed one RNG step (rows of one
        // sequence start from the same seed); the last row's value is the
        // sequence's authoritative next seed.
        for (i, &(start, count, _)) in rows.iter().enumerate() {
            if count > 0 {
                let p = params[start + count - 1];
                self.seqs[i].as_mut().expect("active slot").params = p;
            }
        }

        let mut done_slots = Vec::new();
        let mut outputs = Vec::new();
        for (i, &(start, count, was_prefill)) in rows.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let s = self.seqs[i].as_mut().expect("active slot");
            if was_prefill {
                let last_out = sampled[start + count - 1];
                for _ in 0..count {
                    s.prompt.pop_front();
                }
                s.len += count;
                if s.prompt.is_empty() {
                    s.generated.push(last_out);
                    s.all_tokens.push(last_out);
                    s.logprobs.push(logprobs[start + count - 1]);
                    s.top_logprobs.push(topk[start + count - 1].clone());
                    *s.counts.entry(last_out).or_insert(0) += 1;
                    s.first_decode = Some(last_out);
                    if s.eos.is_some_and(|e| last_out == e)
                        || ends_with_stop(&s.generated, &s.stop_seqs)
                        || s.generated.len() >= s.max_new
                    {
                        done_slots.push(i);
                    }
                    outputs.push((s.id, last_out));
                }
            } else {
                let out = sampled[start];
                s.generated.push(out);
                s.all_tokens.push(out);
                s.logprobs.push(logprobs[start]);
                s.top_logprobs.push(topk[start].clone());
                *s.counts.entry(out).or_insert(0) += 1;
                s.len += 1;
                s.first_decode = Some(out);
                if s.eos.is_some_and(|e| out == e)
                    || ends_with_stop(&s.generated, &s.stop_seqs)
                    || s.generated.len() >= s.max_new
                {
                    done_slots.push(i);
                }
                outputs.push((s.id, out));
            }
        }
        for &slot in done_slots.iter().rev() {
            self.finish(slot);
        }
        Ok(outputs)
    }

    /// True when every sequence has finished.
    #[must_use]
    pub fn all_done(&self) -> bool {
        self.active == 0
    }

    /// Whether the sequence with `id` has finished (or is unknown).
    #[must_use]
    pub fn is_done(&self, id: SeqId) -> bool {
        if self.finished.iter().any(|f| f.id == id) {
            return true;
        }
        self.seqs.iter().flatten().find(|s| s.id == id).is_none()
    }

    /// Generated tokens of the sequence with `id` (empty if unknown).
    #[must_use]
    pub fn generated(&self, id: SeqId) -> Vec<u32> {
        if let Some(f) = self.finished.iter().find(|f| f.id == id) {
            return f.generated.clone();
        }
        self.seqs
            .iter()
            .flatten()
            .find(|s| s.id == id)
            .map(|s| s.generated.clone())
            .unwrap_or_default()
    }

    /// Per-token log-probabilities of the generated output (OpenAI
    /// `logprobs`), empty when unknown.
    #[must_use]
    pub fn generated_logprobs(&self, id: SeqId) -> Vec<f32> {
        if let Some(f) = self.finished.iter().find(|f| f.id == id) {
            return f.logprobs.clone();
        }
        self.seqs
            .iter()
            .flatten()
            .find(|s| s.id == id)
            .map(|s| s.logprobs.clone())
            .unwrap_or_default()
    }

    /// Per-token top-`k` (token, logprob) lists of the generated output
    /// (OpenAI `top_logprobs`), empty when unknown or not requested.
    #[must_use]
    pub fn generated_top_logprobs(&self, id: SeqId) -> Vec<Vec<(u32, f32)>> {
        if let Some(f) = self.finished.iter().find(|f| f.id == id) {
            return f.top_logprobs.clone();
        }
        self.seqs
            .iter()
            .flatten()
            .find(|s| s.id == id)
            .map(|s| s.top_logprobs.clone())
            .unwrap_or_default()
    }

    /// OpenAI finish reason for a finished sequence: `"stop"` when the last
    /// token was the EOS or a stop sequence, `"length"` otherwise. Unknown ids
    /// (or still-active sequences) report `"length"`.
    #[must_use]
    pub fn finish_reason(&self, id: SeqId) -> &'static str {
        if self.finished.iter().any(|f| f.id == id && f.stopped) {
            "stop"
        } else {
            "length"
        }
    }

    /// Removes a finished sequence from the finished list (freeing bookkeeping).
    pub fn ack(&mut self, id: SeqId) {
        self.finished.retain(|f| f.id != id);
    }

    fn finish(&mut self, slot: usize) {
        assert!(slot < self.active, "finish out of range");
        // State-reuse mode: leave a token-boundary anchor at the sequence end
        // so a later turn sharing this prefix skips it (incremental prefill).
        if let Some(sr) = &mut self.state_reuse
            && let Some(s) = self.seqs[slot].as_ref()
            && !s.all_tokens.is_empty()
            && let Ok(anchor) = self
                .model
                .save_anchor(slot, &s.all_tokens, s.all_tokens.len() - 1)
        {
            sr.insert_anchor(anchor);
        }
        let (id, generated, logprobs, top_logprobs, stopped) = {
            let s = self.seqs[slot].as_ref().expect("active slot");
            let stopped = s.generated.last().is_some_and(|&t| s.eos == Some(t))
                || ends_with_stop(&s.generated, &s.stop_seqs);
            (
                s.id,
                s.generated.clone(),
                s.logprobs.clone(),
                s.top_logprobs.clone(),
                stopped,
            )
        };
        self.finished.push(FinishedSeq {
            id,
            generated,
            logprobs,
            top_logprobs,
            stopped,
        });
        // Compact: move every sequence above `slot` down by one, copying KV.
        for i in (slot + 1)..self.active {
            let from = i;
            let to = i - 1;
            let len = self.seqs[i].as_ref().expect("active slot").len;
            self.model
                .copy_seq_kv(from, to, len)
                .expect("compaction KV copy");
            self.seqs[to] = self.seqs[i].take();
        }
        self.seqs[self.active - 1] = None;
        self.active -= 1;
    }
}
