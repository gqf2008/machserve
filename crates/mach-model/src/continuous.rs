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
use crate::{Config, Error, Weights};
use mach_kernel_sys::hip::Hip;
use std::collections::VecDeque;
use std::sync::Arc;

/// Stable sequence identifier.
pub type SeqId = u64;

/// Per-sequence state in the engine.
#[derive(Debug)]
struct SeqState {
    id: SeqId,
    /// Remaining prompt tokens (prefill queue).
    prompt: VecDeque<u32>,
    /// Generated tokens so far.
    generated: Vec<u32>,
    /// Maximum generated tokens.
    max_new: usize,
    /// EOS token id (None disables early stopping).
    eos: Option<u32>,
    /// Number of KV positions consumed (prompt + generated).
    len: usize,
    /// The token to feed next (first generated token after prefill, then each
    /// subsequent sampled token).
    first_decode: Option<u32>,
}

/// Continuous-batching engine over a fixed-capacity batched model.
pub struct ContinuousModel {
    model: BatchedModel,
    /// Active slot state; active slots always occupy `[0, active)`.
    seqs: Vec<Option<SeqState>>,
    active: usize,
    /// Finished sequences' outputs, keyed by stable id.
    finished: Vec<(SeqId, Vec<u32>)>,
    next_id: SeqId,
}

impl ContinuousModel {
    /// Builds an engine with `capacity` concurrent sequence slots.
    pub fn new(hip: Arc<Hip>, cfg: Config, w: &Weights, capacity: usize) -> Result<Self, Error> {
        let model = BatchedModel::new(hip, cfg, w, capacity)?;
        Ok(Self {
            model,
            seqs: (0..capacity).map(|_| None).collect(),
            active: 0,
            finished: Vec::new(),
            next_id: 1,
        })
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

    /// Adds a sequence; returns its stable id.
    pub fn add(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
    ) -> Result<SeqId, Error> {
        if prompt.is_empty() {
            return Err(Error::Model("prompt must not be empty".into()));
        }
        if self.active >= self.capacity() {
            return Err(Error::Model("engine at capacity".into()));
        }
        let id = self.next_id;
        self.next_id += 1;
        let slot = self.active;
        self.seqs[slot] = Some(SeqState {
            id,
            prompt: prompt.iter().copied().collect(),
            generated: Vec::new(),
            max_new,
            eos,
            len: 0,
            first_decode: None,
        });
        self.active += 1;
        Ok(id)
    }

    /// Advances all active sequences by one token (prefill or decode).
    /// Returns `(seq_id, sampled_token)` for each sequence that ran.
    pub fn step(&mut self) -> Result<Vec<(SeqId, u32)>, Error> {
        if self.active == 0 {
            return Ok(Vec::new());
        }
        let mut tokens = Vec::with_capacity(self.active);
        let mut lens = Vec::with_capacity(self.active);
        for i in 0..self.active {
            let s = self.seqs[i].as_ref().expect("active slot");
            let tok = if !s.prompt.is_empty() {
                s.prompt[0]
            } else {
                s.first_decode.expect("decode requires a prior token")
            };
            tokens.push(tok);
            lens.push(s.len as u32);
        }

        let sampled = self.model.decode_step_explicit(&tokens, &lens)?;

        // Update per-sequence state; track finished slots (by slot index).
        let mut done_slots = Vec::new();
        for (i, &out) in sampled.iter().enumerate() {
            let s = self.seqs[i].as_mut().expect("active slot");
            if !s.prompt.is_empty() {
                // This step consumed a prompt token (prefill). The sampled
                // token predicts the next position; it becomes the first
                // generated token only when prefill is complete.
                s.prompt.pop_front();
                s.len += 1;
                if s.prompt.is_empty() {
                    s.generated.push(out);
                    s.first_decode = Some(out);
                    if s.eos.is_some_and(|e| out == e) || s.generated.len() >= s.max_new {
                        done_slots.push(i);
                    }
                }
            } else {
                s.generated.push(out);
                s.len += 1;
                s.first_decode = Some(out);
                if s.eos.is_some_and(|e| out == e) || s.generated.len() >= s.max_new {
                    done_slots.push(i);
                }
            }
        }
        let outputs: Vec<(SeqId, u32)> = (0..self.active)
            .map(|i| (self.seqs[i].as_ref().expect("active slot").id, sampled[i]))
            .collect();
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
        if self.finished.iter().any(|(fid, _)| *fid == id) {
            return true;
        }
        self.seqs.iter().flatten().find(|s| s.id == id).is_none()
    }

    /// Generated tokens of the sequence with `id` (empty if unknown).
    #[must_use]
    pub fn generated(&self, id: SeqId) -> Vec<u32> {
        if let Some((_, g)) = self.finished.iter().find(|(fid, _)| *fid == id) {
            return g.clone();
        }
        self.seqs
            .iter()
            .flatten()
            .find(|s| s.id == id)
            .map(|s| s.generated.clone())
            .unwrap_or_default()
    }

    /// Removes a finished sequence from the finished list (freeing bookkeeping).
    pub fn ack(&mut self, id: SeqId) {
        self.finished.retain(|(fid, _)| *fid != id);
    }

    fn finish(&mut self, slot: usize) {
        assert!(slot < self.active, "finish out of range");
        let (id, generated) = {
            let s = self.seqs[slot].as_ref().expect("active slot");
            (s.id, s.generated.clone())
        };
        self.finished.push((id, generated));
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
