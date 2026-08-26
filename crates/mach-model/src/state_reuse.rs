//! Agentic state reuse: token-boundary anchors + incremental prefill.
//!
//! Multi-turn tool-calling / CoT conversations prefill the same long prefix on
//! every turn. This module provides lightweight, token-boundary checkpoints
//! ("anchors"): the per-layer KV prefix up to a position plus that position's
//! final hidden state. A new sequence that shares the prefix restores the
//! anchor (copies the KV prefix + hidden state) and only prefills the delta —
//! avoiding a full re-computation of the shared prefix. This is the MachServe
//! port of FreeToken's "agentic state reuse" (multi-turn TTFT -65..-80%).
//!
//! The module is **layout-agnostic**: KV snapshots are opaque byte blobs that
//! each model implementation serializes and restores into its own cache layout
//! ([`crate::ref_model::RefModel`] on CPU, [`crate::batched::BatchedModel`] on
//! HIP). Correctness is pinned by the CPU reference pair in
//! `tests/state_reuse.rs`: reuse logits must equal full-recompute logits.

use std::collections::{HashMap, VecDeque};

/// Opaque per-layer KV prefix snapshot.
///
/// `layers[i]` holds the `(k, v)` byte blobs for layer `i` covering positions
/// `0..=token_idx`. The model that created the anchor restores the bytes into
/// its own cache layout (dtype element size included), so the store treats
/// them as opaque.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvSnapshot {
    /// `(k_bytes, v_bytes)` per transformer layer.
    pub layers: Vec<(Vec<u8>, Vec<u8>)>,
}

impl KvSnapshot {
    /// Host bytes held by this snapshot.
    #[must_use]
    pub fn bytes_held(&self) -> usize {
        self.layers.iter().map(|(k, v)| k.len() + v.len()).sum()
    }
}

/// A token-boundary checkpoint: KV prefix + final hidden state.
#[derive(Debug, Clone)]
pub struct Anchor {
    /// Store-assigned id (set by [`AnchorStore::insert`]).
    pub id: u64,
    /// Index of the last token covered by this anchor (0-based); the anchor
    /// covers `token_idx + 1` prefix tokens.
    pub token_idx: usize,
    /// The prefix tokens `[0..=token_idx]` this anchor's KV covers. Used to
    /// verify a new sequence actually shares the prefix.
    pub tokens: Vec<u32>,
    /// Per-layer KV prefix snapshot (opaque bytes).
    pub kv: KvSnapshot,
    /// Final hidden state at `token_idx` (after the last layer, before the
    /// final norm + lm_head). Enables logits-at-anchor without re-running the
    /// last prefix token (continue-without-delta case).
    pub hidden: Vec<f32>,
}

/// A bounded, id-keyed anchor store.
///
/// Anchors can be large (a full KV prefix), so the store enforces a maximum
/// count and evicts oldest-first (FIFO) when full.
#[derive(Debug)]
pub struct AnchorStore {
    anchors: HashMap<u64, Anchor>,
    order: VecDeque<u64>,
    next_id: u64,
    max_anchors: usize,
    bytes_held: usize,
}

impl AnchorStore {
    /// Creates an empty store holding at most `max_anchors` anchors.
    #[must_use]
    pub fn new(max_anchors: usize) -> Self {
        Self {
            anchors: HashMap::new(),
            order: VecDeque::new(),
            next_id: 1,
            max_anchors,
            bytes_held: 0,
        }
    }

    /// Inserts an anchor (assigning its id) and returns the id. Evicts the
    /// oldest anchor when the store is at capacity.
    pub fn insert(&mut self, mut anchor: Anchor) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        anchor.id = id;
        // Zero-capacity store: retain nothing (the pop_front guard below would
        // otherwise no-op and the store would grow unbounded).
        if self.max_anchors == 0 {
            return id;
        }
        if self.anchors.len() >= self.max_anchors
            && let Some(oldest) = self.order.pop_front()
        {
            self.remove(oldest);
        }
        self.bytes_held +=
            anchor.kv.bytes_held() + anchor.tokens.len() * 4 + anchor.hidden.len() * 4;
        self.anchors.insert(id, anchor);
        self.order.push_back(id);
        id
    }

    /// Looks up an anchor by id.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Anchor> {
        self.anchors.get(&id)
    }

    /// Removes an anchor, releasing its host bytes.
    pub fn remove(&mut self, id: u64) {
        if let Some(a) = self.anchors.remove(&id) {
            self.bytes_held = self
                .bytes_held
                .saturating_sub(a.kv.bytes_held() + a.tokens.len() * 4 + a.hidden.len() * 4);
            self.order.retain(|&x| x != id);
        }
    }

    /// Removes every anchor.
    pub fn clear(&mut self) {
        self.anchors.clear();
        self.order.clear();
        self.bytes_held = 0;
    }

    /// Number of anchors currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.anchors.len()
    }

    /// True when no anchors are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }

    /// Approximate host bytes held by all anchors.
    #[must_use]
    pub fn bytes_held(&self) -> usize {
        self.bytes_held
    }

    /// Maximum number of anchors.
    #[must_use]
    pub const fn max_anchors(&self) -> usize {
        self.max_anchors
    }
}

/// A successful prefix reuse decision.
#[derive(Debug, Clone)]
pub struct ReusedPrefix {
    /// Id of the anchor that was matched.
    pub anchor_id: u64,
    /// Number of prefix tokens that will NOT be re-prefilled
    /// (`anchor.token_idx + 1`).
    pub prefix_len: usize,
    /// The saved hidden state at the last prefix token (continue-without-delta
    /// support; ignored when there is a delta to prefill).
    pub hidden: Vec<f32>,
}

/// Reuse accounting (for TTFT-reduction reporting).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReuseStats {
    /// Total `find_reusable` calls.
    pub lookups: u64,
    /// Lookups that matched an anchor.
    pub hits: u64,
    /// Prefix tokens that were skipped across all hits.
    pub tokens_reused: u64,
}

/// The agentic state-reuse orchestrator: anchor store + longest-prefix match.
#[derive(Debug)]
pub struct StateReuse {
    store: AnchorStore,
    stats: ReuseStats,
}

impl StateReuse {
    /// Creates a reuser with a bounded anchor store.
    #[must_use]
    pub fn new(max_anchors: usize) -> Self {
        Self {
            store: AnchorStore::new(max_anchors),
            stats: ReuseStats::default(),
        }
    }

    /// Inserts an anchor (see [`AnchorStore::insert`]).
    pub fn insert_anchor(&mut self, anchor: Anchor) -> u64 {
        self.store.insert(anchor)
    }

    /// The underlying anchor store.
    #[must_use]
    pub fn store(&self) -> &AnchorStore {
        &self.store
    }

    /// Mutable access to the underlying anchor store.
    pub fn store_mut(&mut self) -> &mut AnchorStore {
        &mut self.store
    }

    /// Finds the **longest** anchor whose token prefix is a proper prefix of
    /// `tokens` (at least one delta token must remain; a zero-delta "continue"
    /// is not a prefill decision). Updates hit/miss statistics.
    ///
    /// The caller then restores the anchor's KV + hidden into its model and
    /// prefills only `tokens[prefix_len..]`.
    pub fn find_reusable(&mut self, tokens: &[u32]) -> Option<ReusedPrefix> {
        self.stats.lookups += 1;
        let mut best: Option<(u64, usize)> = None; // (anchor_id, prefix_len)
        for (&id, a) in &self.store.anchors {
            let prefix_len = a.token_idx + 1;
            if prefix_len < tokens.len()
                && tokens.len() >= prefix_len
                && tokens[..prefix_len] == a.tokens[..]
                && best.is_none_or(|(_, bp)| prefix_len > bp)
            {
                best = Some((id, prefix_len));
            }
        }
        let (anchor_id, prefix_len) = best?;
        let hidden = self.store.anchors[&anchor_id].hidden.clone();
        self.stats.hits += 1;
        self.stats.tokens_reused += prefix_len as u64;
        Some(ReusedPrefix {
            anchor_id,
            prefix_len,
            hidden,
        })
    }

    /// Current reuse statistics.
    #[must_use]
    pub fn stats(&self) -> ReuseStats {
        self.stats
    }

    /// Resets reuse statistics (anchors are kept).
    pub fn reset_stats(&mut self) {
        self.stats = ReuseStats::default();
    }
}
