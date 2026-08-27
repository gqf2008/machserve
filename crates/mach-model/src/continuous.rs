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
use crate::{Config, Error, Weights, WeightsFp8, WeightsQ4};
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

/// Reuse registry entry: a request's content prefix whose pages may be
/// aliased by later requests once materialized.
#[derive(Debug)]
struct PagedEntry {
    /// Prompt tokens the entry covers (page-granular prefix match target).
    prefix: Vec<u32>,
    /// True once the pages hold their KV (prefill completed); only then may
    /// later requests reuse them.
    registered: bool,
    /// Precomputed content-hash chain of `prefix` (computed once at
    /// admission; avoids rehashing when the content materializes).
    chain: Vec<String>,
}

/// Paged-KV engine bookkeeping: content-hash page builder + per-slot block
/// tables + the materialization registry that gates cross-request reuse.
struct PagedEngineState {
    tokens_per_page: usize,
    builder: crate::paged_kv::GpuPagedTableBuilder,
    /// Per active slot: the block table installed on the model.
    tables: Vec<Option<crate::paged_kv::PagedTable>>,
    /// Registry parallel to `seqs` (indexed by slot).
    entries: Vec<PagedEntry>,
    /// Finished sequences whose pages persist in the pool (still reusable).
    retired: Vec<PagedEntry>,
    /// Cross-request reuse counters (prompt-token savings).
    requests: usize,
    reused_tokens: usize,
    prompt_tokens: usize,
}

impl PagedEngineState {
    /// Largest page-aligned `r` such that some materialized entry owns
    /// `prompt[..r]` (its pages hold that KV). Reuse is therefore safe: the
    /// builder's hash chain resolves the same pages.
    fn find_reusable(&self, prompt: &[u32]) -> usize {
        let mut best = 0usize;
        for e in self.entries.iter().chain(self.retired.iter()) {
            if !e.registered {
                continue;
            }
            let mut r = 0usize;
            while r + self.tokens_per_page <= e.prefix.len()
                && r + self.tokens_per_page <= prompt.len()
                && e.prefix[r..r + self.tokens_per_page] == prompt[r..r + self.tokens_per_page]
            {
                r += self.tokens_per_page;
            }
            best = best.max(r);
        }
        best
    }

    /// Evicts the oldest retired entry whose pages **no active table
    /// references** (a page still aliased by a live request must stay): frees
    /// its pages back to the pool and unregisters their content mappings, so
    /// future plans allocate fresh pages instead of aliasing evicted ones.
    /// Whole-entry eviction keeps the reuse contract sound — a partially
    /// evicted chain would let requests skip prefill positions whose pages no
    /// longer hold KV. Returns true when an entry was evicted.
    fn evict_one_retired(&mut self) -> bool {
        let referenced: std::collections::HashSet<u32> = self
            .tables
            .iter()
            .flatten()
            .flat_map(|t| t.pages().iter().copied())
            .collect();
        // Oldest first (the retired list is append-ordered).
        for idx in 0..self.retired.len() {
            let entry = &self.retired[idx];
            let any_referenced = entry.chain.iter().any(|h| {
                self.builder
                    .page_of(h)
                    .is_some_and(|p| referenced.contains(&p))
            });
            if any_referenced {
                continue;
            }
            let entry = self.retired.remove(idx);
            for h in &entry.chain {
                self.builder.evict_page(h);
            }
            return true;
        }
        false
    }
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
    /// Paged-KV mode (opt-in): page pool + per-slot block tables with
    /// cross-request prefix sharing (delta-only past the reuse boundary).
    paged: Option<PagedEngineState>,
}

/// Reuse statistics of a paged-KV engine (cross-request prompt sharing).
#[derive(Debug, Clone, Copy, Default)]
pub struct PagedReuseStats {
    /// Requests admitted to the paged engine.
    pub requests: usize,
    /// Prompt tokens served from shared prefix pages (not recomputed).
    pub reused_tokens: usize,
    /// Total prompt tokens admitted.
    pub prompt_tokens: usize,
}

impl PagedReuseStats {
    /// Fraction of prompt tokens reused across requests (`0..=1`).
    #[must_use]
    pub fn reuse_ratio(&self) -> f32 {
        if self.prompt_tokens == 0 {
            0.0
        } else {
            self.reused_tokens as f32 / self.prompt_tokens as f32
        }
    }
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
            paged: None,
        })
    }

    /// Builds a continuous-batching engine from storage-Q4 weights: each GEMM
    /// tensor is dequantized to f16 during upload, so host RAM stays ~= the
    /// packed Q4 weights (experts stay fully GPU-resident).
    pub fn with_prefill_rows_q4(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsQ4,
        capacity: usize,
        prefill_rows: usize,
    ) -> Result<Self, Error> {
        let model = BatchedModel::with_rows_q4(hip, cfg, w, capacity, prefill_rows.max(capacity))?;
        Ok(Self {
            model,
            prefill_rows: prefill_rows.max(capacity),
            seqs: (0..capacity).map(|_| None).collect(),
            active: 0,
            finished: Vec::new(),
            next_id: 1,
            state_reuse: None,
            paged: None,
        })
    }

    /// Builds a continuous-batching engine from storage-FP8 weights: each GEMM
    /// tensor is dequantized to f16 during upload, so host RAM stays ~= the
    /// packed FP8 weights (experts stay fully GPU-resident).
    pub fn with_prefill_rows_fp8(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsFp8,
        capacity: usize,
        prefill_rows: usize,
    ) -> Result<Self, Error> {
        let model = BatchedModel::with_rows_fp8(hip, cfg, w, capacity, prefill_rows.max(capacity))?;
        Ok(Self {
            model,
            prefill_rows: prefill_rows.max(capacity),
            seqs: (0..capacity).map(|_| None).collect(),
            active: 0,
            finished: Vec::new(),
            next_id: 1,
            state_reuse: None,
            paged: None,
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
            paged: None,
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

    /// Builds a **paged-KV** continuous engine: the KV cache is addressed as a
    /// page pool through per-slot block tables, and requests sharing a prefix
    /// alias the same physical pages — the second request prefills only its
    /// delta (positions past the reuse boundary), reading the shared KV from
    /// the first request's pages. Reuse is gated by materialization: a
    /// request's pages enter the content cache only after its prefill
    /// completes, so a request can never read unwritten pages.
    ///
    /// `tokens_per_page` must divide `max_seq_len`. MLA paged mode requires
    /// F32 (asserted by the underlying model ctor).
    pub fn with_paged_prefill_rows(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        capacity: usize,
        prefill_rows: usize,
        tokens_per_page: usize,
    ) -> Result<Self, Error> {
        let model = BatchedModel::with_paged_kv_rows(
            hip,
            cfg,
            w,
            capacity,
            prefill_rows.max(capacity),
            tokens_per_page,
        )?;
        Ok(Self {
            model,
            prefill_rows: prefill_rows.max(capacity),
            seqs: (0..capacity).map(|_| None).collect(),
            active: 0,
            finished: Vec::new(),
            next_id: 1,
            state_reuse: None,
            paged: Some(Self::paged_state(&cfg, capacity, tokens_per_page)),
        })
    }

    /// [`Self::with_paged_prefill_rows`] for storage-Q4 weights (device f16):
    /// cross-request prefix reuse over the dequantized path.
    pub fn with_paged_prefill_rows_q4(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsQ4,
        capacity: usize,
        prefill_rows: usize,
        tokens_per_page: usize,
    ) -> Result<Self, Error> {
        let model = BatchedModel::with_paged_kv_rows_q4(
            hip,
            cfg,
            w,
            capacity,
            prefill_rows.max(capacity),
            tokens_per_page,
        )?;
        Ok(Self {
            model,
            prefill_rows: prefill_rows.max(capacity),
            seqs: (0..capacity).map(|_| None).collect(),
            active: 0,
            finished: Vec::new(),
            next_id: 1,
            state_reuse: None,
            paged: Some(Self::paged_state(&cfg, capacity, tokens_per_page)),
        })
    }

    /// [`Self::with_paged_prefill_rows`] for storage-FP8 weights (device f16).
    pub fn with_paged_prefill_rows_fp8(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsFp8,
        capacity: usize,
        prefill_rows: usize,
        tokens_per_page: usize,
    ) -> Result<Self, Error> {
        let model = BatchedModel::with_paged_kv_rows_fp8(
            hip,
            cfg,
            w,
            capacity,
            prefill_rows.max(capacity),
            tokens_per_page,
        )?;
        Ok(Self {
            model,
            prefill_rows: prefill_rows.max(capacity),
            seqs: (0..capacity).map(|_| None).collect(),
            active: 0,
            finished: Vec::new(),
            next_id: 1,
            state_reuse: None,
            paged: Some(Self::paged_state(&cfg, capacity, tokens_per_page)),
        })
    }

    /// Shared paged-engine bookkeeping factory (page pool sized for the full
    /// slot count; reuse counters zeroed).
    fn paged_state(cfg: &Config, capacity: usize, tokens_per_page: usize) -> PagedEngineState {
        let pages = (cfg.max_seq_len / tokens_per_page) * capacity;
        PagedEngineState {
            tokens_per_page,
            builder: crate::paged_kv::GpuPagedTableBuilder::new(pages as u32, tokens_per_page),
            tables: (0..capacity).map(|_| None).collect(),
            entries: (0..capacity)
                .map(|_| PagedEntry {
                    prefix: Vec::new(),
                    registered: false,
                    chain: Vec::new(),
                })
                .collect(),
            retired: Vec::new(),
            requests: 0,
            reused_tokens: 0,
            prompt_tokens: 0,
        }
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

    /// Cross-request prefix-reuse statistics of a paged engine, else `None`.
    #[must_use]
    pub fn paged_reuse_stats(&self) -> Option<PagedReuseStats> {
        self.paged.as_ref().map(|pg| PagedReuseStats {
            requests: pg.requests,
            reused_tokens: pg.reused_tokens,
            prompt_tokens: pg.prompt_tokens,
        })
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
        if prompt.len() > self.model.max_seq_len() {
            return Err(Error::Model(format!(
                "prompt of {} tokens exceeds max_seq_len {}",
                prompt.len(),
                self.model.max_seq_len()
            )));
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
        if let Some(pg) = &mut self.paged {
            let i32p: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
            // Reuse only materialized pages (see PagedEngineState::find_reusable);
            // otherwise compute everything into fresh pages (registered later
            // when this request's prefill completes).
            let mut r = pg.find_reusable(prompt);
            if r == prompt.len() {
                // Full-prefix reuse would leave an empty prefill queue with no
                // first decode token (step() panics). Keep the last page: it
                // is recomputed into the (identical-content) aliased page —
                // benign — and the prefill produces the first token normally.
                r -= pg.tokens_per_page;
            }
            // One hash-chain computation per admission: plan() resolves and
            // allocates; the chain is stored for materialization-time
            // registration without rehashing. Pool exhaustion evicts cold
            // retired pages and retries (bounded: each eviction removes an
            // entry; a failed plan/pad frees its own pages, so nothing leaks).
            let plan = loop {
                let mut p = match pg.builder.plan(&i32p, r > 0) {
                    Ok(p) => p,
                    Err(e) => {
                        if pg.evict_one_retired() {
                            continue;
                        }
                        return Err(Error::Model(format!("paged plan: {e}")));
                    }
                };
                match pg
                    .builder
                    .pad_table(&mut p.table, self.model.max_pages_per_seq())
                {
                    Ok(()) => break p,
                    Err(e) => {
                        pg.builder.free_plan_pages(&p, &p.table);
                        if pg.evict_one_retired() {
                            continue;
                        }
                        return Err(Error::Model(format!("paged pad_table: {e}")));
                    }
                }
            };
            self.model
                .set_block_table(slot, plan.table.pages())
                .map_err(|e| Error::Model(format!("set_block_table: {e}")))?;
            pg.tables[slot] = Some(plan.table);
            pg.entries[slot] = PagedEntry {
                prefix: prompt.to_vec(),
                registered: false,
                chain: plan.chain,
            };
            pg.requests += 1;
            pg.prompt_tokens += prompt.len();
            pg.reused_tokens += r;
            len = r;
            pending = prompt[r..].iter().copied().collect();
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
                // Hard context limit: stop decoding this sequence without
                // consuming a row (KV store would write past the cache).
                if s.len >= self.model.max_seq_len() {
                    rows.push((tokens.len(), 0, false));
                    continue;
                }
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

        // Skip the batched forward when no row is produced this step (every
        // active sequence is at the hard context limit or over budget): an
        // empty batch would start gridDim=0 grids, which ROCm rejects
        // (hipErrorInvalidConfiguration) and would panic the serving engine.
        // The outputs loop below skips count==0 rows and the hard-stop loop
        // finishes the over-limit sequences.
        let (sampled, logprobs, topk) = if tokens.is_empty() {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            let out = self.model.decode_step_explicit(
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
            out
        };

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
                    // Paged: prefill materialized — register this request's
                    // pages under its content so later requests reuse them,
                    // using the chain computed once at admission (no rehash).
                    // (A reused request re-registers its source's pages: the
                    // builder no-ops on identical mappings.)
                    if let Some(pg) = &mut self.paged {
                        let chain = &pg.entries[i].chain;
                        let t = pg.tables[i].as_ref().expect("paged slot table");
                        pg.builder.register_chain(chain, t);
                        pg.entries[i].registered = true;
                    }
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
        // Sequences that hit the hard context limit mid-decode finish here
        // (they produced no row this step, so the outputs loop skipped them).
        for i in 0..self.active {
            let s = self.seqs[i].as_ref().expect("active slot");
            if s.prompt.is_empty() && s.len >= self.model.max_seq_len() && !done_slots.contains(&i)
            {
                done_slots.push(i);
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
        // (Paged mode reads its KV through block tables — contiguous anchor
        // snapshots do not apply; cross-request reuse is handled by the page
        // registry instead.)
        if self.paged.is_none()
            && let Some(sr) = &mut self.state_reuse
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
        // Retired-metadata cap, computed before the paged-borrow below.
        let retired_cap = self.capacity().saturating_mul(4);
        if let Some(pg) = &mut self.paged {
            // Paged compaction: KV lives in pages addressed by the block
            // tables, so moving a sequence means moving its table (the pages
            // alias follows). The registry entry retires with the pages kept
            // in the pool — later requests may still reuse them.
            // The finishing table's PAD pages (its generated-KV region) are
            // not registered content: free them on retire so long-running
            // pools never fill with orphaned pads (content pages stay for
            // reuse; eviction handles them on demand).
            if let Some(t) = pg.tables[slot].as_ref() {
                let content_pages = pg.entries[slot].chain.len();
                for &page in t.pages().iter().skip(content_pages) {
                    pg.builder.free_page(page);
                }
            }
            let empty = || PagedEntry {
                prefix: Vec::new(),
                registered: false,
                chain: Vec::new(),
            };
            let entry = std::mem::replace(&mut pg.entries[slot], empty());
            for i in (slot + 1)..self.active {
                let table = pg.tables[i].take().expect("active paged table");
                self.model
                    .set_block_table(i - 1, table.pages())
                    .expect("paged compaction table copy");
                pg.tables[i - 1] = Some(table);
                pg.entries[i - 1] = std::mem::replace(&mut pg.entries[i], empty());
                self.seqs[i - 1] = self.seqs[i].take();
            }
            self.seqs[self.active - 1] = None;
            pg.tables[self.active - 1] = None;
            pg.entries[self.active - 1] = empty();
            pg.retired.push(entry);
            // Bound the retired metadata: pages stay in the pool either way,
            // but dropping the oldest entries (via real eviction when
            // unreferenced) caps host memory growth on long-running servers.
            // Entries still aliased by live tables are kept until they free up.
            while pg.retired.len() > retired_cap {
                if !pg.evict_one_retired() {
                    break;
                }
            }
            self.active -= 1;
            return;
        }
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
