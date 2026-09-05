//! Background engine thread bridging HTTP handlers to the continuous-batching
//! model. The model lives only on the engine thread; handlers communicate via
//! channels, so no GPU state crosses thread boundaries.

use mach_kernel_sys::hip::Hip;
use mach_model::batched::BatchedModel;
use mach_model::continuous::{ContinuousModel, SeqId};
use mach_model::sampling::SamplingParams;
use mach_model::speculative::SpeculativeEngine;
use mach_model::{Config, Weights, WeightsFp8, WeightsQ4};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use tokio::sync::oneshot;

/// A submitted generation request.
struct Request {
    prompt: Vec<u32>,
    max_new: usize,
    eos: Option<u32>,
    stop_seqs: Vec<Vec<u32>>,
    logit_bias: Vec<(u32, f32)>,
    params: SamplingParams,
    done: DoneSender,
    /// Streaming: per-token channel (None for non-streaming requests).
    tokens_tx: Option<tokio::sync::mpsc::Sender<u32>>,
}

/// Completion delivery: generated tokens, per-token log-probs, per-token
/// top-`k` log-probs (OpenAI `top_logprobs`), and the OpenAI finish reason.
type DoneSender = oneshot::Sender<(Vec<u32>, Vec<f32>, Vec<Vec<(u32, f32)>>, &'static str)>;
/// Completion delivery receiver (the `submit`/`submit_stream` resolution).
type DoneReceiver = oneshot::Receiver<(Vec<u32>, Vec<f32>, Vec<Vec<(u32, f32)>>, &'static str)>;

/// Shared engine handle (channel side only; the model stays on the engine
/// thread).
pub struct ServerEngine {
    capacity: usize,
    /// Rows per prefill step (>= capacity; larger = faster long-prompt TTFT).
    prefill_rows: usize,
    /// Speculative-decoding mode (greedy-only; draft + target models).
    spec: bool,
    /// Draft tokens per verify round in spec mode.
    spec_k: usize,
    /// MoE offload (cpu backend): GPU-resident expert slots per layer; None = full.
    offload_slots: Option<usize>,
    /// Paged-KV mode: KV in a page pool with cross-request prefix reuse;
    /// `Some(tokens_per_page)` when enabled.
    paged_tpp: Option<usize>,
    pending: Mutex<VecDeque<Request>>,
    cond: Condvar,
    txs: Mutex<HashMap<SeqId, DoneSender>>,
    /// Streaming token channels per active sequence.
    streams: Mutex<HashMap<SeqId, tokio::sync::mpsc::Sender<u32>>>,
    /// Cross-request reuse stats snapshot (paged engines; updated by the
    /// engine thread after every step).
    paged_stats: Mutex<Option<mach_model::continuous::PagedReuseStats>>,
    /// Graceful-shutdown flag: the engine thread drains then exits.
    shutdown: AtomicBool,
}

/// Errors from the engine API.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("engine capacity reached")]
    Busy,
    #[error("engine is shutting down")]
    ShuttingDown,
    #[error("invalid request for spec mode: {0}")]
    InvalidRequest(String),
    #[error("model error: {0}")]
    Model(#[from] mach_model::Error),
}

impl ServerEngine {
    /// Creates an engine handle with `capacity` concurrent sequences.
    #[must_use]
    pub fn new(capacity: usize) -> Arc<Self> {
        Self::with_prefill_rows(capacity, capacity)
    }

    /// Creates an engine handle with `capacity` slots and `prefill_rows` rows
    /// per prefill step.
    #[must_use]
    pub fn with_prefill_rows(capacity: usize, prefill_rows: usize) -> Arc<Self> {
        Self::with_mode(capacity, prefill_rows.max(capacity), false, 0)
    }

    /// Creates a speculative-decoding engine (greedy-only) with `k` draft
    /// tokens per verify round.
    #[must_use]
    pub fn with_spec(capacity: usize, k: usize) -> Arc<Self> {
        Self::with_mode(capacity, capacity, true, k.max(1))
    }

    /// Creates a continuous-batching engine in MoE offload mode (cpu backend) with
    /// `expert_slots` GPU-resident expert slots per layer.
    #[must_use]
    pub fn with_offload(capacity: usize, prefill_rows: usize, expert_slots: usize) -> Arc<Self> {
        Self::with_mode_offload(
            capacity,
            prefill_rows.max(capacity),
            false,
            0,
            Some(expert_slots),
            None,
        )
    }

    fn with_mode(capacity: usize, prefill_rows: usize, spec: bool, spec_k: usize) -> Arc<Self> {
        Self::with_mode_offload(capacity, prefill_rows, spec, spec_k, None, None)
    }

    /// Creates a paged-KV engine (`tokens_per_page` page size): requests whose
    /// prompts share a prefix alias the same physical KV pages and prefill
    /// only their delta (cross-request prefix reuse).
    #[must_use]
    pub fn with_paged(capacity: usize, prefill_rows: usize, tokens_per_page: usize) -> Arc<Self> {
        Self::with_mode_offload(
            capacity,
            prefill_rows.max(capacity),
            false,
            0,
            None,
            Some(tokens_per_page),
        )
    }

    /// Cross-request prefix-reuse statistics of a paged engine (updated by
    /// the engine thread after every step), else `None`.
    #[must_use]
    pub fn paged_reuse_stats(&self) -> Option<mach_model::continuous::PagedReuseStats> {
        *self.paged_stats.lock().unwrap()
    }

    fn with_mode_offload(
        capacity: usize,
        prefill_rows: usize,
        spec: bool,
        spec_k: usize,
        offload_slots: Option<usize>,
        paged_tpp: Option<usize>,
    ) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            prefill_rows,
            spec,
            spec_k,
            offload_slots,
            paged_tpp,
            pending: Mutex::new(VecDeque::new()),
            cond: Condvar::new(),
            txs: Mutex::new(HashMap::new()),
            streams: Mutex::new(HashMap::new()),
            paged_stats: Mutex::new(None),
            shutdown: AtomicBool::new(false),
        })
    }

    /// Rejects requests the (greedy-only) spec engine cannot serve.
    fn check_spec_request(
        &self,
        stop_seqs: &[Vec<u32>],
        logit_bias: &[(u32, f32)],
        params: &SamplingParams,
    ) -> Result<(), EngineError> {
        if !self.spec {
            return Ok(());
        }
        if !stop_seqs.is_empty() || !logit_bias.is_empty() {
            return Err(EngineError::InvalidRequest(
                "stop/logit_bias unsupported in spec mode".into(),
            ));
        }
        if params.temperature != 0.0
            || params.top_k != 0
            || params.top_p != 1.0
            || params.presence_penalty != 0.0
            || params.frequency_penalty != 0.0
        {
            return Err(EngineError::InvalidRequest(
                "spec mode is greedy-only".into(),
            ));
        }
        Ok(())
    }

    /// Submits a generation request; resolves when the sequence finishes.
    pub async fn submit(
        self: &Arc<Self>,
        prompt: Vec<u32>,
        max_new: usize,
        eos: Option<u32>,
        stop_seqs: Vec<Vec<u32>>,
        logit_bias: Vec<(u32, f32)>,
        params: SamplingParams,
    ) -> Result<(Vec<u32>, Vec<f32>, Vec<Vec<(u32, f32)>>, &'static str), EngineError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(EngineError::ShuttingDown);
        }
        self.check_spec_request(&stop_seqs, &logit_bias, &params)?;
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock().unwrap();
            if pending.len() >= self.capacity * 2 {
                return Err(EngineError::Busy);
            }
            pending.push_back(Request {
                prompt,
                max_new,
                eos,
                stop_seqs,
                logit_bias,
                params,
                done: tx,
                tokens_tx: None,
            });
        }
        self.cond.notify_one();
        rx.await.map_err(|_| EngineError::Busy)
    }

    /// Submits a streaming generation request. The returned `Receiver<u32>`
    /// yields one token per generated step; the oneshot resolves with the full
    /// output when the sequence finishes (stream closes at the same time).
    pub async fn submit_stream(
        self: &Arc<Self>,
        prompt: Vec<u32>,
        max_new: usize,
        eos: Option<u32>,
        stop_seqs: Vec<Vec<u32>>,
        logit_bias: Vec<(u32, f32)>,
        params: SamplingParams,
    ) -> Result<(DoneReceiver, tokio::sync::mpsc::Receiver<u32>), EngineError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(EngineError::ShuttingDown);
        }
        self.check_spec_request(&stop_seqs, &logit_bias, &params)?;
        let (tx, rx) = oneshot::channel();
        let (tokens_tx, tokens_rx) = tokio::sync::mpsc::channel(256);
        {
            let mut pending = self.pending.lock().unwrap();
            if pending.len() >= self.capacity * 2 {
                return Err(EngineError::Busy);
            }
            pending.push_back(Request {
                prompt,
                max_new,
                eos,
                stop_seqs,
                logit_bias,
                params,
                done: tx,
                tokens_tx: Some(tokens_tx),
            });
        }
        self.cond.notify_one();
        Ok((rx, tokens_rx))
    }

    /// Requests graceful shutdown: the engine thread drains queued + active
    /// sequences, then exits. New submissions fail with `ShuttingDown`.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.cond.notify_all();
    }

    /// Runs the engine loop until dropped; owns the model.
    pub fn spawn(
        self: Arc<Self>,
        hip: Arc<Hip>,
        cfg: Config,
        w: Weights,
    ) -> Result<std::thread::JoinHandle<()>, EngineError> {
        let mut model = if let Some(tpp) = self.paged_tpp {
            ContinuousModel::with_paged_prefill_rows(
                hip,
                cfg,
                &w,
                self.capacity,
                self.prefill_rows,
                tpp,
            )?
        } else if let Some(slots) = self.offload_slots {
            ContinuousModel::with_prefill_rows_offload(
                hip,
                cfg,
                &w,
                self.capacity,
                self.prefill_rows,
                slots,
            )?
        } else {
            ContinuousModel::with_prefill_rows(hip, cfg, &w, self.capacity, self.prefill_rows)?
        };
        Ok(std::thread::Builder::new()
            .name("mach-engine".into())
            .spawn(move || self.run(&mut model))
            .expect("spawn engine thread"))
    }

    /// Spawns a storage-Q4 engine thread: weights are dequantized to f16 per
    /// tensor during upload, so host RAM stays ~= the packed Q4 weights.
    /// Experts stay fully GPU-resident (the cpu-backend offload path needs f32
    /// `Weights`, which Q4 host layout does not provide).
    pub fn spawn_q4(
        self: Arc<Self>,
        hip: Arc<Hip>,
        cfg: Config,
        w: WeightsQ4,
    ) -> Result<std::thread::JoinHandle<()>, EngineError> {
        if self.offload_slots.is_some() {
            return Err(EngineError::InvalidRequest(
                "Q4 mode does not support MACH_MOE_SLOTS (cpu-backend offload needs f32 Weights)"
                    .into(),
            ));
        }
        let mut model = if let Some(tpp) = self.paged_tpp {
            ContinuousModel::with_paged_prefill_rows_q4(
                hip,
                cfg,
                &w,
                self.capacity,
                self.prefill_rows,
                tpp,
            )?
        } else {
            ContinuousModel::with_prefill_rows_q4(hip, cfg, &w, self.capacity, self.prefill_rows)?
        };
        Ok(std::thread::Builder::new()
            .name("mach-engine".into())
            .spawn(move || self.run(&mut model))
            .expect("spawn engine thread"))
    }

    /// Spawns a storage-Q4 engine with the expert pool kept in Q4 ON DEVICE
    /// (`MACH_Q4_DEVICE=1`; in-kernel dequant in the grouped GEMV kernels) —
    /// the memory path for 30B-class checkpoints whose dequantized f16
    /// expert pool would not fit in VRAM.
    pub fn spawn_q4_device(
        self: Arc<Self>,
        hip: Arc<Hip>,
        cfg: Config,
        w: WeightsQ4,
    ) -> Result<std::thread::JoinHandle<()>, EngineError> {
        if self.offload_slots.is_some() {
            return Err(EngineError::InvalidRequest(
                "Q4 mode does not support MACH_MOE_SLOTS (cpu-backend offload needs f32 Weights)"
                    .into(),
            ));
        }
        let mut model = if let Some(tpp) = self.paged_tpp {
            ContinuousModel::with_paged_prefill_rows_q4_device(
                hip,
                cfg,
                &w,
                self.capacity,
                self.prefill_rows,
                tpp,
            )?
        } else {
            ContinuousModel::with_prefill_rows_q4_device(
                hip,
                cfg,
                &w,
                self.capacity,
                self.prefill_rows,
            )?
        };
        Ok(std::thread::Builder::new()
            .name("mach-engine".into())
            .spawn(move || self.run(&mut model))
            .expect("spawn engine thread"))
    }

    /// Spawns a dense Q4-on-device engine (`MACH_Q4_DEVICE=2`): the expert
    /// pool AND every big dense GEMM tensor stay raw Q4 on device
    /// (in-kernel dequant) — the memory path for DENSE 27B-class checkpoints
    /// whose dequantized f16 weights would not fit in VRAM.
    pub fn spawn_q4_all(
        self: Arc<Self>,
        hip: Arc<Hip>,
        cfg: Config,
        w: WeightsQ4,
    ) -> Result<std::thread::JoinHandle<()>, EngineError> {
        if self.offload_slots.is_some() {
            return Err(EngineError::InvalidRequest(
                "Q4 mode does not support MACH_MOE_SLOTS (cpu-backend offload needs f32 Weights)"
                    .into(),
            ));
        }
        let mut model = if let Some(tpp) = self.paged_tpp {
            ContinuousModel::with_paged_prefill_rows_q4_all(
                hip,
                cfg,
                &w,
                self.capacity,
                self.prefill_rows,
                tpp,
            )?
        } else {
            ContinuousModel::with_prefill_rows_q4_all(
                hip,
                cfg,
                &w,
                self.capacity,
                self.prefill_rows,
            )?
        };
        Ok(std::thread::Builder::new()
            .name("mach-engine".into())
            .spawn(move || self.run(&mut model))
            .expect("spawn engine thread"))
    }

    /// Spawns a storage-FP8 engine thread: weights are dequantized to f16 per
    /// tensor during upload, so host RAM stays ~= the packed FP8 weights.
    /// Experts stay fully GPU-resident (the cpu-backend offload path needs f32
    /// `Weights`, which FP8 host layout does not provide).
    pub fn spawn_fp8(
        self: Arc<Self>,
        hip: Arc<Hip>,
        cfg: Config,
        w: WeightsFp8,
    ) -> Result<std::thread::JoinHandle<()>, EngineError> {
        if self.offload_slots.is_some() {
            return Err(EngineError::InvalidRequest(
                "FP8 mode does not support MACH_MOE_SLOTS (cpu-backend offload needs f32 Weights)"
                    .into(),
            ));
        }
        let mut model = if let Some(tpp) = self.paged_tpp {
            ContinuousModel::with_paged_prefill_rows_fp8(
                hip,
                cfg,
                &w,
                self.capacity,
                self.prefill_rows,
                tpp,
            )?
        } else {
            ContinuousModel::with_prefill_rows_fp8(hip, cfg, &w, self.capacity, self.prefill_rows)?
        };
        Ok(std::thread::Builder::new()
            .name("mach-engine".into())
            .spawn(move || self.run(&mut model))
            .expect("spawn engine thread"))
    }

    /// Spawns a speculative-decoding engine thread (greedy-only).
    pub fn spawn_spec(
        self: Arc<Self>,
        hip: Arc<Hip>,
        cfg: Config,
        w: Weights,
        dcfg: Config,
        dw: Weights,
    ) -> Result<std::thread::JoinHandle<()>, EngineError> {
        if self.paged_tpp.is_some() {
            return Err(EngineError::InvalidRequest(
                "Speculative mode does not support MACH_PAGED (paged spec wiring is a follow-up)"
                    .into(),
            ));
        }
        let k = self.spec_k;
        let draft = BatchedModel::with_rows(hip.clone(), dcfg, &dw, self.capacity, self.capacity)?;
        let target = BatchedModel::with_rows(hip, cfg, &w, self.capacity, self.capacity * (k + 1))?;
        let mut engine = SpeculativeEngine::new(draft, target, k, self.capacity);
        Ok(std::thread::Builder::new()
            .name("mach-engine".into())
            .spawn(move || self.run_spec(&mut engine))
            .expect("spawn engine thread"))
    }

    fn run(self: &Arc<Self>, model: &mut ContinuousModel) {
        loop {
            // Admit pending requests while capacity allows (continuous batching).
            {
                let mut pending = self.pending.lock().unwrap();
                let mut txs = self.txs.lock().unwrap();
                let mut streams = self.streams.lock().unwrap();
                while !pending.is_empty() && model.active() < self.capacity {
                    let r = pending.pop_front().expect("checked non-empty");
                    match model.add(
                        &r.prompt,
                        r.max_new,
                        r.eos,
                        r.stop_seqs,
                        r.logit_bias,
                        r.params,
                    ) {
                        Ok(id) => {
                            txs.insert(id, r.done);
                            if let Some(stx) = r.tokens_tx {
                                streams.insert(id, stx);
                            }
                        }
                        // Admission failure (e.g. paged page-pool exhaustion):
                        // reject the request instead of panicking the engine
                        // thread while holding the pending/txs locks. The
                        // oneshot delivers an empty completion; the caller
                        // sees an empty generation rather than a hang.
                        Err(e) => {
                            eprintln!("engine: rejecting request: {e}");
                            let _ = r.done.send((Vec::new(), Vec::new(), Vec::new(), "error"));
                            drop(r.tokens_tx);
                        }
                    }
                }
                drop(streams);
            }
            if model.active() > 0 {
                let outputs = model.step().expect("engine step");
                if self.paged_tpp.is_some() {
                    *self.paged_stats.lock().unwrap() = model.paged_reuse_stats();
                }
                // Deliver completed sequences.
                let mut txs = self.txs.lock().unwrap();
                let mut streams = self.streams.lock().unwrap();
                for (id, tok) in outputs {
                    if model.is_done(id) {
                        // Stream the final generated token before closing.
                        if let Some(stx) = streams.get(&id) {
                            let _ = stx.try_send(tok);
                        }
                        let output = model.generated(id);
                        let lps = model.generated_logprobs(id);
                        let tlps = model.generated_top_logprobs(id);
                        let reason = model.finish_reason(id);
                        model.ack(id);
                        if let Some(tx) = txs.remove(&id) {
                            let _ = tx.send((output, lps, tlps, reason));
                        }
                        // Closing the stream sender signals end-of-stream.
                        streams.remove(&id);
                    } else if let Some(stx) = streams.get(&id) {
                        // `step()` only returns real tokens (first generated /
                        // decode), so every non-done output is streamed.
                        // Best-effort: drop tokens for a slow/closed client
                        // rather than stalling the whole engine loop.
                        let _ = stx.try_send(tok);
                    }
                }
            } else {
                // Idle: wait for new work, or exit once shutting down.
                let mut pending = self.pending.lock().unwrap();
                if pending.is_empty() {
                    if self.shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    while pending.is_empty() && !self.shutdown.load(Ordering::Acquire) {
                        pending = self.cond.wait(pending).unwrap();
                    }
                }
            }
        }
    }

    /// Speculative-decoding engine loop (greedy-only; draft + target models).
    fn run_spec(self: &Arc<Self>, engine: &mut SpeculativeEngine) {
        loop {
            // Admit pending requests while capacity allows.
            {
                let mut pending = self.pending.lock().unwrap();
                let mut txs = self.txs.lock().unwrap();
                let mut streams = self.streams.lock().unwrap();
                while !pending.is_empty() && engine.active() < self.capacity {
                    let r = pending.pop_front().expect("checked non-empty");
                    let id = engine
                        .add(&r.prompt, r.max_new, r.eos)
                        .expect("capacity guaranteed") as SeqId;
                    txs.insert(id, r.done);
                    if let Some(stx) = r.tokens_tx {
                        streams.insert(id, stx);
                    }
                }
                drop(streams);
            }
            if engine.active() > 0 {
                let outputs = engine.step().expect("spec step");
                let mut txs = self.txs.lock().unwrap();
                let mut streams = self.streams.lock().unwrap();
                for (id, tok) in outputs {
                    let id = id as SeqId;
                    if engine.is_done(id as usize) {
                        if let Some(stx) = streams.get(&id) {
                            let _ = stx.try_send(tok);
                        }
                        let output = engine.generated(id as usize);
                        let reason = engine.finish_reason(id as usize);
                        if let Some(tx) = txs.remove(&id) {
                            // Spec mode is greedy-only: no logprobs tracked.
                            let _ = tx.send((output, Vec::new(), Vec::new(), reason));
                        }
                        streams.remove(&id);
                    } else if let Some(stx) = streams.get(&id) {
                        let _ = stx.try_send(tok);
                    }
                }
            } else {
                let mut pending = self.pending.lock().unwrap();
                if pending.is_empty() {
                    if self.shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    while pending.is_empty() && !self.shutdown.load(Ordering::Acquire) {
                        pending = self.cond.wait(pending).unwrap();
                    }
                }
            }
        }
    }
}
