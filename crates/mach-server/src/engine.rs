//! Background engine thread bridging HTTP handlers to the continuous-batching
//! model. The model lives only on the engine thread; handlers communicate via
//! channels, so no GPU state crosses thread boundaries.

use mach_kernel_sys::hip::Hip;
use mach_model::continuous::{ContinuousModel, SeqId};
use mach_model::sampling::SamplingParams;
use mach_model::{Config, Weights};
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
    pending: Mutex<VecDeque<Request>>,
    cond: Condvar,
    txs: Mutex<HashMap<SeqId, DoneSender>>,
    /// Streaming token channels per active sequence.
    streams: Mutex<HashMap<SeqId, tokio::sync::mpsc::Sender<u32>>>,
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
        Arc::new(Self {
            capacity,
            prefill_rows: prefill_rows.max(capacity),
            pending: Mutex::new(VecDeque::new()),
            cond: Condvar::new(),
            txs: Mutex::new(HashMap::new()),
            streams: Mutex::new(HashMap::new()),
            shutdown: AtomicBool::new(false),
        })
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
        let mut model =
            ContinuousModel::with_prefill_rows(hip, cfg, &w, self.capacity, self.prefill_rows)?;
        Ok(std::thread::Builder::new()
            .name("mach-engine".into())
            .spawn(move || self.run(&mut model))
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
                    let id = model
                        .add(
                            &r.prompt,
                            r.max_new,
                            r.eos,
                            r.stop_seqs,
                            r.logit_bias,
                            r.params,
                        )
                        .expect("capacity guaranteed");
                    txs.insert(id, r.done);
                    if let Some(stx) = r.tokens_tx {
                        streams.insert(id, stx);
                    }
                }
                drop(streams);
            }
            if model.active() > 0 {
                let outputs = model.step().expect("engine step");
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
}
