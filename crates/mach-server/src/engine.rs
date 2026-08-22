//! Background engine thread bridging HTTP handlers to the continuous-batching
//! model. The model lives only on the engine thread; handlers communicate via
//! channels, so no GPU state crosses thread boundaries.

use mach_kernel_sys::hip::Hip;
use mach_model::continuous::{ContinuousModel, SeqId};
use mach_model::sampling::SamplingParams;
use mach_model::{Config, Weights};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Condvar, Mutex};
use tokio::sync::oneshot;

/// A submitted generation request.
struct Request {
    prompt: Vec<u32>,
    max_new: usize,
    eos: Option<u32>,
    params: SamplingParams,
    done: oneshot::Sender<Vec<u32>>,
}

/// Shared engine handle (channel side only; the model stays on the engine
/// thread).
pub struct ServerEngine {
    capacity: usize,
    pending: Mutex<VecDeque<Request>>,
    cond: Condvar,
    txs: Mutex<HashMap<SeqId, oneshot::Sender<Vec<u32>>>>,
    _requests: AtomicU64,
}

/// Errors from the engine API.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("engine capacity reached")]
    Busy,
    #[error("model error: {0}")]
    Model(#[from] mach_model::Error),
}

impl ServerEngine {
    /// Creates an engine handle with `capacity` concurrent sequences.
    #[must_use]
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            pending: Mutex::new(VecDeque::new()),
            cond: Condvar::new(),
            txs: Mutex::new(HashMap::new()),
            _requests: AtomicU64::new(0),
        })
    }

    /// Submits a generation request; resolves when the sequence finishes.
    pub async fn submit(
        self: &Arc<Self>,
        prompt: Vec<u32>,
        max_new: usize,
        eos: Option<u32>,
        params: SamplingParams,
    ) -> Result<Vec<u32>, EngineError> {
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
                params,
                done: tx,
            });
        }
        self.cond.notify_one();
        rx.await.map_err(|_| EngineError::Busy)
    }

    /// Runs the engine loop until dropped; owns the model.
    pub fn spawn(
        self: Arc<Self>,
        hip: Arc<Hip>,
        cfg: Config,
        w: Weights,
    ) -> Result<std::thread::JoinHandle<()>, EngineError> {
        let mut model = ContinuousModel::new(hip, cfg, &w, self.capacity)?;
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
                while !pending.is_empty() && model.active() < self.capacity {
                    let r = pending.pop_front().expect("checked non-empty");
                    let id = model
                        .add(&r.prompt, r.max_new, r.eos, r.params)
                        .expect("capacity guaranteed");
                    txs.insert(id, r.done);
                }
            }
            if model.active() > 0 {
                let outputs = model.step().expect("engine step");
                // Deliver completed sequences.
                let mut txs = self.txs.lock().unwrap();
                for (id, _tok) in outputs {
                    if model.is_done(id) {
                        let output = model.generated(id);
                        model.ack(id);
                        if let Some(tx) = txs.remove(&id) {
                            let _ = tx.send(output);
                        }
                    }
                }
            } else {
                // Idle: wait for new work.
                let mut pending = self.pending.lock().unwrap();
                while pending.is_empty() {
                    pending = self.cond.wait(pending).unwrap();
                }
            }
        }
    }
}
