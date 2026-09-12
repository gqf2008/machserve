//! Background engine thread bridging HTTP handlers to the continuous-batching
//! model. The model lives only on the engine thread; handlers communicate via
//! channels, so no GPU state crosses thread boundaries.

use mach_kernel_sys::hip::Hip;
use mach_model::batched::BatchedModel;
use mach_model::continuous::{ContinuousModel, SeqId};
use mach_model::image_processor::ProcessedImage;
use mach_model::multimodal::{MultimodalPrompt, VisionImage};
use mach_model::sampling::SamplingParams;
use mach_model::speculative::SpeculativeEngine;
use mach_model::vision::{VisionConfig, VisionGrid, VisionWeights};
use mach_model::vision_gpu::VisionGpu;
use mach_model::{Config, Weights, WeightsFp8, WeightsQ4};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use tokio::sync::oneshot;

/// Write the merged vision features (raw little-endian f32) plus metadata
/// for the C4 comparator (MACH_VISION_DUMP=<prefix>).
fn dump_vision_features(
    path: &std::path::Path,
    features: &[f32],
    grids: &[VisionGrid],
) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(features.len().saturating_mul(4));
    for value in features {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path.with_extension("bin"), bytes)?;
    let meta = serde_json::json!({
        "grids": grids,
        "values": features.len(),
        "sum": features.iter().map(|v| *v as f64).sum::<f64>(),
    });
    std::fs::write(path.with_extension("json"), meta.to_string())
}

/// A submitted generation request.
struct Request {
    prompt: Vec<u32>,
    max_new: usize,
    eos: Option<u32>,
    stop_seqs: Vec<Vec<u32>>,
    logit_bias: Vec<(u32, f32)>,
    params: SamplingParams,
    /// Preprocessed images (empty for text-only requests).
    images: Vec<ProcessedImage>,
    done: DoneSender,
    /// Streaming: per-token channel (None for non-streaming requests).
    tokens_tx: Option<tokio::sync::mpsc::Sender<u32>>,
}

/// Vision tower configuration and weights, consumed by the engine thread.
pub struct VisionSetup {
    pub cfg: VisionConfig,
    pub weights: VisionWeights,
    /// Maximum packed vision patches the GPU tower scratch supports.
    pub max_tokens: usize,
}

/// Image preprocessing/token config exposed to the HTTP handler.
#[derive(Debug, Clone)]
pub struct ImageRuntimeConfig {
    pub processor: mach_model::image_processor::ImageProcessorConfig,
    pub image_token_id: u32,
    pub spatial_merge_size: usize,
    /// Maximum total vision patches accepted per request.
    pub max_patches: usize,
    /// `MACH_VISION_DOWNSCALE=1`: an image whose grid exceeds the remaining
    /// patch budget is downscaled into it instead of rejected (400). The
    /// default keeps the strict rejection contract.
    pub downscale_oversized: bool,
}

/// GPU vision runtime owned by (and created on) the engine thread.
struct VisionRuntime {
    cfg: VisionConfig,
    weights: VisionWeights,
    gpu: VisionGpu,
    image_token_id: u32,
    max_tokens: usize,
    /// Dev-only feature dump prefix (MACH_VISION_DUMP), for C4 parity checks.
    dump_path: Option<std::path::PathBuf>,
}

impl VisionRuntime {
    fn new(hip: &Arc<Hip>, setup: VisionSetup) -> Result<Self, EngineError> {
        let gpu = VisionGpu::new(Arc::clone(hip), setup.cfg, &setup.weights, setup.max_tokens)?;
        Ok(Self {
            image_token_id: setup.cfg.image_token_id,
            cfg: setup.cfg,
            weights: setup.weights,
            gpu,
            max_tokens: setup.max_tokens,
            dump_path: std::env::var_os("MACH_VISION_DUMP").map(std::path::PathBuf::from),
        })
    }

    /// Run the tower over `images` and build the text-side prompt overrides.
    fn encode(
        &mut self,
        text_cfg: &Config,
        prompt: &[u32],
        images: &[ProcessedImage],
    ) -> Result<MultimodalPrompt, EngineError> {
        if images.is_empty() {
            return Err(EngineError::InvalidRequest("no images to encode".into()));
        }
        let grids: Vec<VisionGrid> = images.iter().map(|i| i.grid).collect();
        let mut pixel_values = Vec::new();
        let mut patches = 0usize;
        for image in images {
            pixel_values.extend_from_slice(&image.pixel_values);
            let grid_patches = image.grid[0]
                .checked_mul(image.grid[1])
                .and_then(|v| v.checked_mul(image.grid[2]))
                .ok_or_else(|| EngineError::InvalidRequest("vision grid overflow".into()))?;
            patches = patches
                .checked_add(grid_patches)
                .ok_or_else(|| EngineError::InvalidRequest("vision patch count overflow".into()))?;
        }
        if patches > self.max_tokens {
            return Err(EngineError::InvalidRequest(format!(
                "vision patches {patches} exceed cap {}",
                self.max_tokens
            )));
        }
        let prep = VisionGpu::prepare(&self.cfg, &self.weights, &grids)?;
        let features = self.gpu.forward(&pixel_values, &prep)?;
        if let Some(path) = &self.dump_path
            && let Err(e) = dump_vision_features(path, &features, &grids)
        {
            eprintln!("engine: vision dump failed: {e}");
        }

        let merge_unit = self
            .cfg
            .spatial_merge_size
            .checked_mul(self.cfg.spatial_merge_size)
            .ok_or_else(|| EngineError::InvalidRequest("merge size overflow".into()))?;
        let hidden = self.cfg.out_hidden_size;
        let mut offset = 0usize;
        let mut vision_images = Vec::with_capacity(images.len());
        for image in images {
            let grid_patches = image.grid[0]
                .checked_mul(image.grid[1])
                .and_then(|v| v.checked_mul(image.grid[2]))
                .ok_or_else(|| EngineError::InvalidRequest("vision grid overflow".into()))?;
            let rows = grid_patches / merge_unit;
            let end = rows
                .checked_mul(hidden)
                .and_then(|v| offset.checked_add(v))
                .ok_or_else(|| {
                    EngineError::InvalidRequest("vision feature offset overflow".into())
                })?;
            if end > features.len() {
                return Err(EngineError::InvalidRequest(
                    "vision feature buffer shorter than expected".into(),
                ));
            }
            vision_images.push(VisionImage {
                features: features[offset..end].to_vec(),
                grid: image.grid,
            });
            offset = end;
        }
        Ok(MultimodalPrompt::build(
            prompt,
            &vision_images,
            self.image_token_id,
            text_cfg,
            &self.cfg,
        )?)
    }
}

/// Completion delivery: generated tokens, per-token log-probs, per-token
/// top-`k` log-probs (OpenAI `top_logprobs`), and the OpenAI finish reason.
pub(crate) type DonePayload = (Vec<u32>, Vec<f32>, Vec<Vec<(u32, f32)>>, &'static str);
/// Completion delivery: `Err` carries the engine error for the caller.
type DoneSender = oneshot::Sender<Result<DonePayload, EngineError>>;
pub(crate) type DoneReceiver = oneshot::Receiver<Result<DonePayload, EngineError>>;

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
    /// Optional vision tower (consumed by the engine thread at spawn).
    vision: Mutex<Option<(Arc<Hip>, VisionSetup)>>,
    /// Image preprocessing/token config for the HTTP handler.
    image: Mutex<Option<ImageRuntimeConfig>>,
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
    #[error("engine startup failed: {0}")]
    Startup(String),
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

    /// Enables the multimodal vision tower; call before `spawn*`.
    pub fn set_vision(&self, hip: Arc<Hip>, setup: VisionSetup) {
        *self.vision.lock().unwrap() = Some((hip, setup));
    }

    /// Installs the image preprocessing/token config used by the handler.
    pub fn set_image_runtime(&self, cfg: ImageRuntimeConfig) {
        *self.image.lock().unwrap() = Some(cfg);
    }

    /// Image config when multimodal serving is enabled.
    #[must_use]
    pub fn image_runtime(&self) -> Option<ImageRuntimeConfig> {
        self.image.lock().unwrap().clone()
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
            vision: Mutex::new(None),
            image: Mutex::new(None),
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

    /// Submits a text-only generation request; resolves when the sequence finishes.
    pub async fn submit(
        self: &Arc<Self>,
        prompt: Vec<u32>,
        max_new: usize,
        eos: Option<u32>,
        stop_seqs: Vec<Vec<u32>>,
        logit_bias: Vec<(u32, f32)>,
        params: SamplingParams,
    ) -> Result<(Vec<u32>, Vec<f32>, Vec<Vec<(u32, f32)>>, &'static str), EngineError> {
        self.submit_multimodal(
            prompt,
            Vec::new(),
            max_new,
            eos,
            stop_seqs,
            logit_bias,
            params,
        )
        .await
    }

    /// Submits a generation request with preprocessed images (empty = text-only).
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_multimodal(
        self: &Arc<Self>,
        prompt: Vec<u32>,
        images: Vec<ProcessedImage>,
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
            // Re-check under the queue lock: a fatal engine failure may have
            // set shutdown (and drained the queue) after the early check, and
            // enqueueing now would leave the caller waiting forever.
            if self.shutdown.load(Ordering::Acquire) {
                return Err(EngineError::ShuttingDown);
            }
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
                images,
                done: tx,
                tokens_tx: None,
            });
        }
        self.cond.notify_one();
        rx.await.map_err(|_| EngineError::Busy)?
    }

    /// Submits a streaming generation request. The returned `Receiver<u32>`
    /// yields one token per generated step; the oneshot resolves with the full
    /// Submits a streaming text-only generation request.
    pub async fn submit_stream(
        self: &Arc<Self>,
        prompt: Vec<u32>,
        max_new: usize,
        eos: Option<u32>,
        stop_seqs: Vec<Vec<u32>>,
        logit_bias: Vec<(u32, f32)>,
        params: SamplingParams,
    ) -> Result<(DoneReceiver, tokio::sync::mpsc::Receiver<u32>), EngineError> {
        self.submit_stream_multimodal(
            prompt,
            Vec::new(),
            max_new,
            eos,
            stop_seqs,
            logit_bias,
            params,
        )
        .await
    }

    /// Submits a streaming request with preprocessed images (empty = text-only).
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_stream_multimodal(
        self: &Arc<Self>,
        prompt: Vec<u32>,
        images: Vec<ProcessedImage>,
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
            if self.shutdown.load(Ordering::Acquire) {
                return Err(EngineError::ShuttingDown);
            }
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
                images,
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
        let model = if let Some(tpp) = self.paged_tpp {
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
        self.spawn_engine_thread(model)
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
        let model = if let Some(tpp) = self.paged_tpp {
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
        self.spawn_engine_thread(model)
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
        let model = if let Some(tpp) = self.paged_tpp {
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
        self.spawn_engine_thread(model)
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
        int8_kv: bool,
    ) -> Result<std::thread::JoinHandle<()>, EngineError> {
        if self.offload_slots.is_some() {
            return Err(EngineError::InvalidRequest(
                "Q4 mode does not support MACH_MOE_SLOTS (cpu-backend offload needs f32 Weights)"
                    .into(),
            ));
        }
        let model = if let Some(tpp) = self.paged_tpp {
            if int8_kv {
                ContinuousModel::with_paged_prefill_rows_q4_all_int8_kv(
                    hip,
                    cfg,
                    &w,
                    self.capacity,
                    self.prefill_rows,
                    tpp,
                )?
            } else {
                ContinuousModel::with_paged_prefill_rows_q4_all(
                    hip,
                    cfg,
                    &w,
                    self.capacity,
                    self.prefill_rows,
                    tpp,
                )?
            }
        } else if int8_kv {
            ContinuousModel::with_prefill_rows_q4_all_int8_kv(
                hip,
                cfg,
                &w,
                self.capacity,
                self.prefill_rows,
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
        self.spawn_engine_thread(model)
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
        let model = if let Some(tpp) = self.paged_tpp {
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
        self.spawn_engine_thread(model)
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

    /// Spawns the engine thread; builds the GPU vision runtime inside it when
    /// a setup was installed via [`Self::set_vision`].
    fn spawn_engine_thread(
        self: &Arc<Self>,
        mut model: ContinuousModel,
    ) -> Result<std::thread::JoinHandle<()>, EngineError> {
        let engine = Arc::clone(self);
        let setup = self.vision.lock().unwrap().take();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("mach-engine".into())
            .spawn(move || {
                let mut vision = match setup {
                    Some((hip, setup)) => match VisionRuntime::new(&hip, setup) {
                        Ok(runtime) => {
                            model.set_mrope_section(runtime.cfg.mrope_section);
                            Some(runtime)
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    },
                    None => None,
                };
                let _ = ready_tx.send(Ok(()));
                engine.run(&mut model, vision.as_mut());
            })
            .expect("spawn engine thread");
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(handle),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(EngineError::Startup(
                "engine thread exited before signalling readiness".into(),
            )),
        }
    }

    fn run(self: &Arc<Self>, model: &mut ContinuousModel, mut vision: Option<&mut VisionRuntime>) {
        loop {
            // Admit pending requests while capacity allows (continuous
            // batching). Each request is popped under the lock, then vision
            // encoding / admission runs without holding it: a multi-second
            // vision forward must not block HTTP submissions.
            loop {
                let r = {
                    let mut pending = self.pending.lock().unwrap();
                    if pending.is_empty() || model.active() >= self.capacity {
                        None
                    } else {
                        pending.pop_front()
                    }
                };
                let Some(r) = r else {
                    break;
                };
                let admitted: Result<SeqId, EngineError> = if r.images.is_empty() {
                    model
                        .add(
                            &r.prompt,
                            r.max_new,
                            r.eos,
                            r.stop_seqs,
                            r.logit_bias,
                            r.params,
                        )
                        .map_err(EngineError::from)
                } else {
                    match vision.as_mut() {
                        Some(vision) => match vision.encode(model.config(), &r.prompt, &r.images) {
                            Ok(multimodal) => model
                                .add_multimodal(
                                    &r.prompt,
                                    multimodal,
                                    r.max_new,
                                    r.eos,
                                    r.stop_seqs,
                                    r.logit_bias,
                                    r.params,
                                )
                                .map_err(EngineError::from),
                            Err(e) => Err(e),
                        },
                        None => Err(EngineError::InvalidRequest(
                            "vision runtime is not enabled".into(),
                        )),
                    }
                };
                match admitted {
                    Ok(id) => {
                        self.txs.lock().unwrap().insert(id, r.done);
                        if let Some(stx) = r.tokens_tx {
                            self.streams.lock().unwrap().insert(id, stx);
                        }
                    }
                    // Admission failure (e.g. paged page-pool exhaustion):
                    // reject the request instead of panicking the engine
                    // thread. The oneshot carries the engine error; the caller
                    // sees a structured failure rather than a hang.
                    Err(e) => {
                        eprintln!("engine: rejecting request: {e}");
                        let _ = r.done.send(Err(e));
                        drop(r.tokens_tx);
                    }
                }
            }
            if model.active() > 0 {
                let outputs = match model.step() {
                    Ok(outputs) => outputs,
                    Err(e) => {
                        let msg = format!("engine step failed: {e}");
                        eprintln!("{msg}; stopping engine");
                        self.fail_all(&msg);
                        self.shutdown.store(true, Ordering::Release);
                        break;
                    }
                };
                if self.paged_tpp.is_some() {
                    *self.paged_stats.lock().unwrap() = model.paged_reuse_stats();
                }
                // Deliver completed sequences.
                let mut txs = self.txs.lock().unwrap();
                let mut streams = self.streams.lock().unwrap();
                let (complete, cancel) =
                    deliver_step_tokens(outputs, &mut streams, &mut txs, |id| model.is_done(id));
                for id in cancel {
                    model.cancel(id);
                    model.ack(id);
                }
                for id in complete {
                    let output = model.generated(id);
                    let lps = model.generated_logprobs(id);
                    let tlps = model.generated_top_logprobs(id);
                    let reason = model.finish_reason(id);
                    model.ack(id);
                    if let Some(tx) = txs.remove(&id) {
                        let _ = tx.send(Ok((output, lps, tlps, reason)));
                    }
                    // Closing the stream sender signals end-of-stream.
                    streams.remove(&id);
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

    /// Fails every pending and in-flight request after a fatal engine error.
    /// The engine stops afterwards, so callers get a structured error instead
    /// of waiting for a completion that can never arrive.
    fn fail_all(&self, message: &str) {
        // Stop new submissions *before* draining: a submit that already passed
        // its early check re-checks shutdown under the pending lock.
        self.shutdown.store(true, Ordering::Release);
        let mut pending = self.pending.lock().unwrap();
        while let Some(r) = pending.pop_front() {
            let _ = r.done.send(Err(EngineError::Model(mach_model::Error::Model(
                message.to_string(),
            ))));
            drop(r.tokens_tx);
        }
        drop(pending);
        let mut txs = self.txs.lock().unwrap();
        for (_, tx) in txs.drain() {
            let _ = tx.send(Err(EngineError::Model(mach_model::Error::Model(
                message.to_string(),
            ))));
        }
        drop(txs);
        self.streams.lock().unwrap().clear();
        self.cond.notify_all();
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
                let outputs = match engine.step() {
                    Ok(outputs) => outputs,
                    Err(e) => {
                        let msg = format!("spec step failed: {e}");
                        eprintln!("{msg}; stopping engine");
                        self.fail_all(&msg);
                        self.shutdown.store(true, Ordering::Release);
                        break;
                    }
                };
                let mut txs = self.txs.lock().unwrap();
                let mut streams = self.streams.lock().unwrap();
                // A speculative round can emit several tokens for one
                // sequence and mark it finished in the same call, so the
                // completion must wait until the whole group is queued.
                let outputs: Vec<(SeqId, u32)> = outputs
                    .into_iter()
                    .map(|(id, t)| (id as SeqId, t))
                    .collect();
                let (complete, cancel) =
                    deliver_step_tokens(outputs, &mut streams, &mut txs, |id| {
                        engine.is_done(id as usize)
                    });
                for id in cancel {
                    engine.cancel(id as usize);
                }
                for id in complete {
                    let output = engine.generated(id as usize);
                    let reason = engine.finish_reason(id as usize);
                    if let Some(tx) = txs.remove(&id) {
                        // Spec mode is greedy-only: no logprobs tracked.
                        let _ = tx.send(Ok((output, Vec::new(), Vec::new(), reason)));
                    }
                    streams.remove(&id);
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

/// Outcome of handing one step's token to a streaming client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenDelivery {
    /// The token was queued, or the request is not streamed.
    Ok,
    /// The consumer stalled: the token was dropped, the completion has been
    /// failed with a structured error and both channels were removed.
    Stalled,
}

/// Pushes `tok` to the streaming client of `id`, if the request streams.
///
/// A full channel means the consumer stopped draining. Silently dropping the
/// token would truncate the response — and for the final token it would report
/// a clean finish for an incomplete body — so the completion is failed here
/// and the caller only has to cancel the underlying sequence.
fn push_token(
    streams: &mut HashMap<SeqId, tokio::sync::mpsc::Sender<u32>>,
    txs: &mut HashMap<SeqId, DoneSender>,
    id: SeqId,
    tok: u32,
) -> TokenDelivery {
    let stalled = match streams.get(&id) {
        Some(stx) => stx.try_send(tok).is_err(),
        None => false,
    };
    if stalled {
        if let Some(tx) = txs.remove(&id) {
            let _ = tx.send(Err(EngineError::InvalidRequest(
                "stream consumer stalled: token dropped".into(),
            )));
        }
        streams.remove(&id);
        return TokenDelivery::Stalled;
    }
    TokenDelivery::Ok
}

/// Outcome of delivering one step's outputs, grouped per sequence.
type StepDelivery = (Vec<SeqId>, Vec<SeqId>);

/// Delivers one step's outputs to the streaming clients, grouped per
/// sequence.
///
/// A speculative round can emit several tokens for one sequence and mark it
/// finished in the same call, so the completion must not be sent until every
/// token of that round has been queued — otherwise the trailing tokens are
/// silently dropped while the client is told the response finished cleanly.
///
/// Returns `(to_complete, to_cancel)`: ids whose completion payload the caller
/// must send (still present in `txs`/`streams`) and ids whose stalled consumer
/// was failed here and whose sequence the caller must cancel.
fn deliver_step_tokens(
    outputs: Vec<(SeqId, u32)>,
    streams: &mut HashMap<SeqId, tokio::sync::mpsc::Sender<u32>>,
    txs: &mut HashMap<SeqId, DoneSender>,
    is_done: impl Fn(SeqId) -> bool,
) -> StepDelivery {
    let mut groups: Vec<(SeqId, Vec<u32>)> = Vec::new();
    for (id, tok) in outputs {
        match groups.last_mut() {
            Some((gid, toks)) if *gid == id => toks.push(tok),
            _ => groups.push((id, vec![tok])),
        }
    }
    let mut to_complete = Vec::new();
    let mut to_cancel = Vec::new();
    for (id, toks) in groups {
        let done = is_done(id);
        let mut stalled = false;
        for tok in toks {
            if push_token(streams, txs, id, tok) == TokenDelivery::Stalled {
                stalled = true;
                break;
            }
        }
        if stalled {
            to_cancel.push(id);
        } else if done {
            to_complete.push(id);
        }
    }
    (to_complete, to_cancel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_runtime_round_trip() {
        let engine = ServerEngine::new(1);
        assert!(engine.image_runtime().is_none());
        let cfg = ImageRuntimeConfig {
            processor: mach_model::image_processor::ImageProcessorConfig::default(),
            image_token_id: 7,
            spatial_merge_size: 2,
            max_patches: 128,
            downscale_oversized: true,
        };
        engine.set_image_runtime(cfg);
        let got = engine.image_runtime().unwrap();
        assert_eq!(got.image_token_id, 7);
        assert_eq!(got.spatial_merge_size, 2);
        assert_eq!(got.max_patches, 128);
        assert!(got.downscale_oversized);
    }

    /// A fatal engine error must fail every queued *and* in-flight request and
    /// close its token stream, so no caller waits for a completion that can
    /// never arrive.
    #[tokio::test]
    async fn fail_all_fails_pending_and_inflight() {
        let engine = ServerEngine::new(1);

        let (pending_done_tx, pending_done_rx) = tokio::sync::oneshot::channel();
        let (pending_tok_tx, mut pending_tok_rx) = tokio::sync::mpsc::channel(1);
        engine.pending.lock().unwrap().push_back(Request {
            prompt: vec![1],
            max_new: 1,
            eos: None,
            stop_seqs: Vec::new(),
            logit_bias: Vec::new(),
            params: mach_model::sampling::SamplingParams::default(),
            images: Vec::new(),
            done: pending_done_tx,
            tokens_tx: Some(pending_tok_tx),
        });

        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        let (tok_tx, mut tok_rx) = tokio::sync::mpsc::channel(1);
        engine.txs.lock().unwrap().insert(7, done_tx);
        engine.streams.lock().unwrap().insert(7, tok_tx);

        engine.fail_all("boom");

        let err = pending_done_rx
            .await
            .expect("pending completion signalled")
            .expect_err("pending request must fail");
        assert!(err.to_string().contains("boom"), "{err}");
        assert!(
            pending_tok_rx.recv().await.is_none(),
            "pending stream closed"
        );

        let err = done_rx
            .await
            .expect("in-flight completion signalled")
            .expect_err("in-flight request must fail");
        assert!(err.to_string().contains("boom"), "{err}");
        assert!(tok_rx.recv().await.is_none(), "in-flight stream closed");
    }

    /// A submit that passed the early shutdown check but lost the race with a
    /// fatal engine failure must be rejected instead of enqueued into a queue
    /// no engine thread will drain.
    #[test]
    fn submit_rechecks_shutdown_under_pending_lock() {
        let engine = ServerEngine::new(1);
        let guard = engine.pending.lock().unwrap();
        let e2 = Arc::clone(&engine);
        let submit = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(e2.submit(
                vec![1],
                4,
                None,
                Vec::new(),
                Vec::new(),
                mach_model::sampling::SamplingParams::default(),
            ))
        });
        // Give the submit time to pass its early check and block on `pending`.
        std::thread::sleep(std::time::Duration::from_millis(100));
        engine.shutdown.store(true, Ordering::Release);
        drop(guard);
        let err = submit.join().unwrap().expect_err("must be rejected");
        assert!(matches!(err, EngineError::ShuttingDown), "{err}");
        assert!(engine.pending.lock().unwrap().is_empty());
    }

    /// A full token channel must fail the request instead of completing it.
    /// This is what makes the *final* token — which is pushed through the
    /// same path as every other token — unable to truncate a response while
    /// still reporting a clean `finish_reason`.
    #[test]
    fn final_token_full_channel_fails_instead_of_completing() {
        let id: SeqId = 5;
        let (tok_tx, _tok_rx) = tokio::sync::mpsc::channel(1);
        tok_tx.try_send(1).unwrap(); // channel is now full
        let (done_tx, mut done_rx) = tokio::sync::oneshot::channel();
        let mut streams = HashMap::new();
        streams.insert(id, tok_tx);
        let mut txs = HashMap::new();
        txs.insert(id, done_tx);

        let outcome = push_token(&mut streams, &mut txs, id, 2);

        assert_eq!(outcome, TokenDelivery::Stalled);
        assert!(streams.is_empty(), "stream sender removed");
        assert!(txs.is_empty(), "completion sender removed");
        let err = done_rx
            .try_recv()
            .expect("completion must be signalled")
            .expect_err("a dropped token must fail the request");
        assert!(err.to_string().contains("stalled"), "{err}");
    }

    /// With room in the channel (or when the request is not streamed) the
    /// token is queued and the completion is left for the engine to send.
    #[test]
    fn push_token_leaves_completion_when_not_stalled() {
        let id: SeqId = 6;
        let (tok_tx, mut tok_rx) = tokio::sync::mpsc::channel(1);
        let (done_tx, mut done_rx) = tokio::sync::oneshot::channel();
        let mut streams = HashMap::new();
        streams.insert(id, tok_tx);
        let mut txs = HashMap::new();
        txs.insert(id, done_tx);

        assert_eq!(push_token(&mut streams, &mut txs, id, 9), TokenDelivery::Ok);
        assert_eq!(tok_rx.try_recv().unwrap(), 9);
        assert!(txs.contains_key(&id), "completion still pending");
        assert!(done_rx.try_recv().is_err(), "completion not signalled yet");

        // A request that does not stream has no sender to stall.
        let mut streams = HashMap::new();
        assert_eq!(push_token(&mut streams, &mut txs, id, 9), TokenDelivery::Ok);
    }

    /// Regression for the speculative path: one round can emit several
    /// tokens for a sequence *and* be the round that finishes it. All of the
    /// tokens must be queued before the caller is told to complete, otherwise
    /// the trailing ones vanish while the client sees a clean finish.
    #[test]
    fn spec_round_queues_every_token_before_completing() {
        let id: SeqId = 7;
        let (tok_tx, mut tok_rx) = tokio::sync::mpsc::channel(8);
        let (done_tx, _done_rx) = tokio::sync::oneshot::channel();
        let mut streams = HashMap::new();
        streams.insert(id, tok_tx);
        let mut txs = HashMap::new();
        txs.insert(id, done_tx);

        let (complete, cancel) = deliver_step_tokens(
            vec![(id, 11), (id, 12), (id, 13)],
            &mut streams,
            &mut txs,
            |_| true, // finished within this very round
        );

        assert_eq!(complete, vec![id], "completed once, after the group");
        assert!(cancel.is_empty());
        assert_eq!(tok_rx.try_recv().unwrap(), 11);
        assert_eq!(tok_rx.try_recv().unwrap(), 12);
        assert_eq!(tok_rx.try_recv().unwrap(), 13);
        assert!(tok_rx.try_recv().is_err(), "no extra tokens");
        assert!(
            streams.contains_key(&id) && txs.contains_key(&id),
            "the caller still owns the completion"
        );
    }

    /// A stalled consumer on any token of the round fails the whole request
    /// and asks the caller to cancel the sequence — the completion is never
    /// sent.
    #[test]
    fn spec_round_stall_reports_cancel_and_fails_completion() {
        let id: SeqId = 8;
        let (tok_tx, _tok_rx) = tokio::sync::mpsc::channel(2);
        tok_tx.try_send(0).unwrap();
        tok_tx.try_send(0).unwrap(); // full
        let (done_tx, mut done_rx) = tokio::sync::oneshot::channel();
        let mut streams = HashMap::new();
        streams.insert(id, tok_tx);
        let mut txs = HashMap::new();
        txs.insert(id, done_tx);

        let (complete, cancel) =
            deliver_step_tokens(vec![(id, 11), (id, 12)], &mut streams, &mut txs, |_| true);

        assert!(complete.is_empty(), "a stalled round is never completed");
        assert_eq!(cancel, vec![id]);
        let err = done_rx
            .try_recv()
            .expect("completion must be signalled")
            .expect_err("stalled consumer fails the request");
        assert!(err.to_string().contains("stalled"), "{err}");
    }
}
