//! GPU (HIP) transformer for the P1 decode slice.
//!
//! The decode step is split into:
//! - `update_inputs`: copies the token + position into pinned host buffers and
//!   issues async H2D copies on the stream;
//! - `run_kernels`: the whole kernel sequence (GEMMs, norms, attention, KV
//!   store) — this is exactly what gets captured into a HIP graph;
//! - `read_logits`: stream sync + D2H copy.
//!
//! Because `pos` and `token` are read by kernels from device buffers, one
//! captured graph can serve every position: update the buffers between
//! replays, then replay.

use crate::adaptive::{AdaptiveProfile, BandwidthProbe, BandwidthProfile};
use crate::config::ModelDType;
use crate::fp16::f32_to_f16;
use crate::kernels::HipKernels;
use crate::moe_backend::LruExpertCache;
use crate::moe_offload;
use crate::sampling::HipSampler;
use crate::{Config, Error, Weights, WeightsFp8, WeightsQ4};
use mach_engine::graph::{GraphCapture, GraphHandle};
use mach_engine::hip::HipGraphCapture;
use mach_kernel_sys::hip::{self, Hip, HipEvent, HipStream};
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;

/// Per-layer device weight pointers.
#[derive(Clone, Copy)]
struct LayerDev {
    wq: *mut f32,
    wk: *mut f32,
    wv: *mut f32,
    wo: *mut f32,
    rms_attn: *mut f32,
    wg: *mut f32,
    wu: *mut f32,
    wd: *mut f32,
    rms_mlp: *mut f32,
    /// Attention projection biases (null when the checkpoint has none).
    bq: *mut f32,
    bk: *mut f32,
    bv: *mut f32,
    /// QK-norm (Qwen3): per-head RMSNorm weights (null when qk_norm=false).
    q_norm: *mut f32,
    k_norm: *mut f32,
    /// MLA (kv_lora_rank > 0): low-rank Q / compressed KV weights.
    mla_q_a: *mut f32,
    mla_q_a_norm: *mut f32,
    mla_q_b: *mut f32,
    mla_q_rope: *mut f32,
    mla_kv_a: *mut f32,
    mla_kv_a_norm: *mut f32,
    mla_kv_b: *mut f32,
    mla_o: *mut f32,
    /// MoE (num_experts > 0): router [ne,d] + per-expert gate/up/down.
    moe_router: *mut f32,
    moe_wg: *mut f32,
    moe_wu: *mut f32,
    moe_wd: *mut f32,
}

/// Per-layer fp16 device weight pointers (dtype = F16 only).
#[derive(Clone, Copy)]
struct LayerDevF16 {
    wq: *mut u16,
    wk: *mut u16,
    wv: *mut u16,
    wo: *mut u16,
    wg: *mut u16,
    wu: *mut u16,
    wd: *mut u16,
}

/// The GPU transformer.
/// Per-layer GPU-resident expert slot state (offload mode).
struct MoeSlotCtx {
    cap: usize,
    wg: *mut f32,
    wu: *mut f32,
    wd: *mut f32,
    /// slot_expert[slot] = expert id resident there, or -1 if empty.
    slot_expert: Vec<i32>,
    lru: LruExpertCache,
}

pub struct GpuModel {
    cfg: Config,
    k: Arc<HipKernels>,
    // pinned host input buffers
    host_tok: *mut i32,
    host_pos: *mut i32,
    // device input buffers (read by kernels; updated between replays)
    dev_tok: *mut i32,
    dev_pos: *mut i32,
    // activations
    x: *mut f32,
    xn: *mut f32,
    xn2: *mut f32,
    q: *mut f32,
    k_buf: *mut f32,
    v_buf: *mut f32,
    attn: *mut f32,
    proj: *mut f32,
    gate: *mut f32,
    up: *mut f32,
    h: *mut f32,
    logits: *mut f32,
    // weights on device (f32; fp16 copies when cfg.dtype == F16)
    emb_dev: *mut f32,
    rms_final_dev: *mut f32,
    lm_head_dev: *mut f32,
    layers_dev: Vec<LayerDev>,
    emb_f16: *mut u16,
    lm_head_f16: *mut u16,
    layers_f16: Vec<LayerDevF16>,
    /// fp16 scratch for GEMM A operands (size max(d_model, intermediate)).
    xh: *mut u16,
    /// fp16 scratch for hidden GEMM outputs (size max over n of hidden GEMMs).
    yh: *mut u16,
    // MoE scratch (single-token decode; reused across layers).
    router: *mut f32,
    exp_ids: *mut i32,
    exp_w: *mut f32,
    wg_pack: *mut f32,
    wu_pack: *mut f32,
    wd_pack: *mut f32,
    gate_all: *mut f32,
    up_all: *mut f32,
    eh_all: *mut f32,
    down_all: *mut f32,
    // MLA scratch (kv_lora_rank > 0): low-rank Q / compressed KV activations.
    mla_q_lora: *mut f32,
    mla_q_lora_n: *mut f32,
    q_nope: *mut f32,
    q_rope: *mut f32,
    mla_kv_a: *mut f32,
    mla_kv_a_n: *mut f32,
    mla_kv: *mut f32,
    mla_attn: *mut f32,
    // KV caches: (k, v) per layer
    kv_cache: Vec<(*mut f32, *mut f32)>,
    /// MLA KV caches (kv_lora_rank > 0): expanded per-head
    /// k `[max_seq, heads*(nope+rope)]` / v `[max_seq, heads*v_hd]`.
    mla_kv_cache: Vec<(*mut f32, *mut f32)>,
    /// All device allocations (freed on drop).
    allocs: Vec<*mut core::ffi::c_void>,
    /// Number of tokens stored so far.
    pos: usize,
    host_pins: Vec<*mut core::ffi::c_void>,
    /// GPU-side greedy sampler (reads only the sampled token).
    sampler: HipSampler,
    /// MoE offload: max routed experts computed on GPU (`usize::MAX` = full-resident).
    gpu_budget: usize,
    /// Host weight copy for the CPU fallback (set only in offload mode).
    host_w: Option<Arc<Weights>>,
    /// MoE offload: GPU-resident expert slots per layer (usize::MAX = full-resident).
    expert_slots: usize,
    /// Per-layer resident slot state (offload mode).
    slot_ctx: Vec<RefCell<MoeSlotCtx>>,
    /// Small device scratch for slot indices + reordered expert weights.
    slot_ids_dev: *mut i32,
    slot_w_dev: *mut f32,
    /// Bandwidth profile for adaptive (q*) placement; None = static placement.
    adaptive: Option<AdaptiveProfile>,
    /// Auto re-probe cadence: every N decode steps, re-measure PCIe bandwidth and
    /// fold it into the adaptive profile (0 = disabled). Enables real-time q*.
    reprobe_every: usize,
    /// Decode-step counter for the re-probe cadence.
    step_counter: usize,
    /// Dedicated copy stream (owned) for the offload paths' async D2H/H2D:
    /// the host-side MoE work overlaps the GPU-resident part instead of
    /// draining the compute stream every layer.
    xfer_stream: HipStream,
    /// Recorded on the compute stream after the layer's attention + router:
    /// the xfer stream's D2H reads wait on it (the GPU-resident GEMMs do not).
    router_done: HipEvent,
    /// Recorded on the compute stream after the GPU-resident accumulate: the
    /// residual H2D upload waits on it before overwriting `x`.
    gpu_part_done: HipEvent,
    /// Pinned host read-back staging for the offload paths (ids/weights/xn2/
    /// xh): hipMemcpyAsync on non-pinned buffers would fall back to
    /// synchronous copies, blocking the host per copy. Allocated in the
    /// offload constructors; null in full-resident builds.
    offload_ids: *mut i32,
    offload_w: *mut f32,
    offload_xn2: *mut f32,
    offload_xh: *mut f32,
}

impl GpuModel {
    /// Builds a GPU model and uploads `w` to device memory.
    pub fn new(hip: Arc<Hip>, cfg: Config, w: &Weights) -> Result<Self, Error> {
        Self::build(hip, cfg, w, usize::MAX)
    }

    /// Builds a GPU model with expert_slots GPU-resident expert slots per MoE layer
    /// (usize::MAX = full-resident, no offload).
    fn build(hip: Arc<Hip>, cfg: Config, w: &Weights, expert_slots: usize) -> Result<Self, Error> {
        let k = Arc::new(HipKernels::new(Arc::clone(&hip))?);
        let sampler = HipSampler::new(Arc::clone(&hip), k.stream)?;
        let mut m = Self {
            cfg,
            k,
            host_tok: std::ptr::null_mut(),
            host_pos: std::ptr::null_mut(),
            dev_tok: std::ptr::null_mut(),
            dev_pos: std::ptr::null_mut(),
            x: std::ptr::null_mut(),
            xn: std::ptr::null_mut(),
            xn2: std::ptr::null_mut(),
            q: std::ptr::null_mut(),
            k_buf: std::ptr::null_mut(),
            v_buf: std::ptr::null_mut(),
            attn: std::ptr::null_mut(),
            proj: std::ptr::null_mut(),
            gate: std::ptr::null_mut(),
            up: std::ptr::null_mut(),
            h: std::ptr::null_mut(),
            logits: std::ptr::null_mut(),
            emb_dev: std::ptr::null_mut(),
            rms_final_dev: std::ptr::null_mut(),
            lm_head_dev: std::ptr::null_mut(),
            layers_dev: Vec::new(),
            emb_f16: std::ptr::null_mut(),
            lm_head_f16: std::ptr::null_mut(),
            layers_f16: Vec::new(),
            xh: std::ptr::null_mut(),
            yh: std::ptr::null_mut(),
            router: std::ptr::null_mut(),
            exp_ids: std::ptr::null_mut(),
            exp_w: std::ptr::null_mut(),
            wg_pack: std::ptr::null_mut(),
            wu_pack: std::ptr::null_mut(),
            wd_pack: std::ptr::null_mut(),
            gate_all: std::ptr::null_mut(),
            up_all: std::ptr::null_mut(),
            eh_all: std::ptr::null_mut(),
            down_all: std::ptr::null_mut(),
            mla_q_lora: std::ptr::null_mut(),
            mla_q_lora_n: std::ptr::null_mut(),
            q_nope: std::ptr::null_mut(),
            q_rope: std::ptr::null_mut(),
            mla_kv_a: std::ptr::null_mut(),
            mla_kv_a_n: std::ptr::null_mut(),
            mla_kv: std::ptr::null_mut(),
            mla_attn: std::ptr::null_mut(),
            kv_cache: Vec::new(),
            mla_kv_cache: Vec::new(),
            allocs: Vec::new(),
            pos: 0,
            host_pins: Vec::new(),
            sampler,
            gpu_budget: usize::MAX,
            host_w: None,
            expert_slots,
            slot_ctx: Vec::new(),
            slot_ids_dev: std::ptr::null_mut(),
            slot_w_dev: std::ptr::null_mut(),
            adaptive: None,
            reprobe_every: 0,
            step_counter: 0,
            // Copy stream + events for the offload paths' async transfers.
            xfer_stream: {
                let mut s = std::ptr::null_mut();
                unsafe { hip::check(&hip, (hip.api.hip_stream_create)(&mut s))? };
                s
            },
            router_done: {
                let mut e = std::ptr::null_mut();
                unsafe { hip::check(&hip, (hip.api.hip_event_create)(&mut e))? };
                e
            },
            gpu_part_done: {
                let mut e = std::ptr::null_mut();
                unsafe { hip::check(&hip, (hip.api.hip_event_create)(&mut e))? };
                e
            },
            offload_ids: std::ptr::null_mut(),
            offload_w: std::ptr::null_mut(),
            offload_xn2: std::ptr::null_mut(),
            offload_xh: std::ptr::null_mut(),
        };
        m.alloc_buffers()?;
        m.upload_weights(w)?;
        Ok(m)
    }

    /// Builds a GPU model from storage-Q4 weights (dense, F16): each GEMM
    /// weight is dequantized to f16 on the host and uploaded directly, so host
    /// memory stays ~= the packed Q4 weights + one tensor's f16 buffer (8B
    /// model: ~5GB instead of ~48GB).
    pub fn from_q4(hip: Arc<Hip>, cfg: Config, w: &WeightsQ4) -> Result<Self, Error> {
        if cfg.dtype != ModelDType::F16 {
            return Err(Error::Model(
                "from_q4 requires dtype F16 (dequantize to f16 on device)".into(),
            ));
        }
        if cfg.num_experts != 0 || cfg.kv_lora_rank != 0 {
            return Err(Error::Model(
                "GpuModel::from_q4 currently supports dense models only (no MoE/MLA)".into(),
            ));
        }
        let k = Arc::new(HipKernels::new(Arc::clone(&hip))?);
        let sampler = HipSampler::new(Arc::clone(&hip), k.stream)?;
        let mut m = Self {
            cfg,
            k,
            host_tok: std::ptr::null_mut(),
            host_pos: std::ptr::null_mut(),
            dev_tok: std::ptr::null_mut(),
            dev_pos: std::ptr::null_mut(),
            x: std::ptr::null_mut(),
            xn: std::ptr::null_mut(),
            xn2: std::ptr::null_mut(),
            q: std::ptr::null_mut(),
            k_buf: std::ptr::null_mut(),
            v_buf: std::ptr::null_mut(),
            attn: std::ptr::null_mut(),
            proj: std::ptr::null_mut(),
            gate: std::ptr::null_mut(),
            up: std::ptr::null_mut(),
            h: std::ptr::null_mut(),
            logits: std::ptr::null_mut(),
            emb_dev: std::ptr::null_mut(),
            rms_final_dev: std::ptr::null_mut(),
            lm_head_dev: std::ptr::null_mut(),
            layers_dev: Vec::new(),
            emb_f16: std::ptr::null_mut(),
            lm_head_f16: std::ptr::null_mut(),
            layers_f16: Vec::new(),
            xh: std::ptr::null_mut(),
            yh: std::ptr::null_mut(),
            router: std::ptr::null_mut(),
            exp_ids: std::ptr::null_mut(),
            exp_w: std::ptr::null_mut(),
            wg_pack: std::ptr::null_mut(),
            wu_pack: std::ptr::null_mut(),
            wd_pack: std::ptr::null_mut(),
            gate_all: std::ptr::null_mut(),
            up_all: std::ptr::null_mut(),
            eh_all: std::ptr::null_mut(),
            down_all: std::ptr::null_mut(),
            mla_q_lora: std::ptr::null_mut(),
            mla_q_lora_n: std::ptr::null_mut(),
            q_nope: std::ptr::null_mut(),
            q_rope: std::ptr::null_mut(),
            mla_kv_a: std::ptr::null_mut(),
            mla_kv_a_n: std::ptr::null_mut(),
            mla_kv: std::ptr::null_mut(),
            mla_attn: std::ptr::null_mut(),
            kv_cache: Vec::new(),
            mla_kv_cache: Vec::new(),
            allocs: Vec::new(),
            pos: 0,
            host_pins: Vec::new(),
            sampler,
            gpu_budget: usize::MAX,
            host_w: None,
            expert_slots: usize::MAX,
            slot_ctx: Vec::new(),
            slot_ids_dev: std::ptr::null_mut(),
            slot_w_dev: std::ptr::null_mut(),
            adaptive: None,
            reprobe_every: 0,
            step_counter: 0,
            xfer_stream: {
                let mut s = std::ptr::null_mut();
                unsafe { hip::check(&hip, (hip.api.hip_stream_create)(&mut s))? };
                s
            },
            router_done: {
                let mut e = std::ptr::null_mut();
                unsafe { hip::check(&hip, (hip.api.hip_event_create)(&mut e))? };
                e
            },
            gpu_part_done: {
                let mut e = std::ptr::null_mut();
                unsafe { hip::check(&hip, (hip.api.hip_event_create)(&mut e))? };
                e
            },
            offload_ids: std::ptr::null_mut(),
            offload_w: std::ptr::null_mut(),
            offload_xn2: std::ptr::null_mut(),
            offload_xh: std::ptr::null_mut(),
        };
        m.alloc_buffers()?;
        m.upload_weights_q4(w)?;
        Ok(m)
    }

    /// Builds a GPU model from storage-FP8 weights (dense, F16): each GEMM
    /// weight is dequantized to f16 on the host and uploaded directly, so host
    /// memory stays ~= the packed FP8 weights + one tensor's f16 buffer
    /// (8B model: ~8GB instead of ~48GB).
    pub fn from_fp8(hip: Arc<Hip>, cfg: Config, w: &WeightsFp8) -> Result<Self, Error> {
        if cfg.dtype != ModelDType::F16 {
            return Err(Error::Model(
                "from_fp8 requires dtype F16 (dequantize to f16 on device)".into(),
            ));
        }
        if cfg.num_experts != 0 || cfg.kv_lora_rank != 0 {
            return Err(Error::Model(
                "GpuModel::from_fp8 currently supports dense models only (no MoE/MLA)".into(),
            ));
        }
        let k = Arc::new(HipKernels::new(Arc::clone(&hip))?);
        let sampler = HipSampler::new(Arc::clone(&hip), k.stream)?;
        let mut m = Self {
            cfg,
            k,
            host_tok: std::ptr::null_mut(),
            host_pos: std::ptr::null_mut(),
            dev_tok: std::ptr::null_mut(),
            dev_pos: std::ptr::null_mut(),
            x: std::ptr::null_mut(),
            xn: std::ptr::null_mut(),
            xn2: std::ptr::null_mut(),
            q: std::ptr::null_mut(),
            k_buf: std::ptr::null_mut(),
            v_buf: std::ptr::null_mut(),
            attn: std::ptr::null_mut(),
            proj: std::ptr::null_mut(),
            gate: std::ptr::null_mut(),
            up: std::ptr::null_mut(),
            h: std::ptr::null_mut(),
            logits: std::ptr::null_mut(),
            emb_dev: std::ptr::null_mut(),
            rms_final_dev: std::ptr::null_mut(),
            lm_head_dev: std::ptr::null_mut(),
            layers_dev: Vec::new(),
            emb_f16: std::ptr::null_mut(),
            lm_head_f16: std::ptr::null_mut(),
            layers_f16: Vec::new(),
            xh: std::ptr::null_mut(),
            yh: std::ptr::null_mut(),
            router: std::ptr::null_mut(),
            exp_ids: std::ptr::null_mut(),
            exp_w: std::ptr::null_mut(),
            wg_pack: std::ptr::null_mut(),
            wu_pack: std::ptr::null_mut(),
            wd_pack: std::ptr::null_mut(),
            gate_all: std::ptr::null_mut(),
            up_all: std::ptr::null_mut(),
            eh_all: std::ptr::null_mut(),
            down_all: std::ptr::null_mut(),
            mla_q_lora: std::ptr::null_mut(),
            mla_q_lora_n: std::ptr::null_mut(),
            q_nope: std::ptr::null_mut(),
            q_rope: std::ptr::null_mut(),
            mla_kv_a: std::ptr::null_mut(),
            mla_kv_a_n: std::ptr::null_mut(),
            mla_kv: std::ptr::null_mut(),
            mla_attn: std::ptr::null_mut(),
            kv_cache: Vec::new(),
            mla_kv_cache: Vec::new(),
            allocs: Vec::new(),
            pos: 0,
            host_pins: Vec::new(),
            sampler,
            gpu_budget: usize::MAX,
            host_w: None,
            expert_slots: usize::MAX,
            slot_ctx: Vec::new(),
            slot_ids_dev: std::ptr::null_mut(),
            slot_w_dev: std::ptr::null_mut(),
            adaptive: None,
            reprobe_every: 0,
            step_counter: 0,
            xfer_stream: {
                let mut s = std::ptr::null_mut();
                unsafe { hip::check(&hip, (hip.api.hip_stream_create)(&mut s))? };
                s
            },
            router_done: {
                let mut e = std::ptr::null_mut();
                unsafe { hip::check(&hip, (hip.api.hip_event_create)(&mut e))? };
                e
            },
            gpu_part_done: {
                let mut e = std::ptr::null_mut();
                unsafe { hip::check(&hip, (hip.api.hip_event_create)(&mut e))? };
                e
            },
            offload_ids: std::ptr::null_mut(),
            offload_w: std::ptr::null_mut(),
            offload_xn2: std::ptr::null_mut(),
            offload_xh: std::ptr::null_mut(),
        };
        m.alloc_buffers()?;
        m.upload_weights_fp8(w)?;
        Ok(m)
    }

    fn dalloc(&mut self, bytes: usize) -> Result<*mut f32, Error> {
        let p = hip::malloc(self.k.hip(), bytes)
            .map_err(|e| Error::Model(format!("device alloc of {bytes} bytes failed: {e}")))?;
        self.allocs.push(p);
        Ok(p as *mut f32)
    }

    fn alloc_buffers(&mut self) -> Result<(), Error> {
        let c = self.cfg;
        let d = c.d_model;
        let nq = c.n_heads * c.head_dim;
        let nkv = c.n_kv_heads * c.head_dim;

        // pinned host inputs
        let ht = hip::host_malloc(self.k.hip(), 4)?;
        let hp = hip::host_malloc(self.k.hip(), 4)?;
        self.host_tok = ht as *mut i32;
        self.host_pos = hp as *mut i32;
        self.host_pins.push(ht);
        self.host_pins.push(hp);

        self.dev_tok = self.dalloc(4)? as *mut i32;
        self.dev_pos = self.dalloc(4)? as *mut i32;
        self.x = self.dalloc(d * 4)?;
        self.xn = self.dalloc(d * 4)?;
        self.xn2 = self.dalloc(d * 4)?;
        self.q = self.dalloc(nq * 4)?;
        self.k_buf = self.dalloc(nkv * 4)?;
        self.v_buf = self.dalloc(nkv * 4)?;
        self.attn = self.dalloc(nq * 4)?;
        self.proj = self.dalloc(d * 4)?;
        let inter = c.intermediate_size;
        self.gate = self.dalloc(inter * 4)?;
        self.up = self.dalloc(inter * 4)?;
        self.h = self.dalloc(inter * 4)?;
        self.logits = self.dalloc(c.vocab_size * 4)?;
        // fp16 GEMM scratch (only used in F16 mode): A operand + fp16 output
        // (yh must cover the lm_head output = vocab for the c16 logits path).
        let max_n = c
            .d_model
            .max(c.intermediate_size)
            .max(c.n_heads * c.head_dim)
            .max(c.vocab_size);
        let xh_bytes = c.d_model.max(c.intermediate_size) * 2;
        let xh = hip::malloc(self.k.hip(), xh_bytes)?;
        self.xh = xh as *mut u16;
        self.allocs.push(xh);
        let yh = hip::malloc(self.k.hip(), max_n * 2)?;
        self.yh = yh as *mut u16;
        self.allocs.push(yh);

        // Dense KV cache: skipped on the MLA path (kv_lora_rank > 0), which
        // keeps its expanded per-head KV in `mla_kv_cache`; allocating it here
        // would waste VRAM (n_kv_heads = n_heads, head_dim = nope+rope).
        if c.kv_lora_rank == 0 {
            let kv_bytes = c.max_seq_len * c.n_kv_heads * c.head_dim * 4;
            for _ in 0..c.n_layers {
                let kk = self.dalloc(kv_bytes)?;
                let vv = self.dalloc(kv_bytes)?;
                self.kv_cache.push((kk, vv));
            }
        }
        if c.kv_lora_rank > 0 {
            let qlr = c.q_lora_rank;
            let nope = c.qk_nope_head_dim;
            let rope = c.qk_rope_head_dim;
            let v_hd = c.v_head_dim;
            let heads = c.n_heads;
            self.mla_q_lora = self.dalloc(qlr * 4)?;
            self.mla_q_lora_n = self.dalloc(qlr * 4)?;
            self.q_nope = self.dalloc(heads * nope * 4)?;
            self.q_rope = self.dalloc(heads * rope * 4)?;
            self.mla_kv_a = self.dalloc((c.kv_lora_rank + rope) * 4)?;
            self.mla_kv_a_n = self.dalloc(c.kv_lora_rank * 4)?;
            self.mla_kv = self.dalloc(heads * (nope + v_hd) * 4)?;
            self.mla_attn = self.dalloc(heads * v_hd * 4)?;
            let k_bytes = c.max_seq_len * heads * (nope + rope) * 4;
            let v_bytes = c.max_seq_len * heads * v_hd * 4;
            for _ in 0..c.n_layers {
                let kk = self.dalloc(k_bytes)?;
                let vv = self.dalloc(v_bytes)?;
                self.mla_kv_cache.push((kk, vv));
            }
        }
        if c.num_experts > 0 {
            let ne = c.num_experts;
            let topk = c.num_experts_per_tok.min(ne);
            let einter = c.expert_size();
            if topk > 0 {
                self.router = self.dalloc(ne * 4)?;
                let ip = hip::malloc(self.k.hip(), topk * 4)?;
                self.exp_ids = ip as *mut i32;
                self.allocs.push(ip);
                self.exp_w = self.dalloc(topk * 4)?;
                let slot_g = topk * einter * d;
                let slot_d = topk * d * einter;
                self.wg_pack = self.dalloc(slot_g * 4)?;
                self.wu_pack = self.dalloc(slot_g * 4)?;
                self.wd_pack = self.dalloc(slot_d * 4)?;
                self.gate_all = self.dalloc(topk * einter * 4)?;
                self.up_all = self.dalloc(topk * einter * 4)?;
                self.eh_all = self.dalloc(topk * einter * 4)?;
                self.down_all = self.dalloc(topk * d * 4)?;
                self.slot_ids_dev = self.dalloc(topk * 4)? as *mut i32;
                self.slot_w_dev = self.dalloc(topk * 4)?;
            }
        }
        Ok(())
    }

    fn upload(&self, dst: *mut f32, src: &[f32]) -> Result<(), Error> {
        hip::memcpy(
            self.k.hip(),
            dst as *mut core::ffi::c_void,
            src.as_ptr() as *const core::ffi::c_void,
            src.len() * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )?;
        Ok(())
    }

    fn upload_f16(&self, dst: *mut u16, src: &[f32]) -> Result<(), Error> {
        let buf: Vec<u16> = src.iter().map(|&v| f32_to_f16(v)).collect();
        hip::memcpy(
            self.k.hip(),
            dst as *mut core::ffi::c_void,
            buf.as_ptr() as *const core::ffi::c_void,
            buf.len() * 2,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )?;
        Ok(())
    }

    /// Uploads pre-encoded f16 bits directly (no f32 round trip).
    fn upload_f16_bits(&self, dst: *mut u16, src: &[u16]) -> Result<(), Error> {
        hip::memcpy(
            self.k.hip(),
            dst as *mut core::ffi::c_void,
            src.as_ptr() as *const core::ffi::c_void,
            src.len() * 2,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )?;
        Ok(())
    }

    /// Uploads an optional f32 tensor (empty -> null device pointer).
    fn upload_opt(&mut self, v: &[f32]) -> Result<*mut f32, Error> {
        if v.is_empty() {
            Ok(std::ptr::null_mut())
        } else {
            let p = self.dalloc(v.len() * 4)?;
            self.upload(p, v)?;
            Ok(p)
        }
    }

    /// Uploads storage-Q4 weights as f16 (dense F16 path): norms/biases stay
    /// f32; GEMM matrices are dequantized per tensor and freed after upload.
    fn upload_weights_q4(&mut self, w: &WeightsQ4) -> Result<(), Error> {
        self.emb_f16 = self.alloc_f16(w.tok_emb.len())?;
        self.lm_head_f16 = self.alloc_f16(w.lm_head.len())?;
        self.upload_f16_bits(self.emb_f16, &w.tok_emb.dequantize_f16())?;
        self.upload_f16_bits(self.lm_head_f16, &w.lm_head.dequantize_f16())?;
        self.rms_final_dev = self.dalloc(w.rms_final.len() * 4)?;
        self.upload(self.rms_final_dev, &w.rms_final)?;

        for lw in &w.layers {
            let l = LayerDev {
                wq: std::ptr::null_mut(),
                wk: std::ptr::null_mut(),
                wv: std::ptr::null_mut(),
                wo: std::ptr::null_mut(),
                rms_attn: self.dalloc(lw.rms_attn.len() * 4)?,
                wg: std::ptr::null_mut(),
                wu: std::ptr::null_mut(),
                wd: std::ptr::null_mut(),
                rms_mlp: self.dalloc(lw.rms_mlp.len() * 4)?,
                bq: self.upload_opt(&lw.bq)?,
                bk: self.upload_opt(&lw.bk)?,
                bv: self.upload_opt(&lw.bv)?,
                q_norm: self.upload_opt(&lw.q_norm)?,
                k_norm: self.upload_opt(&lw.k_norm)?,
                mla_q_a: std::ptr::null_mut(),
                mla_q_a_norm: std::ptr::null_mut(),
                mla_q_b: std::ptr::null_mut(),
                mla_q_rope: std::ptr::null_mut(),
                mla_kv_a: std::ptr::null_mut(),
                mla_kv_a_norm: std::ptr::null_mut(),
                mla_kv_b: std::ptr::null_mut(),
                mla_o: std::ptr::null_mut(),
                moe_router: std::ptr::null_mut(),
                moe_wg: std::ptr::null_mut(),
                moe_wu: std::ptr::null_mut(),
                moe_wd: std::ptr::null_mut(),
            };
            self.upload(l.rms_attn, &lw.rms_attn)?;
            self.upload(l.rms_mlp, &lw.rms_mlp)?;
            self.layers_dev.push(l);
            let l16 = LayerDevF16 {
                wq: self.alloc_f16(lw.wq.len())?,
                wk: self.alloc_f16(lw.wk.len())?,
                wv: self.alloc_f16(lw.wv.len())?,
                wo: self.alloc_f16(lw.wo.len())?,
                wg: self.alloc_f16(lw.wg.len())?,
                wu: self.alloc_f16(lw.wu.len())?,
                wd: self.alloc_f16(lw.wd.len())?,
            };
            self.upload_f16_bits(l16.wq, &lw.wq.dequantize_f16())?;
            self.upload_f16_bits(l16.wk, &lw.wk.dequantize_f16())?;
            self.upload_f16_bits(l16.wv, &lw.wv.dequantize_f16())?;
            self.upload_f16_bits(l16.wo, &lw.wo.dequantize_f16())?;
            self.upload_f16_bits(l16.wg, &lw.wg.dequantize_f16())?;
            self.upload_f16_bits(l16.wu, &lw.wu.dequantize_f16())?;
            self.upload_f16_bits(l16.wd, &lw.wd.dequantize_f16())?;
            self.layers_f16.push(l16);
        }
        Ok(())
    }

    /// Uploads storage-FP8 weights as f16 (dense F16 path): norms/biases stay
    /// f32; GEMM matrices are dequantized per tensor and freed after upload.
    fn upload_weights_fp8(&mut self, w: &WeightsFp8) -> Result<(), Error> {
        self.emb_f16 = self.alloc_f16(w.tok_emb.len())?;
        self.lm_head_f16 = self.alloc_f16(w.lm_head.len())?;
        self.upload_f16_bits(self.emb_f16, &w.tok_emb.dequantize_f16())?;
        self.upload_f16_bits(self.lm_head_f16, &w.lm_head.dequantize_f16())?;
        self.rms_final_dev = self.dalloc(w.rms_final.len() * 4)?;
        self.upload(self.rms_final_dev, &w.rms_final)?;

        for lw in &w.layers {
            let l = LayerDev {
                wq: std::ptr::null_mut(),
                wk: std::ptr::null_mut(),
                wv: std::ptr::null_mut(),
                wo: std::ptr::null_mut(),
                rms_attn: self.dalloc(lw.rms_attn.len() * 4)?,
                wg: std::ptr::null_mut(),
                wu: std::ptr::null_mut(),
                wd: std::ptr::null_mut(),
                rms_mlp: self.dalloc(lw.rms_mlp.len() * 4)?,
                bq: self.upload_opt(&lw.bq)?,
                bk: self.upload_opt(&lw.bk)?,
                bv: self.upload_opt(&lw.bv)?,
                q_norm: self.upload_opt(&lw.q_norm)?,
                k_norm: self.upload_opt(&lw.k_norm)?,
                mla_q_a: std::ptr::null_mut(),
                mla_q_a_norm: std::ptr::null_mut(),
                mla_q_b: std::ptr::null_mut(),
                mla_q_rope: std::ptr::null_mut(),
                mla_kv_a: std::ptr::null_mut(),
                mla_kv_a_norm: std::ptr::null_mut(),
                mla_kv_b: std::ptr::null_mut(),
                mla_o: std::ptr::null_mut(),
                moe_router: std::ptr::null_mut(),
                moe_wg: std::ptr::null_mut(),
                moe_wu: std::ptr::null_mut(),
                moe_wd: std::ptr::null_mut(),
            };
            self.upload(l.rms_attn, &lw.rms_attn)?;
            self.upload(l.rms_mlp, &lw.rms_mlp)?;
            self.layers_dev.push(l);
            let l16 = LayerDevF16 {
                wq: self.alloc_f16(lw.wq.len())?,
                wk: self.alloc_f16(lw.wk.len())?,
                wv: self.alloc_f16(lw.wv.len())?,
                wo: self.alloc_f16(lw.wo.len())?,
                wg: self.alloc_f16(lw.wg.len())?,
                wu: self.alloc_f16(lw.wu.len())?,
                wd: self.alloc_f16(lw.wd.len())?,
            };
            self.upload_f16_bits(l16.wq, &lw.wq.dequantize_f16())?;
            self.upload_f16_bits(l16.wk, &lw.wk.dequantize_f16())?;
            self.upload_f16_bits(l16.wv, &lw.wv.dequantize_f16())?;
            self.upload_f16_bits(l16.wo, &lw.wo.dequantize_f16())?;
            self.upload_f16_bits(l16.wg, &lw.wg.dequantize_f16())?;
            self.upload_f16_bits(l16.wu, &lw.wu.dequantize_f16())?;
            self.upload_f16_bits(l16.wd, &lw.wd.dequantize_f16())?;
            self.layers_f16.push(l16);
        }
        Ok(())
    }

    fn alloc_f16(&mut self, n: usize) -> Result<*mut u16, Error> {
        let p = hip::malloc(self.k.hip(), n * 2)?;
        self.allocs.push(p);
        Ok(p as *mut u16)
    }

    /// Allocates + uploads an f32 matrix, or returns null on the F16 path
    /// (fp16 weights live in `LayerDevF16`; the f32 copy is not needed).
    fn upload_mat32(&mut self, src: &[f32], f16: bool) -> Result<*mut f32, Error> {
        if f16 {
            Ok(std::ptr::null_mut())
        } else {
            let p = self.dalloc(src.len() * 4)?;
            self.upload(p, src)?;
            Ok(p)
        }
    }

    fn upload_weights(&mut self, w: &Weights) -> Result<(), Error> {
        let c = self.cfg;
        let f16 = c.dtype == ModelDType::F16;
        self.emb_dev = self.upload_mat32(&w.tok_emb, f16)?;
        self.rms_final_dev = self.dalloc(w.rms_final.len() * 4)?;
        self.lm_head_dev = self.upload_mat32(&w.lm_head, f16)?;
        self.upload(self.rms_final_dev, &w.rms_final)?;
        if c.dtype == ModelDType::F16 {
            self.emb_f16 = self.alloc_f16(w.tok_emb.len())?;
            self.lm_head_f16 = self.alloc_f16(w.lm_head.len())?;
            self.upload_f16(self.emb_f16, &w.tok_emb)?;
            self.upload_f16(self.lm_head_f16, &w.lm_head)?;
        }

        let d = c.d_model;
        let nq = c.n_heads * c.head_dim;
        let nkv = c.n_kv_heads * c.head_dim;
        for lw in &w.layers {
            let l = LayerDev {
                wq: self.upload_mat32(&lw.wq, f16)?,
                wk: self.upload_mat32(&lw.wk, f16)?,
                wv: self.upload_mat32(&lw.wv, f16)?,
                wo: self.upload_mat32(&lw.wo, f16)?,
                rms_attn: self.dalloc(lw.rms_attn.len() * 4)?,
                wg: self.upload_mat32(&lw.wg, f16)?,
                wu: self.upload_mat32(&lw.wu, f16)?,
                wd: self.upload_mat32(&lw.wd, f16)?,
                rms_mlp: self.dalloc(lw.rms_mlp.len() * 4)?,
                bq: if lw.bq.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.bq.len() * 4)?;
                    self.upload(p, &lw.bq)?;
                    p
                },
                bk: if lw.bk.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.bk.len() * 4)?;
                    self.upload(p, &lw.bk)?;
                    p
                },
                bv: if lw.bv.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.bv.len() * 4)?;
                    self.upload(p, &lw.bv)?;
                    p
                },
                q_norm: if lw.q_norm.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.q_norm.len() * 4)?;
                    self.upload(p, &lw.q_norm)?;
                    p
                },
                k_norm: if lw.k_norm.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.k_norm.len() * 4)?;
                    self.upload(p, &lw.k_norm)?;
                    p
                },
                mla_q_a: if lw.mla_q_a.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.mla_q_a.len() * 4)?;
                    self.upload(p, &lw.mla_q_a)?;
                    p
                },
                mla_q_a_norm: if lw.mla_q_a_norm.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.mla_q_a_norm.len() * 4)?;
                    self.upload(p, &lw.mla_q_a_norm)?;
                    p
                },
                mla_q_b: if lw.mla_q_b.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.mla_q_b.len() * 4)?;
                    self.upload(p, &lw.mla_q_b)?;
                    p
                },
                mla_q_rope: if lw.mla_q_rope.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.mla_q_rope.len() * 4)?;
                    self.upload(p, &lw.mla_q_rope)?;
                    p
                },
                mla_kv_a: if lw.mla_kv_a.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.mla_kv_a.len() * 4)?;
                    self.upload(p, &lw.mla_kv_a)?;
                    p
                },
                mla_kv_a_norm: if lw.mla_kv_a_norm.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.mla_kv_a_norm.len() * 4)?;
                    self.upload(p, &lw.mla_kv_a_norm)?;
                    p
                },
                mla_kv_b: if lw.mla_kv_b.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.mla_kv_b.len() * 4)?;
                    self.upload(p, &lw.mla_kv_b)?;
                    p
                },
                mla_o: if lw.mla_o.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.mla_o.len() * 4)?;
                    self.upload(p, &lw.mla_o)?;
                    p
                },
                moe_router: if lw.moe_router.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.moe_router.len() * 4)?;
                    self.upload(p, &lw.moe_router)?;
                    p
                },
                moe_wg: if lw.moe_wg.is_empty() {
                    std::ptr::null_mut()
                } else if self.expert_slots < c.num_experts {
                    self.dalloc(self.expert_slots * c.expert_size() * c.d_model * 4)?
                } else {
                    let p = self.dalloc(lw.moe_wg.len() * 4)?;
                    self.upload(p, &lw.moe_wg)?;
                    p
                },
                moe_wu: if lw.moe_wu.is_empty() {
                    std::ptr::null_mut()
                } else if self.expert_slots < c.num_experts {
                    self.dalloc(self.expert_slots * c.expert_size() * c.d_model * 4)?
                } else {
                    let p = self.dalloc(lw.moe_wu.len() * 4)?;
                    self.upload(p, &lw.moe_wu)?;
                    p
                },
                moe_wd: if lw.moe_wd.is_empty() {
                    std::ptr::null_mut()
                } else if self.expert_slots < c.num_experts {
                    self.dalloc(self.expert_slots * c.expert_size() * c.d_model * 4)?
                } else {
                    let p = self.dalloc(lw.moe_wd.len() * 4)?;
                    self.upload(p, &lw.moe_wd)?;
                    p
                },
            };
            self.upload(l.rms_attn, &lw.rms_attn)?;
            self.upload(l.rms_mlp, &lw.rms_mlp)?;
            let _ = (d, nq, nkv);
            self.slot_ctx.push(RefCell::new(
                if !lw.moe_wg.is_empty() && self.expert_slots < c.num_experts {
                    MoeSlotCtx {
                        cap: self.expert_slots,
                        wg: l.moe_wg,
                        wu: l.moe_wu,
                        wd: l.moe_wd,
                        slot_expert: vec![-1; self.expert_slots],
                        lru: LruExpertCache::new(self.expert_slots),
                    }
                } else {
                    MoeSlotCtx {
                        cap: 0,
                        wg: std::ptr::null_mut(),
                        wu: std::ptr::null_mut(),
                        wd: std::ptr::null_mut(),
                        slot_expert: Vec::new(),
                        lru: LruExpertCache::new(0),
                    }
                },
            ));
            self.layers_dev.push(l);
            if c.dtype == ModelDType::F16 {
                let l16 = LayerDevF16 {
                    wq: self.alloc_f16(lw.wq.len())?,
                    wk: self.alloc_f16(lw.wk.len())?,
                    wv: self.alloc_f16(lw.wv.len())?,
                    wo: self.alloc_f16(lw.wo.len())?,
                    wg: self.alloc_f16(lw.wg.len())?,
                    wu: self.alloc_f16(lw.wu.len())?,
                    wd: self.alloc_f16(lw.wd.len())?,
                };
                self.upload_f16(l16.wq, &lw.wq)?;
                self.upload_f16(l16.wk, &lw.wk)?;
                self.upload_f16(l16.wv, &lw.wv)?;
                self.upload_f16(l16.wo, &lw.wo)?;
                self.upload_f16(l16.wg, &lw.wg)?;
                self.upload_f16(l16.wu, &lw.wu)?;
                self.upload_f16(l16.wd, &lw.wd)?;
                self.layers_f16.push(l16);
            }
        }
        Ok(())
    }

    /// Sets the token/position host buffers and issues async H2D copies.
    fn update_inputs(&self, token: u32) -> Result<(), Error> {
        unsafe {
            *self.host_tok = token as i32;
            *self.host_pos = self.pos as i32;
        }
        let k = &self.k;
        hip::memcpy_async(
            k.hip(),
            self.dev_tok as *mut core::ffi::c_void,
            self.host_tok as *const core::ffi::c_void,
            4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            k.stream,
        )?;
        hip::memcpy_async(
            k.hip(),
            self.dev_pos as *mut core::ffi::c_void,
            self.host_pos as *const core::ffi::c_void,
            4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            k.stream,
        )?;
        Ok(())
    }

    /// Runs the full decode kernel sequence (capturable).
    fn run_kernels(&self) -> Result<(), Error> {
        let c = self.cfg;
        let d = c.d_model as i32;
        let k = &self.k;
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let f16 = c.dtype == ModelDType::F16;

        // Selects fp16 (cast activations + fp16 weights, fp32 output) or f32.
        // The single-sequence decode is always m=1: the custom GEMV kernel
        // replaces the rocBLAS m=1 path (60x over the memory bound there —
        // see the GEMV_F16 contract). GEMV_MAX_D guards the 48 KB shared
        // staging (f16 MLA o_proj at DeepSeek-level kk exceeds it and falls
        // back to the hipBLAS path).
        let gemm = |out: *mut f32,
                    x: *const f32,
                    w32: *mut f32,
                    w16: *mut u16,
                    n: i32,
                    kk: i32|
         -> Result<(), Error> {
            if f16 {
                if kk <= crate::batched::GEMV_MAX_D {
                    k.launch_gemv_f16(out, x, w16, n, kk, 1)
                } else {
                    k.gemm_f16(out, x, w16, n, kk, self.xh, self.yh)
                }
            } else {
                k.gemm(out, x, w32, n, kk)
            }
        };

        if f16 {
            k.launch_embed_f16(self.dev_tok, self.emb_f16, self.x, d, 1)?;
        } else {
            k.launch_embed(self.dev_tok, self.emb_dev, self.x, d)?;
        }
        for (li, lw) in self.layers_dev.iter().enumerate() {
            let nq = (c.n_heads * c.head_dim) as i32;
            let nkv = (c.n_kv_heads * c.head_dim) as i32;
            let l16 = if f16 { Some(self.layers_f16[li]) } else { None };
            k.launch_rms_norm(self.x, lw.rms_attn, self.xn, 1, d, c.rms_eps)?;
            if c.kv_lora_rank > 0 {
                let heads = c.n_heads as i32;
                let qlr = c.q_lora_rank as i32;
                let nope = c.qk_nope_head_dim as i32;
                let rope = c.qk_rope_head_dim as i32;
                let v_hd = c.v_head_dim as i32;
                let kvlr = c.kv_lora_rank as i32;
                // q_lora = q_a(xn); rms; q_nope = q_b(q_lora)
                k.gemm(self.mla_q_lora, self.xn, lw.mla_q_a, qlr, d)?;
                k.launch_rms_norm(
                    self.mla_q_lora,
                    lw.mla_q_a_norm,
                    self.mla_q_lora_n,
                    1,
                    qlr,
                    c.rms_eps,
                )?;
                k.gemm(
                    self.q_nope,
                    self.mla_q_lora_n,
                    lw.mla_q_b,
                    heads * nope,
                    qlr,
                )?;
                // q_rope = q_rope_proj(xn) + RoPE (k_buf is scratch on this path).
                k.gemm(self.q_rope, self.xn, lw.mla_q_rope, heads * rope, d)?;
                k.launch_rope(
                    self.q_rope,
                    self.k_buf,
                    self.dev_pos,
                    heads,
                    heads,
                    rope,
                    c.rope_theta,
                )?;
                // compressed_kv = kv_a(xn); latent rms; k_rope (shared) + RoPE.
                k.gemm(self.mla_kv_a, self.xn, lw.mla_kv_a, kvlr + rope, d)?;
                k.launch_rms_norm(
                    self.mla_kv_a,
                    lw.mla_kv_a_norm,
                    self.mla_kv_a_n,
                    1,
                    kvlr,
                    c.rms_eps,
                )?;
                let k_rope_ptr = unsafe { self.mla_kv_a.add(kvlr as usize) };
                k.launch_rope(
                    k_rope_ptr,
                    self.v_buf,
                    self.dev_pos,
                    1,
                    1,
                    rope,
                    c.rope_theta,
                )?;
                // kv = kv_b_proj(latent): [heads*(nope + v_hd)].
                k.gemm(
                    self.mla_kv,
                    self.mla_kv_a_n,
                    lw.mla_kv_b,
                    heads * (nope + v_hd),
                    kvlr,
                )?;
                // Assemble per-head q/k/v and store into the MLA caches.
                k.launch_mla_assemble_q(self.q_nope, self.q_rope, self.q, heads, nope, rope)?;
                let (kc, vc) = self.mla_kv_cache[li];
                k.launch_mla_assemble_kv(
                    self.mla_kv,
                    k_rope_ptr,
                    kc,
                    vc,
                    self.dev_pos,
                    heads,
                    nope,
                    rope,
                    v_hd,
                )?;
                let scale = 1.0 / ((nope + rope) as f32).sqrt();
                k.launch_mla_attn_decode(
                    self.q,
                    kc,
                    vc,
                    self.mla_attn,
                    self.dev_pos,
                    heads,
                    nope + rope,
                    v_hd,
                    scale,
                    c.max_seq_len as i32,
                )?;
                k.gemm(self.proj, self.mla_attn, lw.mla_o, d, heads * v_hd)?;
                k.launch_add(self.x, self.proj, d)?;
            } else {
                gemm(
                    self.q,
                    self.xn,
                    lw.wq,
                    l16.map_or(std::ptr::null_mut(), |l| l.wq),
                    nq,
                    d,
                )?;
                gemm(
                    self.k_buf,
                    self.xn,
                    lw.wk,
                    l16.map_or(std::ptr::null_mut(), |l| l.wk),
                    nkv,
                    d,
                )?;
                gemm(
                    self.v_buf,
                    self.xn,
                    lw.wv,
                    l16.map_or(std::ptr::null_mut(), |l| l.wv),
                    nkv,
                    d,
                )?;
                // Qwen2 checkpoints ship q/k/v biases.
                if !lw.bq.is_null() {
                    k.launch_add_bias(self.q, lw.bq, 1, nq)?;
                }
                if !lw.bk.is_null() {
                    k.launch_add_bias(self.k_buf, lw.bk, 1, nkv)?;
                }
                if !lw.bv.is_null() {
                    k.launch_add_bias(self.v_buf, lw.bv, 1, nkv)?;
                }
                // Qwen3 QK-norm: per-head RMSNorm after projection, before RoPE.
                if !lw.q_norm.is_null() {
                    k.launch_qk_norm(
                        self.q,
                        self.k_buf,
                        lw.q_norm,
                        lw.k_norm,
                        1,
                        c.n_heads as i32,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        c.rms_eps,
                    )?;
                }
                k.launch_rope(
                    self.q,
                    self.k_buf,
                    self.dev_pos,
                    c.n_heads as i32,
                    c.n_kv_heads as i32,
                    c.head_dim as i32,
                    c.rope_theta,
                )?;

                let (kc, vc) = self.kv_cache[li];
                k.launch_kv_store(
                    self.k_buf,
                    kc,
                    self.dev_pos,
                    c.n_kv_heads as i32,
                    c.head_dim as i32,
                    c.max_seq_len as i32,
                )?;
                k.launch_kv_store(
                    self.v_buf,
                    vc,
                    self.dev_pos,
                    c.n_kv_heads as i32,
                    c.head_dim as i32,
                    c.max_seq_len as i32,
                )?;

                k.launch_attn_decode(
                    self.q,
                    kc,
                    vc,
                    self.attn,
                    self.dev_pos,
                    c.n_heads as i32,
                    c.n_kv_heads as i32,
                    c.head_dim as i32,
                    scale,
                    c.max_seq_len as i32,
                )?;
                gemm(
                    self.proj,
                    self.attn,
                    lw.wo,
                    l16.map_or(std::ptr::null_mut(), |l| l.wo),
                    d,
                    nq,
                )?;
                k.launch_add(self.x, self.proj, d)?;
            }

            let inter = c.intermediate_size as i32;
            let einter = c.expert_size() as i32;
            k.launch_rms_norm(self.x, lw.rms_mlp, self.xn2, 1, d, c.rms_eps)?;
            if !lw.moe_router.is_null() {
                let ne = c.num_experts as i32;
                let topk = c.num_experts_per_tok.min(c.num_experts) as i32;
                if topk > 0 {
                    // Router logits [ne] = xn2 @ moe_router^T.
                    k.gemm(self.router, self.xn2, lw.moe_router, ne, d)?;
                    k.launch_moe_router(self.router, self.exp_ids, self.exp_w, ne, topk)?;
                    if self.expert_slots < ne as usize {
                        self.forward_moe_slots(lw, li, ne, topk, d, einter)?;
                    } else if self.gpu_budget < topk as usize {
                        self.forward_moe_offload(lw, li, ne, topk, d, einter)?;
                    } else {
                        // Pack the selected experts into contiguous scratch (f32 path;
                        // fp16 MoE weights are a later slice).
                        k.launch_moe_gather_weights(
                            lw.moe_wg,
                            lw.moe_wu,
                            lw.moe_wd,
                            self.exp_ids,
                            self.wg_pack,
                            self.wu_pack,
                            self.wd_pack,
                            ne,
                            einter,
                            d,
                            topk,
                        )?;
                        // Concatenated per-expert gate/up GEMMs over the topk slots.
                        k.gemm(self.gate_all, self.xn2, self.wg_pack, topk * einter, d)?;
                        k.gemm(self.up_all, self.xn2, self.wu_pack, topk * einter, d)?;
                        k.launch_silu_mul(self.up_all, self.gate_all, self.eh_all, topk * einter)?;
                        // Per-slot down projections: each slot has its own hidden
                        // state, so the concat-GEMM trick (shared input) does not
                        // apply here — launch one small GEMM per selected expert.
                        for slot in 0..topk {
                            k.gemm(
                                unsafe { self.down_all.add((slot * d) as usize) },
                                unsafe { self.eh_all.add((slot * einter) as usize) },
                                unsafe { self.wd_pack.add((slot * d * einter) as usize) },
                                d,
                                einter,
                            )?;
                        }
                        k.launch_moe_accumulate(self.x, self.down_all, self.exp_w, d, topk)?;
                    }
                }
                // topk == 0: MoE contributes nothing (matches ref_model).
            } else {
                gemm(
                    self.gate,
                    self.xn2,
                    lw.wg,
                    l16.map_or(std::ptr::null_mut(), |l| l.wg),
                    inter,
                    d,
                )?;
                gemm(
                    self.up,
                    self.xn2,
                    lw.wu,
                    l16.map_or(std::ptr::null_mut(), |l| l.wu),
                    inter,
                    d,
                )?;
                // SwiGLU: h = silu(gate) * up, so silu applies to `gate`.
                k.launch_silu_mul(self.up, self.gate, self.h, inter)?;
                gemm(
                    self.proj,
                    self.h,
                    lw.wd,
                    l16.map_or(std::ptr::null_mut(), |l| l.wd),
                    d,
                    inter,
                )?;
                k.launch_add(self.x, self.proj, d)?;
            }
        }
        k.launch_rms_norm(self.x, self.rms_final_dev, self.xn, 1, d, c.rms_eps)?;
        if f16 {
            if d <= crate::batched::GEMV_MAX_D {
                k.launch_gemv_f16(
                    self.logits,
                    self.xn,
                    self.lm_head_f16,
                    c.vocab_size as i32,
                    d,
                    1,
                )?;
            } else {
                k.gemm_f16(
                    self.logits,
                    self.xn,
                    self.lm_head_f16,
                    c.vocab_size as i32,
                    d,
                    self.xh,
                    self.yh,
                )?;
            }
        } else {
            k.gemm(
                self.logits,
                self.xn,
                self.lm_head_dev,
                c.vocab_size as i32,
                d,
            )?;
        }
        Ok(())
    }

    /// Reads logits back to host after a stream sync.
    fn read_logits(&self) -> Result<Vec<f32>, Error> {
        self.k.sync()?;
        let mut out = vec![0.0f32; self.cfg.vocab_size];
        hip::memcpy(
            self.k.hip(),
            out.as_mut_ptr() as *mut core::ffi::c_void,
            self.logits as *const core::ffi::c_void,
            out.len() * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )?;
        Ok(out)
    }

    /// Eager decode step without logits readback: update inputs, launch all
    /// kernels on the stream. Ordering is guaranteed by the stream, so a whole
    /// sequence can run without per-token sync (readback once at the end).
    pub fn step_eager(&mut self, token: u32) -> Result<(), Error> {
        if self.pos >= self.cfg.max_seq_len {
            return Err(Error::Model("sequence length exceeded".into()));
        }
        self.update_inputs(token)?;
        self.run_kernels()?;
        self.pos += 1;
        Ok(())
    }

    /// Graph-driven step without logits readback (see [`step_eager`]).
    pub fn step_graph(&mut self, graph: &dyn GraphHandle, token: u32) -> Result<(), Error> {
        if self.pos >= self.cfg.max_seq_len {
            return Err(Error::Model("sequence length exceeded".into()));
        }
        self.update_inputs(token)?;
        // SAFETY: inputs updated on the capture stream before the replay; the
        // caller syncs before any readback.
        unsafe { graph.replay()? };
        self.pos += 1;
        Ok(())
    }

    /// One eager decode step for `token` at position `self.pos`.
    /// Builds a GPU model with a MoE offload budget: at most `gpu_budget` of the
    /// top-k routed experts are computed on the GPU per step; the rest fall back
    /// to the CPU reference. Keeps a host weight copy for the CPU fallback.
    pub fn with_gpu_budget(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        gpu_budget: usize,
    ) -> Result<Self, Error> {
        let mut m = Self::new(hip, cfg, w)?;
        m.gpu_budget = gpu_budget;
        m.host_w = Some(Arc::new(w.clone()));
        m.alloc_offload_pins()?;
        Ok(m)
    }

    /// Pinned host staging for the offload paths' async D2H/H2D read-backs
    /// (ids/weights/xn2/xh): hipMemcpyAsync on ordinary heap would fall back
    /// to synchronous copies and block the host per copy. Called by the
    /// offload constructors; full-resident builds keep the fields null.
    fn alloc_offload_pins(&mut self) -> Result<(), Error> {
        let topk = self.cfg.num_experts_per_tok.min(self.cfg.num_experts);
        let d = self.cfg.d_model;
        let hip = self.k.hip();
        let mut pin = |bytes: usize| -> Result<*mut core::ffi::c_void, Error> {
            let b = hip::host_malloc(hip, bytes)?;
            self.host_pins.push(b);
            Ok(b)
        };
        self.offload_ids = pin(topk * 4)? as *mut i32;
        self.offload_w = pin(topk * 4)? as *mut f32;
        self.offload_xn2 = pin(d * 4)? as *mut f32;
        self.offload_xh = pin(d * 4)? as *mut f32;
        Ok(())
    }

    /// Builds a GPU model with expert_slots GPU-resident expert slots per MoE layer;
    /// experts beyond the slots are kept in host RAM and fetched on demand through
    /// the LRU cache.
    pub fn with_expert_slots(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        expert_slots: usize,
    ) -> Result<Self, Error> {
        let mut m = Self::build(Arc::clone(&hip), cfg, w, expert_slots)?;
        m.host_w = Some(Arc::new(w.clone()));
        m.alloc_offload_pins()?;
        Ok(m)
    }

    /// MoE step with a GPU/CPU split: the first `gpu_budget` routed experts are
    /// computed on GPU via the gather+GEMM path; the rest are computed on the CPU
    /// reference and added back. The output is placement-invariant.
    fn forward_moe_offload(
        &self,
        lw: &LayerDev,
        li: usize,
        ne: i32,
        topk: i32,
        d: i32,
        inter: i32,
    ) -> Result<(), Error> {
        let k = self.k.clone();
        let gpu_n = self.gpu_budget.min(topk as usize) as i32;
        // Everything enqueued so far (the layer's attention + router) is done
        // after this event fires: the copy stream's D2H reads wait on it
        // instead of a full stream sync, so the GPU-resident GEMMs below run
        // concurrently with the host-side transfers and CPU experts.
        let hip = self.k.hip();
        unsafe {
            hip::check(
                hip,
                (hip.api.hip_event_record)(self.router_done, self.k.stream),
            )?;
        }

        // GPU-resident part: first gpu_n routed experts, existing gather+GEMM.
        // `gpu_n` depends only on the budget, not on the router ids, so this
        // is enqueued before the ids are read back.
        if gpu_n > 0 {
            k.launch_moe_gather_weights(
                lw.moe_wg,
                lw.moe_wu,
                lw.moe_wd,
                self.exp_ids,
                self.wg_pack,
                self.wu_pack,
                self.wd_pack,
                ne,
                inter,
                d,
                gpu_n,
            )?;
            k.gemm(self.gate_all, self.xn2, self.wg_pack, gpu_n * inter, d)?;
            k.gemm(self.up_all, self.xn2, self.wu_pack, gpu_n * inter, d)?;
            k.launch_silu_mul(self.up_all, self.gate_all, self.eh_all, gpu_n * inter)?;
            for slot in 0..gpu_n {
                k.gemm(
                    unsafe { self.down_all.add((slot * d) as usize) },
                    unsafe { self.eh_all.add((slot * inter) as usize) },
                    unsafe { self.wd_pack.add((slot * d * inter) as usize) },
                    d,
                    inter,
                )?;
            }
            k.launch_moe_accumulate(self.x, self.down_all, self.exp_w, d, gpu_n)?;
            // The residual upload and the post-accumulate `x` read wait on
            // this before touching `x`.
            unsafe {
                hip::check(
                    hip,
                    (hip.api.hip_event_record)(self.gpu_part_done, self.k.stream),
                )?;
            }
        }

        // CPU fallback: routed experts beyond gpu_n.
        let n_cpu = topk - gpu_n;
        if n_cpu > 0 {
            // Read the top-k ids + weights and the attention output back to
            // host on the copy stream — it waits only for the router, so the
            // GPU-resident GEMMs above keep running underneath. Targets are
            // the model's pinned staging (async copies need pinned host
            // memory, else hipMemcpyAsync falls back to synchronous).
            let s = self.xfer_stream;
            unsafe {
                hip::check(hip, (hip.api.hip_stream_wait_event)(s, self.router_done, 0))?;
            }
            hip::memcpy_async(
                hip,
                self.offload_ids as *mut core::ffi::c_void,
                self.exp_ids as *const core::ffi::c_void,
                (topk as usize) * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
                s,
            )?;
            hip::memcpy_async(
                hip,
                self.offload_w as *mut core::ffi::c_void,
                self.exp_w as *const core::ffi::c_void,
                (topk as usize) * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
                s,
            )?;
            hip::memcpy_async(
                hip,
                self.offload_xn2 as *mut core::ffi::c_void,
                self.xn2 as *const core::ffi::c_void,
                (d as usize) * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
                s,
            )?;
            unsafe {
                hip::check(hip, (hip.api.hip_stream_synchronize)(s))?;
            }

            let host_w = self
                .host_w
                .as_ref()
                .ok_or_else(|| Error::Model("offload CPU path requires host weights".into()))?;
            let lw_h = &host_w.layers[li];

            // Pinned staging read back above: id/weight/xn2 slices.
            let ids = unsafe { std::slice::from_raw_parts(self.offload_ids, topk as usize) };
            let weights = unsafe { std::slice::from_raw_parts(self.offload_w, topk as usize) };
            let xn2 = unsafe { std::slice::from_raw_parts(self.offload_xn2, d as usize) };

            let mut residual = vec![0.0f32; d as usize];
            for i in (gpu_n as usize)..(topk as usize) {
                let e = ids[i] as usize;
                let w = weights[i];
                let (inter_us, d_us) = (inter as usize, d as usize);
                let wg = &lw_h.moe_wg[e * inter_us * d_us..(e + 1) * inter_us * d_us];
                let wu = &lw_h.moe_wu[e * inter_us * d_us..(e + 1) * inter_us * d_us];
                let wd = &lw_h.moe_wd[e * d_us * inter_us..(e + 1) * d_us * inter_us];
                let down = moe_offload::expert_mlp(xn2, wg, wu, wd, inter_us, d_us);
                for kk in 0..d_us {
                    residual[kk] += w * down[kk];
                }
            }

            // Read the POST-accumulate `x` (the GPU part's contribution is
            // already folded in) and upload the residual back: both wait for
            // `gpu_part_done`, by which time the accumulate above has long
            // finished under the CPU work.
            let xh = unsafe { std::slice::from_raw_parts_mut(self.offload_xh, d as usize) };
            unsafe {
                hip::check(
                    hip,
                    (hip.api.hip_stream_wait_event)(s, self.gpu_part_done, 0),
                )?;
            }
            hip::memcpy_async(
                hip,
                self.offload_xh as *mut core::ffi::c_void,
                self.x as *const core::ffi::c_void,
                (d as usize) * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
                s,
            )?;
            unsafe {
                hip::check(hip, (hip.api.hip_stream_synchronize)(s))?;
            }
            for kk in 0..d as usize {
                xh[kk] += residual[kk];
            }
            hip::memcpy_async(
                hip,
                self.x as *mut core::ffi::c_void,
                self.offload_xh as *const core::ffi::c_void,
                (d as usize) * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
                s,
            )?;
            unsafe {
                hip::check(hip, (hip.api.hip_stream_synchronize)(s))?;
            }
        }
        Ok(())
    }
    /// Builds a GPU model in adaptive (q*) mode: measures the machine PCIe
    /// bandwidth and CPU expert cost at init, and per miss decides whether to
    /// fetch an expert to a GPU slot or compute it on the CPU (the BandwidthProbe).
    pub fn with_adaptive(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        expert_slots: usize,
    ) -> Result<Self, Error> {
        let mut m = Self::build(Arc::clone(&hip), cfg, w, expert_slots)?;
        m.host_w = Some(Arc::new(w.clone()));
        let a = BandwidthProbe::measure(&hip, &cfg)?.profile;
        m.adaptive = Some(AdaptiveProfile::new(
            a.pcie_bytes_per_sec,
            a.cpu_expert_sec,
            0.9,
        ));
        m.alloc_offload_pins()?;
        Ok(m)
    }

    /// Re-measures PCIe bandwidth + CPU expert cost and folds the sample into the
    /// adaptive (q*) profile, so a contended bus shifts the per-miss decision to
    /// CPU. Call periodically from the serving loop or via `set_reprobe_every`.
    pub fn reprobe_bandwidth(&mut self) -> Result<(), Error> {
        if let Some(prof) = &mut self.adaptive {
            let a = BandwidthProbe::measure(self.k.hip(), &self.cfg)?.profile;
            prof.observe(a.pcie_bytes_per_sec);
        }
        Ok(())
    }
    /// Sets the auto re-probe cadence: every `n` decode steps, re-measure PCIe
    /// bandwidth and fold it into the adaptive profile. `0` disables.
    pub fn set_reprobe_every(&mut self, n: usize) {
        self.reprobe_every = n;
    }

    #[must_use]
    fn should_reprobe(counter: usize, every: usize) -> bool {
        every > 0 && counter > 0 && counter.is_multiple_of(every)
    }

    /// Advances the step counter and re-probes bandwidth on the cadence (if the
    /// adaptive profile is present). Real-time q*: a contended bus shifts the
    /// per-miss decision to CPU on the following steps.
    fn maybe_reprobe(&mut self) -> Result<(), Error> {
        self.step_counter += 1;
        if Self::should_reprobe(self.step_counter, self.reprobe_every) {
            self.reprobe_bandwidth()?;
        }
        Ok(())
    }
    /// Places a step routed experts into GPU slots (bounded by `cap`), applying
    /// the adaptive q* choice when `adaptive` is provided. Returns (to_upload,
    /// slot_list, gpu_w, cpu): experts to copy into slots, slot indices for the GPU
    /// gather (in routed order), their weights, and the CPU fallback.
    #[allow(clippy::type_complexity)]
    fn moe_slot_place(
        &self,
        li: usize,
        ids: &[i32],
        weights: &[f32],
        adaptive: Option<BandwidthProfile>,
        expert_bytes: usize,
    ) -> (Vec<(usize, usize)>, Vec<i32>, Vec<f32>, Vec<(usize, f32)>) {
        let mut to_upload: Vec<(usize, usize)> = Vec::new();
        let mut slot_list: Vec<i32> = Vec::new();
        let mut gpu_w: Vec<f32> = Vec::new();
        let mut cpu: Vec<(usize, f32)> = Vec::new();
        let mut ctx = self.slot_ctx[li].borrow_mut();
        let cap = ctx.cap;
        let routed_set: HashSet<u32> = ids.iter().map(|&e| e as u32).collect();
        for (i, &e) in ids.iter().enumerate() {
            let e = e as usize;
            let w = weights[i];
            let id = e as u32;
            if let Some(slot) = ctx.lru.get(id) {
                slot_list.push(slot as i32);
                gpu_w.push(w);
            } else if adaptive
                .is_some_and(|p| p.choose(expert_bytes) == crate::adaptive::FetchChoice::ComputeCpu)
            {
                cpu.push((e, w));
            } else if ctx.lru.len() < cap || ctx.lru.evict_lru_not_in(&routed_set).is_some() {
                let put = ctx.lru.put(id);
                if ctx.slot_expert[put.slot] != e as i32 {
                    ctx.slot_expert[put.slot] = e as i32;
                    to_upload.push((put.slot, e));
                }
                slot_list.push(put.slot as i32);
                gpu_w.push(w);
            } else {
                cpu.push((e, w));
            }
        }
        (to_upload, slot_list, gpu_w, cpu)
    }
    /// MoE step with bounded GPU-resident expert slots: experts stay in host RAM
    /// and are fetched on demand into a fixed set of per-layer slots via the LRU
    /// cache (`plan_step`); misses beyond capacity are computed on the CPU. The
    /// output is placement-invariant and the GPU-resident footprint is bounded by
    /// `expert_slots` per layer, not by the total expert count.
    fn forward_moe_slots(
        &self,
        _lw: &LayerDev,
        li: usize,
        _ne: i32,
        topk: i32,
        d: i32,
        inter: i32,
    ) -> Result<(), Error> {
        let k = self.k.clone();
        // The placement below needs the router ids on the host, but the reads
        // only wait for the router (copy stream), never draining the compute
        // stream — the GPU-resident GEMMs below overlap the CPU fallback.
        let hip = self.k.hip();
        unsafe {
            hip::check(
                hip,
                (hip.api.hip_event_record)(self.router_done, self.k.stream),
            )?;
        }
        let s = self.xfer_stream;
        unsafe {
            hip::check(hip, (hip.api.hip_stream_wait_event)(s, self.router_done, 0))?;
        }
        // Read the top-k ids + weights back to host into the pinned staging
        // (async copies need pinned host memory, else hipMemcpyAsync falls
        // back to synchronous).
        hip::memcpy_async(
            hip,
            self.offload_ids as *mut core::ffi::c_void,
            self.exp_ids as *const core::ffi::c_void,
            (topk as usize) * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
            s,
        )?;
        hip::memcpy_async(
            hip,
            self.offload_w as *mut core::ffi::c_void,
            self.exp_w as *const core::ffi::c_void,
            (topk as usize) * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
            s,
        )?;
        unsafe {
            hip::check(hip, (hip.api.hip_stream_synchronize)(s))?;
        }
        let ids = unsafe { std::slice::from_raw_parts(self.offload_ids, topk as usize) };
        let weights = unsafe { std::slice::from_raw_parts(self.offload_w, topk as usize) };

        let host_w = self
            .host_w
            .as_ref()
            .ok_or_else(|| Error::Model("offload slots require host weights".into()))?;
        let lw_h = &host_w.layers[li];
        let (ctx_wg, ctx_wu, ctx_wd, cap) = {
            let ctx = self.slot_ctx[li].borrow();
            (ctx.wg, ctx.wu, ctx.wd, ctx.cap)
        };

        // Decide placement: fill free slots, overflow to CPU, no intra-step eviction;
        // in adaptive mode a miss is computed on the CPU when it is cheaper than the
        // PCIe fetch pull for this machine (q*).
        let expert_bytes = 3 * inter as usize * d as usize * 4;
        let (to_upload, slot_list, gpu_w, cpu) = self.moe_slot_place(
            li,
            ids,
            weights,
            self.adaptive.as_ref().map(|p| p.profile()),
            expert_bytes,
        );

        // Upload experts that changed slots (on-demand fetch from host RAM).
        if !to_upload.is_empty() {
            let (inter_us, d_us) = (inter as usize, d as usize);
            for (s, e) in &to_upload {
                let wg = &lw_h.moe_wg[e * inter_us * d_us..(e + 1) * inter_us * d_us];
                let wu = &lw_h.moe_wu[e * inter_us * d_us..(e + 1) * inter_us * d_us];
                let wd = &lw_h.moe_wd[e * d_us * inter_us..(e + 1) * d_us * inter_us];
                self.upload(unsafe { ctx_wg.add(s * inter_us * d_us) }, wg)?;
                self.upload(unsafe { ctx_wu.add(s * inter_us * d_us) }, wu)?;
                self.upload(unsafe { ctx_wd.add(s * d_us * inter_us) }, wd)?;
            }
        }

        // GPU-resident part: gather from the slot buffer by slot index.
        let gpu_count = slot_list.len() as i32;
        if gpu_count > 0 {
            hip::memcpy(
                hip,
                self.slot_ids_dev as *mut core::ffi::c_void,
                slot_list.as_ptr() as *const core::ffi::c_void,
                (gpu_count as usize) * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )?;
            hip::memcpy(
                hip,
                self.slot_w_dev as *mut core::ffi::c_void,
                gpu_w.as_ptr() as *const core::ffi::c_void,
                (gpu_count as usize) * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )?;
            k.launch_moe_gather_weights(
                ctx_wg,
                ctx_wu,
                ctx_wd,
                self.slot_ids_dev,
                self.wg_pack,
                self.wu_pack,
                self.wd_pack,
                cap as i32,
                inter,
                d,
                gpu_count,
            )?;
            k.gemm(self.gate_all, self.xn2, self.wg_pack, gpu_count * inter, d)?;
            k.gemm(self.up_all, self.xn2, self.wu_pack, gpu_count * inter, d)?;
            k.launch_silu_mul(self.up_all, self.gate_all, self.eh_all, gpu_count * inter)?;
            for slot in 0..gpu_count {
                k.gemm(
                    unsafe { self.down_all.add((slot * d) as usize) },
                    unsafe { self.eh_all.add((slot * inter) as usize) },
                    unsafe { self.wd_pack.add((slot * d * inter) as usize) },
                    d,
                    inter,
                )?;
            }
            k.launch_moe_accumulate(self.x, self.down_all, self.slot_w_dev, d, gpu_count)?;
            unsafe {
                hip::check(
                    hip,
                    (hip.api.hip_event_record)(self.gpu_part_done, self.k.stream),
                )?;
            }
        }

        // CPU fallback: routed experts that did not fit into a slot. The reads
        // and CPU work overlap the GPU-resident part above (they wait only for
        // the router / the accumulate, not the full stream).
        if !cpu.is_empty() {
            let (inter_us, d_us) = (inter as usize, d as usize);
            unsafe {
                hip::check(hip, (hip.api.hip_stream_wait_event)(s, self.router_done, 0))?;
            }
            hip::memcpy_async(
                hip,
                self.offload_xn2 as *mut core::ffi::c_void,
                self.xn2 as *const core::ffi::c_void,
                d_us * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
                s,
            )?;
            unsafe {
                hip::check(hip, (hip.api.hip_stream_synchronize)(s))?;
            }
            let xn2 = unsafe { std::slice::from_raw_parts(self.offload_xn2, d_us) };
            let mut residual = vec![0.0f32; d_us];
            for (e, w) in &cpu {
                let wg = &lw_h.moe_wg[e * inter_us * d_us..(e + 1) * inter_us * d_us];
                let wu = &lw_h.moe_wu[e * inter_us * d_us..(e + 1) * inter_us * d_us];
                let wd = &lw_h.moe_wd[e * d_us * inter_us..(e + 1) * d_us * inter_us];
                let down = moe_offload::expert_mlp(xn2, wg, wu, wd, inter_us, d_us);
                for kk in 0..d_us {
                    residual[kk] += w * down[kk];
                }
            }
            let xh = unsafe { std::slice::from_raw_parts_mut(self.offload_xh, d_us) };
            unsafe {
                hip::check(
                    hip,
                    (hip.api.hip_stream_wait_event)(s, self.gpu_part_done, 0),
                )?;
            }
            hip::memcpy_async(
                hip,
                self.offload_xh as *mut core::ffi::c_void,
                self.x as *const core::ffi::c_void,
                d_us * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
                s,
            )?;
            unsafe {
                hip::check(hip, (hip.api.hip_stream_synchronize)(s))?;
            }
            for kk in 0..d_us {
                xh[kk] += residual[kk];
            }
            hip::memcpy_async(
                hip,
                self.x as *mut core::ffi::c_void,
                self.offload_xh as *const core::ffi::c_void,
                d_us * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
                s,
            )?;
            unsafe {
                hip::check(hip, (hip.api.hip_stream_synchronize)(s))?;
            }
        }
        Ok(())
    }
    pub fn decode_step(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        if self.pos >= self.cfg.max_seq_len {
            return Err(Error::Model("sequence length exceeded".into()));
        }
        self.update_inputs(token)?;
        self.run_kernels()?;
        let out = self.read_logits()?;
        self.maybe_reprobe()?;
        self.pos += 1;
        Ok(out)
    }

    /// Syncs the model stream (debug/measurement).
    pub fn sync(&self) -> Result<(), Error> {
        self.k.sync()
    }

    /// One decode step returning the greedy-sampled next token, reading back
    /// only 4 bytes instead of the full logits vector.
    pub fn decode_step_sampled(&mut self, token: u32) -> Result<u32, Error> {
        if self.pos >= self.cfg.max_seq_len {
            return Err(Error::Model("sequence length exceeded".into()));
        }
        self.update_inputs(token)?;
        self.run_kernels()?;
        let next = self.sampler.argmax(self.logits, self.cfg.vocab_size)?;
        self.pos += 1;
        Ok(next)
    }

    /// Runs `tokens` one by one and returns logits of the final token.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        let mut logits = Vec::new();
        for &t in tokens {
            logits = self.decode_step(t)?;
        }
        Ok(logits)
    }

    /// Zeroes the KV cache and resets the position (before graph capture).
    pub fn reset_state(&mut self) -> Result<(), Error> {
        let bytes = self.cfg.max_seq_len * self.cfg.n_kv_heads * self.cfg.head_dim * 4;
        for (kc, vc) in &self.kv_cache {
            unsafe {
                hip::check(
                    self.k.hip(),
                    (self.k.hip().api.hip_memset)(*kc as *mut _, 0, bytes),
                )?;
                hip::check(
                    self.k.hip(),
                    (self.k.hip().api.hip_memset)(*vc as *mut _, 0, bytes),
                )?;
            }
        }
        if !self.mla_kv_cache.is_empty() {
            let c = self.cfg;
            let heads = c.n_heads;
            let k_bytes = c.max_seq_len * heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) * 4;
            let v_bytes = c.max_seq_len * heads * c.v_head_dim * 4;
            for (kc, vc) in &self.mla_kv_cache {
                unsafe {
                    hip::check(
                        self.k.hip(),
                        (self.k.hip().api.hip_memset)(*kc as *mut _, 0, k_bytes),
                    )?;
                    hip::check(
                        self.k.hip(),
                        (self.k.hip().api.hip_memset)(*vc as *mut _, 0, v_bytes),
                    )?;
                }
            }
        }
        unsafe { *self.host_pos = 0 };
        hip::memcpy_async(
            self.k.hip(),
            self.dev_pos as *mut core::ffi::c_void,
            self.host_pos as *const core::ffi::c_void,
            4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.k.stream,
        )?;
        self.k.sync()?;
        self.pos = 0;
        Ok(())
    }

    /// Warms up, resets state, then captures the decode kernel sequence into a
    /// HIP graph. Replays are driven with
    /// [`decode_step_graph`](Self::decode_step_graph).
    pub fn capture_decode(&mut self) -> Result<Box<dyn GraphHandle>, Error> {
        // Warmup: compile everything and let hipBLAS allocate workspace.
        for _ in 0..3 {
            self.update_inputs(0)?;
            self.run_kernels()?;
            self.k.sync()?;
        }
        self.reset_state()?;

        let cap = HipGraphCapture::with_stream(Arc::clone(self.k.hip()), self.k.stream)?;
        cap.prepare()?;
        self.k.sync()?;
        cap.begin()?;
        self.run_kernels()?;
        let graph = cap.end()?;
        Ok(graph)
    }

    /// One graph-driven decode step: update inputs, replay, read logits.
    pub fn decode_step_graph(
        &mut self,
        graph: &dyn GraphHandle,
        token: u32,
    ) -> Result<Vec<f32>, Error> {
        if self.pos >= self.cfg.max_seq_len {
            return Err(Error::Model("sequence length exceeded".into()));
        }
        self.update_inputs(token)?;
        // SAFETY: input buffers are updated on the capture stream before the
        // replay, and the read syncs the stream before touching the output.
        unsafe { graph.replay()? };
        let out = self.read_logits()?;
        self.pos += 1;
        Ok(out)
    }
}

impl Drop for GpuModel {
    fn drop(&mut self) {
        let hip = self.k.hip();
        // Drain any in-flight async transfers before freeing the buffers.
        unsafe {
            let _ = (hip.api.hip_stream_synchronize)(self.xfer_stream);
            let _ = (hip.api.hip_event_destroy)(self.router_done);
            let _ = (hip.api.hip_event_destroy)(self.gpu_part_done);
            let _ = (hip.api.hip_stream_destroy)(self.xfer_stream);
        }
        for &p in &self.allocs {
            let _ = hip::free(hip, p);
        }
        for &p in &self.host_pins {
            let _ = hip::host_free(hip, p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reprobe_cadence() {
        // Disabled (every == 0) never reprobes.
        assert!(!GpuModel::should_reprobe(0, 0));
        assert!(!GpuModel::should_reprobe(5, 0));
        // Enabled: triggers exactly on multiples of `every`.
        assert!(!GpuModel::should_reprobe(0, 10));
        assert!(!GpuModel::should_reprobe(9, 10));
        assert!(GpuModel::should_reprobe(10, 10));
        assert!(GpuModel::should_reprobe(20, 10));
    }
}
