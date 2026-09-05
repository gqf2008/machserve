//! Batched decode: multiple sequences share one forward pass.
//!
//! This is the P2 fix for the single-sequence bottleneck: projections become
//! `m = batch` GEMMs instead of `m = 1` (hipBLAS m=1 is extremely slow), which
//! also enables continuous batching. Each sequence keeps its own length, so
//! sequences at different positions are processed together (attention masks by
//! per-sequence position).

use crate::config::ModelDType;
use crate::fp16::f32_to_f16;
use crate::kernels::HipKernels;
use crate::kernels::RopeParams;
use crate::moe_offload;
use crate::sampling::{BatchedSampler, SampleOutput, SamplingParams};
use crate::{Config, Error, Weights, WeightsFp8, WeightsQ4};
use mach_engine::graph::GraphCapture;
use mach_kernel_sys::hip::{self, Hip, HipEvent};
use std::sync::Arc;

/// Per-layer device weight pointers (same layout as the single-seq model).
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
    /// Shared experts (DeepSeek-V2): one dense SwiGLU MLP of width
    /// `n_shared_experts * expert_size`, added to the routed experts' sum. Null
    /// (and `shared_size() == 0`) for checkpoints without shared experts.
    shared_wg: *mut f32,
    shared_wu: *mut f32,
    shared_wd: *mut f32,
    /// Whether this layer HAS shared experts — dtype-independent existence
    /// flag. The f32 pointers above are null in every F16 upload path (the
    /// weights live in `LayerDevF16` only), so testing them silently SKIPPED
    /// the shared experts for all F16/Q4/FP8 models (issue #107: DeepSeek-V2
    /// Q4-on-device decoded garbage; Qwen-MoE was unaffected because it ships
    /// no shared experts).
    has_shared: bool,
    /// Gated DeltaNet (Qwen3.5 linear-attention layers). Null on full-attn
    /// layers: `gdn_in_qkv` doubles as the layer-type dispatch marker
    /// (dtype-independent, unlike the f32 pointers — see `has_shared`).
    has_gdn: bool,
    gdn_in_qkv: *mut f32,
    gdn_in_z: *mut f32,
    gdn_in_a: *mut f32,
    gdn_in_b: *mut f32,
    gdn_out: *mut f32,
    /// Depthwise conv taps `[conv_dim, k]` (newest last) — f32 even on the
    /// F16 path (tiny, elementwise).
    gdn_conv_w: *mut f32,
    /// Bare per-v-head scalars `A_log` / `dt_bias` and the shared gated-norm
    /// weight `[head_dim]`.
    gdn_a_log: *mut f32,
    gdn_dt_bias: *mut f32,
    gdn_norm: *mut f32,
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
    /// MoE fp16 (num_experts > 0): router + per-expert gate/up/down.
    moe_router: *mut u16,
    moe_wg: *mut u16,
    moe_wu: *mut u16,
    moe_wd: *mut u16,
    /// Shared experts fp16 (DeepSeek-V2): dense SwiGLU MLP, see `LayerDev`.
    shared_wg: *mut u16,
    shared_wu: *mut u16,
    shared_wd: *mut u16,
    /// MLA fp16 (kv_lora_rank > 0): low-rank Q / compressed KV weights.
    mla_q_a: *mut u16,
    mla_q_b: *mut u16,
    mla_q_rope: *mut u16,
    mla_kv_a: *mut u16,
    mla_kv_b: *mut u16,
    mla_o: *mut u16,
    /// GDN GEMM matrices mirrored in f16 (a/b are [vh, d] matrices, not
    /// small vectors — they contract over d like the other projections).
    gdn_in_qkv: *mut u16,
    gdn_in_z: *mut u16,
    gdn_in_a: *mut u16,
    gdn_in_b: *mut u16,
    gdn_out: *mut u16,
}

/// Storage-Q4 expert weights on device (Q4-on-device mode): the raw packed
/// int4 bytes + per-group f32 scales per tensor, read with in-kernel dequant
/// by the Q4 grouped GEMV kernels. No f16/f32 device copy of the expert pool
/// — a 30B-class checkpoint's ~16GB of Q4 experts fits in VRAM where the
/// dequantized f16 pool (~50GB) would not.
struct Q4ExpertDev {
    wg_q: *mut u8,
    wg_s: *mut f32,
    wu_q: *mut u8,
    wu_s: *mut f32,
    wd_q: *mut u8,
    wd_s: *mut f32,
}

/// One Q4 GEMM tensor on device: the raw packed int4 bytes + per-group f32
/// scales, dequantized in-kernel by `gemv_q4` / `embed_gather_q4` (dense
/// Q4-on-device mode).
#[derive(Clone, Copy)]
struct Q4TensorDev {
    q: *mut u8,
    s: *mut f32,
}

impl Q4TensorDev {
    const EMPTY: Self = Self {
        q: std::ptr::null_mut(),
        s: std::ptr::null_mut(),
    };

    fn is_null(&self) -> bool {
        self.q.is_null()
    }
}

/// Dense (non-expert) Q4 weights per layer (dense Q4-on-device mode,
/// `MACH_Q4_DEVICE=2`): every big GEMM tensor stays raw Q4 on device — the
/// memory path for DENSE 27B-class checkpoints, where the dequantized f16
/// copy (~54GB) does not fit VRAM but the all-Q4 pool (~13.5GB) does. Each
/// GEMV runs the in-kernel-dequant `gemv_q4` for EVERY step (no f16/f32
/// copy of these tensors exists to fall back on — same contract as the Q4
/// expert pool). Tensors a layer type does not carry keep null pairs;
/// layer-type dispatch itself stays dtype-independent (`has_gdn` /
/// `moe_router`, the #107 lesson). Unquantized tensors (router, GDN a/b)
/// and the small always-dequantized shared MLP keep their f16 copies.
struct Q4DenseDev {
    wq: Q4TensorDev,
    wk: Q4TensorDev,
    wv: Q4TensorDev,
    wo: Q4TensorDev,
    wg: Q4TensorDev,
    wu: Q4TensorDev,
    wd: Q4TensorDev,
    gdn_in_qkv: Q4TensorDev,
    gdn_in_z: Q4TensorDev,
    gdn_out: Q4TensorDev,
}

/// Step profiler state (`Config.step_profile`, parsed from the server's
/// `MACH_STEP_PROFILE`): per-layer event pairs bracketing [layer start,
/// attention done, MoE done] plus the whole-step pair. `report` prints the
/// per-step phase breakdown after the sampling sync.
struct ProfState {
    hip: std::sync::Arc<Hip>,
    /// `[layer] -> (layer_start, attn_done, moe_done)`.
    events: Vec<(HipEvent, HipEvent, HipEvent)>,
    /// Whole-step (start, end).
    step_events: (HipEvent, HipEvent),
    /// Decoded-step counter for the report line.
    step: usize,
}

impl ProfState {
    fn new(n_layers: usize, hip: &std::sync::Arc<Hip>) -> Result<Self, crate::Error> {
        // NOTE: if event creation fails midway, the already-created handles
        // leak (Self is never constructed, so Drop does not run). Accepted:
        // this only happens on a dying HIP context, where the process exits
        // anyway; no guard is added for the catastrophic path.
        let mk = || -> Result<HipEvent, crate::Error> {
            let mut e = std::ptr::null_mut();
            unsafe { hip::check(hip, (hip.api.hip_event_create)(&mut e))? };
            Ok(e)
        };
        let mut events = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            events.push((mk()?, mk()?, mk()?));
        }
        Ok(Self {
            hip: hip.clone(),
            events,
            step_events: (mk()?, mk()?),
            step: 0,
        })
    }

    /// Prints the per-step phase breakdown (call after the step's sync):
    /// per-layer event pairs give attention vs MoE vs other, plus the
    /// whole-step wall time from the step event pair.
    fn report(&mut self) {
        let mut attn = 0f32;
        let mut moe = 0f32;
        for (l0, l1, l2) in &self.events {
            let mut a = 0f32;
            unsafe {
                let _ = (self.hip.api.hip_event_elapsed_time)(&mut a, *l0, *l1);
            }
            attn += a;
            unsafe {
                let _ = (self.hip.api.hip_event_elapsed_time)(&mut a, *l1, *l2);
            }
            moe += a;
        }
        let mut total = 0f32;
        unsafe {
            let _ = (self.hip.api.hip_event_elapsed_time)(
                &mut total,
                self.step_events.0,
                self.step_events.1,
            );
        }
        eprintln!(
            "[prof] step {}: attn={attn:.2}ms moe={moe:.2}ms other={:.2}ms total={total:.2}ms",
            self.step,
            (total - attn - moe).max(0.0)
        );
        self.step += 1;
    }
}

impl Drop for ProfState {
    fn drop(&mut self) {
        unsafe {
            let flat = self
                .events
                .iter()
                .flat_map(|(a, b, c)| [*a, *b, *c])
                .chain([self.step_events.0, self.step_events.1]);
            for e in flat {
                let _ = (self.hip.api.hip_event_destroy)(e);
            }
        }
    }
}

/// Max m (active rows) dispatched to the custom GEMV kernel: below this the
/// rocBLAS m-small kernels waste 60x over the memory bound (measured on the
/// 30B decode); above it, weight reuse across rows makes hipBLAS tensor-core
/// GEMMs win (a 512-row prefill re-reads the weights m× through the GEMV).
const GEMV_MAX_M: i32 = 8;
/// Max reduction width the GEMV kernel can stage in its 48 KB shared row
/// buffer (`d * 4` bytes): dispatch sites must fall back to hipBLAS above
/// this (e.g. f16 MLA o_proj with DeepSeek-V2-level kk = 128 heads × 192).
pub(crate) const GEMV_MAX_D: i32 = 12_288;

/// Multi-sequence transformer on the GPU.
pub struct BatchedModel {
    cfg: Config,
    /// KV slot capacity (concurrent sequences).
    batch: usize,
    /// Row capacity (>= batch): prefill may pack more prompt positions than
    /// there are slots; all rows of one sequence share its slot.
    rows: usize,
    k: Arc<HipKernels>,
    sampler: BatchedSampler,
    // device inputs
    tokens_dev: *mut i32,
    pos_dev: *mut i32,
    // pinned host inputs
    tokens_host: *mut i32,
    pos_host: *mut i32,
    /// Per-row KV cache slot (row index != slot during chunked prefill).
    slots_host: *mut i32,
    slots_dev: *mut i32,
    /// Prefill-attention run descriptors `[qoff, count, base, slot] x N` and
    /// per-row mask (1 = row covered by a run -> prefill attention).
    runs_host: *mut i32,
    runs_dev: *mut i32,
    run_mask_host: *mut i32,
    run_mask_dev: *mut i32,
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
    out_tok_dev: *mut i32,
    out_tok_host: *mut i32,
    // weights (f32; fp16 copies when cfg.dtype == F16)
    emb_dev: *mut f32,
    rms_final_dev: *mut f32,
    lm_head_dev: *mut f32,
    layers_dev: Vec<LayerDev>,
    emb_f16: *mut u16,
    lm_head_f16: *mut u16,
    layers_f16: Vec<LayerDevF16>,
    /// fp16 scratch for GEMM A operands (batch * max(d_model, intermediate)).
    xh: *mut u16,
    /// fp16 scratch for hidden GEMM outputs (batch * max hidden n).
    yh: *mut u16,
    // MoE scratch (grouped per-expert GEMM; sized by rows * topk capacity).
    router: *mut f32,
    exp_ids: *mut i32,
    exp_w: *mut f32,
    counts_dev: *mut i32,
    moe_pos_dev: *mut i32,
    offsets_dev: *mut i32,
    counts_host: *mut i32,
    xg: *mut f32,
    gw: *mut f32,
    row_idx: *mut i32,
    h_acc: *mut f32,
    gate_all: *mut f32,
    up_all: *mut f32,
    eh_all: *mut f32,
    down_all: *mut f32,
    /// fp16 scratch for grouped MoE GEMMs (rows * topk * max(d, inter)).
    xh_moe: *mut u16,
    yh_moe: *mut u16,
    // MLA scratch (kv_lora_rank > 0): low-rank Q / compressed KV activations.
    mla_q_lora: *mut f32,
    mla_q_lora_n: *mut f32,
    q_nope: *mut f32,
    q_rope: *mut f32,
    mla_kv_a: *mut f32,
    mla_kv_a_n: *mut f32,
    mla_kv_lora: *mut f32,
    mla_kv: *mut f32,
    mla_k_rope: *mut f32,
    mla_attn: *mut f32,
    // Qwen3.5 `attn_output_gate` scratch: the doubled q_proj output before
    // the per-head [query | gate] split, and the gate.
    qg: *mut f32,
    attn_gate: *mut f32,
    // Qwen3.5 GDN decode scratch (row-capacity sized).
    gdn_qkv_pre: *mut f32,
    gdn_qkv: *mut f32,
    gdn_z: *mut f32,
    gdn_a: *mut f32,
    gdn_b: *mut f32,
    gdn_o: *mut f32,
    /// Per GDN layer recurrent `[batch, vh, kd, vd]` / conv `[batch,
    /// conv_dim, k-1]` state, SLOT-indexed by the kernels via `slots_dev`;
    /// zeroed at build and in `reset_state` (the pos-0 recurrence reads
    /// S = 0 and an empty conv window; hipMalloc does not zero).
    gdn_state: Vec<*mut f32>,
    gdn_conv: Vec<*mut f32>,
    /// KV caches: (k, v) per layer, layout `[batch, max_seq, kv_heads, head_dim]`.
    /// KV caches as opaque pointers (f32 or fp16 per dtype), layout
    /// `[batch, max_seq, kv_heads, head_dim]`.
    kv_cache: Vec<(*mut core::ffi::c_void, *mut core::ffi::c_void)>,
    /// MLA KV caches (kv_lora_rank > 0): expanded per-head k/v, layout
    /// `[batch, max_seq, heads, hd]` / `[batch, max_seq, heads, v_hd]`.
    mla_kv_cache: Vec<(*mut core::ffi::c_void, *mut core::ffi::c_void)>,
    /// Per-sequence lengths (host).
    lens: Vec<u32>,
    /// Last row index each slot occupied in the most recent forward (for
    /// reading that slot's hidden state back to host — e.g. anchor saving).
    last_row_by_slot: Vec<usize>,
    allocs: Vec<*mut core::ffi::c_void>,
    host_pins: Vec<*mut core::ffi::c_void>,
    /// MoE offload: CPU-backend mode (experts stay in host RAM) when < num_experts.
    expert_slots: usize,
    /// Host weight copy for the CPU MoE compute (offload mode).
    host_w: Option<Arc<Weights>>,
    /// Full-layer double-buffered prefill engine (dedicated prefetch stream +
    /// ping-pong expert buffers). `Some` only in buffered-prefill mode.
    prefetch: Option<crate::prefill_buffered::PrefetchEngine>,
    /// Paged-KV mode: the KV caches are addressed as a page pool via per-slot
    /// block tables (`kv_store_paged` / `attn_decode_paged`) instead of the
    /// contiguous `[slot, max_seq, kv, dim]` layout. Enabled by
    /// [`Self::with_paged_kv`]; the KV allocation itself is unchanged (the
    /// default identity block-table mapping gives the same physical layout and
    /// [`Self::set_block_table`] installs reuse-planner tables on top).
    paged: bool,
    /// Batched-MoE decode uses the device-side grouped GEMV kernels (no
    /// counts readback, no host loop). `MACH_MOE_GROUPED=0` falls back to
    /// the hipBLAS host loop (A/B switch + ops lever). Known limits: a step
    /// containing any prefill row runs the hipBLAS path for the WHOLE step
    /// (mixed decode+prefill steps do not split rows between the two
    /// implementations), and decode on a prefill-buffered model still waits
    /// on the per-layer prefetch stream.
    moe_grouped: bool,
    /// Q4-on-device expert weights per layer (non-empty only in the
    /// `with_rows_q4_device` mode): the grouped GEMV kernels dequantize
    /// in-kernel; all steps (prefill included) run the grouped path, since
    /// no f16/f32 expert copy exists for the hipBLAS path.
    q4_experts: Vec<Q4ExpertDev>,
    /// Dense Q4-on-device weights per layer (non-empty only in the
    /// `with_rows_q4_all` mode): the big dense GEMM tensors stay raw Q4,
    /// read by `gemv_q4` in-kernel dequant for every step. Index-aligned
    /// with `layers_dev` / `layers_f16`.
    q4_layers: Vec<Q4DenseDev>,
    /// Dense Q4-on-device embedding table (`with_rows_q4_all` mode); null
    /// otherwise (the f16 table `emb_f16` is used).
    emb_q4: Q4TensorDev,
    /// Dense Q4-on-device lm_head (`with_rows_q4_all` mode); null otherwise.
    lm_head_q4: Q4TensorDev,
    /// Step profiler (`Config.step_profile`): per-layer event pairs bracketing
    /// the attention and MoE phases plus a whole-step pair; `report` prints
    /// the per-step breakdown after the sampling sync.
    prof: Option<ProfState>,
    /// Tokens per KV page (paged mode).
    tokens_per_page: usize,
    /// Pages per sequence (`max_seq / tokens_per_page`, divisibility required).
    max_pages_per_seq: usize,
    /// Device block tables: `[slots * max_pages_per_seq]` ints; slot `S`'s
    /// logical page `L` sits at `[S*max_pages_per_seq + L]`. Default is the
    /// static identity mapping; `set_block_table` overwrites per-slot entries
    /// (shared-prefix pages alias the same physical ids across slots).
    block_tables: *mut i32,
    /// Host mirror of the device table region (one upload source per slot).
    tables_host: Vec<i32>,
    /// Pinned host mirror of the per-row table offsets, refilled per step
    /// (`offsets[row] = slot[row] * max_pages_per_seq`; prefill rows may all
    /// point at one slot's pages).
    offsets_host: *mut i32,
    /// True when the device offsets buffer may differ from the identity
    /// mapping (an explicit step packed non-identity slots). `decode_step`
    /// refreshes only when dirty — its rows are always identity.
    offsets_dirty: bool,
    /// Device per-row table offsets: `[rows]` ints, `offset[row] = slots[row] *
    /// max_pages_per_seq`, refreshed per step.
    table_offsets: *mut i32,
    /// #103: lazily captured HIP graphs for pure-decode steps, keyed by the
    /// active row count `n` (`1..=GEMV_MAX_M`). Capture folds the whole decode
    /// step (~700 module launches + the input-upload memcpys on this
    /// host-launch-bound platform) into one `hipGraphLaunch`; replay re-reads
    /// the pinned staging buffers, so only the sampler readback (sync + D2H)
    /// stays outside the graph.
    decode_graphs: std::collections::HashMap<i32, Box<dyn mach_engine::graph::GraphHandle>>,
    /// The greedy [`Self::decode_step`] graph (its row count is always
    /// `self.batch`, so a single graph suffices — kept separate from
    /// `decode_graphs`, whose captures embed the full sampler instead of the
    /// argmax readback kernel).
    greedy_graph: Option<Box<dyn mach_engine::graph::GraphHandle>>,
    /// MACH_GRAPH=1 gate for service-chain decode-graph capture (also
    /// switchable via [`Self::set_decode_graph_enabled`]). Experimental: on
    /// ROCm 6.2 / Windows the 30B service chain degenerates after ~1-4k graph
    /// replays (hipGraphLaunch/sync report success while the GPU work silently
    /// vanishes — see the churn repro `qwen3_30b_graph_churn`); pure single-
    /// stream replay survived 12k steps, so the knob stays opt-in until the
    /// driver-side behavior is re-verified on a newer ROCm.
    graph_enabled: bool,
    /// Debug (issue #107 layer-divergence localization): when set, every
    /// forward snapshots the LAST active row's residual (`self.x`) to
    /// `<dir>/hs_L{li:02}.npy` after each layer body — the same per-layer
    /// hidden-state trace `.scratch/np_ref.py` records at `REC_POS` — so the
    /// first layer where the device forward diverges from the host reference
    /// can be read straight off the files. Syncs per layer and is mutually
    /// exclusive with graph capture (see [`Self::graph_capture_ok`]).
    layer_dump: Option<std::path::PathBuf>,
}

/// Minimal NumPy `.npy` writer (f32, C-order, format version 1.0) for the
/// `layer_dump` debug hook — hand-rolled so the diagnostic path pulls no
/// extra dependency. `np.load` reads what this emits.
fn write_npy_f32(path: &std::path::Path, v: &[f32]) -> Result<(), Error> {
    let mut hdr = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        v.len()
    );
    // numpy pads the header so the total prefix (8 magic + 2 len) is a
    // multiple of 64 and terminates it with a newline.
    let pad = (64 - (10 + hdr.len() + 1) % 64) % 64;
    hdr.push_str(&" ".repeat(pad));
    hdr.push('\n');
    let mut buf = Vec::with_capacity(10 + hdr.len() + v.len() * 4);
    buf.extend_from_slice(b"\x93NUMPY\x01\x00");
    buf.extend_from_slice(&(hdr.len() as u16).to_le_bytes());
    buf.extend_from_slice(hdr.as_bytes());
    for f in v {
        buf.extend_from_slice(&f.to_le_bytes());
    }
    std::fs::write(path, buf)
        .map_err(|e| Error::Model(format!("layer dump write {}: {e}", path.display())))
}

impl BatchedModel {
    /// Builds a batched model for `batch` sequences and uploads `w`.
    pub fn new(hip: Arc<Hip>, cfg: Config, w: &Weights, batch: usize) -> Result<Self, Error> {
        Self::with_rows(hip, cfg, w, batch, batch)
    }

    /// Builds a batched model with `slots` KV slots and `rows` row capacity
    /// (`rows >= slots`; prefill can pack more prompt positions per step).
    pub fn with_rows(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        slots: usize,
        rows: usize,
    ) -> Result<Self, Error> {
        Self::build(hip, cfg, w, slots, rows, usize::MAX)
    }

    /// Builds a batched model in **paged-KV** mode (dense F32): the KV caches
    /// are addressed via per-slot block tables (page pool) using
    /// `kv_store_paged` / `attn_decode_paged`. The default mapping is the
    /// static identity (same physical layout as the contiguous path); call
    /// [`Self::set_block_table`] to install reuse-planner tables so sequences
    /// share prefix physical pages. Dense F16 and MLA (F32) paged kernels are
    /// wired too (`with_paged_kv_rows` dtype variants; fused
    /// `kv_store_paged_mla`).
    pub fn with_paged_kv(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        slots: usize,
        tokens_per_page: usize,
    ) -> Result<Self, Error> {
        Self::with_paged_kv_rows(hip, cfg, w, slots, slots, tokens_per_page)
    }

    /// [`Self::with_paged_kv`] variant with an independent prefill row
    /// capacity (`rows >= slots`): a prefill step may pack several prompt
    /// positions of one sequence into distinct rows, each addressed through
    /// the sequence's block table via per-row table offsets.
    pub fn with_paged_kv_rows(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        slots: usize,
        rows: usize,
        tokens_per_page: usize,
    ) -> Result<Self, Error> {
        Self::paged_guards(&cfg, tokens_per_page)?;
        let mut m = Self::build(hip, cfg, w, slots, rows, usize::MAX)?;
        m.init_paged(tokens_per_page)?;
        Ok(m)
    }

    /// [`Self::with_paged_kv_rows`] for storage-Q4 weights: each tensor is
    /// dequantized to f16 on upload, so the wired f16 paged kernels serve the
    /// device path directly.
    pub fn with_paged_kv_rows_q4(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsQ4,
        slots: usize,
        rows: usize,
        tokens_per_page: usize,
    ) -> Result<Self, Error> {
        Self::paged_guards(&cfg, tokens_per_page)?;
        let mut m = Self::build_q4(hip, cfg, w, slots, rows)?;
        m.init_paged(tokens_per_page)?;
        Ok(m)
    }

    /// Paged Q4-on-device variant: the expert pool stays raw Q4 on device
    /// (in-kernel dequant) — the memory path for 30B-class checkpoints under
    /// paged KV with cross-request prefix reuse.
    pub fn with_paged_kv_rows_q4_device(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsQ4,
        slots: usize,
        rows: usize,
        tokens_per_page: usize,
    ) -> Result<Self, Error> {
        Self::paged_guards(&cfg, tokens_per_page)?;
        let mut m = Self::build_common(hip, cfg, slots, rows, usize::MAX, |m| {
            m.upload_weights_q4(w, true, false)
        })?;
        m.init_paged(tokens_per_page)?;
        Ok(m)
    }

    /// Paged dense Q4-on-device variant (see [`Self::with_rows_q4_all`]):
    /// every big dense tensor AND the expert pool stay raw Q4 on device —
    /// the memory path for DENSE 27B-class checkpoints under paged KV.
    pub fn with_paged_kv_rows_q4_all(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsQ4,
        slots: usize,
        rows: usize,
        tokens_per_page: usize,
    ) -> Result<Self, Error> {
        Self::paged_guards(&cfg, tokens_per_page)?;
        let mut m = Self::build_common(hip, cfg, slots, rows, usize::MAX, |m| {
            m.upload_weights_q4(w, true, true)
        })?;
        m.init_paged(tokens_per_page)?;
        Ok(m)
    }

    /// [`Self::with_paged_kv_rows`] for storage-FP8 weights: E4M3 tensors are
    /// dequantized to f16 on upload; the f16 paged kernels serve the device.
    pub fn with_paged_kv_rows_fp8(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsFp8,
        slots: usize,
        rows: usize,
        tokens_per_page: usize,
    ) -> Result<Self, Error> {
        Self::paged_guards(&cfg, tokens_per_page)?;
        let mut m = Self::build_fp8(hip, cfg, w, slots, rows)?;
        m.init_paged(tokens_per_page)?;
        Ok(m)
    }

    /// Construction-time paged-mode guards shared by every build variant:
    /// page geometry, attention smem bound, dtype coverage (dense F32/F16;
    /// MLA stays F32-only) — fail loudly (as `Err`, not a panic) here rather
    /// than falling through to contiguous kernels while owning block tables,
    /// or failing mid-request.
    fn paged_guards(cfg: &Config, tokens_per_page: usize) -> Result<(), Error> {
        if tokens_per_page == 0 || !cfg.max_seq_len.is_multiple_of(tokens_per_page) {
            return Err(Error::InvalidArgument(format!(
                "tokens_per_page {tokens_per_page} must be a non-zero divisor of max_seq_len {}",
                cfg.max_seq_len
            )));
        }
        // The paged attention kernels stage the full per-row score array in
        // dynamic shared memory: `(max_pages * tokens_per_page + 256) * 4` ==
        // `(max_seq_len + 256) * 4` bytes. Enforce the 64 KiB device limit
        // here so an unsupported context size fails at construction with a
        // clear error, not with a cryptic launch failure mid-request. (The
        // contiguous attention kernels share the same smem bound.)
        let smem = (cfg.max_seq_len + 256) * 4;
        if smem > 64 * 1024 {
            return Err(Error::InvalidArgument(format!(
                "max_seq_len {} needs {smem} bytes of attention smem (64 KiB device limit); \
                 paged-KV mode supports contexts up to {} tokens",
                cfg.max_seq_len,
                (64 * 1024) / 4 - 256
            )));
        }
        // The paged decode branch dispatches dense F32/F16 and MLA (F32); any
        // other dtype would silently fall through to the contiguous kernels
        // while still owning block tables. Reject loudly at construction.
        if !matches!(cfg.dtype, ModelDType::F32 | ModelDType::F16) {
            return Err(Error::InvalidArgument(format!(
                "paged-KV mode supports dense F32/F16 only (got {:?})",
                cfg.dtype
            )));
        }
        if cfg.kv_lora_rank > 0 && cfg.dtype != ModelDType::F32 {
            return Err(Error::InvalidArgument(format!(
                "paged-KV mode supports MLA in F32 only (got {:?})",
                cfg.dtype
            )));
        }
        // GDN (hybrid linear-attention) layers carry recurrent state OUTSIDE
        // the KV pages: a shared-prefix reuse would restore only the
        // full-attn layers' KV and silently desync the hybrid stack. Stage B
        // (#112, chunked scan + state-aware reuse) revisits; contiguous-KV
        // decode supports GDN today.
        if cfg.gdn_enabled() {
            return Err(Error::InvalidArgument(
                "paged-KV mode does not support GDN (hybrid linear-attention) models yet".into(),
            ));
        }
        Ok(())
    }

    /// Authoritative pre-flight for paged-KV support: the same checks
    /// [`Self::paged_guards`] enforces at construction (page geometry,
    /// attention-smem bound, dtype coverage), exposed so callers (the
    /// server's pre-load gate) validate a model/mode combination BEFORE the
    /// multi-minute weight load instead of hand-copying the constraints.
    /// Pure `cfg` logic — CPU-runnable.
    pub fn check_paged_support(cfg: &Config, tokens_per_page: usize) -> Result<(), Error> {
        Self::paged_guards(cfg, tokens_per_page)
    }

    /// Pages per sequence in paged mode (`max_seq / tokens_per_page`).
    #[must_use]
    pub fn max_pages_per_seq(&self) -> usize {
        self.max_pages_per_seq
    }

    /// Paged mode: replaces slot `slot`'s logical→physical block table with
    /// `phys_pages` (logical page `L` → `phys_pages[L]`) and uploads the slot's
    /// table region to the device. Table updates run between steps (before the
    /// next `decode_step*`), and the upload is stream-ordered against them.
    ///
    /// This is the shared-prefix entry point: build tables with
    /// [`GpuPagedTableBuilder`](crate::paged_kv::GpuPagedTableBuilder) so
    /// concurrent requests alias the same physical prefix pages, then feed a
    /// reused request only its delta (its first stored/decoded position starts
    /// at the reuse boundary — the pages already hold that KV).
    ///
    /// Requirements: `!phys_pages.is_empty()`, `len <= max_pages_per_seq`, and
    /// every id `< batch * max_pages_per_seq` (ids index the single shared pool
    /// allocation). Entries past `phys_pages.len()` are padded by repeating the
    /// last id so even a misaddressed read stays inside the pool; they must not
    /// be addressed while the sequence length stays within the written range
    /// (the kernels never do — they bound pages by position).
    pub fn set_block_table(&mut self, slot: usize, phys_pages: &[u32]) -> Result<(), Error> {
        if !self.paged {
            return Err(Error::InvalidArgument(
                "set_block_table requires paged mode (with_paged_kv)".into(),
            ));
        }
        if slot >= self.batch {
            return Err(Error::InvalidArgument(format!(
                "slot {slot} out of range (batch {})",
                self.batch
            )));
        }
        let maxp = self.max_pages_per_seq;
        let pool_pages = self.batch * maxp;
        if phys_pages.is_empty() || phys_pages.len() > maxp {
            return Err(Error::InvalidArgument(format!(
                "block table length {} outside 1..={maxp}",
                phys_pages.len()
            )));
        }
        if let Some(&bad) = phys_pages.iter().find(|&&p| (p as usize) >= pool_pages) {
            return Err(Error::InvalidArgument(format!(
                "physical page {bad} out of pool range ({pool_pages} pages)"
            )));
        }
        let start = slot * maxp;
        for (i, &p) in phys_pages.iter().enumerate() {
            self.tables_host[start + i] = p as i32;
        }
        // Pad trailing logical pages with the last valid id (never addressed
        // while position-bounded; keeps any stray read inside the pool).
        for t in self.tables_host[start + phys_pages.len()..start + maxp].iter_mut() {
            *t = phys_pages[phys_pages.len() - 1] as i32;
        }
        // Stream-ordering note: this runs between steps, and hipMemcpy serializes
        // against subsequent kernel launches on the engine stream.
        let dst = unsafe { self.block_tables.add(start) };
        hip::memcpy(
            self.k.hip(),
            dst as *mut core::ffi::c_void,
            self.tables_host[start..start + maxp].as_ptr() as *const core::ffi::c_void,
            maxp * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )?;
        Ok(())
    }

    /// Builds a batched model from storage-Q4 weights (dense + MoE, F16): each
    /// GEMM tensor is dequantized to f16 on the host and uploaded directly, so
    /// host memory stays ~= the packed Q4 weights + one tensor's f16 buffer.
    /// Experts stay fully GPU-resident (the CPU-backend offload path needs f32
    /// `Weights`, which the Q4 host layout does not provide).
    pub fn from_q4(hip: Arc<Hip>, cfg: Config, w: &WeightsQ4, batch: usize) -> Result<Self, Error> {
        Self::with_rows_q4(hip, cfg, w, batch, batch)
    }

    /// Builds a batched model from storage-Q4 weights (F16) with the EXPERT
    /// POOL KEPT IN Q4 ON DEVICE: the raw packed int4 bytes + per-group f32
    /// scales are uploaded as-is and dequantized in-kernel by the Q4 grouped
    /// GEMV kernels. This is the memory path for 30B-class checkpoints — the
    /// dequantized f16 expert pool (~50GB) would not fit in VRAM, the Q4 pool
    /// (~16GB) does. Non-expert tensors (attention/MLP/router) dequantize to
    /// f16 on upload as usual. All steps run the grouped path (no f16/f32
    /// expert copy exists for the hipBLAS path).
    pub fn with_rows_q4_device(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsQ4,
        slots: usize,
        rows: usize,
    ) -> Result<Self, Error> {
        if cfg.dtype != ModelDType::F16 {
            return Err(Error::Model(
                "from_q4 requires dtype F16 (dequantize to f16 on device)".into(),
            ));
        }
        Self::build_common(hip, cfg, slots, rows, usize::MAX, |m| {
            m.upload_weights_q4(w, true, false)
        })
    }

    /// Q4-everything variant of [`Self::with_rows_q4_device`] (dense
    /// Q4-on-device, server `MACH_Q4_DEVICE=2`): the expert pool AND every
    /// big dense GEMM tensor (q/k/v/o projections, dense MLP, GDN
    /// projections, embedding, lm_head) stay raw Q4 on device, read with
    /// in-kernel dequant — the memory path for DENSE 27B-class checkpoints
    /// (all-Q4 ~13.5GB fits VRAM; the dequantized f16 copy ~54GB does not).
    /// Small/unquantized tensors (norms, biases, router, GDN a/b) and the
    /// shared MLP keep their f32/f16 copies as in the other Q4 modes. MLA
    /// (`kv_lora_rank > 0`) is not wired on this path yet — loud error.
    pub fn with_rows_q4_all(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsQ4,
        slots: usize,
        rows: usize,
    ) -> Result<Self, Error> {
        if cfg.dtype != ModelDType::F16 {
            return Err(Error::Model(
                "from_q4 requires dtype F16 (dequantize to f16 on device)".into(),
            ));
        }
        if cfg.kv_lora_rank > 0 {
            return Err(Error::Model(
                "dense Q4-on-device does not support MLA yet (kv_lora_rank > 0)".into(),
            ));
        }
        Self::build_common(hip, cfg, slots, rows, usize::MAX, |m| {
            m.upload_weights_q4(w, true, true)
        })
    }

    /// Q4 variant of [`with_rows`]: `slots` KV slots and `rows` row capacity
    /// (prefill can pack more prompt positions per step).
    pub fn with_rows_q4(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsQ4,
        slots: usize,
        rows: usize,
    ) -> Result<Self, Error> {
        Self::build_q4(hip, cfg, w, slots, rows)
    }

    /// Builds a batched model from storage-FP8 weights (F16): each GEMM tensor
    /// is dequantized to f16 on the host and uploaded directly, so host memory
    /// stays ~= the packed FP8 weights + one tensor's f16 buffer. Experts stay
    /// fully GPU-resident (the CPU-backend offload path needs f32 `Weights`,
    /// which the FP8 host layout does not provide).
    pub fn from_fp8(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsFp8,
        batch: usize,
    ) -> Result<Self, Error> {
        Self::with_rows_fp8(hip, cfg, w, batch, batch)
    }

    /// FP8 variant of [`with_rows`]: `slots` KV slots and `rows` row capacity
    /// (prefill can pack more prompt positions per step).
    pub fn with_rows_fp8(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsFp8,
        slots: usize,
        rows: usize,
    ) -> Result<Self, Error> {
        Self::build_fp8(hip, cfg, w, slots, rows)
    }

    /// Builds a batched model in MoE offload mode: experts stay in host RAM and the
    /// MoE layer is computed on the CPU (FreeToken cpu backend), so GPU memory is
    /// bounded regardless of the expert count. GPU-slot fast path is a follow-up.
    pub fn with_expert_slots(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        slots: usize,
        rows: usize,
        expert_slots: usize,
    ) -> Result<Self, Error> {
        let mut m = Self::build(hip, cfg, w, slots, rows, expert_slots)?;
        m.host_w = Some(Arc::new(w.clone()));
        Ok(m)
    }

    /// Builds a batched model in full-layer double-buffered prefill mode
    /// (FreeToken-style): MoE expert weights stay in host RAM (pinned) and each
    /// MoE layer's experts are prefetched host→device on a separate stream
    /// while the previous layer is computed, overlapping the H2D with the
    /// GEMMs. The grouped expert GEMMs then read the prefetched weights, so the
    /// math is identical to the full-resident path; device memory holds only
    /// two ping-pong expert pools. F32 only for now.
    pub fn with_prefill_buffer(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        slots: usize,
        rows: usize,
    ) -> Result<Self, Error> {
        let mut m = Self::build(hip, cfg, w, slots, rows, 0)?;
        let engine = crate::prefill_buffered::PrefetchEngine::new(Arc::clone(m.k.hip()), cfg, w)?;
        m.prefetch = Some(engine);
        m.host_w = Some(Arc::new(w.clone()));
        Ok(m)
    }

    /// Enables paged-KV mode: allocates the static block tables and the
    /// per-row table-offset buffers and fills them with the default identity
    /// mapping. Decode always runs with `row == slot`, so both mappings are
    /// static and need no per-step refresh.
    fn init_paged(&mut self, tokens_per_page: usize) -> Result<(), Error> {
        let max_pages = self.cfg.max_seq_len / tokens_per_page;
        self.paged = true;
        self.tokens_per_page = tokens_per_page;
        self.max_pages_per_seq = max_pages;

        // Default mapping (identity): slot S's logical page L -> S*max_pages + L.
        let mut tables: Vec<i32> = Vec::with_capacity(self.batch * max_pages);
        for s in 0..self.batch {
            let base = (s * max_pages) as i32;
            for l in 0..max_pages {
                tables.push(base + l as i32);
            }
        }
        self.tables_host = tables;
        // Per-row offsets: row r -> r*max_pages (row == slot during decode).
        let mut offsets: Vec<i32> = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            offsets.push((r * max_pages) as i32);
        }
        let hip = self.k.hip();
        let oh = hip::host_malloc(hip, offsets.len() * 4)?;
        self.host_pins.push(oh);
        self.offsets_host = oh as *mut i32;
        let bt = hip::malloc(hip, self.tables_host.len() * 4)?;
        self.allocs.push(bt);
        hip::memcpy(
            hip,
            bt,
            self.tables_host.as_ptr() as *const core::ffi::c_void,
            self.tables_host.len() * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )?;
        self.block_tables = bt as *mut i32;
        let toff = hip::malloc(hip, offsets.len() * 4)?;
        self.allocs.push(toff);
        hip::memcpy(
            hip,
            toff,
            offsets.as_ptr() as *const core::ffi::c_void,
            offsets.len() * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )?;
        self.table_offsets = toff as *mut i32;
        Ok(())
    }

    fn build(
        hip: Arc<Hip>,
        cfg: Config,
        w: &Weights,
        slots: usize,
        rows: usize,
        expert_slots: usize,
    ) -> Result<Self, Error> {
        Self::build_common(hip, cfg, slots, rows, expert_slots, |m| m.upload_weights(w))
    }

    /// Q4 (storage-int4) batched build: same device layout as the F16 path, but
    /// each GEMM tensor is dequantized from packed Q4 to f16 during upload.
    fn build_q4(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsQ4,
        slots: usize,
        rows: usize,
    ) -> Result<Self, Error> {
        if cfg.dtype != ModelDType::F16 {
            return Err(Error::Model(
                "from_q4 requires dtype F16 (dequantize to f16 on device)".into(),
            ));
        }
        Self::build_common(hip, cfg, slots, rows, usize::MAX, |m| {
            m.upload_weights_q4(w, false, false)
        })
    }

    /// FP8 (storage-E4M3) batched build: same device layout as the F16 path,
    /// but each GEMM tensor is dequantized from E4M3 to f16 during upload.
    fn build_fp8(
        hip: Arc<Hip>,
        cfg: Config,
        w: &WeightsFp8,
        slots: usize,
        rows: usize,
    ) -> Result<Self, Error> {
        if cfg.dtype != ModelDType::F16 {
            return Err(Error::Model(
                "from_fp8 requires dtype F16 (dequantize to f16 on device)".into(),
            ));
        }
        Self::build_common(hip, cfg, slots, rows, usize::MAX, |m| {
            m.upload_weights_fp8(w)
        })
    }

    fn build_common(
        hip: Arc<Hip>,
        cfg: Config,
        slots: usize,
        rows: usize,
        expert_slots: usize,
        upload: impl FnOnce(&mut Self) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        assert!(slots >= 1, "slots must be >= 1");
        assert!(rows >= slots, "rows must be >= slots");
        let k = Arc::new(HipKernels::new(Arc::clone(&hip))?);
        let sampler = BatchedSampler::new(Arc::clone(&hip), k.stream, rows)?;
        let mut m = Self {
            cfg,
            batch: slots,
            rows,
            k,
            sampler,
            tokens_dev: std::ptr::null_mut(),
            pos_dev: std::ptr::null_mut(),
            tokens_host: std::ptr::null_mut(),
            pos_host: std::ptr::null_mut(),
            slots_host: std::ptr::null_mut(),
            slots_dev: std::ptr::null_mut(),
            runs_host: std::ptr::null_mut(),
            runs_dev: std::ptr::null_mut(),
            run_mask_host: std::ptr::null_mut(),
            run_mask_dev: std::ptr::null_mut(),
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
            out_tok_dev: std::ptr::null_mut(),
            out_tok_host: std::ptr::null_mut(),
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
            counts_dev: std::ptr::null_mut(),
            moe_pos_dev: std::ptr::null_mut(),
            offsets_dev: std::ptr::null_mut(),
            counts_host: std::ptr::null_mut(),
            xg: std::ptr::null_mut(),
            gw: std::ptr::null_mut(),
            row_idx: std::ptr::null_mut(),
            h_acc: std::ptr::null_mut(),
            gate_all: std::ptr::null_mut(),
            up_all: std::ptr::null_mut(),
            eh_all: std::ptr::null_mut(),
            down_all: std::ptr::null_mut(),
            xh_moe: std::ptr::null_mut(),
            yh_moe: std::ptr::null_mut(),
            mla_q_lora: std::ptr::null_mut(),
            mla_q_lora_n: std::ptr::null_mut(),
            q_nope: std::ptr::null_mut(),
            q_rope: std::ptr::null_mut(),
            mla_kv_a: std::ptr::null_mut(),
            mla_kv_a_n: std::ptr::null_mut(),
            mla_kv_lora: std::ptr::null_mut(),
            mla_kv: std::ptr::null_mut(),
            mla_k_rope: std::ptr::null_mut(),
            mla_attn: std::ptr::null_mut(),
            qg: std::ptr::null_mut(),
            attn_gate: std::ptr::null_mut(),
            gdn_qkv_pre: std::ptr::null_mut(),
            gdn_qkv: std::ptr::null_mut(),
            gdn_z: std::ptr::null_mut(),
            gdn_a: std::ptr::null_mut(),
            gdn_b: std::ptr::null_mut(),
            gdn_o: std::ptr::null_mut(),
            gdn_state: Vec::new(),
            gdn_conv: Vec::new(),
            q4_experts: Vec::new(),
            q4_layers: Vec::new(),
            emb_q4: Q4TensorDev::EMPTY,
            lm_head_q4: Q4TensorDev::EMPTY,
            kv_cache: Vec::new(),
            mla_kv_cache: Vec::new(),
            lens: vec![0; slots],
            last_row_by_slot: vec![0; slots],
            allocs: Vec::new(),
            host_pins: Vec::new(),
            expert_slots,
            host_w: None,
            prefetch: None,
            paged: false,
            moe_grouped: cfg.moe_grouped,
            prof: None,
            tokens_per_page: 0,
            max_pages_per_seq: 0,
            block_tables: std::ptr::null_mut(),
            tables_host: Vec::new(),
            offsets_host: std::ptr::null_mut(),
            offsets_dirty: false,
            table_offsets: std::ptr::null_mut(),
            decode_graphs: std::collections::HashMap::new(),
            greedy_graph: None,
            graph_enabled: std::env::var("MACH_GRAPH").is_ok_and(|v| v == "1"),
            layer_dump: None,
        };
        m.alloc_buffers()?;
        upload(&mut m)?;
        // Step profiler (Config.step_profile, from the server's
        // MACH_STEP_PROFILE): per-layer attention/MoE event bracketing,
        // reported after each decode step's sampling sync. A diagnostic —
        // if event creation fails we warn and continue without it rather
        // than failing the model load.
        if cfg.step_profile {
            m.prof = match ProfState::new(cfg.n_layers, &hip) {
                Ok(p) => Some(p),
                Err(e) => {
                    eprintln!("warning: MACH_STEP_PROFILE unavailable: {e}; continuing without");
                    None
                }
            };
        }
        Ok(m)
    }

    /// CPU-backend MoE offload for the batch: experts live in host RAM and the MoE
    /// layer is computed on the CPU from the host weights (FreeToken `cpu` backend).
    /// The router still runs on the GPU (moe_router is uploaded); the grouped-GEMM
    /// path is bypassed, so GPU memory is bounded by the router, not the experts.
    /// Pending stable-GPU parity; the GPU-slot fast path is a follow-up.
    fn forward_moe_cpu_batched(
        &self,
        li: usize,
        _ne: i32,
        topk: i32,
        b: i32,
        d: i32,
        inter: i32,
    ) -> Result<(), Error> {
        self.k.sync()?;
        let n_entries = (b * topk) as usize;
        let mut ids = vec![0i32; n_entries];
        let mut weights = vec![0.0f32; n_entries];
        hip::memcpy(
            self.k.hip(),
            ids.as_mut_ptr() as *mut core::ffi::c_void,
            self.exp_ids as *const core::ffi::c_void,
            n_entries * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )?;
        hip::memcpy(
            self.k.hip(),
            weights.as_mut_ptr() as *mut core::ffi::c_void,
            self.exp_w as *const core::ffi::c_void,
            n_entries * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )?;
        let (b_us, d_us, inter_us, topk_us) =
            (b as usize, d as usize, inter as usize, topk as usize);
        let mut xn2 = vec![0.0f32; b_us * d_us];
        hip::memcpy(
            self.k.hip(),
            xn2.as_mut_ptr() as *mut core::ffi::c_void,
            self.xn2 as *const core::ffi::c_void,
            xn2.len() * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )?;
        let host_w = self
            .host_w
            .as_ref()
            .ok_or_else(|| Error::Model("cpu-backend offload requires host weights".into()))?;
        let lw_h = &host_w.layers[li];
        let residual = moe_offload::moe_batch_cpu_residual(
            &ids, &weights, &xn2, lw_h, b_us, d_us, inter_us, topk_us,
        );
        let mut x = vec![0.0f32; b_us * d_us];
        hip::memcpy(
            self.k.hip(),
            x.as_mut_ptr() as *mut core::ffi::c_void,
            self.x as *const core::ffi::c_void,
            x.len() * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )?;
        for i in 0..x.len() {
            x[i] += residual[i];
        }
        self.upload(self.x, &x)?;
        Ok(())
    }
    fn dalloc(&mut self, bytes: usize) -> Result<*mut f32, Error> {
        let p = hip::malloc(self.k.hip(), bytes)
            .map_err(|e| Error::Model(format!("device alloc of {bytes} bytes failed: {e}")))?;
        self.allocs.push(p);
        Ok(p as *mut f32)
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

    fn alloc_f16(&mut self, n: usize) -> Result<*mut u16, Error> {
        let p = hip::malloc(self.k.hip(), n * 2)?;
        self.allocs.push(p);
        Ok(p as *mut u16)
    }

    /// Allocates + uploads an f32 matrix, or returns null on the F16 path
    /// (fp16 weights live in `LayerDevF16`; the f32 copy is not needed).
    /// Empty slices (absent tensors, e.g. wq on a GDN layer) return null on
    /// both paths — no zero-byte allocation.
    fn upload_mat32(&mut self, src: &[f32], f16: bool) -> Result<*mut f32, Error> {
        if src.is_empty() || f16 {
            Ok(std::ptr::null_mut())
        } else {
            let p = self.dalloc(src.len() * 4)?;
            self.upload(p, src)?;
            Ok(p)
        }
    }

    /// Allocates + uploads an optional f16 matrix (empty -> null device
    /// pointer): the F16-path mirror of [`Self::upload_mat32`].
    fn upload_f16_mat(&mut self, src: &[f32]) -> Result<*mut u16, Error> {
        if src.is_empty() {
            Ok(std::ptr::null_mut())
        } else {
            let p = self.alloc_f16(src.len())?;
            self.upload_f16(p, src)?;
            Ok(p)
        }
    }

    fn alloc_buffers(&mut self) -> Result<(), Error> {
        let c = self.cfg;
        let b = self.rows; // row buffers sized for the prefill row capacity
        let d = c.d_model;
        let nq = c.n_heads * c.head_dim;
        let nkv = c.n_kv_heads * c.head_dim;
        // The shared experts (DeepSeek-V2) are one dense MLP of width
        // `n_shared_experts * expert_size`, which can exceed the dense
        // `intermediate_size`; the gate/up/h scratch must cover both.
        let inter = c.intermediate_size.max(c.shared_size());

        self.tokens_dev = self.dalloc(b * 4)? as *mut i32;
        self.pos_dev = self.dalloc(b * 4)? as *mut i32;
        self.slots_dev = self.dalloc(b * 4)? as *mut i32;
        let max_runs = b.div_ceil(2);
        self.runs_dev = self.dalloc(max_runs * 4 * 4)? as *mut i32;
        self.run_mask_dev = self.dalloc(b * 4)? as *mut i32;
        let th = hip::host_malloc(self.k.hip(), b * 4)?;
        let ph = hip::host_malloc(self.k.hip(), b * 4)?;
        let sh = hip::host_malloc(self.k.hip(), b * 4)?;
        let rh = hip::host_malloc(self.k.hip(), max_runs * 4 * 4)?;
        let mh = hip::host_malloc(self.k.hip(), b * 4)?;
        self.tokens_host = th as *mut i32;
        self.pos_host = ph as *mut i32;
        self.slots_host = sh as *mut i32;
        self.runs_host = rh as *mut i32;
        self.run_mask_host = mh as *mut i32;
        self.host_pins.push(th);
        self.host_pins.push(ph);
        self.host_pins.push(sh);
        self.host_pins.push(rh);
        self.host_pins.push(mh);

        self.x = self.dalloc(b * d * 4)?;
        self.xn = self.dalloc(b * d * 4)?;
        self.xn2 = self.dalloc(b * d * 4)?;
        self.q = self.dalloc(b * nq * 4)?;
        self.k_buf = self.dalloc(b * nkv * 4)?;
        self.v_buf = self.dalloc(b * nkv * 4)?;
        self.attn = self.dalloc(b * nq * 4)?;
        self.proj = self.dalloc(b * d * 4)?;
        self.gate = self.dalloc(b * inter * 4)?;
        self.up = self.dalloc(b * inter * 4)?;
        self.h = self.dalloc(b * inter * 4)?;
        self.logits = self.dalloc(b * c.vocab_size * 4)?;
        // fp16 GEMM scratch (only used in F16 mode): A operand + fp16 output
        // (yh must cover the lm_head output = batch * vocab for the c16 logits
        // path, which is then cast back to fp32 for the sampler).
        let max_n = c
            .d_model
            .max(inter)
            .max(c.n_heads * c.head_dim)
            .max(c.vocab_size);
        let xh_bytes = b * c.d_model.max(inter) * 2;
        let xh = hip::malloc(self.k.hip(), xh_bytes)?;
        self.xh = xh as *mut u16;
        self.allocs.push(xh);
        let yh = hip::malloc(self.k.hip(), b * max_n * 2)?;
        self.yh = yh as *mut u16;
        self.allocs.push(yh);
        self.out_tok_dev = self.dalloc(b * 4)? as *mut i32;
        let oh = hip::host_malloc(self.k.hip(), b * 4)?;
        self.out_tok_host = oh as *mut i32;
        self.host_pins.push(oh);

        let kv_elem = if c.dtype == ModelDType::F16 { 2 } else { 4 };
        // KV caches are sized by the slot count, not the row capacity.
        // Dense KV cache: skipped on the MLA path (kv_lora_rank > 0), which
        // keeps its expanded per-head KV in `mla_kv_cache`; allocating it here
        // would waste VRAM (n_kv_heads = n_heads, head_dim = nope+rope).
        if c.kv_lora_rank == 0 {
            let kv_bytes = self.batch * c.max_seq_len * c.n_kv_heads * c.head_dim * kv_elem;
            for _ in 0..c.n_layers {
                let kk = self.dalloc(kv_bytes)?;
                let vv = self.dalloc(kv_bytes)?;
                self.kv_cache
                    .push((kk as *mut core::ffi::c_void, vv as *mut core::ffi::c_void));
            }
        }
        if c.kv_lora_rank > 0 {
            let qlr = c.q_lora_rank;
            let nope = c.qk_nope_head_dim;
            let rope = c.qk_rope_head_dim;
            let v_hd = c.v_head_dim;
            let heads = c.n_heads;
            self.mla_q_lora = self.dalloc(b * qlr * 4)?;
            self.mla_q_lora_n = self.dalloc(b * qlr * 4)?;
            self.q_nope = self.dalloc(b * heads * nope * 4)?;
            self.q_rope = self.dalloc(b * heads * rope * 4)?;
            self.mla_kv_a = self.dalloc(b * (c.kv_lora_rank + rope) * 4)?;
            self.mla_kv_a_n = self.dalloc(b * c.kv_lora_rank * 4)?;
            self.mla_kv_lora = self.dalloc(b * c.kv_lora_rank * 4)?;
            self.mla_kv = self.dalloc(b * heads * (nope + v_hd) * 4)?;
            self.mla_k_rope = self.dalloc(b * rope * 4)?;
            self.mla_attn = self.dalloc(b * heads * v_hd * 4)?;
            // MLA KV caches are stored as f32 (expanded per-head layout).
            let k_bytes = self.batch * c.max_seq_len * heads * (nope + rope) * 4;
            let v_bytes = self.batch * c.max_seq_len * heads * v_hd * 4;
            for _ in 0..c.n_layers {
                let kk = self.dalloc(k_bytes)?;
                let vv = self.dalloc(v_bytes)?;
                self.mla_kv_cache
                    .push((kk as *mut core::ffi::c_void, vv as *mut core::ffi::c_void));
            }
        }
        // Qwen3.5 `attn_output_gate`: the doubled q_proj output + gate split.
        if c.attn_output_gate {
            self.qg = self.dalloc(b * 2 * nq * 4)?;
            self.attn_gate = self.dalloc(b * nq * 4)?;
        }
        // Qwen3.5 GDN decode scratch (row-sized) + per-layer SLOT-sized
        // recurrent/conv state (persistent per request; zeroed — the pos-0
        // recurrence reads S = 0 and an empty conv window, and hipMalloc
        // does not zero).
        if c.gdn_enabled() {
            let kd = c.gdn_key_dim();
            let vd = c.gdn_value_dim();
            let conv_dim = 2 * kd + vd;
            let keep = c.gdn_conv_kernel - 1;
            self.gdn_qkv_pre = self.dalloc(b * conv_dim * 4)?;
            self.gdn_qkv = self.dalloc(b * conv_dim * 4)?;
            self.gdn_z = self.dalloc(b * vd * 4)?;
            self.gdn_a = self.dalloc(b * c.gdn_v_heads * 4)?;
            self.gdn_b = self.dalloc(b * c.gdn_v_heads * 4)?;
            self.gdn_o = self.dalloc(b * vd * 4)?;
            // Per-head recurrent state is [hd, hd] (GDN_STEP's layout
            // `[slots, vh, hd, hd]`) — NOT kd×vd: the full dims over-allocate
            // by (kd*vd)/(hd*hd) (768x on the 27B = 2.4GB/layer/slot vs the
            // 3.1MB actually addressed — 116GB at capacity 1, a build-time
            // OOM on every real-size config; the small test configs masked it).
            let state_zeros =
                vec![0.0f32; self.batch * c.gdn_v_heads * c.gdn_head_dim * c.gdn_head_dim];
            let conv_zeros = vec![0.0f32; self.batch * conv_dim * keep];
            for _ in 0..c.n_layers {
                let s = self.dalloc(state_zeros.len() * 4)?;
                self.upload(s, &state_zeros)?;
                let cv = self.dalloc(conv_zeros.len() * 4)?;
                self.upload(cv, &conv_zeros)?;
                self.gdn_state.push(s);
                self.gdn_conv.push(cv);
            }
        }
        if c.num_experts > 0 {
            let ne = c.num_experts;
            let topk = c.num_experts_per_tok.min(ne);
            if topk > 0 {
                let cap = b * topk; // packed grouped-row capacity
                self.router = self.dalloc(b * ne * 4)?;
                self.exp_ids = self.dalloc(cap * 4)? as *mut i32;
                self.exp_w = self.dalloc(cap * 4)?;
                self.counts_dev = self.dalloc(ne * 4)? as *mut i32;
                self.moe_pos_dev = self.dalloc(ne * 4)? as *mut i32;
                self.offsets_dev = self.dalloc(ne * 4)? as *mut i32;
                self.xg = self.dalloc(cap * d * 4)?;
                self.gw = self.dalloc(cap * 4)?;
                self.row_idx = self.dalloc(cap * 4)? as *mut i32;
                self.h_acc = self.dalloc(b * d * 4)?;
                // Expert scratch must cover the wider of dense/MoE widths.
                let moe_w = c.intermediate_size.max(c.expert_size());
                self.gate_all = self.dalloc(cap * moe_w * 4)?;
                self.up_all = self.dalloc(cap * moe_w * 4)?;
                self.eh_all = self.dalloc(cap * moe_w * 4)?;
                self.down_all = self.dalloc(cap * d * 4)?;
                let ch = hip::host_malloc(self.k.hip(), ne * 4)?;
                self.counts_host = ch as *mut i32;
                self.host_pins.push(ch);
                if c.dtype == ModelDType::F16 {
                    let m = c.d_model.max(moe_w);
                    self.xh_moe = self.alloc_f16(cap * m)?;
                    self.yh_moe = self.alloc_f16(cap * m)?;
                }
            }
        }
        Ok(())
    }

    fn upload_weights(&mut self, w: &Weights) -> Result<(), Error> {
        let c = self.cfg;
        let d = c.d_model;
        let nq = c.n_heads * c.head_dim;
        let nkv = c.n_kv_heads * c.head_dim;
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
                mla_q_a: if f16 || lw.mla_q_a.is_empty() {
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
                mla_q_b: if f16 || lw.mla_q_b.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.mla_q_b.len() * 4)?;
                    self.upload(p, &lw.mla_q_b)?;
                    p
                },
                mla_q_rope: if f16 || lw.mla_q_rope.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.mla_q_rope.len() * 4)?;
                    self.upload(p, &lw.mla_q_rope)?;
                    p
                },
                mla_kv_a: if f16 || lw.mla_kv_a.is_empty() {
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
                mla_kv_b: if f16 || lw.mla_kv_b.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.mla_kv_b.len() * 4)?;
                    self.upload(p, &lw.mla_kv_b)?;
                    p
                },
                mla_o: if f16 || lw.mla_o.is_empty() {
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
                moe_wg: if lw.moe_wg.is_empty() || self.expert_slots < c.num_experts {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.moe_wg.len() * 4)?;
                    self.upload(p, &lw.moe_wg)?;
                    p
                },
                moe_wu: if lw.moe_wu.is_empty() || self.expert_slots < c.num_experts {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.moe_wu.len() * 4)?;
                    self.upload(p, &lw.moe_wu)?;
                    p
                },
                moe_wd: if lw.moe_wd.is_empty() || self.expert_slots < c.num_experts {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.moe_wd.len() * 4)?;
                    self.upload(p, &lw.moe_wd)?;
                    p
                },
                // Shared experts are always resident (one small dense MLP), so
                // they follow the same f32/f16 split as the dense MLP weights.
                shared_wg: if f16 || lw.shared_wg.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.shared_wg.len() * 4)?;
                    self.upload(p, &lw.shared_wg)?;
                    p
                },
                shared_wu: if f16 || lw.shared_wu.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.shared_wu.len() * 4)?;
                    self.upload(p, &lw.shared_wu)?;
                    p
                },
                shared_wd: if f16 || lw.shared_wd.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.shared_wd.len() * 4)?;
                    self.upload(p, &lw.shared_wd)?;
                    p
                },
                has_shared: !lw.shared_wg.is_empty(),
                // GDN tensors (empty on full-attn layers / non-GDN models);
                // `has_gdn` is the dtype-independent dispatch flag.
                has_gdn: !lw.gdn_in_qkv.is_empty(),
                gdn_in_qkv: if f16 || lw.gdn_in_qkv.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.gdn_in_qkv.len() * 4)?;
                    self.upload(p, &lw.gdn_in_qkv)?;
                    p
                },
                gdn_in_z: if f16 || lw.gdn_in_z.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.gdn_in_z.len() * 4)?;
                    self.upload(p, &lw.gdn_in_z)?;
                    p
                },
                gdn_in_a: if f16 || lw.gdn_in_a.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.gdn_in_a.len() * 4)?;
                    self.upload(p, &lw.gdn_in_a)?;
                    p
                },
                gdn_in_b: if f16 || lw.gdn_in_b.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.gdn_in_b.len() * 4)?;
                    self.upload(p, &lw.gdn_in_b)?;
                    p
                },
                gdn_out: if f16 || lw.gdn_out.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.gdn_out.len() * 4)?;
                    self.upload(p, &lw.gdn_out)?;
                    p
                },
                gdn_conv_w: self.upload_opt(&lw.gdn_conv_w)?,
                gdn_a_log: self.upload_opt(&lw.gdn_a_log)?,
                gdn_dt_bias: self.upload_opt(&lw.gdn_dt_bias)?,
                gdn_norm: self.upload_opt(&lw.gdn_norm)?,
            };
            self.upload(l.rms_attn, &lw.rms_attn)?;
            self.upload(l.rms_mlp, &lw.rms_mlp)?;
            let _ = (d, nq, nkv);
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
                    moe_router: self.alloc_f16(lw.moe_router.len())?,
                    moe_wg: if self.expert_slots < c.num_experts {
                        std::ptr::null_mut()
                    } else {
                        self.alloc_f16(lw.moe_wg.len())?
                    },
                    moe_wu: if self.expert_slots < c.num_experts {
                        std::ptr::null_mut()
                    } else {
                        self.alloc_f16(lw.moe_wu.len())?
                    },
                    moe_wd: if self.expert_slots < c.num_experts {
                        std::ptr::null_mut()
                    } else {
                        self.alloc_f16(lw.moe_wd.len())?
                    },
                    mla_q_a: self.alloc_f16(lw.mla_q_a.len())?,
                    mla_q_b: self.alloc_f16(lw.mla_q_b.len())?,
                    mla_q_rope: self.alloc_f16(lw.mla_q_rope.len())?,
                    mla_kv_a: self.alloc_f16(lw.mla_kv_a.len())?,
                    mla_kv_b: self.alloc_f16(lw.mla_kv_b.len())?,
                    mla_o: self.alloc_f16(lw.mla_o.len())?,
                    shared_wg: self.alloc_f16(lw.shared_wg.len())?,
                    shared_wu: self.alloc_f16(lw.shared_wu.len())?,
                    shared_wd: self.alloc_f16(lw.shared_wd.len())?,
                    gdn_in_qkv: self.upload_f16_mat(&lw.gdn_in_qkv)?,
                    gdn_in_z: self.upload_f16_mat(&lw.gdn_in_z)?,
                    gdn_in_a: self.upload_f16_mat(&lw.gdn_in_a)?,
                    gdn_in_b: self.upload_f16_mat(&lw.gdn_in_b)?,
                    gdn_out: self.upload_f16_mat(&lw.gdn_out)?,
                };
                self.upload_f16(l16.wq, &lw.wq)?;
                self.upload_f16(l16.wk, &lw.wk)?;
                self.upload_f16(l16.wv, &lw.wv)?;
                self.upload_f16(l16.wo, &lw.wo)?;
                self.upload_f16(l16.wg, &lw.wg)?;
                self.upload_f16(l16.wu, &lw.wu)?;
                self.upload_f16(l16.wd, &lw.wd)?;
                self.upload_f16(l16.moe_router, &lw.moe_router)?;
                if self.expert_slots >= c.num_experts {
                    self.upload_f16(l16.moe_wg, &lw.moe_wg)?;
                    self.upload_f16(l16.moe_wu, &lw.moe_wu)?;
                    self.upload_f16(l16.moe_wd, &lw.moe_wd)?;
                }
                self.upload_f16(l16.mla_q_a, &lw.mla_q_a)?;
                self.upload_f16(l16.mla_q_b, &lw.mla_q_b)?;
                self.upload_f16(l16.mla_q_rope, &lw.mla_q_rope)?;
                self.upload_f16(l16.mla_kv_a, &lw.mla_kv_a)?;
                self.upload_f16(l16.mla_kv_b, &lw.mla_kv_b)?;
                self.upload_f16(l16.mla_o, &lw.mla_o)?;
                self.upload_f16(l16.shared_wg, &lw.shared_wg)?;
                self.upload_f16(l16.shared_wu, &lw.shared_wu)?;
                self.upload_f16(l16.shared_wd, &lw.shared_wd)?;
                self.layers_f16.push(l16);
            }
        }
        Ok(())
    }

    /// Uploads storage-Q4 weights as f16 (dense/MoE/MLA F16 path): norms and
    /// biases stay f32; GEMM matrices are dequantized per tensor and uploaded.
    /// The router keeps its exact f32 copy in `LayerDev` (Q4 does not quantize
    /// it) plus the usual f16 copy for the fp16 GEMM path. With
    /// `q4_on_device`, the MoE EXPERT tensors stay raw Q4 (packed int4
    /// bytes + per-group f32 scales) in `self.q4_experts`, dequantized
    /// in-kernel by the Q4 grouped GEMV kernels — the memory path for
    /// 30B-class checkpoints (Q4 pool ~16GB vs the dequantized f16 ~50GB).
    /// `experts_on_device` keeps the MoE expert pool raw Q4 (`=1` mode);
    /// `dense_on_device` additionally keeps every big dense tensor raw Q4
    /// (`=2` mode — the embedding/lm_head + layer tables in `q4_layers`,
    /// read by `gemv_q4`/`embed_gather_q4` with in-kernel dequant).
    fn upload_weights_q4(
        &mut self,
        w: &WeightsQ4,
        experts_on_device: bool,
        dense_on_device: bool,
    ) -> Result<(), Error> {
        if dense_on_device {
            self.emb_q4 = Q4TensorDev {
                q: self.upload_bytes(w.tok_emb.q_bytes())?,
                s: self.upload_f32s(w.tok_emb.scales())?,
            };
            self.lm_head_q4 = Q4TensorDev {
                q: self.upload_bytes(w.lm_head.q_bytes())?,
                s: self.upload_f32s(w.lm_head.scales())?,
            };
        } else {
            self.emb_f16 = self.alloc_f16(w.tok_emb.len())?;
            self.lm_head_f16 = self.alloc_f16(w.lm_head.len())?;
            self.upload_f16_bits(self.emb_f16, &w.tok_emb.dequantize_f16())?;
            self.upload_f16_bits(self.lm_head_f16, &w.lm_head.dequantize_f16())?;
        }
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
                mla_q_a_norm: self.upload_opt(&lw.mla_q_a_norm)?,
                mla_q_b: std::ptr::null_mut(),
                mla_q_rope: std::ptr::null_mut(),
                mla_kv_a: std::ptr::null_mut(),
                mla_kv_a_norm: self.upload_opt(&lw.mla_kv_a_norm)?,
                mla_kv_b: std::ptr::null_mut(),
                mla_o: std::ptr::null_mut(),
                moe_router: self.upload_opt(&lw.moe_router)?,
                moe_wg: std::ptr::null_mut(),
                moe_wu: std::ptr::null_mut(),
                moe_wd: std::ptr::null_mut(),
                // Shared experts live in the f16 copy only (the Q4/FP8 paths
                // always run f16 GEMMs); the f32 pointers stay null.
                shared_wg: std::ptr::null_mut(),
                shared_wu: std::ptr::null_mut(),
                shared_wd: std::ptr::null_mut(),
                has_shared: !lw.shared_wg.is_empty(),
                // GDN: the GEMM matrices live in the f16 copy below; the small
                // f32 tensors (conv taps, A_log/dt_bias, gated-norm weight)
                // stay f32. `has_gdn` is the dtype-independent dispatch flag.
                has_gdn: !lw.gdn_in_qkv.is_empty(),
                gdn_in_qkv: std::ptr::null_mut(),
                gdn_in_z: std::ptr::null_mut(),
                gdn_in_a: std::ptr::null_mut(),
                gdn_in_b: std::ptr::null_mut(),
                gdn_out: std::ptr::null_mut(),
                gdn_conv_w: self.upload_opt(&lw.gdn_conv_w)?,
                gdn_a_log: self.upload_opt(&lw.gdn_a_log)?,
                gdn_dt_bias: self.upload_opt(&lw.gdn_dt_bias)?,
                gdn_norm: self.upload_opt(&lw.gdn_norm)?,
            };
            self.upload(l.rms_attn, &lw.rms_attn)?;
            self.upload(l.rms_mlp, &lw.rms_mlp)?;
            self.layers_dev.push(l);
            // Dense Q4-on-device skips the f16 copies of the big tensors:
            // they go up as raw Q4 pairs (`Q4DenseDev`) and the f16 pointers
            // stay null, so the GEMM dispatch reads the Q4 pair instead of
            // ever handing a null weight to the f16 kernels.
            let d16 = !dense_on_device;
            let l16 = LayerDevF16 {
                wq: if d16 {
                    self.alloc_f16(lw.wq.len())?
                } else {
                    std::ptr::null_mut()
                },
                wk: if d16 {
                    self.alloc_f16(lw.wk.len())?
                } else {
                    std::ptr::null_mut()
                },
                wv: if d16 {
                    self.alloc_f16(lw.wv.len())?
                } else {
                    std::ptr::null_mut()
                },
                wo: if d16 {
                    self.alloc_f16(lw.wo.len())?
                } else {
                    std::ptr::null_mut()
                },
                wg: if d16 {
                    self.alloc_f16(lw.wg.len())?
                } else {
                    std::ptr::null_mut()
                },
                wu: if d16 {
                    self.alloc_f16(lw.wu.len())?
                } else {
                    std::ptr::null_mut()
                },
                wd: if d16 {
                    self.alloc_f16(lw.wd.len())?
                } else {
                    std::ptr::null_mut()
                },
                moe_router: self.alloc_f16(lw.moe_router.len())?,
                moe_wg: if experts_on_device {
                    std::ptr::null_mut()
                } else {
                    self.alloc_f16(lw.moe_wg.len())?
                },
                moe_wu: if experts_on_device {
                    std::ptr::null_mut()
                } else {
                    self.alloc_f16(lw.moe_wu.len())?
                },
                moe_wd: if experts_on_device {
                    std::ptr::null_mut()
                } else {
                    self.alloc_f16(lw.moe_wd.len())?
                },
                mla_q_a: self.alloc_f16(lw.mla_q_a.len())?,
                mla_q_b: self.alloc_f16(lw.mla_q_b.len())?,
                mla_q_rope: self.alloc_f16(lw.mla_q_rope.len())?,
                mla_kv_a: self.alloc_f16(lw.mla_kv_a.len())?,
                mla_kv_b: self.alloc_f16(lw.mla_kv_b.len())?,
                mla_o: self.alloc_f16(lw.mla_o.len())?,
                shared_wg: self.alloc_f16(lw.shared_wg.len())?,
                shared_wu: self.alloc_f16(lw.shared_wu.len())?,
                shared_wd: self.alloc_f16(lw.shared_wd.len())?,
                gdn_in_qkv: if d16 && !lw.gdn_in_qkv.is_empty() {
                    self.alloc_f16(lw.gdn_in_qkv.len())?
                } else {
                    std::ptr::null_mut()
                },
                gdn_in_z: if d16 && !lw.gdn_in_z.is_empty() {
                    self.alloc_f16(lw.gdn_in_z.len())?
                } else {
                    std::ptr::null_mut()
                },
                gdn_in_a: if lw.gdn_in_a.is_empty() {
                    std::ptr::null_mut()
                } else {
                    self.alloc_f16(lw.gdn_in_a.len())?
                },
                gdn_in_b: if lw.gdn_in_b.is_empty() {
                    std::ptr::null_mut()
                } else {
                    self.alloc_f16(lw.gdn_in_b.len())?
                },
                gdn_out: if d16 && !lw.gdn_out.is_empty() {
                    self.alloc_f16(lw.gdn_out.len())?
                } else {
                    std::ptr::null_mut()
                },
            };
            if d16 {
                self.upload_f16_bits(l16.wq, &lw.wq.dequantize_f16())?;
                self.upload_f16_bits(l16.wk, &lw.wk.dequantize_f16())?;
                self.upload_f16_bits(l16.wv, &lw.wv.dequantize_f16())?;
                self.upload_f16_bits(l16.wo, &lw.wo.dequantize_f16())?;
                self.upload_f16_bits(l16.wg, &lw.wg.dequantize_f16())?;
                self.upload_f16_bits(l16.wu, &lw.wu.dequantize_f16())?;
                self.upload_f16_bits(l16.wd, &lw.wd.dequantize_f16())?;
            }
            self.upload_f16(l16.moe_router, &lw.moe_router)?;
            if experts_on_device {
                // Expert pool stays raw Q4: upload the packed bytes + per-group
                // scales as-is (in-kernel dequant reads them directly). Mixed
                // checkpoints (dense + MoE layers) push a zero-length entry
                // for the dense layers — never read (the layer dispatch
                // skips layers without a router), but index-aligned with
                // layers_dev/layers_f16.
                let q4 = Q4ExpertDev {
                    wg_q: self.upload_bytes(lw.moe_wg.q_bytes())?,
                    wg_s: self.upload_f32s(lw.moe_wg.scales())?,
                    wu_q: self.upload_bytes(lw.moe_wu.q_bytes())?,
                    wu_s: self.upload_f32s(lw.moe_wu.scales())?,
                    wd_q: self.upload_bytes(lw.moe_wd.q_bytes())?,
                    wd_s: self.upload_f32s(lw.moe_wd.scales())?,
                };
                self.q4_experts.push(q4);
            } else {
                self.upload_f16_bits(l16.moe_wg, &lw.moe_wg.dequantize_f16())?;
                self.upload_f16_bits(l16.moe_wu, &lw.moe_wu.dequantize_f16())?;
                self.upload_f16_bits(l16.moe_wd, &lw.moe_wd.dequantize_f16())?;
            }
            self.upload_f16_bits(l16.mla_q_a, &lw.mla_q_a.dequantize_f16())?;
            self.upload_f16_bits(l16.mla_q_b, &lw.mla_q_b.dequantize_f16())?;
            self.upload_f16_bits(l16.mla_q_rope, &lw.mla_q_rope.dequantize_f16())?;
            self.upload_f16_bits(l16.mla_kv_a, &lw.mla_kv_a.dequantize_f16())?;
            self.upload_f16_bits(l16.mla_kv_b, &lw.mla_kv_b.dequantize_f16())?;
            self.upload_f16_bits(l16.mla_o, &lw.mla_o.dequantize_f16())?;
            // Shared experts: the Q4/FP8 pool stays packed for the routed
            // experts, but the shared MLP is small and always dequantized.
            self.upload_f16_bits(l16.shared_wg, &lw.shared_wg.dequantize_f16())?;
            self.upload_f16_bits(l16.shared_wu, &lw.shared_wu.dequantize_f16())?;
            self.upload_f16_bits(l16.shared_wd, &lw.shared_wd.dequantize_f16())?;
            // GDN (Qwen3.5): qkv/z/out dequantize from the packed tensor; a/b
            // are stored f32 and convert like the other f16 GEMM matrices.
            // Dense Q4-on-device keeps qkv/z/out raw; a/b still take the f16
            // copy (unquantized in `WeightsQ4`).
            if !lw.gdn_in_qkv.is_empty() {
                if d16 {
                    self.upload_f16_bits(l16.gdn_in_qkv, &lw.gdn_in_qkv.dequantize_f16())?;
                    self.upload_f16_bits(l16.gdn_in_z, &lw.gdn_in_z.dequantize_f16())?;
                    self.upload_f16_bits(l16.gdn_out, &lw.gdn_out.dequantize_f16())?;
                }
                self.upload_f16(l16.gdn_in_a, &lw.gdn_in_a)?;
                self.upload_f16(l16.gdn_in_b, &lw.gdn_in_b)?;
            }
            if dense_on_device {
                // Local first: each field upload needs `&mut self` while the
                // push receiver holds `self.q4_layers`.
                let lq4 = Q4DenseDev {
                    wq: self.upload_q4_tensor(&lw.wq)?,
                    wk: self.upload_q4_tensor(&lw.wk)?,
                    wv: self.upload_q4_tensor(&lw.wv)?,
                    wo: self.upload_q4_tensor(&lw.wo)?,
                    wg: self.upload_q4_tensor(&lw.wg)?,
                    wu: self.upload_q4_tensor(&lw.wu)?,
                    wd: self.upload_q4_tensor(&lw.wd)?,
                    gdn_in_qkv: self.upload_q4_tensor(&lw.gdn_in_qkv)?,
                    gdn_in_z: self.upload_q4_tensor(&lw.gdn_in_z)?,
                    gdn_out: self.upload_q4_tensor(&lw.gdn_out)?,
                };
                self.q4_layers.push(lq4);
            }
            self.layers_f16.push(l16);
        }
        Ok(())
    }

    /// Uploads raw bytes to a fresh device allocation.
    fn upload_bytes(&mut self, src: &[u8]) -> Result<*mut u8, Error> {
        let p = self.dalloc(src.len())?;
        hip::memcpy(
            self.k.hip(),
            p as *mut core::ffi::c_void,
            src.as_ptr() as *const core::ffi::c_void,
            src.len(),
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )?;
        Ok(p as *mut u8)
    }

    /// Uploads a Q4 tensor as its raw packed bytes + per-group scales (dense
    /// Q4-on-device); empty tensors (absent on this layer type) return the
    /// null pair — existence stays dtype-independent (`has_gdn` etc.).
    fn upload_q4_tensor(&mut self, t: &crate::q4::Q4Tensor) -> Result<Q4TensorDev, Error> {
        if t.is_empty() {
            Ok(Q4TensorDev::EMPTY)
        } else {
            Ok(Q4TensorDev {
                q: self.upload_bytes(t.q_bytes())?,
                s: self.upload_f32s(t.scales())?,
            })
        }
    }

    /// Uploads f32s to a fresh device allocation.
    fn upload_f32s(&mut self, src: &[f32]) -> Result<*mut f32, Error> {
        let p = self.dalloc(src.len() * 4)?;
        self.upload(p, src)?;
        Ok(p)
    }

    /// Uploads storage-FP8 weights as f16 (dense/MoE/MLA F16 path): norms and
    /// biases stay f32; GEMM matrices are dequantized per tensor and the f16
    /// buffer is freed after each upload. The router keeps its exact f32 copy
    /// in `LayerDev` (FP8 does not quantize it) plus the usual f16 copy for the
    /// fp16 GEMM path.
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
                mla_q_a_norm: self.upload_opt(&lw.mla_q_a_norm)?,
                mla_q_b: std::ptr::null_mut(),
                mla_q_rope: std::ptr::null_mut(),
                mla_kv_a: std::ptr::null_mut(),
                mla_kv_a_norm: self.upload_opt(&lw.mla_kv_a_norm)?,
                mla_kv_b: std::ptr::null_mut(),
                mla_o: std::ptr::null_mut(),
                moe_router: self.upload_opt(&lw.moe_router)?,
                moe_wg: std::ptr::null_mut(),
                moe_wu: std::ptr::null_mut(),
                moe_wd: std::ptr::null_mut(),
                // Shared experts live in the f16 copy only (the Q4/FP8 paths
                // always run f16 GEMMs); the f32 pointers stay null.
                shared_wg: std::ptr::null_mut(),
                shared_wu: std::ptr::null_mut(),
                shared_wd: std::ptr::null_mut(),
                has_shared: !lw.shared_wg.is_empty(),
                // GDN: the GEMM matrices live in the f16 copy below; the small
                // f32 tensors (conv taps, A_log/dt_bias, gated-norm weight)
                // stay f32. `has_gdn` is the dtype-independent dispatch flag.
                has_gdn: !lw.gdn_in_qkv.is_empty(),
                gdn_in_qkv: std::ptr::null_mut(),
                gdn_in_z: std::ptr::null_mut(),
                gdn_in_a: std::ptr::null_mut(),
                gdn_in_b: std::ptr::null_mut(),
                gdn_out: std::ptr::null_mut(),
                gdn_conv_w: self.upload_opt(&lw.gdn_conv_w)?,
                gdn_a_log: self.upload_opt(&lw.gdn_a_log)?,
                gdn_dt_bias: self.upload_opt(&lw.gdn_dt_bias)?,
                gdn_norm: self.upload_opt(&lw.gdn_norm)?,
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
                moe_router: self.alloc_f16(lw.moe_router.len())?,
                moe_wg: self.alloc_f16(lw.moe_wg.len())?,
                moe_wu: self.alloc_f16(lw.moe_wu.len())?,
                moe_wd: self.alloc_f16(lw.moe_wd.len())?,
                mla_q_a: self.alloc_f16(lw.mla_q_a.len())?,
                mla_q_b: self.alloc_f16(lw.mla_q_b.len())?,
                mla_q_rope: self.alloc_f16(lw.mla_q_rope.len())?,
                mla_kv_a: self.alloc_f16(lw.mla_kv_a.len())?,
                mla_kv_b: self.alloc_f16(lw.mla_kv_b.len())?,
                mla_o: self.alloc_f16(lw.mla_o.len())?,
                shared_wg: self.alloc_f16(lw.shared_wg.len())?,
                shared_wu: self.alloc_f16(lw.shared_wu.len())?,
                shared_wd: self.alloc_f16(lw.shared_wd.len())?,
                gdn_in_qkv: if lw.gdn_in_qkv.is_empty() {
                    std::ptr::null_mut()
                } else {
                    self.alloc_f16(lw.gdn_in_qkv.len())?
                },
                gdn_in_z: if lw.gdn_in_z.is_empty() {
                    std::ptr::null_mut()
                } else {
                    self.alloc_f16(lw.gdn_in_z.len())?
                },
                gdn_in_a: if lw.gdn_in_a.is_empty() {
                    std::ptr::null_mut()
                } else {
                    self.alloc_f16(lw.gdn_in_a.len())?
                },
                gdn_in_b: if lw.gdn_in_b.is_empty() {
                    std::ptr::null_mut()
                } else {
                    self.alloc_f16(lw.gdn_in_b.len())?
                },
                gdn_out: if lw.gdn_out.is_empty() {
                    std::ptr::null_mut()
                } else {
                    self.alloc_f16(lw.gdn_out.len())?
                },
            };
            self.upload_f16_bits(l16.wq, &lw.wq.dequantize_f16())?;
            self.upload_f16_bits(l16.wk, &lw.wk.dequantize_f16())?;
            self.upload_f16_bits(l16.wv, &lw.wv.dequantize_f16())?;
            self.upload_f16_bits(l16.wo, &lw.wo.dequantize_f16())?;
            self.upload_f16_bits(l16.wg, &lw.wg.dequantize_f16())?;
            self.upload_f16_bits(l16.wu, &lw.wu.dequantize_f16())?;
            self.upload_f16_bits(l16.wd, &lw.wd.dequantize_f16())?;
            self.upload_f16(l16.moe_router, &lw.moe_router)?;
            self.upload_f16_bits(l16.moe_wg, &lw.moe_wg.dequantize_f16())?;
            self.upload_f16_bits(l16.moe_wu, &lw.moe_wu.dequantize_f16())?;
            self.upload_f16_bits(l16.moe_wd, &lw.moe_wd.dequantize_f16())?;
            self.upload_f16_bits(l16.mla_q_a, &lw.mla_q_a.dequantize_f16())?;
            self.upload_f16_bits(l16.mla_q_b, &lw.mla_q_b.dequantize_f16())?;
            self.upload_f16_bits(l16.mla_q_rope, &lw.mla_q_rope.dequantize_f16())?;
            self.upload_f16_bits(l16.mla_kv_a, &lw.mla_kv_a.dequantize_f16())?;
            self.upload_f16_bits(l16.mla_kv_b, &lw.mla_kv_b.dequantize_f16())?;
            self.upload_f16_bits(l16.mla_o, &lw.mla_o.dequantize_f16())?;
            // Shared experts: the Q4/FP8 pool stays packed for the routed
            // experts, but the shared MLP is small and always dequantized.
            self.upload_f16_bits(l16.shared_wg, &lw.shared_wg.dequantize_f16())?;
            self.upload_f16_bits(l16.shared_wu, &lw.shared_wu.dequantize_f16())?;
            self.upload_f16_bits(l16.shared_wd, &lw.shared_wd.dequantize_f16())?;
            // GDN (Qwen3.5): qkv/z/out dequantize from the packed tensor; a/b
            // are stored f32 and convert like the other f16 GEMM matrices.
            if !lw.gdn_in_qkv.is_empty() {
                self.upload_f16_bits(l16.gdn_in_qkv, &lw.gdn_in_qkv.dequantize_f16())?;
                self.upload_f16_bits(l16.gdn_in_z, &lw.gdn_in_z.dequantize_f16())?;
                self.upload_f16_bits(l16.gdn_out, &lw.gdn_out.dequantize_f16())?;
                self.upload_f16(l16.gdn_in_a, &lw.gdn_in_a)?;
                self.upload_f16(l16.gdn_in_b, &lw.gdn_in_b)?;
            }
            self.layers_f16.push(l16);
        }
        Ok(())
    }

    /// Zeroes KV caches and resets all sequence lengths.
    pub fn reset_state(&mut self) -> Result<(), Error> {
        let kv_elem = if self.cfg.dtype == ModelDType::F16 {
            2
        } else {
            4
        };
        let kv_bytes =
            self.batch * self.cfg.max_seq_len * self.cfg.n_kv_heads * self.cfg.head_dim * kv_elem;
        for (kc, vc) in &self.kv_cache {
            unsafe {
                hip::check(
                    self.k.hip(),
                    (self.k.hip().api.hip_memset)(*kc as *mut _, 0, kv_bytes),
                )?;
                hip::check(
                    self.k.hip(),
                    (self.k.hip().api.hip_memset)(*vc as *mut _, 0, kv_bytes),
                )?;
            }
        }
        if !self.mla_kv_cache.is_empty() {
            let c = self.cfg;
            let heads = c.n_heads;
            let k_bytes =
                self.batch * c.max_seq_len * heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) * 4;
            let v_bytes = self.batch * c.max_seq_len * heads * c.v_head_dim * 4;
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
        // GDN recurrent + conv state back to the pos-0 state (S = 0, empty
        // conv window) — same contract as the KV caches above.
        if !self.gdn_state.is_empty() {
            let c = self.cfg;
            let state_bytes = self.batch * c.gdn_v_heads * c.gdn_key_dim() * c.gdn_value_dim() * 4;
            let conv_dim = 2 * c.gdn_key_dim() + c.gdn_value_dim();
            let conv_bytes = self.batch * conv_dim * (c.gdn_conv_kernel - 1) * 4;
            for (&s, &cv) in self.gdn_state.iter().zip(&self.gdn_conv) {
                unsafe {
                    hip::check(
                        self.k.hip(),
                        (self.k.hip().api.hip_memset)(s as *mut _, 0, state_bytes),
                    )?;
                    hip::check(
                        self.k.hip(),
                        (self.k.hip().api.hip_memset)(cv as *mut _, 0, conv_bytes),
                    )?;
                }
            }
        }
        for l in self.lens.iter_mut() {
            *l = 0;
        }
        unsafe {
            for i in 0..self.batch {
                *self.pos_host.add(i) = 0;
            }
            hip::memcpy_async(
                self.k.hip(),
                self.pos_dev as *mut core::ffi::c_void,
                self.pos_host as *const core::ffi::c_void,
                self.batch * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
                self.k.stream,
            )?;
        }
        self.k.sync()?;
        Ok(())
    }

    /// Runs a batched decode step for `tokens` (one per sequence), returning the
    /// greedy-sampled next token for each sequence.
    pub fn decode_step(&mut self, tokens: &[u32]) -> Result<Vec<u32>, Error> {
        assert_eq!(tokens.len(), self.batch, "tokens length must equal batch");
        for (i, l) in self.lens.iter().enumerate() {
            if *l as usize >= self.cfg.max_seq_len {
                return Err(Error::Model(format!("seq {i} exceeds max_seq_len")));
            }
        }
        let b = self.batch;
        if self.graph_capture_ok(b, true) {
            // Greedy graph path (#103): stage every pinned mirror, then one
            // replay covers uploads + run_kernels + argmax; only the D2H
            // readback stays outside.
            unsafe {
                for (i, (&t, &l)) in tokens.iter().zip(&self.lens).enumerate() {
                    *self.tokens_host.add(i) = t as i32;
                    *self.pos_host.add(i) = l as i32;
                    *self.slots_host.add(i) = i as i32; // row == slot for decode
                    *self.run_mask_host.add(i) = 0;
                    self.last_row_by_slot[i] = i;
                }
            }
            if self.paged {
                // The graph always re-uploads the offsets: keep the staging at
                // the identity mapping this entry point guarantees.
                let identity: Vec<u32> = (0..b as u32).collect();
                self.stage_table_offsets(&identity);
            }
            if self.greedy_graph.is_none() {
                self.capture_greedy_decode_graph()?;
            }
            // SAFETY: every captured buffer is owned by `self` and alive; the
            // pinned staging mirrors were refreshed above on this thread.
            unsafe { self.greedy_graph.as_deref().unwrap().replay()? }
            let next = self.sample_readback(b)?;
            // Only now is the device offsets buffer known to hold the identity
            // mapping (the graph's recorded memcpy uploaded it): clearing the
            // dirty flag earlier would let a failed capture/replay leave a
            // stale non-identity table in place while the eager fallback
            // skips its refresh.
            self.offsets_dirty = false;
            for l in self.lens.iter_mut() {
                *l += 1;
            }
            return Ok(next);
        }
        unsafe {
            for (i, (&t, &l)) in tokens.iter().zip(&self.lens).enumerate() {
                *self.tokens_host.add(i) = t as i32;
                *self.pos_host.add(i) = l as i32;
                *self.slots_host.add(i) = i as i32; // row == slot for decode
                self.last_row_by_slot[i] = i;
            }
            hip::memcpy_async(
                self.k.hip(),
                self.tokens_dev as *mut core::ffi::c_void,
                self.tokens_host as *const core::ffi::c_void,
                self.batch * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
                self.k.stream,
            )?;
            hip::memcpy_async(
                self.k.hip(),
                self.pos_dev as *mut core::ffi::c_void,
                self.pos_host as *const core::ffi::c_void,
                self.batch * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
                self.k.stream,
            )?;
            hip::memcpy_async(
                self.k.hip(),
                self.slots_dev as *mut core::ffi::c_void,
                self.slots_host as *const core::ffi::c_void,
                self.batch * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
                self.k.stream,
            )?;
            // Greedy decode: every row is a single decode position (no runs).
            for i in 0..self.batch {
                *self.run_mask_host.add(i) = 0;
            }
            hip::memcpy_async(
                self.k.hip(),
                self.run_mask_dev as *mut core::ffi::c_void,
                self.run_mask_host as *const core::ffi::c_void,
                self.batch * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
                self.k.stream,
            )?;
        }
        self.refresh_offsets_if_dirty()?;
        self.run_kernels(
            self.batch as i32,
            self.slots_dev,
            self.run_mask_dev,
            true,
            0,
        )?;
        let next = self.sample(self.batch)?;
        // Step profiler: the sampling sync above drained the stream, so all
        // event records have executed — safe to read elapsed times.
        if let Some(pf_state) = self.prof.as_mut() {
            pf_state.report();
        }
        for l in self.lens.iter_mut() {
            *l += 1;
        }
        Ok(next)
    }

    /// Paged mode: refills the per-row table offsets for the next step's
    /// active rows (`offsets[row] = slot[row] * pages_per_seq`) and uploads
    /// them stream-ordered before the kernels. Prefill packing means `row ==
    /// slot` no longer holds — every row addresses its sequence's pages via
    /// this offset. No-op outside paged mode.
    fn refresh_table_offsets(&mut self, slots: &[u32]) -> Result<(), Error> {
        self.stage_table_offsets(slots);
        self.upload_table_offsets(slots.len())
    }

    /// Host half of [`Self::refresh_table_offsets`]: refills the pinned
    /// offsets staging only. Split from the upload so decode-graph capture can
    /// record the memcpy (the graph then re-reads this staging on replay)
    /// while the per-step host write stays outside the graph. No-op outside
    /// paged mode.
    fn stage_table_offsets(&mut self, slots: &[u32]) {
        if !self.paged {
            return;
        }
        debug_assert!(slots.len() <= self.rows, "active rows exceed capacity");
        unsafe {
            for (r, &s) in slots.iter().enumerate() {
                *self.offsets_host.add(r) = (s as usize * self.max_pages_per_seq) as i32;
            }
        }
    }

    /// Device half of [`Self::refresh_table_offsets`]: stream-ordered H2D of
    /// the pinned offsets staging. No-op outside paged mode.
    fn upload_table_offsets(&self, n: usize) -> Result<(), Error> {
        if !self.paged {
            return Ok(());
        }
        hip::memcpy_async(
            self.k.hip(),
            self.table_offsets as *mut core::ffi::c_void,
            self.offsets_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.k.stream,
        )?;
        Ok(())
    }

    /// `decode_step` fast path: refresh the offsets only when a previous
    /// explicit step left them non-identity. `decode_step` always packs
    /// rows == slots 0..batch, matching the initial identity upload.
    fn refresh_offsets_if_dirty(&mut self) -> Result<(), Error> {
        if self.offsets_dirty {
            self.refresh_table_offsets(&(0..self.batch as u32).collect::<Vec<_>>())?;
            self.offsets_dirty = false;
        }
        Ok(())
    }

    /// Device half of the decode-step input staging: tokens/pos/slots/run_mask
    /// (+ paged table offsets) H2D from the pinned mirrors. When captured into
    /// a decode graph these become memcpy nodes that re-copy the current
    /// staging contents on every replay, so the host only rewrites the pinned
    /// buffers between steps.
    fn upload_decode_inputs(&self, n: usize) -> Result<(), Error> {
        hip::memcpy_async(
            self.k.hip(),
            self.tokens_dev as *mut core::ffi::c_void,
            self.tokens_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.k.stream,
        )?;
        hip::memcpy_async(
            self.k.hip(),
            self.pos_dev as *mut core::ffi::c_void,
            self.pos_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.k.stream,
        )?;
        hip::memcpy_async(
            self.k.hip(),
            self.slots_dev as *mut core::ffi::c_void,
            self.slots_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.k.stream,
        )?;
        hip::memcpy_async(
            self.k.hip(),
            self.run_mask_dev as *mut core::ffi::c_void,
            self.run_mask_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.k.stream,
        )?;
        self.upload_table_offsets(n)
    }

    /// #103: a step is graph-capturable only when it is a pure decode step on
    /// the custom-GEMV path with a fully device-driven MoE and no outside
    /// stream/sync involvement:
    /// - `decode_only` (prefill/mixed steps keep the hipBLAS/host-loop paths,
    ///   which sync and read counts back to the host);
    /// - `n <= GEMV_MAX_M` so every projection's grid is fixed at capture
    ///   time, and every projection's reduction width `<= GEMV_MAX_D` so
    ///   every projection takes the custom GEMV kernels — the hipBLAS
    ///   `GemmEx` fallback may lazily allocate workspace, which aborts
    ///   capture. The widths are the input side of each projection: dense
    ///   models use d / nq / inter, MLA adds q_lora_rank / kv_lora_rank;
    /// - f16 (f32 routes even small GEMMs through hipBLAS);
    /// - no step profiler (event records would be baked into the graph) and no
    ///   prefetch engine (its cross-stream H2D pipeline cannot be captured);
    /// - MoE models must take the device-driven grouped path with all experts
    ///   resident (no CPU-expert host loop).
    fn graph_capture_ok(&self, n: usize, decode_only: bool) -> bool {
        if !(self.graph_enabled && decode_only)
            || self.prof.is_some()
            || self.prefetch.is_some()
            || self.layer_dump.is_some()
            || self.cfg.dtype != ModelDType::F16
            || n == 0
            || n > GEMV_MAX_M as usize
        {
            return false;
        }
        let c = &self.cfg;
        let max_kk = if c.kv_lora_rank > 0 {
            // MLA: q_a(d) / kv_a(d) / q_b(q_lora_rank) / kv_b(kv_lora_rank)
            // / o(n_heads * v_head_dim).
            c.d_model
                .max(c.q_lora_rank)
                .max(c.kv_lora_rank)
                .max(c.n_heads * c.v_head_dim)
        } else {
            // Dense/MoE: q/k/v + router + lm_head(d), o(nq), dense-MLP
            // down(inter); MoE experts ride the device-driven grouped kernels
            // (required below), not the gemm closure. Qwen3.5 adds the GDN
            // projections (all contract over d, except out_proj over
            // gdn_value_dim) and the doubled q_proj row count (n, not kk —
            // the GEMV grid scales with it, n is unbounded by GEMV_MAX_D).
            c.d_model
                .max(c.n_heads * c.head_dim)
                .max(c.intermediate_size)
                .max(c.gdn_value_dim())
        };
        if max_kk > GEMV_MAX_D as usize {
            return false;
        }
        if c.num_experts > 0 {
            let grouped = self.moe_grouped || !self.q4_experts.is_empty();
            if !grouped || self.expert_slots < c.num_experts {
                return false;
            }
        }
        true
    }

    /// Enables/disables decode-graph capture (MACH_GRAPH=1 sets the initial
    /// value at construction; tests use this setter instead of the process-wide
    /// env). Existing captured graphs stay cached.
    pub fn set_decode_graph_enabled(&mut self, on: bool) {
        self.graph_enabled = on;
    }

    /// Debug (issue #107): after every forward, write the LAST active row's
    /// per-layer residual (post full layer — attention + MLP/MoE residual, the
    /// same point `.scratch/np_ref.py` records) to `<dir>/hs_L{li:02}.npy`
    /// (f32, C-order). Files always carry the most recent step. Forces graph
    /// capture off — the per-layer D2H read cannot be captured. Library-code
    /// debug hook: the server/examples choose the directory; no env is read
    /// here.
    ///
    /// Interaction caveat: the per-layer sync + D2H + file write lands inside
    /// the [`MACH_STEP_PROFILE`] phase brackets, so phase timings measured
    /// with the dump armed include dump I/O — don't combine the two knobs for
    /// performance analysis.
    pub fn set_layer_dump(&mut self, dir: impl Into<std::path::PathBuf>) -> Result<(), Error> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::Model(format!("layer dump dir {}: {e}", dir.display())))?;
        self.graph_enabled = false;
        self.decode_graphs.clear();
        self.greedy_graph = None;
        self.layer_dump = Some(dir);
        Ok(())
    }

    /// Turns the layer dump off (files stay on disk). The dump example uses
    /// this before generating, so the prompt's final-step files survive.
    /// Graph capture stays disabled — re-enabling is the caller's call
    /// ([`Self::set_decode_graph_enabled`]); the prior setting is not
    /// remembered.
    pub fn clear_layer_dump(&mut self) {
        self.layer_dump = None;
    }

    /// Debug (issue #107): syncs the engine stream and snapshots the LAST
    /// active row of the residual `self.x` (`d` f32) to `<dir>/<name>.npy`.
    /// Shared by the embedding / mid-layer / layer-tail dump sites, which
    /// only vary in the file name.
    fn dump_last_x_row(
        &self,
        dir: &std::path::Path,
        name: &str,
        b: i32,
        d: i32,
    ) -> Result<(), Error> {
        let mut hs = vec![0.0f32; d as usize];
        self.k.sync()?;
        let src = (self.x as usize + b.saturating_sub(1) as usize * d as usize * 4)
            as *const core::ffi::c_void;
        hip::memcpy(
            self.k.hip(),
            hs.as_mut_ptr() as *mut core::ffi::c_void,
            src,
            d as usize * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )?;
        write_npy_f32(&dir.join(name), &hs)
    }

    /// Number of captured decode graphs (one per active-row-count bucket seen
    /// so far). Diagnostics/tests for the #103 graph path.
    #[must_use]
    pub fn decode_graph_count(&self) -> usize {
        self.decode_graphs.len()
    }

    /// Captures the greedy [`Self::decode_step`] sequence (input uploads +
    /// `run_kernels` + argmax) into a single HIP graph. Same capture-then-
    /// immediately-replay contract as [`Self::capture_decode_graph`].
    fn capture_greedy_decode_graph(&mut self) -> Result<(), Error> {
        let cap =
            mach_engine::hip::HipGraphCapture::with_stream(Arc::clone(self.k.hip()), self.k.stream)
                .inspect_err(|_| self.graph_enabled = false)?;
        // Any capture failure permanently disables the graph path: a caller
        // must fall back to eager rather than re-failing every step.
        if let Err(e) = cap.prepare() {
            self.graph_enabled = false;
            return Err(e.into());
        }
        self.k.sync()?;
        if let Err(e) = cap.begin() {
            self.graph_enabled = false;
            return Err(e.into());
        }
        let record = (|| {
            self.upload_decode_inputs(self.batch)?;
            self.run_kernels(
                self.batch as i32,
                self.slots_dev,
                self.run_mask_dev,
                true,
                0,
            )?;
            self.sample_device(self.batch)
        })();
        // `end` must run even when recording failed, to leave capture mode.
        match (record, cap.end()) {
            (Ok(()), Ok(g)) => {
                self.greedy_graph = Some(g);
                eprintln!("greedy decode graph captured (batch={})", self.batch);
                Ok(())
            }
            (Err(e), _) => {
                self.graph_enabled = false;
                Err(e)
            }
            (Ok(()), Err(e)) => {
                self.graph_enabled = false;
                Err(e.into())
            }
        }
    }

    /// Captures one pure-decode step (input uploads + `run_kernels` + sampler
    /// upload/kernel) into a HIP graph keyed by the active row count `n`.
    /// Stream capture records without executing, so the first replay right
    /// after capture performs exactly the step that was just recorded — no
    /// separate warmup is needed (all kernels were compiled at
    /// [`HipKernels::new`] and all device buffers were allocated at
    /// construction, so the capture window allocates and compiles nothing).
    fn capture_decode_graph(&mut self, n: i32) -> Result<(), Error> {
        let cap =
            mach_engine::hip::HipGraphCapture::with_stream(Arc::clone(self.k.hip()), self.k.stream)
                .inspect_err(|_| self.graph_enabled = false)?;
        // Any capture failure permanently disables the graph path: a caller
        // must fall back to eager rather than re-failing every step.
        if let Err(e) = cap.prepare() {
            self.graph_enabled = false;
            return Err(e.into());
        }
        self.k.sync()?;
        if let Err(e) = cap.begin() {
            self.graph_enabled = false;
            return Err(e.into());
        }
        let record = (|| {
            self.upload_decode_inputs(n as usize)?;
            self.run_kernels(n, self.slots_dev, self.run_mask_dev, true, 0)?;
            self.sampler.sample_batched_upload(n as usize)?;
            self.sampler
                .sample_batched_kernel(self.logits, n, self.cfg.vocab_size)
        })();
        // `end` must run even when recording failed, to leave capture mode.
        match (record, cap.end()) {
            (Ok(()), Ok(g)) => {
                self.decode_graphs.insert(n, g);
                eprintln!(
                    "decode graph captured: n={n} ({} total)",
                    self.decode_graphs.len()
                );
                Ok(())
            }
            (Err(e), _) => {
                self.graph_enabled = false; // don't retry a failing shape
                Err(e)
            }
            (Ok(()), Err(e)) => {
                self.graph_enabled = false;
                Err(e.into())
            }
        }
    }

    fn run_kernels(
        &self,
        count: i32,
        slots: *const i32,
        run_mask: *const i32,
        decode_only: bool,
        num_runs: i32,
    ) -> Result<(), Error> {
        let c = self.cfg;
        let b = count;
        let d = c.d_model as i32;
        let nq = (c.n_heads * c.head_dim) as i32;
        let nkv = (c.n_kv_heads * c.head_dim) as i32;
        let inter = c.intermediate_size as i32;
        let scale = c.attn_scale(c.head_dim);
        let k = &self.k;
        let f16 = c.dtype == ModelDType::F16;

        // Selects fp16 (cast activations + fp16 weights, fp32 output) or f32.
        // Small m (decode rows) takes the custom GEMV kernel — rocBLAS m=1
        // runs the 30B-class decode 60x over the memory bound; large m keeps
        // hipBLAS (weight reuse across rows pays on tensor cores).
        // `t` is the dense Q4-on-device pair for this tensor (null in every
        // other mode): when present, the raw packed tensor + scales go to
        // `gemv_q4` for EVERY step — no f16/f32 copy exists to fall back on
        // (same contract as the Q4 expert pool), and that kernel is
        // batch-general (prefill rows included) with no 48 KB staging limit
        // (oversize contractions read x straight from global).
        let gemm = |out: *mut f32,
                    x: *const f32,
                    w32: *mut f32,
                    w16: *mut u16,
                    t: Q4TensorDev,
                    n: i32,
                    kk: i32|
         -> Result<(), Error> {
            if !t.is_null() {
                return k.launch_gemv_q4(x, t.q, t.s, out, n, kk, b);
            }
            if f16 {
                if b <= GEMV_MAX_M && kk <= GEMV_MAX_D {
                    k.launch_gemv_f16(out, x, w16, n, kk, b, std::ptr::null_mut())
                } else {
                    k.gemm_batched_f16(out, x, w16, b, n, kk, self.xh, self.yh)
                }
            } else {
                k.gemm_batched(out, x, w32, b, n, kk)
            }
        };

        if !self.emb_q4.is_null() {
            // Dense Q4-on-device: embedding rows dequantize in the gather.
            k.launch_embed_gather_q4(self.tokens_dev, self.emb_q4.q, self.emb_q4.s, self.x, d, b)?;
        } else if f16 {
            k.launch_embed_f16(self.tokens_dev, self.emb_f16, self.x, d, b)?;
        } else {
            k.launch_embed_batched(self.tokens_dev, self.emb_dev, self.x, d, b)?;
        }
        // Debug (issue #107): pre-layer-0 embedding snapshot (row 0 of the
        // diff chain — if this is off, nothing downstream matters).
        if let Some(dir) = &self.layer_dump {
            self.dump_last_x_row(dir, "hs_emb.npy", b, d)?;
        }
        // Step profiler (MACH_STEP_PROFILE): whole-step start bracket.
        if let Some(pf_state) = &self.prof {
            unsafe {
                hip::check(
                    self.k.hip(),
                    (self.k.hip().api.hip_event_record)(pf_state.step_events.0, self.k.stream),
                )?;
            }
        }
        // Double-buffered prefill: start the first MoE layer's H2D prefetch so
        // it overlaps the first layers' compute.
        if let Some(pf) = &self.prefetch {
            pf.begin()?;
        }
        for (li, lw) in self.layers_dev.iter().enumerate() {
            let l16 = if f16 { Some(self.layers_f16[li]) } else { None };
            // Dense Q4-on-device pairs for this layer's big tensors (`None`
            // in every other mode; a layer type without the tensor maps to
            // the null pair, so the branch is inert for it).
            let lq4 = self.q4_layers.get(li);
            // Step profiler: bracket this layer's attention phase.
            if let Some(pf_state) = &self.prof
                && let Some((l0, _, _)) = pf_state.events.get(li)
            {
                unsafe {
                    hip::check(
                        self.k.hip(),
                        (self.k.hip().api.hip_event_record)(*l0, self.k.stream),
                    )?;
                }
            }
            if let Some(pf) = &self.prefetch {
                pf.layer_begin(li)?;
            }
            k.launch_rms_norm(self.x, lw.rms_attn, self.xn, b, d, c.rms_eps)?;
            if c.kv_lora_rank > 0 {
                let heads = c.n_heads as i32;
                let qlr = c.q_lora_rank as i32;
                let nope = c.qk_nope_head_dim as i32;
                let rope = c.qk_rope_head_dim as i32;
                let v_hd = c.v_head_dim as i32;
                let kvlr = c.kv_lora_rank as i32;
                let max_seq = c.max_seq_len as i32;
                // DeepSeek-V2-Lite ships `q_lora_rank: null`: the fused q
                // projection reads the layer input directly, so both halves
                // contract over `d` instead of a normalized low-rank q.
                let low_rank_q = c.q_lora_rank > 0;
                let q_kk = if low_rank_q { qlr } else { d };
                // q_lora = q_a(xn); rms; q_nope = q_b(q_lora).
                if low_rank_q {
                    gemm(
                        self.mla_q_lora,
                        self.xn,
                        lw.mla_q_a,
                        l16.map_or(std::ptr::null_mut(), |l| l.mla_q_a),
                        Q4TensorDev::EMPTY,
                        qlr,
                        d,
                    )?;
                    k.launch_rms_norm(
                        self.mla_q_lora,
                        lw.mla_q_a_norm,
                        self.mla_q_lora_n,
                        b,
                        qlr,
                        c.rms_eps,
                    )?;
                }
                let q_src = if low_rank_q {
                    self.mla_q_lora_n
                } else {
                    self.xn
                };
                gemm(
                    self.q_nope,
                    q_src,
                    lw.mla_q_b,
                    l16.map_or(std::ptr::null_mut(), |l| l.mla_q_b),
                    Q4TensorDev::EMPTY,
                    heads * nope,
                    q_kk,
                )?;
                // q_rope = q_rope_proj(xn) + RoPE (k_buf is scratch here).
                gemm(
                    self.q_rope,
                    q_src,
                    lw.mla_q_rope,
                    l16.map_or(std::ptr::null_mut(), |l| l.mla_q_rope),
                    Q4TensorDev::EMPTY,
                    heads * rope,
                    q_kk,
                )?;
                k.launch_rope_batched(
                    self.q_rope,
                    self.k_buf,
                    self.pos_dev,
                    b,
                    heads,
                    heads,
                    rope,
                    rope,
                    RopeParams::from(c),
                )?;
                // compressed_kv = kv_a(xn); latent is followed by k_rope in
                // kv_a, so extract it to a contiguous buffer before the RMSNorm
                // (rms_norm assumes row stride == cols); shared k_rope + RoPE.
                gemm(
                    self.mla_kv_a,
                    self.xn,
                    lw.mla_kv_a,
                    l16.map_or(std::ptr::null_mut(), |l| l.mla_kv_a),
                    Q4TensorDev::EMPTY,
                    kvlr + rope,
                    d,
                )?;
                k.launch_mla_extract_kv_lora(self.mla_kv_a, self.mla_kv_lora, b, kvlr, rope)?;
                k.launch_rms_norm(
                    self.mla_kv_lora,
                    lw.mla_kv_a_norm,
                    self.mla_kv_a_n,
                    b,
                    kvlr,
                    c.rms_eps,
                )?;
                k.launch_mla_extract_k_rope(self.mla_kv_a, self.mla_k_rope, b, kvlr, rope)?;
                k.launch_rope_batched(
                    self.mla_k_rope,
                    self.v_buf,
                    self.pos_dev,
                    b,
                    1,
                    1,
                    rope,
                    rope,
                    RopeParams::from(c),
                )?;
                // kv = kv_b_proj(latent): [batch, heads*(nope + v_hd)].
                gemm(
                    self.mla_kv,
                    self.mla_kv_a_n,
                    lw.mla_kv_b,
                    l16.map_or(std::ptr::null_mut(), |l| l.mla_kv_b),
                    Q4TensorDev::EMPTY,
                    heads * (nope + v_hd),
                    kvlr,
                )?;
                // Assemble per-head q/k/v and store into the MLA caches.
                k.launch_mla_assemble_q_batched(
                    self.q_nope,
                    self.q_rope,
                    self.q,
                    b,
                    heads,
                    nope,
                    rope,
                )?;
                let (kc, vc) = self.mla_kv_cache[li];
                let scale = c.attn_scale((nope + rope) as usize);
                if self.paged {
                    // Paged MLA: one fused kernel expands kv + shared k_rope
                    // on the fly while storing through the block tables into
                    // the per-head page pools, then batched attention over
                    // them (no intermediate scratch roundtrip).
                    let tpp = self.tokens_per_page as i32;
                    let max_pages = self.max_pages_per_seq as i32;
                    k.launch_kv_store_paged_mla(
                        self.mla_kv,
                        self.mla_k_rope,
                        kc as *mut f32,
                        vc as *mut f32,
                        self.pos_dev,
                        self.table_offsets,
                        self.block_tables,
                        b,
                        heads,
                        nope,
                        rope,
                        v_hd,
                        tpp,
                    )?;
                    k.launch_attn_decode_paged_mla_batched(
                        self.q,
                        kc as *const f32,
                        vc as *const f32,
                        self.block_tables,
                        self.mla_attn,
                        self.pos_dev,
                        self.table_offsets,
                        b,
                        heads,
                        nope,
                        rope,
                        v_hd,
                        scale,
                        tpp,
                        max_pages,
                    )?;
                } else {
                    k.launch_mla_assemble_kv_batched(
                        self.mla_kv,
                        self.mla_k_rope,
                        kc as *mut f32,
                        vc as *mut f32,
                        self.pos_dev,
                        slots,
                        b,
                        max_seq,
                        heads,
                        nope,
                        rope,
                        v_hd,
                    )?;
                    k.launch_mla_attn_decode_batched(
                        self.q,
                        kc as *const f32,
                        vc as *const f32,
                        self.mla_attn,
                        self.pos_dev,
                        slots,
                        b,
                        heads,
                        nope + rope,
                        v_hd,
                        scale,
                        max_seq,
                    )?;
                }
                gemm(
                    self.proj,
                    self.mla_attn,
                    lw.mla_o,
                    l16.map_or(std::ptr::null_mut(), |l| l.mla_o),
                    Q4TensorDev::EMPTY,
                    d,
                    heads * v_hd,
                )?;
                k.launch_add(self.x, self.proj, b * d)?;
            } else if lw.has_gdn {
                // Qwen3.5 hybrid: linear-attention layer (gated DeltaNet).
                // Weights-driven dispatch (`has_gdn`, dtype-independent). The
                // kernels index the persistent state by SLOT via `slots`;
                // one row per slot per step — chunked prefill (several rows
                // of one sequence in a single step) is Stage B (#112), the
                // graph/prefill guards below keep this path decode-shaped.
                let kd = (c.gdn_k_heads * c.gdn_head_dim) as i32;
                let vd = (c.gdn_v_heads * c.gdn_head_dim) as i32;
                let kh = c.gdn_k_heads as i32;
                let vh = c.gdn_v_heads as i32;
                let hd = c.gdn_head_dim as i32;
                let conv_dim = 2 * kd + vd;
                let keep = c.gdn_conv_kernel as i32 - 1;
                gemm(
                    self.gdn_qkv_pre,
                    self.xn,
                    lw.gdn_in_qkv,
                    l16.map_or(std::ptr::null_mut(), |l| l.gdn_in_qkv),
                    lq4.map_or(Q4TensorDev::EMPTY, |l| l.gdn_in_qkv),
                    conv_dim,
                    d,
                )?;
                gemm(
                    self.gdn_z,
                    self.xn,
                    lw.gdn_in_z,
                    l16.map_or(std::ptr::null_mut(), |l| l.gdn_in_z),
                    lq4.map_or(Q4TensorDev::EMPTY, |l| l.gdn_in_z),
                    vd,
                    d,
                )?;
                gemm(
                    self.gdn_a,
                    self.xn,
                    lw.gdn_in_a,
                    l16.map_or(std::ptr::null_mut(), |l| l.gdn_in_a),
                    Q4TensorDev::EMPTY,
                    vh,
                    d,
                )?;
                gemm(
                    self.gdn_b,
                    self.xn,
                    lw.gdn_in_b,
                    l16.map_or(std::ptr::null_mut(), |l| l.gdn_in_b),
                    Q4TensorDev::EMPTY,
                    vh,
                    d,
                )?;
                k.launch_gdn_conv_update(
                    self.gdn_qkv,
                    self.gdn_qkv_pre,
                    self.gdn_conv[li],
                    slots,
                    b,
                    conv_dim,
                    keep,
                    lw.gdn_conv_w,
                )?;
                k.launch_gdn_l2norm_qk(self.gdn_qkv, b, kh, hd, conv_dim)?;
                k.launch_gdn_step(
                    self.gdn_qkv,
                    self.gdn_state[li],
                    self.gdn_o,
                    slots,
                    lw.gdn_a_log,
                    lw.gdn_dt_bias,
                    self.gdn_a,
                    self.gdn_b,
                    b,
                    vh,
                    kh,
                    hd,
                    conv_dim,
                )?;
                k.launch_gdn_gated_norm(self.gdn_o, self.gdn_z, lw.gdn_norm, b, vh, hd, c.rms_eps)?;
                gemm(
                    self.proj,
                    self.gdn_o,
                    lw.gdn_out,
                    l16.map_or(std::ptr::null_mut(), |l| l.gdn_out),
                    lq4.map_or(Q4TensorDev::EMPTY, |l| l.gdn_out),
                    d,
                    vd,
                )?;
                k.launch_add(self.x, self.proj, b * d)?;
            } else {
                // Qwen3.5 `attn_output_gate`: the doubled q_proj carries each
                // head's block as [query | gate]; split before QK-norm (the
                // gate skips norm + RoPE) and apply sigmoid after attention.
                // Small-m f16 keeps the fused QKV GEMV (the doubled q rows
                // are just more rows of the same GEMV).
                let gate_on = c.attn_output_gate;
                let nq_rows = if gate_on { 2 * nq } else { nq };
                // Dense Q4-on-device has no f16 wq (the raw pair is the only
                // copy), so the fused f16 QKV GEMV fast path is off; the
                // generic branch below runs per-tensor Q4 GEMVs.
                if f16 && b <= GEMV_MAX_M && lq4.map_or(Q4TensorDev::EMPTY, |l| l.wq).is_null() {
                    let l = l16.expect("f16 layer");
                    k.launch_gemv_f16_qkv(
                        self.xn,
                        l.wq,
                        l.wk,
                        l.wv,
                        if gate_on { self.qg } else { self.q },
                        self.k_buf,
                        self.v_buf,
                        nq_rows,
                        nkv,
                        d,
                        b,
                        std::ptr::null_mut(),
                    )?;
                } else {
                    gemm(
                        if gate_on { self.qg } else { self.q },
                        self.xn,
                        lw.wq,
                        l16.map_or(std::ptr::null_mut(), |l| l.wq),
                        lq4.map_or(Q4TensorDev::EMPTY, |l| l.wq),
                        nq_rows,
                        d,
                    )?;
                    gemm(
                        self.k_buf,
                        self.xn,
                        lw.wk,
                        l16.map_or(std::ptr::null_mut(), |l| l.wk),
                        lq4.map_or(Q4TensorDev::EMPTY, |l| l.wk),
                        nkv,
                        d,
                    )?;
                    gemm(
                        self.v_buf,
                        self.xn,
                        lw.wv,
                        l16.map_or(std::ptr::null_mut(), |l| l.wv),
                        lq4.map_or(Q4TensorDev::EMPTY, |l| l.wv),
                        nkv,
                        d,
                    )?;
                }
                if gate_on {
                    k.launch_qg_split(
                        self.qg,
                        self.q,
                        self.attn_gate,
                        b,
                        c.n_heads as i32,
                        c.head_dim as i32,
                    )?;
                }
                // Qwen2 checkpoints ship q/k/v biases.
                if !lw.bq.is_null() {
                    k.launch_add_bias(self.q, lw.bq, b, nq)?;
                }
                if !lw.bk.is_null() {
                    k.launch_add_bias(self.k_buf, lw.bk, b, nkv)?;
                }
                if !lw.bv.is_null() {
                    k.launch_add_bias(self.v_buf, lw.bv, b, nkv)?;
                }
                // Qwen3 QK-norm: per-head RMSNorm after projection, before RoPE.
                if !lw.q_norm.is_null() {
                    k.launch_qk_norm(
                        self.q,
                        self.k_buf,
                        lw.q_norm,
                        lw.k_norm,
                        b,
                        c.n_heads as i32,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        c.rms_eps,
                    )?;
                }
                // Partial rotary (Qwen3.5 full-attn layers): only the first
                // `attn_rotary_dim` coordinates rotate (== head_dim for every
                // pre-Qwen3.5 family).
                let rot = c.attn_rotary_dim() as i32;
                k.launch_rope_batched(
                    self.q,
                    self.k_buf,
                    self.pos_dev,
                    b,
                    c.n_heads as i32,
                    c.n_kv_heads as i32,
                    c.head_dim as i32,
                    rot,
                    RopeParams::from(c),
                )?;
                let (kc, vc) = self.kv_cache[li];
                if f16 && self.paged {
                    // Paged decode (F16 KV): store + attention through the
                    // block tables; same addressing as the F32 paged branch.
                    let tpp = self.tokens_per_page as i32;
                    let max_pages = self.max_pages_per_seq as i32;
                    k.launch_kv_store_paged_f16(
                        self.k_buf,
                        kc as *mut u16,
                        self.pos_dev,
                        self.table_offsets,
                        self.block_tables,
                        b,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        tpp,
                    )?;
                    k.launch_kv_store_paged_f16(
                        self.v_buf,
                        vc as *mut u16,
                        self.pos_dev,
                        self.table_offsets,
                        self.block_tables,
                        b,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        tpp,
                    )?;
                    k.launch_attn_decode_paged_f16_gqa(
                        self.q,
                        kc as *const u16,
                        vc as *const u16,
                        self.block_tables,
                        self.attn,
                        self.pos_dev,
                        self.table_offsets,
                        b,
                        c.n_heads as i32,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        scale,
                        tpp,
                        max_pages,
                    )?;
                } else if f16 {
                    k.launch_kv_store_batched_f16(
                        self.k_buf,
                        kc as *mut u16,
                        self.pos_dev,
                        slots,
                        b,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        c.max_seq_len as i32,
                    )?;
                    k.launch_kv_store_batched_f16(
                        self.v_buf,
                        vc as *mut u16,
                        self.pos_dev,
                        slots,
                        b,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        c.max_seq_len as i32,
                    )?;
                    k.launch_attn_decode_batched_f16_gqa(
                        self.q,
                        kc as *const u16,
                        vc as *const u16,
                        self.attn,
                        self.pos_dev,
                        slots,
                        run_mask,
                        b,
                        c.n_heads as i32,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        scale,
                        c.max_seq_len as i32,
                    )?;
                    // Shared-KV prefill attention for detected runs.
                    // Run descriptors are read from the pinned host copy.
                    let runs = self.runs_host;
                    for ri in 0..num_runs {
                        let qoff = unsafe { *runs.add((ri * 4) as usize) };
                        let cc = unsafe { *runs.add((ri * 4 + 1) as usize) };
                        let base = unsafe { *runs.add((ri * 4 + 2) as usize) };
                        let slot = unsafe { *runs.add((ri * 4 + 3) as usize) };
                        k.launch_attn_prefill_f16(
                            self.q,
                            kc as *const u16,
                            vc as *const u16,
                            self.attn,
                            qoff,
                            cc,
                            base,
                            c.n_heads as i32,
                            c.n_kv_heads as i32,
                            c.head_dim as i32,
                            scale,
                            c.max_seq_len as i32,
                            slot,
                        )?;
                    }
                } else if self.paged {
                    // Paged decode (dense F32): KV store + attention go through
                    // the per-slot block tables into the page pool. Rows are
                    // addressed by per-row table offsets (`refresh_table_
                    // offsets`), so prefill packing (several rows on one slot)
                    // works alongside plain decode. `set_block_table` may
                    // alias prefix physical pages across slots between steps.
                    let tpp = self.tokens_per_page as i32;
                    let max_pages = self.max_pages_per_seq as i32;
                    k.launch_kv_store_paged(
                        self.k_buf,
                        kc as *mut f32,
                        self.pos_dev,
                        self.table_offsets,
                        self.block_tables,
                        b,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        tpp,
                    )?;
                    k.launch_kv_store_paged(
                        self.v_buf,
                        vc as *mut f32,
                        self.pos_dev,
                        self.table_offsets,
                        self.block_tables,
                        b,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        tpp,
                    )?;
                    k.launch_attn_decode_paged(
                        self.q,
                        kc as *const f32,
                        vc as *const f32,
                        self.block_tables,
                        self.attn,
                        self.pos_dev,
                        self.table_offsets,
                        b,
                        c.n_heads as i32,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        scale,
                        tpp,
                        max_pages,
                    )?;
                } else {
                    k.launch_kv_store_batched(
                        self.k_buf,
                        kc as *mut f32,
                        self.pos_dev,
                        slots,
                        b,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        c.max_seq_len as i32,
                    )?;
                    k.launch_kv_store_batched(
                        self.v_buf,
                        vc as *mut f32,
                        self.pos_dev,
                        slots,
                        b,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        c.max_seq_len as i32,
                    )?;
                    k.launch_attn_decode_batched(
                        self.q,
                        kc as *const f32,
                        vc as *const f32,
                        self.attn,
                        self.pos_dev,
                        slots,
                        b,
                        c.n_heads as i32,
                        c.n_kv_heads as i32,
                        c.head_dim as i32,
                        scale,
                        c.max_seq_len as i32,
                    )?;
                }
                if c.attn_output_gate {
                    k.launch_attn_gate_apply(self.attn, self.attn_gate, (b * nq) as usize)?;
                }
                gemm(
                    self.proj,
                    self.attn,
                    lw.wo,
                    l16.map_or(std::ptr::null_mut(), |l| l.wo),
                    lq4.map_or(Q4TensorDev::EMPTY, |l| l.wo),
                    d,
                    nq,
                )?;
                k.launch_add(self.x, self.proj, b * d)?;
            }
            // Step profiler: attention phase done.
            if let Some(pf_state) = &self.prof
                && let Some((_, l1, _)) = pf_state.events.get(li)
            {
                unsafe {
                    hip::check(
                        self.k.hip(),
                        (self.k.hip().api.hip_event_record)(*l1, self.k.stream),
                    )?;
                }
            }
            // Debug (issue #107): mid-layer snapshot — attention residual has
            // landed, MLP/MoE not yet run. Together with the layer-tail dump
            // this splits a divergent layer into its attention vs MLP halves.
            if let Some(dir) = &self.layer_dump {
                self.dump_last_x_row(dir, &format!("hsMid_L{li:02}.npy"), b, d)?;
            }
            k.launch_rms_norm(self.x, lw.rms_mlp, self.xn2, b, d, c.rms_eps)?;
            // Per-layer MoE dispatch: mixed checkpoints (Qwen3-MoE style) have
            // dense layers with empty MoE tensors alongside routed-expert
            // layers; a layer is MoE iff it carries a router (mirrors
            // ref_model and the single-sequence path).
            if c.num_experts > 0 && !lw.moe_router.is_null() {
                let ne = c.num_experts as i32;
                let topk = c.num_experts_per_tok.min(c.num_experts) as i32;
                // Routed-expert layers use the MoE expert width (Qwen-MoE:
                // moe_intermediate_size), which may differ from the dense
                // intermediate_size.
                let einter = c.expert_size() as i32;
                if topk > 0 {
                    // Router logits [B, ne] (shared input -> batched GEMM).
                    // The router is unquantized (f32 in `WeightsQ4`) — no Q4
                    // pair exists for it.
                    gemm(
                        self.router,
                        self.xn2,
                        lw.moe_router,
                        l16.map_or(std::ptr::null_mut(), |l| l.moe_router),
                        Q4TensorDev::EMPTY,
                        ne,
                        d,
                    )?;
                    k.launch_moe_router_batched(
                        self.router,
                        self.exp_ids,
                        self.exp_w,
                        ne,
                        topk,
                        b,
                        c.moe_norm_topk,
                        c.moe_routed_scale,
                    )?;
                    // Decode/small steps run the whole grouped MoE on device:
                    // the per-row GEMV kernels need no per-expert counts, so
                    // the counts D2H readback, the full-stream sync, and the
                    // `3*ne` small host-launched GEMMs per layer all disappear
                    // (they serialized every MoE layer of every decode step on
                    // the host). Larger chunked-prefill steps keep the hipBLAS
                    // path, where m>1 rows per expert make weight reuse pay.
                    // Q4-on-device mode has no f16/f32 expert copy, so it runs
                    // the grouped path for EVERY step (prefill included).
                    let grouped = (self.moe_grouped && decode_only) || !self.q4_experts.is_empty();
                    // Buffered prefill: run the grouped GEMMs from the
                    // weights prefetched on the separate stream. Placement is a
                    // numeric no-op, so output matches the full-resident path.
                    let buffered = self.prefetch.is_some();
                    if !buffered && self.expert_slots < ne as usize {
                        self.forward_moe_cpu_batched(li, ne, topk, b, d, einter)?;
                    } else {
                        // Buffered prefill: the weights were prefetched on the
                        // separate stream while this layer computed; wait for
                        // them before either MoE path reads them. The wait is
                        // stream-ordered, so doing it above the grouped/prefill
                        // split is a numeric no-op (both branches share it).
                        if buffered {
                            self.prefetch
                                .as_ref()
                                .expect("prefetch engine")
                                .weights_ready(li, self.k.stream)?;
                        }
                        // Full-layer expert weights for this layer (prefetch
                        // ping-pong buffer or resident), shared by the grouped
                        // GEMV decode path and the hipBLAS prefill path (the
                        // latter slices per expert).
                        let (wg_base, wu_base, wd_base) = if buffered {
                            self.prefetch
                                .as_ref()
                                .expect("prefetch engine")
                                .weights(li)
                                .expect("buffered MoE layer")
                        } else {
                            (lw.moe_wg, lw.moe_wu, lw.moe_wd)
                        };
                        if grouped {
                            // No gather launch: the token-major layout is the
                            // identity — gate_up/down read x + ids directly
                            // (row r's token is r / topk), scatter_all
                            // computes row positions inline. 4 launches per
                            // MoE layer.
                            let rows_total = b * topk;
                            if let Some(q) = self.q4_experts.get(li) {
                                // Q4-on-device: in-kernel dequant from the raw
                                // packed int4 + per-group scales (the only
                                // expert copy on device).
                                k.launch_moe_grouped_gate_up_q4(
                                    self.xn2,
                                    self.exp_ids,
                                    q.wg_q,
                                    q.wg_s,
                                    q.wu_q,
                                    q.wu_s,
                                    self.gate_all,
                                    self.up_all,
                                    rows_total,
                                    einter,
                                    d,
                                    topk,
                                    std::ptr::null_mut(),
                                )?;
                                k.launch_silu_mul(
                                    self.up_all,
                                    self.gate_all,
                                    self.eh_all,
                                    rows_total * einter,
                                )?;
                                k.launch_moe_grouped_down_q4(
                                    self.eh_all,
                                    self.exp_ids,
                                    q.wd_q,
                                    q.wd_s,
                                    self.down_all,
                                    rows_total,
                                    d,
                                    einter,
                                    std::ptr::null_mut(),
                                )?;
                            } else {
                                // Full-layer expert weights for this layer.
                                let (wg16, wu16, wd16) = if f16 {
                                    let l = self.layers_f16[li];
                                    (l.moe_wg, l.moe_wu, l.moe_wd)
                                } else {
                                    (
                                        std::ptr::null_mut(),
                                        std::ptr::null_mut(),
                                        std::ptr::null_mut(),
                                    )
                                };
                                k.launch_moe_grouped_gate_up(
                                    self.xn2,
                                    self.exp_ids,
                                    wg_base,
                                    wu_base,
                                    wg16,
                                    wu16,
                                    self.gate_all,
                                    self.up_all,
                                    rows_total,
                                    einter,
                                    d,
                                    topk,
                                    f16,
                                )?;
                                k.launch_silu_mul(
                                    self.up_all,
                                    self.gate_all,
                                    self.eh_all,
                                    rows_total * einter,
                                )?;
                                k.launch_moe_grouped_down(
                                    self.eh_all,
                                    self.exp_ids,
                                    wd_base,
                                    wd16,
                                    self.down_all,
                                    rows_total,
                                    d,
                                    einter,
                                    f16,
                                )?;
                            }
                            k.launch_moe_scatter_all(
                                self.h_acc,
                                self.exp_w,
                                self.down_all,
                                b,
                                topk,
                                d,
                            )?;
                        } else {
                            // Prefill chunk: expert-major prelude — counts on
                            // device, exclusive prefix sum, atomic-placement
                            // gather, then the per-expert host GEMM loop
                            // (hipBLAS batch counts are host-side).
                            unsafe {
                                hip::check(
                                    self.k.hip(),
                                    (self.k.hip().api.hip_memset)(
                                        self.counts_dev as *mut _,
                                        0,
                                        (ne as usize) * 4,
                                    ),
                                )?;
                                hip::check(
                                    self.k.hip(),
                                    (self.k.hip().api.hip_memset)(
                                        self.moe_pos_dev as *mut _,
                                        0,
                                        (ne as usize) * 4,
                                    ),
                                )?;
                            }
                            k.launch_moe_count_experts(self.exp_ids, self.counts_dev, b, topk)?;
                            // GPU-side exclusive prefix sum -> gather offsets.
                            k.launch_moe_prefix_sum(self.counts_dev, self.offsets_dev, ne)?;
                            k.launch_moe_gather_rows(
                                self.xn2,
                                self.exp_ids,
                                self.exp_w,
                                self.offsets_dev,
                                self.moe_pos_dev,
                                self.xg,
                                self.gw,
                                self.row_idx,
                                b,
                                topk,
                                d,
                            )?;
                            // The per-expert counts are read back once per
                            // layer for the host GEMM loop.
                            hip::memcpy_async(
                                self.k.hip(),
                                self.counts_host as *mut core::ffi::c_void,
                                self.counts_dev as *const core::ffi::c_void,
                                (ne as usize) * 4,
                                hip::HIP_MEMCPY_DEVICE_TO_HOST,
                                self.k.stream,
                            )?;
                            // Make the async counts readback visible to the host loop.
                            self.k.sync()?;
                            let counts: Vec<i32> = (0..ne)
                                .map(|e| unsafe { *self.counts_host.add(e as usize) })
                                .collect();
                            // h_acc accumulation base for the segment scatters.
                            unsafe {
                                hip::check(
                                    self.k.hip(),
                                    (self.k.hip().api.hip_memset)(
                                        self.h_acc as *mut _,
                                        0,
                                        (b as usize) * (d as usize) * 4,
                                    ),
                                )?;
                            }
                            // `wg_base`/`wu_base`/`wd_base` and the prefetch
                            // wait come from the shared block above the
                            // grouped/prefill split; the loop slices per
                            // expert below.
                            // Per-expert grouped GEMMs (counts known on host after the
                            // single D2H read; no per-expert sync). The running base
                            // mirrors the device prefix-sum output.
                            let d_usize = d as usize;
                            let einter_usize = einter as usize;
                            let mut base = 0usize;
                            for (e, &cnt) in counts.iter().enumerate() {
                                let base_e = base;
                                base += cnt as usize;
                                if cnt <= 0 {
                                    continue;
                                }
                                let base = base_e;
                                let xg_e = unsafe { self.xg.add(base * d_usize) };
                                let down_e = unsafe { self.down_all.add(base * d_usize) };
                                let wg32 = unsafe { wg_base.add(e * einter_usize * d_usize) };
                                let wu32 = unsafe { wu_base.add(e * einter_usize * d_usize) };
                                let wd32 = unsafe { wd_base.add(e * d_usize * einter_usize) };
                                let (wg16, wu16, wd16) = if f16 {
                                    let l = self.layers_f16[li];
                                    (
                                        unsafe { l.moe_wg.add(e * einter_usize * d_usize) },
                                        unsafe { l.moe_wu.add(e * einter_usize * d_usize) },
                                        unsafe { l.moe_wd.add(e * d_usize * einter_usize) },
                                    )
                                } else {
                                    (
                                        std::ptr::null_mut(),
                                        std::ptr::null_mut(),
                                        std::ptr::null_mut(),
                                    )
                                };
                                let gemm_e = |out: *mut f32,
                                              x: *const f32,
                                              w32: *mut f32,
                                              w16: *mut u16,
                                              n: i32,
                                              kk: i32|
                                 -> Result<(), Error> {
                                    if f16 {
                                        k.gemm_batched_f16(
                                            out,
                                            x,
                                            w16,
                                            cnt,
                                            n,
                                            kk,
                                            self.xh_moe,
                                            self.yh_moe,
                                        )
                                    } else {
                                        k.gemm_batched(out, x, w32, cnt, n, kk)
                                    }
                                };
                                gemm_e(self.gate_all, xg_e, wg32, wg16, einter, d)?;
                                gemm_e(self.up_all, xg_e, wu32, wu16, einter, d)?;
                                k.launch_silu_mul(
                                    self.up_all,
                                    self.gate_all,
                                    self.eh_all,
                                    cnt * einter,
                                )?;
                                gemm_e(down_e, self.eh_all, wd32, wd16, d, einter)?;
                                k.launch_moe_scatter_add(
                                    self.h_acc,
                                    unsafe { self.row_idx.add(base) },
                                    unsafe { self.gw.add(base) },
                                    down_e,
                                    cnt,
                                    d,
                                )?;
                            }
                        }
                        // Shared experts (DeepSeek-V2): a dense SwiGLU MLP of
                        // width `n_shared_experts * expert_size` on the same
                        // normalized input, ADDED on top of the routed
                        // experts' weighted sum in `h_acc`. Existence is the
                        // dtype-independent `has_shared` flag — the f32
                        // pointers are null on every F16 upload path, and
                        // testing them silently dropped the shared experts
                        // there (#107).
                        if lw.has_shared {
                            // Freeze the invariant the three construction sites
                            // must keep: has_shared (derived from the host
                            // weights) implies one dtype's pointer set was
                            // actually uploaded — otherwise the GEMMs below
                            // read a null weight pointer (a silent-garbage or
                            // illegal-address failure of the #107 class).
                            debug_assert!(
                                if f16 {
                                    l16.is_some_and(|l| !l.shared_wg.is_null())
                                } else {
                                    !lw.shared_wg.is_null()
                                },
                                "has_shared but no f32/f16 shared-expert weights uploaded"
                            );
                            let shinter = c.shared_size() as i32;
                            gemm(
                                self.gate,
                                self.xn2,
                                lw.shared_wg,
                                l16.map_or(std::ptr::null_mut(), |l| l.shared_wg),
                                Q4TensorDev::EMPTY,
                                shinter,
                                d,
                            )?;
                            gemm(
                                self.up,
                                self.xn2,
                                lw.shared_wu,
                                l16.map_or(std::ptr::null_mut(), |l| l.shared_wu),
                                Q4TensorDev::EMPTY,
                                shinter,
                                d,
                            )?;
                            // SwiGLU: h = silu(gate) * up, so silu applies to `gate`.
                            k.launch_silu_mul(self.up, self.gate, self.h, b * shinter)?;
                            gemm(
                                self.proj,
                                self.h,
                                lw.shared_wd,
                                l16.map_or(std::ptr::null_mut(), |l| l.shared_wd),
                                Q4TensorDev::EMPTY,
                                d,
                                shinter,
                            )?;
                            k.launch_add(self.h_acc, self.proj, b * d)?;
                        }
                        k.launch_add(self.x, self.h_acc, b * d)?;
                    }
                }
                // topk == 0: MoE contributes nothing (matches ref_model).
            } else {
                gemm(
                    self.gate,
                    self.xn2,
                    lw.wg,
                    l16.map_or(std::ptr::null_mut(), |l| l.wg),
                    lq4.map_or(Q4TensorDev::EMPTY, |l| l.wg),
                    inter,
                    d,
                )?;
                gemm(
                    self.up,
                    self.xn2,
                    lw.wu,
                    l16.map_or(std::ptr::null_mut(), |l| l.wu),
                    lq4.map_or(Q4TensorDev::EMPTY, |l| l.wu),
                    inter,
                    d,
                )?;
                // SwiGLU: h = silu(gate) * up, so silu applies to `gate`.
                k.launch_silu_mul(self.up, self.gate, self.h, b * inter)?;
                gemm(
                    self.proj,
                    self.h,
                    lw.wd,
                    l16.map_or(std::ptr::null_mut(), |l| l.wd),
                    lq4.map_or(Q4TensorDev::EMPTY, |l| l.wd),
                    d,
                    inter,
                )?;
                k.launch_add(self.x, self.proj, b * d)?;
            }
            // Debug (issue #107): the layer's final residual add has landed —
            // snapshot the last active row for the per-layer divergence diff
            // against the numpy reference (`np_ref.py` records the same point).
            // Syncs per layer; debug-only (set_layer_dump).
            if let Some(dir) = &self.layer_dump {
                self.dump_last_x_row(dir, &format!("hs_L{li:02}.npy"), b, d)?;
                // MoE layers: also snapshot this row's routing (expert ids +
                // weights) — a top-k flip between the f16 device router and the
                // f64 reference would masquerade as numeric divergence. topk > 0
                // is required for buffer existence: alloc_buffers skips the MoE
                // scratch entirely when num_experts_per_tok == 0 (the forward
                // itself treats topk == 0 as "MoE contributes nothing").
                let topk = c.num_experts_per_tok.min(c.num_experts);
                if c.num_experts > 0 && topk > 0 && !lw.moe_router.is_null() {
                    let mut ids = vec![0i32; topk];
                    let mut wts = vec![0.0f32; topk];
                    let off = b.saturating_sub(1) as usize * topk;
                    hip::memcpy(
                        self.k.hip(),
                        ids.as_mut_ptr() as *mut core::ffi::c_void,
                        (self.exp_ids as usize + off * 4) as *const core::ffi::c_void,
                        topk * 4,
                        hip::HIP_MEMCPY_DEVICE_TO_HOST,
                    )?;
                    hip::memcpy(
                        self.k.hip(),
                        wts.as_mut_ptr() as *mut core::ffi::c_void,
                        (self.exp_w as usize + off * 4) as *const core::ffi::c_void,
                        topk * 4,
                        hip::HIP_MEMCPY_DEVICE_TO_HOST,
                    )?;
                    write_npy_f32(
                        &dir.join(format!("route_ids_L{li:02}.npy")),
                        &ids.iter().map(|&e| e as f32).collect::<Vec<_>>(),
                    )?;
                    write_npy_f32(&dir.join(format!("route_w_L{li:02}.npy")), &wts)?;
                }
                // Layer 1 (the first MoE layer on DeepSeek-V2-style checkpoints
                // with first_k_dense_replace = 1) gets the full intermediate
                // trace: router logits, xn2, the LAST active token's per-slot
                // gate/up/eh/down, h_acc, and the shared-expert chain — matching
                // the row the hs_/route_ dumps above snapshot. Only meaningful
                // on the grouped on-device decode path, the only one that leaves
                // these buffers token-major for this layer: the hipBLAS prefill
                // path overwrites gate/up/eh per expert segment and packs
                // down_all expert-major, and the CPU-offload path
                // (forward_moe_cpu_batched) never writes them at all. Models
                // whose first MoE layer is not index 1 need this index adjusted.
                let grouped = (self.moe_grouped && decode_only) || !self.q4_experts.is_empty();
                let offload = self.prefetch.is_none() && self.expert_slots < c.num_experts;
                if li == 1
                    && c.num_experts > 0
                    && topk > 0
                    && grouped
                    && !offload
                    && !lw.moe_router.is_null()
                {
                    let einter = c.expert_size();
                    let ne = c.num_experts;
                    let shinter = c.shared_size();
                    let d_us = d as usize;
                    let last = b.saturating_sub(1) as usize;
                    // Element offset of the LAST active row: [rows, width]
                    // buffers are row-major, and the grouped kernels pack the
                    // per-slot rows token-major (row r = token r/topk, slot
                    // r%topk), so token `last`'s topk slots start at row
                    // last*topk.
                    let grab =
                        |name: &str, base: *mut f32, off: usize, n: usize| -> Result<(), Error> {
                            let mut v = vec![0.0f32; n];
                            let src = unsafe { base.add(off) } as *const core::ffi::c_void;
                            hip::memcpy(
                                self.k.hip(),
                                v.as_mut_ptr() as *mut core::ffi::c_void,
                                src,
                                n * 4,
                                hip::HIP_MEMCPY_DEVICE_TO_HOST,
                            )?;
                            write_npy_f32(&dir.join(format!("moe_{name}.npy")), &v)
                        };
                    grab("router", self.router, last * ne, ne)?;
                    grab("xn2", self.xn2, last * d_us, d_us)?;
                    grab(
                        "gate_all",
                        self.gate_all,
                        last * topk * einter,
                        topk * einter,
                    )?;
                    grab("up_all", self.up_all, last * topk * einter, topk * einter)?;
                    grab("eh_all", self.eh_all, last * topk * einter, topk * einter)?;
                    grab("down_all", self.down_all, last * topk * d_us, topk * d_us)?;
                    grab("h_acc", self.h_acc, last * d_us, d_us)?;
                    // The shared-expert chain only exists when this layer has
                    // shared experts; otherwise gate/up/h/proj hold values from
                    // some other consumer (dense MLP of an earlier layer, the
                    // attention projection) and dumping them as "sh_*" would
                    // mislead the diff.
                    if lw.has_shared {
                        grab("sh_gate", self.gate, last * shinter, shinter)?;
                        grab("sh_up", self.up, last * shinter, shinter)?;
                        grab("sh_h", self.h, last * shinter, shinter)?;
                        grab("sh_proj", self.proj, last * d_us, d_us)?;
                    }
                }
            }
            // Step profiler: MLP/MoE phase done (dense MLP or routed experts).
            if let Some(pf_state) = &self.prof
                && let Some((_, _, l2)) = pf_state.events.get(li)
            {
                unsafe {
                    hip::check(
                        self.k.hip(),
                        (self.k.hip().api.hip_event_record)(*l2, self.k.stream),
                    )?;
                }
            }
            // The layer's compute is fully queued: let the prefetch engine know
            // (its ping-pong slot is now free for the layer after next).
            if let Some(pf) = &self.prefetch {
                pf.layer_end(li, self.k.stream)?;
            }
        }
        // Step profiler: whole-step end bracket — placed AFTER the output head
        // so the reported total covers the full step (final norm + lm_head in
        // the `other` bucket).
        k.launch_rms_norm(self.x, self.rms_final_dev, self.xn, b, d, c.rms_eps)?;
        if !self.lm_head_q4.is_null() {
            // Dense Q4-on-device: every step, any row count — the kernel is
            // batch-general and vocab-sized n is just more blocks.
            k.launch_gemv_q4(
                self.xn,
                self.lm_head_q4.q,
                self.lm_head_q4.s,
                self.logits,
                c.vocab_size as i32,
                d,
                b,
            )?;
        } else if f16 {
            if b <= GEMV_MAX_M && d <= GEMV_MAX_D {
                k.launch_gemv_f16(
                    self.logits,
                    self.xn,
                    self.lm_head_f16,
                    c.vocab_size as i32,
                    d,
                    b,
                    std::ptr::null_mut(),
                )?;
            } else {
                k.gemm_batched_f16(
                    self.logits,
                    self.xn,
                    self.lm_head_f16,
                    b,
                    c.vocab_size as i32,
                    d,
                    self.xh,
                    self.yh,
                )?;
            }
        } else {
            k.gemm_batched(
                self.logits,
                self.xn,
                self.lm_head_dev,
                b,
                c.vocab_size as i32,
                d,
            )?;
        }
        if let Some(pf_state) = &self.prof {
            unsafe {
                hip::check(
                    self.k.hip(),
                    (self.k.hip().api.hip_event_record)(pf_state.step_events.1, self.k.stream),
                )?;
            }
        }
        Ok(())
    }

    /// Batched greedy sampling: argmax per row, read back only `batch` tokens.
    fn sample(&self, n: usize) -> Result<Vec<u32>, Error> {
        self.sample_device(n)?;
        self.sample_readback(n)
    }

    /// Device half of [`Self::sample`]: the argmax kernel only (graph-
    /// capturable, #103).
    fn sample_device(&self, n: usize) -> Result<(), Error> {
        self.k.launch_argmax_batched(
            self.logits,
            self.out_tok_dev,
            self.cfg.vocab_size as i32,
            n as i32,
        )
    }

    /// Host half of [`Self::sample`]: sync + D2H of the sampled tokens. This
    /// is the graph boundary — it cannot be captured.
    fn sample_readback(&self, n: usize) -> Result<Vec<u32>, Error> {
        unsafe {
            hip::check(
                self.k.hip(),
                (self.k.hip().api.hip_stream_synchronize)(self.k.stream),
            )?;
            hip::memcpy(
                self.k.hip(),
                self.out_tok_host as *mut core::ffi::c_void,
                self.out_tok_dev as *const core::ffi::c_void,
                n * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )?;
        }
        let mut out = Vec::with_capacity(n);
        unsafe {
            for i in 0..n {
                out.push(*self.out_tok_host.add(i) as u32);
            }
        }
        Ok(out)
    }

    /// Batched decode with explicit per-sequence lengths and a variable active
    /// count (`tokens.len()` may be <= capacity). The engine owns `lens`; this
    /// method does not touch the internal lens used by [`decode_step`](Self::decode_step).
    /// Row capacity of the model (max rows per step).
    #[must_use]
    pub const fn row_capacity(&self) -> usize {
        self.rows
    }

    /// Maximum KV positions per sequence (hard context limit).
    #[must_use]
    pub const fn max_seq_len(&self) -> usize {
        self.cfg.max_seq_len
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_explicit(
        &mut self,
        tokens: &[u32],
        lens: &[u32],
        slots: &[u32],
        params: &mut [SamplingParams],
        counts: &[Vec<(u32, u32)>],
        bias: &[Vec<(u32, f32)>],
        decode_only: bool,
    ) -> Result<SampleOutput, Error> {
        let n = tokens.len();
        assert_eq!(n, lens.len(), "tokens and lens must be equal length");
        assert_eq!(n, slots.len(), "tokens and slots must be equal length");
        // The sampler stages/reads per row; a mismatch would silently sample
        // stale staging rows in the graph path (its grid uses `n`).
        assert_eq!(n, params.len(), "tokens and params must be equal length");
        assert!(n <= self.rows, "active count exceeds row capacity");
        // Positions and slots must stay inside the device buffers: an out-of-
        // range length would make the KV store write past the cache (silent
        // corruption). Guarded here for the explicit-entry API, which the
        // continuous/speculative engines use directly.
        if let Some(&l) = lens.iter().find(|&&l| l as usize >= self.cfg.max_seq_len) {
            return Err(Error::Model(format!(
                "row at position {l} exceeds max_seq_len {}",
                self.cfg.max_seq_len
            )));
        }
        if let Some(&s) = slots.iter().find(|&&s| s as usize >= self.batch) {
            return Err(Error::Model(format!(
                "row slot {s} out of batch range {}",
                self.batch
            )));
        }
        // GDN (hybrid linear-attention) layers apply the per-slot recurrent
        // update exactly once per step: two rows on one slot would
        // double-apply the delta rule. Chunked/packed prefill for GDN models
        // is Stage B (#112) — reject duplicate slots loudly instead of
        // corrupting the state silently.
        if self.cfg.gdn_enabled() {
            let mut seen = vec![false; self.batch];
            for &s in slots {
                if std::mem::replace(&mut seen[s as usize], true) {
                    return Err(Error::InvalidArgument(format!(
                        "slot {s} appears twice in one step; GDN models need one row per slot (chunked prefill is #112 Stage B)"
                    )));
                }
            }
        }
        // Prefill-attention runs are currently disabled: the naive shared-KV
        // kernel is occupancy-bound and slower than decode attention on this
        // GPU (see roadmap). All rows use decode attention (run_mask = 0).
        let run_mask = vec![0i32; n];
        let num_runs = 0;
        // Host staging: refresh every pinned input mirror. Replayable graphs
        // re-read these buffers on every launch, so this write is the ONLY
        // per-step input work in the graph path.
        unsafe {
            for i in 0..n {
                *self.tokens_host.add(i) = tokens[i] as i32;
                *self.pos_host.add(i) = lens[i] as i32;
                *self.slots_host.add(i) = slots[i] as i32;
                *self.run_mask_host.add(i) = run_mask[i];
            }
        }
        for (r, &s) in slots.iter().enumerate() {
            self.last_row_by_slot[s as usize] = r;
        }
        self.stage_table_offsets(slots);
        self.offsets_dirty = true;
        // Sampler parameter staging is host-only; the upload + kernel either
        // run eagerly below or are recorded inside the decode graph.
        self.sampler.sample_batched_stage(params, counts, bias)?;
        let vocab = self.cfg.vocab_size;
        if self.graph_capture_ok(n, decode_only) {
            let key = n as i32;
            if !self.decode_graphs.contains_key(&key) {
                self.capture_decode_graph(key)?;
            }
            // SAFETY: every buffer the graph touches is owned by `self` and
            // alive; the pinned staging mirrors (inputs + sampler params) were
            // refreshed above on this thread, and no other stream uses them.
            unsafe { self.decode_graphs[&key].replay()? }
            return self
                .sampler
                .sample_batched_readback(self.logits, params, vocab);
        }
        self.upload_decode_inputs(n)?;
        self.run_kernels(
            n as i32,
            self.slots_dev,
            self.run_mask_dev,
            decode_only,
            num_runs,
        )?;
        self.sampler
            .sample_batched_after_stage(self.logits, params, vocab)
    }

    /// Saves a lightweight token-boundary anchor for `slot`: the per-layer KV
    /// prefix `[0..=token_idx]` plus the hidden state of the last token.
    ///
    /// Requires `tokens.len() == token_idx + 1` and that the slot's KV for
    /// those positions is up to date (call after a step that processed the
    /// sequence; the stream is synced inside). The hidden state is read from
    /// the slot's last forward row (`last_row_by_slot`).
    pub fn save_anchor(
        &self,
        slot: usize,
        tokens: &[u32],
        token_idx: usize,
    ) -> Result<crate::state_reuse::Anchor, Error> {
        use crate::state_reuse::{Anchor, KvSnapshot};
        if slot >= self.batch {
            return Err(Error::InvalidArgument(format!(
                "slot {slot} out of range (batch {})",
                self.batch
            )));
        }
        if tokens.len() != token_idx + 1 {
            return Err(Error::InvalidArgument(format!(
                "anchor token_idx {token_idx} does not match {} prefix tokens",
                tokens.len()
            )));
        }
        // The anchor format carries KV + hidden state only; GDN layers would
        // additionally need their recurrent + conv state at the boundary.
        // Reject loudly until the anchor grows that payload (#112 follow-up)
        // instead of restoring a silently desynced hybrid stack.
        if self.cfg.gdn_enabled() {
            return Err(Error::InvalidArgument(
                "state-reuse anchors do not support GDN (hybrid linear-attention) models yet"
                    .into(),
            ));
        }
        self.k.sync()?;
        let c = self.cfg;
        let kv_elem = if c.dtype == ModelDType::F16 { 2 } else { 4 };
        let row_bytes = c.max_seq_len * c.n_kv_heads * c.head_dim * kv_elem;
        let copy = (token_idx + 1) * c.n_kv_heads * c.head_dim * kv_elem;
        let mut layers = Vec::with_capacity(c.n_layers);
        for (kc, vc) in &self.kv_cache {
            let src_k = (*kc as usize + slot * row_bytes) as *const core::ffi::c_void;
            let src_v = (*vc as usize + slot * row_bytes) as *const core::ffi::c_void;
            let mut kb = vec![0u8; copy];
            let mut vb = vec![0u8; copy];
            hip::memcpy(
                self.k.hip(),
                kb.as_mut_ptr() as *mut core::ffi::c_void,
                src_k,
                copy,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )?;
            hip::memcpy(
                self.k.hip(),
                vb.as_mut_ptr() as *mut core::ffi::c_void,
                src_v,
                copy,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )?;
            layers.push((kb, vb));
        }
        if !self.mla_kv_cache.is_empty() {
            let heads = c.n_heads;
            let k_row = c.max_seq_len * heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) * 4;
            let v_row = c.max_seq_len * heads * c.v_head_dim * 4;
            let k_bytes = (token_idx + 1) * heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) * 4;
            let v_bytes = (token_idx + 1) * heads * c.v_head_dim * 4;
            for (kc, vc) in &self.mla_kv_cache {
                let src_k = (*kc as usize + slot * k_row) as *const core::ffi::c_void;
                let src_v = (*vc as usize + slot * v_row) as *const core::ffi::c_void;
                let mut kb = vec![0u8; k_bytes];
                let mut vb = vec![0u8; v_bytes];
                hip::memcpy(
                    self.k.hip(),
                    kb.as_mut_ptr() as *mut core::ffi::c_void,
                    src_k,
                    k_bytes,
                    hip::HIP_MEMCPY_DEVICE_TO_HOST,
                )?;
                hip::memcpy(
                    self.k.hip(),
                    vb.as_mut_ptr() as *mut core::ffi::c_void,
                    src_v,
                    v_bytes,
                    hip::HIP_MEMCPY_DEVICE_TO_HOST,
                )?;
                layers.push((kb, vb));
            }
        }
        let d = c.d_model;
        let row = self.last_row_by_slot[slot];
        let mut hidden = vec![0.0f32; d];
        let src = (self.x as usize + row * d * 4) as *const core::ffi::c_void;
        hip::memcpy(
            self.k.hip(),
            hidden.as_mut_ptr() as *mut core::ffi::c_void,
            src,
            d * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )?;
        Ok(Anchor {
            id: 0,
            token_idx,
            tokens: tokens.to_vec(),
            kv: KvSnapshot { layers },
            hidden,
        })
    }

    /// Restores an anchor into `slot`: copies the per-layer KV prefix into the
    /// slot's cache, sets its length to `token_idx + 1`, and restores the
    /// saved hidden state into `x[slot]`. The next decode/prefill continues at
    /// position `token_idx + 1`, so only the delta needs computing.
    pub fn restore_anchor(
        &mut self,
        slot: usize,
        anchor: &crate::state_reuse::Anchor,
    ) -> Result<(), Error> {
        if slot >= self.batch {
            return Err(Error::InvalidArgument(format!(
                "slot {slot} out of range (batch {})",
                self.batch
            )));
        }
        let c = self.cfg;
        if anchor.kv.layers.len() != c.n_layers {
            return Err(Error::InvalidArgument(format!(
                "anchor layer count {} != model {}",
                anchor.kv.layers.len(),
                c.n_layers
            )));
        }
        let prefix = anchor.token_idx + 1;
        if prefix > c.max_seq_len {
            return Err(Error::InvalidArgument(format!(
                "anchor prefix {prefix} exceeds max_seq_len {}",
                c.max_seq_len
            )));
        }
        let kv_elem = if c.dtype == ModelDType::F16 { 2 } else { 4 };
        let row_bytes = c.max_seq_len * c.n_kv_heads * c.head_dim * kv_elem;
        let copy = prefix * c.n_kv_heads * c.head_dim * kv_elem;
        for (li, (kc, vc)) in self.kv_cache.iter().enumerate() {
            let (kb, vb) = &anchor.kv.layers[li];
            if kb.len() != copy || vb.len() != copy {
                return Err(Error::InvalidArgument(format!(
                    "anchor KV size mismatch at layer {li} (expected {copy} bytes, got {} / {})",
                    kb.len(),
                    vb.len()
                )));
            }
            let dst_k = (*kc as usize + slot * row_bytes) as *mut core::ffi::c_void;
            let dst_v = (*vc as usize + slot * row_bytes) as *mut core::ffi::c_void;
            hip::memcpy(
                self.k.hip(),
                dst_k,
                kb.as_ptr() as *const core::ffi::c_void,
                copy,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )?;
            hip::memcpy(
                self.k.hip(),
                dst_v,
                vb.as_ptr() as *const core::ffi::c_void,
                copy,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )?;
        }
        if !self.mla_kv_cache.is_empty() {
            let heads = c.n_heads;
            let k_row = c.max_seq_len * heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) * 4;
            let v_row = c.max_seq_len * heads * c.v_head_dim * 4;
            let k_bytes = prefix * heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) * 4;
            let v_bytes = prefix * heads * c.v_head_dim * 4;
            for (li, (kc, vc)) in self.mla_kv_cache.iter().enumerate() {
                let (kb, vb) = &anchor.kv.layers[li];
                if kb.len() != k_bytes || vb.len() != v_bytes {
                    return Err(Error::InvalidArgument(format!(
                        "anchor MLA KV size mismatch at layer {li}"
                    )));
                }
                let dst_k = (*kc as usize + slot * k_row) as *mut core::ffi::c_void;
                let dst_v = (*vc as usize + slot * v_row) as *mut core::ffi::c_void;
                hip::memcpy(
                    self.k.hip(),
                    dst_k,
                    kb.as_ptr() as *const core::ffi::c_void,
                    k_bytes,
                    hip::HIP_MEMCPY_HOST_TO_DEVICE,
                )?;
                hip::memcpy(
                    self.k.hip(),
                    dst_v,
                    vb.as_ptr() as *const core::ffi::c_void,
                    v_bytes,
                    hip::HIP_MEMCPY_HOST_TO_DEVICE,
                )?;
            }
        }
        self.lens[slot] = prefix as u32;
        // Restore the hidden state into x[slot] so logits-at-anchor works if a
        // caller wants it; decode steps overwrite x from embeddings regardless.
        if anchor.hidden.len() == c.d_model {
            let d = c.d_model;
            let dst = (self.x as usize + slot * d * 4) as *mut core::ffi::c_void;
            hip::memcpy(
                self.k.hip(),
                dst,
                anchor.hidden.as_ptr() as *const core::ffi::c_void,
                d * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )?;
            self.last_row_by_slot[slot] = slot;
        }
        Ok(())
    }

    /// Moves a sequence's KV rows from `from` to `to` (compaction). Only the
    /// first `len` positions are copied.
    pub fn copy_seq_kv(&self, from: usize, to: usize, len: usize) -> Result<(), Error> {
        let kv_elem = if self.cfg.dtype == ModelDType::F16 {
            2
        } else {
            4
        };
        let row_bytes = self.cfg.max_seq_len * self.cfg.n_kv_heads * self.cfg.head_dim * kv_elem;
        let copy_bytes = len * self.cfg.n_kv_heads * self.cfg.head_dim * kv_elem;
        for (kc, vc) in &self.kv_cache {
            let src_k = (*kc as usize + from * row_bytes) as *const core::ffi::c_void;
            let dst_k = (*kc as usize + to * row_bytes) as *mut core::ffi::c_void;
            let src_v = (*vc as usize + from * row_bytes) as *const core::ffi::c_void;
            let dst_v = (*vc as usize + to * row_bytes) as *mut core::ffi::c_void;
            hip::memcpy(
                self.k.hip(),
                dst_k,
                src_k,
                copy_bytes,
                hip::HIP_MEMCPY_DEVICE_TO_DEVICE,
            )?;
            hip::memcpy(
                self.k.hip(),
                dst_v,
                src_v,
                copy_bytes,
                hip::HIP_MEMCPY_DEVICE_TO_DEVICE,
            )?;
        }
        // MLA (kv_lora_rank > 0) keeps its expanded per-head KV in a separate
        // cache; compaction must move it with the slot or the sequence's KV
        // silently points at the wrong slot after a lower slot finishes.
        if !self.mla_kv_cache.is_empty() {
            let c = self.cfg;
            let heads = c.n_heads;
            let k_row = c.max_seq_len * heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) * 4;
            let v_row = c.max_seq_len * heads * c.v_head_dim * 4;
            let k_bytes = len * heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) * 4;
            let v_bytes = len * heads * c.v_head_dim * 4;
            for (kc, vc) in &self.mla_kv_cache {
                let src_k = (*kc as usize + from * k_row) as *const core::ffi::c_void;
                let dst_k = (*kc as usize + to * k_row) as *mut core::ffi::c_void;
                let src_v = (*vc as usize + from * v_row) as *const core::ffi::c_void;
                let dst_v = (*vc as usize + to * v_row) as *mut core::ffi::c_void;
                hip::memcpy(
                    self.k.hip(),
                    dst_k,
                    src_k,
                    k_bytes,
                    hip::HIP_MEMCPY_DEVICE_TO_DEVICE,
                )?;
                hip::memcpy(
                    self.k.hip(),
                    dst_v,
                    src_v,
                    v_bytes,
                    hip::HIP_MEMCPY_DEVICE_TO_DEVICE,
                )?;
            }
        }
        Ok(())
    }

    /// The configured batch size.
    #[must_use]
    pub const fn batch(&self) -> usize {
        self.batch
    }

    /// Syncs the stream and copies the last step's logits (`[batch, vocab]`)
    /// back to host (debug / numeric validation).
    ///
    /// **Note**: when the last `decode_step_explicit` applied presence/frequency
    /// penalties or logit_bias, the device logits buffer holds the
    /// **post-penalty** values (the sampler and `top_logprobs` intentionally
    /// operate on those, matching the sampled distribution). For the raw model
    /// output, read before sampling (or use a penalty-free greedy step).
    pub fn read_logits(&self) -> Result<Vec<f32>, Error> {
        self.read_logits_rows(self.batch)
    }

    /// [`Self::read_logits`] for `n` forwarded rows: a prefill step packed
    /// into more rows than slots writes `[rows, vocab]`, so row-level
    /// inspection needs the active count, not the slot count.
    pub fn read_logits_rows(&self, n: usize) -> Result<Vec<f32>, Error> {
        if n > self.rows {
            return Err(Error::InvalidArgument(format!(
                "row count {n} exceeds capacity {}",
                self.rows
            )));
        }
        self.k.sync()?;
        let elems = n * self.cfg.vocab_size;
        let mut out = vec![0.0f32; elems];
        hip::memcpy(
            self.k.hip(),
            out.as_mut_ptr() as *mut core::ffi::c_void,
            self.logits as *const core::ffi::c_void,
            out.len() * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )?;
        Ok(out)
    }
}

// SAFETY: BatchedModel is confined to one engine thread; its raw device
// pointers are only dereferenced there, and the loaded HIP runtime is Send.
unsafe impl Send for BatchedModel {}

impl Drop for BatchedModel {
    fn drop(&mut self) {
        let hip = self.k.hip();
        for &p in &self.allocs {
            let _ = hip::free(hip, p);
        }
        for &p in &self.host_pins {
            let _ = hip::host_free(hip, p);
        }
    }
}

#[cfg(all(test, feature = "hip"))]
mod paged_support_tests {
    use super::*;

    /// The pre-flight must mirror `paged_guards` exactly — the server's
    /// pre-load gate degrades on these same conditions.
    #[test]
    fn check_paged_support_mirrors_paged_guards() {
        let cfg = Config::tiny(); // max_seq_len 256; dense F32
        assert!(BatchedModel::check_paged_support(&cfg, 64).is_ok());

        // Page geometry: zero and non-divisor tpp are rejected.
        assert!(BatchedModel::check_paged_support(&cfg, 0).is_err());
        assert!(BatchedModel::check_paged_support(&cfg, 48).is_err());

        // Attention-smem bound: max_seq_len beyond 16128 tokens.
        let mut big = Config::tiny();
        big.max_seq_len = 32768;
        assert!(BatchedModel::check_paged_support(&big, 64).is_err());

        // MLA is F32-only: F16 MLA rejected, F32 MLA accepted.
        let mut mla = Config::mla(128, 2, 4, 1024, 256, 32, 16, 16, 8, 16);
        mla.dtype = ModelDType::F16;
        assert!(BatchedModel::check_paged_support(&mla, 64).is_err());
        mla.dtype = ModelDType::F32;
        assert!(BatchedModel::check_paged_support(&mla, 64).is_ok());
    }
}

#[cfg(all(test, feature = "hip"))]
mod npy_writer_tests {
    use super::*;

    /// `write_npy_f32` emits a numpy 1.0 file `np.load` can read: magic,
    /// version, u16-LE header length, a 64-byte-aligned header block
    /// terminated by a newline, then little-endian f32 payload. The padding
    /// arithmetic is exactly the kind that breaks silently, so pin the bytes.
    #[test]
    fn write_npy_f32_header_and_payload() {
        let dir = std::env::temp_dir().join(format!("mach_npy_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Non-empty vector: shape (3,), three LE f32 values.
        let path = dir.join("t.npy");
        write_npy_f32(&path, &[1.5f32, -2.25, 0.0]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..6], b"\x93NUMPY", "magic");
        assert_eq!(&bytes[6..8], &[1u8, 0], "format version 1.0");
        let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!(10 + hlen, bytes.len() - 3 * 4, "payload size");
        assert_eq!((10 + hlen) % 64, 0, "header block 64-byte alignment");
        let hdr = core::str::from_utf8(&bytes[10..10 + hlen]).unwrap();
        assert!(hdr.ends_with('\n'), "header terminated by newline");
        assert!(hdr.contains("'descr': '<f4'"), "{hdr}");
        assert!(hdr.contains("'fortran_order': False"), "{hdr}");
        assert!(hdr.contains("'shape': (3,)"), "{hdr}");
        let mut want = Vec::new();
        want.extend_from_slice(&1.5f32.to_le_bytes());
        want.extend_from_slice(&(-2.25f32).to_le_bytes());
        want.extend_from_slice(&0.0f32.to_le_bytes());
        assert_eq!(&bytes[10 + hlen..], want.as_slice(), "LE f32 payload");

        // Empty vector: shape (0,), zero payload — every `grab` offset in the
        // layer dump can hit this when a width is 0.
        let empty = dir.join("empty.npy");
        write_npy_f32(&empty, &[]).unwrap();
        let bytes = std::fs::read(&empty).unwrap();
        let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!(10 + hlen, bytes.len(), "no payload");
        assert_eq!((10 + hlen) % 64, 0, "header block 64-byte alignment");
        let hdr = core::str::from_utf8(&bytes[10..10 + hlen]).unwrap();
        assert!(hdr.contains("'shape': (0,)"), "{hdr}");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&empty);
        let _ = std::fs::remove_dir(&dir);
    }
}
