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
use crate::sampling::{BatchedSampler, SampleOutput, SamplingParams};
use crate::{Config, Error, Weights};
use mach_kernel_sys::hip::{self, Hip};
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
    /// MoE fp16 (num_experts > 0): router + per-expert gate/up/down.
    moe_router: *mut u16,
    moe_wg: *mut u16,
    moe_wu: *mut u16,
    moe_wd: *mut u16,
}

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
    /// KV caches: (k, v) per layer, layout `[batch, max_seq, kv_heads, head_dim]`.
    /// KV caches as opaque pointers (f32 or fp16 per dtype), layout
    /// `[batch, max_seq, kv_heads, head_dim]`.
    kv_cache: Vec<(*mut core::ffi::c_void, *mut core::ffi::c_void)>,
    /// Per-sequence lengths (host).
    lens: Vec<u32>,
    allocs: Vec<*mut core::ffi::c_void>,
    host_pins: Vec<*mut core::ffi::c_void>,
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
            kv_cache: Vec::new(),
            lens: vec![0; slots],
            allocs: Vec::new(),
            host_pins: Vec::new(),
        };
        m.alloc_buffers()?;
        m.upload_weights(w)?;
        Ok(m)
    }

    fn dalloc(&mut self, bytes: usize) -> Result<*mut f32, Error> {
        let p = hip::malloc(self.k.hip(), bytes)?;
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

    fn alloc_f16(&mut self, n: usize) -> Result<*mut u16, Error> {
        let p = hip::malloc(self.k.hip(), n * 2)?;
        self.allocs.push(p);
        Ok(p as *mut u16)
    }

    fn alloc_buffers(&mut self) -> Result<(), Error> {
        let c = self.cfg;
        let b = self.rows; // row buffers sized for the prefill row capacity
        let d = c.d_model;
        let nq = c.n_heads * c.head_dim;
        let nkv = c.n_kv_heads * c.head_dim;
        let inter = c.intermediate_size;

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
            .max(c.intermediate_size)
            .max(c.n_heads * c.head_dim)
            .max(c.vocab_size);
        let xh_bytes = b * c.d_model.max(c.intermediate_size) * 2;
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
        let kv_bytes = self.batch * c.max_seq_len * c.n_kv_heads * c.head_dim * kv_elem;
        for _ in 0..c.n_layers {
            let kk = self.dalloc(kv_bytes)?;
            let vv = self.dalloc(kv_bytes)?;
            self.kv_cache
                .push((kk as *mut core::ffi::c_void, vv as *mut core::ffi::c_void));
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
                self.gate_all = self.dalloc(cap * inter * 4)?;
                self.up_all = self.dalloc(cap * inter * 4)?;
                self.eh_all = self.dalloc(cap * inter * 4)?;
                self.down_all = self.dalloc(cap * d * 4)?;
                let ch = hip::host_malloc(self.k.hip(), ne * 4)?;
                self.counts_host = ch as *mut i32;
                self.host_pins.push(ch);
                if c.dtype == ModelDType::F16 {
                    let m = c.d_model.max(c.intermediate_size);
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
        self.emb_dev = self.dalloc(w.tok_emb.len() * 4)?;
        self.rms_final_dev = self.dalloc(w.rms_final.len() * 4)?;
        self.lm_head_dev = self.dalloc(w.lm_head.len() * 4)?;
        self.upload(self.emb_dev, &w.tok_emb)?;
        self.upload(self.rms_final_dev, &w.rms_final)?;
        self.upload(self.lm_head_dev, &w.lm_head)?;
        if c.dtype == ModelDType::F16 {
            self.emb_f16 = self.alloc_f16(w.tok_emb.len())?;
            self.lm_head_f16 = self.alloc_f16(w.lm_head.len())?;
            self.upload_f16(self.emb_f16, &w.tok_emb)?;
            self.upload_f16(self.lm_head_f16, &w.lm_head)?;
        }
        for lw in &w.layers {
            let l = LayerDev {
                wq: self.dalloc(lw.wq.len() * 4)?,
                wk: self.dalloc(lw.wk.len() * 4)?,
                wv: self.dalloc(lw.wv.len() * 4)?,
                wo: self.dalloc(lw.wo.len() * 4)?,
                rms_attn: self.dalloc(lw.rms_attn.len() * 4)?,
                wg: self.dalloc(lw.wg.len() * 4)?,
                wu: self.dalloc(lw.wu.len() * 4)?,
                wd: self.dalloc(lw.wd.len() * 4)?,
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
                moe_router: if lw.moe_router.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.moe_router.len() * 4)?;
                    self.upload(p, &lw.moe_router)?;
                    p
                },
                moe_wg: if lw.moe_wg.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.moe_wg.len() * 4)?;
                    self.upload(p, &lw.moe_wg)?;
                    p
                },
                moe_wu: if lw.moe_wu.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.moe_wu.len() * 4)?;
                    self.upload(p, &lw.moe_wu)?;
                    p
                },
                moe_wd: if lw.moe_wd.is_empty() {
                    std::ptr::null_mut()
                } else {
                    let p = self.dalloc(lw.moe_wd.len() * 4)?;
                    self.upload(p, &lw.moe_wd)?;
                    p
                },
            };
            self.upload(l.wq, &lw.wq)?;
            self.upload(l.wk, &lw.wk)?;
            self.upload(l.wv, &lw.wv)?;
            self.upload(l.wo, &lw.wo)?;
            self.upload(l.rms_attn, &lw.rms_attn)?;
            self.upload(l.wg, &lw.wg)?;
            self.upload(l.wu, &lw.wu)?;
            self.upload(l.wd, &lw.wd)?;
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
                    moe_wg: self.alloc_f16(lw.moe_wg.len())?,
                    moe_wu: self.alloc_f16(lw.moe_wu.len())?,
                    moe_wd: self.alloc_f16(lw.moe_wd.len())?,
                };
                self.upload_f16(l16.wq, &lw.wq)?;
                self.upload_f16(l16.wk, &lw.wk)?;
                self.upload_f16(l16.wv, &lw.wv)?;
                self.upload_f16(l16.wo, &lw.wo)?;
                self.upload_f16(l16.wg, &lw.wg)?;
                self.upload_f16(l16.wu, &lw.wu)?;
                self.upload_f16(l16.wd, &lw.wd)?;
                self.upload_f16(l16.moe_router, &lw.moe_router)?;
                self.upload_f16(l16.moe_wg, &lw.moe_wg)?;
                self.upload_f16(l16.moe_wu, &lw.moe_wu)?;
                self.upload_f16(l16.moe_wd, &lw.moe_wd)?;
                self.layers_f16.push(l16);
            }
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
        unsafe {
            for (i, (&t, &l)) in tokens.iter().zip(&self.lens).enumerate() {
                *self.tokens_host.add(i) = t as i32;
                *self.pos_host.add(i) = l as i32;
                *self.slots_host.add(i) = i as i32; // row == slot for decode
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
        self.run_kernels(self.batch as i32, self.slots_dev, self.run_mask_dev, 0)?;
        let next = self.sample(self.batch)?;
        for l in self.lens.iter_mut() {
            *l += 1;
        }
        Ok(next)
    }

    fn run_kernels(
        &self,
        count: i32,
        slots: *const i32,
        run_mask: *const i32,
        num_runs: i32,
    ) -> Result<(), Error> {
        let c = self.cfg;
        let b = count;
        let d = c.d_model as i32;
        let nq = (c.n_heads * c.head_dim) as i32;
        let nkv = (c.n_kv_heads * c.head_dim) as i32;
        let inter = c.intermediate_size as i32;
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let k = &self.k;
        let f16 = c.dtype == ModelDType::F16;

        // Selects fp16 (cast activations + fp16 weights, fp32 output) or f32.
        let gemm = |out: *mut f32,
                    x: *const f32,
                    w32: *mut f32,
                    w16: *mut u16,
                    n: i32,
                    kk: i32|
         -> Result<(), Error> {
            if f16 {
                k.gemm_batched_f16(out, x, w16, b, n, kk, self.xh, self.yh)
            } else {
                k.gemm_batched(out, x, w32, b, n, kk)
            }
        };

        if f16 {
            k.launch_embed_f16(self.tokens_dev, self.emb_f16, self.x, d, b)?;
        } else {
            k.launch_embed_batched(self.tokens_dev, self.emb_dev, self.x, d, b)?;
        }
        for (li, lw) in self.layers_dev.iter().enumerate() {
            let l16 = if f16 { Some(self.layers_f16[li]) } else { None };
            k.launch_rms_norm(self.x, lw.rms_attn, self.xn, b, d, c.rms_eps)?;
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
                k.launch_add_bias(self.q, lw.bq, b, nq)?;
            }
            if !lw.bk.is_null() {
                k.launch_add_bias(self.k_buf, lw.bk, b, nkv)?;
            }
            if !lw.bv.is_null() {
                k.launch_add_bias(self.v_buf, lw.bv, b, nkv)?;
            }
            k.launch_rope_batched(
                self.q,
                self.k_buf,
                self.pos_dev,
                b,
                c.n_heads as i32,
                c.n_kv_heads as i32,
                c.head_dim as i32,
                c.rope_theta,
            )?;
            let (kc, vc) = self.kv_cache[li];
            if f16 {
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
            gemm(
                self.proj,
                self.attn,
                lw.wo,
                l16.map_or(std::ptr::null_mut(), |l| l.wo),
                d,
                nq,
            )?;
            k.launch_add(self.x, self.proj, b * d)?;
            k.launch_rms_norm(self.x, lw.rms_mlp, self.xn2, b, d, c.rms_eps)?;
            if c.num_experts > 0 {
                let ne = c.num_experts as i32;
                let topk = c.num_experts_per_tok.min(c.num_experts) as i32;
                if topk > 0 {
                    // Router logits [B, ne] (shared input -> batched GEMM).
                    gemm(
                        self.router,
                        self.xn2,
                        lw.moe_router,
                        l16.map_or(std::ptr::null_mut(), |l| l.moe_router),
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
                    )?;
                    // Count routed (token, slot) pairs per expert on device.
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
                    // GPU-side exclusive prefix sum -> gather offsets. The
                    // per-expert counts are still read back once per layer for
                    // the host GEMM loop (hipBLAS batch counts are host-side);
                    // the gather itself no longer needs a host round-trip.
                    k.launch_moe_prefix_sum(self.counts_dev, self.offsets_dev, ne)?;
                    hip::memcpy_async(
                        self.k.hip(),
                        self.counts_host as *mut core::ffi::c_void,
                        self.counts_dev as *const core::ffi::c_void,
                        (ne as usize) * 4,
                        hip::HIP_MEMCPY_DEVICE_TO_HOST,
                        self.k.stream,
                    )?;
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
                    // Make the async counts readback visible to the host loop.
                    self.k.sync()?;
                    let counts: Vec<i32> = (0..ne)
                        .map(|e| unsafe { *self.counts_host.add(e as usize) })
                        .collect();
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
                    // Per-expert grouped GEMMs (counts known on host after the
                    // single D2H read; no per-expert sync). The running base
                    // mirrors the device prefix-sum output.
                    let d_usize = d as usize;
                    let inter_usize = inter as usize;
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
                        let wg32 = unsafe { lw.moe_wg.add(e * inter_usize * d_usize) };
                        let wu32 = unsafe { lw.moe_wu.add(e * inter_usize * d_usize) };
                        let wd32 = unsafe { lw.moe_wd.add(e * d_usize * inter_usize) };
                        let (wg16, wu16, wd16) = if f16 {
                            let l = self.layers_f16[li];
                            (
                                unsafe { l.moe_wg.add(e * inter_usize * d_usize) },
                                unsafe { l.moe_wu.add(e * inter_usize * d_usize) },
                                unsafe { l.moe_wd.add(e * d_usize * inter_usize) },
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
                        gemm_e(self.gate_all, xg_e, wg32, wg16, inter, d)?;
                        gemm_e(self.up_all, xg_e, wu32, wu16, inter, d)?;
                        k.launch_silu_mul(self.up_all, self.gate_all, self.eh_all, cnt * inter)?;
                        gemm_e(down_e, self.eh_all, wd32, wd16, d, inter)?;
                        k.launch_moe_scatter_add(
                            self.h_acc,
                            unsafe { self.row_idx.add(base) },
                            unsafe { self.gw.add(base) },
                            down_e,
                            cnt,
                            d,
                        )?;
                    }
                    k.launch_add(self.x, self.h_acc, b * d)?;
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
                k.launch_silu_mul(self.up, self.gate, self.h, b * inter)?;
                gemm(
                    self.proj,
                    self.h,
                    lw.wd,
                    l16.map_or(std::ptr::null_mut(), |l| l.wd),
                    d,
                    inter,
                )?;
                k.launch_add(self.x, self.proj, b * d)?;
            }
        }
        k.launch_rms_norm(self.x, self.rms_final_dev, self.xn, b, d, c.rms_eps)?;
        if f16 {
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
        Ok(())
    }

    /// Batched greedy sampling: argmax per row, read back only `batch` tokens.
    fn sample(&self, n: usize) -> Result<Vec<u32>, Error> {
        let b = n as i32;
        self.k.launch_argmax_batched(
            self.logits,
            self.out_tok_dev,
            self.cfg.vocab_size as i32,
            b,
        )?;
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

    pub fn decode_step_explicit(
        &mut self,
        tokens: &[u32],
        lens: &[u32],
        slots: &[u32],
        params: &mut [SamplingParams],
        counts: &[Vec<(u32, u32)>],
        bias: &[Vec<(u32, f32)>],
    ) -> Result<SampleOutput, Error> {
        let n = tokens.len();
        assert_eq!(n, lens.len(), "tokens and lens must be equal length");
        assert_eq!(n, slots.len(), "tokens and slots must be equal length");
        assert!(n <= self.rows, "active count exceeds row capacity");
        // Prefill-attention runs are currently disabled: the naive shared-KV
        // kernel is occupancy-bound and slower than decode attention on this
        // GPU (see roadmap). All rows use decode attention (run_mask = 0).
        let runs: Vec<i32> = Vec::new();
        let run_mask = vec![0i32; n];
        let num_runs = 0;
        unsafe {
            for i in 0..n {
                *self.tokens_host.add(i) = tokens[i] as i32;
                *self.pos_host.add(i) = lens[i] as i32;
                *self.slots_host.add(i) = slots[i] as i32;
                *self.run_mask_host.add(i) = run_mask[i];
            }
            for (k, &v) in runs.iter().enumerate() {
                *self.runs_host.add(k) = v;
            }
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
            if num_runs > 0 {
                hip::memcpy_async(
                    self.k.hip(),
                    self.runs_dev as *mut core::ffi::c_void,
                    self.runs_host as *const core::ffi::c_void,
                    num_runs * 4 * 4,
                    hip::HIP_MEMCPY_HOST_TO_DEVICE,
                    self.k.stream,
                )?;
            }
        }
        self.run_kernels(n as i32, self.slots_dev, self.run_mask_dev, num_runs as i32)?;
        self.sampler
            .sample_batched(self.logits, params, counts, bias, self.cfg.vocab_size)
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
        Ok(())
    }

    /// The configured batch size.
    #[must_use]
    pub const fn batch(&self) -> usize {
        self.batch
    }

    /// Syncs the stream and copies the last step's logits (`[batch, vocab]`)
    /// back to host (debug / numeric validation).
    pub fn read_logits(&self) -> Result<Vec<f32>, Error> {
        self.k.sync()?;
        let n = self.batch * self.cfg.vocab_size;
        let mut out = vec![0.0f32; n];
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
