//! Batched decode: multiple sequences share one forward pass.
//!
//! This is the P2 fix for the single-sequence bottleneck: projections become
//! `m = batch` GEMMs instead of `m = 1` (hipBLAS m=1 is extremely slow), which
//! also enables continuous batching. Each sequence keeps its own length, so
//! sequences at different positions are processed together (attention masks by
//! per-sequence position).

use crate::kernels::HipKernels;
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
}

/// Multi-sequence transformer on the GPU.
pub struct BatchedModel {
    cfg: Config,
    /// Max sequences per step.
    batch: usize,
    k: Arc<HipKernels>,
    // device inputs
    tokens_dev: *mut i32,
    pos_dev: *mut i32,
    // pinned host inputs
    tokens_host: *mut i32,
    pos_host: *mut i32,
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
    // weights
    emb_dev: *mut f32,
    rms_final_dev: *mut f32,
    lm_head_dev: *mut f32,
    layers_dev: Vec<LayerDev>,
    /// KV caches: (k, v) per layer, layout `[batch, max_seq, kv_heads, head_dim]`.
    kv_cache: Vec<(*mut f32, *mut f32)>,
    /// Per-sequence lengths (host).
    lens: Vec<u32>,
    allocs: Vec<*mut core::ffi::c_void>,
    host_pins: Vec<*mut core::ffi::c_void>,
}

impl BatchedModel {
    /// Builds a batched model for `batch` sequences and uploads `w`.
    pub fn new(hip: Arc<Hip>, cfg: Config, w: &Weights, batch: usize) -> Result<Self, Error> {
        assert!(batch >= 1, "batch must be >= 1");
        let k = Arc::new(HipKernels::new(Arc::clone(&hip))?);
        let mut m = Self {
            cfg,
            batch,
            k,
            tokens_dev: std::ptr::null_mut(),
            pos_dev: std::ptr::null_mut(),
            tokens_host: std::ptr::null_mut(),
            pos_host: std::ptr::null_mut(),
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
            kv_cache: Vec::new(),
            lens: vec![0; batch],
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

    fn alloc_buffers(&mut self) -> Result<(), Error> {
        let c = self.cfg;
        let b = self.batch;
        let d = c.d_model;
        let nq = c.n_heads * c.head_dim;
        let nkv = c.n_kv_heads * c.head_dim;
        let inter = c.intermediate_size;

        self.tokens_dev = self.dalloc(b * 4)? as *mut i32;
        self.pos_dev = self.dalloc(b * 4)? as *mut i32;
        let th = hip::host_malloc(self.k.hip(), b * 4)?;
        let ph = hip::host_malloc(self.k.hip(), b * 4)?;
        self.tokens_host = th as *mut i32;
        self.pos_host = ph as *mut i32;
        self.host_pins.push(th);
        self.host_pins.push(ph);

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
        self.out_tok_dev = self.dalloc(b * 4)? as *mut i32;
        let oh = hip::host_malloc(self.k.hip(), b * 4)?;
        self.out_tok_host = oh as *mut i32;
        self.host_pins.push(oh);

        let kv_bytes = b * c.max_seq_len * c.n_kv_heads * c.head_dim * 4;
        for _ in 0..c.n_layers {
            let kk = self.dalloc(kv_bytes)?;
            let vv = self.dalloc(kv_bytes)?;
            self.kv_cache.push((kk, vv));
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
        }
        Ok(())
    }

    /// Zeroes KV caches and resets all sequence lengths.
    pub fn reset_state(&mut self) -> Result<(), Error> {
        let kv_bytes =
            self.batch * self.cfg.max_seq_len * self.cfg.n_kv_heads * self.cfg.head_dim * 4;
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
        }
        self.run_kernels()?;
        let next = self.sample()?;
        for l in self.lens.iter_mut() {
            *l += 1;
        }
        Ok(next)
    }

    fn run_kernels(&self) -> Result<(), Error> {
        let c = self.cfg;
        let b = self.batch as i32;
        let d = c.d_model as i32;
        let nq = (c.n_heads * c.head_dim) as i32;
        let nkv = (c.n_kv_heads * c.head_dim) as i32;
        let inter = c.intermediate_size as i32;
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let k = &self.k;

        k.launch_embed_batched(self.tokens_dev, self.emb_dev, self.x, d, b)?;
        for (li, lw) in self.layers_dev.iter().enumerate() {
            k.launch_rms_norm(self.x, lw.rms_attn, self.xn, b, d, c.rms_eps)?;
            k.gemm_batched(self.q, self.xn, lw.wq, b, nq, d)?;
            k.gemm_batched(self.k_buf, self.xn, lw.wk, b, nkv, d)?;
            k.gemm_batched(self.v_buf, self.xn, lw.wv, b, nkv, d)?;
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
            k.launch_kv_store_batched(
                self.k_buf,
                kc,
                self.pos_dev,
                b,
                c.n_kv_heads as i32,
                c.head_dim as i32,
                c.max_seq_len as i32,
            )?;
            k.launch_kv_store_batched(
                self.v_buf,
                vc,
                self.pos_dev,
                b,
                c.n_kv_heads as i32,
                c.head_dim as i32,
                c.max_seq_len as i32,
            )?;
            k.launch_attn_decode_batched(
                self.q,
                kc,
                vc,
                self.attn,
                self.pos_dev,
                b,
                c.n_heads as i32,
                c.n_kv_heads as i32,
                c.head_dim as i32,
                scale,
                c.max_seq_len as i32,
            )?;
            k.gemm_batched(self.proj, self.attn, lw.wo, b, d, nq)?;
            k.launch_add(self.x, self.proj, b * d)?;
            k.launch_rms_norm(self.x, lw.rms_mlp, self.xn2, b, d, c.rms_eps)?;
            k.gemm_batched(self.gate, self.xn2, lw.wg, b, inter, d)?;
            k.gemm_batched(self.up, self.xn2, lw.wu, b, inter, d)?;
            k.launch_silu_mul(self.gate, self.up, self.h, b * inter)?;
            k.gemm_batched(self.proj, self.h, lw.wd, b, d, inter)?;
            k.launch_add(self.x, self.proj, b * d)?;
        }
        k.launch_rms_norm(self.x, self.rms_final_dev, self.xn, b, d, c.rms_eps)?;
        k.gemm_batched(
            self.logits,
            self.xn,
            self.lm_head_dev,
            b,
            c.vocab_size as i32,
            d,
        )?;
        Ok(())
    }

    /// Batched greedy sampling: argmax per row, read back only `batch` tokens.
    fn sample(&self) -> Result<Vec<u32>, Error> {
        let b = self.batch as i32;
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
                self.batch * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )?;
        }
        let mut out = Vec::with_capacity(self.batch);
        unsafe {
            for i in 0..self.batch {
                out.push(*self.out_tok_host.add(i) as u32);
            }
        }
        Ok(out)
    }

    /// The configured batch size.
    #[must_use]
    pub const fn batch(&self) -> usize {
        self.batch
    }
}

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
