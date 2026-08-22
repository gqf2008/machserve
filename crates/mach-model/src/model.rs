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

use crate::kernels::HipKernels;
use crate::{Config, Error, Weights};
use mach_engine::graph::{GraphCapture, GraphHandle};
use mach_engine::hip::HipGraphCapture;
use mach_kernel_sys::hip::{self, Hip};
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
}

/// The GPU transformer.
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
    // weights on device
    emb_dev: *mut f32,
    rms_final_dev: *mut f32,
    lm_head_dev: *mut f32,
    layers_dev: Vec<LayerDev>,
    // KV caches: (k, v) per layer
    kv_cache: Vec<(*mut f32, *mut f32)>,
    /// All device allocations (freed on drop).
    allocs: Vec<*mut core::ffi::c_void>,
    /// Number of tokens stored so far.
    pos: usize,
    host_pins: Vec<*mut core::ffi::c_void>,
}

impl GpuModel {
    /// Builds a GPU model and uploads `w` to device memory.
    pub fn new(hip: Arc<Hip>, cfg: Config, w: &Weights) -> Result<Self, Error> {
        let k = Arc::new(HipKernels::new(Arc::clone(&hip))?);
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
            kv_cache: Vec::new(),
            allocs: Vec::new(),
            pos: 0,
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

        let kv_bytes = c.max_seq_len * c.n_kv_heads * c.head_dim * 4;
        for _ in 0..c.n_layers {
            let kk = self.dalloc(kv_bytes)?;
            let vv = self.dalloc(kv_bytes)?;
            self.kv_cache.push((kk, vv));
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

    fn upload_weights(&mut self, w: &Weights) -> Result<(), Error> {
        let c = self.cfg;
        self.emb_dev = self.dalloc(w.tok_emb.len() * 4)?;
        self.rms_final_dev = self.dalloc(w.rms_final.len() * 4)?;
        self.lm_head_dev = self.dalloc(w.lm_head.len() * 4)?;
        self.upload(self.emb_dev, &w.tok_emb)?;
        self.upload(self.rms_final_dev, &w.rms_final)?;
        self.upload(self.lm_head_dev, &w.lm_head)?;

        let d = c.d_model;
        let nq = c.n_heads * c.head_dim;
        let nkv = c.n_kv_heads * c.head_dim;
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

        k.launch_embed(self.dev_tok, self.emb_dev, self.x, d)?;
        for (li, lw) in self.layers_dev.iter().enumerate() {
            let nq = (c.n_heads * c.head_dim) as i32;
            let nkv = (c.n_kv_heads * c.head_dim) as i32;

            k.launch_rms_norm(self.x, lw.rms_attn, self.xn, 1, d, c.rms_eps)?;
            k.gemm(self.q, self.xn, lw.wq, nq, d)?;
            k.gemm(self.k_buf, self.xn, lw.wk, nkv, d)?;
            k.gemm(self.v_buf, self.xn, lw.wv, nkv, d)?;
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
            )?;
            k.gemm(self.proj, self.attn, lw.wo, d, nq)?;
            k.launch_add(self.x, self.proj, d)?;

            let inter = c.intermediate_size as i32;
            k.launch_rms_norm(self.x, lw.rms_mlp, self.xn2, 1, d, c.rms_eps)?;
            k.gemm(self.gate, self.xn2, lw.wg, inter, d)?;
            k.gemm(self.up, self.xn2, lw.wu, inter, d)?;
            k.launch_silu_mul(self.gate, self.up, self.h, inter)?;
            k.gemm(self.proj, self.h, lw.wd, d, inter)?;
            k.launch_add(self.x, self.proj, d)?;
        }
        k.launch_rms_norm(self.x, self.rms_final_dev, self.xn, 1, d, c.rms_eps)?;
        k.gemm(
            self.logits,
            self.xn,
            self.lm_head_dev,
            c.vocab_size as i32,
            d,
        )?;
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
    pub fn decode_step(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        if self.pos >= self.cfg.max_seq_len {
            return Err(Error::Model("sequence length exceeded".into()));
        }
        self.update_inputs(token)?;
        self.run_kernels()?;
        let out = self.read_logits()?;
        self.pos += 1;
        Ok(out)
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
        for &p in &self.allocs {
            let _ = hip::free(hip, p);
        }
        for &p in &self.host_pins {
            let _ = hip::host_free(hip, p);
        }
    }
}
