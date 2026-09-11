//! HIP vision-tower runtime for Qwen3.5/Qwen3.8.
//!
//! The first GPU path is deliberately correctness-first: weights and
//! activations stay f32, linear layers use hipBLAS, and the four elementwise /
//! attention kernels are the new vision kernels in `kernels.rs`. Position
//! interpolation and RoPE tables are prepared on the host (small relative to
//! the transformer) and uploaded with the pixel patches.

use crate::Error;
use crate::kernels::HipKernels;
use crate::vision::{
    VisionConfig, VisionGrid, VisionPrepared, VisionWeights, prepare_vision_inputs,
    validate_vision_weights,
};
use mach_kernel_sys::hip;
use std::sync::Arc;

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize, Error> {
    a.checked_mul(b)
        .ok_or_else(|| Error::InvalidArgument(format!("vision {what} size overflow")))
}

fn validate_scratch_sizes(cfg: &VisionConfig, max_tokens: usize) -> Result<(), Error> {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim();
    let inter = cfg.intermediate_size;
    let merge_unit = cfg.spatial_merge_size * cfg.spatial_merge_size;
    let patch_dim = checked_mul(
        checked_mul(cfg.in_channels, cfg.temporal_patch_size, "patch")?,
        checked_mul(cfg.patch_size, cfg.patch_size, "patch")?,
        "patch",
    )?;
    let qkv_dim = checked_mul(3, h, "qkv")?;
    let merger_dim = checked_mul(h, merge_unit, "merger")?;
    let max_merged = max_tokens / merge_unit.max(1);
    let _seg_bytes = checked_mul(max_tokens, 4, "segment")?;
    for (name, size) in [
        ("pixel", checked_mul(max_tokens, patch_dim, "pixel")?),
        ("pos", checked_mul(max_tokens, h, "pos")?),
        ("rope", checked_mul(max_tokens, hd, "rope")?),
        ("qkv", checked_mul(max_tokens, qkv_dim, "qkv")?),
        ("mlp", checked_mul(max_tokens, inter, "mlp")?),
        ("out", checked_mul(max_merged, cfg.out_hidden_size, "out")?),
    ] {
        if size > i32::MAX as usize {
            return Err(Error::InvalidArgument(format!(
                "vision {name} scratch {size} exceeds i32"
            )));
        }
    }
    for (name, size) in [
        ("hidden", h),
        ("head_dim", hd),
        ("intermediate", inter),
        ("qkv", qkv_dim),
        ("merger", merger_dim),
        ("out", cfg.out_hidden_size),
    ] {
        if size > i32::MAX as usize {
            return Err(Error::InvalidArgument(format!(
                "vision {name} dimension {size} exceeds i32"
            )));
        }
    }
    Ok(())
}
struct LayerDev {
    norm1_w: *mut f32,
    norm1_b: *mut f32,
    qkv_w: *mut f32,
    qkv_b: *mut f32,
    proj_w: *mut f32,
    proj_b: *mut f32,
    norm2_w: *mut f32,
    norm2_b: *mut f32,
    fc1_w: *mut f32,
    fc1_b: *mut f32,
    fc2_w: *mut f32,
    fc2_b: *mut f32,
}

/// f32 GPU vision tower with persistent weights and scratch buffers.
pub struct VisionGpu {
    hip: Arc<hip::Hip>,
    k: HipKernels,
    cfg: VisionConfig,
    max_tokens: usize,
    allocs: Vec<*mut core::ffi::c_void>,
    layers: Vec<LayerDev>,
    patch_w: *mut f32,
    patch_b: *mut f32,
    merger_norm_w: *mut f32,
    merger_norm_b: *mut f32,
    merger_fc1_w: *mut f32,
    merger_fc1_b: *mut f32,
    merger_fc2_w: *mut f32,
    merger_fc2_b: *mut f32,
    pixel_dev: *mut f32,
    pos_dev: *mut f32,
    cos_dev: *mut f32,
    sin_dev: *mut f32,
    seg_start_dev: *mut i32,
    seg_len_dev: *mut i32,
    x: *mut f32,
    norm: *mut f32,
    qkv: *mut f32,
    attn: *mut f32,
    proj: *mut f32,
    fc1: *mut f32,
    out: *mut f32,
}

impl VisionGpu {
    /// Build a GPU vision tower with scratch sized for `max_tokens` patches.
    pub fn new(
        hip: Arc<hip::Hip>,
        cfg: VisionConfig,
        w: &VisionWeights,
        max_tokens: usize,
    ) -> Result<Self, Error> {
        cfg.validate()?;
        validate_vision_weights(&cfg, w)?;
        validate_scratch_sizes(&cfg, max_tokens)?;
        if max_tokens == 0 {
            return Err(Error::InvalidArgument(
                "vision max_tokens must be positive".into(),
            ));
        }
        if w.layers.len() != cfg.depth {
            return Err(Error::Model(format!(
                "vision weights have {} layers, expected {}",
                w.layers.len(),
                cfg.depth
            )));
        }
        let k = HipKernels::new(Arc::clone(&hip))?;
        let mut this = Self {
            hip,
            k,
            cfg,
            max_tokens,
            allocs: Vec::new(),
            layers: Vec::with_capacity(cfg.depth),
            patch_w: std::ptr::null_mut(),
            patch_b: std::ptr::null_mut(),
            merger_norm_w: std::ptr::null_mut(),
            merger_norm_b: std::ptr::null_mut(),
            merger_fc1_w: std::ptr::null_mut(),
            merger_fc1_b: std::ptr::null_mut(),
            merger_fc2_w: std::ptr::null_mut(),
            merger_fc2_b: std::ptr::null_mut(),
            pixel_dev: std::ptr::null_mut(),
            pos_dev: std::ptr::null_mut(),
            cos_dev: std::ptr::null_mut(),
            sin_dev: std::ptr::null_mut(),
            seg_start_dev: std::ptr::null_mut(),
            seg_len_dev: std::ptr::null_mut(),
            x: std::ptr::null_mut(),
            norm: std::ptr::null_mut(),
            qkv: std::ptr::null_mut(),
            attn: std::ptr::null_mut(),
            proj: std::ptr::null_mut(),
            fc1: std::ptr::null_mut(),
            out: std::ptr::null_mut(),
        };
        this.patch_w = this.upload(&w.patch_embed_weight)?;
        this.patch_b = this.upload(&w.patch_embed_bias)?;
        for layer in &w.layers {
            let norm1_w = this.upload(&layer.norm1_weight)?;
            let norm1_b = this.upload(&layer.norm1_bias)?;
            let qkv_w = this.upload(&layer.qkv.weight)?;
            let qkv_b = this.upload(&layer.qkv.bias)?;
            let proj_w = this.upload(&layer.attn_proj.weight)?;
            let proj_b = this.upload(&layer.attn_proj.bias)?;
            let norm2_w = this.upload(&layer.norm2_weight)?;
            let norm2_b = this.upload(&layer.norm2_bias)?;
            let fc1_w = this.upload(&layer.mlp_fc1.weight)?;
            let fc1_b = this.upload(&layer.mlp_fc1.bias)?;
            let fc2_w = this.upload(&layer.mlp_fc2.weight)?;
            let fc2_b = this.upload(&layer.mlp_fc2.bias)?;
            this.layers.push(LayerDev {
                norm1_w,
                norm1_b,
                qkv_w,
                qkv_b,
                proj_w,
                proj_b,
                norm2_w,
                norm2_b,
                fc1_w,
                fc1_b,
                fc2_w,
                fc2_b,
            });
        }
        this.merger_norm_w = this.upload(&w.merger_norm_weight)?;
        this.merger_norm_b = this.upload(&w.merger_norm_bias)?;
        this.merger_fc1_w = this.upload(&w.merger_fc1.weight)?;
        this.merger_fc1_b = this.upload(&w.merger_fc1.bias)?;
        this.merger_fc2_w = this.upload(&w.merger_fc2.weight)?;
        this.merger_fc2_b = this.upload(&w.merger_fc2.bias)?;
        this.alloc_scratch()?;
        Ok(this)
    }

    /// Host-side preparation helper for callers that want to reuse the CPU
    /// reference's position/RoPE/segment conventions.
    pub fn prepare(
        cfg: &VisionConfig,
        w: &VisionWeights,
        grids: &[VisionGrid],
    ) -> Result<VisionPrepared, Error> {
        prepare_vision_inputs(cfg, w, grids)
    }

    /// Run the vision tower and return merged features `[tokens/merge^2, out]`.
    pub fn forward(
        &mut self,
        pixel_values: &[f32],
        prep: &VisionPrepared,
    ) -> Result<Vec<f32>, Error> {
        let hidden = self.cfg.hidden_size;
        let hd = self.cfg.head_dim();
        let heads = self.cfg.num_heads;
        let inter = self.cfg.intermediate_size;
        let merge_unit = self.cfg.spatial_merge_size * self.cfg.spatial_merge_size;
        let merger_dim = hidden * merge_unit;
        let patch_dim = self.cfg.in_channels
            * self.cfg.temporal_patch_size
            * self.cfg.patch_size
            * self.cfg.patch_size;
        let tokens = prep.tokens;
        let merged = prep.merged_tokens;
        if tokens == 0 || merged == 0 {
            return Err(Error::InvalidArgument("vision batch is empty".into()));
        }
        if tokens > self.max_tokens {
            return Err(Error::InvalidArgument(format!(
                "vision batch has {tokens} patches, scratch holds {}",
                self.max_tokens
            )));
        }
        let expected_pixels = checked_mul(tokens, patch_dim, "pixel input")?;
        if pixel_values.len() != expected_pixels {
            return Err(Error::InvalidArgument(format!(
                "vision pixel_values has {} elements, expected {expected_pixels}",
                pixel_values.len()
            )));
        }
        if !tokens.is_multiple_of(merge_unit)
            || merged != tokens / merge_unit
            || prep.pos_embeddings.len() != checked_mul(tokens, hidden, "pos")?
            || prep.cos.len() != checked_mul(tokens, hd, "rope")?
            || prep.sin.len() != checked_mul(tokens, hd, "rope")?
            || prep.seg_start.len() != tokens
            || prep.seg_len.len() != tokens
        {
            return Err(Error::InvalidArgument(
                "vision prepared tensor shape mismatch".into(),
            ));
        }
        let mut actual_max = 0usize;
        for i in 0..tokens {
            let start = prep.seg_start[i];
            let len = prep.seg_len[i];
            if start < 0 || len <= 0 {
                return Err(Error::InvalidArgument(format!(
                    "vision segment {i} has invalid start/len ({start}, {len})"
                )));
            }
            let end = (start as usize)
                .checked_add(len as usize)
                .ok_or_else(|| Error::InvalidArgument("vision segment end overflow".into()))?;
            if end > tokens {
                return Err(Error::InvalidArgument(format!(
                    "vision segment {i} [{start}, {end}) exceeds {tokens} tokens"
                )));
            }
            actual_max = actual_max.max(len as usize);
        }
        if prep.max_seg != actual_max {
            return Err(Error::InvalidArgument(format!(
                "vision prepared max_seg={} but descriptors imply {actual_max}",
                prep.max_seg
            )));
        }
        if prep.max_seg > 8192 {
            return Err(Error::InvalidArgument(format!(
                "vision attention segment {} exceeds the 8192-token GPU limit",
                prep.max_seg
            )));
        }

        self.copy_h2d(self.pixel_dev, pixel_values)?;
        self.copy_h2d(self.pos_dev, &prep.pos_embeddings)?;
        self.copy_h2d(self.cos_dev, &prep.cos)?;
        self.copy_h2d(self.sin_dev, &prep.sin)?;
        self.copy_h2d(self.seg_start_dev, &prep.seg_start)?;
        self.copy_h2d(self.seg_len_dev, &prep.seg_len)?;

        let t = i32::try_from(tokens)
            .map_err(|_| Error::InvalidArgument("vision token count exceeds i32".into()))?;
        self.k.gemm_batched(
            self.x,
            self.pixel_dev,
            self.patch_w,
            t,
            hidden as i32,
            patch_dim as i32,
        )?;
        self.k
            .launch_add_bias(self.x, self.patch_b, t, hidden as i32)?;
        self.k
            .launch_add(self.x, self.pos_dev, (tokens * hidden) as i32)?;

        let scale = 1.0 / (hd as f32).sqrt();
        for layer in &self.layers {
            self.k.launch_layer_norm(
                self.x,
                layer.norm1_w,
                layer.norm1_b,
                self.norm,
                t,
                hidden as i32,
                1e-6,
            )?;
            self.k.gemm_batched(
                self.qkv,
                self.norm,
                layer.qkv_w,
                t,
                (3 * hidden) as i32,
                hidden as i32,
            )?;
            self.k
                .launch_add_bias(self.qkv, layer.qkv_b, t, (3 * hidden) as i32)?;
            self.k.launch_vision_rope_apply(
                self.qkv,
                self.cos_dev,
                self.sin_dev,
                t,
                heads as i32,
                hd as i32,
            )?;
            self.k.launch_vision_attn(
                self.qkv,
                self.attn,
                self.seg_start_dev,
                self.seg_len_dev,
                t,
                heads as i32,
                hd as i32,
                scale,
                prep.max_seg as i32,
            )?;
            self.k.gemm_batched(
                self.proj,
                self.attn,
                layer.proj_w,
                t,
                hidden as i32,
                hidden as i32,
            )?;
            self.k
                .launch_add_bias(self.proj, layer.proj_b, t, hidden as i32)?;
            self.k
                .launch_add(self.x, self.proj, (tokens * hidden) as i32)?;

            self.k.launch_layer_norm(
                self.x,
                layer.norm2_w,
                layer.norm2_b,
                self.norm,
                t,
                hidden as i32,
                1e-6,
            )?;
            self.k.gemm_batched(
                self.fc1,
                self.norm,
                layer.fc1_w,
                t,
                inter as i32,
                hidden as i32,
            )?;
            self.k
                .launch_add_bias(self.fc1, layer.fc1_b, t, inter as i32)?;
            self.k
                .launch_gelu(self.fc1, (tokens * inter) as i32, true)?;
            self.k.gemm_batched(
                self.proj,
                self.fc1,
                layer.fc2_w,
                t,
                hidden as i32,
                inter as i32,
            )?;
            self.k
                .launch_add_bias(self.proj, layer.fc2_b, t, hidden as i32)?;
            self.k
                .launch_add(self.x, self.proj, (tokens * hidden) as i32)?;
        }

        // Merger: the normalized per-token rows are contiguous; viewing four
        // adjacent rows as one 4*hidden vector is the HF `view(-1, 4*hidden)`.
        self.k.launch_layer_norm(
            self.x,
            self.merger_norm_w,
            self.merger_norm_b,
            self.x,
            t,
            hidden as i32,
            1e-6,
        )?;
        self.k.gemm_batched(
            self.proj,
            self.x,
            self.merger_fc1_w,
            i32::try_from(merged).map_err(|_| {
                Error::InvalidArgument("vision merged token count exceeds i32".into())
            })?,
            merger_dim as i32,
            merger_dim as i32,
        )?;
        self.k.launch_add_bias(
            self.proj,
            self.merger_fc1_b,
            i32::try_from(merged).map_err(|_| {
                Error::InvalidArgument("vision merged token count exceeds i32".into())
            })?,
            merger_dim as i32,
        )?;
        self.k
            .launch_gelu(self.proj, (merged * merger_dim) as i32, false)?;
        self.k.gemm_batched(
            self.out,
            self.proj,
            self.merger_fc2_w,
            i32::try_from(merged).map_err(|_| {
                Error::InvalidArgument("vision merged token count exceeds i32".into())
            })?,
            self.cfg.out_hidden_size as i32,
            merger_dim as i32,
        )?;
        self.k.launch_add_bias(
            self.out,
            self.merger_fc2_b,
            i32::try_from(merged).map_err(|_| {
                Error::InvalidArgument("vision merged token count exceeds i32".into())
            })?,
            self.cfg.out_hidden_size as i32,
        )?;

        let mut host = vec![0.0f32; merged * self.cfg.out_hidden_size];
        // SAFETY: self.k.stream is owned by this VisionGpu and was created
        // by HipKernels::new; no other thread can destroy it while self is
        // mutably borrowed for this synchronous forward.
        unsafe {
            hip::check(
                &self.hip,
                (self.hip.api.hip_stream_synchronize)(self.k.stream),
            )?;
        }
        hip::memcpy(
            &self.hip,
            host.as_mut_ptr() as *mut core::ffi::c_void,
            self.out as *const core::ffi::c_void,
            host.len() * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )?;
        Ok(host)
    }

    fn alloc_bytes(&mut self, bytes: usize) -> Result<*mut core::ffi::c_void, Error> {
        let bytes = bytes.max(4);
        let p = hip::malloc(&self.hip, bytes)?;
        self.allocs.push(p);
        Ok(p)
    }

    fn alloc_f32(&mut self, n: usize) -> Result<*mut f32, Error> {
        Ok(self.alloc_bytes(n.saturating_mul(4).max(4))? as *mut f32)
    }

    fn upload(&mut self, data: &[f32]) -> Result<*mut f32, Error> {
        let p = self.alloc_f32(data.len())?;
        if !data.is_empty() {
            hip::memcpy(
                &self.hip,
                p as *mut core::ffi::c_void,
                data.as_ptr() as *const core::ffi::c_void,
                data.len() * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )?;
        }
        Ok(p)
    }

    fn copy_h2d<T>(&self, dst: *mut T, data: &[T]) -> Result<(), Error> {
        if !data.is_empty() {
            hip::memcpy(
                &self.hip,
                dst as *mut core::ffi::c_void,
                data.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(data),
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )?;
        }
        Ok(())
    }

    fn alloc_scratch(&mut self) -> Result<(), Error> {
        let t = self.max_tokens;
        let hidden = self.cfg.hidden_size;
        let hd = self.cfg.head_dim();
        let inter = self.cfg.intermediate_size;
        let merge_unit = self.cfg.spatial_merge_size * self.cfg.spatial_merge_size;
        let patch_dim = self.cfg.in_channels
            * self.cfg.temporal_patch_size
            * self.cfg.patch_size
            * self.cfg.patch_size;
        self.pixel_dev = self.alloc_f32(t * patch_dim)?;
        self.pos_dev = self.alloc_f32(t * hidden)?;
        self.cos_dev = self.alloc_f32(t * hd)?;
        self.sin_dev = self.alloc_f32(t * hd)?;
        self.seg_start_dev = self.alloc_bytes(t * 4)? as *mut i32;
        self.seg_len_dev = self.alloc_bytes(t * 4)? as *mut i32;
        self.x = self.alloc_f32(t * hidden)?;
        self.norm = self.alloc_f32(t * hidden)?;
        self.qkv = self.alloc_f32(t * 3 * hidden)?;
        self.attn = self.alloc_f32(t * hidden)?;
        self.proj = self.alloc_f32(t * hidden)?;
        self.fc1 = self.alloc_f32(t * inter)?;
        let max_merged = t / merge_unit.max(1);
        self.out = self.alloc_f32(max_merged * self.cfg.out_hidden_size)?;
        Ok(())
    }
}

impl Drop for VisionGpu {
    fn drop(&mut self) {
        for &p in &self.allocs {
            let _ = hip::free(&self.hip, p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_size_overflow_is_rejected() {
        let cfg = VisionConfig::default();
        assert!(validate_scratch_sizes(&cfg, 16).is_ok());
        assert!(validate_scratch_sizes(&cfg, 1usize << 63).is_err());
    }
}
