//! Qwen3.5-family vision tower configuration and checkpoint metadata.
//!
//! Stage C is intentionally split from the text `Config`: the vision stack has
//! its own dimensions, token ids and M-RoPE metadata, while the text runtime
//! keeps its existing shape. This module currently owns parsing and
//! header-only checkpoint validation; tensor loading and the CPU/GPU forward
//! live in follow-up increments.

use crate::Error;

/// Qwen3.5/Qwen3.8 vision-tower configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionConfig {
    /// Number of transformer blocks in the vision tower.
    pub depth: usize,
    /// Vision hidden dimension.
    pub hidden_size: usize,
    /// Vision MLP intermediate dimension.
    pub intermediate_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Input image channels.
    pub in_channels: usize,
    /// Spatial patch size.
    pub patch_size: usize,
    /// Temporal patch size used by the 3D patch embedding.
    pub temporal_patch_size: usize,
    /// Spatial merge factor between the vision tower and text embeddings.
    pub spatial_merge_size: usize,
    /// Number of learned square position embeddings.
    pub num_position_embeddings: usize,
    /// Output width consumed by the text hidden size.
    pub out_hidden_size: usize,
    /// `<|image_pad|>` token id.
    pub image_token_id: u32,
    /// `<|video_pad|>` token id.
    pub video_token_id: u32,
    /// `<|vision_start|>` token id.
    pub vision_start_token_id: u32,
    /// `<|vision_end|>` token id.
    pub vision_end_token_id: u32,
    /// M-RoPE section split `(temporal, height, width)`.
    pub mrope_section: [usize; 3],
    /// Whether M-RoPE uses the interleaved layout.
    pub mrope_interleaved: bool,
}

/// Header-only summary for a checkpoint's vision tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionCheckpointLayout {
    /// Number of safetensors shards inspected.
    pub shards: usize,
    /// Number of `model.visual.*` tensors found.
    pub tensors: usize,
    /// Sum of vision tensor payload bytes.
    pub payload_bytes: u64,
}

impl Default for VisionConfig {
    fn default() -> Self {
        // Qwen3.8-27B / Qwen3.5-27B text-backbone defaults.
        Self {
            depth: 27,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_heads: 16,
            in_channels: 3,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            num_position_embeddings: 2304,
            out_hidden_size: 5120,
            image_token_id: 248056,
            video_token_id: 248057,
            vision_start_token_id: 248053,
            vision_end_token_id: 248054,
            mrope_section: [11, 11, 10],
            mrope_interleaved: true,
        }
    }
}

impl VisionConfig {
    /// Vision attention head dimension.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Width of the merger input after the `spatial_merge_size^2` shuffle.
    #[must_use]
    pub fn merger_input_dim(&self) -> usize {
        self.hidden_size * self.spatial_merge_size * self.spatial_merge_size
    }

    /// Parse the `vision_config` object from a Qwen3.5/Qwen3.8 HF config.
    pub fn from_hf_json(v: &serde_json::Value) -> Result<Self, Error> {
        let vc = v
            .get("vision_config")
            .and_then(|x| x.as_object())
            .ok_or_else(|| Error::Model("Qwen3.5 config is missing vision_config".into()))?;
        let text = v.get("text_config").unwrap_or(v);
        let rope = text
            .get("rope_parameters")
            .unwrap_or(&serde_json::Value::Null);

        let mut cfg = Self::default();
        cfg.depth = json_usize(vc, "depth", cfg.depth)?;
        cfg.hidden_size = json_usize(vc, "hidden_size", cfg.hidden_size)?;
        cfg.intermediate_size = json_usize(vc, "intermediate_size", cfg.intermediate_size)?;
        cfg.num_heads = json_usize(vc, "num_heads", cfg.num_heads)?;
        cfg.in_channels = json_usize(vc, "in_channels", cfg.in_channels)?;
        cfg.patch_size = json_usize(vc, "patch_size", cfg.patch_size)?;
        cfg.temporal_patch_size = json_usize(vc, "temporal_patch_size", cfg.temporal_patch_size)?;
        cfg.spatial_merge_size = json_usize(vc, "spatial_merge_size", cfg.spatial_merge_size)?;
        cfg.num_position_embeddings =
            json_usize(vc, "num_position_embeddings", cfg.num_position_embeddings)?;
        cfg.out_hidden_size = json_usize(vc, "out_hidden_size", cfg.out_hidden_size)?;
        cfg.image_token_id = json_u32(v, "image_token_id", cfg.image_token_id)?;
        cfg.video_token_id = json_u32(v, "video_token_id", cfg.video_token_id)?;
        cfg.vision_start_token_id =
            json_u32(v, "vision_start_token_id", cfg.vision_start_token_id)?;
        cfg.vision_end_token_id = json_u32(v, "vision_end_token_id", cfg.vision_end_token_id)?;
        if let Some(section) = rope.get("mrope_section").and_then(|x| x.as_array()) {
            if section.len() != 3 {
                return Err(Error::Model(format!(
                    "mrope_section must have 3 entries, got {}",
                    section.len()
                )));
            }
            for (i, x) in section.iter().enumerate() {
                cfg.mrope_section[i] = x
                    .as_u64()
                    .ok_or_else(|| Error::Model("mrope_section entry must be a u64".into()))?
                    as usize;
            }
        }
        if let Some(v) = rope.get("mrope_interleaved").and_then(|x| x.as_bool()) {
            cfg.mrope_interleaved = v;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate dimensions before any checkpoint metadata is compared.
    pub fn validate(&self) -> Result<(), Error> {
        if self.depth == 0 || self.hidden_size == 0 || self.num_heads == 0 {
            return Err(Error::Model(
                "vision depth/hidden_size/num_heads must be positive".into(),
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(Error::Model(format!(
                "vision hidden_size {} is not divisible by num_heads {}",
                self.hidden_size, self.num_heads
            )));
        }
        if self.head_dim() == 0 || !self.head_dim().is_multiple_of(2) {
            return Err(Error::Model(format!(
                "vision head_dim {} must be even and positive for vision RoPE",
                self.head_dim()
            )));
        }
        if self.patch_size == 0 || self.temporal_patch_size == 0 || self.in_channels == 0 {
            return Err(Error::Model(
                "vision patch/temporal/channel sizes must be positive".into(),
            ));
        }
        if self.spatial_merge_size == 0
            || self.num_position_embeddings == 0
            || self.out_hidden_size == 0
            || self.intermediate_size == 0
        {
            return Err(Error::Model(
                "vision merge/position/output/intermediate sizes must be positive".into(),
            ));
        }
        let side = (self.num_position_embeddings as f64).sqrt().round() as usize;
        if side * side != self.num_position_embeddings {
            return Err(Error::Model(format!(
                "num_position_embeddings {} must be a perfect square",
                self.num_position_embeddings
            )));
        }
        if self.mrope_section.iter().all(|&x| x == 0) {
            return Err(Error::Model("mrope_section must not be all zero".into()));
        }
        Ok(())
    }

    /// Expected names and shapes for every vision tensor in the HF checkpoint.
    #[must_use]
    pub fn expected_tensors(&self) -> Vec<(String, Vec<usize>)> {
        let h = self.hidden_size;
        let inter = self.intermediate_size;
        let merge = self.merger_input_dim();
        let mut t = Vec::with_capacity(10 + 12 * self.depth);
        t.push((
            "model.visual.patch_embed.proj.weight".into(),
            vec![
                h,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ],
        ));
        t.push(("model.visual.patch_embed.proj.bias".into(), vec![h]));
        t.push((
            "model.visual.pos_embed.weight".into(),
            vec![self.num_position_embeddings, h],
        ));
        for li in 0..self.depth {
            let p = |suffix: &str| format!("model.visual.blocks.{li}.{suffix}");
            t.push((p("norm1.weight"), vec![h]));
            t.push((p("norm1.bias"), vec![h]));
            t.push((p("attn.qkv.weight"), vec![3 * h, h]));
            t.push((p("attn.qkv.bias"), vec![3 * h]));
            t.push((p("attn.proj.weight"), vec![h, h]));
            t.push((p("attn.proj.bias"), vec![h]));
            t.push((p("norm2.weight"), vec![h]));
            t.push((p("norm2.bias"), vec![h]));
            t.push((p("mlp.linear_fc1.weight"), vec![inter, h]));
            t.push((p("mlp.linear_fc1.bias"), vec![inter]));
            t.push((p("mlp.linear_fc2.weight"), vec![h, inter]));
            t.push((p("mlp.linear_fc2.bias"), vec![h]));
        }
        t.push(("model.visual.merger.norm.weight".into(), vec![h]));
        t.push(("model.visual.merger.norm.bias".into(), vec![h]));
        t.push((
            "model.visual.merger.linear_fc1.weight".into(),
            vec![merge, merge],
        ));
        t.push(("model.visual.merger.linear_fc1.bias".into(), vec![merge]));
        t.push((
            "model.visual.merger.linear_fc2.weight".into(),
            vec![self.out_hidden_size, merge],
        ));
        t.push((
            "model.visual.merger.linear_fc2.bias".into(),
            vec![self.out_hidden_size],
        ));
        t
    }
}

fn json_usize(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: usize,
) -> Result<usize, Error> {
    match obj.get(key) {
        Some(v) => Ok(v
            .as_u64()
            .ok_or_else(|| Error::Model(format!("vision_config.{key} must be a u64")))?
            as usize),
        None => Ok(default),
    }
}

fn json_u32(v: &serde_json::Value, key: &str, default: u32) -> Result<u32, Error> {
    match v.get(key) {
        Some(x) => {
            Ok(x.as_u64()
                .ok_or_else(|| Error::Model(format!("{key} must be a u64")))? as u32)
        }
        None => Ok(default),
    }
}

/// A row-major linear layer (`weight` is `[out_features, in_features]`).
#[derive(Debug, Clone, PartialEq)]
pub struct VisionLinear {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// Per-block vision weights.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionLayerWeights {
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub qkv: VisionLinear,
    pub attn_proj: VisionLinear,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub mlp_fc1: VisionLinear,
    pub mlp_fc2: VisionLinear,
}

/// Host-side f32 vision tower weights.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionWeights {
    pub patch_embed_weight: Vec<f32>,
    pub patch_embed_bias: Vec<f32>,
    pub pos_embed_weight: Vec<f32>,
    pub layers: Vec<VisionLayerWeights>,
    pub merger_norm_weight: Vec<f32>,
    pub merger_norm_bias: Vec<f32>,
    pub merger_fc1: VisionLinear,
    pub merger_fc2: VisionLinear,
}

/// One image/video grid `[temporal, height, width]` in patch units.
pub type VisionGrid = [usize; 3];

/// Runs the Qwen3.5/Qwen3.8 vision tower on packed patch values.
///
/// `pixel_values` is `[sum(t*h*w), in_channels * temporal_patch_size *
/// patch_size * patch_size]` in the processor's patch order. The return value
/// is the merged vision feature sequence `[sum(t*h*w / merge^2),
/// out_hidden_size]`.
pub fn vision_forward(
    cfg: &VisionConfig,
    w: &VisionWeights,
    pixel_values: &[f32],
    grids: &[VisionGrid],
) -> Result<Vec<f32>, Error> {
    cfg.validate()?;
    if grids.is_empty() {
        return Err(Error::InvalidArgument(
            "vision grids must not be empty".into(),
        ));
    }
    let patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim();
    let merge_unit = cfg.spatial_merge_size * cfg.spatial_merge_size;
    let mut total_patches = 0usize;
    let mut total_merged = 0usize;
    let mut segments = Vec::new();
    for (image, g) in grids.iter().enumerate() {
        let [t, h, wd] = *g;
        if t == 0 || h == 0 || wd == 0 {
            return Err(Error::InvalidArgument(format!(
                "vision grid {image} has a zero dimension: {g:?}"
            )));
        }
        if !h.is_multiple_of(cfg.spatial_merge_size) || !wd.is_multiple_of(cfg.spatial_merge_size) {
            return Err(Error::InvalidArgument(format!(
                "vision grid {image} must be divisible by spatial_merge_size {}: {g:?}",
                cfg.spatial_merge_size
            )));
        }
        let frame = h * wd;
        for ti in 0..t {
            segments.push((total_patches + ti * frame, frame));
        }
        total_patches = total_patches
            .checked_add(t * frame)
            .ok_or_else(|| Error::InvalidArgument("vision patch count overflow".into()))?;
        total_merged = total_merged
            .checked_add(t * (h / cfg.spatial_merge_size) * (wd / cfg.spatial_merge_size))
            .ok_or_else(|| Error::InvalidArgument("vision merged token count overflow".into()))?;
    }
    let want_pixels = total_patches
        .checked_mul(patch_dim)
        .ok_or_else(|| Error::InvalidArgument("vision pixel buffer size overflow".into()))?;
    if pixel_values.len() != want_pixels {
        return Err(Error::InvalidArgument(format!(
            "vision pixel_values has {} elements, expected {want_pixels}",
            pixel_values.len()
        )));
    }
    if w.patch_embed_weight.len() != hidden * patch_dim {
        return Err(Error::Model(format!(
            "vision patch weight has {} elements, expected {}",
            w.patch_embed_weight.len(),
            hidden * patch_dim
        )));
    }

    // Patch embedding (Conv3d with kernel == stride), then learned positional
    // interpolation. The processor packs each patch as [C,T,P,P].
    let patch_linear = VisionLinear {
        weight: w.patch_embed_weight.clone(),
        bias: w.patch_embed_bias.clone(),
    };
    let mut x = linear_forward(pixel_values, total_patches, patch_dim, &patch_linear)?;
    let mut pos = position_embeddings(cfg, &w.pos_embed_weight, grids, total_patches)?;
    for (v, p) in x.iter_mut().zip(pos.drain(..)) {
        *v += p;
    }
    let (cos, sin) = vision_rope(cfg, grids, total_patches)?;

    for layer in &w.layers {
        let norm1 = layer_norm(
            &x,
            total_patches,
            hidden,
            &layer.norm1_weight,
            &layer.norm1_bias,
            1e-6,
        )?;
        let mut qkv = linear_forward(&norm1, total_patches, hidden, &layer.qkv)?;
        vision_rope_apply_inplace(&mut qkv, total_patches, cfg.num_heads, head_dim, &cos, &sin)?;
        let attn = vision_attention(cfg, &qkv, &segments, total_patches)?;
        let proj = linear_forward(&attn, total_patches, hidden, &layer.attn_proj)?;
        for (v, p) in x.iter_mut().zip(proj) {
            *v += p;
        }
        let norm2 = layer_norm(
            &x,
            total_patches,
            hidden,
            &layer.norm2_weight,
            &layer.norm2_bias,
            1e-6,
        )?;
        let fc1 = linear_forward(&norm2, total_patches, hidden, &layer.mlp_fc1)?;
        let mut gelu = fc1;
        gelu_inplace(&mut gelu);
        let fc2 = linear_forward(&gelu, total_patches, cfg.intermediate_size, &layer.mlp_fc2)?;
        for (v, p) in x.iter_mut().zip(fc2) {
            *v += p;
        }
    }

    let norm = layer_norm(
        &x,
        total_patches,
        hidden,
        &w.merger_norm_weight,
        &w.merger_norm_bias,
        1e-6,
    )?;
    let merge_dim = hidden * merge_unit;
    if !total_patches.is_multiple_of(merge_unit) {
        return Err(Error::InvalidArgument(format!(
            "vision patch count {total_patches} is not divisible by merge unit {merge_unit}"
        )));
    }
    let mut merged = vec![0.0f32; total_merged * merge_dim];
    for token in 0..total_merged {
        let src = &norm[token * merge_dim..(token + 1) * merge_dim];
        merged[token * merge_dim..(token + 1) * merge_dim].copy_from_slice(src);
    }
    let fc1 = linear_forward(&merged, total_merged, merge_dim, &w.merger_fc1)?;
    let mut gelu = fc1;
    gelu_inplace(&mut gelu);
    let out = linear_forward(&gelu, total_merged, merge_dim, &w.merger_fc2)?;
    Ok(out)
}

fn linear_forward(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    l: &VisionLinear,
) -> Result<Vec<f32>, Error> {
    let out_dim = l.bias.len();
    if l.weight.len() != out_dim * in_dim {
        return Err(Error::Model(format!(
            "vision linear weight has {} elements, expected {}",
            l.weight.len(),
            out_dim * in_dim
        )));
    }
    if x.len() != rows * in_dim {
        return Err(Error::InvalidArgument(format!(
            "vision linear input has {} elements, expected {}",
            x.len(),
            rows * in_dim
        )));
    }
    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut s = l.bias[o];
            for i in 0..in_dim {
                s += l.weight[o * in_dim + i] * x[r * in_dim + i];
            }
            out[r * out_dim + o] = s;
        }
    }
    Ok(out)
}

fn layer_norm(
    x: &[f32],
    rows: usize,
    dim: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<Vec<f32>, Error> {
    if x.len() != rows * dim || weight.len() != dim || bias.len() != dim {
        return Err(Error::Model("vision LayerNorm shape mismatch".into()));
    }
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..dim {
            out[r * dim + i] = (row[i] - mean) * inv * weight[i] + bias[i];
        }
    }
    Ok(out)
}

fn gelu_inplace(x: &mut [f32]) {
    for v in x {
        *v = 0.5 * *v * (1.0 + erf(*v / std::f32::consts::SQRT_2));
    }
}

fn erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26, max abs error < 1.5e-7.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp());
    sign * y
}

fn vision_rope_apply_inplace(
    qkv: &mut [f32],
    tokens: usize,
    heads: usize,
    hd: usize,
    cos: &[f32],
    sin: &[f32],
) -> Result<(), Error> {
    if qkv.len() != tokens * 3 * heads * hd || cos.len() != tokens * hd || sin.len() != tokens * hd
    {
        return Err(Error::Model("vision RoPE shape mismatch".into()));
    }
    let half = hd / 2;
    for t in 0..tokens {
        for head in 0..heads {
            for part in 0..2 {
                let base = t * 3 * heads * hd + part * heads * hd + head * hd;
                for i in 0..half {
                    let a = qkv[base + i];
                    let b = qkv[base + half + i];
                    let c = cos[t * hd + i];
                    let s = sin[t * hd + i];
                    qkv[base + i] = a * c - b * s;
                    qkv[base + half + i] = b * c + a * s;
                }
            }
        }
    }
    Ok(())
}
fn vision_attention(
    cfg: &VisionConfig,
    qkv: &[f32],
    segments: &[(usize, usize)],
    tokens: usize,
) -> Result<Vec<f32>, Error> {
    let h = cfg.num_heads;
    let hd = cfg.head_dim();
    let hidden = cfg.hidden_size;
    if qkv.len() != tokens * 3 * hidden {
        return Err(Error::Model("vision attention shape mismatch".into()));
    }
    let mut out = vec![0.0f32; tokens * hidden];
    let scale = 1.0 / (hd as f32).sqrt();
    for &(start, len) in segments {
        for head in 0..h {
            for t in 0..len {
                let token = start + t;
                let q_base = token * 3 * hidden + head * hd;
                let mut scores = vec![0.0f32; len];
                let mut maxv = f32::NEG_INFINITY;
                for (s, score) in scores.iter_mut().enumerate() {
                    let other = start + s;
                    let kk = other * 3 * hidden + hidden + head * hd;
                    let mut dot = 0.0;
                    for d in 0..hd {
                        dot += qkv[q_base + d] * qkv[kk + d];
                    }
                    let sscore = dot * scale;
                    // Vision attention is bidirectional within each temporal
                    // frame (cu_seqlens repeat H*W across T).
                    *score = sscore;
                    maxv = maxv.max(sscore);
                }
                let mut sum = 0.0;
                for score in &mut scores {
                    *score = (*score - maxv).exp();
                    sum += *score;
                }
                for d in 0..hd {
                    let mut acc = 0.0;
                    for (s, score) in scores.iter().enumerate() {
                        let other = start + s;
                        let vv = other * 3 * hidden + 2 * hidden + head * hd + d;
                        acc += *score * qkv[vv];
                    }
                    out[token * hidden + head * hd + d] = acc / sum;
                }
                // RoPE is applied to q/k in-place logically here; the caller
                // already rotated qkv in place before this function.
            }
        }
    }
    Ok(out)
}

fn vision_rope(
    cfg: &VisionConfig,
    grids: &[VisionGrid],
    tokens: usize,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let hd = cfg.head_dim();
    let half = hd / 2;
    let freq_dim = half / 2;
    let theta = 10000.0f32;
    let mut freqs = vec![0.0f32; freq_dim];
    for (i, f) in freqs.iter_mut().enumerate() {
        let j = (2 * i) as f32;
        *f = 1.0 / theta.powf(j / half as f32);
    }
    let positions = spatial_position_ids(cfg, grids)?;
    if positions.len() != tokens * 2 {
        return Err(Error::Model(
            "vision spatial position count mismatch".into(),
        ));
    }
    let mut cos = vec![0.0f32; tokens * hd];
    let mut sin = vec![0.0f32; tokens * hd];
    for t in 0..tokens {
        let hpos = positions[t * 2] as f32;
        let wpos = positions[t * 2 + 1] as f32;
        for i in 0..freq_dim {
            let r = hpos * freqs[i];
            let r2 = wpos * freqs[i];
            cos[t * hd + i] = r.cos();
            cos[t * hd + freq_dim + i] = r2.cos();
            sin[t * hd + i] = r.sin();
            sin[t * hd + freq_dim + i] = r2.sin();
            cos[t * hd + half + i] = r.cos();
            cos[t * hd + half + freq_dim + i] = r2.cos();
            sin[t * hd + half + i] = r.sin();
            sin[t * hd + half + freq_dim + i] = r2.sin();
        }
    }
    Ok((cos, sin))
}

fn spatial_position_ids(cfg: &VisionConfig, grids: &[VisionGrid]) -> Result<Vec<usize>, Error> {
    let merge = cfg.spatial_merge_size;
    let mut out = Vec::new();
    for (image, g) in grids.iter().enumerate() {
        let [t, h, w] = *g;
        if !h.is_multiple_of(merge) || !w.is_multiple_of(merge) {
            return Err(Error::InvalidArgument(format!(
                "vision grid {image} must be divisible by {merge}: {g:?}"
            )));
        }
        let blocks_h = h / merge;
        let blocks_w = w / merge;
        let mut hp = Vec::with_capacity(h * w);
        let mut wp = Vec::with_capacity(h * w);
        for br in 0..blocks_h {
            for bc in 0..blocks_w {
                for ir in 0..merge {
                    for ic in 0..merge {
                        hp.push(br * merge + ir);
                        wp.push(bc * merge + ic);
                    }
                }
            }
        }
        for _ in 0..t {
            for i in 0..h * w {
                out.push(hp[i]);
                out.push(wp[i]);
            }
        }
    }
    Ok(out)
}

fn position_embeddings(
    cfg: &VisionConfig,
    table: &[f32],
    grids: &[VisionGrid],
    tokens: usize,
) -> Result<Vec<f32>, Error> {
    let hidden = cfg.hidden_size;
    if table.len() != cfg.num_position_embeddings * hidden {
        return Err(Error::Model("vision position table shape mismatch".into()));
    }
    let side = (cfg.num_position_embeddings as f64).sqrt().round() as usize;
    let mut out = vec![0.0f32; tokens * hidden];
    let mut dst = 0usize;
    for (image, g) in grids.iter().enumerate() {
        let [t, h, w] = *g;
        let merge = cfg.spatial_merge_size;
        if !h.is_multiple_of(merge) || !w.is_multiple_of(merge) {
            return Err(Error::InvalidArgument(format!(
                "vision grid {image} must be divisible by {merge}: {g:?}"
            )));
        }
        let frame = h * w;
        for _ in 0..t {
            for within in 0..frame {
                let blocks_w = w / merge;
                let in_col = within % merge;
                let in_row = (within / merge) % merge;
                let block_col = (within / (merge * merge)) % blocks_w;
                let block_row = within / (merge * merge * blocks_w);
                let row = block_row * merge + in_row;
                let col = block_col * merge + in_col;
                let (h_taps, h_weights) = axis_taps(row, h, side);
                let (w_taps, w_weights) = axis_taps(col, w, side);
                for ih in 0..2 {
                    for iw in 0..2 {
                        let idx = h_taps[ih] * side + w_taps[iw];
                        let weight = h_weights[ih] * w_weights[iw];
                        let src = &table[idx * hidden..(idx + 1) * hidden];
                        let dst_row = &mut out[dst * hidden..(dst + 1) * hidden];
                        for d in 0..hidden {
                            dst_row[d] += src[d] * weight;
                        }
                    }
                }
                dst += 1;
            }
        }
    }
    if dst != tokens {
        return Err(Error::Model(format!(
            "vision position embedding produced {dst} tokens, expected {tokens}"
        )));
    }
    Ok(out)
}

fn axis_taps(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let src = index as f32 * (side - 1) as f32 / (size.saturating_sub(1)).max(1) as f32;
    let floor = src.floor();
    let mut taps = [0usize; 2];
    let mut weights = [0.0f32; 2];
    for (i, tap) in taps.iter_mut().enumerate() {
        let raw = floor as isize + i as isize;
        *tap = raw.clamp(0, side as isize - 1) as usize;
        let dist = (src - floor - i as f32).abs();
        weights[i] = (1.0 - dist).max(0.0);
    }
    (taps, weights)
}
