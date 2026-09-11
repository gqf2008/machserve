//! Model-side assembly of a multimodal prompt.
//!
//! Turns an already-expanded token sequence (one `<|image_pad|>` token per
//! merged vision token) plus per-image merged features into the row-level
//! overrides consumed by [`crate::batched::BatchedModel`] and the M-RoPE
//! tables consumed by the GPU rope kernel.
//!
//! Layout matches HF `Qwen3VLProcessor.replace_image_token` +
//! `Qwen3_5Model.get_rope_index`: image rows are the contiguous pad-token
//! runs, `mm_token_type_ids` is 1 on those rows, and text positions after the
//! image continue at `index + delta`.

use crate::mrope::{MropePositions, qwen3_5_mrope_positions};
use crate::vision::{VisionConfig, VisionGrid};
use crate::{Config, Error};

/// Merged vision features for one image.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionImage {
    /// Row-major `[merged_tokens, d_model]` features from the vision tower.
    pub features: Vec<f32>,
    /// Vision grid `[temporal, height, width]` in patch units.
    pub grid: VisionGrid,
}

impl VisionImage {
    /// Merged token count for this grid: `t * h * w / merge_size^2`.
    pub fn merged_tokens(&self, merge_size: usize) -> Result<usize, Error> {
        let merge_unit = merge_size
            .checked_mul(merge_size)
            .ok_or_else(|| Error::InvalidArgument("multimodal merge size overflow".into()))?;
        if merge_unit == 0 {
            return Err(Error::InvalidArgument(
                "multimodal merge size must be positive".into(),
            ));
        }
        let total = self.grid[0]
            .checked_mul(self.grid[1])
            .and_then(|v| v.checked_mul(self.grid[2]))
            .ok_or_else(|| Error::InvalidArgument("vision grid size overflow".into()))?;
        if total == 0 || !total.is_multiple_of(merge_unit) {
            return Err(Error::InvalidArgument(format!(
                "vision grid {:?} is not divisible by merge unit {merge_unit}",
                self.grid
            )));
        }
        Ok(total / merge_unit)
    }
}

/// Row-level multimodal overrides for one prompt sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct MultimodalPrompt {
    /// Prompt rows whose embedding is replaced by the vision features.
    pub image_rows: Vec<usize>,
    /// Concatenated features for `image_rows`, `d_model` per row.
    pub image_features: Vec<f32>,
    /// `mm_token_type_ids`: 1 on image rows, 0 elsewhere.
    pub mm_token_type_ids: Vec<u8>,
    /// Full-prompt M-RoPE positions.
    pub positions: MropePositions,
    /// Full-prompt token-major cos/sin tables `[prompt_len, rotary_dim]`.
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    /// M-RoPE section this prompt was built with.
    pub mrope_section: [usize; 3],
    d_model: usize,
    rotary_dim: usize,
}

impl MultimodalPrompt {
    /// Assemble the overrides for `tokens` (pad tokens already expanded).
    pub fn build(
        tokens: &[u32],
        images: &[VisionImage],
        image_token_id: u32,
        cfg: &Config,
        vision: &VisionConfig,
    ) -> Result<Self, Error> {
        if tokens.is_empty() {
            return Err(Error::InvalidArgument("multimodal prompt is empty".into()));
        }
        if images.is_empty() {
            return Err(Error::InvalidArgument(
                "multimodal prompt needs at least one image".into(),
            ));
        }
        if cfg.d_model == 0 {
            return Err(Error::InvalidArgument("d_model must be positive".into()));
        }
        let merge = vision.spatial_merge_size;
        let mut grids = Vec::with_capacity(images.len());
        let mut merged = Vec::with_capacity(images.len());
        let mut total_rows = 0usize;
        for (i, image) in images.iter().enumerate() {
            let n = image.merged_tokens(merge)?;
            let expect = n
                .checked_mul(cfg.d_model)
                .ok_or_else(|| Error::InvalidArgument("vision feature buffer overflow".into()))?;
            if image.features.len() != expect {
                return Err(Error::InvalidArgument(format!(
                    "image {i} has {} feature values, expected {expect} ({n} merged tokens x {} d_model)",
                    image.features.len(),
                    cfg.d_model
                )));
            }
            grids.push([
                u32::try_from(image.grid[0])
                    .map_err(|_| Error::InvalidArgument("image grid t exceeds u32".into()))?,
                u32::try_from(image.grid[1])
                    .map_err(|_| Error::InvalidArgument("image grid h exceeds u32".into()))?,
                u32::try_from(image.grid[2])
                    .map_err(|_| Error::InvalidArgument("image grid w exceeds u32".into()))?,
            ]);
            merged.push(n);
            total_rows = total_rows
                .checked_add(n)
                .ok_or_else(|| Error::InvalidArgument("multimodal row count overflow".into()))?;
        }
        let image_features_len = total_rows
            .checked_mul(cfg.d_model)
            .ok_or_else(|| Error::InvalidArgument("multimodal feature buffer overflow".into()))?;
        let mut mm_token_type_ids = vec![0u8; tokens.len()];
        let mut image_rows = Vec::with_capacity(total_rows);
        let mut image_features = Vec::with_capacity(image_features_len);
        let mut next_image = 0usize;
        let mut index = 0usize;
        while index < tokens.len() {
            if tokens[index] != image_token_id {
                index += 1;
                continue;
            }
            if next_image >= images.len() {
                return Err(Error::InvalidArgument(
                    "prompt has more image pad groups than images".into(),
                ));
            }
            let start = index;
            while index < tokens.len() && tokens[index] == image_token_id {
                index += 1;
            }
            let run = index - start;
            if run != merged[next_image] {
                return Err(Error::InvalidArgument(format!(
                    "image {next_image} pad run has {run} tokens, expected {}",
                    merged[next_image]
                )));
            }
            image_rows.extend(start..index);
            mm_token_type_ids[start..index].fill(1);
            image_features.extend_from_slice(&images[next_image].features);
            next_image += 1;
        }
        if next_image != images.len() {
            return Err(Error::InvalidArgument(format!(
                "{} image(s) were not matched by any pad run",
                images.len() - next_image
            )));
        }
        let positions = qwen3_5_mrope_positions(tokens, &mm_token_type_ids, &grids, &[], vision)?;
        let (cos, sin) = positions.cos_sin(cfg, vision.mrope_section)?;
        Ok(Self {
            image_rows,
            image_features,
            mm_token_type_ids,
            positions,
            cos,
            sin,
            mrope_section: vision.mrope_section,
            d_model: cfg.d_model,
            rotary_dim: cfg.attn_rotary_dim(),
        })
    }

    /// Prompt row whose embedding is overridden, in ascending order.
    #[must_use]
    pub fn image_rows(&self) -> &[usize] {
        &self.image_rows
    }

    /// Feature stride of [`Self::image_features`].
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Rotary dimension of one cos/sin row.
    #[must_use]
    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    /// Number of prompt rows covered by the position tables.
    #[must_use]
    pub fn prompt_len(&self) -> usize {
        self.positions.len()
    }

    /// Explicit feature row for prompt row `row`, when overridden.
    #[must_use]
    pub fn feature_row(&self, row: usize) -> Option<&[f32]> {
        let i = self.image_rows.binary_search(&row).ok()?;
        Some(&self.image_features[i * self.d_model..(i + 1) * self.d_model])
    }

    /// Scalar position for the generated token at absolute `index`.
    pub fn generated_position(&self, index: usize) -> Result<[i32; 3], Error> {
        let index = i32::try_from(index)
            .map_err(|_| Error::InvalidArgument("generated index exceeds i32".into()))?;
        let p = index
            .checked_add(self.positions.delta)
            .ok_or_else(|| Error::InvalidArgument("generated M-RoPE position overflow".into()))?;
        Ok([p, p, p])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (Config, VisionConfig) {
        (Config::tiny(), VisionConfig::default())
    }

    fn features(rows: usize, d: usize) -> Vec<f32> {
        (0..rows * d).map(|i| i as f32).collect()
    }

    #[test]
    fn text_only_prompt_is_rejected() {
        let (cfg, vision) = tiny();
        let err = MultimodalPrompt::build(&[1, 2, 3], &[], 20, &cfg, &vision)
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least one image"), "{err}");
    }

    #[test]
    fn single_image_maps_pad_rows_and_positions() {
        let (cfg, vision) = tiny();
        let tokens = [10, 11, 20, 20, 20, 20, 12];
        let d = cfg.d_model;
        let image = VisionImage {
            features: features(4, d),
            grid: [1, 4, 4],
        };
        let out = MultimodalPrompt::build(&tokens, &[image], 20, &cfg, &vision).unwrap();
        assert_eq!(out.image_rows(), [2, 3, 4, 5]);
        assert_eq!(out.mm_token_type_ids, [0, 0, 1, 1, 1, 1, 0]);
        assert_eq!(out.image_features.len(), 4 * d);
        assert_eq!(out.feature_row(2), Some(&out.image_features[0..d]));
        assert_eq!(out.feature_row(5), Some(&out.image_features[3 * d..4 * d]));
        assert!(out.feature_row(1).is_none());
        assert_eq!(
            out.positions.pos,
            vec![
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 3],
                [2, 3, 2],
                [2, 3, 3],
                [4, 4, 4]
            ]
        );
        assert_eq!(out.positions.delta, -2);
        assert_eq!(out.prompt_len(), 7);
        assert_eq!(out.cos.len(), 7 * out.rotary_dim());
        assert_eq!(out.sin.len(), out.cos.len());
        assert_eq!(out.generated_position(7).unwrap(), [5, 5, 5]);
    }

    #[test]
    fn two_images_consume_pad_runs_in_order() {
        let (cfg, vision) = tiny();
        let tokens = [1, 20, 20, 20, 20, 2, 20];
        let d = cfg.d_model;
        let a = VisionImage {
            features: features(4, d),
            grid: [1, 4, 4],
        };
        let b = VisionImage {
            features: vec![9.0; d],
            grid: [1, 2, 2],
        };
        let out = MultimodalPrompt::build(&tokens, &[a, b], 20, &cfg, &vision).unwrap();
        assert_eq!(out.image_rows(), [1, 2, 3, 4, 6]);
        assert_eq!(out.image_features.len(), 5 * d);
        assert_eq!(&out.image_features[4 * d..], vec![9.0; d]);
        assert_eq!(out.mm_token_type_ids, [0, 1, 1, 1, 1, 0, 1]);
    }

    #[test]
    fn rejects_inconsistent_image_inputs() {
        let (cfg, vision) = tiny();
        let d = cfg.d_model;
        let short = VisionImage {
            features: features(3, d),
            grid: [1, 4, 4],
        };
        assert!(MultimodalPrompt::build(&[20, 20, 20, 20], &[short], 20, &cfg, &vision).is_err());

        let full = VisionImage {
            features: features(4, d),
            grid: [1, 4, 4],
        };
        let err = MultimodalPrompt::build(
            &[20, 20, 20],
            std::slice::from_ref(&full),
            20,
            &cfg,
            &vision,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("pad run"), "{err}");

        assert!(
            MultimodalPrompt::build(
                &[20, 20, 20, 20],
                &[full.clone(), full.clone()],
                20,
                &cfg,
                &vision,
            )
            .is_err()
        );
        assert!(
            MultimodalPrompt::build(&[20, 20, 20, 20, 20], &[full], 20, &cfg, &vision).is_err()
        );

        let indivisible = VisionImage {
            features: Vec::new(),
            grid: [1, 3, 3],
        };
        assert!(indivisible.merged_tokens(2).is_err());
    }
}

/// One sequence contribution to a batched step.
#[derive(Debug, Clone, Copy)]
pub struct StepSeq<'a> {
    /// Absolute prompt row for the first prefill row, or the generated-token
    /// index for a decode row.
    pub offset: usize,
    /// Rows this sequence contributes to the step.
    pub count: usize,
    /// True when the rows are prompt tokens, false for a generated token.
    pub prefill: bool,
    /// Multimodal overrides for this sequence, when it carries an image.
    pub prompt: Option<&'a MultimodalPrompt>,
}

/// Per-batch row overrides for one forward step.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StepOverrides {
    /// Batch rows covered by this step.
    pub rows: usize,
    pub d_model: usize,
    pub rotary_dim: usize,
    /// `[rows, d_model]` features; only masked rows are applied.
    pub row_embeddings: Vec<f32>,
    /// Per-row override mask (`1` = replace the token embedding).
    pub row_mask: Vec<i32>,
    /// `[rows, rotary_dim]` M-RoPE tables for the whole batch.
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    /// True when the batch needs M-RoPE tables (any sequence has an image).
    pub mrope: bool,
    /// True when at least one row needs an explicit embedding.
    pub row_embed: bool,
}

/// Assemble the per-row overrides for one engine step.
///
/// Any sequence carrying an image forces M-RoPE tables for the whole batch:
/// image rows use their stored `(t,h,w)` position, decode rows continue at
/// `index + delta`, and text-only sequences use scalar `(p,p,p)` positions.
pub fn step_overrides(
    seqs: &[StepSeq<'_>],
    cfg: &Config,
    mrope_section: [usize; 3],
) -> Result<StepOverrides, Error> {
    let rows = seqs.iter().try_fold(0usize, |acc, s| {
        acc.checked_add(s.count)
            .ok_or_else(|| Error::InvalidArgument("step row count overflow".into()))
    })?;
    let d_model = cfg.d_model;
    let rotary_dim = cfg.attn_rotary_dim();
    let mrope = seqs.iter().any(|s| s.prompt.is_some());
    if !mrope {
        return Ok(StepOverrides {
            rows,
            d_model,
            rotary_dim,
            ..StepOverrides::default()
        });
    }
    if d_model == 0 || rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(Error::InvalidArgument(
            "multimodal step needs positive d_model and even rotary dim".into(),
        ));
    }
    let mut row_embeddings = vec![
        0.0f32;
        rows.checked_mul(d_model).ok_or_else(|| {
            Error::InvalidArgument("step embedding buffer overflow".into())
        })?
    ];
    let mut row_mask = vec![0i32; rows];
    let mut positions = Vec::with_capacity(rows);
    let mut cursor = 0usize;
    let mut row_embed = false;
    for seq in seqs {
        if let Some(prompt) = seq.prompt
            && prompt.mrope_section != mrope_section
        {
            return Err(Error::InvalidArgument(format!(
                "prompt M-RoPE section {:?} does not match engine section {mrope_section:?}",
                prompt.mrope_section
            )));
        }
        for k in 0..seq.count {
            let feature = match seq.prompt {
                Some(prompt) if seq.prefill => {
                    let row = seq.offset.checked_add(k).ok_or_else(|| {
                        Error::InvalidArgument("multimodal prompt row overflow".into())
                    })?;
                    if row >= prompt.prompt_len() {
                        return Err(Error::InvalidArgument(format!(
                            "multimodal prompt row {row} exceeds prompt length {}",
                            prompt.prompt_len()
                        )));
                    }
                    positions.push(prompt.positions.pos[row]);
                    prompt.feature_row(row)
                }
                Some(prompt) => {
                    let index = seq.offset.checked_add(k).ok_or_else(|| {
                        Error::InvalidArgument("multimodal decode offset overflow".into())
                    })?;
                    positions.push(prompt.generated_position(index)?);
                    None
                }
                None => {
                    let index = seq
                        .offset
                        .checked_add(k)
                        .ok_or_else(|| Error::InvalidArgument("text position overflow".into()))?;
                    let index = i32::try_from(index)
                        .map_err(|_| Error::InvalidArgument("text position exceeds i32".into()))?;
                    positions.push([index, index, index]);
                    None
                }
            };
            if let Some(feature) = feature {
                let dst = (cursor + k)
                    .checked_mul(d_model)
                    .ok_or_else(|| Error::InvalidArgument("row offset overflow".into()))?;
                row_embeddings[dst..dst + d_model].copy_from_slice(feature);
                row_mask[cursor + k] = 1;
                row_embed = true;
            }
        }
        cursor += seq.count;
    }
    let (cos, sin) = MropePositions {
        pos: positions,
        delta: 0,
    }
    .cos_sin(cfg, mrope_section)?;
    Ok(StepOverrides {
        rows,
        d_model,
        rotary_dim,
        row_embeddings,
        row_mask,
        cos,
        sin,
        mrope: true,
        row_embed,
    })
}

#[cfg(test)]
mod step_tests {
    use super::*;

    fn tiny() -> (Config, VisionConfig) {
        (Config::tiny(), VisionConfig::default())
    }

    fn features(rows: usize, d: usize) -> Vec<f32> {
        (0..rows * d).map(|i| i as f32).collect()
    }

    fn single_image_prompt() -> (Config, VisionConfig, MultimodalPrompt) {
        let (cfg, vision) = tiny();
        let tokens = [10, 11, 20, 20, 20, 20, 12];
        let image = VisionImage {
            features: features(4, cfg.d_model),
            grid: [1, 4, 4],
        };
        let prompt = MultimodalPrompt::build(&tokens, &[image], 20, &cfg, &vision).unwrap();
        (cfg, vision, prompt)
    }

    #[test]
    fn step_overrides_text_only_is_empty() {
        let (cfg, _vision) = tiny();
        let seqs = [StepSeq {
            offset: 0,
            count: 3,
            prefill: true,
            prompt: None,
        }];
        let out = step_overrides(&seqs, &cfg, [1, 1, 1]).unwrap();
        assert!(!out.mrope);
        assert!(!out.row_embed);
        assert_eq!(out.rows, 3);
        assert!(out.row_embeddings.is_empty());
        assert!(out.cos.is_empty());
    }

    #[test]
    fn step_overrides_mix_image_prefill_and_text_decode() {
        let (cfg, vision, prompt) = single_image_prompt();
        let seqs = [
            StepSeq {
                offset: 0,
                count: 7,
                prefill: true,
                prompt: Some(&prompt),
            },
            StepSeq {
                offset: 20,
                count: 1,
                prefill: false,
                prompt: None,
            },
        ];
        let out = step_overrides(&seqs, &cfg, vision.mrope_section).unwrap();
        assert!(out.mrope);
        assert!(out.row_embed);
        assert_eq!(out.rows, 8);
        assert_eq!(out.row_mask, [0, 0, 1, 1, 1, 1, 0, 0]);
        let d = cfg.d_model;
        for row in 2..6 {
            assert_eq!(
                &out.row_embeddings[row * d..(row + 1) * d],
                &prompt.image_features[(row - 2) * d..(row - 1) * d]
            );
        }
        let dim = out.rotary_dim;
        for row in 0..7 {
            assert_eq!(
                &out.cos[row * dim..(row + 1) * dim],
                &prompt.cos[row * dim..(row + 1) * dim]
            );
        }
        let (want_cos, want_sin) = MropePositions {
            pos: vec![[20, 20, 20]],
            delta: 0,
        }
        .cos_sin(&cfg, vision.mrope_section)
        .unwrap();
        assert_eq!(&out.cos[7 * dim..8 * dim], &want_cos);
        assert_eq!(&out.sin[7 * dim..8 * dim], &want_sin);
    }

    #[test]
    fn step_overrides_image_decode_uses_delta() {
        let (cfg, vision, prompt) = single_image_prompt();
        let seqs = [StepSeq {
            offset: 7,
            count: 1,
            prefill: false,
            prompt: Some(&prompt),
        }];
        let out = step_overrides(&seqs, &cfg, vision.mrope_section).unwrap();
        assert!(out.mrope);
        assert!(!out.row_embed);
        assert_eq!(out.row_mask, [0]);
        let (want_cos, want_sin) = MropePositions {
            pos: vec![[5, 5, 5]],
            delta: 0,
        }
        .cos_sin(&cfg, vision.mrope_section)
        .unwrap();
        assert_eq!(out.cos, want_cos);
        assert_eq!(out.sin, want_sin);
    }

    #[test]
    fn step_overrides_prefill_chunk_starts_inside_image() {
        let (cfg, vision, prompt) = single_image_prompt();
        let seqs = [StepSeq {
            offset: 3,
            count: 2,
            prefill: true,
            prompt: Some(&prompt),
        }];
        let out = step_overrides(&seqs, &cfg, vision.mrope_section).unwrap();
        assert_eq!(out.row_mask, [1, 1]);
        let d = cfg.d_model;
        assert_eq!(&out.row_embeddings[0..d], &prompt.image_features[d..2 * d]);
        assert_eq!(
            &out.row_embeddings[d..2 * d],
            &prompt.image_features[2 * d..3 * d]
        );
        let dim = out.rotary_dim;
        assert_eq!(
            &out.cos[0..dim],
            &prompt.cos[3 * dim..4 * dim],
            "chunk start must use the absolute prompt row"
        );
        assert_eq!(&out.cos[dim..2 * dim], &prompt.cos[4 * dim..5 * dim]);
    }

    #[test]
    fn step_overrides_prefill_chunk_after_image_has_no_override() {
        let (cfg, vision, prompt) = single_image_prompt();
        let seqs = [StepSeq {
            offset: 6,
            count: 1,
            prefill: true,
            prompt: Some(&prompt),
        }];
        let out = step_overrides(&seqs, &cfg, vision.mrope_section).unwrap();
        assert_eq!(out.row_mask, [0]);
        assert!(!out.row_embed);
        let dim = out.rotary_dim;
        assert_eq!(&out.cos[0..dim], &prompt.cos[6 * dim..7 * dim]);
    }

    #[test]
    fn step_overrides_rejects_section_mismatch() {
        let (cfg, _vision, prompt) = single_image_prompt();
        let seqs = [StepSeq {
            offset: 0,
            count: 1,
            prefill: true,
            prompt: Some(&prompt),
        }];
        let err = step_overrides(&seqs, &cfg, [1, 1, 1])
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not match"), "{err}");
    }
}
