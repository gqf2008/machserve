//! Qwen3.5/Qwen3.8 text-side M-RoPE position generation.
//!
//! Mirrors `Qwen3_5Model.get_rope_index` from transformers 5.16.1: text
//! groups advance a scalar position, image/video groups expand their
//! `(t,h,w)` grid into 3-axis positions, and the returned delta is used for
//! incremental text decoding after a multimodal prompt.

use crate::Error;
use crate::vision::VisionConfig;

/// Per-token M-RoPE positions `[temporal, height, width]` and the text
/// continuation delta (`max_pos + 1 - prompt_len`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MropePositions {
    pub pos: Vec<[i32; 3]>,
    pub delta: i32,
}

impl MropePositions {
    #[must_use]
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// Compute the per-token interleaved M-RoPE cos/sin tables for `cfg`.
    ///
    /// The returned vectors are token-major `[len, rotary_dim]`; the future
    /// GPU rope kernel consumes them directly, while the CPU reference can use
    /// the same tables to avoid a second interpretation of the interleave.
    pub fn cos_sin(
        &self,
        cfg: &crate::Config,
        mrope_section: [usize; 3],
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let dim = cfg.attn_rotary_dim();
        if dim == 0 || !dim.is_multiple_of(2) {
            return Err(Error::InvalidArgument(format!(
                "M-RoPE rotary dim {dim} must be positive and even"
            )));
        }
        let half = dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            let j = (2 * i) as f32;
            inv_freq.push(1.0 / cfg.rope_theta.powf(j / dim as f32));
        }
        let mut cos = vec![0.0f32; self.pos.len() * dim];
        let mut sin = vec![0.0f32; self.pos.len() * dim];
        let mut freqs = [vec![0.0f32; half], vec![0.0f32; half], vec![0.0f32; half]];
        for (token, p) in self.pos.iter().enumerate() {
            for axis in 0..3 {
                for i in 0..half {
                    freqs[axis][i] = p[axis] as f32 * inv_freq[i];
                }
            }
            let mut interleaved = freqs[0].clone();
            for axis in 1..3 {
                let length = mrope_section[axis]
                    .checked_mul(3)
                    .ok_or_else(|| Error::InvalidArgument("mrope section overflow".into()))?;
                let mut j = axis;
                while j < length && j < half {
                    interleaved[j] = freqs[axis][j];
                    j += 3;
                }
            }
            for i in 0..half {
                let c = interleaved[i].cos();
                let s = interleaved[i].sin();
                cos[token * dim + i] = c;
                cos[token * dim + half + i] = c;
                sin[token * dim + i] = s;
                sin[token * dim + half + i] = s;
            }
        }
        Ok((cos, sin))
    }
}

/// Generate Qwen3.5/Qwen3.8 M-RoPE positions.
///
/// `mm_token_type_ids` uses 0=text, 1=image, 2=video. `image_grid_thw` entries
/// are consumed in order by image groups; `video_grid_thw` entries are split
/// into one `t=1` grid per video frame, matching the timestamp-separated
/// processor layout.
pub fn qwen3_5_mrope_positions(
    input_ids: &[u32],
    mm_token_type_ids: &[u8],
    image_grid_thw: &[[u32; 3]],
    video_grid_thw: &[[u32; 3]],
    vision: &VisionConfig,
) -> Result<MropePositions, Error> {
    if input_ids.len() != mm_token_type_ids.len() {
        return Err(Error::InvalidArgument(format!(
            "input_ids length {} != mm_token_type_ids length {}",
            input_ids.len(),
            mm_token_type_ids.len()
        )));
    }
    if input_ids.is_empty() {
        return Err(Error::InvalidArgument("input_ids must not be empty".into()));
    }
    let merge = vision.spatial_merge_size;
    if merge == 0 {
        return Err(Error::InvalidArgument(
            "vision spatial_merge_size must be positive".into(),
        ));
    }
    let mut video_frames = Vec::new();
    for g in video_grid_thw {
        let [t, h, w] = *g;
        for _ in 0..t {
            video_frames.push([1, h, w]);
        }
    }

    let mut pos = vec![[0i32; 3]; input_ids.len()];
    let mut image_idx = 0usize;
    let mut video_idx = 0usize;
    let mut current_pos = 0i32;
    let mut i = 0usize;
    while i < mm_token_type_ids.len() {
        let ty = mm_token_type_ids[i];
        let start = i;
        while i < mm_token_type_ids.len() && mm_token_type_ids[i] == ty {
            i += 1;
        }
        let len = i - start;
        match ty {
            0 => {
                for (k, slot) in pos[start..i].iter_mut().enumerate() {
                    let p = current_pos
                        .checked_add(i32::try_from(k).map_err(|_| {
                            Error::InvalidArgument("text position offset exceeds i32".into())
                        })?)
                        .ok_or_else(|| Error::InvalidArgument("text position overflow".into()))?;
                    *slot = [p, p, p];
                }
                current_pos = current_pos
                    .checked_add(i32::try_from(len).map_err(|_| {
                        Error::InvalidArgument("text group length exceeds i32".into())
                    })?)
                    .ok_or_else(|| Error::InvalidArgument("text position overflow".into()))?;
            }
            1 => {
                let g = image_grid_thw.get(image_idx).ok_or_else(|| {
                    Error::InvalidArgument("image token group has no image_grid_thw entry".into())
                })?;
                image_idx += 1;
                let expected = vision_group_len(*g, merge)?;
                if expected != len {
                    return Err(Error::InvalidArgument(format!(
                        "image group has {len} tokens, grid {g:?} expands to {expected}"
                    )));
                }
                vision_positions(*g, merge, current_pos, &mut pos[start..i])?;
                current_pos = advance_vision_pos(*g, merge, current_pos)?;
            }
            2 => {
                let g = video_frames.get(video_idx).ok_or_else(|| {
                    Error::InvalidArgument("video token group has no video frame grid".into())
                })?;
                video_idx += 1;
                let expected = vision_group_len(*g, merge)?;
                if expected != len {
                    return Err(Error::InvalidArgument(format!(
                        "video frame group has {len} tokens, grid {g:?} expands to {expected}"
                    )));
                }
                vision_positions(*g, merge, current_pos, &mut pos[start..i])?;
                current_pos = advance_vision_pos(*g, merge, current_pos)?;
            }
            other => {
                return Err(Error::InvalidArgument(format!(
                    "unknown mm_token_type_id {other} (expected 0/1/2)"
                )));
            }
        }
    }
    if image_idx != image_grid_thw.len() {
        return Err(Error::InvalidArgument(format!(
            "{} image_grid_thw entries were not consumed",
            image_grid_thw.len() - image_idx
        )));
    }
    if video_idx != video_frames.len() {
        return Err(Error::InvalidArgument(format!(
            "{} video frame grids were not consumed",
            video_frames.len() - video_idx
        )));
    }
    let max_pos = pos
        .iter()
        .flat_map(|p| p.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let delta = max_pos
        .checked_add(1)
        .and_then(|v| v.checked_sub(i32::try_from(input_ids.len()).ok()?))
        .ok_or_else(|| Error::InvalidArgument("M-RoPE delta overflow".into()))?;
    Ok(MropePositions { pos, delta })
}

fn vision_group_len(g: [u32; 3], merge: usize) -> Result<usize, Error> {
    let t = usize::try_from(g[0]).map_err(|_| Error::InvalidArgument("grid t too large".into()))?;
    let h = usize::try_from(g[1]).map_err(|_| Error::InvalidArgument("grid h too large".into()))?;
    let w = usize::try_from(g[2]).map_err(|_| Error::InvalidArgument("grid w too large".into()))?;
    if t == 0 || h == 0 || w == 0 || !h.is_multiple_of(merge) || !w.is_multiple_of(merge) {
        return Err(Error::InvalidArgument(format!(
            "invalid or unmerged vision grid {g:?} for merge {merge}"
        )));
    }
    t.checked_mul(h / merge)
        .and_then(|v| v.checked_mul(w / merge))
        .ok_or_else(|| Error::InvalidArgument("vision group length overflow".into()))
}

fn vision_positions(
    g: [u32; 3],
    merge: usize,
    start: i32,
    out: &mut [[i32; 3]],
) -> Result<(), Error> {
    let t = usize::try_from(g[0]).map_err(|_| Error::InvalidArgument("grid t too large".into()))?;
    let h = usize::try_from(g[1]).map_err(|_| Error::InvalidArgument("grid h too large".into()))?
        / merge;
    let w = usize::try_from(g[2]).map_err(|_| Error::InvalidArgument("grid w too large".into()))?
        / merge;
    let mut k = 0usize;
    for tt in 0..t {
        for hh in 0..h {
            for ww in 0..w {
                let axis = |v: usize| -> Result<i32, Error> {
                    start
                        .checked_add(i32::try_from(v).map_err(|_| {
                            Error::InvalidArgument("vision position exceeds i32".into())
                        })?)
                        .ok_or_else(|| Error::InvalidArgument("vision position overflow".into()))
                };
                out[k] = [axis(tt)?, axis(hh)?, axis(ww)?];
                k += 1;
            }
        }
    }
    Ok(())
}

fn advance_vision_pos(g: [u32; 3], merge: usize, current: i32) -> Result<i32, Error> {
    let h = usize::try_from(g[1]).map_err(|_| Error::InvalidArgument("grid h too large".into()))?;
    let w = usize::try_from(g[2]).map_err(|_| Error::InvalidArgument("grid w too large".into()))?;
    let step = h.max(w) / merge;
    current
        .checked_add(
            i32::try_from(step)
                .map_err(|_| Error::InvalidArgument("vision position step exceeds i32".into()))?,
        )
        .ok_or_else(|| Error::InvalidArgument("vision position overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn axes(p: &MropePositions) -> [Vec<i32>; 3] {
        [
            p.pos.iter().map(|x| x[0]).collect(),
            p.pos.iter().map(|x| x[1]).collect(),
            p.pos.iter().map(|x| x[2]).collect(),
        ]
    }

    #[test]
    fn text_positions_are_sequential() {
        let p = qwen3_5_mrope_positions(
            &[10, 11, 12],
            &[0, 0, 0],
            &[],
            &[],
            &VisionConfig::default(),
        )
        .unwrap();
        assert_eq!(p.pos, vec![[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
        assert_eq!(p.delta, 0);
    }

    #[test]
    fn image_positions_match_hf_golden() {
        let p = qwen3_5_mrope_positions(
            &[10, 11, 20, 20, 20, 20, 12],
            &[0, 0, 1, 1, 1, 1, 0],
            &[[1, 4, 4]],
            &[],
            &VisionConfig::default(),
        )
        .unwrap();
        assert_eq!(
            axes(&p),
            [
                vec![0, 1, 2, 2, 2, 2, 4],
                vec![0, 1, 2, 2, 3, 3, 4],
                vec![0, 1, 2, 3, 2, 3, 4],
            ]
        );
        assert_eq!(p.delta, -2);
    }

    #[test]
    fn video_frame_positions_match_hf_golden() {
        let p = qwen3_5_mrope_positions(
            &[10, 20, 20, 20, 20, 11, 20, 20, 20, 20, 12],
            &[0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0],
            &[],
            &[[2, 4, 4]],
            &VisionConfig::default(),
        )
        .unwrap();
        assert_eq!(
            axes(&p),
            [
                vec![0, 1, 1, 1, 1, 3, 4, 4, 4, 4, 6],
                vec![0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6],
                vec![0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6],
            ]
        );
        assert_eq!(p.delta, -4);
    }

    #[test]
    fn cos_sin_uses_interleaved_mrope() {
        let p = MropePositions {
            pos: vec![[5, 7, 11], [6, 8, 12]],
            delta: 0,
        };
        let cfg = Config::llama(8, 1, 1, 1, 16, 32);
        let (cos, sin) = p.cos_sin(&cfg, [1, 1, 1]).unwrap();
        assert_eq!(cos.len(), 2 * cfg.attn_rotary_dim());
        assert_eq!(sin.len(), cos.len());
        assert!(cos.iter().all(|v| v.is_finite()));
        assert!(sin.iter().all(|v| v.is_finite()));
    }
}
