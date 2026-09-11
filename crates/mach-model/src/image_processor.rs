//! Qwen2/3-VL-compatible image preprocessing on the host CPU.
//!
//! Reference: `transformers` 5.16.1 `Qwen2VLImageProcessorPil`
//! (`models/qwen2_vl/image_processing_pil_qwen2_vl.py`) and Pillow 12.3.0
//! (`src/libImaging/Resample.c`). Qwen3.8-27B ships
//! `image_processor_type = Qwen2VLImageProcessorFast`; that backend differs
//! only in the torchvision resampling kernel, while grid/token semantics are
//! identical. This module matches the framework-independent PIL reference
//! bit-for-bit, including the 22-bit fixed-point BICUBIC resampler, so image
//! inputs are reproducible without torchvision.
//!
//! For one image the output layout matches HF exactly:
//! - `grid = [1, resized_height / patch_size, resized_width / patch_size]`;
//! - patch vectors are ordered `(channel, temporal, patch_y, patch_x)`, with
//!   the single frame repeated `temporal_patch_size` times.

use crate::Error;
use crate::vision::VisionGrid;

/// Sampling configuration parsed from `preprocessor_config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageProcessorConfig {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    /// `size.shortest_edge`: minimum number of pixels after resize.
    pub min_pixels: usize,
    /// `size.longest_edge`: maximum number of pixels after resize.
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        // Qwen3.8-27B `preprocessor_config.json`.
        Self {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
        }
    }
}

impl ImageProcessorConfig {
    /// Parse the fields consumed from an HF `preprocessor_config.json`.
    pub fn from_hf_json(v: &serde_json::Value) -> Result<Self, Error> {
        let mut cfg = Self::default();
        if let Some(size) = v.get("size") {
            if let Some(x) = size.get("shortest_edge") {
                cfg.min_pixels = json_usize(x, "size.shortest_edge")?;
            }
            if let Some(x) = size.get("longest_edge") {
                cfg.max_pixels = json_usize(x, "size.longest_edge")?;
            }
        }
        if let Some(x) = v.get("patch_size") {
            cfg.patch_size = json_usize(x, "patch_size")?;
        }
        if let Some(x) = v.get("temporal_patch_size") {
            cfg.temporal_patch_size = json_usize(x, "temporal_patch_size")?;
        }
        if let Some(x) = v.get("merge_size") {
            cfg.merge_size = json_usize(x, "merge_size")?;
        }
        if let Some(x) = v.get("image_mean") {
            cfg.image_mean = json_rgb(x, "image_mean")?;
        }
        if let Some(x) = v.get("image_std") {
            cfg.image_std = json_rgb(x, "image_std")?;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Pixel-count divisor: `patch_size * merge_size`.
    #[must_use]
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Reject configurations that would make the resize layout ill-defined.
    pub fn validate(&self) -> Result<(), Error> {
        if self.patch_size == 0 || self.temporal_patch_size == 0 || self.merge_size == 0 {
            return Err(Error::InvalidArgument(
                "image processor patch/temporal/merge sizes must be positive".into(),
            ));
        }
        if self.patch_size > 256 || self.merge_size > 256 || self.temporal_patch_size > 16 {
            return Err(Error::InvalidArgument(
                "image processor patch/temporal/merge sizes out of range".into(),
            ));
        }
        if self.min_pixels == 0 || self.max_pixels < self.min_pixels {
            return Err(Error::InvalidArgument(format!(
                "image processor needs 0 < min_pixels <= max_pixels, got {}..{}",
                self.min_pixels, self.max_pixels
            )));
        }
        for (i, (m, s)) in self.image_mean.iter().zip(&self.image_std).enumerate() {
            if !m.is_finite() || !s.is_finite() || *s == 0.0 {
                return Err(Error::InvalidArgument(format!(
                    "image processor channel {i} has invalid mean/std {m}/{s}"
                )));
            }
        }
        Ok(())
    }
}

/// One image after HF-compatible preprocessing.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedImage {
    /// Row-major patch vectors, `(patches, channels * temporal * patch^2)`.
    pub pixel_values: Vec<f32>,
    /// `[1, grid_h, grid_w]` for a single image.
    pub grid: VisionGrid,
}

impl ProcessedImage {
    /// Number of vision patches in this image.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.grid[0] * self.grid[1] * self.grid[2]
    }
}

/// HF `smart_resize`: snap both sides to `factor` while keeping the pixel
/// count in `[min_pixels, max_pixels]` and the aspect ratio as close as
/// possible. Mirrors the Python float/round/floor/ceil sequence, including
/// banker-rounding of the initial snap.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), Error> {
    if height == 0 || width == 0 || factor == 0 {
        return Err(Error::InvalidArgument(
            "smart_resize needs positive height/width/factor".into(),
        ));
    }
    if min_pixels == 0 || max_pixels < min_pixels {
        return Err(Error::InvalidArgument(
            "smart_resize needs 0 < min_pixels <= max_pixels".into(),
        ));
    }
    let h = height as f64;
    let w = width as f64;
    let ratio = h.max(w) / h.min(w);
    if ratio > 200.0 {
        return Err(Error::InvalidArgument(format!(
            "image aspect ratio {ratio:.3} exceeds 200"
        )));
    }
    let factor_f = factor as f64;
    let mut h_bar = (h / factor_f).round_ties_even() * factor_f;
    let mut w_bar = (w / factor_f).round_ties_even() * factor_f;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = factor_f.max((h / beta / factor_f).floor() * factor_f);
        w_bar = factor_f.max((w / beta / factor_f).floor() * factor_f);
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / factor_f).ceil() * factor_f;
        w_bar = (w * beta / factor_f).ceil() * factor_f;
    }
    let resized_h = f64_to_usize(h_bar, "resized height")?;
    let resized_w = f64_to_usize(w_bar, "resized width")?;
    if resized_h.checked_mul(resized_w).is_none() {
        return Err(Error::InvalidArgument(
            "smart_resize output pixel count overflows usize".into(),
        ));
    }
    Ok((resized_h, resized_w))
}

/// Decode-free preprocessing: `rgb8` is `height * width * 3` bytes in
/// row-major RGB order. Produces `pixel_values` and `grid` ready for
/// [`crate::vision::vision_forward`] / the GPU vision runtime.
pub fn preprocess_image(
    cfg: &ImageProcessorConfig,
    rgb8: &[u8],
    height: usize,
    width: usize,
) -> Result<ProcessedImage, Error> {
    cfg.validate()?;
    if height == 0 || width == 0 {
        return Err(Error::InvalidArgument(
            "image height/width must be positive".into(),
        ));
    }
    let expect = height
        .checked_mul(width)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| Error::InvalidArgument("image size overflow".into()))?;
    if rgb8.len() != expect {
        return Err(Error::InvalidArgument(format!(
            "image buffer has {} bytes, expected {expect} for {height}x{width} RGB",
            rgb8.len()
        )));
    }
    let (resized_h, resized_w) =
        smart_resize(height, width, cfg.factor(), cfg.min_pixels, cfg.max_pixels)?;
    let resized = resize_rgb8_bicubic(rgb8, height, width, resized_h, resized_w)?;
    let grid_h = resized_h / cfg.patch_size;
    let grid_w = resized_w / cfg.patch_size;
    let rows = grid_h / cfg.merge_size;
    let cols = grid_w / cfg.merge_size;
    let patch = cfg.patch_size;
    let merge = cfg.merge_size;
    let temporal = cfg.temporal_patch_size;
    let patch_dim = 3usize
        .checked_mul(temporal)
        .and_then(|v| v.checked_mul(patch))
        .and_then(|v| v.checked_mul(patch))
        .ok_or_else(|| Error::InvalidArgument("image patch dimension overflow".into()))?;
    let patches = grid_h
        .checked_mul(grid_w)
        .ok_or_else(|| Error::InvalidArgument("image patch count overflow".into()))?;
    let mut pixel_values = Vec::with_capacity(
        patches
            .checked_mul(patch_dim)
            .ok_or_else(|| Error::InvalidArgument("image patch buffer overflow".into()))?,
    );
    for row in 0..rows {
        for col in 0..cols {
            for mh in 0..merge {
                for mw in 0..merge {
                    let base_y = (row * merge + mh) * patch;
                    let base_x = (col * merge + mw) * patch;
                    for c in 0..3 {
                        let mean = cfg.image_mean[c];
                        let std = cfg.image_std[c];
                        for _ in 0..temporal {
                            for py in 0..patch {
                                for px in 0..patch {
                                    let raw =
                                        resized[((base_y + py) * resized_w + base_x + px) * 3 + c];
                                    let v = (raw as f32 / 255.0 - mean) / std;
                                    pixel_values.push(v);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(pixel_values.len(), patches * patch_dim);
    Ok(ProcessedImage {
        pixel_values,
        grid: [1, grid_h, grid_w],
    })
}

/// Pillow `ImagingResampleHorizontal/Vertical_8bpc` with BICUBIC.
fn resize_rgb8_bicubic(
    src: &[u8],
    height: usize,
    width: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<u8>, Error> {
    let tmp_len = height
        .checked_mul(out_w)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| Error::InvalidArgument("resized intermediate too large".into()))?;
    let out_len = out_h
        .checked_mul(out_w)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| Error::InvalidArgument("resized image too large".into()))?;
    let hcoeff = resample_coeffs(width, out_w);
    let mut tmp = vec![0u8; tmp_len];
    for y in 0..height {
        let row = &src[y * width * 3..(y + 1) * width * 3];
        for (x, (xmin, ks)) in hcoeff.iter().enumerate() {
            for c in 0..3 {
                let mut ss = 1i32 << 21;
                for (j, &k) in ks.iter().enumerate() {
                    ss += row[(xmin + j) * 3 + c] as i32 * k;
                }
                tmp[(y * out_w + x) * 3 + c] = clip8(ss);
            }
        }
    }
    let vcoeff = resample_coeffs(height, out_h);
    let mut out = vec![0u8; out_len];
    for (y, (ymin, ks)) in vcoeff.iter().enumerate() {
        for x in 0..out_w {
            for c in 0..3 {
                let mut ss = 1i32 << 21;
                for (j, &k) in ks.iter().enumerate() {
                    ss += tmp[((ymin + j) * out_w + x) * 3 + c] as i32 * k;
                }
                out[(y * out_w + x) * 3 + c] = clip8(ss);
            }
        }
    }
    Ok(out)
}

/// `(first source index, fixed-point weights)` per output sample.
fn resample_coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let inv_filterscale = 1.0 / filterscale;
    let mut out = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
        let count = xmax - xmin;
        let mut weights = Vec::with_capacity(count);
        let mut ww = 0.0;
        for x in 0..count {
            let w = bicubic((x as f64 + xmin as f64 - center + 0.5) * inv_filterscale);
            weights.push(w);
            ww += w;
        }
        let ks = weights
            .into_iter()
            .map(|w| fixed_point(if ww != 0.0 { w / ww } else { w }))
            .collect();
        out.push((xmin, ks));
    }
    out
}

/// Pillow `bicubic_filter` with `a = -0.5`.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Pillow `normalize_coeffs_8bpc`: round-half-away-from-zero into Q22.
fn fixed_point(w: f64) -> i32 {
    let scaled = w * ((1i32 << 22) as f64);
    if w < 0.0 {
        (-0.5 + scaled) as i32
    } else {
        (0.5 + scaled) as i32
    }
}

/// Pillow `clip8`: shift back out of Q22 and clamp to `[0, 255]`.
fn clip8(ss: i32) -> u8 {
    if ss < 0 {
        0
    } else if ss >= 255 << 22 {
        255
    } else {
        (ss >> 22) as u8
    }
}

fn f64_to_usize(v: f64, what: &str) -> Result<usize, Error> {
    if !v.is_finite() || v < 0.0 || v > usize::MAX as f64 {
        return Err(Error::InvalidArgument(format!(
            "{what} is not a valid size"
        )));
    }
    Ok(v as usize)
}

fn json_usize(v: &serde_json::Value, key: &str) -> Result<usize, Error> {
    v.as_u64()
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| Error::InvalidArgument(format!("{key} must be a non-negative integer")))
}

fn json_rgb(v: &serde_json::Value, key: &str) -> Result<[f32; 3], Error> {
    let arr = v
        .as_array()
        .filter(|a| a.len() == 3)
        .ok_or_else(|| Error::InvalidArgument(format!("{key} must be an array of 3 numbers")))?;
    let mut out = [0.0f32; 3];
    for (i, item) in arr.iter().enumerate() {
        out[i] = item
            .as_f64()
            .map(|x| x as f32)
            .ok_or_else(|| Error::InvalidArgument(format!("{key}[{i}] must be a number")))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_matches_pillow_bicubic() {
        const SRC: [u8; 60] = [
            225, 83, 51, 234, 213, 237, 28, 225, 137, 23, 40, 1, 13, 31, 78, 212, 115, 135, 89,
            186, 143, 58, 11, 170, 193, 205, 168, 120, 182, 147, 51, 251, 18, 2, 76, 16, 11, 154,
            158, 52, 207, 135, 237, 178, 135, 148, 92, 218, 247, 42, 161, 31, 50, 139, 118, 206,
            209, 9, 31, 244,
        ];
        const WANT: [u8; 144] = [
            224, 72, 32, 244, 133, 128, 236, 219, 245, 87, 253, 182, 7, 184, 85, 12, 37, 0, 8, 13,
            34, 5, 20, 80, 232, 82, 90, 212, 141, 140, 165, 206, 202, 72, 145, 175, 56, 110, 129,
            110, 118, 84, 82, 108, 96, 53, 101, 113, 200, 131, 123, 143, 157, 120, 63, 166, 122,
            43, 46, 157, 90, 61, 176, 181, 208, 174, 167, 211, 158, 140, 190, 146, 70, 247, 21, 37,
            181, 15, 0, 88, 27, 5, 117, 125, 26, 163, 165, 60, 207, 138, 173, 200, 132, 249, 187,
            130, 91, 179, 119, 112, 123, 102, 124, 50, 86, 45, 81, 128, 26, 142, 158, 75, 206, 171,
            108, 147, 186, 126, 92, 194, 147, 86, 235, 209, 65, 209, 255, 39, 167, 92, 29, 139, 44,
            94, 155, 122, 205, 210, 49, 102, 241, 0, 8, 254,
        ];
        let got = resize_rgb8_bicubic(&SRC, 4, 5, 6, 8).unwrap();
        assert_eq!(got, WANT);
    }
}
