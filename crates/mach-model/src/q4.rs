//! Storage-level symmetric int4 quantization with a per-group f32 scale.
//!
//! This is a *storage* format: weights are held on the host as packed int4
//! nibbles (2 values/byte) plus one scale per group, and dequantized to f16 on
//! the device during upload. It cuts host RAM ~8x vs f32 (8B model: 32GB ->
//! ~5GB) while keeping the existing f16 GEMM compute path. It does not speed
//! up compute; true int4 GEMM would need custom kernels (hipBLAS has none on
//! gfx1100).

/// Group size for the per-group scale.
pub const Q4_GROUP: usize = 32;

/// A quantized tensor: packed int4 (low nibble first) + per-group f32 scales.
#[derive(Clone, Debug, Default)]
pub struct Q4Tensor {
    q: Vec<u8>,
    scales: Vec<f32>,
    n: usize,
}

fn signed_nibble(v: u8) -> i32 {
    let lo = v & 0x0F;
    if lo < 8 { lo as i32 } else { lo as i32 - 16 }
}

impl Q4Tensor {
    /// Symmetric per-group quantization: `scale = max(|w|)/7` so the packed
    #[allow(clippy::needless_range_loop)]
    /// signed int4 stays within `[-7, 7]`.
    pub fn quantize(w: &[f32]) -> Self {
        let n = w.len();
        let groups = n.div_ceil(Q4_GROUP);
        let mut q = vec![0u8; n.div_ceil(2)];
        let mut scales = vec![0f32; groups];
        for g in 0..groups {
            let start = g * Q4_GROUP;
            let end = (start + Q4_GROUP).min(n);
            let mut max_abs = 0f32;
            for &v in &w[start..end] {
                max_abs = max_abs.max(v.abs());
            }
            let scale = max_abs / 7.0;
            scales[g] = scale;
            for i in start..end {
                let qi = if scale > 0.0 {
                    (w[i] / scale).round().clamp(-7.0, 7.0) as i8
                } else {
                    0i8
                };
                let byte = i / 2;
                let nib = (qi as u8 & 0x0F) << if i % 2 == 0 { 0 } else { 4 };
                q[byte] |= nib;
            }
        }
        Self { q, scales, n }
    }

    /// Dequantizes back to f32.
    #[allow(clippy::needless_range_loop)]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.n];
        for i in 0..self.n {
            let byte = self.q[i / 2];
            let nib = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            let g = i / Q4_GROUP;
            out[i] = signed_nibble(nib) as f32 * self.scales[g];
        }
        out
    }

    /// Dequantizes to f16 bit patterns (for direct device upload).
    pub fn dequantize_f16(&self) -> Vec<u16> {
        self.dequantize()
            .iter()
            .map(|&v| crate::fp16::f32_to_f16(v))
            .collect()
    }

    /// Number of stored elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// True when no elements are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_error_is_bounded_by_half_scale() {
        let mut w = vec![0f32; 1000];
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i as f32) * 0.7).sin() * 3.0 + 0.1 * (i as f32);
        }
        let q = Q4Tensor::quantize(&w);
        assert_eq!(q.len(), 1000);
        let d = q.dequantize();
        for i in 0..w.len() {
            let err = (d[i] - w[i]).abs();
            let g = i / Q4_GROUP;
            // int4 step = scale / 2 -> max abs error <= scale/2.
            assert!(err <= q.scales[g] / 2.0 + 1e-6, "i={i} err={err}");
        }
    }

    #[test]
    fn empty_tensor_is_empty() {
        let q = Q4Tensor::quantize(&[]);
        assert!(q.is_empty());
        assert_eq!(q.dequantize(), Vec::<f32>::new());
    }

    #[test]
    fn dequant_f16_length_matches() {
        let w: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.0).collect();
        let q = Q4Tensor::quantize(&w);
        assert_eq!(q.dequantize_f16().len(), 64);
    }
}
