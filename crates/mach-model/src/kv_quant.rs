//! KV-cache quantization primitives.
//!
//! Stage 1 implements only the full-attention INT8 layout and its CPU oracle.
//! Quantization is symmetric per `(token, kv_head, head_dim block)`: the
//! payload is one `i8` per K/V element and one `f32` scale per head row. This
//! keeps the layout identical for the future contiguous and paged GPU stores;
//! it is intentionally not wired into the f16/f32 runtime yet.

use crate::Error;

/// One-byte INT8 KV storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuantFormat {
    F16,
    Int8,
}

impl KvQuantFormat {
    /// Bytes per K/V scalar, excluding scale metadata.
    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::Int8 => 1,
        }
    }
}

/// Dequantize one symmetric INT8 value without overflowing at finite f32
/// extremes. The f64 product is saturated back to the finite f32 range.
fn dequant_value(q: i8, scale: f32) -> f32 {
    let v = q as f64 * scale as f64;
    if v > f32::MAX as f64 {
        f32::MAX
    } else if v < f32::MIN as f64 {
        f32::MIN
    } else {
        v as f32
    }
}

/// Symmetric INT8 KV payload plus one f32 scale per token/head row.
#[derive(Debug, Clone, PartialEq)]
pub struct Int8Kv {
    q: Vec<i8>,
    scales: Vec<f32>,
    heads: usize,
    head_dim: usize,
}

impl Int8Kv {
    /// Quantize `values` shaped `[tokens, heads, head_dim]` with one symmetric
    /// scale per `(token, head)`. The quantized range is `[-127, 127]`; `-128`
    /// is deliberately unused so dequantization stays exactly symmetric.
    pub fn quantize(values: &[f32], heads: usize, head_dim: usize) -> Result<Self, Error> {
        if heads == 0 || head_dim == 0 {
            return Err(Error::InvalidArgument(
                "INT8 KV requires non-zero heads and head_dim".into(),
            ));
        }
        let block = heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::InvalidArgument("INT8 KV head block overflow".into()))?;
        if !values.len().is_multiple_of(block) {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV values length {} is not a multiple of heads*head_dim {block}",
                values.len()
            )));
        }
        if let Some((i, v)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, v)| !v.is_finite())
        {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV input has non-finite value at index {i}: {v}"
            )));
        }

        let rows = values.len() / block;
        let mut q = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(rows * heads);
        for row in 0..rows {
            for head in 0..heads {
                let start = row * block + head * head_dim;
                let end = start + head_dim;
                let max_abs = values[start..end]
                    .iter()
                    .fold(0.0f32, |m, &v| m.max(v.abs()));
                let scale = if max_abs == 0.0 {
                    1.0
                } else {
                    let s = max_abs / 127.0;
                    if s == 0.0 { max_abs } else { s }
                };
                scales.push(scale);
                for &v in &values[start..end] {
                    let r = (v / scale).round();
                    let clamped = if r >= 127.0 {
                        127i8
                    } else if r <= -127.0 {
                        -127i8
                    } else {
                        r as i8
                    };
                    q.push(clamped);
                }
            }
        }
        Ok(Self {
            q,
            scales,
            heads,
            head_dim,
        })
    }

    pub fn tokens(&self) -> usize {
        if self.heads == 0 || self.head_dim == 0 {
            0
        } else {
            self.q.len() / (self.heads * self.head_dim)
        }
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Packed payload + scale bytes for this K or V tensor.
    pub fn bytes(&self) -> usize {
        self.q.len() + self.scales.len() * std::mem::size_of::<f32>()
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn quantized(&self) -> &[i8] {
        &self.q
    }

    pub fn scale(&self, token: usize, head: usize) -> Option<f32> {
        if token >= self.tokens() || head >= self.heads {
            return None;
        }
        Some(self.scales[token * self.heads + head])
    }

    pub fn head_row(&self, token: usize, head: usize) -> Option<&[i8]> {
        if token >= self.tokens() || head >= self.heads {
            return None;
        }
        let block = self.heads * self.head_dim;
        let start = token * block + head * self.head_dim;
        Some(&self.q[start..start + self.head_dim])
    }

    /// Quantized K dot an f32 query row: `sum(q_i * k_i) * scale`. Q is
    /// intentionally not quantized; attention dequantization multiplies the
    /// completed dot by the single per-head scale.
    pub fn dot_q(&self, q: &[f32], token: usize, head: usize) -> Result<f32, Error> {
        if q.len() != self.head_dim {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV dot query length {} != head_dim {}",
                q.len(),
                self.head_dim
            )));
        }
        if let Some((i, v)) = q.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV dot query has non-finite value at index {i}: {v}"
            )));
        }
        let row = self.head_row(token, head).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "INT8 KV dot index out of range: token={token} head={head}"
            ))
        })?;
        let mut acc = 0.0f64;
        for (&qi, &ki) in q.iter().zip(row) {
            acc += qi as f64 * ki as f64;
        }
        let out = acc * self.scale(token, head).unwrap() as f64;
        if !out.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV dot produced a non-finite value: token={token} head={head} value={out}"
            )));
        }
        let scale = self.scale(token, head).unwrap() as f64;
        // The scale itself is rounded to f32, so a mathematically finite dot
        // exactly at f32::MAX can land a fraction of one quantization step
        // outside the f32 range. Accept that rounding margin; larger overflow
        // is a real out-of-range dot.
        let tol = scale * 0.5;
        if out > f32::MAX as f64 {
            if out - f32::MAX as f64 <= tol {
                return Ok(f32::MAX);
            }
            return Err(Error::InvalidArgument(format!(
                "INT8 KV dot overflow: token={token} head={head} value={out}"
            )));
        }
        if out < f32::MIN as f64 {
            if f32::MIN as f64 - out <= tol {
                return Ok(f32::MIN);
            }
            return Err(Error::InvalidArgument(format!(
                "INT8 KV dot underflow: token={token} head={head} value={out}"
            )));
        }
        Ok(out as f32)
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.q.len());
        let block = self.heads * self.head_dim;
        for row in 0..self.tokens() {
            for head in 0..self.heads {
                let scale = self.scales[row * self.heads + head];
                let start = row * block + head * self.head_dim;
                let end = start + self.head_dim;
                out.extend(self.q[start..end].iter().map(|&v| dequant_value(v, scale)));
            }
        }
        out
    }

    pub fn dequantize_into(&self, out: &mut [f32]) -> Result<(), Error> {
        if out.len() != self.q.len() {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV dequantize output length {} != {}",
                out.len(),
                self.q.len()
            )));
        }
        let block = self.heads * self.head_dim;
        for row in 0..self.tokens() {
            for head in 0..self.heads {
                let scale = self.scales[row * self.heads + head];
                let start = row * block + head * self.head_dim;
                let end = start + self.head_dim;
                for (dst, &v) in out[start..end].iter_mut().zip(&self.q[start..end]) {
                    *dst = dequant_value(v, scale);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    }

    #[test]
    fn int8_roundtrip_error_is_bounded_by_half_scale() {
        let mut seed = 7u64;
        let values: Vec<f32> = (0..3 * 4 * 256).map(|_| lcg(&mut seed) * 8.0).collect();
        let kv = Int8Kv::quantize(&values, 4, 256).unwrap();
        let out = kv.dequantize();
        assert_eq!(out.len(), values.len());
        for token in 0..3 {
            for head in 0..4 {
                let scale = kv.scale(token, head).unwrap();
                let block = 4 * 256;
                let start = token * block + head * 256;
                for i in start..start + 256 {
                    assert!(
                        (out[i] - values[i]).abs() <= scale * 0.500_01,
                        "token={token} head={head} i={i} err={} scale={scale}",
                        (out[i] - values[i]).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn zero_head_roundtrips_exactly_with_finite_scale() {
        let mut values = vec![0.0f32; 2 * 4 * 256];
        values[256] = 1.0;
        let kv = Int8Kv::quantize(&values, 4, 256).unwrap();
        assert!(kv.scales().iter().all(|x| x.is_finite()));
        assert_eq!(kv.scale(0, 0), Some(1.0));
        let out = kv.dequantize();
        assert_eq!(out[0..256], values[0..256]);
    }

    #[test]
    fn subnormal_input_does_not_poison_scale() {
        let tiny = f32::from_bits(1);
        let values = [tiny, tiny * 2.0, -tiny, 0.0];
        let kv = Int8Kv::quantize(&values, 1, 4).unwrap();
        let out = kv.dequantize();
        assert!(kv.scales().iter().all(|x| x.is_finite()));
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(kv.quantized().iter().any(|&q| q != 0));
        for (a, b) in values.iter().zip(&out) {
            assert!((a - b).abs() <= tiny);
        }
    }

    #[test]
    fn dot_rejects_non_finite_query_and_extreme_overflow() {
        let kv = Int8Kv::quantize(&[f32::MAX, 1.0], 1, 2).unwrap();
        assert!(kv.dot_q(&[1.0, f32::NAN], 0, 0).is_err());
        assert!(kv.dot_q(&[2.0, 0.0], 0, 0).is_err());
        assert!(kv.dot_q(&[1.0, 0.0], 0, 0).is_ok());
    }

    #[test]
    fn finite_extreme_roundtrip_does_not_overflow() {
        let values = [f32::MAX, f32::MIN, 0.0, 1.0];
        let kv = Int8Kv::quantize(&values, 1, 4).unwrap();
        let out = kv.dequantize();
        assert!(out.iter().all(|x| x.is_finite()));
        assert_eq!(out[0], f32::MAX);
        assert_eq!(out[1], f32::MIN);
    }

    #[test]
    fn non_finite_input_is_rejected() {
        let values = [0.0f32, f32::NAN];
        assert!(Int8Kv::quantize(&values, 1, 2).is_err());
        let values = [0.0f32, f32::INFINITY];
        assert!(Int8Kv::quantize(&values, 1, 2).is_err());
    }

    #[test]
    fn layout_counts_and_bytes_match_contract() {
        let values = vec![0.25f32; 5 * 4 * 256];
        let kv = Int8Kv::quantize(&values, 4, 256).unwrap();
        assert_eq!(kv.tokens(), 5);
        assert_eq!(kv.heads(), 4);
        assert_eq!(kv.head_dim(), 256);
        assert_eq!(kv.scales().len(), 5 * 4);
        assert_eq!(kv.quantized().len(), values.len());
        assert_eq!(kv.bytes(), values.len() + 5 * 4 * 4);
        assert!(kv.bytes() < values.len() * std::mem::size_of::<f32>() / 2);
    }

    #[test]
    fn qk_dot_error_tracks_quantization_error() {
        let mut seed = 11u64;
        let head_dim = 256usize;
        let k: Vec<f32> = (0..head_dim).map(|_| lcg(&mut seed) * 3.0).collect();
        let q: Vec<f32> = (0..head_dim).map(|_| lcg(&mut seed)).collect();
        let kv = Int8Kv::quantize(&k, 1, head_dim).unwrap();
        let exact: f32 = q.iter().zip(&k).map(|(a, b)| a * b).sum();
        let got = kv.dot_q(&q, 0, 0).unwrap();
        let scale = kv.scale(0, 0).unwrap();
        let bound = q.iter().map(|x| x.abs()).sum::<f32>() * scale * 0.500_01;
        assert!(
            (got - exact).abs() <= bound,
            "dot err {} bound {bound}",
            (got - exact).abs()
        );
    }

    #[test]
    fn qk_softmax_stays_close_with_int8_k() {
        let mut seed = 19u64;
        let tokens = 64usize;
        let head_dim = 256usize;
        let q: Vec<f32> = (0..head_dim).map(|_| lcg(&mut seed)).collect();
        let k: Vec<f32> = (0..tokens * head_dim)
            .map(|_| lcg(&mut seed) * 2.0)
            .collect();
        let kv = Int8Kv::quantize(&k, 1, head_dim).unwrap();

        let mut exact = vec![0.0f32; tokens];
        let mut quant = vec![0.0f32; tokens];
        for t in 0..tokens {
            exact[t] = q
                .iter()
                .zip(&k[t * head_dim..(t + 1) * head_dim])
                .map(|(a, b)| a * b)
                .sum();
            quant[t] = kv.dot_q(&q, t, 0).unwrap();
        }
        let max = exact.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exact_sum: f32 = exact.iter().map(|x| (x - max).exp()).sum();
        let quant_max = quant.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let quant_sum: f32 = quant.iter().map(|x| (x - quant_max).exp()).sum();
        let mut max_prob_diff = 0.0f32;
        for t in 0..tokens {
            let pe = (exact[t] - max).exp() / exact_sum;
            let pq = (quant[t] - quant_max).exp() / quant_sum;
            assert!(pe.is_finite() && pq.is_finite());
            max_prob_diff = max_prob_diff.max((pe - pq).abs());
        }
        assert!(
            max_prob_diff < 0.02,
            "max softmax prob diff {max_prob_diff}"
        );
    }
}
