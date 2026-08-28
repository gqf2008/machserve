//! Storage-level FP8 (E4M3) quantization with a per-block f32 scale.
//!
//! This is a *storage* format: weights are held on the host as one E4M3 byte
//! per element plus one f32 scale per block (default: one scale per tensor /
//! per MoE expert), and dequantized to f16 on the device during upload. It
//! cuts host RAM ~2x vs f16 (~4x vs f32) while keeping the existing f16 GEMM
//! compute path. It does not speed up compute; true FP8 GEMM is rejected by
//! hipBLAS on gfx1100/ROCm 6.2 and would need NVIDIA or custom kernels.
//!
//! Why the scale: E4M3 alone has a hard subnormal floor at 2^-9 (~0.002) —
//! weights below that flush to zero (100% relative error). Per-tensor scales
//! map each tensor's range onto [-448, 448], keeping E4M3's native ~6% relative
//! precision everywhere. Measured on qwen3-moe-tiny (CPU ref, 40 prompts):
//! plain E4M3 max logits diff 0.91 vs Q4 2.49 (same prompts); per-tensor scale
//! improves it to 0.77 at zero storage cost (1 byte/element + 4 bytes/tensor).
//! Per-group scales (group=32) reach 0.67 but add 12.5% storage; FP8's 3
//! mantissa bits make that unnecessary.
//!
//! E4M3 (per the FP8 deep-learning formats): 1 sign + 4 exponent + 3 mantissa
//! bits, bias 7. Range [-448, 448]; there is **no infinity** — the all-ones
//! exponent with mantissa `0b111` (0x7f / 0xff) is NaN, and exponent `0b1111`
//! with mantissa < 7 encodes normal values 256..448 (0x7e = 448).

/// Converts an f32 to E4M3 bits (round-to-nearest-even, NaN/Inf -> NaN,
/// overflow saturates to ±448, underflow flushes to zero).
#[must_use]
pub fn f32_to_e4m3(x: f32) -> u8 {
    let b = x.to_bits();
    let sign = ((b >> 24) & 0x80) as u8;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;

    if exp == 0xff {
        // NaN / Inf: E4M3 has no infinity, so both map to NaN.
        return sign | 0x7f;
    }
    // Rebias exponent from f32 (127) to E4M3 (7).
    let mut e = exp - 127 + 7;
    if e >= 0x10 {
        // > 448 -> saturate to max normal.
        return sign | 0x7e;
    }
    if e <= 0 {
        // Subnormal / zero. E4M3 subnormals encode m * 2^-9 (m in 1..=7).
        if e < -3 {
            return sign; // too small -> zero
        }
        let m = mant | 0x80_0000;
        let shift = 21 - e; // e in [-3, 0] -> shift in [21, 24]
        let half = 1u32 << (shift - 1);
        let mut m8 = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && (m8 & 1) == 1) {
            m8 += 1;
        }
        if m8 == 8 {
            // Rounds up to the min normal (2^-6).
            return sign | 0x08;
        }
        return sign | m8 as u8;
    }
    // Normal: 3 mantissa bits.
    let mut m8 = (mant >> 20) as u8;
    let rem = mant & 0x0f_ffff;
    if rem > 0x08_0000 || (rem == 0x08_0000 && (m8 & 1) == 1) {
        m8 += 1;
        if m8 == 8 {
            m8 = 0;
            e += 1;
        }
    }
    // Exponent 0b1111 with mantissa 7 is NaN (0x7f/0xff); values that would
    // round there (or overflow past 0b1111) saturate to 448.
    if e >= 0x10 || (e == 0x0f && m8 >= 7) {
        return sign | 0x7e;
    }
    sign | ((e as u8) << 3) | m8
}

/// Expands E4M3 bits to f32 (exact; NaN stays NaN).
#[must_use]
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = ((b as u32) & 0x80) << 24;
    let e = ((b >> 3) & 0x0f) as i32;
    let m = (b & 0x07) as u32;
    let bits = if e == 0 {
        if m == 0 {
            sign // zero
        } else {
            // Subnormal: value = m * 2^-9, m in 1..=7. Normalize exactly into
            // f32: the leading set bit of m selects the exponent (2^-9 for
            // m=1 .. 2^-7 for m=4..7), the rest becomes the mantissa.
            let k = m.ilog2(); // bit position of leading 1 (0..=2)
            let e2 = 118 + k; // biased exponent for 1.0 * 2^-9
            let frac = (m - (1 << k)) << (23 - k);
            sign | (e2 << 23) | frac
        }
    } else if e == 0x0f && m == 7 {
        // NaN (0x7f / 0xff).
        sign | 0x7fc0_0000
    } else {
        // Normal (includes exponent 0b1111 with m < 7: 256..448).
        sign | (((e - 7 + 127) as u32) << 23) | (m << 20)
    };
    f32::from_bits(bits)
}

/// A quantized tensor: one E4M3 byte per element plus one f32 scale per
/// `block` elements. The default is a single per-tensor scale (`block = n`);
/// MoE per-expert tensors keep `block = expert size` so concatenation appends
/// packed bytes + scales directly.
#[derive(Clone, Debug, Default)]
pub struct Fp8Tensor {
    q: Vec<u8>,
    scales: Vec<f32>,
    block: usize,
    n: usize,
}

/// Maximum E4M3 finite value (the scale denominator).
const E4M3_MAX: f32 = 448.0;

impl Fp8Tensor {
    /// Quantizes f32 weights to E4M3 bytes with a single per-tensor scale
    /// (`scale = max_abs / 448`, so the tensor's max maps to 448).
    pub fn quantize(w: &[f32]) -> Self {
        if w.is_empty() {
            return Self::default();
        }
        Self::quantize_block(w, w.len())
    }
    /// Quantizes with one scale per `block` elements (`block = n` is a single
    /// per-tensor scale; the MoE loader uses per-expert blocks).
    pub fn quantize_block(w: &[f32], block: usize) -> Self {
        let n = w.len();
        let block = block.max(1);
        let groups = n.div_ceil(block);
        let mut q = Vec::with_capacity(n);
        let mut scales = Vec::with_capacity(groups);
        for g in 0..groups {
            let start = g * block;
            let end = (start + block).min(n);
            let scale = w[start..end].iter().fold(0.0f32, |m, x| m.max(x.abs())) / E4M3_MAX;
            scales.push(scale);
            for &x in &w[start..end] {
                let qi = if scale > 0.0 {
                    f32_to_e4m3(x / scale)
                } else {
                    0
                };
                q.push(qi);
            }
        }
        Self {
            q,
            scales,
            block,
            n,
        }
    }

    /// Parallel quantize (per-tensor scale, bit-identical to [`Self::quantize`]):
    /// the block max is computed once, then element conversion is split across
    /// threads. Used by the FP8 loader so large single tensors
    /// (embedding/lm_head) do not serialize the whole load.
    pub fn quantize_par(w: &[f32]) -> Self {
        let n = w.len();
        if n == 0 {
            return Self::default();
        }
        let scale = w.iter().fold(0.0f32, |m, x| m.max(x.abs())) / E4M3_MAX;
        let n_threads = std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(4)
            .min(16);
        let per_thread = n.div_ceil(n_threads).max(1);
        let mut q = Vec::with_capacity(n);
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_threads);
            for t in 0..n_threads {
                let start = t * per_thread;
                let end = (start + per_thread).min(n);
                if start >= end {
                    continue;
                }
                handles.push(s.spawn(move || {
                    w[start..end]
                        .iter()
                        .map(|&x| {
                            if scale > 0.0 {
                                f32_to_e4m3(x / scale)
                            } else {
                                0
                            }
                        })
                        .collect::<Vec<u8>>()
                }));
            }
            for h in handles {
                q.extend_from_slice(&h.join().unwrap());
            }
        });
        Self {
            q,
            scales: vec![scale],
            block: n.max(1),
            n,
        }
    }

    /// Dequantizes back to f32.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n);
        for (i, &b) in self.q.iter().enumerate() {
            let s = self.scales[i / self.block.max(1)];
            out.push(e4m3_to_f32(b) * s);
        }
        out
    }

    /// Dequantizes directly to f16 bit patterns (for device upload), without
    /// materializing the full f32 vector (transient = 2 bytes/element).
    pub fn dequantize_f16(&self) -> Vec<u16> {
        let mut out = Vec::with_capacity(self.n);
        for (i, &b) in self.q.iter().enumerate() {
            let s = self.scales[i / self.block.max(1)];
            out.push(crate::fp16::f32_to_f16(e4m3_to_f32(b) * s));
        }
        out
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

    /// Byte size of the packed payload (without scales).
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.q.len()
    }

    /// Elements per scale block (1 for a single per-tensor scale; per-expert
    /// size for concatenated MoE tensors).
    #[must_use]
    pub fn block(&self) -> usize {
        self.block
    }

    /// Per-block f32 scales (one per `block` elements).
    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Concatenates two quantized tensors (per-expert MoE tensors). When both
    /// use the same block size and `self` ends on a block boundary (the loader
    /// always satisfies this: each expert is quantized with `block = expert
    /// size`), the packed bytes and scales append directly: exact, O(1) per
    /// expert. Unaligned tensors fall back to dequantize + requantize with a
    /// single per-tensor scale, preserving validity.
    pub fn concat(&self, other: &Self) -> Self {
        if self.n == 0 {
            return other.clone();
        }
        if other.n == 0 {
            return self.clone();
        }
        if self.block == other.block && self.n.is_multiple_of(self.block) {
            let mut q = self.q.clone();
            q.extend_from_slice(&other.q);
            let mut scales = self.scales.clone();
            scales.extend_from_slice(&other.scales);
            return Self {
                q,
                scales,
                block: self.block,
                n: self.n + other.n,
            };
        }
        let mut v = self.dequantize();
        v.extend(other.dequantize());
        Self::quantize(&v)
    }

    /// Single-pass append of `parts` in order, O(total) bytes — folding
    /// [`Self::concat`] instead re-clones the growing prefix per part
    /// (O(n²) over many MoE experts). Byte/scale-identical to that fold
    /// **iff every part is block-aligned with a common block** — the only
    /// shape real checkpoints hit (per-expert quantize sets
    /// `block = expert size`); otherwise both re-quantize, but scale
    /// grouping can differ from the sequential fold (see the
    /// `concat_many_matches_concat_fold` test).
    pub fn concat_many(parts: &[Self]) -> Self {
        if parts.is_empty() {
            return Self::default();
        }
        let block = parts[0].block;
        let n: usize = parts.iter().map(|p| p.n).sum();
        if block > 0
            && parts
                .iter()
                .all(|p| p.block == block && p.n.is_multiple_of(block))
        {
            let mut q = Vec::with_capacity(n);
            let mut scales = Vec::with_capacity(n / block);
            for p in parts {
                q.extend_from_slice(&p.q);
                scales.extend_from_slice(&p.scales);
            }
            return Self {
                q,
                scales,
                block,
                n,
            };
        }
        let mut v = Vec::with_capacity(n);
        for p in parts {
            v.extend(p.dequantize());
        }
        Self::quantize(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_known_values() {
        assert_eq!(f32_to_e4m3(0.0), 0x00);
        assert_eq!(f32_to_e4m3(-0.0), 0x80);
        assert_eq!(f32_to_e4m3(1.0), 0x38);
        assert_eq!(f32_to_e4m3(-1.0), 0xb8);
        assert_eq!(f32_to_e4m3(2.0), 0x40);
        assert_eq!(f32_to_e4m3(0.5), 0x30);
        assert_eq!(f32_to_e4m3(448.0), 0x7e); // max normal
        assert_eq!(f32_to_e4m3(-448.0), 0xfe);
        assert_eq!(f32_to_e4m3(1000.0), 0x7e); // saturate
        assert_eq!(f32_to_e4m3(-1000.0), 0xfe);
        assert_eq!(f32_to_e4m3(256.0), 0x78); // exp 0b1111, mant 0 is normal
        assert_eq!(f32_to_e4m3(f32::INFINITY), 0x7f); // NaN
        assert_eq!(f32_to_e4m3(f32::NEG_INFINITY), 0xff); // NaN
        assert_eq!(f32_to_e4m3(f32::NAN), 0x7f);
        assert_eq!(f32_to_e4m3(1.0 / 64.0), 0x08); // min normal 2^-6
        assert_eq!(f32_to_e4m3(1.0 / 512.0), 0x01); // min subnormal 2^-9
        assert_eq!(f32_to_e4m3(1e-20), 0x00); // flush to zero
    }

    #[test]
    fn e4m3_dequant_known_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x80), -0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0xb8), -1.0);
        assert_eq!(e4m3_to_f32(0x40), 2.0);
        assert_eq!(e4m3_to_f32(0x30), 0.5);
        assert_eq!(e4m3_to_f32(0x7e), 448.0);
        assert_eq!(e4m3_to_f32(0xfe), -448.0);
        assert_eq!(e4m3_to_f32(0x78), 256.0);
        assert_eq!(e4m3_to_f32(0x01), 1.0 / 512.0);
        assert_eq!(e4m3_to_f32(0x08), 1.0 / 64.0);
        assert!(e4m3_to_f32(0x7f).is_nan());
        assert!(e4m3_to_f32(0xff).is_nan());
    }

    #[test]
    fn e4m3_round_trip_error_is_bounded() {
        // E4M3 relative precision ~= 2^-3 per step, so relative error of a
        // single value stays well under 8%.
        let cases = [
            0.0f32,
            1.0,
            -1.0,
            0.5,
            std::f32::consts::PI,
            -std::f32::consts::E,
            1e-2,
            123.456,
            -0.25,
            255.0,
            440.0,
            0.015625, // min normal
            0.004,    // subnormal
            -0.004,
        ];
        for c in cases {
            let rt = e4m3_to_f32(f32_to_e4m3(c));
            let err = (rt - c).abs();
            let rel = err / c.abs().max(1e-9);
            assert!(
                rel < 0.08,
                "E4M3 round-trip rel err too large for {c}: {rt} (err {err})"
            );
        }
    }

    #[test]
    fn quantize_scales_into_range() {
        let w: Vec<f32> = (0..1024)
            .map(|i| ((i as f64) * 0.13).cos() as f32 * 0.5)
            .collect();
        let t = Fp8Tensor::quantize(&w);
        assert_eq!(t.block, w.len());
        assert_eq!(t.scales.len(), 1);
        // max |w| = 0.5 -> scale = 0.5/448; dequant max ~= 0.5.
        let d = t.dequantize();
        let max_d = d.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((max_d - 0.5).abs() < 0.05, "dequant max {max_d}");
        // relative error stays small (per-tensor scale keeps values in range).
        for (&a, &b) in w.iter().zip(&d) {
            let rel = (a - b).abs() / a.abs().max(1e-9);
            assert!(rel < 0.1, "rel err {rel} for {a} -> {b}");
        }
    }

    #[test]
    fn quantize_par_matches_quantize_bitwise() {
        for n in [0usize, 1, 7, 64, 100, 1000, 65536] {
            let w: Vec<f32> = (0..n)
                .map(|i| ((i as f64) * 0.13).cos() as f32 * 100.0)
                .collect();
            let a = Fp8Tensor::quantize(&w);
            let b = Fp8Tensor::quantize_par(&w);
            assert_eq!(a.q, b.q, "bytes n={n}");
            assert_eq!(a.scales, b.scales, "scales n={n}");
            assert_eq!(a.block, b.block, "block n={n}");
            assert_eq!(a.n, b.n, "n n={n}");
        }
    }

    #[test]
    fn concat_aligned_appends_exactly() {
        for (n1, n2) in [(0usize, 64usize), (64, 128), (128, 96), (32, 1024)] {
            let a = Fp8Tensor::quantize_block(&wave(n1, 0.37), 32);
            let b = Fp8Tensor::quantize_block(&wave(n2, 0.11), 32);
            // n1 must land on a block boundary for the fast path.
            let mut w = wave(n1, 0.37);
            w.extend(wave(n2, 0.11));
            let c = a.concat(&b);
            let want = Fp8Tensor::quantize_block(&w, 32);
            assert_eq!(c.q, want.q, "bytes n1={n1} n2={n2}");
            assert_eq!(c.scales, want.scales, "scales n1={n1} n2={n2}");
            assert_eq!(c.n, n1 + n2);
        }
    }

    #[test]
    fn concat_unaligned_falls_back_to_requant() {
        let a = Fp8Tensor::quantize_block(&wave(33, 0.7), 32);
        let b = Fp8Tensor::quantize_block(&wave(64, 0.2), 32);
        let mut v = a.dequantize();
        v.extend(b.dequantize());
        let want = Fp8Tensor::quantize(&v);
        let c = a.concat(&b);
        assert_eq!(c.q, want.q);
        assert_eq!(c.scales, want.scales);
        assert_eq!(c.n, want.n);
    }

    fn wave(n: usize, k: f64) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f64) * k).sin() as f32 * 4.0)
            .collect()
    }

    /// concat_many must be byte-identical to the loader's sequential concat
    /// fold whenever every part shares its block (per-expert quantize sets
    /// `block = expert size` — the only shape real MoE checkpoints produce).
    #[test]
    fn concat_many_matches_concat_fold_when_aligned() {
        let cases: &[&[usize]] = &[&[8, 8], &[32, 64, 16], &[128], &[4, 4, 4, 4]];
        for parts in cases {
            let tensors: Vec<Fp8Tensor> = parts
                .iter()
                .map(|&n| Fp8Tensor::quantize(&wave(n, 0.07)))
                .collect();
            let folded = tensors
                .iter()
                .fold(Fp8Tensor::default(), |a, b| a.concat(b));
            let many = Fp8Tensor::concat_many(&tensors);
            assert_eq!(many.q, folded.q, "parts {parts:?}: packed bytes");
            assert_eq!(many.scales, folded.scales, "parts {parts:?}: scales");
            assert_eq!(many.block, folded.block, "parts {parts:?}: block");
            assert_eq!(many.n, folded.n, "parts {parts:?}: n");
        }
    }
}
