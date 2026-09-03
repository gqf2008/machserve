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

    /// Parallel quantize: groups are independent (disjoint byte ranges), so the
    /// work is split across threads and the per-thread packed outputs are
    /// concatenated in group order. Bit-identical to [`Self::quantize`]; used
    /// by the Q4 loader so large single tensors (embedding/lm_head) do not
    /// serialize the whole load.
    #[allow(clippy::needless_range_loop)]
    pub fn quantize_par(w: &[f32]) -> Self {
        let n = w.len();
        let groups = n.div_ceil(Q4_GROUP);
        let n_threads = std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(4)
            .min(16);
        let per_thread = groups.div_ceil(n_threads).max(1);
        let mut q = Vec::with_capacity(n.div_ceil(2));
        let mut scales = Vec::with_capacity(groups);
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_threads);
            for t in 0..n_threads {
                let g0 = t * per_thread;
                let g1 = (g0 + per_thread).min(groups);
                if g0 >= g1 {
                    continue;
                }
                handles.push(s.spawn(move || {
                    let span_start = g0 * Q4_GROUP;
                    let span_end = (g1 * Q4_GROUP).min(n);
                    // Only the final partial group is padded differently from
                    // `quantize`: size by the actual element span, so the
                    // merged byte array is bit-identical to the scalar path.
                    let mut tq = vec![0u8; (span_end - span_start).div_ceil(2)];
                    let mut ts = vec![0f32; g1 - g0];
                    for g in g0..g1 {
                        let start = g * Q4_GROUP;
                        let end = (start + Q4_GROUP).min(n);
                        let mut max_abs = 0f32;
                        for &v in &w[start..end] {
                            max_abs = max_abs.max(v.abs());
                        }
                        let scale = max_abs / 7.0;
                        ts[g - g0] = scale;
                        for i in start..end {
                            let qi = if scale > 0.0 {
                                (w[i] / scale).round().clamp(-7.0, 7.0) as i8
                            } else {
                                0i8
                            };
                            let byte = (i - span_start) / 2;
                            let nib = (qi as u8 & 0x0F)
                                << if (i - span_start).is_multiple_of(2) {
                                    0
                                } else {
                                    4
                                };
                            tq[byte] |= nib;
                        }
                    }
                    (tq, ts)
                }));
            }
            for h in handles {
                let (tq, ts) = h.join().unwrap();
                q.extend_from_slice(&tq);
                scales.extend_from_slice(&ts);
            }
        });
        Self { q, scales, n }
    }

    /// Raw packed int4 bytes (2 elements per byte, low nibble first).
    pub fn q_bytes(&self) -> &[u8] {
        &self.q
    }

    /// Per-group f32 scales (`groups = n.div_ceil(Q4_GROUP)`), flat-aligned.
    pub fn scales(&self) -> &[f32] {
        &self.scales
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

    /// Dequantizes directly to f16 bit patterns (for device upload), without
    #[allow(clippy::needless_range_loop)]
    /// materializing the full f32 vector (transient = 2 bytes/element).
    pub fn dequantize_f16(&self) -> Vec<u16> {
        let mut out = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let byte = self.q[i / 2];
            let nib = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            let g = i / Q4_GROUP;
            let v = signed_nibble(nib) as f32 * self.scales[g];
            out.push(crate::fp16::f32_to_f16(v));
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
    /// Concatenates two quantized tensors (per-expert MoE tensors).
    ///
    /// When `self` ends on a group boundary the packed int4 bytes and
    /// per-group scales concatenate directly: exact, O(1) per append instead
    /// of the loader's old dequantize+requantize (O(n) per expert, O(ne^2)
    /// across ne experts). Unaligned tensors fall back to dequantize +
    /// requantize, preserving the historical result.
    pub fn concat(&self, other: &Self) -> Self {
        if self.n.is_multiple_of(Q4_GROUP) {
            let mut q = self.q.clone();
            q.extend_from_slice(&other.q);
            let mut scales = self.scales.clone();
            scales.extend_from_slice(&other.scales);
            return Self {
                q,
                scales,
                n: self.n + other.n,
            };
        }
        let mut v = self.dequantize();
        v.extend(other.dequantize());
        Self::quantize(&v)
    }

    /// Byte/scale-identical row-block split of a `[rows, kk]` quantized matrix.
    ///
    /// `blocks` are `(start_row, n_rows)` ranges; each piece keeps the packed
    /// bytes and scales of exactly its own rows, so nothing is requantized.
    /// Requires `kk` to be a multiple of [`Q4_GROUP`] (and therefore each
    /// block's element count to be too), which holds for every real fused-MLA
    /// projection (`kk` = `d_model` or `q_lora_rank`, both 32-aligned).
    ///
    /// Used to break a fused `q_proj`/`q_b_proj` `[heads*(nope+rope), kk]` into
    /// the per-head non-RoPE and RoPE halves the runtime keeps separate.
    #[must_use]
    pub fn split_row_blocks(&self, kk: usize, blocks: &[(usize, usize)]) -> Vec<Self> {
        assert!(
            kk.is_multiple_of(Q4_GROUP),
            "q4 row split: kk={kk} is not a multiple of Q4_GROUP"
        );
        let g_per_row = kk / Q4_GROUP;
        blocks
            .iter()
            .map(|&(r0, nr)| {
                let i0 = r0 * kk;
                let i1 = i0 + nr * kk;
                assert!(
                    i1 <= self.n,
                    "q4 row split: rows {r0}..{} out of range",
                    r0 + nr
                );
                let g0 = r0 * g_per_row;
                let g1 = (r0 + nr) * g_per_row;
                Self {
                    q: self.q[i0 / 2..i1 / 2].to_vec(),
                    scales: self.scales[g0..g1].to_vec(),
                    n: nr * kk,
                }
            })
            .collect()
    }

    /// Single-pass append of `parts` in order, O(total) bytes — folding
    /// [`Self::concat`] instead re-clones the growing prefix per part
    /// (O(n²) over many MoE experts). Byte/scale-identical to that fold
    /// **iff every part is group-aligned** — the only shape real
    /// checkpoints hit (expert sizes are [`Q4_GROUP`] multiples);
    /// otherwise both re-quantize, but scale grouping can differ from the
    /// sequential fold (see the `concat_many_matches_concat_fold` test).
    pub fn concat_many(parts: &[Self]) -> Self {
        if parts.is_empty() {
            return Self::default();
        }
        let n: usize = parts.iter().map(|p| p.n).sum();
        if parts.iter().all(|p| p.n.is_multiple_of(Q4_GROUP)) {
            let mut q = Vec::with_capacity(n / 2);
            let mut scales = Vec::with_capacity(n / Q4_GROUP);
            for p in parts {
                q.extend_from_slice(&p.q);
                scales.extend_from_slice(&p.scales);
            }
            return Self { q, scales, n };
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
    fn wave(n: usize, k: f64) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f64) * k).sin() as f32 * 4.0)
            .collect()
    }
    fn assert_same(c: &Q4Tensor, w: &[f32]) {
        let want = Q4Tensor::quantize(w);
        assert_eq!(c.q, want.q, "packed bytes");
        assert_eq!(c.scales, want.scales, "scales");
        assert_eq!(c.n, want.n, "n");
    }
    #[test]
    fn concat_group_aligned_is_bitwise_quantize_of_concat() {
        for (n1, n2) in [
            (0usize, 64usize),
            (64, 128),
            (128, 96),
            (32, 1024),
            (1024, 32),
        ] {
            let a = Q4Tensor::quantize(&wave(n1, 0.37));
            let b = Q4Tensor::quantize(&wave(n2, 0.11));
            let mut w = wave(n1, 0.37);
            w.extend(wave(n2, 0.11));
            assert_same(&a.concat(&b), &w);
        }
    }
    #[test]
    fn quantize_par_matches_quantize_bitwise() {
        for n in [0usize, 1, 31, 32, 64, 100, 1000, 65536] {
            let w: Vec<f32> = (0..n)
                .map(|i| ((i as f64) * 0.13).cos() as f32 * 7.0)
                .collect();
            let a = Q4Tensor::quantize(&w);
            let b = Q4Tensor::quantize_par(&w);
            assert_eq!(a.q, b.q, "bytes n={n}");
            assert_eq!(a.scales, b.scales, "scales n={n}");
            assert_eq!(a.n, b.n, "n n={n}");
        }
    }

    #[test]
    fn concat_unaligned_falls_back_to_requant() {
        let a = Q4Tensor::quantize(&wave(33, 0.7));
        let b = Q4Tensor::quantize(&wave(64, 0.2));
        let mut v = a.dequantize();
        v.extend(b.dequantize());
        let want = Q4Tensor::quantize(&v);
        let c = a.concat(&b);
        assert_eq!(c.q, want.q);
        assert_eq!(c.scales, want.scales);
        assert_eq!(c.n, want.n);
    }

    /// concat_many must be byte-identical to the loader's sequential concat
    /// fold whenever every part is group-aligned — the only shape real MoE
    /// checkpoints produce (expert sizes are Q4_GROUP multiples).
    #[test]
    fn concat_many_matches_concat_fold_when_aligned() {
        let cases: &[&[usize]] = &[&[8, 8], &[32, 64, 16], &[128], &[4, 4, 4, 4]];
        for parts in cases {
            let tensors: Vec<Q4Tensor> = parts
                .iter()
                .map(|&n| Q4Tensor::quantize(&wave(n, 0.07)))
                .collect();
            let folded = tensors.iter().fold(Q4Tensor::default(), |a, b| a.concat(b));
            let many = Q4Tensor::concat_many(&tensors);
            assert_eq!(many.q, folded.q, "parts {parts:?}: packed bytes");
            assert_eq!(many.scales, folded.scales, "parts {parts:?}: scales");
            assert_eq!(many.n, folded.n, "parts {parts:?}: n");
        }
    }
}
