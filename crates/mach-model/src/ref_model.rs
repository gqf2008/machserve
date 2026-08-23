//! CPU f32 reference implementation of the slice transformer.
//!
//! Used as the golden reference for the GPU path: identical math, naive
//! loops, no SIMD. The GPU tests compare against this within tolerance.

use crate::{Config, Weights};

/// CPU reference model with an explicit KV cache.
#[derive(Debug)]
pub struct RefModel {
    cfg: Config,
    w: Weights,
    /// Per layer: (k, v), each `[n_kv_heads, max_seq_len, head_dim]`.
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// Number of tokens stored so far.
    pos: usize,
}

impl RefModel {
    /// Builds a reference model from weights.
    #[must_use]
    pub fn new(cfg: Config, w: Weights) -> Self {
        let kv_slots = cfg.n_kv_heads * cfg.max_seq_len * cfg.head_dim;
        let kv = (0..cfg.n_layers)
            .map(|_| (vec![0.0; kv_slots], vec![0.0; kv_slots]))
            .collect();
        Self { cfg, w, kv, pos: 0 }
    }

    /// Processes `tokens` one by one and returns logits of the final token.
    pub fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        let mut logits = Vec::new();
        for &t in tokens {
            logits = self.decode_step(t);
        }
        logits
    }

    /// One decode step: `token` at position `self.pos`, returns `[vocab]` logits.
    pub fn decode_step(&mut self, token: u32) -> Vec<f32> {
        let cfg = self.cfg;
        let d = cfg.d_model;
        let pos = self.pos;
        assert!(
            pos < cfg.max_seq_len,
            "sequence length exceeded max_seq_len"
        );

        let mut x = self.w.tok_emb[token as usize * d..(token as usize + 1) * d].to_vec();

        for (li, lw) in self.w.layers.iter().enumerate() {
            let xn = rms_norm(&x, &lw.rms_attn, cfg.rms_eps);
            let mut q = matvec_t(&xn, &lw.wq, cfg.n_heads * cfg.head_dim);
            let mut k = matvec_t(&xn, &lw.wk, cfg.n_kv_heads * cfg.head_dim);
            let v = matvec_t(&xn, &lw.wv, cfg.n_kv_heads * cfg.head_dim);
            // Qwen3 QK-norm: per-head RMSNorm after projection, before RoPE.
            if !lw.q_norm.is_empty() {
                qk_norm(&mut q, &lw.q_norm, cfg.n_heads, cfg.head_dim, cfg.rms_eps);
                qk_norm(
                    &mut k,
                    &lw.k_norm,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    cfg.rms_eps,
                );
            }
            apply_rope(&mut q, cfg.n_heads, cfg.head_dim, pos, cfg.rope_theta);
            apply_rope(&mut k, cfg.n_kv_heads, cfg.head_dim, pos, cfg.rope_theta);

            // Store into KV cache.
            store_row(&mut self.kv[li].0, &k, pos, cfg);
            store_row(&mut self.kv[li].1, &v, pos, cfg);

            // Attention over positions 0..=pos.
            let attn = attention_decode(&q, &self.kv[li].0, &self.kv[li].1, pos, cfg);
            let attn_proj = matvec_t(&attn, &lw.wo, d);
            for i in 0..d {
                x[i] += attn_proj[i];
            }

            let inter = cfg.intermediate_size;
            let xn2 = rms_norm(&x, &lw.rms_mlp, cfg.rms_eps);
            let moe = cfg.num_experts > 0 && !lw.moe_router.is_empty();
            if moe {
                // MoE: router softmax -> top-k experts -> weighted sum of
                // per-expert SwiGLU MLPs.
                let ne = cfg.num_experts;
                let topk = cfg.num_experts_per_tok.min(ne);
                let router = matvec_t(&xn2, &lw.moe_router, ne);
                let mut probs = vec![0.0; ne];
                let mut maxr = f32::NEG_INFINITY;
                for r in &router {
                    maxr = maxr.max(*r);
                }
                let mut sumr = 0.0f32;
                for i in 0..ne {
                    probs[i] = (router[i] - maxr).exp();
                    sumr += probs[i];
                }
                for p in probs.iter_mut() {
                    *p /= sumr;
                }
                // Top-k expert indices by probability (ties: lower index).
                let mut order: Vec<usize> = (0..ne).collect();
                order.sort_by(|&a, &b| {
                    probs[b]
                        .partial_cmp(&probs[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.cmp(&b))
                });
                let mut norm = 0.0f32;
                for &e in order.iter().take(topk) {
                    norm += probs[e];
                }
                let mut h = vec![0.0; d];
                for &e in order.iter().take(topk) {
                    // Expert e: gate/up [inter, d], down [d, inter].
                    let wg = &lw.moe_wg[e * inter * d..(e + 1) * inter * d];
                    let wu = &lw.moe_wu[e * inter * d..(e + 1) * inter * d];
                    let wd = &lw.moe_wd[e * d * inter..(e + 1) * d * inter];
                    let gate = matvec_t(&xn2, wg, inter);
                    let up = matvec_t(&xn2, wu, inter);
                    let mut eh = vec![0.0; inter];
                    for i in 0..inter {
                        eh[i] = silu(gate[i]) * up[i];
                    }
                    let down = matvec_t(&eh, wd, d);
                    let w = probs[e] / norm;
                    for i in 0..d {
                        h[i] += w * down[i];
                    }
                }
                for i in 0..d {
                    x[i] += h[i];
                }
            } else {
                let gate = matvec_t(&xn2, &lw.wg, inter);
                let up = matvec_t(&xn2, &lw.wu, inter);
                let mut h = vec![0.0; inter];
                for i in 0..inter {
                    // SwiGLU: h = silu(gate) * up.
                    h[i] = silu(gate[i]) * up[i];
                }
                let down = matvec_t(&h, &lw.wd, d);
                for i in 0..d {
                    x[i] += down[i];
                }
            }
        }

        let xf = rms_norm(&x, &self.w.rms_final, cfg.rms_eps);
        self.pos += 1;
        matvec_t(&xf, &self.w.lm_head, cfg.vocab_size)
    }
}

/// Per-head RMSNorm (Qwen3 QK-norm): each head's `head_dim` slice is
/// normalized independently, scaled by that head's weight vector.
fn qk_norm(x: &mut [f32], w: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let s = h * head_dim;
        let mut ss = 0.0;
        for &v in &x[s..s + head_dim] {
            ss += v * v;
        }
        let inv = 1.0 / (ss / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            x[s + i] = x[s + i] * inv * w[s + i];
        }
    }
}

fn apply_rope(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        for d in 0..half {
            let freq = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
            let ang = pos as f32 * freq;
            let c = ang.cos();
            let sn = ang.sin();
            // GPT-NeoX rotary: pairs (d, d + half), matching HF rotate_half.
            let idx = h * head_dim + d;
            let a = x[idx];
            let b = x[idx + half];
            x[idx] = a * c - b * sn;
            x[idx + half] = a * sn + b * c;
        }
    }
}

fn store_row(cache: &mut [f32], row: &[f32], pos: usize, cfg: Config) {
    let off = pos * cfg.n_kv_heads * cfg.head_dim;
    cache[off..off + row.len()].copy_from_slice(row);
}

#[allow(clippy::needless_range_loop)]
fn attention_decode(q: &[f32], kc: &[f32], vc: &[f32], pos: usize, cfg: Config) -> Vec<f32> {
    let hd = cfg.head_dim;
    let groups = cfg.n_heads / cfg.n_kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0; cfg.n_heads * hd];
    for h in 0..cfg.n_heads {
        let kv = h / groups;
        let qh = &q[h * hd..(h + 1) * hd];
        let mut scores = vec![0.0; pos + 1];
        let mut maxv = f32::NEG_INFINITY;
        for p in 0..=pos {
            let kp = &kc[(p * cfg.n_kv_heads + kv) * hd..(p * cfg.n_kv_heads + kv + 1) * hd];
            let mut s = 0.0;
            for dd in 0..hd {
                s += qh[dd] * kp[dd];
            }
            s *= scale;
            scores[p] = s;
            maxv = maxv.max(s);
        }
        let mut sum = 0.0;
        for p in 0..=pos {
            scores[p] = (scores[p] - maxv).exp();
            sum += scores[p];
        }
        for dd in 0..hd {
            let mut acc = 0.0;
            for p in 0..=pos {
                let vp = &vc[(p * cfg.n_kv_heads + kv) * hd + dd];
                acc += scores[p] * vp;
            }
            out[h * hd + dd] = acc / sum;
        }
    }
    out
}

fn matvec_t(x: &[f32], w: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0; out_dim];
    for o in 0..out_dim {
        let mut s = 0.0;
        for i in 0..in_dim {
            s += w[o * in_dim + i] * x[i];
        }
        out[o] = s;
    }
    out
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0.0;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    (0..n).map(|i| x[i] * inv * w[i]).collect()
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}
