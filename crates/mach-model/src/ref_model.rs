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
            let q = matvec_t(&xn, &lw.wq, cfg.n_heads * cfg.head_dim);
            let k = matvec_t(&xn, &lw.wk, cfg.n_kv_heads * cfg.head_dim);
            let v = matvec_t(&xn, &lw.wv, cfg.n_kv_heads * cfg.head_dim);

            // Store into KV cache.
            store_row(&mut self.kv[li].0, &k, pos, cfg);
            store_row(&mut self.kv[li].1, &v, pos, cfg);

            // Attention over positions 0..=pos.
            let attn = attention_decode(&q, &self.kv[li].0, &self.kv[li].1, pos, cfg);
            let attn_proj = matvec_t(&attn, &lw.wo, d);
            for i in 0..d {
                x[i] += attn_proj[i];
            }

            let xn2 = rms_norm(&x, &lw.rms_mlp, cfg.rms_eps);
            let gate = matvec_t(&xn2, &lw.wg, d);
            let up = matvec_t(&xn2, &lw.wu, d);
            let mut h = vec![0.0; d];
            for i in 0..d {
                h[i] = gate[i] * silu(up[i]);
            }
            let down = matvec_t(&h, &lw.wd, d);
            for i in 0..d {
                x[i] += down[i];
            }
        }

        let xf = rms_norm(&x, &self.w.rms_final, cfg.rms_eps);
        self.pos += 1;
        matvec_t(&xf, &self.w.lm_head, cfg.vocab_size)
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
