//! fp64 CPU reference for the MoE offload math and the transformer forward.
//!
//! Independent f64 implementation of the MoE path — matvec, expert SwiGLU
//! (gate/up/down), router softmax + top-k weighted sum, batched CPU residual —
//! and of the full non-MLA forward. It is the higher-precision golden
//! reference for the f32 CPU/GPU paths (issue #22): f32 → f64 conversion is
//! exact, so computing in f64 with the same weights exposes only the f32
//! rounding error of the CPU/GPU paths.
//!
//! This module deliberately shares **no code** with [`crate::ref_model`] /
//! [`crate::moe_offload`]: a numerical bug must not be able to appear
//! identically in both references.

use crate::{Config, LayerWeights, Weights};

/// f64 mirror of [`LayerWeights`] (standard attention + dense/MoE MLP).
/// MLA configs are rejected in [`Fp64RefModel::new`].
#[derive(Debug, Clone)]
pub struct LayerWeights64 {
    /// `[d_model, n_heads * head_dim]`
    pub wq: Vec<f64>,
    /// `[d_model, n_kv_heads * head_dim]`
    pub wk: Vec<f64>,
    /// `[d_model, n_kv_heads * head_dim]`
    pub wv: Vec<f64>,
    /// `[n_heads * head_dim, d_model]`
    pub wo: Vec<f64>,
    /// `[d_model]`
    pub rms_attn: Vec<f64>,
    /// `[intermediate_size, d_model]`
    pub wg: Vec<f64>,
    /// `[intermediate_size, d_model]`
    pub wu: Vec<f64>,
    /// `[d_model, intermediate_size]`
    pub wd: Vec<f64>,
    /// `[d_model]`
    pub rms_mlp: Vec<f64>,
    /// Per-head QK-norm weight `[n_heads * head_dim]`; empty when disabled.
    pub q_norm: Vec<f64>,
    /// Per-head QK-norm weight `[n_kv_heads * head_dim]`; empty when disabled.
    pub k_norm: Vec<f64>,
    /// MoE router `[num_experts, d_model]`; empty for dense layers.
    pub moe_router: Vec<f64>,
    /// Per-expert gate/up `[num_experts, intermediate_size, d_model]`.
    pub moe_wg: Vec<f64>,
    pub moe_wu: Vec<f64>,
    /// Per-expert down `[num_experts, d_model, intermediate_size]`.
    pub moe_wd: Vec<f64>,
}

impl From<&LayerWeights> for LayerWeights64 {
    fn from(w: &LayerWeights) -> Self {
        Self {
            wq: to_f64(&w.wq),
            wk: to_f64(&w.wk),
            wv: to_f64(&w.wv),
            wo: to_f64(&w.wo),
            rms_attn: to_f64(&w.rms_attn),
            wg: to_f64(&w.wg),
            wu: to_f64(&w.wu),
            wd: to_f64(&w.wd),
            rms_mlp: to_f64(&w.rms_mlp),
            q_norm: to_f64(&w.q_norm),
            k_norm: to_f64(&w.k_norm),
            moe_router: to_f64(&w.moe_router),
            moe_wg: to_f64(&w.moe_wg),
            moe_wu: to_f64(&w.moe_wu),
            moe_wd: to_f64(&w.moe_wd),
        }
    }
}

/// f64 mirror of [`Weights`].
#[derive(Debug, Clone)]
pub struct Weights64 {
    /// `[vocab_size, d_model]`
    pub tok_emb: Vec<f64>,
    /// `[d_model]`
    pub rms_final: Vec<f64>,
    /// `[vocab_size, d_model]`
    pub lm_head: Vec<f64>,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights64>,
}

impl From<&Weights> for Weights64 {
    fn from(w: &Weights) -> Self {
        Self {
            tok_emb: to_f64(&w.tok_emb),
            rms_final: to_f64(&w.rms_final),
            lm_head: to_f64(&w.lm_head),
            layers: w.layers.iter().map(LayerWeights64::from).collect(),
        }
    }
}

/// Exact f32 → f64 conversion (every f32 is representable in f64).
#[must_use]
pub fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// Row-major `[out, in]` matrix times a vector, returning `out`. Independent
/// f64 implementation (the f32 `ref_model::matvec_t` / `moe_offload::matvec_t`
/// are intentionally not reused).
#[must_use]
pub fn matvec_t(x: &[f64], w: &[f64], out_dim: usize) -> Vec<f64> {
    let in_dim = x.len();
    let mut out = vec![0.0; out_dim];
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0.0;
        for (k, &xk) in x.iter().enumerate() {
            acc += xk * row[k];
        }
        out[o] = acc;
    }
    out
}

#[must_use]
pub fn rms_norm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len();
    let mut ss = 0.0;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f64 + eps).sqrt();
    (0..n).map(|i| x[i] * inv * w[i]).collect()
}

/// Per-head RMSNorm (Qwen3 QK-norm): each head's `head_dim` slice is
/// normalized independently, scaled by that head's weight vector.
fn qk_norm(x: &mut [f64], w: &[f64], n_heads: usize, head_dim: usize, eps: f64) {
    for h in 0..n_heads {
        let s = h * head_dim;
        let mut ss = 0.0;
        for &v in &x[s..s + head_dim] {
            ss += v * v;
        }
        let inv = 1.0 / (ss / head_dim as f64 + eps).sqrt();
        for i in 0..head_dim {
            x[s + i] *= inv * w[s + i];
        }
    }
}

/// GPT-NeoX rotary embedding (pairs `(d, d + half)`), matching the f32 paths.
fn apply_rope(x: &mut [f64], n_heads: usize, head_dim: usize, pos: usize, theta: f64) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        for d in 0..half {
            let freq = 1.0 / theta.powf(2.0 * d as f64 / head_dim as f64);
            let ang = pos as f64 * freq;
            let (sn, c) = ang.sin_cos();
            let idx = h * head_dim + d;
            let a = x[idx];
            let b = x[idx + half];
            x[idx] = a * c - b * sn;
            x[idx + half] = a * sn + b * c;
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn attention_decode(q: &[f64], kc: &[f64], vc: &[f64], pos: usize, cfg: Config) -> Vec<f64> {
    let hd = cfg.head_dim;
    let groups = cfg.n_heads / cfg.n_kv_heads;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut out = vec![0.0; cfg.n_heads * hd];
    for h in 0..cfg.n_heads {
        let kv = h / groups;
        let qh = &q[h * hd..(h + 1) * hd];
        let mut scores = vec![0.0; pos + 1];
        let mut maxv = f64::NEG_INFINITY;
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
                acc += scores[p] * vc[(p * cfg.n_kv_heads + kv) * hd + dd];
            }
            out[h * hd + dd] = acc / sum;
        }
    }
    out
}

#[must_use]
pub fn silu(v: f64) -> f64 {
    v / (1.0 + (-v).exp())
}

/// f64 SwiGLU MLP for one expert (`gate/up [inter, d]`, `down [d, inter]`),
/// returning `[d]`. Mirrors `moe_offload::expert_mlp` in f64.
#[must_use]
pub fn expert_mlp(
    xn: &[f64],
    wg: &[f64],
    wu: &[f64],
    wd: &[f64],
    inter: usize,
    d: usize,
) -> Vec<f64> {
    let gate = matvec_t(xn, wg, inter);
    let up = matvec_t(xn, wu, inter);
    let mut eh = vec![0.0; inter];
    for i in 0..inter {
        eh[i] = silu(gate[i]) * up[i];
    }
    matvec_t(&eh, wd, d)
}

/// f64 MoE routing for one row: router logits -> softmax -> top-k (ties
/// resolved by lower index, matching the f32 paths) -> normalized weights.
/// Returns `(ids, weights)`, each `[topk]`, where `weights` sums to 1.
#[must_use]
pub fn moe_route(xn: &[f64], lw: &LayerWeights64, cfg: &Config) -> (Vec<i32>, Vec<f64>) {
    let ne = cfg.num_experts;
    let topk = cfg.num_experts_per_tok.min(ne);
    let router = matvec_t(xn, &lw.moe_router, ne);
    let mut maxr = f64::NEG_INFINITY;
    for r in &router {
        maxr = maxr.max(*r);
    }
    let mut probs = vec![0.0; ne];
    let mut sumr = 0.0;
    for i in 0..ne {
        probs[i] = (router[i] - maxr).exp();
        sumr += probs[i];
    }
    for p in &mut probs {
        *p /= sumr;
    }
    let mut order: Vec<usize> = (0..ne).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let chosen: Vec<usize> = order.into_iter().take(topk).collect();
    let mut norm = 0.0;
    for &e in &chosen {
        norm += probs[e];
    }
    let ids: Vec<i32> = chosen.iter().map(|&e| e as i32).collect();
    let weights: Vec<f64> = chosen.iter().map(|&e| probs[e] / norm).collect();
    (ids, weights)
}

/// f64 MoE layer residual for one row: router softmax -> top-k -> weighted
/// sum of per-expert SwiGLU MLPs. Returns `[d]`. This is the fp64 counterpart
/// of `MoeOffload::step` / the MoE section of `ref_model::RefModel`.
#[must_use]
pub fn moe_layer(xn: &[f64], lw: &LayerWeights64, cfg: &Config) -> Vec<f64> {
    let d = cfg.d_model;
    let inter = cfg.expert_size();
    let (ids, weights) = moe_route(xn, lw, cfg);
    let mut residual = vec![0.0; d];
    for (j, e) in ids.iter().enumerate() {
        let e = *e as usize;
        let w = weights[j];
        let wg = &lw.moe_wg[e * inter * d..(e + 1) * inter * d];
        let wu = &lw.moe_wu[e * inter * d..(e + 1) * inter * d];
        let wd = &lw.moe_wd[e * d * inter..(e + 1) * d * inter];
        let down = expert_mlp(xn, wg, wu, wd, inter, d);
        for kk in 0..d {
            residual[kk] += w * down[kk];
        }
    }
    residual
}

/// f64 batched MoE residual for `[b, d]` rows given precomputed routed
/// `ids`/`weights` (same contract as `moe_offload::moe_batch_cpu_residual`,
/// but with f64 weights). Returns `[b, d]`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn moe_batch_cpu_residual(
    ids: &[i32],
    weights: &[f64],
    xn2: &[f64],
    lw: &LayerWeights64,
    b: usize,
    d: usize,
    inter: usize,
    topk: usize,
) -> Vec<f64> {
    let mut residual = vec![0.0; b * d];
    for t in 0..b {
        let row = &xn2[t * d..(t + 1) * d];
        for j in 0..topk {
            let e = ids[t * topk + j] as usize;
            let w = weights[t * topk + j];
            let wg = &lw.moe_wg[e * inter * d..(e + 1) * inter * d];
            let wu = &lw.moe_wu[e * inter * d..(e + 1) * inter * d];
            let wd = &lw.moe_wd[e * d * inter..(e + 1) * d * inter];
            let down = expert_mlp(row, wg, wu, wd, inter, d);
            for kk in 0..d {
                residual[t * d + kk] += w * down[kk];
            }
        }
    }
    residual
}

/// fp64 reference model with an explicit KV cache: the f64 counterpart of
/// [`crate::ref_model::RefModel`]. Standard attention + GQA + optional
/// QK-norm + dense/MoE MLP. MLA is not implemented (asserts on construction).
#[derive(Debug)]
pub struct Fp64RefModel {
    cfg: Config,
    w: Weights64,
    /// Per layer: (k, v), each `[n_kv_heads, max_seq_len, head_dim]`.
    kv: Vec<(Vec<f64>, Vec<f64>)>,
    /// Number of tokens stored so far.
    pos: usize,
}

impl Fp64RefModel {
    /// Builds the fp64 reference from f32 weights (converted exactly).
    #[must_use]
    pub fn new(cfg: Config, w: &Weights) -> Self {
        assert!(
            cfg.kv_lora_rank == 0,
            "fp64 reference does not implement MLA (kv_lora_rank > 0)"
        );
        let kv_slots = cfg.n_kv_heads * cfg.max_seq_len * cfg.head_dim;
        let kv = (0..cfg.n_layers)
            .map(|_| (vec![0.0; kv_slots], vec![0.0; kv_slots]))
            .collect();
        Self {
            cfg,
            w: Weights64::from(w),
            kv,
            pos: 0,
        }
    }

    /// Number of tokens processed so far (next position).
    #[must_use]
    pub const fn pos(&self) -> usize {
        self.pos
    }

    /// Processes `tokens` one by one and returns fp64 logits of the final token.
    pub fn forward(&mut self, tokens: &[u32]) -> Vec<f64> {
        let mut logits = Vec::new();
        for &t in tokens {
            logits = self.decode_step(t);
        }
        logits
    }

    /// One decode step: `token` at position `self.pos`, returns `[vocab]` logits.
    pub fn decode_step(&mut self, token: u32) -> Vec<f64> {
        let d = self.cfg.d_model;
        let x0 = self.w.tok_emb[token as usize * d..(token as usize + 1) * d].to_vec();
        self.forward_from(x0)
    }

    /// Runs one transformer position starting from an input hidden state,
    /// storing KV at the current position and returning `[vocab]` logits.
    fn forward_from(&mut self, mut x: Vec<f64>) -> Vec<f64> {
        let cfg = self.cfg;
        let d = cfg.d_model;
        let pos = self.pos;
        assert!(
            pos < cfg.max_seq_len,
            "sequence length exceeded max_seq_len"
        );
        let eps = cfg.rms_eps as f64;
        let theta = cfg.rope_theta as f64;

        for (li, lw) in self.w.layers.iter().enumerate() {
            let xn = rms_norm(&x, &lw.rms_attn, eps);
            let mut q = matvec_t(&xn, &lw.wq, cfg.n_heads * cfg.head_dim);
            let mut k = matvec_t(&xn, &lw.wk, cfg.n_kv_heads * cfg.head_dim);
            let v = matvec_t(&xn, &lw.wv, cfg.n_kv_heads * cfg.head_dim);
            if !lw.q_norm.is_empty() {
                qk_norm(&mut q, &lw.q_norm, cfg.n_heads, cfg.head_dim, eps);
                qk_norm(&mut k, &lw.k_norm, cfg.n_kv_heads, cfg.head_dim, eps);
            }
            apply_rope(&mut q, cfg.n_heads, cfg.head_dim, pos, theta);
            apply_rope(&mut k, cfg.n_kv_heads, cfg.head_dim, pos, theta);

            let row = cfg.n_kv_heads * cfg.head_dim;
            let off = pos * row;
            self.kv[li].0[off..off + row].copy_from_slice(&k);
            self.kv[li].1[off..off + row].copy_from_slice(&v);

            let attn = attention_decode(&q, &self.kv[li].0, &self.kv[li].1, pos, cfg);
            let attn_proj = matvec_t(&attn, &lw.wo, d);
            for i in 0..d {
                x[i] += attn_proj[i];
            }

            let inter = cfg.intermediate_size;
            let xn2 = rms_norm(&x, &lw.rms_mlp, eps);
            let moe = cfg.num_experts > 0 && !lw.moe_router.is_empty();
            if moe {
                let h = moe_layer(&xn2, lw, &cfg);
                for i in 0..d {
                    x[i] += h[i];
                }
            } else {
                let gate = matvec_t(&xn2, &lw.wg, inter);
                let up = matvec_t(&xn2, &lw.wu, inter);
                let mut h = vec![0.0; inter];
                for i in 0..inter {
                    h[i] = silu(gate[i]) * up[i];
                }
                let down = matvec_t(&h, &lw.wd, d);
                for i in 0..d {
                    x[i] += down[i];
                }
            }
        }

        let xf = rms_norm(&x, &self.w.rms_final, eps);
        self.pos += 1;
        matvec_t(&xf, &self.w.lm_head, cfg.vocab_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe_offload::{self, MoeOffload};
    use crate::ref_model::RefModel;

    /// Shapes for the MoE-scope parity tests: (d_model, intermediate, experts, topk).
    const MOE_SHAPES: &[(usize, usize, usize, usize)] = &[
        (128, 256, 8, 2),
        (128, 128, 16, 4),
        (256, 512, 4, 2),
        (64, 128, 8, 3),
    ];
    const SEEDS: &[u64] = &[1, 7, 42, 2024];

    fn moe_cfg(d: usize, inter: usize, ne: usize, topk: usize) -> Config {
        let mut c = Config::tiny();
        c.d_model = d;
        c.head_dim = d / c.n_heads;
        c.intermediate_size = inter;
        c.num_experts = ne;
        c.num_experts_per_tok = topk.min(ne);
        c
    }

    fn xn_for(d: usize, seed: u64) -> Vec<f32> {
        (0..d)
            .map(|i| (i as f32) * 0.017 + (seed as f32) * 0.011 - 1.5)
            .collect()
    }

    fn max_abs(a: &[f64]) -> f64 {
        a.iter().fold(0.0f64, |m, &v| m.max(v.abs()))
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    fn argmax(a: &[f64]) -> usize {
        let mut best = 0;
        for (i, &v) in a.iter().enumerate() {
            if v > a[best] {
                best = i;
            }
        }
        best
    }

    /// f32 rounding tolerance for a chain of `dots` length-`n` dot products.
    /// Each f32 dot product has typical relative error ~sqrt(n)*eps32 (and a
    /// worst case of n*eps32); 8x margin over the typical-error estimate.
    fn moe_tol(n: usize, dots: usize, scale: f64) -> f64 {
        let eps = f64::from(f32::EPSILON);
        8.0 * dots as f64 * (n as f64).sqrt() * eps * scale + eps
    }

    /// f32 router (independent copy for routing-stability diagnostics).
    fn f32_route(cfg: &Config, lw: &LayerWeights, xn: &[f32]) -> (Vec<i32>, Vec<f32>) {
        let ne = cfg.num_experts;
        let topk = cfg.num_experts_per_tok.min(ne);
        let d = cfg.d_model;
        let mut router = vec![0.0f32; ne];
        for (o, r) in router.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for (k, &xk) in xn.iter().enumerate() {
                s += xk * lw.moe_router[o * d + k];
            }
            *r = s;
        }
        let maxr = router.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0.0f32; ne];
        let mut sumr = 0.0f32;
        for i in 0..ne {
            probs[i] = (router[i] - maxr).exp();
            sumr += probs[i];
        }
        for p in &mut probs {
            *p /= sumr;
        }
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let chosen: Vec<usize> = order.into_iter().take(topk).collect();
        let mut norm = 0.0f32;
        for &e in &chosen {
            norm += probs[e];
        }
        let ids: Vec<i32> = chosen.iter().map(|&e| e as i32).collect();
        let weights: Vec<f32> = chosen.iter().map(|&e| probs[e] / norm).collect();
        (ids, weights)
    }

    #[test]
    fn expert_mlp_f64_matches_f32_path() {
        for &(d, inter, ne, _) in MOE_SHAPES {
            for &seed in SEEDS {
                let cfg = moe_cfg(d, inter, ne, 2);
                let w = Weights::random(&cfg, seed).unwrap();
                let lw = &w.layers[0];
                let xn = xn_for(d, seed);
                let wg = &lw.moe_wg[0..inter * d];
                let wu = &lw.moe_wu[0..inter * d];
                let wd = &lw.moe_wd[0..d * inter];
                let f32_out = moe_offload::expert_mlp(&xn, wg, wu, wd, inter, d);
                let lw64 = LayerWeights64::from(lw);
                let f64_out = expert_mlp(
                    &to_f64(&xn),
                    &lw64.moe_wg[0..inter * d],
                    &lw64.moe_wu[0..inter * d],
                    &lw64.moe_wd[0..d * inter],
                    inter,
                    d,
                );
                let scale = max_abs(&to_f64(&f32_out));
                let diff = max_abs_diff(&to_f64(&f32_out), &f64_out);
                let tol = moe_tol(d.max(inter), 3, scale);
                eprintln!(
                    "expert_mlp f32-vs-f64: d={d} inter={inter} seed={seed} max_diff={diff:.3e} tol={tol:.3e} scale={scale:.3e}"
                );
                assert!(
                    diff <= tol,
                    "expert_mlp f32 vs f64: d={d} inter={inter} seed={seed} max_diff={diff:.3e} tol={tol:.3e} scale={scale:.3e}"
                );
            }
        }
    }

    #[test]
    fn moe_layer_f64_matches_offload_step() {
        for &(d, inter, ne, topk) in MOE_SHAPES {
            for &seed in SEEDS {
                let cfg = moe_cfg(d, inter, ne, topk);
                let w = Weights::random(&cfg, seed).unwrap();
                let lw = &w.layers[0];
                let xn = xn_for(d, seed);

                let mut moe = MoeOffload::new(ne);
                let out = moe.step(&cfg, lw, &xn, usize::MAX);
                let lw64 = LayerWeights64::from(lw);
                let f64_out = moe_layer(&to_f64(&xn), &lw64, &cfg);

                // Routing must be identical: fp64 top-k == f32 top-k (near-tie
                // experts would show up here as a routing instability, which is
                // itself a finding).
                let (f32_ids, _) = f32_route(&cfg, lw, &xn);
                let (f64_ids, _) = moe_route(&to_f64(&xn), &lw64, &cfg);
                assert_eq!(
                    f32_ids, f64_ids,
                    "router top-k differs f32 vs f64: d={d} inter={inter} ne={ne} topk={topk} seed={seed}"
                );

                let scale = max_abs(&to_f64(&out.residual));
                let diff = max_abs_diff(&to_f64(&out.residual), &f64_out);
                let tol = moe_tol(d.max(inter), 3 * topk + 1, scale);
                eprintln!(
                    "moe_layer f32-vs-f64: d={d} inter={inter} ne={ne} topk={topk} seed={seed} max_diff={diff:.3e} tol={tol:.3e} scale={scale:.3e}"
                );
                assert!(
                    diff <= tol,
                    "moe_layer f32 step vs f64: d={d} inter={inter} ne={ne} topk={topk} seed={seed} max_diff={diff:.3e} tol={tol:.3e} scale={scale:.3e}"
                );
            }
        }
    }

    #[test]
    fn moe_batch_cpu_residual_f64_matches_f32() {
        for &(d, inter, ne, topk) in MOE_SHAPES {
            for &seed in SEEDS {
                let cfg = moe_cfg(d, inter, ne, topk);
                let w = Weights::random(&cfg, seed).unwrap();
                let lw = &w.layers[0];
                let lw64 = LayerWeights64::from(lw);
                let b = 2usize;
                let xn2: Vec<f32> = (0..b * d)
                    .map(|i| (i as f32) * 0.013 + (seed as f32) * 0.007 - 1.2)
                    .collect();

                // Same routed ids/weights for both functions (derived from fp64
                // routing, then weights cast to f32 for the f32 function).
                let mut ids = Vec::new();
                let mut w_f64 = Vec::new();
                for t in 0..b {
                    let row = &xn2[t * d..(t + 1) * d];
                    let (r_ids, r_w) = moe_route(&to_f64(row), &lw64, &cfg);
                    ids.extend(r_ids);
                    w_f64.extend(r_w);
                }
                let w_f32: Vec<f32> = w_f64.iter().map(|&x| x as f32).collect();

                let f32_res =
                    moe_offload::moe_batch_cpu_residual(&ids, &w_f32, &xn2, lw, b, d, inter, topk);
                let f64_res =
                    moe_batch_cpu_residual(&ids, &w_f64, &to_f64(&xn2), &lw64, b, d, inter, topk);

                let scale = max_abs(&to_f64(&f32_res));
                let diff = max_abs_diff(&to_f64(&f32_res), &f64_res);
                let tol = moe_tol(d.max(inter), 3 * topk + 1, scale);
                eprintln!(
                    "batch residual f32-vs-f64: d={d} inter={inter} ne={ne} topk={topk} seed={seed} max_diff={diff:.3e} tol={tol:.3e} scale={scale:.3e}"
                );
                assert!(
                    diff <= tol,
                    "batch residual f32 vs f64: d={d} inter={inter} ne={ne} topk={topk} seed={seed} max_diff={diff:.3e} tol={tol:.3e} scale={scale:.3e}"
                );
            }
        }
    }

    /// The fp64 residual must sit within f32 rounding of both f32 placements
    /// (resident vs CPU-overflow): placement invariance also holds in fp64.
    #[test]
    fn fp64_matches_both_f32_placements() {
        let (d, inter, ne, topk) = (128usize, 256usize, 8usize, 2usize);
        for &seed in SEEDS {
            let cfg = moe_cfg(d, inter, ne, topk);
            let w = Weights::random(&cfg, seed).unwrap();
            let lw = &w.layers[0];
            let xn = xn_for(d, seed);
            let lw64 = LayerWeights64::from(lw);
            let f64_out = moe_layer(&to_f64(&xn), &lw64, &cfg);

            let mut resident = MoeOffload::new(ne);
            let r = resident.step(&cfg, lw, &xn, usize::MAX);
            let mut overflow = MoeOffload::new(1);
            let o = overflow.step(&cfg, lw, &xn, usize::MAX);
            assert_eq!(r.residual, o.residual, "f32 placement invariance");

            for (name, f32_out) in [("resident", &r.residual), ("cpu-overflow", &o.residual)] {
                let scale = max_abs(&to_f64(f32_out));
                let diff = max_abs_diff(&to_f64(f32_out), &f64_out);
                let tol = moe_tol(d.max(inter), 3 * topk + 1, scale);
                eprintln!(
                    "placement fp64-vs-f32-{name}: seed={seed} max_diff={diff:.3e} tol={tol:.3e} scale={scale:.3e}"
                );
                assert!(
                    diff <= tol,
                    "fp64 vs f32 {name} placement: seed={seed} max_diff={diff:.3e} tol={tol:.3e} scale={scale:.3e}"
                );
            }
        }
    }

    #[test]
    fn fp64_forward_matches_f32_ref() {
        // Two configs: tiny MoE and a slightly larger Qwen3-style MoE with
        // QK-norm (qk_norm=true), several token sequences per seed.
        let mut cfgs = Vec::new();
        for &(d, inter, ne, topk) in MOE_SHAPES {
            cfgs.push(moe_cfg(d, inter, ne, topk));
        }
        let mut qwen = Config::qwen3(256, 2, 8, 4, 1024, 64);
        qwen.num_experts = 8;
        qwen.num_experts_per_tok = 2;
        qwen.moe_intermediate_size = 128;
        cfgs.push(qwen);

        for cfg in &cfgs {
            for &seed in SEEDS {
                let w = Weights::random(cfg, seed).unwrap();
                let tokens = [seed as u32 % 512, 7, 42, 300, 11, 2024 % 512];
                let mut f32_m = RefModel::new(*cfg, w.clone());
                let f32_logits = f32_m.forward(&tokens);
                let mut f64_m = Fp64RefModel::new(*cfg, &w);
                let f64_logits = f64_m.forward(&tokens);

                let scale = max_abs(&to_f64(&f32_logits));
                let diff = max_abs_diff(&to_f64(&f32_logits), &f64_logits);
                let tol = forward_tol(cfg, scale);
                assert!(
                    diff <= tol,
                    "forward f32 vs f64: d={} ne={} topk={} qk_norm={} seed={seed} max_diff={diff:.3e} tol={tol:.3e} scale={scale:.3e}",
                    cfg.d_model,
                    cfg.num_experts,
                    cfg.num_experts_per_tok,
                    cfg.qk_norm,
                );
                let (am32, am64) = (argmax(&to_f64(&f32_logits)), argmax(&f64_logits));
                eprintln!(
                    "forward f32-vs-f64: d={} ne={} topk={} qk_norm={} seed={seed} max_diff={diff:.3e} tol={tol:.3e} scale={scale:.3e} argmax=({am32},{am64})",
                    cfg.d_model, cfg.num_experts, cfg.num_experts_per_tok, cfg.qk_norm
                );
                assert_eq!(
                    am32, am64,
                    "forward argmax f32 vs f64: d={} ne={} topk={} qk_norm={} seed={seed}",
                    cfg.d_model, cfg.num_experts, cfg.num_experts_per_tok, cfg.qk_norm,
                );
            }
        }
    }

    /// Accumulated f32 rounding bound for a full forward: `layers` layers x
    /// ~7 chained dot products (attention q/k/v/o + MLP g/u/d), each with
    /// typical relative error sqrt(n)*eps32; 8x margin.
    fn forward_tol(cfg: &Config, scale: f64) -> f64 {
        let eps = f64::from(f32::EPSILON);
        let n = cfg.d_model.max(cfg.expert_size()) as f64;
        let dots_per_layer = 7.0;
        let layers = cfg.n_layers as f64;
        8.0 * dots_per_layer * layers * n.sqrt() * eps * scale + eps
    }
}
