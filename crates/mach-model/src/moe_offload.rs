//! Host-side MoE offload orchestration.
//!
//! Combines [`LruExpertCache`] with the per-expert SwiGLU MLP math from the CPU
//! reference. For a decode step it produces the MoE residual (what the reference
//! adds to `x`), and drives the offload decision through the LRU cache so the
//! caller knows which experts are GPU-resident vs must be computed on CPU.
//!
//! The key property it guarantees is **placement invariance**: where an expert is
//! computed (GPU slot vs CPU) does not change the output, only the fetch/evict
//! bookkeeping. That is what P1 must hold before the GPU upload path lands.

use crate::config::Config;
use crate::moe_backend::{LruExpertCache, StepPlan};
use crate::weights::LayerWeights;

/// Row-major `[out, in]` matrix times a vector, returning `out` (same as
/// `ref_model::matvec_t`). `w` length must be `out * in`.
fn matvec_t(x: &[f32], w: &[f32], out_dim: usize) -> Vec<f32> {
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

pub(crate) fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Gate/up/down SwiGLU MLP for one expert, returning `[d]`.
fn expert_mlp(xn: &[f32], wg: &[f32], wu: &[f32], wd: &[f32], inter: usize, d: usize) -> Vec<f32> {
    let gate = matvec_t(xn, wg, inter);
    let up = matvec_t(xn, wu, inter);
    let mut eh = vec![0.0; inter];
    for i in 0..inter {
        eh[i] = silu(gate[i]) * up[i];
    }
    matvec_t(&eh, wd, d)
}

/// Decode-step MoE offload engine for one layer (host side).
#[derive(Debug)]
pub struct MoeOffload {
    cache: LruExpertCache,
}

/// Output of a MoE offload step.
#[derive(Debug)]
pub struct StepOut {
    /// MoE residual to add to `x` (`[d_model]`).
    pub residual: Vec<f32>,
    /// The LRU placement plan for this step.
    pub plan: StepPlan,
}

impl MoeOffload {
    /// Creates an offload engine with `gpu_slots` resident expert slots.
    #[must_use]
    pub fn new(gpu_slots: usize) -> Self {
        Self {
            cache: LruExpertCache::new(gpu_slots),
        }
    }

    /// Computes the MoE residual + placement plan for one step.
    ///
    /// `fetch_budget` caps GPU fetches this step; routed experts beyond it are
    /// placed on the CPU (the P2 q* hook). The output is placement-invariant:
    /// whether an expert is in a GPU slot or computed on CPU does not change the
    /// residual, only the plan.
    pub fn step(
        &mut self,
        cfg: &Config,
        lw: &LayerWeights,
        xn: &[f32],
        fetch_budget: usize,
    ) -> StepOut {
        let ne = cfg.num_experts;
        let topk = cfg.num_experts_per_tok.min(ne);
        let d = cfg.d_model;
        let inter = cfg.intermediate_size;

        let router = matvec_t(xn, &lw.moe_router, ne);
        // Softmax over all experts (max-subtracted for stability).
        let mut maxr = f32::NEG_INFINITY;
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

        // Top-k by probability descending, ties resolved by lower index.
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let topk_experts: Vec<usize> = order.into_iter().take(topk).collect();
        let mut norm = 0.0;
        for &e in &topk_experts {
            norm += probs[e];
        }

        // Drive the offload decision through the LRU cache.
        let routed: Vec<u32> = topk_experts.iter().map(|&e| e as u32).collect();
        let plan = self.cache.plan(&routed, fetch_budget);

        // Weighted sum of each top-k expert SwiGLU MLP.
        let mut residual = vec![0.0; d];
        for &e in &topk_experts {
            let wg = &lw.moe_wg[e * inter * d..(e + 1) * inter * d];
            let wu = &lw.moe_wu[e * inter * d..(e + 1) * inter * d];
            let wd = &lw.moe_wd[e * d * inter..(e + 1) * d * inter];
            let down = expert_mlp(xn, wg, wu, wd, inter, d);
            let w = probs[e] / norm;
            for k in 0..d {
                residual[k] += w * down[k];
            }
        }
        StepOut { residual, plan }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::weights::Weights;

    fn moe_cfg() -> Config {
        let mut c = Config::tiny();
        c.num_experts = 8;
        c.num_experts_per_tok = 2;
        c.intermediate_size = 256;
        c
    }

    /// Independent naive reference: collects contributions first, then weights.
    fn reference(cfg: &Config, lw: &LayerWeights, xn: &[f32]) -> Vec<f32> {
        let ne = cfg.num_experts;
        let topk = cfg.num_experts_per_tok.min(ne);
        let d = cfg.d_model;
        let inter = cfg.intermediate_size;
        let router = matvec_t(xn, &lw.moe_router, ne);
        let maxr = router.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0.0; ne];
        let mut sumr = 0.0;
        for i in 0..ne {
            probs[i] = (router[i] - maxr).exp();
            sumr += probs[i];
        }
        for p in &mut probs {
            *p /= sumr;
        }
        let mut idx: Vec<usize> = (0..ne).collect();
        idx.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let chosen: Vec<usize> = idx.into_iter().take(topk).collect();
        let mut norm = 0.0;
        for &e in &chosen {
            norm += probs[e];
        }
        let contribs: Vec<Vec<f32>> = chosen
            .iter()
            .map(|&e| {
                let wg = &lw.moe_wg[e * inter * d..(e + 1) * inter * d];
                let wu = &lw.moe_wu[e * inter * d..(e + 1) * inter * d];
                let wd = &lw.moe_wd[e * d * inter..(e + 1) * d * inter];
                expert_mlp(xn, wg, wu, wd, inter, d)
            })
            .collect();
        let mut residual = vec![0.0; d];
        for (i, &e) in chosen.iter().enumerate() {
            let w = probs[e] / norm;
            for k in 0..d {
                residual[k] += w * contribs[i][k];
            }
        }
        residual
    }

    #[test]
    fn matches_independent_reference() {
        let cfg = moe_cfg();
        let w = Weights::random(&cfg, 42).unwrap();
        let lw = &w.layers[0];
        let xn: Vec<f32> = (0..cfg.d_model).map(|i| (i as f32) * 0.01 - 1.5).collect();
        let mut moe = MoeOffload::new(4);
        let out = moe.step(&cfg, lw, &xn, usize::MAX);
        let refout = reference(&cfg, lw, &xn);
        assert_eq!(out.residual, refout);
    }

    #[test]
    fn placement_invariance() {
        let cfg = moe_cfg();
        let w = Weights::random(&cfg, 7).unwrap();
        let lw = &w.layers[0];
        let xn: Vec<f32> = (0..cfg.d_model).map(|i| (i as f32) * 0.02 - 2.0).collect();
        // All experts resident on GPU (budget = max) vs all on CPU (budget = 0).
        let mut gpu = MoeOffload::new(4);
        let g = gpu.step(&cfg, lw, &xn, usize::MAX);
        let mut cpu = MoeOffload::new(4);
        let c = cpu.step(&cfg, lw, &xn, 0);
        assert_eq!(g.residual, c.residual);
        // Bookkeeping differs as expected.
        assert!(
            !g.plan.fetches.is_empty()
                || g.plan
                    .placements
                    .iter()
                    .all(|p| matches!(p, crate::moe_backend::Placement::Gpu(_)))
        );
        assert!(
            c.plan
                .placements
                .iter()
                .all(|p| matches!(p, crate::moe_backend::Placement::Cpu))
        );
    }
}
