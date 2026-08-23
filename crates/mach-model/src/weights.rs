//! Model weights (host side, fp32) for the P1 decode slice.
//!
//! Randomly initialized with a deterministic LCG so CPU and GPU runs share
//! identical weights. Real safetensors loading is a follow-up (P1b); the
//! slice verifies the execution path first.

use crate::{Config, Error};

/// Per-layer weights. All matrices are row-major `[out, in]`.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// `[d_model, n_heads * head_dim]`
    pub wq: Vec<f32>,
    /// `[d_model, n_kv_heads * head_dim]`
    pub wk: Vec<f32>,
    /// `[d_model, n_kv_heads * head_dim]`
    pub wv: Vec<f32>,
    /// `[n_heads * head_dim, d_model]`
    pub wo: Vec<f32>,
    /// `[d_model]`
    pub rms_attn: Vec<f32>,
    /// `[intermediate_size, d_model]`
    pub wg: Vec<f32>,
    /// `[intermediate_size, d_model]`
    pub wu: Vec<f32>,
    /// `[d_model, intermediate_size]`
    pub wd: Vec<f32>,
    /// `[d_model]`
    pub rms_mlp: Vec<f32>,
    /// Attention projection biases (Qwen2 checkpoints ship them despite
    /// `attention_bias: false` in some configs). Empty when absent.
    pub bq: Vec<f32>,
    pub bk: Vec<f32>,
    pub bv: Vec<f32>,
    /// QK-norm (Qwen3): per-head RMSNorm weight `[n_heads * head_dim]` for q;
    /// empty when qk_norm=false.
    pub q_norm: Vec<f32>,
    /// QK-norm per-head RMSNorm weight `[n_kv_heads * head_dim]` for k.
    pub k_norm: Vec<f32>,
    /// MoE (num_experts > 0): router `[num_experts, d_model]`; empty for dense.
    pub moe_router: Vec<f32>,
    /// Per-expert gate/up `[num_experts, intermediate_size, d_model]`.
    pub moe_wg: Vec<f32>,
    pub moe_wu: Vec<f32>,
    /// Per-expert down `[num_experts, d_model, intermediate_size]`.
    pub moe_wd: Vec<f32>,
}

/// All model weights.
#[derive(Debug, Clone)]
pub struct Weights {
    /// `[vocab_size, d_model]`
    pub tok_emb: Vec<f32>,
    /// `[d_model]`
    pub rms_final: Vec<f32>,
    /// `[vocab_size, d_model]`
    pub lm_head: Vec<f32>,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
}

impl Weights {
    /// Builds deterministic random weights for `cfg` (seeded).
    pub fn random(cfg: &Config, seed: u64) -> Result<Self, Error> {
        let mut rng = Lcg::new(seed);
        let d = cfg.d_model;
        let scale = 1.0 / (d as f32).sqrt();

        fn mat(rng: &mut Lcg, rows: usize, cols: usize, scale: f32) -> Vec<f32> {
            let n = rows * cols;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push((rng.next_f32() - 0.5) * 2.0 * scale);
            }
            v
        }
        fn vec1(rng: &mut Lcg, n: usize) -> Vec<f32> {
            (0..n).map(|_| 1.0 + (rng.next_f32() - 0.5) * 0.1).collect()
        }

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let (moe_router, moe_wg, moe_wu, moe_wd) = if cfg.num_experts > 0 {
                (
                    mat(&mut rng, cfg.num_experts, d, scale),
                    mat(&mut rng, cfg.num_experts * cfg.intermediate_size, d, scale),
                    mat(&mut rng, cfg.num_experts * cfg.intermediate_size, d, scale),
                    mat(&mut rng, cfg.num_experts * d, cfg.intermediate_size, scale),
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
            layers.push(LayerWeights {
                wq: mat(&mut rng, d, cfg.n_heads * cfg.head_dim, scale),
                wk: mat(&mut rng, d, cfg.n_kv_heads * cfg.head_dim, scale),
                wv: mat(&mut rng, d, cfg.n_kv_heads * cfg.head_dim, scale),
                wo: mat(&mut rng, cfg.n_heads * cfg.head_dim, d, scale),
                rms_attn: vec1(&mut rng, d),
                wg: mat(&mut rng, cfg.intermediate_size, d, scale),
                wu: mat(&mut rng, cfg.intermediate_size, d, scale),
                wd: mat(&mut rng, d, cfg.intermediate_size, scale),
                rms_mlp: vec1(&mut rng, d),
                bq: Vec::new(),
                bk: Vec::new(),
                bv: Vec::new(),
                q_norm: if cfg.qk_norm {
                    vec1(&mut rng, cfg.n_heads * cfg.head_dim)
                } else {
                    Vec::new()
                },
                k_norm: if cfg.qk_norm {
                    vec1(&mut rng, cfg.n_kv_heads * cfg.head_dim)
                } else {
                    Vec::new()
                },
                moe_router,
                moe_wg,
                moe_wu,
                moe_wd,
            });
        }

        Ok(Self {
            tok_emb: mat(&mut rng, cfg.vocab_size, d, scale),
            rms_final: vec1(&mut rng, d),
            lm_head: mat(&mut rng, cfg.vocab_size, d, scale),
            layers,
        })
    }

    /// Byte size of all weights.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        let mut n = self.tok_emb.len() + self.rms_final.len() + self.lm_head.len();
        for l in &self.layers {
            n += l.wq.len()
                + l.wk.len()
                + l.wv.len()
                + l.wo.len()
                + l.rms_attn.len()
                + l.wg.len()
                + l.wu.len()
                + l.wd.len()
                + l.rms_mlp.len()
                + l.q_norm.len()
                + l.k_norm.len()
                + l.moe_router.len()
                + l.moe_wg.len()
                + l.moe_wu.len()
                + l.moe_wd.len();
        }
        n * 4
    }
}

/// Minimal deterministic LCG (splitmix-style) for reproducible weights.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9e3779b97f4a7c15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add(0xbf58476d1ce4e5b9);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
