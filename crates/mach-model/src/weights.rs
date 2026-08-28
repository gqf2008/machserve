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
    /// MLA (kv_lora_rank > 0, DeepSeek-V2 style): low-rank Q / compressed KV.
    /// `q_a [q_lora_rank, d]`, `q_a_norm [q_lora_rank]`,
    /// `q_b [n_heads*qk_nope, q_lora_rank]`, `q_rope [n_heads*qk_rope, d]`,
    /// `kv_a [kv_lora_rank + qk_rope, d]`, `kv_a_norm [kv_lora_rank]`,
    /// `kv_b [n_heads*(qk_nope + v_head), kv_lora_rank]`,
    /// `o [d, n_heads*v_head]`. Empty for standard attention.
    pub mla_q_a: Vec<f32>,
    pub mla_q_a_norm: Vec<f32>,
    pub mla_q_b: Vec<f32>,
    pub mla_q_rope: Vec<f32>,
    pub mla_kv_a: Vec<f32>,
    pub mla_kv_a_norm: Vec<f32>,
    pub mla_kv_b: Vec<f32>,
    pub mla_o: Vec<f32>,
    /// MoE (num_experts > 0): router `[num_experts, d_model]`; empty for dense.
    pub moe_router: Vec<f32>,
    /// Per-expert gate/up `[num_experts, intermediate_size, d_model]`.
    pub moe_wg: Vec<f32>,
    pub moe_wu: Vec<f32>,
    /// Per-expert down `[num_experts, d_model, intermediate_size]`.
    pub moe_wd: Vec<f32>,
}

/// Q4 (storage-level int4) layer weights: GEMM tensors quantized, norms and
/// biases kept as f32. Mirrors [`LayerWeights`].
#[derive(Debug, Clone)]
pub struct LayerWeightsQ4 {
    pub wq: crate::q4::Q4Tensor,
    pub wk: crate::q4::Q4Tensor,
    pub wv: crate::q4::Q4Tensor,
    pub wo: crate::q4::Q4Tensor,
    pub rms_attn: Vec<f32>,
    pub wg: crate::q4::Q4Tensor,
    pub wu: crate::q4::Q4Tensor,
    pub wd: crate::q4::Q4Tensor,
    pub rms_mlp: Vec<f32>,
    pub bq: Vec<f32>,
    pub bk: Vec<f32>,
    pub bv: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub mla_q_a: crate::q4::Q4Tensor,
    pub mla_q_a_norm: Vec<f32>,
    pub mla_q_b: crate::q4::Q4Tensor,
    pub mla_q_rope: crate::q4::Q4Tensor,
    pub mla_kv_a: crate::q4::Q4Tensor,
    pub mla_kv_a_norm: Vec<f32>,
    pub mla_kv_b: crate::q4::Q4Tensor,
    pub mla_o: crate::q4::Q4Tensor,
    // Router is tiny and int4 error can flip critical expert selection, so it
    // stays f32 (same reasoning as norms/biases).
    pub moe_router: Vec<f32>,
    pub moe_wg: crate::q4::Q4Tensor,
    pub moe_wu: crate::q4::Q4Tensor,
    pub moe_wd: crate::q4::Q4Tensor,
}

/// All model weights in storage-Q4 form (host memory ~4x smaller than f32 for
/// the GEMM tensors; norms/biases stay f32).
#[derive(Debug, Clone)]
pub struct WeightsQ4 {
    pub tok_emb: crate::q4::Q4Tensor,
    pub rms_final: Vec<f32>,
    pub lm_head: crate::q4::Q4Tensor,
    pub layers: Vec<LayerWeightsQ4>,
}

/// FP8 (storage-level E4M3) layer weights: GEMM tensors quantized, norms and
/// biases kept as f32. Mirrors [`LayerWeights`] / [`LayerWeightsQ4`].
#[derive(Debug, Clone)]
pub struct LayerWeightsFp8 {
    pub wq: crate::fp8::Fp8Tensor,
    pub wk: crate::fp8::Fp8Tensor,
    pub wv: crate::fp8::Fp8Tensor,
    pub wo: crate::fp8::Fp8Tensor,
    pub rms_attn: Vec<f32>,
    pub wg: crate::fp8::Fp8Tensor,
    pub wu: crate::fp8::Fp8Tensor,
    pub wd: crate::fp8::Fp8Tensor,
    pub rms_mlp: Vec<f32>,
    pub bq: Vec<f32>,
    pub bk: Vec<f32>,
    pub bv: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub mla_q_a: crate::fp8::Fp8Tensor,
    pub mla_q_a_norm: Vec<f32>,
    pub mla_q_b: crate::fp8::Fp8Tensor,
    pub mla_q_rope: crate::fp8::Fp8Tensor,
    pub mla_kv_a: crate::fp8::Fp8Tensor,
    pub mla_kv_a_norm: Vec<f32>,
    pub mla_kv_b: crate::fp8::Fp8Tensor,
    pub mla_o: crate::fp8::Fp8Tensor,
    // Router is tiny and quantization error can flip critical expert
    // selection, so it stays f32 (same reasoning as norms/biases).
    pub moe_router: Vec<f32>,
    pub moe_wg: crate::fp8::Fp8Tensor,
    pub moe_wu: crate::fp8::Fp8Tensor,
    pub moe_wd: crate::fp8::Fp8Tensor,
}

/// All model weights in storage-FP8 form (host memory ~2x smaller than f16 /
/// ~4x smaller than f32 for the GEMM tensors; norms/biases stay f32).
#[derive(Debug, Clone)]
pub struct WeightsFp8 {
    pub tok_emb: crate::fp8::Fp8Tensor,
    pub rms_final: Vec<f32>,
    pub lm_head: crate::fp8::Fp8Tensor,
    pub layers: Vec<LayerWeightsFp8>,
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

        let mla = cfg.kv_lora_rank > 0;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let (moe_router, moe_wg, moe_wu, moe_wd) = if cfg.num_experts > 0 {
                (
                    mat(&mut rng, cfg.num_experts, d, scale),
                    mat(&mut rng, cfg.num_experts * cfg.expert_size(), d, scale),
                    mat(&mut rng, cfg.num_experts * cfg.expert_size(), d, scale),
                    mat(&mut rng, cfg.num_experts * d, cfg.expert_size(), scale),
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
            layers.push(LayerWeights {
                // MLA replaces the standard q/k/v/o projections.
                wq: if mla {
                    Vec::new()
                } else {
                    mat(&mut rng, d, cfg.n_heads * cfg.head_dim, scale)
                },
                wk: if mla {
                    Vec::new()
                } else {
                    mat(&mut rng, d, cfg.n_kv_heads * cfg.head_dim, scale)
                },
                wv: if mla {
                    Vec::new()
                } else {
                    mat(&mut rng, d, cfg.n_kv_heads * cfg.head_dim, scale)
                },
                wo: if mla {
                    Vec::new()
                } else {
                    mat(&mut rng, cfg.n_heads * cfg.head_dim, d, scale)
                },
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
                mla_q_a: if mla {
                    mat(&mut rng, cfg.q_lora_rank, d, scale)
                } else {
                    Vec::new()
                },
                mla_q_a_norm: if mla {
                    vec1(&mut rng, cfg.q_lora_rank)
                } else {
                    Vec::new()
                },
                mla_q_b: if mla {
                    mat(
                        &mut rng,
                        cfg.n_heads * cfg.qk_nope_head_dim,
                        cfg.q_lora_rank,
                        scale,
                    )
                } else {
                    Vec::new()
                },
                mla_q_rope: if mla {
                    mat(&mut rng, cfg.n_heads * cfg.qk_rope_head_dim, d, scale)
                } else {
                    Vec::new()
                },
                mla_kv_a: if mla {
                    mat(&mut rng, cfg.kv_lora_rank + cfg.qk_rope_head_dim, d, scale)
                } else {
                    Vec::new()
                },
                mla_kv_a_norm: if mla {
                    vec1(&mut rng, cfg.kv_lora_rank)
                } else {
                    Vec::new()
                },
                mla_kv_b: if mla {
                    mat(
                        &mut rng,
                        cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
                        cfg.kv_lora_rank,
                        scale,
                    )
                } else {
                    Vec::new()
                },
                mla_o: if mla {
                    mat(&mut rng, d, cfg.n_heads * cfg.v_head_dim, scale)
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
                + l.mla_q_a.len()
                + l.mla_q_a_norm.len()
                + l.mla_q_b.len()
                + l.mla_q_rope.len()
                + l.mla_kv_a.len()
                + l.mla_kv_a_norm.len()
                + l.mla_kv_b.len()
                + l.mla_o.len()
                + l.moe_router.len()
                + l.moe_wg.len()
                + l.moe_wu.len()
                + l.moe_wd.len();
        }
        n * 4
    }
}

impl WeightsQ4 {
    /// Converts f32 weights to storage-Q4 (GEMM tensors quantized, norms/biases
    /// and the tiny router copied) — the same conversion the safetensors Q4
    /// loader applies, shared by tests and future re-quantization paths.
    /// `cfg.num_experts` splits the concatenated MoE expert tensors so each
    /// expert is quantized with its own groups (exactly what a checkpoint
    /// load produces).
    pub fn from_weights(w: &Weights, cfg: &Config) -> Self {
        let q = |v: &[f32]| crate::q4::Q4Tensor::quantize(v);
        // MoE expert tensors are stored concatenated; the loader quantizes
        // each expert separately (per-expert groups via `concat`), so split
        // by `cfg.num_experts` here too — a whole-tensor quantize of the
        // concatenation would let one outlier expert flatten every other.
        let qm = |v: &[f32]| -> crate::q4::Q4Tensor {
            let ne = cfg.num_experts;
            if ne == 0 {
                return q(v);
            }
            let per = v.len() / ne;
            assert_eq!(per * ne, v.len(), "MoE tensor must split evenly per expert");
            let mut out = crate::q4::Q4Tensor::default();
            for e in 0..ne {
                out = out.concat(&q(&v[e * per..(e + 1) * per]));
            }
            out
        };
        Self {
            tok_emb: q(&w.tok_emb),
            rms_final: w.rms_final.clone(),
            lm_head: q(&w.lm_head),
            layers: w
                .layers
                .iter()
                .map(|l| LayerWeightsQ4 {
                    wq: q(&l.wq),
                    wk: q(&l.wk),
                    wv: q(&l.wv),
                    wo: q(&l.wo),
                    rms_attn: l.rms_attn.clone(),
                    wg: q(&l.wg),
                    wu: q(&l.wu),
                    wd: q(&l.wd),
                    rms_mlp: l.rms_mlp.clone(),
                    bq: l.bq.clone(),
                    bk: l.bk.clone(),
                    bv: l.bv.clone(),
                    q_norm: l.q_norm.clone(),
                    k_norm: l.k_norm.clone(),
                    mla_q_a: q(&l.mla_q_a),
                    mla_q_a_norm: l.mla_q_a_norm.clone(),
                    mla_q_b: q(&l.mla_q_b),
                    mla_q_rope: q(&l.mla_q_rope),
                    mla_kv_a: q(&l.mla_kv_a),
                    mla_kv_a_norm: l.mla_kv_a_norm.clone(),
                    mla_kv_b: q(&l.mla_kv_b),
                    mla_o: q(&l.mla_o),
                    moe_router: l.moe_router.clone(),
                    moe_wg: qm(&l.moe_wg),
                    moe_wu: qm(&l.moe_wu),
                    moe_wd: qm(&l.moe_wd),
                })
                .collect(),
        }
    }
}

impl WeightsFp8 {
    /// Converts f32 weights to storage-FP8 (GEMM tensors E4M3-quantized with
    /// per-tensor scales, norms/biases and the tiny router copied) — the same
    /// conversion the safetensors FP8 loader applies, shared by tests and
    /// future re-quantization paths. `cfg.num_experts` splits the
    /// concatenated MoE expert tensors so each expert is quantized with its
    /// own per-expert scale and `block = expert size` (exactly what a
    /// checkpoint load produces).
    pub fn from_weights(w: &Weights, cfg: &Config) -> Self {
        let q = |v: &[f32]| crate::fp8::Fp8Tensor::quantize(v);
        // Per-expert scales, mirroring the loader's concat of independently
        // quantized expert tensors (see the Q4 sibling for the rationale).
        let qm = |v: &[f32]| -> crate::fp8::Fp8Tensor {
            let ne = cfg.num_experts;
            if ne == 0 {
                return q(v);
            }
            let per = v.len() / ne;
            assert_eq!(per * ne, v.len(), "MoE tensor must split evenly per expert");
            let mut out = crate::fp8::Fp8Tensor::default();
            for e in 0..ne {
                out = out.concat(&q(&v[e * per..(e + 1) * per]));
            }
            out
        };
        Self {
            tok_emb: q(&w.tok_emb),
            rms_final: w.rms_final.clone(),
            lm_head: q(&w.lm_head),
            layers: w
                .layers
                .iter()
                .map(|l| LayerWeightsFp8 {
                    wq: q(&l.wq),
                    wk: q(&l.wk),
                    wv: q(&l.wv),
                    wo: q(&l.wo),
                    rms_attn: l.rms_attn.clone(),
                    wg: q(&l.wg),
                    wu: q(&l.wu),
                    wd: q(&l.wd),
                    rms_mlp: l.rms_mlp.clone(),
                    bq: l.bq.clone(),
                    bk: l.bk.clone(),
                    bv: l.bv.clone(),
                    q_norm: l.q_norm.clone(),
                    k_norm: l.k_norm.clone(),
                    mla_q_a: q(&l.mla_q_a),
                    mla_q_a_norm: l.mla_q_a_norm.clone(),
                    mla_q_b: q(&l.mla_q_b),
                    mla_q_rope: q(&l.mla_q_rope),
                    mla_kv_a: q(&l.mla_kv_a),
                    mla_kv_a_norm: l.mla_kv_a_norm.clone(),
                    mla_kv_b: q(&l.mla_kv_b),
                    mla_o: q(&l.mla_o),
                    moe_router: l.moe_router.clone(),
                    moe_wg: qm(&l.moe_wg),
                    moe_wu: qm(&l.moe_wu),
                    moe_wd: qm(&l.moe_wd),
                })
                .collect(),
        }
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
