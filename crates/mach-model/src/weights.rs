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
    /// Shared experts (DeepSeek-V2 `n_shared_experts > 0`, MoE layers only):
    /// one dense SwiGLU MLP of width `n_shared_experts * expert_size()` whose
    /// output is ADDED to the routed experts' weighted sum. Empty when the
    /// checkpoint has no shared experts.
    pub shared_wg: Vec<f32>,
    pub shared_wu: Vec<f32>,
    pub shared_wd: Vec<f32>,
    /// Gated DeltaNet (Qwen3.5 linear-attention layers): fused QKV projection
    /// `[2*key_dim + value_dim, d_model]` — q rows `[0, key_dim)`, k rows
    /// `[key_dim, 2*key_dim)`, v rows `[2*key_dim, 2*key_dim + value_dim)`.
    /// Empty for full-attention layers.
    pub gdn_in_qkv: Vec<f32>,
    /// Output gate projection `[value_dim, d_model]`. Empty otherwise.
    pub gdn_in_z: Vec<f32>,
    /// Per v-head decay input `[gdn_v_heads, d_model]`. Empty otherwise.
    pub gdn_in_a: Vec<f32>,
    /// Per v-head beta input `[gdn_v_heads, d_model]`. Empty otherwise.
    pub gdn_in_b: Vec<f32>,
    /// Depthwise causal conv1d weight `[2*key_dim + value_dim, conv_kernel]`.
    /// Empty otherwise.
    pub gdn_conv_w: Vec<f32>,
    /// Decay log-scale `[gdn_v_heads]` (gate = -exp(A_log)). Empty otherwise.
    pub gdn_a_log: Vec<f32>,
    /// Delta-rule time-step bias `[gdn_v_heads]`. Empty otherwise.
    pub gdn_dt_bias: Vec<f32>,
    /// Gated RMSNorm weight over v-head dim `[gdn_head_dim]`, shared across
    /// v-heads. Empty otherwise.
    pub gdn_norm: Vec<f32>,
    /// Output projection `[d_model, value_dim]`. Empty otherwise.
    pub gdn_out: Vec<f32>,
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
    /// Shared experts (DeepSeek-V2): gate/up `[shared_size, d_model]`, down
    /// `[d_model, shared_size]`; empty when absent.
    pub shared_wg: crate::q4::Q4Tensor,
    pub shared_wu: crate::q4::Q4Tensor,
    pub shared_wd: crate::q4::Q4Tensor,
    // GDN (Qwen3.5 linear attention). The three big projections quantize; the
    // a/b gate projections feed softplus/sigmoid (tiny — `[48, d_model]` — but
    // error there shifts decay for a whole head), and conv_w/a_log/dt_bias/
    // norm are structured or per-head small, so they stay f32.
    pub gdn_in_qkv: crate::q4::Q4Tensor,
    pub gdn_in_z: crate::q4::Q4Tensor,
    pub gdn_in_a: Vec<f32>,
    pub gdn_in_b: Vec<f32>,
    pub gdn_conv_w: Vec<f32>,
    pub gdn_a_log: Vec<f32>,
    pub gdn_dt_bias: Vec<f32>,
    pub gdn_norm: Vec<f32>,
    pub gdn_out: crate::q4::Q4Tensor,
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
    /// Shared experts (DeepSeek-V2): gate/up `[shared_size, d_model]`, down
    /// `[d_model, shared_size]`; empty when absent.
    pub shared_wg: crate::fp8::Fp8Tensor,
    pub shared_wu: crate::fp8::Fp8Tensor,
    pub shared_wd: crate::fp8::Fp8Tensor,
    // GDN (Qwen3.5 linear attention), same split as the Q4 mirror: the three
    // big projections quantize, the gate/small tensors stay f32.
    pub gdn_in_qkv: crate::fp8::Fp8Tensor,
    pub gdn_in_z: crate::fp8::Fp8Tensor,
    pub gdn_in_a: Vec<f32>,
    pub gdn_in_b: Vec<f32>,
    pub gdn_conv_w: Vec<f32>,
    pub gdn_a_log: Vec<f32>,
    pub gdn_dt_bias: Vec<f32>,
    pub gdn_norm: Vec<f32>,
    pub gdn_out: crate::fp8::Fp8Tensor,
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

        // MLA: the q projection's contraction width. With a low-rank q
        // (`q_lora_rank > 0`, DeepSeek-V2 236B) both halves take the
        // normalized q_lora; without one (DeepSeek-V2-Lite, `q_lora_rank` is
        // null) they take the layer input directly, so the width is d_model.
        let mla = cfg.kv_lora_rank > 0;
        let q_kk = if cfg.q_lora_rank > 0 {
            cfg.q_lora_rank
        } else {
            d
        };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            // Qwen3.5 hybrid: linear-attention (GDN) layers replace the
            // standard q/k/v/o path entirely.
            let gdn = cfg.gdn_enabled() && !cfg.layer_is_full_attn(li);
            // Shared experts (DeepSeek-V2): a dense SwiGLU MLP of width
            // `n_shared_experts * expert_size`, present on every routed layer.
            let (shared_wg, shared_wu, shared_wd) =
                if cfg.num_experts > 0 && cfg.n_shared_experts > 0 {
                    (
                        mat(&mut rng, cfg.shared_size(), d, scale),
                        mat(&mut rng, cfg.shared_size(), d, scale),
                        mat(&mut rng, d, cfg.shared_size(), scale),
                    )
                } else {
                    (Vec::new(), Vec::new(), Vec::new())
                };
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
            // GDN tensors, following the checkpoint's own init conventions:
            // conv1d is the identity-like depthwise init (columns 1..=1, last
            // column 2), A_log is log-uniform over U(0.01, 16) so decay spans
            // fast/slow heads, dt_bias and the gated-norm weight start at one.
            let (
                gdn_in_qkv,
                gdn_in_z,
                gdn_in_a,
                gdn_in_b,
                gdn_conv_w,
                gdn_a_log,
                gdn_dt_bias,
                gdn_norm,
                gdn_out,
            ) = if gdn {
                let kd = cfg.gdn_key_dim();
                let vd = cfg.gdn_value_dim();
                let conv_dim = 2 * kd + vd;
                let conv_k = cfg.gdn_conv_kernel;
                let mut conv_w = vec![0.0f32; conv_dim * conv_k];
                for c in 0..conv_dim {
                    for k in 0..conv_k - 1 {
                        conv_w[c * conv_k + k] = 1.0;
                    }
                    conv_w[c * conv_k + conv_k - 1] = 2.0;
                }
                (
                    mat(&mut rng, conv_dim, d, scale),
                    mat(&mut rng, vd, d, scale),
                    mat(&mut rng, cfg.gdn_v_heads, d, scale),
                    mat(&mut rng, cfg.gdn_v_heads, d, scale),
                    conv_w,
                    (0..cfg.gdn_v_heads)
                        .map(|_| (0.01f32 * 1600.0f32.powf(rng.next_f32())).ln())
                        .collect(),
                    vec![1.0f32; cfg.gdn_v_heads],
                    vec1(&mut rng, cfg.gdn_head_dim),
                    mat(&mut rng, d, vd, scale),
                )
            } else {
                (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            };
            layers.push(LayerWeights {
                // MLA replaces the standard q/k/v/o projections; so do GDN
                // linear-attention layers.
                // Qwen3.5 `attn_output_gate`: q_proj carries a per-head
                // sigmoid gate in the second half of each head's
                // `2 * head_dim` block, doubling the projection width.
                wq: if mla || gdn {
                    Vec::new()
                } else {
                    mat(
                        &mut rng,
                        d,
                        cfg.n_heads * cfg.head_dim * if cfg.attn_output_gate { 2 } else { 1 },
                        scale,
                    )
                },
                wk: if mla || gdn {
                    Vec::new()
                } else {
                    mat(&mut rng, d, cfg.n_kv_heads * cfg.head_dim, scale)
                },
                wv: if mla || gdn {
                    Vec::new()
                } else {
                    mat(&mut rng, d, cfg.n_kv_heads * cfg.head_dim, scale)
                },
                wo: if mla || gdn {
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
                q_norm: if cfg.qk_norm && !gdn {
                    vec1(&mut rng, cfg.n_heads * cfg.head_dim)
                } else {
                    Vec::new()
                },
                k_norm: if cfg.qk_norm && !gdn {
                    vec1(&mut rng, cfg.n_kv_heads * cfg.head_dim)
                } else {
                    Vec::new()
                },
                // Low-rank q exists only when `q_lora_rank > 0`; DeepSeek-V2
                // -Lite ships `q_lora_rank: null`, so its q path is a single
                // full-width projection straight from the layer input.
                mla_q_a: if mla && cfg.q_lora_rank > 0 {
                    mat(&mut rng, cfg.q_lora_rank, d, scale)
                } else {
                    Vec::new()
                },
                mla_q_a_norm: if mla && cfg.q_lora_rank > 0 {
                    vec1(&mut rng, cfg.q_lora_rank)
                } else {
                    Vec::new()
                },
                // MLA q: ONE fused projection per checkpoint family, split per
                // head into the non-RoPE and RoPE halves at load time:
                //   q_lora_rank > 0 (DeepSeek-V2 236B): `q_b_proj`
                //     [heads*(nope+rope), q_lora_rank], applied to the
                //     normalized q_lora.
                //   q_lora_rank == 0 (DeepSeek-V2-Lite): `q_proj`
                //     [heads*(nope+rope), d_model], applied to the layer
                //     input.
                // Both halves share the same input and contraction width,
                // which is what the runtime GEMMs below assume.
                mla_q_b: if mla {
                    mat(&mut rng, cfg.n_heads * cfg.qk_nope_head_dim, q_kk, scale)
                } else {
                    Vec::new()
                },
                mla_q_rope: if mla {
                    mat(&mut rng, cfg.n_heads * cfg.qk_rope_head_dim, q_kk, scale)
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
                shared_wg,
                shared_wu,
                shared_wd,
                gdn_in_qkv,
                gdn_in_z,
                gdn_in_a,
                gdn_in_b,
                gdn_conv_w,
                gdn_a_log,
                gdn_dt_bias,
                gdn_norm,
                gdn_out,
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
                + l.moe_wd.len()
                + l.shared_wg.len()
                + l.shared_wu.len()
                + l.shared_wd.len()
                + l.gdn_in_qkv.len()
                + l.gdn_in_z.len()
                + l.gdn_in_a.len()
                + l.gdn_in_b.len()
                + l.gdn_conv_w.len()
                + l.gdn_a_log.len()
                + l.gdn_dt_bias.len()
                + l.gdn_norm.len()
                + l.gdn_out.len();
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
            let parts: Vec<_> = (0..ne).map(|e| q(&v[e * per..(e + 1) * per])).collect();
            crate::q4::Q4Tensor::concat_many(&parts)
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
                    shared_wg: q(&l.shared_wg),
                    shared_wu: q(&l.shared_wu),
                    shared_wd: q(&l.shared_wd),
                    gdn_in_qkv: q(&l.gdn_in_qkv),
                    gdn_in_z: q(&l.gdn_in_z),
                    gdn_in_a: l.gdn_in_a.clone(),
                    gdn_in_b: l.gdn_in_b.clone(),
                    gdn_conv_w: l.gdn_conv_w.clone(),
                    gdn_a_log: l.gdn_a_log.clone(),
                    gdn_dt_bias: l.gdn_dt_bias.clone(),
                    gdn_norm: l.gdn_norm.clone(),
                    gdn_out: q(&l.gdn_out),
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
            let parts: Vec<_> = (0..ne).map(|e| q(&v[e * per..(e + 1) * per])).collect();
            crate::fp8::Fp8Tensor::concat_many(&parts)
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
                    shared_wg: qm(&l.shared_wg),
                    shared_wu: qm(&l.shared_wu),
                    shared_wd: qm(&l.shared_wd),
                    gdn_in_qkv: q(&l.gdn_in_qkv),
                    gdn_in_z: q(&l.gdn_in_z),
                    gdn_in_a: l.gdn_in_a.clone(),
                    gdn_in_b: l.gdn_in_b.clone(),
                    gdn_conv_w: l.gdn_conv_w.clone(),
                    gdn_a_log: l.gdn_a_log.clone(),
                    gdn_dt_bias: l.gdn_dt_bias.clone(),
                    gdn_norm: l.gdn_norm.clone(),
                    gdn_out: q(&l.gdn_out),
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
