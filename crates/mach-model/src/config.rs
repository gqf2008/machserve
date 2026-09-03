//! Transformer configuration for the P1 decode slice.

/// Device compute dtype for weights + GEMM operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelDType {
    /// Full fp32 path (default; reference behavior).
    #[default]
    F32,
    /// fp16 weights + fp16 GEMM inputs with fp32 accumulation (2x GEMM rate).
    F16,
}

/// Configuration of the small transformer used in the P1 slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Compute dtype for device weights and GEMM operands.
    pub dtype: ModelDType,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden (model) dimension.
    pub d_model: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of KV heads (GQA).
    pub n_kv_heads: usize,
    /// Head dimension (d_model / n_heads).
    pub head_dim: usize,
    /// MLP intermediate size (gate/up width).
    pub intermediate_size: usize,
    /// MoE expert FFN intermediate size (Qwen-MoE style); 0 = use
    /// `intermediate_size` for experts (single-size families).
    pub moe_intermediate_size: usize,
    /// Number of experts (0 = dense MLP, no MoE).
    pub num_experts: usize,
    /// Active experts per token (MoE routing; 0 = dense).
    pub num_experts_per_tok: usize,
    /// Maximum sequence length (static KV cache size).
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// Rotary embedding theta.
    pub rope_theta: f32,
    /// QK-norm (Qwen3): per-head RMSNorm on q/k after projection, before RoPE.
    pub qk_norm: bool,
    /// MLA (DeepSeek-style): q low-rank projection rank; 0 = disabled.
    pub q_lora_rank: usize,
    /// MLA: compressed KV latent rank; 0 = disabled (standard attention).
    pub kv_lora_rank: usize,
    /// MLA: non-RoPE head dim for q/k.
    pub qk_nope_head_dim: usize,
    /// MLA: decoupled RoPE head dim for q/k.
    pub qk_rope_head_dim: usize,
    /// MLA: value head dim.
    pub v_head_dim: usize,
    /// Batched-MoE decode uses the device-side grouped GEMV kernels when true
    /// (the only batched MoE implementation; `false` keeps the hipBLAS host
    /// loop for A/B). The server parses `MACH_MOE_GROUPED` into this field —
    /// the library itself reads no MoE env.
    pub moe_grouped: bool,
    /// Number of shared (always-active) experts, added to the routed-expert
    /// output — DeepSeek-V2 style (`n_shared_experts`). 0 = none.
    pub n_shared_experts: usize,
    /// MoE routing: renormalize the selected experts' scores to sum to 1
    /// (Qwen-MoE: always). DeepSeek-V2 sets `norm_topk_prob=false`, which
    /// leaves the softmax scores un-renormalized.
    pub moe_norm_topk: bool,
    /// MoE routing: multiplier applied to every selected expert's weight
    /// (`routed_scaling_factor`; 1.0 for DeepSeek-V2-Lite).
    pub moe_routed_scale: f32,
    /// YaRN RoPE context extension (`rope_scaling.type == "yarn"`): the
    /// position-scaling factor. 0 = plain RoPE (no YaRN).
    pub rope_yarn_factor: f32,
    /// YaRN: `original_max_position_embeddings` (the un-extended context).
    pub rope_yarn_orig_len: usize,
    /// YaRN: `beta_fast` / `beta_slow` correction-range bounds.
    pub rope_yarn_beta_fast: f32,
    pub rope_yarn_beta_slow: f32,
    /// YaRN: `mscale` (numerator of the cos/sin `attention_factor`).
    pub rope_yarn_mscale: f32,
    /// YaRN: `mscale_all_dim`; > 0 applies the `mscale^2` attention-logit
    /// correction (`0.1 * mscale_all_dim * ln(factor) + 1`, squared).
    pub rope_yarn_mscale_all_dim: f32,
    /// RoPE pairing convention: which two coordinates rotate together.
    ///
    /// `false` (GPT-NeoX / HF `rotate_half`): coordinate `d` pairs with
    /// `d + head_dim/2`. This is what Llama, Qwen2 and Qwen3 do — their
    /// `apply_rotary_pos_emb` is a bare `q * cos + rotate_half(q) * sin`.
    ///
    /// `true` (interleaved): coordinate `2d` pairs with `2d + 1`. DeepSeek-V2
    /// needs this — its `apply_rotary_pos_emb` first permutes
    /// `view(d//2, 2).transpose(4, 3)` from interleaved to split-halves
    /// before applying `rotate_half`, and transformers' current builtin
    /// rotates `view_as_complex(x.reshape(-1, 2))`, i.e. adjacent pairs.
    ///
    /// The two produce identical output at `pos == 0` (cos=1, sin=0 makes
    /// RoPE the identity), so a diff taken at position 0 cannot tell them
    /// apart. Compare at `pos > 0`.
    pub rope_interleave: bool,
    /// Step profiler diagnostic: per-layer attention/MoE HIP event
    /// bracketing, reported after each decode step. The server parses
    /// `MACH_STEP_PROFILE` into this field — the library reads no env.
    pub step_profile: bool,
}

impl Config {
    /// Expert FFN width: `moe_intermediate_size` when set (Qwen-MoE
    /// checkpoints), else `intermediate_size` (single-size families).
    #[must_use]
    pub fn expert_size(&self) -> usize {
        if self.moe_intermediate_size > 0 {
            self.moe_intermediate_size
        } else {
            self.intermediate_size
        }
    }

    /// Shared-expert FFN width: `n_shared_experts * expert_size()`. The shared
    /// experts are stored as one dense MLP (DeepSeek-V2 ships them as
    /// `mlp.shared_experts.*_proj` with the experts' width each), so the
    /// per-layer gate/up matrices are `[shared_size, d_model]`.
    #[must_use]
    pub fn shared_size(&self) -> usize {
        self.n_shared_experts * self.expert_size()
    }

    /// YaRN enabled (rope_scaling.type == "yarn" with a factor > 1).
    #[must_use]
    pub fn yarn(&self) -> bool {
        self.rope_yarn_factor > 1.0 && self.rope_yarn_orig_len > 0
    }

    /// YaRN `mscale` logit correction: `0.1 * mscale_all_dim * ln(factor) + 1`.
    /// Attention logits are multiplied by `mscale^2` when this is > 1
    /// (DeepSeek-V2 rope_scaling). Returns 1.0 when YaRN/mscale is off.
    #[must_use]
    pub fn yarn_mscale(&self) -> f32 {
        if !self.yarn() || self.rope_yarn_mscale_all_dim <= 0.0 {
            return 1.0;
        }
        0.1 * self.rope_yarn_mscale_all_dim * self.rope_yarn_factor.ln() + 1.0
    }

    /// Attention logit scale for head dim `hd`: `1/sqrt(hd)` with the YaRN
    /// `mscale^2` correction folded in when it applies (DeepSeek-V2
    /// `rope_scaling` sets `mscale_all_dim`, and its attention multiplies
    /// `softmax_scale` by `mscale^2`). Plain RoPE / no YaRN returns the usual
    /// `1/sqrt(hd)`.
    #[must_use]
    pub fn attn_scale(&self, hd: usize) -> f32 {
        let m = self.yarn_mscale();
        m * m / (hd as f32).sqrt()
    }

    /// YaRN cos/sin `attention_factor`: `mscale(factor) / mscale_all_dim(factor)`
    /// with `mscale(x) = 0.1 * x * ln(factor) + 1` (HF
    /// `_yarn_get_mscale`). DeepSeek-V2 sets both to 0.707, so this is 1.0 and
    /// cos/sin are unscaled; the logit correction above is what actually bites.
    /// Returns 1.0 when YaRN is off or both scales are absent.
    #[must_use]
    pub fn yarn_attention_factor(&self) -> f32 {
        if !self.yarn() {
            return 1.0;
        }
        // HF `yarn_get_mscale(scale, mscale)` with the config defaults
        // (`mscale = 1` when absent, `mscale_all_dim = 0`), so a missing
        // `mscale_all_dim` leaves the cos/sin unscaled.
        let ln = self.rope_yarn_factor.ln();
        let num = 0.1 * self.rope_yarn_mscale * ln + 1.0;
        let den = 0.1 * self.rope_yarn_mscale_all_dim * ln + 1.0;
        num / den
    }

    /// Minimal config for tests: fast to run, exercises GQA + capture.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            dtype: ModelDType::F32,
            vocab_size: 1024,
            d_model: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 32,
            intermediate_size: 512,
            moe_intermediate_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            max_seq_len: 256,
            rms_eps: 1e-6,
            rope_theta: 10000.0,
            qk_norm: false,
            q_lora_rank: 0,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            n_shared_experts: 0,
            moe_norm_topk: true,
            moe_routed_scale: 1.0,
            rope_yarn_factor: 0.0,
            rope_yarn_orig_len: 0,
            rope_yarn_beta_fast: 0.0,
            rope_yarn_beta_slow: 0.0,
            rope_yarn_mscale: 1.0,
            rope_yarn_mscale_all_dim: 0.0,
            rope_interleave: false,
            step_profile: false,
            moe_grouped: true,
        }
    }

    /// Qwen3 dense-family config (QK-norm, 3x hidden MLP, theta=1e6).
    #[must_use]
    pub fn qwen3(
        hidden_size: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        vocab_size: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            dtype: ModelDType::F32,
            vocab_size,
            d_model: hidden_size,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim: hidden_size / n_heads,
            intermediate_size: 3 * hidden_size,
            moe_intermediate_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            max_seq_len,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
            qk_norm: true,
            q_lora_rank: 0,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            n_shared_experts: 0,
            moe_norm_topk: true,
            moe_routed_scale: 1.0,
            rope_yarn_factor: 0.0,
            rope_yarn_orig_len: 0,
            rope_yarn_beta_fast: 0.0,
            rope_yarn_beta_slow: 0.0,
            rope_yarn_mscale: 1.0,
            rope_yarn_mscale_all_dim: 0.0,
            rope_interleave: false,
            step_profile: false,
            moe_grouped: true,
        }
    }

    /// Llama/Qwen-style config from raw hyperparameters.
    #[must_use]
    pub fn llama(
        hidden_size: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        vocab_size: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            dtype: ModelDType::F32,
            vocab_size,
            d_model: hidden_size,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim: hidden_size / n_heads,
            intermediate_size: 4 * hidden_size,
            moe_intermediate_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            max_seq_len,
            rms_eps: 1e-6,
            rope_theta: 10000.0,
            qk_norm: false,
            q_lora_rank: 0,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            n_shared_experts: 0,
            moe_norm_topk: true,
            moe_routed_scale: 1.0,
            rope_yarn_factor: 0.0,
            rope_yarn_orig_len: 0,
            rope_yarn_beta_fast: 0.0,
            rope_yarn_beta_slow: 0.0,
            rope_yarn_mscale: 1.0,
            rope_yarn_mscale_all_dim: 0.0,
            rope_interleave: false,
            step_profile: false,
            moe_grouped: true,
        }
    }

    /// A more representative config for benchmarking on the 7900 XTX.
    #[must_use]
    pub fn small() -> Self {
        Self {
            dtype: ModelDType::F32,
            vocab_size: 32000,
            d_model: 512,
            n_layers: 2,
            n_heads: 8,
            n_kv_heads: 4,
            head_dim: 64,
            intermediate_size: 2048,
            moe_intermediate_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            max_seq_len: 1024,
            rms_eps: 1e-6,
            rope_theta: 10000.0,
            qk_norm: false,
            q_lora_rank: 0,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            n_shared_experts: 0,
            moe_norm_topk: true,
            moe_routed_scale: 1.0,
            rope_yarn_factor: 0.0,
            rope_yarn_orig_len: 0,
            rope_yarn_beta_fast: 0.0,
            rope_yarn_beta_slow: 0.0,
            rope_yarn_mscale: 1.0,
            rope_yarn_mscale_all_dim: 0.0,
            rope_interleave: false,
            step_profile: false,
            moe_grouped: true,
        }
    }

    /// DeepSeek-V2-style MLA config (low-rank Q + compressed KV).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn mla(
        hidden_size: usize,
        n_layers: usize,
        n_heads: usize,
        vocab_size: usize,
        max_seq_len: usize,
        q_lora_rank: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
    ) -> Self {
        Self {
            dtype: ModelDType::F32,
            vocab_size,
            d_model: hidden_size,
            n_layers,
            n_heads,
            n_kv_heads: n_heads,
            head_dim: qk_nope_head_dim + qk_rope_head_dim,
            intermediate_size: 4 * hidden_size,
            moe_intermediate_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            max_seq_len,
            rms_eps: 1e-6,
            rope_theta: 10000.0,
            qk_norm: false,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            n_shared_experts: 0,
            moe_norm_topk: true,
            moe_routed_scale: 1.0,
            rope_yarn_factor: 0.0,
            rope_yarn_orig_len: 0,
            rope_yarn_beta_fast: 0.0,
            rope_yarn_beta_slow: 0.0,
            rope_yarn_mscale: 1.0,
            rope_yarn_mscale_all_dim: 0.0,
            rope_interleave: false,
            step_profile: false,
            moe_grouped: true,
        }
    }
}
