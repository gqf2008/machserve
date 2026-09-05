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
    /// Attention output gate (Qwen3.5 `attn_output_gate`): q_proj carries
    /// `n_heads * head_dim * 2` rows — each head's block is `[query | gate]`
    /// (HF chunks per head after the projection). The gate skips QK-norm and
    /// RoPE and multiplies the attention output elementwise (`sigmoid`)
    /// before o_proj.
    pub attn_output_gate: bool,
    /// Zero-centered RMSNorm weights (Qwen3.5 `Qwen3_5RMSNorm`): the
    /// checkpoint stores `w` zero-init with forward `x * (1 + w)`. The loader
    /// shifts these tensors by +1 at load (layer norms, final norm, q/k
    /// norms) so the runtime's plain `x * w` matches. The GDN gated norm is
    /// ones-init and multiplies plainly — it is NOT shifted.
    pub zero_centered_norm: bool,
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
    /// and its successors need this — their `apply_rotary_pos_emb` permutes
    /// `view(d//2, 2).transpose(4, 3)` from interleaved to split-halves
    /// before applying `rotate_half`, i.e. the pair that ends up rotating
    /// together came from adjacent coordinates. transformers' dedicated
    /// `apply_rotary_pos_emb_interleave` (selected per-checkpoint, not the
    /// default path) does the same.
    ///
    /// The two produce identical output at `pos == 0` (cos=1, sin=0 makes
    /// RoPE the identity), so a diff taken at position 0 cannot tell them
    /// apart. Compare at `pos > 0`.
    pub rope_interleave: bool,
    /// Partial rotary (Qwen3.5/Qwen3.8): fraction of the head dim that gets
    /// RoPE; the tail coordinates pass through unrotated. The inv_freq table
    /// is built over the ROTARY dim (`theta^(-2d/rotary_dim)`), not the full
    /// head dim, and the pairing happens inside the rotary slice. 1.0 = full
    /// rotation (all prior families).
    pub rope_rotary_pct: f32,
    /// Hybrid linear attention (Qwen3.5/Qwen3.8 gated DeltaNet): every Nth
    /// layer is full attention (`li + 1) % N == 0`, e.g. interval 4 puts
    /// full attention at layers 3, 7, 11, ... of a 64-layer stack); the rest
    /// are linear-attention gated-DeltaNet layers. 0 = no linear-attention
    /// layers (every layer full attention — all prior families).
    pub full_attention_interval: usize,
    /// GDN: number of key heads (l2-normalized; each shared by
    /// `gdn_v_heads / gdn_k_heads` consecutive value heads via
    /// `repeat_interleave`).
    pub gdn_k_heads: usize,
    /// GDN: number of value heads (also the recurrent-state head count).
    pub gdn_v_heads: usize,
    /// GDN: per-head dim, shared by key and value heads.
    pub gdn_head_dim: usize,
    /// GDN: short depthwise causal conv kernel size on the fused qkv
    /// (Qwen3.5: 4).
    pub gdn_conv_kernel: usize,
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

    /// Rotary coordinates per head: `head_dim * rope_rotary_pct` (rounded
    /// down to an even number of PAIRS). Partial-rotary checkpoints
    /// (Qwen3.5: 0.25 of 256 = 64) rotate only this leading slice; the tail
    /// passes through. Full rotation returns `head_dim`.
    #[must_use]
    pub fn attn_rotary_dim(&self) -> usize {
        if self.rope_rotary_pct >= 1.0 {
            self.head_dim
        } else {
            let dim = (self.head_dim as f32 * self.rope_rotary_pct) as usize & !1;
            dim.max(2).min(self.head_dim)
        }
    }

    /// Hybrid attention layout: whether layer `li` is a full-attention layer
    /// (true) or a gated-DeltaNet linear-attention layer (false). With
    /// `full_attention_interval == 0` every layer is full attention (all
    /// prior families).
    #[must_use]
    pub fn layer_is_full_attn(&self, li: usize) -> bool {
        self.full_attention_interval == 0 || (li + 1).is_multiple_of(self.full_attention_interval)
    }

    /// GDN enabled: the config describes a hybrid stack with linear-attention
    /// layers (`full_attention_interval > 0` with value heads).
    #[must_use]
    pub fn gdn_enabled(&self) -> bool {
        self.full_attention_interval > 0 && self.gdn_v_heads > 0
    }

    /// GDN fused q/k/v width: `gdn_k_heads * gdn_head_dim`.
    #[must_use]
    pub fn gdn_key_dim(&self) -> usize {
        self.gdn_k_heads * self.gdn_head_dim
    }

    /// GDN value width: `gdn_v_heads * gdn_head_dim` (input to the gated
    /// RMSNorm + out_proj).
    #[must_use]
    pub fn gdn_value_dim(&self) -> usize {
        self.gdn_v_heads * self.gdn_head_dim
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
            attn_output_gate: false,
            zero_centered_norm: false,
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
            rope_rotary_pct: 1.0,
            full_attention_interval: 0,
            gdn_k_heads: 0,
            gdn_v_heads: 0,
            gdn_head_dim: 0,
            gdn_conv_kernel: 0,
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
            attn_output_gate: false,
            zero_centered_norm: false,
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
            rope_rotary_pct: 1.0,
            full_attention_interval: 0,
            gdn_k_heads: 0,
            gdn_v_heads: 0,
            gdn_head_dim: 0,
            gdn_conv_kernel: 0,
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
            attn_output_gate: false,
            zero_centered_norm: false,
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
            rope_rotary_pct: 1.0,
            full_attention_interval: 0,
            gdn_k_heads: 0,
            gdn_v_heads: 0,
            gdn_head_dim: 0,
            gdn_conv_kernel: 0,
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
            attn_output_gate: false,
            zero_centered_norm: false,
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
            rope_rotary_pct: 1.0,
            full_attention_interval: 0,
            gdn_k_heads: 0,
            gdn_v_heads: 0,
            gdn_head_dim: 0,
            gdn_conv_kernel: 0,
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
            attn_output_gate: false,
            zero_centered_norm: false,
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
            // MLA is DeepSeek's attention, and every DeepSeek checkpoint uses
            // the interleaved convention — defaulting to `false` here would
            // make this constructor disagree with the real checkpoints it is
            // shaped for, and would leave the interleaved branch with no
            // model-level coverage (parity tests compare both sides under the
            // same flag, so either default passes; only this one is truthful).
            rope_interleave: true,
            rope_rotary_pct: 1.0,
            full_attention_interval: 0,
            gdn_k_heads: 0,
            gdn_v_heads: 0,
            gdn_head_dim: 0,
            gdn_conv_kernel: 0,
            step_profile: false,
            moe_grouped: true,
        }
    }

    /// Qwen3.5/Qwen3.8 hybrid config (gated-DeltaNet linear attention + every
    /// `full_attention_interval`-th layer full attention, QK-norm, partial
    /// rotary 0.25, theta=1e7). `gdn_*` describe the linear-attention layers;
    /// pass zeros with `full_attention_interval = 0` for a pure full-attention
    /// variant.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn qwen3_5(
        hidden_size: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
        gdn_k_heads: usize,
        gdn_v_heads: usize,
        gdn_head_dim: usize,
        gdn_conv_kernel: usize,
    ) -> Self {
        let full_attention_interval = if gdn_v_heads > 0 { 4 } else { 0 };
        Self {
            dtype: ModelDType::F32,
            vocab_size,
            d_model: hidden_size,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate_size,
            moe_intermediate_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            max_seq_len,
            rms_eps: 1e-6,
            rope_theta: 10_000_000.0,
            qk_norm: true,
            // Family constants: full-attn layers carry the doubled q_proj
            // with its sigmoid output gate, and RMSNorm weights ship
            // zero-centered (`x * (1 + w)`).
            attn_output_gate: true,
            zero_centered_norm: true,
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
            // Qwen3.5 rotates with plain rotate_half pairing (the M-RoPE
            // machinery degenerates to standard sequential positions for
            // text-only input), but only over the leading 25% of each head.
            rope_interleave: false,
            rope_rotary_pct: 0.25,
            full_attention_interval,
            gdn_k_heads,
            gdn_v_heads,
            gdn_head_dim,
            gdn_conv_kernel,
            step_profile: false,
            moe_grouped: true,
        }
    }
}
