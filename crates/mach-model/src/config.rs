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
}

impl Config {
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
        }
    }
}
