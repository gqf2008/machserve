//! Transformer configuration for the P1 decode slice.

/// Configuration of the small transformer used in the P1 slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
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
    /// Maximum sequence length (static KV cache size).
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
}

impl Config {
    /// Minimal config for tests: fast to run, exercises GQA + capture.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1024,
            d_model: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 32,
            max_seq_len: 256,
            rms_eps: 1e-6,
        }
    }

    /// A more representative config for benchmarking on the 7900 XTX.
    #[must_use]
    pub fn small() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 512,
            n_layers: 2,
            n_heads: 8,
            n_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 1024,
            rms_eps: 1e-6,
        }
    }
}
