//! Kernel backend identity and capabilities.

use core::fmt;

/// Which platform a kernel implementation targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendId {
    /// NVIDIA CUDA.
    Cuda,
    /// AMD ROCm.
    Hip,
    /// Host CPU (reference).
    Cpu,
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda => write!(f, "cuda"),
            Self::Hip => write!(f, "hip"),
            Self::Cpu => write!(f, "cpu"),
        }
    }
}

/// Static capabilities of a kernel implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackendCaps {
    /// Backend this kernel runs on.
    pub backend: BackendId,
    /// Supports FP8 (e4m3fn / e5m2) GEMM.
    pub fp8: bool,
    /// Capturable inside a CUDA graph window (no host sync, no allocation).
    pub graph_capturable: bool,
    /// Supports in-place epilogue fusion (e.g. bias+activation fused into GEMM).
    pub fused_epilogue: bool,
}

impl BackendCaps {
    /// CPU reference capabilities.
    #[must_use]
    pub const fn cpu() -> Self {
        Self {
            backend: BackendId::Cpu,
            fp8: false,
            graph_capturable: true,
            fused_epilogue: false,
        }
    }
}
