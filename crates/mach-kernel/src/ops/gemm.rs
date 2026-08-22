//! GEMM family: dense and FP8 matrix multiplications.

use crate::buffer::Buffer;
use crate::{BackendCaps, Kernel, KernelRegistry, RegistryError};
use std::sync::Arc;

/// GEMM kernel contract.
pub trait Gemm: Kernel {
    /// `out = alpha * a @ b + beta * c` with optional fused epilogue.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        _a: &Buffer,
        _b: &Buffer,
        _c: Option<&Buffer>,
        _out: &mut Buffer,
        _alpha: f32,
        _beta: f32,
    ) -> Result<(), mach_engine::Error> {
        Err(mach_engine::Error::BackendUnavailable(
            "gemm.gemm not implemented in P0".into(),
        ))
    }
}

crate::kernel!(
    CpuGemm,
    "gemm",
    "cpu.reference",
    BackendCaps::cpu(),
    "0.1.0"
);
impl Gemm for CpuGemm {}

/// Registers CPU reference GEMM kernels.
pub fn register_cpu(registry: &KernelRegistry) -> Result<(), RegistryError> {
    registry.register(Arc::new(CpuGemm))
}
