//! MoE family: expert routing and grouped expert GEMMs.

use crate::buffer::Buffer;
use crate::{BackendCaps, Kernel, KernelRegistry, RegistryError};
use std::sync::Arc;

/// MoE kernel contract.
pub trait Moe: Kernel {
    /// Grouped expert GEMM for a MoE layer.
    fn grouped_expert_gemm(
        &self,
        _input: &Buffer,
        _weights: &Buffer,
        _out: &mut Buffer,
    ) -> Result<(), mach_engine::Error> {
        Err(mach_engine::Error::BackendUnavailable(
            "moe.grouped_expert_gemm not implemented in P0".into(),
        ))
    }
}

crate::kernel!(CpuMoe, "moe", "cpu.reference", BackendCaps::cpu(), "0.1.0");
impl Moe for CpuMoe {}

/// Registers CPU reference MoE kernels.
pub fn register_cpu(registry: &KernelRegistry) -> Result<(), RegistryError> {
    registry.register(Arc::new(CpuMoe))
}
