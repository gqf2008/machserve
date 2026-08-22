//! Sampling family: fused sampling kernels (optional; host-side fallback exists).

use crate::buffer::Buffer;
use crate::{BackendCaps, Kernel, KernelRegistry, RegistryError};
use std::sync::Arc;

/// Sampling kernel contract.
pub trait Sampling: Kernel {
    /// Fused top-k/top-p sampling over logits.
    fn sample(&self, _logits: &Buffer, _out: &mut Buffer) -> Result<(), mach_engine::Error> {
        Err(mach_engine::Error::BackendUnavailable(
            "sampling.sample not implemented in P0".into(),
        ))
    }
}

crate::kernel!(
    CpuSampling,
    "sampling",
    "cpu.reference",
    BackendCaps::cpu(),
    "0.1.0"
);
impl Sampling for CpuSampling {}

/// Registers CPU reference sampling kernels.
pub fn register_cpu(registry: &KernelRegistry) -> Result<(), RegistryError> {
    registry.register(Arc::new(CpuSampling))
}
