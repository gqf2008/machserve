//! Attention family: decode (one token per sequence) and prefill kernels.

use crate::buffer::Buffer;
use crate::{BackendCaps, Kernel, KernelRegistry, RegistryError};
use std::sync::Arc;

/// Attention kernel contract.
pub trait Attention: Kernel {
    /// Decode step: one new token per sequence against the KV cache.
    ///
    /// P1 wires real buffers; the signature is frozen as the contract.
    fn decode(
        &self,
        _q: &Buffer,
        _k: &Buffer,
        _v: &Buffer,
        _out: &mut Buffer,
    ) -> Result<(), mach_engine::Error> {
        Err(mach_engine::Error::BackendUnavailable(
            "attention.decode not implemented in P0".into(),
        ))
    }

    /// Prefill step: process a batch of input tokens.
    fn prefill(
        &self,
        _q: &Buffer,
        _k: &Buffer,
        _v: &Buffer,
        _out: &mut Buffer,
    ) -> Result<(), mach_engine::Error> {
        Err(mach_engine::Error::BackendUnavailable(
            "attention.prefill not implemented in P0".into(),
        ))
    }
}

crate::kernel!(
    CpuAttention,
    "attention",
    "cpu.reference",
    BackendCaps::cpu(),
    "0.1.0"
);
impl Attention for CpuAttention {}

/// Registers CPU reference attention kernels.
pub fn register_cpu(registry: &KernelRegistry) -> Result<(), RegistryError> {
    registry.register(Arc::new(CpuAttention))
}
