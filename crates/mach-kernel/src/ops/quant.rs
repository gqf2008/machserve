//! Quantization family: FP8 quantize/dequantize and KV-cache helpers.

use crate::buffer::Buffer;
use crate::{BackendCaps, Kernel, KernelRegistry, RegistryError};
use std::sync::Arc;

/// Quantization kernel contract.
pub trait Quant: Kernel {
    /// Quantize fp16/bf16 activations to FP8 with per-token scales.
    fn quantize_fp8(
        &self,
        _src: &Buffer,
        _dst: &mut Buffer,
        _scales: &mut Buffer,
    ) -> Result<(), mach_engine::Error> {
        Err(mach_engine::Error::BackendUnavailable(
            "quant.quantize_fp8 not implemented in P0".into(),
        ))
    }
}

crate::kernel!(
    CpuQuant,
    "quant",
    "cpu.reference",
    BackendCaps::cpu(),
    "0.1.0"
);
impl Quant for CpuQuant {}

/// Registers CPU reference quantization kernels.
pub fn register_cpu(registry: &KernelRegistry) -> Result<(), RegistryError> {
    registry.register(Arc::new(CpuQuant))
}
