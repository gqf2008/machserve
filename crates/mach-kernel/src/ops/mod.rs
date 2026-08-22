//! Kernel op families.
//!
//! Each family defines the trait the engine calls and registers concrete
//! implementations (`<solution>`), following the `<family>/<solution>` layout:
//! `attention/flashinfer`, `gemm/cutlass`, `moe/trtllm`, etc. P0 ships the
//! trait contracts plus CPU reference descriptors; real kernel wiring happens
//! in P1+ via `mach-kernel-sys`.

pub mod attention;
pub mod gemm;
pub mod moe;
pub mod quant;
pub mod sampling;

use crate::{KernelRegistry, RegistryError};

/// Registers every CPU reference kernel into `registry`.
pub fn register_cpu_reference(registry: &KernelRegistry) -> Result<(), RegistryError> {
    attention::register_cpu(registry)?;
    gemm::register_cpu(registry)?;
    moe::register_cpu(registry)?;
    quant::register_cpu(registry)?;
    sampling::register_cpu(registry)?;
    Ok(())
}
