//! CUDA backend placeholders (P0).
//!
//! This module is compiled only when the `cuda` feature is enabled. In P0 it
//! defines the type layout for the CUDA memory pool and graph capture; the
//! actual cudarc-backed implementation lands in P1 when a CUDA toolkit is
//! available in the build environment.
//!
//! Planned mapping (design frozen in `docs/roadmap.md`):
//!
//! - `CudaMemoryPool`  → caching allocator over `cuMemAllocAsync` with a
//!   persistent pool for graph capture, mirroring `mach-engine::memory`.
//! - `CudaGraphCapture` → `cuStreamBeginCapture` / `cuStreamEndCapture` /
//!   `cuGraphInstantiate` / `cuGraphLaunch`, with the strict
//!   `NoCapture → Prepare → Capture` lifecycle and memory-node rejection.

use crate::graph::{GraphCapture, GraphError, GraphHandle};

/// CUDA memory pool placeholder (see module docs).
#[derive(Debug, Default)]
pub struct CudaMemoryPool;

/// CUDA graph capture placeholder (see module docs).
#[derive(Debug, Default)]
pub struct CudaGraphCapture;

impl GraphCapture for CudaGraphCapture {
    fn supported(&self) -> bool {
        false
    }

    fn prepare(&self) -> Result<(), GraphError> {
        Err(GraphError::Unsupported)
    }

    fn begin(&self) -> Result<(), GraphError> {
        Err(GraphError::Unsupported)
    }

    fn end(&self) -> Result<Box<dyn GraphHandle>, GraphError> {
        Err(GraphError::Unsupported)
    }

    fn abort(&self) -> Result<(), GraphError> {
        Ok(())
    }
}
