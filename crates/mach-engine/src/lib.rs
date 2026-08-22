//! MachServe core runtime primitives.
//!
//! This crate owns the narrow set of concepts every other crate builds on:
//! devices, dtypes, shapes, memory pools, stream/event ordering and the
//! CUDA-graph capture lifecycle. It is deliberately **backend-agnostic**: the
//! default build is pure Rust (CPU, no CUDA dependency) and every hardware
//! path is gated behind the optional `cuda` feature.
//!
//! The graph-capture lifecycle mirrors the design proven in burn/cubecl
//! (`NoCapture → Prepare → Capture → NoCapture`, strict ordering, zero
//! allocation inside the capture window) while keeping the implementation
//! independent of any particular framework.

pub mod device;
pub mod dtype;
pub mod graph;
pub mod memory;
pub mod shape;
pub mod stream;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use device::Device;
pub use dtype::DType;
pub use graph::{CaptureState, GraphCapture, GraphError, GraphHandle, SoftwareGraphCapture};
pub use memory::{Allocation, MemoryPool};
pub use shape::Shape;
pub use stream::{Event, StreamId};

/// Error type used across the core crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("graph error: {0}")]
    Graph(#[from] graph::GraphError),
    #[error("memory error: {0}")]
    Memory(String),
}
