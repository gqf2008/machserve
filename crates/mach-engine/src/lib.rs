//! MachServe core runtime primitives.
//!
//! After the batch-1 dead-code cleanup this crate retains exactly one
//! subsystem: the graph-capture lifecycle ([`graph`]) plus its HIP
//! implementation ([`hip`], behind the `hip` feature). It is slated for
//! removal in refactor batch 3, when `hip_arch()` moves into
//! `mach-kernel-sys` and the HIP graph experiment surface is dropped.

pub mod graph;

#[cfg(feature = "hip")]
pub mod hip;

pub use graph::{CaptureState, GraphCapture, GraphError, GraphHandle};

/// Error type used across the core crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("graph error: {0}")]
    Graph(#[from] graph::GraphError),

    #[cfg(feature = "hip")]
    #[error("hip error: {0}")]
    Hip(#[from] mach_kernel_sys::hip::HipError),
}
