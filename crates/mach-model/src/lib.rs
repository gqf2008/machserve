//! MachServe model layer.
//!
//! P1 decode slice: a small transformer (attention + MLP, GQA) with a static
//! KV cache, runnable on CPU (reference) and on AMD GPUs via the HIP path.
//! The GPU decode step is structured so the kernel sequence can be captured
//! once into a HIP graph and replayed for every token — the serving pattern
//! used by TokenSpeed on CUDA.

#[cfg(feature = "hip")]
#[cfg(feature = "hip")]
pub mod batched;
pub mod config;
#[cfg(feature = "hip")]
pub mod continuous;
pub mod fp16;
pub mod loader;
pub mod ref_model;
#[cfg(feature = "hip")]
pub mod sampling;
pub mod tokenizer;
pub mod weights;

pub use config::Config;
pub use weights::{LayerWeights, Weights};

/// Errors from the model layer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("hip error: {0}")]
    #[cfg(feature = "hip")]
    Hip(#[from] mach_kernel_sys::hip::HipError),
    #[error("engine error: {0}")]
    #[cfg(feature = "hip")]
    Engine(#[from] mach_engine::Error),
    #[error("graph error: {0}")]
    #[cfg(feature = "hip")]
    Graph(#[from] mach_engine::graph::GraphError),
}

#[cfg(feature = "hip")]
pub mod kernels;
#[cfg(feature = "hip")]
pub mod model;
