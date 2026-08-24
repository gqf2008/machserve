//! MachServe model layer.
//!
//! P1 decode slice: a small transformer (attention + MLP, GQA) with a static
//! KV cache, runnable on CPU (reference) and on AMD GPUs via the HIP path.
//! The GPU decode step is structured so the kernel sequence can be captured
//! once into a HIP graph and replayed for every token — the serving pattern
//! used by TokenSpeed on CUDA.

#[cfg(feature = "hip")]
pub mod adaptive;
#[cfg(feature = "hip")]
#[cfg(feature = "hip")]
pub mod batched;
pub mod config;
#[cfg(feature = "hip")]
pub mod continuous;
pub mod fp16;
pub mod loader;
pub mod moe_backend;
pub mod moe_offload;
pub mod prefill_buffered;
pub mod q4;
pub mod ref_model;
#[cfg(feature = "hip")]
pub mod sampling;
#[cfg(feature = "hip")]
pub mod speculative;
pub mod state_reuse;
pub mod tokenizer;
pub mod weights;

pub use config::Config;
use std::path::PathBuf;
pub use weights::{LayerWeights, LayerWeightsQ4, Weights, WeightsQ4};

/// Opt-in path for real-model integration tests. Returns the configured model
/// path only when MACH_TEST_MODEL is set and exists; real-model tests skip
/// otherwise (they load multi-GB weights, so default-off prevents accidental
/// loads during a plain cargo test).
#[doc(hidden)]
pub fn real_test_model_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("MACH_TEST_MODEL").ok()?);
    p.exists().then_some(p)
}
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
