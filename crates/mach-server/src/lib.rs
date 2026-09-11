//! MachServe OpenAI-compatible HTTP server over the continuous-batching engine.

#[cfg(feature = "hip")]
pub mod engine;
pub mod multimodal;
#[cfg(feature = "hip")]
pub mod routes;

#[cfg(feature = "hip")]
pub use engine::{ImageRuntimeConfig, ServerEngine, VisionSetup};
#[cfg(feature = "hip")]
pub use routes::{AppState, ChatFormat, router};
