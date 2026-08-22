//! MachServe OpenAI-compatible HTTP server over the continuous-batching engine.

#[cfg(feature = "hip")]
pub mod engine;
#[cfg(feature = "hip")]
pub mod routes;

#[cfg(feature = "hip")]
pub use engine::ServerEngine;
#[cfg(feature = "hip")]
pub use routes::{AppState, router};
