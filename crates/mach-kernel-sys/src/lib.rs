//! MachServe FFI boundary.
//!
//! This crate is the **only** place the Rust runtime talks to vendor GPU
//! libraries. The HIP side dynamically loads `amdhip64_6.dll` (fallback
//! `amdhip64.dll`), `hiprtc0602.dll` and `hipblas.dll` at runtime — no
//! link-time dependency; the ROCm bin directory can be overridden with
//! `MACH_HIP_PATH`.

#[cfg(feature = "hip")]
pub mod hip;

#[cfg(feature = "hip")]
pub mod hipblas;

/// Version of the FFI contract (bump on breaking ABI changes).
pub const FFI_VERSION: u32 = 1;
