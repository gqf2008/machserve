//! MachServe FFI boundary.
//!
//! This crate is the **only** place the Rust runtime links against third-party
//! C/CUDA kernel libraries. Contract:
//!
//! - Every linked library must live under the repo `thirdparty/` directory.
//! - Bindings are written by hand (or generated with bindgen and reviewed);
//!   each `extern "C"` block is feature-gated on the matching thirdparty
//!   component being built.
//! - `mach-kernel` (the op layer) depends on this crate; runtime crates must
//!   not.
//!
//! See [`ffi`] for the declared entry points and `thirdparty/README.md` for
//! the pinning policy.

pub mod ffi;

/// Version of the FFI contract (bump on breaking ABI changes).
pub const FFI_VERSION: u32 = 1;
