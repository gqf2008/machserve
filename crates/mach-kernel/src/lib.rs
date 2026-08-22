//! MachServe kernel boundary.
//!
//! This crate is the **only** place runtime code interacts with kernels.
//! Rules:
//!
//! 1. Runtime crates depend on `mach-kernel`, never on a kernel library
//!    directly.
//! 2. All third-party kernel code lives in the repo `thirdparty/` directory
//!    and is reached exclusively through `mach-kernel-sys` FFI.
//! 3. Ops follow `<family>/<solution>` layout under [`ops`], e.g.
//!    `ops::attention::flashinfer` or `ops::gemm::cutlass`.
//! 4. Concrete kernel implementations register themselves into the
//!    [`KernelRegistry`] at startup.

pub mod backend;
pub mod buffer;
pub mod kernel;
pub mod ops;
pub mod registry;

pub use backend::{BackendCaps, BackendId};
pub use buffer::Buffer;
pub use kernel::Kernel;
pub use registry::{KernelRegistry, RegistryError};
