//! Build script for `mach-kernel-sys`.
//!
//! Locates third-party kernel libraries. The `MACH_THIRDPARTY` environment
//! variable points at a directory containing prebuilt libraries (e.g. a CMake
//! build dir); when absent, FFI features are expected to fail at link time,
//! which is correct: nothing links them unless the feature is enabled.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=MACH_THIRDPARTY");
    if let Ok(dir) = env::var("MACH_THIRDPARTY") {
        let dir = PathBuf::from(dir);
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:metadata=MACH_THIRDPARTY={}", dir.display());
    }
}
