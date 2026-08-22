//! Declared FFI entry points into third-party kernel libraries.
//!
//! Each block is gated behind the feature matching its library and requires
//! that library to be built from `thirdparty/` (see the repo README for the
//! build flow). The signatures below are the **P1 contract** and will be
//! finalized against the actual pinned library versions.

/// flashinfer attention entry points (feature `flashinfer`).
///
/// P1: verify against the pinned flashinfer tag in `thirdparty/`.
#[cfg(feature = "flashinfer")]
pub mod flashinfer {
    // Example of the intended binding shape. Not compiled in default builds.
    #[link(name = "flashinfer")]
    extern "C" {
        /// Returns the flashinfer library version string.
        pub fn flashinfer_version() -> *const std::ffi::c_char;
    }

    /// Safety wrapper for `flashinfer_version`.
    ///
    /// # Safety
    ///
    /// The returned pointer is a NUL-terminated static string owned by the
    /// library; valid for the lifetime of the loaded library.
    pub unsafe fn version() -> String {
        unsafe {
            let p = flashinfer_version();
            if p.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }
}

/// cutlass entry points (feature `cutlass`).
#[cfg(feature = "cutlass")]
pub mod cutlass {
    // P1: declare GEMM launch entry points here.
}
