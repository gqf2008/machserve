//! Kernel registry: a process-wide, thread-safe map of registered kernels.

use crate::Kernel;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Errors produced by the registry.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RegistryError {
    #[error("kernel already registered: {family}/{name}")]
    AlreadyRegistered { family: String, name: String },
    #[error("kernel not found: {family}/{name}")]
    NotFound { family: String, name: String },
}

/// Process-wide kernel registry.
///
/// Kernels register themselves at startup; the engine looks them up by
/// `(family, name)` and dispatches through the [`crate::ops`] traits.
#[derive(Default)]
pub struct KernelRegistry {
    inner: RwLock<HashMap<(String, String), Arc<dyn Kernel>>>,
}

impl core::fmt::Debug for KernelRegistry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KernelRegistry")
            .field("len", &self.len())
            .finish()
    }
}

impl KernelRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a kernel, rejecting duplicates.
    pub fn register(&self, kernel: Arc<dyn Kernel>) -> Result<(), RegistryError> {
        let key = (kernel.family().to_string(), kernel.name().to_string());
        let mut map = self.inner.write().unwrap();
        if map.contains_key(&key) {
            return Err(RegistryError::AlreadyRegistered {
                family: key.0,
                name: key.1,
            });
        }
        map.insert(key, kernel);
        Ok(())
    }

    /// Looks up a kernel by family and name.
    pub fn get(&self, family: &str, name: &str) -> Result<Arc<dyn Kernel>, RegistryError> {
        self.inner
            .read()
            .unwrap()
            .get(&(family.to_string(), name.to_string()))
            .cloned()
            .ok_or_else(|| RegistryError::NotFound {
                family: family.to_string(),
                name: name.to_string(),
            })
    }

    /// All registered kernels (for diagnostics and benchmark enumeration).
    #[must_use]
    pub fn all(&self) -> Vec<Arc<dyn Kernel>> {
        self.inner.read().unwrap().values().cloned().collect()
    }

    /// Number of registered kernels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendCaps, kernel};

    kernel!(
        RegKernel,
        "attention",
        "cpu.reference",
        BackendCaps::cpu(),
        "0.1.0"
    );

    #[test]
    fn register_get_reject_duplicate() {
        let reg = KernelRegistry::new();
        reg.register(Arc::new(RegKernel)).unwrap();
        assert_eq!(reg.len(), 1);
        assert!(reg.get("attention", "cpu.reference").is_ok());
        assert!(reg.get("attention", "missing").is_err());
        assert!(reg.register(Arc::new(RegKernel)).is_err());
    }
}
