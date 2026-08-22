//! Stream and event primitives.
//!
//! A stream is an ordered execution queue; an event marks a point in a stream
//! that other streams can wait on. The CPU implementation is a no-op marker;
//! the CUDA path maps these onto `cudaStream_t` / `cudaEvent_t` (via cudarc)
//! behind the `cuda` feature.

use core::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic stream identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(u64);

impl StreamId {
    /// Allocates a fresh stream id.
    #[must_use]
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// The raw id value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl Default for StreamId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for StreamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stream#{}", self.0)
    }
}

/// A synchronization event, opaque on CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Event {
    /// Monotonic sequence used to order events on the host.
    seq: u64,
}

impl Event {
    /// Records a new event with the next sequence number.
    #[must_use]
    pub fn record() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self {
            seq: NEXT.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Sequence number used for host-side ordering.
    #[must_use]
    pub const fn seq(self) -> u64 {
        self.seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_and_events_are_monotonic() {
        let a = StreamId::new();
        let b = StreamId::new();
        assert_ne!(a, b);
        let e1 = Event::record();
        let e2 = Event::record();
        assert!(e2.seq() > e1.seq());
    }
}
