//! CUDA-graph capture lifecycle and replay.
//!
//! The lifecycle mirrors the design proven in burn/cubecl: a strict
//! `NoCapture → Prepare → Capture → NoCapture` progression, a mandatory warmup
//! run that primes the persistent memory pool, and a capture window that must
//! allocate nothing (an allocation mid-capture becomes a memory node and the
//! resulting graph cannot be relaunched).
//!
//! [`SoftwareGraphCapture`] is the reference implementation used on CPU and in
//! tests: it records operation descriptors instead of launching kernels and
//! replays them by name. The CUDA implementation (behind the `cuda` feature)
//! drives the real driver API through cudarc.

mod software;

pub use software::SoftwareGraphCapture;

use core::fmt;

/// Errors produced by the graph-capture lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GraphError {
    #[error("capture must be prepared before starting")]
    NotPrepared,
    #[error("a capture is already prepared or recording on this stream")]
    AlreadyActive,
    #[error("no capture is recording on this stream")]
    NotRecording,
    #[error("capture allocated inside the recording window (memory node): {0}")]
    AllocatedDuringCapture(String),
    #[error("backend does not support hardware graph capture")]
    Unsupported,
}

/// Where a stream sits in the graph-capture lifecycle.
///
/// Only this type is allowed to move the state: transitions are strict and
/// out-of-order calls are rejected, so a capture can never start unprepared
/// and two captures can never overlap on one stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaptureState {
    /// No capture is prepared or recording.
    #[default]
    NoCapture,
    /// `prepare` has armed the persistent pools for the warmup run.
    Prepare,
    /// Launches are being recorded into a graph instead of executing.
    Capture,
}

impl CaptureState {
    /// `NoCapture → Prepare`.
    pub fn prepare(self) -> Result<Self, GraphError> {
        match self {
            Self::NoCapture => Ok(Self::Prepare),
            Self::Prepare | Self::Capture => Err(GraphError::AlreadyActive),
        }
    }

    /// `Prepare → Capture`.
    pub fn begin(self) -> Result<Self, GraphError> {
        match self {
            Self::Prepare => Ok(Self::Capture),
            Self::NoCapture => Err(GraphError::NotPrepared),
            Self::Capture => Err(GraphError::AlreadyActive),
        }
    }

    /// `Capture → NoCapture`.
    pub fn end(self) -> Result<Self, GraphError> {
        match self {
            Self::Capture => Ok(Self::NoCapture),
            Self::NoCapture | Self::Prepare => Err(GraphError::NotRecording),
        }
    }

    /// Returns to `NoCapture` from anywhere, for the failure path.
    #[must_use]
    pub fn abort(self) -> Self {
        Self::NoCapture
    }

    /// Whether launches are currently being recorded.
    #[must_use]
    pub const fn is_recording(self) -> bool {
        matches!(self, Self::Capture)
    }
}

/// A captured, replayable graph.
///
/// Replay re-runs the recorded operations against the exact buffers captured;
/// the caller is responsible for keeping those buffers alive and refreshing
/// inputs in place between replays.
pub trait GraphHandle: Send + Sync {
    /// Replays the captured graph once. `unsafe` because nothing tracks
    /// whether the captured buffers are still valid.
    ///
    /// # Safety
    ///
    /// Every buffer the graph touches must still be alive and must not be
    /// concurrently read or written by another stream/thread while the replay
    /// executes.
    unsafe fn replay(&self) -> Result<(), GraphError>;
}

/// Backend graph-capture contract.
///
/// The caller drives the sequence `prepare → (warmup runs) → begin → (record
/// the workload) → end`, then keeps the returned [`GraphHandle`] and replays it.
pub trait GraphCapture: Send + Sync {
    /// Whether this backend supports hardware graph capture.
    fn supported(&self) -> bool;

    /// Arms the persistent pool for an upcoming capture. Call before warmup.
    fn prepare(&self) -> Result<(), GraphError>;

    /// Opens the recording window. Must follow [`prepare`](Self::prepare).
    fn begin(&self) -> Result<(), GraphError>;

    /// Closes the window and instantiates the graph.
    ///
    /// A capture that allocated inside the window must be rejected, since a
    /// graph holding a memory node cannot be relaunched.
    fn end(&self) -> Result<Box<dyn GraphHandle>, GraphError>;

    /// Aborts a capture that failed to open, returning the stream to normal.
    fn abort(&self) -> Result<(), GraphError>;
}

impl fmt::Display for CaptureState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::NoCapture => "no-capture",
            Self::Prepare => "prepare",
            Self::Capture => "capture",
        };
        f.write_str(s)
    }
}
