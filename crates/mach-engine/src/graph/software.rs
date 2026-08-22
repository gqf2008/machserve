//! Software (reference) graph capture.
//!
//! Records operation descriptors instead of launching kernels. Used as the
//! CPU/CI reference implementation and as the fallback for backends without
//! hardware graph support.

use super::{CaptureState, GraphCapture, GraphError, GraphHandle};
use std::sync::{Arc, Mutex};

/// Replay sink: invoked once per recorded op during a software replay.
type ReplaySink = Box<dyn Fn(&RecordedOp) + Send + Sync>;

/// A recorded operation descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedOp {
    /// Operation name, e.g. `"flashinfer.decode"`.
    pub name: String,
    /// Number of arguments this op consumed (for sanity checks).
    pub arity: usize,
}

/// A software-captured graph; replay emits the recorded op names to the sink.
pub struct SoftwareGraph {
    ops: Vec<RecordedOp>,
    sink: Arc<Mutex<Option<ReplaySink>>>,
}

impl core::fmt::Debug for SoftwareGraph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SoftwareGraph")
            .field("ops", &self.ops)
            .finish_non_exhaustive()
    }
}

impl GraphHandle for SoftwareGraph {
    unsafe fn replay(&self) -> Result<(), GraphError> {
        let sink = self.sink.lock().unwrap();
        for op in &self.ops {
            if let Some(f) = sink.as_ref() {
                f(op);
            }
        }
        Ok(())
    }
}

/// Software capture backend. `supported()` is `false` by contract: the caller
/// uses it as an explicit fallback, never as a hardware acceleration.
#[derive(Default)]
pub struct SoftwareGraphCapture {
    state: Mutex<CaptureState>,
    ops: Mutex<Vec<RecordedOp>>,
    sink: Arc<Mutex<Option<ReplaySink>>>,
}

impl core::fmt::Debug for SoftwareGraphCapture {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SoftwareGraphCapture")
            .finish_non_exhaustive()
    }
}

impl SoftwareGraphCapture {
    /// Creates a fresh software capture backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs a callback invoked for each recorded op on replay.
    pub fn set_sink(&self, sink: Option<ReplaySink>) {
        *self.sink.lock().unwrap() = sink;
    }

    /// Records an operation into the current capture window (no-op otherwise).
    pub fn record(&self, name: impl Into<String>, arity: usize) -> Result<(), GraphError> {
        if self.state.lock().unwrap().is_recording() {
            self.ops.lock().unwrap().push(RecordedOp {
                name: name.into(),
                arity,
            });
        }
        Ok(())
    }
}

impl GraphCapture for SoftwareGraphCapture {
    fn supported(&self) -> bool {
        false
    }

    fn prepare(&self) -> Result<(), GraphError> {
        let mut state = self.state.lock().unwrap();
        *state = state.prepare()?;
        Ok(())
    }

    fn begin(&self) -> Result<(), GraphError> {
        let mut state = self.state.lock().unwrap();
        *state = state.begin()?;
        self.ops.lock().unwrap().clear();
        Ok(())
    }

    fn end(&self) -> Result<Box<dyn GraphHandle>, GraphError> {
        let mut state = self.state.lock().unwrap();
        let next = state.end()?;
        *state = next;
        let ops = core::mem::take(&mut *self.ops.lock().unwrap());
        Ok(Box::new(SoftwareGraph {
            ops,
            sink: Arc::clone(&self.sink),
        }))
    }

    fn abort(&self) -> Result<(), GraphError> {
        let mut state = self.state.lock().unwrap();
        *state = state.abort();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CaptureState;

    #[test]
    fn lifecycle_transitions_are_strict() {
        let mut s = CaptureState::NoCapture;
        assert!(s.begin().is_err());
        assert!(s.end().is_err());

        s = s.prepare().unwrap();
        assert_eq!(s, CaptureState::Prepare);
        assert!(s.prepare().is_err());
        assert!(s.end().is_err());

        s = s.begin().unwrap();
        assert_eq!(s, CaptureState::Capture);
        assert!(s.begin().is_err());
        assert!(s.prepare().is_err());

        s = s.end().unwrap();
        assert_eq!(s, CaptureState::NoCapture);
    }

    #[test]
    fn software_capture_records_and_replays() {
        let cap = SoftwareGraphCapture::new();
        cap.prepare().unwrap();
        // Warmup run: ops recorded outside the window are dropped.
        cap.record("warmup.op", 1).unwrap();
        cap.begin().unwrap();
        cap.record("flashinfer.decode", 3).unwrap();
        cap.record("cutlass.gemm_fp8", 2).unwrap();
        let graph = cap.end().unwrap();

        // Replay must invoke the sink in recorded order, repeatedly.
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink_seen = Arc::clone(&seen);
        cap.set_sink(Some(Box::new(move |op| {
            sink_seen.lock().unwrap().push(op.name.clone());
        })));
        // SAFETY: test-only graph over owned descriptors; no external buffers.
        unsafe { graph.replay().unwrap() };
        unsafe { graph.replay().unwrap() };
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[
                "flashinfer.decode",
                "cutlass.gemm_fp8",
                "flashinfer.decode",
                "cutlass.gemm_fp8"
            ]
        );
    }

    #[test]
    fn capture_that_never_ends_can_be_aborted() {
        let cap = SoftwareGraphCapture::new();
        cap.prepare().unwrap();
        cap.begin().unwrap();
        cap.abort().unwrap();
        cap.prepare().unwrap();
    }
}
