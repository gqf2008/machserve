//! Request-lifecycle FSM (TokenSpeed scheduler contract, ported).
//!
//! Ported from LightSeek TokenSpeed `ts-scheduler-core/src/fsm.rs` (MIT):
//! the seven request states and the event transition table. TokenSpeed's
//! C++ control plane models a request as a type-safe FSM where illegal
//! transitions throw; Rust makes the transition table exhaustive and we panic
//! on an illegal event for the same reason (`unreachable!`-style safety).
//!
//! This module is deliberately **pure**: it only tracks lifecycle state and
//! token counts. The cache side effects TokenSpeed attaches to `Finish`,
//! `Abort`, `Retraction` and `Retract` (releasing block tables / prefix-hash
//! updates / WriteBack-LoadBack) are expressed as `on_transition` hooks so a
//! future scheduler can wire the block pool / prefix cache without changing
//! the transition table.
//!
//! ```text
//! Bootstrapping --Bootstrapped--> Submitted
//! Submitted --SchedulePrefillFirstChunk--> Prefilling | PrefillDone
//! Prefilling --SchedulePrefill--> Prefilling | PrefillDone
//! PrefillDone --ScheduleDecode--> Decoding
//! Decoding --ScheduleDecode--> Decoding
//! Decoding --Retraction--> Retracted
//! PrefillDone/Decoding --Retract--> Submitted   (capacity eviction back to queue)
//! Retracted --SchedulePrefillFirstChunk--> Prefilling | PrefillDone
//! * --Finish/Abort--> Finished   Decoding --Succeeded--> Finished
//! ```

/// Lifecycle state of one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestState {
    /// Request created; host-side metadata staged.
    Bootstrapping(Bootstrapping),
    /// Tokens known; waiting for the first prefill schedule.
    Submitted(Submitted),
    /// Prefill in progress; `num_prefill_tokens` of `tokens` are computed.
    Prefilling(Prefilling),
    /// Prefix fully computed; decode may start.
    PrefillDone(PrefillDone),
    /// Decoding in progress.
    Decoding(Decoding),
    /// Decode evicted to host (capacity / PD) while retaining its token
    /// stream; resumable via prefill-first-chunk.
    Retracted(Retracted),
    /// Terminal.
    Finished,
}

/// Payloads (minimal: token counts; the scheduler attaches KV/block state).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Bootstrapping;

/// `tokens` = full request token ids, `max_new_tokens` = generation budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submitted {
    pub tokens: Vec<i32>,
    /// Generated tokens already produced before an eviction (0 for fresh).
    pub num_decoded_tokens: i32,
    pub max_new_tokens: i32,
}

/// `num_prefill_tokens` of `tokens` have been written to the KV cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prefilling {
    pub tokens: Vec<i32>,
    pub num_prefill_tokens: i32,
    pub num_decoded_tokens: i32,
    pub reserve_tokens: i32,
    pub max_new_tokens: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillDone {
    pub tokens: Vec<i32>,
    pub num_decoded_tokens: i32,
    pub reserve_tokens: i32,
    pub max_new_tokens: i32,
}

/// `num_decoded_tokens` generated tokens so far.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoding {
    pub tokens: Vec<i32>,
    pub num_decoded_tokens: i32,
    pub reserve_tokens: i32,
    pub max_new_tokens: i32,
}

/// Evicted request; the token stream (and budget) is retained for a later
/// resume, but any KV/block resources were released.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Retracted {
    pub tokens: Vec<i32>,
    pub num_decoded_tokens: i32,
    pub max_new_tokens: i32,
}

/// Events that drive the FSM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsmEvent {
    /// Bootstrapping -> Submitted.
    Bootstrapped,
    /// First prefill chunk (may complete the whole prefix in one shot).
    SchedulePrefillFirstChunk(SchedulePrefillFirstChunk),
    /// Continue an in-progress prefill.
    SchedulePrefill(SchedulePrefill),
    /// Enter or continue decode.
    ScheduleDecode(ScheduleDecode),
    /// Normal completion of a done/decoding/retracted request.
    Finish,
    /// Abort from any non-terminal state.
    Abort,
    /// Capacity/PD eviction of a decoding request into [`RequestState::Retracted`].
    Retraction,
    /// Capacity eviction that releases the request's KV resources and puts it
    /// back on the submission queue ([`RequestState::Submitted`]).
    Retract,
    /// Adjust the decode reserve (e.g. speculative window).
    UpdateReserveNumTokens(i32),
    /// A batch produced `tokens`; append to the request's token stream.
    ExtendResult(Vec<i32>),
    /// Decode finished with all `max_new_tokens` produced.
    Succeeded,
    /// A remote (PD) engine finished the prefill; `bootstrap_token` is the
    /// first decode token it appended.
    RemotePrefillDone(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulePrefillFirstChunk {
    /// Max prompt tokens this chunk computes.
    pub chunk_size: i32,
    /// Tokens kept free behind the last computed page.
    pub reserve_tokens: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulePrefill {
    /// Max prompt tokens this chunk computes.
    pub chunk_size: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleDecode {
    /// Tokens kept free behind the last page during decode.
    pub reserve_tokens: i32,
}

/// Optional side-effect hook invoked on every transition
/// `(event, from, to)`. Default is a no-op.
pub trait TransitionHook {
    fn on_transition(&mut self, event: &FsmEvent, from: &RequestState, to: &RequestState);
}

impl TransitionHook for () {
    fn on_transition(&mut self, _event: &FsmEvent, _from: &RequestState, _to: &RequestState) {}
}

fn invalid_transition(event: &str, state: &str) -> ! {
    panic!("FSM transition invalid: event=scheduler_fsm::{event}; state=scheduler_fsm::{state}");
}

impl FsmEvent {
    /// Applies the event, returning the next state. Panics on an illegal
    /// transition (mirrors TokenSpeed's `std::logic_error` for the same
    /// condition).
    #[must_use]
    pub fn apply(self, state: RequestState) -> RequestState {
        self.apply_with_hook(state, &mut ())
    }

    /// Applies the event with a caller-provided transition hook (cache
    /// side effects, e.g. releasing block tables on Finish/Abort/Retraction).
    #[must_use]
    pub fn apply_with_hook(
        self,
        state: RequestState,
        hook: &mut dyn TransitionHook,
    ) -> RequestState {
        let from = state.clone();
        let to = match self.clone() {
            FsmEvent::Bootstrapped => match state {
                RequestState::Bootstrapping(Bootstrapping) => RequestState::Submitted(Submitted {
                    tokens: Vec::new(),
                    num_decoded_tokens: 0,
                    max_new_tokens: 0,
                }),
                other => invalid_transition("Bootstrapped", other.state_name()),
            },
            FsmEvent::SchedulePrefillFirstChunk(ev) => match state {
                RequestState::Submitted(s) => first_chunk(s, ev),
                RequestState::Retracted(s) => first_chunk(
                    Submitted {
                        tokens: s.tokens,
                        num_decoded_tokens: s.num_decoded_tokens,
                        max_new_tokens: s.max_new_tokens,
                    },
                    ev,
                ),
                other => invalid_transition("SchedulePrefillFirstChunk", other.state_name()),
            },
            FsmEvent::SchedulePrefill(ev) => match state {
                RequestState::Prefilling(mut s) => {
                    s.num_prefill_tokens =
                        (s.num_prefill_tokens + ev.chunk_size).min(s.tokens.len() as i32);
                    if s.num_prefill_tokens >= s.tokens.len() as i32 {
                        RequestState::PrefillDone(PrefillDone {
                            tokens: s.tokens,
                            num_decoded_tokens: s.num_decoded_tokens,
                            reserve_tokens: s.reserve_tokens,
                            max_new_tokens: s.max_new_tokens,
                        })
                    } else {
                        RequestState::Prefilling(s)
                    }
                }
                other => invalid_transition("SchedulePrefill", other.state_name()),
            },
            FsmEvent::ScheduleDecode(ev) => match state {
                RequestState::PrefillDone(s) => RequestState::Decoding(Decoding {
                    tokens: s.tokens,
                    // A resumed (previously retracted/retracted-to-queue)
                    // request keeps its decoded count so the generation
                    // budget `max_new - num_decoded` stays correct.
                    num_decoded_tokens: s.num_decoded_tokens,
                    reserve_tokens: ev.reserve_tokens,
                    max_new_tokens: s.max_new_tokens,
                }),
                RequestState::Decoding(mut s) => {
                    s.reserve_tokens = ev.reserve_tokens;
                    RequestState::Decoding(s)
                }
                other => invalid_transition("ScheduleDecode", other.state_name()),
            },
            FsmEvent::Finish => match state {
                RequestState::PrefillDone(_)
                | RequestState::Decoding(_)
                | RequestState::Retracted(_) => RequestState::Finished,
                other => invalid_transition("Finish", other.state_name()),
            },
            FsmEvent::Abort => match state {
                RequestState::Bootstrapping(_)
                | RequestState::Submitted(_)
                | RequestState::Prefilling(_)
                | RequestState::PrefillDone(_)
                | RequestState::Decoding(_)
                | RequestState::Retracted(_) => RequestState::Finished,
                RequestState::Finished => RequestState::Finished,
            },
            FsmEvent::Retraction => match state {
                RequestState::Decoding(s) => RequestState::Retracted(Retracted {
                    tokens: s.tokens,
                    num_decoded_tokens: s.num_decoded_tokens,
                    max_new_tokens: s.max_new_tokens,
                }),
                other => invalid_transition("Retraction", other.state_name()),
            },
            FsmEvent::Retract => match state {
                // Upstream `retract()` frees the request's block tables and
                // returns to the submission queue; only the token stream (and
                // generation budget) survive.
                RequestState::PrefillDone(s) => RequestState::Submitted(Submitted {
                    tokens: s.tokens,
                    num_decoded_tokens: s.num_decoded_tokens,
                    max_new_tokens: s.max_new_tokens,
                }),
                RequestState::Decoding(s) => RequestState::Submitted(Submitted {
                    tokens: s.tokens,
                    num_decoded_tokens: s.num_decoded_tokens,
                    max_new_tokens: s.max_new_tokens,
                }),
                other => invalid_transition("Retract", other.state_name()),
            },
            FsmEvent::UpdateReserveNumTokens(v) => match state {
                RequestState::Decoding(mut s) => {
                    s.reserve_tokens = v;
                    RequestState::Decoding(s)
                }
                RequestState::Finished => RequestState::Finished,
                other => invalid_transition("UpdateReserveNumTokens", other.state_name()),
            },
            FsmEvent::ExtendResult(tokens) => match state {
                RequestState::PrefillDone(mut s) => {
                    s.tokens.extend(tokens);
                    RequestState::PrefillDone(s)
                }
                RequestState::Decoding(mut s) => {
                    s.num_decoded_tokens += tokens.len() as i32;
                    s.tokens.extend(tokens);
                    RequestState::Decoding(s)
                }
                RequestState::Finished => RequestState::Finished,
                other => invalid_transition("ExtendResult", other.state_name()),
            },
            FsmEvent::Succeeded => match state {
                RequestState::Decoding(_) => RequestState::Finished,
                other => invalid_transition("Succeeded", other.state_name()),
            },
            FsmEvent::RemotePrefillDone(bootstrap_token) => match state {
                // Upstream extends the token container with the bootstrap
                // token (the first decode token of the remote engine).
                RequestState::Prefilling(mut s) => {
                    s.tokens.push(bootstrap_token);
                    RequestState::PrefillDone(PrefillDone {
                        tokens: s.tokens,
                        num_decoded_tokens: s.num_decoded_tokens,
                        reserve_tokens: s.reserve_tokens,
                        max_new_tokens: s.max_new_tokens,
                    })
                }
                other => invalid_transition("RemotePrefillDone", other.state_name()),
            },
        };
        hook.on_transition(&self, &from, &to);
        to
    }
}

impl RequestState {
    /// Stable name for diagnostics / panic messages.
    #[must_use]
    pub fn state_name(&self) -> &'static str {
        match self {
            RequestState::Bootstrapping(_) => "Bootstrapping",
            RequestState::Submitted(_) => "Submitted",
            RequestState::Prefilling(_) => "Prefilling",
            RequestState::PrefillDone(_) => "PrefillDone",
            RequestState::Decoding(_) => "Decoding",
            RequestState::Retracted(_) => "Retracted",
            RequestState::Finished => "Finished",
        }
    }

    /// Prefix tokens computed so far (0 unless prefilling/done).
    #[must_use]
    pub fn num_prefill_tokens(&self) -> i32 {
        match self {
            RequestState::Prefilling(s) => s.num_prefill_tokens,
            RequestState::PrefillDone(s) => s.tokens.len() as i32,
            _ => 0,
        }
    }
}

/// Submitted -> Prefilling | PrefillDone for the first chunk.
fn first_chunk(s: Submitted, ev: SchedulePrefillFirstChunk) -> RequestState {
    let total = s.tokens.len() as i32;
    let chunk = ev.chunk_size.min(total);
    if chunk >= total {
        RequestState::PrefillDone(PrefillDone {
            tokens: s.tokens,
            num_decoded_tokens: s.num_decoded_tokens,
            reserve_tokens: ev.reserve_tokens,
            max_new_tokens: s.max_new_tokens,
        })
    } else {
        RequestState::Prefilling(Prefilling {
            tokens: s.tokens,
            num_prefill_tokens: chunk,
            num_decoded_tokens: s.num_decoded_tokens,
            reserve_tokens: ev.reserve_tokens,
            max_new_tokens: s.max_new_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bootstrapped(tokens: &[i32], max_new: i32) -> RequestState {
        FsmEvent::Bootstrapped
            .apply(RequestState::Bootstrapping(Bootstrapping))
            .pipe(|mut s| {
                if let RequestState::Submitted(su) = &mut s {
                    su.tokens = tokens.to_vec();
                    su.max_new_tokens = max_new;
                }
                s
            })
    }

    fn first_chunk(state: RequestState, chunk_size: i32, reserve: i32) -> RequestState {
        FsmEvent::SchedulePrefillFirstChunk(SchedulePrefillFirstChunk {
            chunk_size,
            reserve_tokens: reserve,
        })
        .apply(state)
    }

    fn to_decode(state: RequestState, reserve: i32) -> RequestState {
        FsmEvent::ScheduleDecode(ScheduleDecode {
            reserve_tokens: reserve,
        })
        .apply(state)
    }

    /// Tiny pipe helper to keep tests readable.
    trait Pipe: Sized {
        fn pipe<F: FnOnce(Self) -> Self>(self, f: F) -> Self {
            f(self)
        }
    }
    impl<T: Sized> Pipe for T {}

    #[test]
    fn submitted_first_chunk_small_fits_in_one() {
        // tokens fit one chunk -> PrefillDone directly.
        let s = bootstrapped(&[1, 2, 3, 4], 16);
        let next = first_chunk(s, 8, 2);
        assert!(matches!(next, RequestState::PrefillDone(_)));
        assert_eq!(next.num_prefill_tokens(), 4);
    }

    #[test]
    fn submitted_first_chunk_large_starts_prefilling() {
        let s = bootstrapped(&[1, 2, 3, 4, 5, 6, 7, 8], 16);
        let next = first_chunk(s, 4, 2);
        let RequestState::Prefilling(p) = &next else {
            panic!("expected Prefilling, got {}", next.state_name());
        };
        assert_eq!(p.num_prefill_tokens, 4);
        assert_eq!(p.tokens.len(), 8);
        assert_eq!(p.max_new_tokens, 16);
    }

    #[test]
    fn prefill_continues_until_done() {
        let s = bootstrapped(&[1, 2, 3, 4, 5, 6, 7, 8], 16);
        let mut state = first_chunk(s, 4, 2);
        state = FsmEvent::SchedulePrefill(SchedulePrefill { chunk_size: 4 }).apply(state);
        assert!(matches!(state, RequestState::PrefillDone(_)));
        assert_eq!(state.num_prefill_tokens(), 8);
    }

    #[test]
    fn prefill_done_then_decode() {
        let s = bootstrapped(&[1, 2], 16);
        let done = first_chunk(s, 8, 0);
        assert!(matches!(done, RequestState::PrefillDone(_)));
        let decoding = to_decode(done, 4);
        let RequestState::Decoding(d) = &decoding else {
            panic!("expected Decoding");
        };
        assert_eq!(d.num_decoded_tokens, 0);
        assert_eq!(d.reserve_tokens, 4);
        assert_eq!(d.max_new_tokens, 16);
        // ExtendResult appends + advances the decoded count.
        let decoding = FsmEvent::ExtendResult(vec![9, 10]).apply(decoding);
        let RequestState::Decoding(d) = &decoding else {
            panic!("expected Decoding");
        };
        assert_eq!(d.num_decoded_tokens, 2);
        // Succeeded -> Finished.
        let finished = FsmEvent::Succeeded.apply(decoding);
        assert_eq!(finished, RequestState::Finished);
    }

    #[test]
    fn retraction_evicts_to_retracted_and_resumes() {
        let s = bootstrapped(&[1, 2], 16);
        let done = first_chunk(s, 8, 0);
        let mut decoding = to_decode(done, 4);
        decoding = FsmEvent::ExtendResult(vec![9, 10]).apply(decoding);
        let retracted = FsmEvent::Retraction.apply(decoding);
        let RequestState::Retracted(r) = &retracted else {
            panic!("expected Retracted");
        };
        assert_eq!(r.max_new_tokens, 16);
        assert_eq!(
            r.num_decoded_tokens, 2,
            "Retraction must preserve the decoded count for the remaining budget"
        );
        // Resume via SchedulePrefillFirstChunk (prefix already computed) and
        // decode again: the decoded count must survive into Decoding.
        let resumed = first_chunk(retracted, 8, 4);
        let RequestState::PrefillDone(d) = &resumed else {
            panic!("expected PrefillDone");
        };
        assert_eq!(d.num_decoded_tokens, 2);
        let redecoding = to_decode(resumed, 4);
        let RequestState::Decoding(dd) = &redecoding else {
            panic!("expected Decoding");
        };
        assert_eq!(
            dd.num_decoded_tokens, 2,
            "resumed decode must keep the pre-retraction decoded count"
        );
    }

    #[test]
    fn retract_to_queue_keeps_decoded_count() {
        // Capacity eviction (Retract) also must not lose the decoded count:
        // the request re-prefills its full stream and continues with the
        // remaining budget `max_new - num_decoded`.
        let s = bootstrapped(&[1, 2, 3, 4], 16);
        let done = first_chunk(s, 8, 0);
        let decoding = FsmEvent::ExtendResult(vec![9]).apply(to_decode(done, 4));
        let RequestState::Submitted(sub) = FsmEvent::Retract.apply(decoding) else {
            panic!("expected Submitted");
        };
        assert_eq!(sub.num_decoded_tokens, 1);
        let redecoding = to_decode(first_chunk(RequestState::Submitted(sub), 8, 4), 4);
        let RequestState::Decoding(d) = &redecoding else {
            panic!("expected Decoding");
        };
        assert_eq!(d.num_decoded_tokens, 1);
    }

    #[test]
    fn apply_with_hook_invokes_the_hook() {
        // The transition hook must actually fire when supplied (regression:
        // it was hard-wired to a no-op).
        #[derive(Debug)]
        struct Recorder {
            events: Vec<String>,
        }
        impl TransitionHook for Recorder {
            fn on_transition(
                &mut self,
                event: &FsmEvent,
                _from: &RequestState,
                _to: &RequestState,
            ) {
                self.events.push(format!("{event:?}"));
            }
        }
        let mut rec = Recorder { events: Vec::new() };
        let s = bootstrapped(&[1, 2], 16);
        let done = FsmEvent::SchedulePrefillFirstChunk(SchedulePrefillFirstChunk {
            chunk_size: 8,
            reserve_tokens: 0,
        })
        .apply_with_hook(s, &mut rec);
        assert!(
            rec.events
                .iter()
                .any(|e| e.contains("SchedulePrefillFirstChunk")),
            "hook must be invoked: {rec:?}"
        );
        let _ = done;
    }

    #[test]
    fn retract_returns_to_submitted_queue() {
        // Capacity eviction returns the request to Submitted (upstream
        // `retract()` semantics), so it can be rescheduled from scratch.
        let s = bootstrapped(&[1, 2, 3, 4], 16);
        let done = first_chunk(s, 8, 0);
        let decoding = to_decode(done, 4);
        let submitted = FsmEvent::Retract.apply(decoding);
        let RequestState::Submitted(su) = &submitted else {
            panic!("expected Submitted, got {}", submitted.state_name());
        };
        assert_eq!(su.tokens, vec![1, 2, 3, 4]);
        assert_eq!(su.max_new_tokens, 16);
        // Rescheduled: first chunk recomputes the prefix.
        let redone = first_chunk(submitted, 8, 0);
        assert!(matches!(redone, RequestState::PrefillDone(_)));
    }

    #[test]
    fn retract_from_prefill_done_also_returns_to_submitted() {
        let s = bootstrapped(&[1, 2], 16);
        let done = first_chunk(s, 8, 0);
        let submitted = FsmEvent::Retract.apply(done);
        assert!(matches!(submitted, RequestState::Submitted(_)));
    }

    #[test]
    fn update_reserve_and_remote_prefill_done() {
        let s = bootstrapped(&[1, 2, 3], 16);
        let prefilling = first_chunk(s, 2, 0);
        let RequestState::Prefilling(p) = &prefilling else {
            panic!("expected Prefilling");
        };
        assert_eq!(p.num_prefill_tokens, 2);
        // Remote engine finishes the prefill and appends its bootstrap token.
        let done = FsmEvent::RemotePrefillDone(42).apply(prefilling);
        let RequestState::PrefillDone(d) = &done else {
            panic!("expected PrefillDone");
        };
        assert_eq!(d.tokens, vec![1, 2, 3, 42]);
        // Reserve can be adjusted while decoding.
        let decoding = to_decode(done, 0);
        let decoding = FsmEvent::UpdateReserveNumTokens(8).apply(decoding);
        let RequestState::Decoding(d) = &decoding else {
            panic!("expected Decoding");
        };
        assert_eq!(d.reserve_tokens, 8);
    }

    #[test]
    fn abort_and_finish_reach_terminal() {
        for state in [
            RequestState::Bootstrapping(Bootstrapping),
            bootstrapped(&[1], 4),
            first_chunk(bootstrapped(&[1, 2, 3, 4, 5, 6, 7, 8], 4), 4, 0),
        ] {
            assert_eq!(FsmEvent::Abort.apply(state), RequestState::Finished);
        }
        let s = bootstrapped(&[1], 4);
        let done = first_chunk(s, 8, 0);
        assert_eq!(FsmEvent::Finish.apply(done), RequestState::Finished);
    }

    #[test]
    #[should_panic(
        expected = "FSM transition invalid: event=scheduler_fsm::ScheduleDecode; state=scheduler_fsm::Submitted"
    )]
    fn illegal_schedule_decode_from_submitted_panics() {
        let s = bootstrapped(&[1, 2], 8);
        let _ = FsmEvent::ScheduleDecode(ScheduleDecode { reserve_tokens: 0 }).apply(s);
    }

    #[test]
    #[should_panic(
        expected = "FSM transition invalid: event=scheduler_fsm::Finish; state=scheduler_fsm::Bootstrapping"
    )]
    fn illegal_finish_from_bootstrapping_panics() {
        let _ = FsmEvent::Finish.apply(RequestState::Bootstrapping(Bootstrapping));
    }

    #[test]
    #[should_panic(
        expected = "FSM transition invalid: event=scheduler_fsm::Retraction; state=scheduler_fsm::PrefillDone"
    )]
    fn illegal_retraction_from_prefill_done_panics() {
        let s = bootstrapped(&[1], 4);
        let done = first_chunk(s, 8, 0);
        let _ = FsmEvent::Retraction.apply(done);
    }

    #[test]
    #[should_panic(
        expected = "FSM transition invalid: event=scheduler_fsm::Succeeded; state=scheduler_fsm::PrefillDone"
    )]
    fn illegal_succeeded_from_prefill_done_panics() {
        let s = bootstrapped(&[1], 4);
        let done = first_chunk(s, 8, 0);
        let _ = FsmEvent::Succeeded.apply(done);
    }
}
