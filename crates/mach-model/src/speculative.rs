//! Speculative decoding for a single sequence.
//!
//! A small draft model proposes `k` tokens; the target model verifies them in
//! ONE batched forward (`k+1` rows: the last accepted token at position
//! `len-1`, then the drafts at `len..len+k-1`). Argmax acceptance keeps the
//! output **identical to plain greedy** on the target: the longest prefix of
//! drafts matching the target's own argmax is accepted, and the next token is
//! the target's argmax at the rejection point (or after the last draft).
//!
//! The algorithm is validated by simulation (`docs/roadmap.md` P3al) and by
//! the tiny-model parity test in `tests/spec_decode.rs`.

use crate::Error;
use crate::batched::BatchedModel;
use crate::sampling::SamplingParams;

/// Single-sequence speculative decoder (draft + target sharing one vocab).
#[allow(clippy::len_without_is_empty)]
pub struct SpeculativeDecoder {
    draft: BatchedModel,
    target: BatchedModel,
    /// Draft tokens proposed per verify round.
    k: usize,
    /// Accepted context length (prompt + accepted tokens).
    len: usize,
    /// Token at position `len - 1` (the last accepted token).
    draft_last: u32,
}

impl SpeculativeDecoder {
    /// Prefills both models with `prompt`; `k` draft tokens per round.
    pub fn new(
        mut draft: BatchedModel,
        mut target: BatchedModel,
        k: usize,
        prompt: &[u32],
    ) -> Result<Self, Error> {
        assert!(!prompt.is_empty(), "prompt must not be empty");
        assert!(k >= 1, "k must be >= 1");
        let lens: Vec<u32> = (0..prompt.len() as u32).collect();
        let slots = vec![0u32; prompt.len()];
        let empty_counts: Vec<Vec<(u32, u32)>> = vec![Vec::new(); prompt.len()];
        let empty_bias: Vec<Vec<(u32, f32)>> = vec![Vec::new(); prompt.len()];
        let mut gp = vec![SamplingParams::greedy(0); prompt.len()];
        target.decode_step_explicit(prompt, &lens, &slots, &mut gp, &empty_counts, &empty_bias)?;
        let mut dp = vec![SamplingParams::greedy(0); prompt.len()];
        draft.decode_step_explicit(prompt, &lens, &slots, &mut dp, &empty_counts, &empty_bias)?;
        Ok(Self {
            draft,
            target,
            k,
            len: prompt.len(),
            draft_last: *prompt.last().expect("non-empty"),
        })
    }

    /// One speculative round: draft `k` tokens, verify on the target, accept
    /// the longest matching prefix. Returns the accepted tokens (>= 1), so the
    /// caller accumulates until it has enough.
    pub fn step(&mut self) -> Result<Vec<u32>, Error> {
        let l = self.len;
        // 1. Draft k tokens: c[i] is the draft's guess for position l+i.
        let mut c: Vec<u32> = Vec::with_capacity(self.k);
        for i in 0..self.k {
            let input = if i == 0 { self.draft_last } else { c[i - 1] };
            let pos = (l - 1 + i) as u32;
            let mut p = [SamplingParams::greedy(0)];
            let out = self.draft.decode_step_explicit(
                &[input],
                &[pos],
                &[0],
                &mut p,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
            )?;
            c.push(out.0[0]);
        }
        // 2. Verify k+1 rows on the target: [draft_last, c[0..k-1]] at
        //    positions [l-1 .. l+k-1]; pred[i] = target's guess for position
        //    l+i (pred[0] = fresh next-token prediction).
        let mut rows = Vec::with_capacity(self.k + 1);
        rows.push(self.draft_last);
        rows.extend_from_slice(&c);
        let rlens: Vec<u32> = (0..=self.k as u32).map(|i| l as u32 - 1 + i).collect();
        let mut vp = vec![SamplingParams::greedy(0); self.k + 1];
        let vout = self.target.decode_step_explicit(
            &rows,
            &rlens,
            &vec![0u32; self.k + 1],
            &mut vp,
            &vec![Vec::new(); self.k + 1],
            &vec![Vec::new(); self.k + 1],
        )?;
        let pred = vout.0;
        // 3. Accept the longest prefix c[0..a-1] matching the target argmax.
        let mut a = 0usize;
        while a < self.k && pred[a] == c[a] {
            a += 1;
        }
        let next = pred[a];
        // 4. Advance the draft context with c[0..a] + next.
        let mut advance = c[..a].to_vec();
        advance.push(next);
        for (j, &t) in advance.iter().enumerate() {
            let mut p = [SamplingParams::greedy(0)];
            self.draft.decode_step_explicit(
                &[t],
                &[(l + j) as u32],
                &[0],
                &mut p,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
            )?;
        }
        self.len += advance.len();
        self.draft_last = *advance.last().expect("non-empty");
        Ok(advance)
    }

    /// Current accepted context length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }
}
