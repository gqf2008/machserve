# Benchmark: Agentic State Reuse (multi-turn TTFT A/B)

Corresponds to P3 issue #3 checklist item "多轮 TTFT 降幅数据（对标 FreeToken 65–80%）".

## Goal

In a multi-turn tool-calling / CoT conversation, each new user turn re-sends the
whole chat history. Without reuse the engine re-prefills that shared prefix on
every turn; with **agentic state reuse** (`ContinuousModel::with_state_reuse`,
`crates/mach-model/src/state_reuse.rs`) the engine restores the turn-boundary
anchor (per-layer KV prefix + hidden state) and prefills only the delta.

The benchmark measures the turn-2 **TTFT** (request → first generated token)
with and without reuse on the same request, on `qwen3-moe-tiny` (real MoE
checkpoint, `.models/model.safetensors` + `.models/config.json`).

## Environment

- AMD RX 7900 XTX (gfx1100), ROCm 6.2, Windows native.
- Model: qwen3-moe-tiny — hidden 1024, 24 layers (layer 0 dense MLP, layers
  1–23 MoE), 16 experts / top-4, `moe_intermediate_size` 256, QK-norm on,
  vocab 151936, fp32.

## Methodology

1. **Warm-up**: a throwaway engine runs a short turn so hiprtc compilation and
   weight upload are excluded from the measurement (the in-process kernel cache
   keeps compiled modules alive across model instances).
2. **Turn 1** (both arms): `turn1_prompt` (`MACH_PREFIX_TOKENS`, default 128)
   is prefilled and a greedy deterministic response of `MACH_RESP_TOKENS`
   (default 32) is generated. In the reuse arm the engine leaves a token-boundary
   anchor when the sequence finishes.
3. **Turn 2** (measured): `prompt2 = turn1_prompt + turn1_response +
   turn2_delta` (`MACH_DELTA_TOKENS`, default 16). The no-reuse arm re-prefills
   the whole `prompt2`; the reuse arm restores the anchor and prefills only the
   delta. TTFT = elapsed time from `add(prompt2)` to the step that returns the
   first generated token. Both arms generate `MACH_TURN2_GEN` (default 8)
   tokens greedily.
4. **Correctness gate**: the reuse arm's generated output must equal the
   baseline's (greedy + deterministic), printed as `output matches baseline`.
5. **Metric**: `TTFT reduction % = (no_reuse_ttft − reuse_ttft) / no_reuse_ttft`.

Env knobs: `MACH_MODELS`, `MACH_MODEL`, `MACH_CONFIG`, `MACH_PREFIX_TOKENS`,
`MACH_RESP_TOKENS`, `MACH_DELTA_TOKENS`, `MACH_TURN2_GEN`.

## How to run

```powershell
MACH_MODELS=.models MACH_MODEL=model.safetensors MACH_CONFIG=config.json `
  cargo run -p mach-model --release --features hip --example state_reuse_bench
```

## Results

Run: 2026-08-25, RX 7900 XTX / ROCm 6.2, qwen3-moe-tiny, fp32, release.
(The first launch after idle occasionally crashes inside `amdhip64_6.dll` —
the known cold-start driver flake on this box; retry once, see
`~/.agents/rules/LESSON_MachServe全量HIP回归偶发退出码1...`.)

### Default: turn-1 prompt 128 + response 32, turn-2 delta 16

| arm | TTFT (ms) | tokens reused | output matches baseline |
|-----|-----------|---------------|-------------------------|
| no-reuse (full prefill) | 1891.43 | 0 | - |
| reuse (anchor + delta prefill) | 187.24 | 160 | true |

- TTFT reduction: **90.10%** (no-reuse vs reuse).
- Turn-2 prefill: 176 tokens total; reuse skips 160 (turn-1 prompt + response),
  a theoretical upper bound of 90.9% of the prefill FLOPs — the measured
  reduction tracks the bound within ~1%.

### Longer context: turn-1 prompt 512 + response 128, turn-2 delta 32

| arm | TTFT (ms) | tokens reused | output matches baseline |
|-----|-----------|---------------|-------------------------|
| no-reuse (full prefill) | 7274.45 | 0 | - |
| reuse (anchor + delta prefill) | 358.53 | 640 | true |

- TTFT reduction: **95.07%** (theoretical bound 95.2%).

### Analysis vs FreeToken 65–80%

On this tiny model the measured reduction (90–95%) actually **exceeds** the
FreeToken 65–80% band, because the shared prefix is the dominant cost of
turn-2 (90–95% of the prefill tokens are skipped) and the anchor restore is a
cheap host-side KV copy relative to the skipped prefill. FreeToken's band is
reported for frontier-MoE contexts where other terms (expert-cache reload,
memory rebalancing, bandwidth-adaptive placement) reduce the *net* gain; on a
20–70B MoE with 4–8k-token histories the same 90%+ shape is expected as long
as the shared prefix dominates the delta. The reduction grows with context
length (90.1% @ 176 tokens → 95.1% @ 672 tokens), approaching the theoretical
bound `1 − delta/context`.

## Regression notes

- Anchor save/restore correctness is pinned by `tests/state_reuse.rs`
  (CPU exact-equal pair + GPU `#[ignore]` parity test); run with
  `cargo test -p mach-model --features hip --test state_reuse -- --ignored --test-threads=1`.
- The elastic-memory pressure simulation (`region_shrink_to_under_pressure_no_oom`,
  CPU + HIP `#[ignore]`) verifies the pool never OOMs when VRAM is squeezed.

