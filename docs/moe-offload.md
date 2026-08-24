# MachServe MoE Offload 引擎

> 面向个人/单卡场景的 MoE 推理 offload：专家驻留 host RAM、按需取到 GPU、带宽自适应分流、批量 serving 全链路支持。

## 能力

- **单序列 offload（`GpuModel`）**：有界 GPU 专家槽位 + LRU 按需取 + CPU 兜底 + 带宽自适应 q* + 实时 q*（自动重测）。
- **批量 offload（`BatchedModel`，cpu backend）**：专家驻留 host RAM，MoE 在 CPU 算（显存受控），数学有 CPU 单测。
- **服务端全链路**：`ContinuousModel::with_prefill_rows_offload` + `ServerEngine::with_offload` + CLI `MACH_MOE_SLOTS`。

## 模块

- `mach-model/src/moe_backend.rs`：`LruExpertCache`（`plan`/`plan_step`/`plan_lru` + 稀疏槽位修复）。
- `mach-model/src/moe_offload.rs`：`MoeOffload::step`（host 编排）、`moe_batch_cpu_residual`（批量 CPU 残差，有单测）。
- `mach-model/src/adaptive.rs`：`BandwidthProbe`（实测 PCIe/CPU）、`BandwidthProfile::choose`（q*）、`AdaptiveProfile`（实时 q* 平滑）。
- `GpuModel`：`with_expert_slots` / `with_adaptive` / `reprobe_bandwidth` / `set_reprobe_every` / `moe_slot_place`。
- `BatchedModel::with_expert_slots`（cpu backend）→ `ContinuousModel::with_prefill_rows_offload` → `ServerEngine::with_offload` → CLI `MACH_MOE_SLOTS`。

## 用法

```bash
# 用 offload serve 一个 MoE 模型（cpu backend：专家在 host RAM，MoE 走 CPU）
MACH_MODEL=<moE.safetensors> MACH_CONFIG=<config.json> MACH_MOE_SLOTS=2 \\
  cargo run -p mach-server --release --features hip

# 基准（TTFT / TPOT，三种放置对比）
MACH_MODEL=<moE.safetensors> MACH_CONFIG=<config.json> MACH_MOE_SLOTS=2 \\
  cargo run -p mach-model --release --features hip --example moe_offload_bench

# GPU 对拍测试（默认 #[ignore]，需显式 --ignored；在稳定 GPU 上跑）
cargo test -p mach-model --features hip --test moe -- --ignored --test-threads=1
```

## 验证状态

- **CPU 已验证**（`cargo test -p mach-model --lib`，20 passed）：LRU 语义、放置无关性、批量 CPU 残差、q* 决策、实时 q* 节拍、tokenizer/fp16。
- **GPU 已实测过**（之前机器稳定时）：`moe_gpu_offload_placement_invariant` / `moe_gpu_slot_offload_matches_full` / `moe_gpu_adaptive_offload_matches_full` —— 三种 offload 与全驻留一致。
- **待稳定 GPU 验证**：批量 offload 对拍、批量 GPU 槽位快速路径（当前 cpu backend 是正确性第一版）、基准实跑（TTFT/TPOT）。

## 环境注意

- 当前 7900 XTX + Windows ROCm 6.2 在**持续 GPU 负载下会触发驱动 TDR（Event 4101），可能整机硬锁**；GPU 测试已默认 `#[ignore]`，验证请换稳定机器/远程，并遵守 `-- --ignored` + `--test-threads=1` + 时间盒。

