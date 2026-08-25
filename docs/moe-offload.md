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
- **GPU 已实测**（本机 7900 XTX 无头卡，2026-08-24）：`moe_gpu_offload_placement_invariant` / `moe_gpu_slot_offload_matches_full` / `moe_gpu_adaptive_offload_matches_full` / `batched_moe_cpu_offload_matches_full_resident` / `moe_gpu_forward_matches_cpu_reference` —— 单序列、批量、自适应三种 offload 与全驻留一致。
- **真实 MoE checkpoint 已实跑**：`PrimeIntellect/qwen3-moe-tiny`（Qwen3-MoE，~670M）加载 + 三档基准（TTFT/TPOT），放置无关性成立（max logit diff ≤ 6e-6，argmax 一致）；结果见 `benchmark-results-moe-offload.md`。
- **fp64 三路对拍已绿（issue #22）**：独立 `fp64_ref` 参考（matvec/SwiGLU/router top-k/批量残差 + 完整 forward），GPU(f32) == CPU(f32) == fp64 误差在 f32 舍入量级（合成 ~1e-6，真实 qwen3-moe-tiny ~9.5e-6 绝对 / ~1.9e-6 相对），argmax 全一致，放置无关性在 fp64 下成立；数字见 `benchmark-fp64-parity.md`。
- **loader 新增 Qwen3-MoE 族支持**：`moe_intermediate_size`（专家 FFN 宽度）+ 混合 dense/MoE 层（`mlp_only_layers`，按 `mlp.gate.weight` 逐层判定）+ Qwen3 共享 qk-norm 权重（`[head_dim]`，loader 平铺为 per-head）。

## 待办（非阻塞）

- 批量 GPU 槽位快速路径（当前 cpu backend 是正确性第一版）：基准显示同步/D2H 是主要开销，是后续优化方向。

## 环境注意

- 本机 7900 XTX 现为**无头计算卡**（显示器已改接核显），GPU 计算可稳定运行；GPU 测试默认 `#[ignore]`，跑时用 `-- --ignored` + `--test-threads=1`。
- 历史：显示器直连 7900 时，持续 GPU 负载会触发驱动 TDR（Event 4101），可能整机硬锁只能硬复位。

