# MoE Offload 基准结果（真机实测）

> 记录每次真机基准，含环境、模型、命令与结果，供复现与对比。

## 2026-08-24 · PrimeIntellect/qwen3-moe-tiny

- **GPU / 驱动**：AMD RX 7900 XTX（无头计算卡，显示器已改接核显），Windows 原生 ROCm 6.2，HIP 后端，`--release`。
- **模型**：`PrimeIntellect/qwen3-moe-tiny`（qwen3_moe，d_model=1024，24 层，16 专家，top-4，moe_inter=256，~670M 参数）。
  - 单文件 `model.safetensors`（BF16，1.34GB）+ `config.json`，放 `.models/`。
- **命令**：
  ```bash
  MACH_MODEL=model.safetensors MACH_CONFIG=config.json MACH_BENCH_TOKENS=32 \
    cargo run -p mach-model --release --features hip --example moe_offload_bench
  ```

### 结果（TTFT / TPOT / tok/s）

| MACH_MOE_SLOTS | full TTFT(ms) | full TPOT(ms) | full tok/s | slots TTFT(ms) | slots TPOT(ms) | slots tok/s | adaptive TPOT(ms) | adaptive tok/s |
|---|---|---|---|---|---|---|---|---|
| 2   | 7.97  | 7.99  | 125.1 | 556.96 | 245.24 | 4.1  | 55.23 | 18.1 |
| 4   | 8.15  | 8.06  | 124.1 | 55.56  | 31.69  | 31.6 | 56.64 | 17.7 |
| 8   | 8.25  | 8.01  | 124.8 | 60.11  | 24.68  | 40.5 | 54.50 | 18.3 |

### 解读

- **放置无关性在真实 checkpoint 上成立**：`slots`/`adaptive` 与 `full` 的最大 logit 差 ≤ 6e-6，argmax 完全一致 —— offload 只改变调度，不改数学。
- **offload 同步/回读开销是主要成本**：全驻留 ~8ms/token；slots=4 为 ~32ms/token（+24ms 调度/同步），slots=8 为 ~25ms/token。TPOT 随槽位数单调下降，说明开销来自每步 stream sync + D2H/H2D 往返，而非专家计算本身。
- **slots=2 < topk=4 时最差**（245ms/token）：每步都要换入 2 个新专家（LRU 颠簸）+ 2 个专家走 CPU，往返次数最多。
- **adaptive q\*** 稳定在 ~55ms/token：探针测得本模型 CPU 专家计算便宜（expert_size=256），q* 倾向 CPU 兜底，避免 GPU 同步；相对 slots=2 提升 4.4x，但不如容量足够的纯 GPU 槽位路径（slots≥4）。
- **结论 / 下一步**：同步与 D2H 是主要瓶颈 → 后续做 GPU 槽位快速路径（批量取专家、减少每步 host 往返），并对齐 FreeToken 的长 context prefill offload 口径。
