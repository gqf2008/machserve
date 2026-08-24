# 可复现基准：全层双缓冲 prefill（TTFT / TPOT A/B）

> 目的：用一种可复现、可对比的方式，测出 MachServe 全层双缓冲 prefill（`BatchedModel::with_prefill_buffer`）
> 相对「全驻留（无逐层 H2D）」与「P1 CPU-offload（非缓冲）」的 TTFT / TPOT，验证 FreeToken 论文
> 「带宽自适应执行 / 长 context prefill 提速」在 MachServe 上的落地效果。

## 背景

- P2 新增 `crates/mach-model/src/prefill_buffered.rs`：算第 l 层时在**独立 stream** 上预取第 l+1 层
  MoE 专家权重（host→device，pinned 内存 + ping-pong 双缓冲），预取与计算拆 stream，避免与 graph capture 冲突。
- 三种模式跑**同一个模型、同一 prompt**：
  - `full`：全部专家常驻 GPU，无逐层 H2D（顺序执行的「上限」）；
  - `buffered`：专家按层 H2D 预取 + 双缓冲（新增模式，GPU 只保留 ~2 层专家 + pinned host staging）；
  - `cpu`：P1 offload 基线，每层 D2H + CPU 计算 MoE（非缓冲）。

## 环境

- GPU / OS：AMD RX 7900 XTX（24GB, gfx1100），Windows 原生 ROCm 6.2（与 `docs/benchmark-protocol.md` 一致）。
- 模型：`PrimeIntellect/qwen3-moe-tiny`（MoE：16 专家 / top-4 / 专家宽 256 / 层 0 dense / 24 层，d_model=1024）。
  权重在 `.models/model.safetensors`（fp32），`config.json` 在 `.models/`。
- 编译：`--release --features hip`。

## 运行

```bash
cargo run -p mach-model --release --features hip --example prefill_bench
```

环境变量（均有默认值）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `MACH_MODEL` | `model.safetensors` | checkpoint 文件名 |
| `MACH_CONFIG` | `config.json` | 模型 config 文件名 |
| `MACH_MODELS` | `.models` | 模型目录 |
| `MACH_CTX` | `2048` | prompt 长度（长 context prefill） |
| `MACH_PREFILL_ROWS` | `512` | 每步 prefill 行数（分块大小） |
| `MACH_DECODE` | `16` | TPOT 的 decode token 数 |
| `MACH_BENCH_MODE` | `all` | `all` / `full` / `buffered` / `cpu`（跑单个模式） |

## 指标口径

- **TTFT**：完整 prompt 的分块 prefill 总耗时（`decode_step_explicit` 每步内部同步，与 continuous 引擎一致）。
- **TPOT**：prefill 后逐 token decode 的平均耗时。
- **greedy tokens match full**：buffered/cpu 的贪心输出 token 序列是否与 full 完全一致（数值对拍的可观测锚点）。

## 结果（2026-08-25，7900 XTX，qwen3-moe-tiny，fp32）

### ctx=2048，rows=512，decode=16

| mode | TTFT(ms) | TPOT(ms/tok) | tok/s | greedy match full |
|---|---|---|---|---|
| full | 543.2 | 23.2 | 43.1 | - |
| **buffered** | **563.8** | 59.7 | 16.8 | true |
| cpu-offload | 51098.1 | 62.2 | 16.1 | true |

### ctx=4096，rows=512，decode=8

| mode | TTFT(ms) | TPOT(ms/tok) | tok/s |
|---|---|---|---|
| full | 1961.8 | 26.8 | 37.3 |
| **buffered** | **2004.7** | 60.2 | 16.6 |

（ctx=4096 未跑 cpu-offload：非缓冲 CPU 基线单次约 100s，见下。）

## 解读

1. **TTFT：buffered ≈ full（+2% ~ +4%）**。逐层 H2D（~50MB/层 ≈ 2ms@PCIe4 x16）被双缓冲几乎完全隐藏在
   层计算（~20ms/层）之下；相对非缓冲的 `cpu` 基线（每层 D2H + CPU 计算 + H2D + 同步），buffered 快 **~90x**。
   这就是 FreeToken「长 context prefill 提速」在 MachServe 上的可复现结果：**专家不必常驻 GPU，
   流式预取也几乎不损失 prefill 速度**。
2. **数值对拍**：`buffered` 与 `full` 的贪心输出逐 token 一致（CPU 单测逐位一致 + GPU 对拍 bitwise，
   见 `tests/prefill_buffered.rs`）；`cpu` 因求和顺序不同有极小浮点差，但 argmax 一致。
3. **TPOT 的诚实口径**：buffered 的 decode 仍按层预取**全部**专家（~1.15GB/token H2D），所以 TPOT 与 `cpu`
   相当、明显慢于 `full`。decode 的正确优化是 P1 的 LRU 专家缓存 / q* 分流（`moe_offload_bench`），
   不是全层预取 —— 本基准的 TPOT 列展示的是「全层双缓冲用于 prefill」的真实边界。

## 稳定性备注

- 文档数字均在 `MACH_PREFILL_ROWS=512`（continuous 引擎默认分块）下取得，多次复现稳定。
- 在 `MACH_PREFILL_ROWS >= 1024` + 完整尺寸 checkpoint 时，本机 Windows ROCm 驱动**间歇性**失败
  （hipMalloc 报 OOM 但显存空闲、偶发 router 回读损坏）；同规模合成模型 5/5 稳定，未见逻辑错误，
  疑似驱动在大量分配/hiprtc 编译下的不稳定。跑长 context 时请保持 `MACH_PREFILL_ROWS=512`。

## 复现

```bash
# 主表（约 1 分钟：cpu 基线 ~51s）
cargo run -p mach-model --release --features hip --example prefill_bench

# 只跑 buffered/full（ctx=4096 秒级）
MACH_CTX=4096 MACH_PREFILL_ROWS=512 MACH_DECODE=8 MACH_BENCH_MODE=full \
  cargo run -p mach-model --release --features hip --example prefill_bench
MACH_CTX=4096 MACH_PREFILL_ROWS=512 MACH_DECODE=8 MACH_BENCH_MODE=buffered \
  cargo run -p mach-model --release --features hip --example prefill_bench
```
