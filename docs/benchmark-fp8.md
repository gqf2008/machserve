# FP8 存储 vs f16：真实模型基准（host RAM / 加载 / TTFT / TPOT / logits）

> 目的：在真实 MoE checkpoint 上做 **f16 vs 存储-FP8（E4M3 + per-tensor scale）** 的
> 可复现 A/B —— FP8 只改变主机侧存储精度（1 字节/元素 + 每张量 1 个 f32 scale，上传时
> 逐张量反量化到 f16），设备计算路径与 f16 完全相同，所以预期差异是：host RAM 大幅下降
> + logits 出现 E4M3 量化误差（介于 f16 与 Q4 之间），TTFT/TPOT 不变。

## 环境

- GPU / OS：AMD RX 7900 XTX（gfx1100，无头计算卡），Windows 原生 ROCm 6.2，HIP 后端，`--release`。
- 模型：`PrimeIntellect/qwen3-moe-tiny`（qwen3_moe，d_model=1024，24 层，16 专家 / top-4，
  moe_inter=256，~670M 参数；单文件 `model.safetensors` 1.25 GiB + `config.json`，放 `.models/`）。

## 运行

```bash
# 完整 A/B（f16 + FP8 两个 leg + logits 对比）
cargo run -p mach-model --release --features hip --example fp8_bench

# 单 leg（干净的 host RAM 口径；两次运行分开测）
MACH_FP8=0 cargo run -p mach-model --release --features hip --example fp8_bench
MACH_FP8=1 cargo run -p mach-model --release --features hip --example fp8_bench
```

Env：`MACH_MODELS`（默认 `.models`）、`MACH_MODEL`（默认 `model.safetensors`）、
`MACH_CONFIG`（默认 `config.json`）、`MACH_FP8`（不设=双 leg；`0`=f16；`1`=FP8）、
`MACH_BATCH`（默认 8）、`MACH_BENCH_TOKENS`（默认 32）、`MACH_PROMPT_LEN`（默认 128）、
`MACH_CAPACITY`（默认 4）。

## 指标口径

- **host RAM**：`GetProcessMemoryInfo`（Windows）当前进程 working set 相对 HIP 初始化后的增量，
  即「加载权重后多占的内存」。单 leg 运行是干净口径（同进程双 leg 时，第二个 leg 会带上
  第一个 leg 释放后留在工作集里的内存，故 host RAM 以单 leg 数字为准）。
- **加载时间**：从读取 safetensors 到权重全部量化/就绪（f16 leg = f32 解码；FP8 leg = f32
  解码 + E4M3 量化 + 并行转换）。
- **TTFT**：128-token prompt、capacity=4 的 continuous engine，从 add 到产出第一个生成
  token 的端到端耗时（含分块 prefill）。
- **TPOT**：batch=8 的 batched `decode_step` 平均值（固定输入流，`MACH_BENCH_TOKENS` 步）。
- **logits diff**：两个 leg 用**同一条固定输入流**跑 `MACH_BENCH_TOKENS` 步后，最后一步
  logits 的 max|f16 - FP8| 与贪心 argmax 是否一致。

## 结果（2026-08-25，7900 XTX，--release）

### host RAM（加载后，单 leg 干净口径）

| 存储 | host RAM delta | 相对 f32 加载 |
|---|---|---|
| f32（f16 计算腿） | **2.50 GiB**（2682 MiB） | 1.0x |
| FP8（E4M3 + per-tensor scale） | **0.63 GiB**（677 MiB） | **4.0x 更小** |

- 预期：FP8 1 字节/元素 ≈ f16 的一半 ≈ f32 的 1/4（GEMM 权重）；qwen3-moe-tiny
  ~670M 参数 → FP8 ~0.63 GiB 与预期一致。
- f32 加载 ~1.3-1.5 s；FP8 加载+量化 **~2.3-3.4 s**（一次性成本；量化按张量并行 +
  大张量按元素范围跨线程 `Fp8Tensor::quantize_par`，与 `quantize` 逐位一致）。
- 同进程 A/B 里 FP8 leg 显示 0.77 GiB，是 f32 权重释放后留在工作集里的内存，非 FP8 真实占用。

### TTFT / TPOT（同一 f16 计算路径）

| leg | TTFT(128 tok, cap 4) | TPOT(batch 8) | seq tok/s |
|---|---|---|---|
| f16 | 682–684 ms | 23.3–23.6 ms/step | 339–344 |
| FP8 | 645–744 ms | 23.3–24.4 ms/step | 328–343 |

- TTFT 区间跨多次运行（首跑含冷启动/顺序效应）；FP8 与 f16 在同一区间，**无系统性差异**。
- TPOT 逐 run 差 <5%，在噪声内 —— 符合「FP8 只改存储，设备 GEMM 仍是同一套 f16」的预期。

### GPU logits 差（FP8 vs f16，同一输入流）

```
max|f16 - FP8| = 0.533（logits scale 7.445）| greedy argmax match: true
```

- 单元素 E4M3 相对误差 ~6.25%（3 位尾数），远小于 Q4 的 ~12.5%；24 层 MoE 逐层累积后
  最终 logits 差 0.53，**远小于 Q4 的 1.44**（同一输入流上 Q4 为 1.44、FP8 为 0.53），
  且**贪心 argmax 一致**（真实模型采样行为不变）。
- 上传路径正确性由 `fp8_batched_matches_dequantized_f16` 保证（FP8→f16 与参考 f32→f16
  逐位一致），因此上面的差纯粹是量化误差，不是计算路径分歧。
- 选型依据（CPU 对拍，40 个合成 prompt，同一口径）：纯 E4M3（无 scale）max logits 差
  0.91、per-tensor scale 0.77、group-32 0.67、Q4 2.49。per-tensor scale 在 0 存储开销下
  再降 ~15%，并消除 E4M3 次正规数下限（<2^-9 的元素会冲刷为 0）对小权重张量的影响；
  group-32 只再降 ~13% 但多 12.5% 存储，FP8 的 3 位尾数不需要。

## FP8 现状（一句话）

gfx1100 / ROCm 6.2 下 hipBLAS 原生拒绝 fp8 GEMM（`gemm_ex_fp8_probe` 探针，见
`docs/roadmap.md` P3aq）；**FP8 存储 → f16 计算是 AMD 上现实可行的中间精度路径**
（host RAM ~4x 小于 f32，logits 差 0.53 介于 f16(~0) 与 Q4(1.44) 之间），
原生 fp8 GEMM 另需 NVIDIA 或自研 hiprtc 内核。

## 复现

```bash
MACH_MODELS=.models MACH_MODEL=model.safetensors MACH_CONFIG=config.json \
  cargo run -p mach-model --release --features hip --example fp8_bench
```

服务端等价入口：`MACH_FP8=1 cargo run -p mach-server --release --features hip`
（`MACH_FP8` 走 `load_safetensors_fp8` → FP8 批量引擎；preflight 会标注 storage FP8；
与 `MACH_Q4` / `MACH_SPEC` / `MACH_MOE_SLOTS` 互斥，要求 `MACH_DTYPE=f16`）。
