# Q4 存储 vs f16：真实模型基准（host RAM / TTFT / TPOT / logits）

> 目的：在真实 MoE checkpoint 上做 **f16 vs 存储-Q4** 的可复现 A/B —— Q4 只改变
> 主机侧存储精度（packed int4，上传时逐张量反量化到 f16），设备计算路径与 f16 完全
> 相同，所以预期差异是：host RAM 大幅下降 + logits 出现 Q4 量化误差，TTFT/TPOT 不变。

## 环境

- GPU / OS：AMD RX 7900 XTX（gfx1100，无头计算卡），Windows 原生 ROCm 6.2，HIP 后端，`--release`。
- 模型：`PrimeIntellect/qwen3-moe-tiny`（qwen3_moe，d_model=1024，24 层，16 专家 / top-4，
  moe_inter=256，~670M 参数；单文件 `model.safetensors` 1.25 GiB + `config.json`，放 `.models/`）。

## 运行

```bash
# 完整 A/B（f16 + Q4 两个 leg + logits 对比）
cargo run -p mach-model --release --features hip --example q4_bench

# 单 leg（干净的 host RAM 口径；两次运行分开测）
MACH_Q4=0 cargo run -p mach-model --release --features hip --example q4_bench
MACH_Q4=1 cargo run -p mach-model --release --features hip --example q4_bench
```

Env：`MACH_MODELS`（默认 `.models`）、`MACH_MODEL`（默认 `model.safetensors`）、
`MACH_CONFIG`（默认 `config.json`）、`MACH_Q4`（不设=双 leg；`0`=f16；`1`=Q4）、
`MACH_BATCH`（默认 8）、`MACH_BENCH_TOKENS`（默认 32）、`MACH_PROMPT_LEN`（默认 128）、
`MACH_CAPACITY`（默认 4）。

## 指标口径

- **host RAM**：`GetProcessMemoryInfo`（Windows）当前进程 working set 相对 HIP 初始化后的增量，
  即「加载权重后多占的内存」。单 leg 运行是干净口径（同进程双 leg 时，第二个 leg 会带上
  第一个 leg 释放后留在工作集里的内存，故 host RAM 以单 leg 数字为准）。
- **TTFT**：128-token prompt、capacity=4 的 continuous engine，从 add 到产出第一个生成
  token 的端到端耗时（含分块 prefill）。
- **TPOT**：batch=8 的 batched `decode_step` 平均值（固定输入流，`MACH_BENCH_TOKENS` 步）。
- **logits diff**：两个 leg 用**同一条固定输入流**跑 `MACH_BENCH_TOKENS` 步后，最后一步
  logits 的 max|f16 - q4| 与贪心 argmax 是否一致。

## 结果（2026-08-25，7900 XTX，--release）

### host RAM（加载后，单 leg 干净口径）

| 存储 | host RAM delta | 相对 f32 |
|---|---|---|
| f32（f16 计算腿） | **2.50 GiB**（2682 MiB） | 1.0x |
| Q4（packed int4） | **0.40 GiB**（433 MiB） | **6.2x 更小** |

- f32 加载 ~1.2-1.4 s；Q4 加载+量化 ~37 s（一次性成本；MoE 专家张量的
  `concat_q4` 是逐专家 dequant+requant，O(n²)，后续可优化）。
- 同进程 A/B 里 Q4 leg 显示 0.68 GiB，是 f32 权重释放后留在工作集里的内存，非 Q4 真实占用。

### TTFT / TPOT（同一 f16 计算路径）

| leg | TTFT(128 tok, cap 4) | TPOT(batch 8) | seq tok/s |
|---|---|---|---|
| f16 | 636–726 ms | 22.7–23.0 ms/step | 348–352 |
| Q4 | 557–664 ms | 22.7–22.7 ms/step | 352–353 |

- TTFT 区间跨多次运行（首跑含冷启动/顺序效应）；Q4 与 f16 在同一区间，**无系统性差异**。
- TPOT 逐 run 差 <2%，在噪声内 —— 符合「Q4 只改存储，设备 GEMM 仍是同一套 f16」的预期。

### GPU logits 差（Q4 vs f16，同一输入流）

```
max|f16 - q4| = 1.441（logits scale 7.445）| greedy argmax match: true
```

- 单元素 Q4 量化误差 ≈ 组内 scale/2 ≈ 1.5e-2（qwen3-moe-tiny 权重 ~1/sqrt(1024)）；
  24 层 MoE 逐层累积后最终 logits 差 ~1.4，但**贪心 argmax 一致**（真实模型采样行为不变）。
- 上传路径正确性由 `q4_batched_matches_dequantized_f16` 保证（Q4→f16 与参考 f32→f16
  逐位一致），因此上面的差纯粹是量化误差，不是计算路径分歧。

## FP8 现状（一句话）

gfx1100 / ROCm 6.2 下 hipBLAS 原生拒绝 fp8 GEMM（`gemm_ex_fp8_probe` 探针，见
`docs/roadmap.md` P3aq），原生 fp8 路径需自研 hiprtc 内核；**Q4 存储 → f16 计算是 AMD
上现实可行的大模型存储路径**（host RAM ~6x 下降，复用已验证的 f16 GEMM 路径）。

## 复现

```bash
MACH_MODELS=.models MACH_MODEL=model.safetensors MACH_CONFIG=config.json \
  cargo run -p mach-model --release --features hip --example q4_bench
```

服务端等价入口：`MACH_Q4=1 cargo run -p mach-server --release --features hip`
（`MACH_Q4` 走 `load_safetensors_q4` → Q4 批量引擎；preflight 会标注 storage Q4）。
