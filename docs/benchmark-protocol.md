# GPU 对标: MachServe vs llama.cpp(7900 XTX, Windows 原生)

> 2026-08-22 更新:按项目定位,**对标物不再是 TokenSpeed**(MachServe 是对 TokenSpeed 的
> 重写,且 TokenSpeed 无法在 Windows 原生运行);改为与本机可运行的 **llama.cpp(Vulkan 后端)**
> 做同 GPU、同模型的真实对比。llama.cpp 构建于 Windows 原生(ggml 2100e59,Vulkan SDK 1.4.357)。

## 基准环境

- GPU:AMD Radeon RX 7900 XTX(24GB, gfx1100),Windows 原生
- 模型:Qwen2.5-0.5B-Instruct(24 层 / hidden 896 / GQA 14:2 / intermediate 4864)
- MachServe:fp32 权重(原始 safetensors → f32),HIP 后端,`cargo run -p mach-model --release --features hip --example qwen_bench`
- llama.cpp:Q8_0 GGUF(644MB, qwen2.5-0.5b-instruct-q8_0.gguf),Vulkan 后端
  `llama-bench -m ... -p 512 -n 128 -b 1,16,128 -r 2`

## 结果(2026-08-22,7900 XTX;2026-08-22 修正)

| 指标 | MachServe | llama.cpp (Vulkan) |
|---|---|---|
| **decode TPOT(真实 GPU 完成率)** | **~7.1 ms/tok**(50 步 + 1 次同步实测) | **1.55 ms/tok**(tg=643 tok/s) |
| decode TPOT(GPU 采样生成,4B 回读) | ~7.6-10.7 ms/tok | 1.55 ms/tok |
| CPU 提交速率(无同步,易误读) | eager 1.39 / graph 0.57 ms/tok | — |
| prompt 处理(pp512, batch=128) | 未实现(逐 token) | 19782 tok/s |
| 数值正确性 | 相对误差 4.4e-6(vs fp64 参考) | (GGUF 量化,无独立对拍) |

## 结论与口径(修正)

1. **真实 TPOT 约 7.1 ms/token**,比 llama.cpp Vulkan(1.55 ms)慢约 4.6x。
   此前"launch-only 0.30-0.47 ms/tok、2-5x 快"的结论是**误读**——那是 CPU 提交速率
   (kernel 排队未执行),不是 GPU 完成时间;加入一次同步后测得真实 GPU 完成率 ~7.1 ms;
2. **真正瓶颈是 GPU kernel 效率,不是 logits 读回**:GPU 采样(只回读 4 字节)仍 ~7.6-10.7 ms,
   与全量回读相当。根因是 **m=1 的 hipBLAS GEMM 极低效**(decode 每层 7 个 m=1 GEMM × 24 层),
   以及 fp32 计算;llama.cpp 靠批量/融合 kernel + flash-attn + fp16 拿到 1.55 ms;
3. **P2 的正确方向**:批量 decode(m=B 使 GEMM 高效,同时支撑连续批处理)、fp16/bf16 计算、
   更好的 decode GEMV——而不是采样/读回(那个已被证伪)。GPU 采样已实现,用于后续批量采样;
4. **口径差异**:MachServe fp32 vs llama.cpp Q8_0;Vulkan vs HIP。

## 批量 decode 结果(2026-08-22,P2b)

连续批处理(B 序列共享一次前向,批量 GEMM m=B)后,每序列 TPOT 按 1/B 下降:

| batch | ms/step | ms/seq-tok | tok/s(seq) |
|---|---|---|---|
| 1 | 11.89 | 11.886 | 84 |
| 8 | 12.78 | 1.597 | 626 |
| 16 | 13.42 | 0.839 | 1193 |
| 32 | 14.93 | 0.467 | 2143 |
| **64** | 13.79 | **0.216** | **4640** |

对比 llama.cpp Vulkan:1.55 ms/seq-tok(643 tok/s)→ **batch=64 时 MachServe 快 7.2x**。
结论:批量(m=B)GEMM 直接修复 m=1 瓶颈;连续批处理既是性能方案也是引擎模型。

## 复现

```bash
# MachServe
cargo run -p mach-model --release --features hip --example qwen_bench
# llama.cpp
llama-bench -m models/qwen2.5-0.5b-instruct-q8_0.gguf -p 512 -n 128 -b 1,16,128 -r 2
```
