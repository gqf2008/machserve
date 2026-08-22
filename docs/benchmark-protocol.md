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

## 结果(2026-08-22,7900 XTX)

| 指标 | MachServe | llama.cpp (Vulkan) |
|---|---|---|
| **decode TPOT(GPU 计算)** | **0.30-0.47 ms/tok**(HIP graph 重放) | 1.55 ms/tok(tg=643 tok/s) |
| **decode 吞吐(按 TPOT 反推)** | ~2100-3300 tok/s | 643 tok/s |
| decode TPOT(完整步,含 logits 读回) | 7-10 ms/tok | 1.55 ms/tok(采样在 GPU 侧完成) |
| prompt 处理(pp512, batch=128) | 未实现(逐 token) | 19782 tok/s |
| 数值正确性 | 相对误差 4.4e-6(vs fp64 参考) | (GGUF 量化,无独立对拍) |

## 结论与口径

1. **GPU 解码计算(纯 kernel + 调度)**:**MachServe 比 llama.cpp Vulkan 快约 2-5x**。
   这是 MachServe 的 HIP kernel + graph 重放的真实收益;
2. **端到端 TPOT 落后**:MachServe 完整步 7-10 ms/token 远慢于 llama.cpp 的 1.55 ms,
   瓶颈是**每 token 把全部 151936 个 logits(607KB)读回主机**;llama.cpp 在 GPU 侧采样,
   不读全量 logits。这是 MachServe 明确的下一步优化(GPU 侧 sampling kernel / 只读采样
   token);
3. **口径差异**:MachServe 为 fp32 权重,llama.cpp 为 Q8_0 量化(通常更快);Vulkan vs HIP
   后端差异。更严格的对比需把 MachServe 换成 fp16 权重、或给 llama.cpp 用 fp16 GGUF。

## 复现

```bash
# MachServe
cargo run -p mach-model --release --features hip --example qwen_bench
# llama.cpp
llama-bench -m models/qwen2.5-0.5b-instruct-q8_0.gguf -p 512 -n 128 -b 1,16,128 -r 2
```
