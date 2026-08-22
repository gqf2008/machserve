import io
p = "E:/Users/gxh/Documents/GitHub/machserve/docs/benchmark-protocol.md"
s = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")

old = """## 结果(2026-08-22,7900 XTX)

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
   后端差异。更严格的对比需把 MachServe 换成 fp16 权重、或给 llama.cpp 用 fp16 GGUF。"""

new = """## 结果(2026-08-22,7900 XTX;2026-08-22 修正)

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
4. **口径差异**:MachServe fp32 vs llama.cpp Q8_0;Vulkan vs HIP。"""

assert old in s, "doc anchor"
s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("benchmark doc corrected")
