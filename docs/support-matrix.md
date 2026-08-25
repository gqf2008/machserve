# 支持矩阵（说人话版）

> 目标：让个人用户先选「我是什么卡 + 想跑哪个模型」，再决定怎么装。所有数字要么是**本机实测**，要么是**标注清楚的估算**，不画饼。

## 一、本机实测（AMD RX 7900 XTX / ROCm 6.2，唯一实测基准机）

| 场景 | 配置 | 结果 |
|---|---|---|
| 小模型 decode（Qwen2.5-0.5B fp16） | capacity 64，批 512 | **35251 tok/s** |
| 小模型 decode（同上） | 批 64，短 ctx | 12887 tok/s（4.97 ms/step） |
| 长 context decode（2048 ctx） | Qwen2.5-0.5B fp16 | 13.4 ms/step（单序列 4778 tok/s） |
| 长 prompt TTFT | 512-token / 2048-token | 57 ms / 289 ms（分块 prefill） |
| **MoE 全驻留（qwen3-moe-tiny，~0.67B）** | 16 专家 / top-4，f32 单序列 | **125 tok/s**（~8 ms/tok） |
| MoE offload（同上） | slots=4 / slots=8 | 31.6 / 40.5 tok/s |
| MoE offload 自适应 q* | 同上 | ~18 tok/s |
| 双缓冲 prefill（长 ctx） | qwen3-moe-tiny，ctx 2048 | buffered TTFT ≈ 全驻留（+2~4%），比非缓冲 CPU-offload 快 ~90x |
| Agent 多轮复用 | 同上下文 + 新轮次 | 多轮 TTFT **-90~95%**（前缀 128/512） |

## 二、MoE offload 的带宽账（估算方法学，最诚实的部分）

专家权重驻留 host RAM、按需过 PCIe 时，**decode 吞吐的上限由 PCIe 带宽决定**：

```
每 token 专家字节 ≈ topk × expert_size × d_model × 3 × 4 B（f32，bf16 减半）
理论 tok/s ≈ PCIe 带宽 / 每 token 专家字节
```

| 模型规模（举例） | topk / expert 宽 / d | 每 token 专家字节（bf16） | PCIe 4.0 x16（~28 GB/s） | PCIe 5.0 x16（~64 GB/s） |
|---|---|---|---|---|
| qwen3-moe-tiny（0.67B） | 4 / 256 / 1024 | ~6.3 MB | 实测瓶颈在同步开销，不在带宽 | 同上 |
| 30B 级 A3B（如 Qwen3-30B-A3B） | 8 / 1024 / 4096 | ~134 MB | **~200 tok/s 带宽上限** | ~470 tok/s |
| 70B 级（如 Qwen2.5-MoE-A2.7B 类） | 8 / 1408 / 2048 | ~46 MB | ~600 tok/s 带宽上限 | ~1400 tok/s |

> 注：这是**纯 offload（专家全在 host）**的上限；专家常驻 GPU 时不受此限。实测的小模型数字远低于带宽上限，因为当前实现的主成本是每步同步/D2H 往返（P2 基准结论），不是带宽——大模型反而更接近带宽上限。

## 三、说人话：什么卡能跑什么

| 你的情况 | 建议 |
|---|---|
| 8 GB 显存卡 | 专家全驻留的 ≤2B 级 MoE（如 qwen3-moe-tiny）轻松跑；30B 级 A3B 走 offload（专家在内存），吞吐看内存带宽 + PCIe |
| 16 GB 显存卡 | ≤8B 级密集模型全驻留；30B 级 A3B offload 可跑，~100-200 tok/s 量级 |
| 24 GB 显存卡（如 7900 XTX / 3090） | 本机实测场景全覆盖；30B 级 A3B offload + 长 context / 多轮复用体验最好 |
| 内存 < 32 GB | 别碰 30B+ offload（专家 + KV 都在内存，会吃紧）；先跑 ≤7B 级 |
| 没有 AMD 卡 | 目前只支持 AMD ROCm/HIP（Windows 原生）；NVIDIA 路径未就绪 |

## 四、判断准绳

- **先 `mach-server doctor`**：一次看清显卡 / VRAM / ROCm / 模型文件 / 估算占用。
- **再 `install.ps1`**：自检 + 构建 + 下模型 + 冒烟，全程说人话。
- **别信跑分、信对拍**：仓库所有性能改动都带「GPU==CPU 对拍 + 可复现 A/B」，README「当前战绩」与 `docs/benchmark-*` 均有实测口径。
