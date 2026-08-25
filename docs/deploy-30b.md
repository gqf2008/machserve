# 30B+ MoE 部署配方（Qwen3-30B-A3B 示例）

> 目标：把「大模型路径」落到真机上。本文给出可复现的部署步骤、资源账目与现状结论。核心代码路径（多 shard Q4 流式加载 → offload 服务）已被测试验证；**本机（网络/内存）未实际跑通 30B 实体的原因是下载量，不是引擎能力**。

## 1. 为什么选 Qwen3-30B-A3B

- 30B 总参 / 3B 激活，**无 shared_expert**（loader 直接支持），Qwen-MoE 命名，config：hidden 2048、48 层、128 专家、top-8、moe_inter 768。
- BF16 约 60GB（16 shards）；Q4 后 host 约 15GB，7900 XTX 24GB VRAM + offload 可跑。

## 2. 资源账（估算，方法学见 docs/support-matrix.md）

| 项 | 数值 |
|---|---|
| 权重（BF16 下载） | ~60 GB（16 × ~3.75 GB） |
| host RAM（Q4 加载后） | ~15 GB + KV ~3 GB + 激活 ≈ **< 24 GB**（32GB 机器可行） |
| GPU VRAM | 全驻留不可能；offload：路由 + 注意力 + 少量专家槽位，几 GB |
| decode 吞吐（offload） | 每 token 专家字节 ≈ 8 × 768 × 2048 × 3 × 2B ≈ 75 MB（bf16 口径）；PCIe 4.0 x16 ~28 GB/s → **~370 tok/s 带宽上限**（Q4/fp8 存储只降 host RAM，不降每 token 移动量；上限由专家字节 × PCIe 决定） |

## 3. 部署步骤（网络稳定的机器上执行）

```bash
# 1) 下载 16 shards（hf-mirror 回退；脚本见 scripts/install.ps1 的下载段）
mkdir -p .models/qwen3-30b-a3b
for i in $(seq -w 1 16); do
  curl -L --retry 5 --retry-all-errors -o .models/qwen3-30b-a3b/model-000$i-of-00016.safetensors \
    https://hf-mirror.com/Qwen/Qwen3-30B-A3B/resolve/main/model-000$i-of-00016.safetensors
done
curl -L -o .models/qwen3-30b-a3b/config.json \
  https://hf-mirror.com/Qwen/Qwen3-30B-A3B/resolve/main/config.json

# 2) Q4 流式加载（多 shard 已测：q4_sharded_load_matches_single_file）+ offload serve
MACH_MODELS=.models/qwen3-30b-a3b MACH_MODEL=model-00001-of-00016.safetensors \
MACH_CONFIG=config.json MACH_Q4=1 MACH_MOE_SLOTS=2 MACH_CAPACITY=8 MACH_PREFILL_ROWS=256 \
  cargo run -p mach-server --release --features hip

# 3) 冒烟
curl http://127.0.0.1:8080/v1/completions -d '{"model":"x","prompt":"你好","max_tokens":16}'
```

> 注意：MACH_MODEL 只需指向任一 shard 文件名（loader 按目录发现全部 shard）；Q4 要求 f16 计算路径（默认）。

## 4. 现状结论（2026-08-25）

- **引擎就绪**：多 shard Q4 流式加载（4-shard 测试与单文件逐位一致）、Q4/FP8 存储、MoE offload（LRU + q*）、双缓冲 prefill、Agent 状态复用、服务端全链路均已在 qwen3-moe-tiny 上验证。
- **未跑通的唯一原因**：~60GB 下载在本网络（1.34GB 需约半小时）不可行。网络稳定后按 §3 一键部署即可；届时把真实 TTFT/TPOT/降幅记录回本文。
- **预估**：decode 受 PCIe 带宽约束 ~370 tok/s 上限（见 §2），长 context prefill 用双缓冲、多轮用状态复用（-90%+）。
