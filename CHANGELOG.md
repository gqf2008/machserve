# Changelog

## 0.2.0 (2026-08-25)

FreeToken 思想移植完成 + 个人产品化首批：

### MoE offload（P1）
- LRU 专家缓存 + host RAM offload + CPU/GPU 双路径 + 批量 offload + 服务端全链路（PR #4）
- Qwen3-MoE 族加载：moe_intermediate_size + 混合 dense/MoE 层 + 共享 qk-norm（PR #4）
- 真实模型基准：qwen3-moe-tiny 放置无关性 + TTFT/TPOT（docs/benchmark-moe-offload.md）

### 双缓冲 prefill（P2）
- 算第 l 层时预取第 l+1 层专家权重；独立预取 stream + ping-pong 设备专家池（PR #17）
- buffered TTFT ≈ 全驻留（+2~4%），比非缓冲 CPU-offload 快 ~90x

### 弹性显存 + Agent 状态复用（P3）
- mach-engine TaggedPool：带 tag / 可 resize 的显存区域，运行时收缩 / expert⇄KV 让渡（PR #18）
- state_reuse：token 边界锚点 + 增量 prefill；多轮 TTFT -90~95%（超过 FreeToken 65-80% 带）

### 稳定性
- GPU 采样测试 #[ignore] 化（AMD Windows ROCm 并发 GPU setup 死锁，PR #15）
- batched MoE 按层 dense/MoE 分发 + expert_size()（P1 遗留修复，PR #17/#18）

### 个人产品化（P4）
- mach-server doctor（一键排障）/ --version
- 一键安装 scripts/install.ps1（自检 → 构建 → 模型下载回退 → 冒烟）
- 支持矩阵 docs/support-matrix.md（实测 + 带宽估算方法学）
- CI 门禁 .github/workflows/ci.yml（fmt / clippy -D warnings / check / CPU 测试 + GPU opt-in）

## 0.1.0 (2026-08-22)

- 首个可用版本：Qwen2.5-0.5B/1.5B 推理（fp16）、连续批处理、OpenAI 兼容 API、分块 prefill、spec-decode（实验，已暂停）、MLA（实验）
