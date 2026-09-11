# Qwen3.8 Vision E2E（C4）Runbook

Stage C 的离线链路已经合入；本文档描述真机 GPU 窗口打开后如何完成 C4 硬门禁。
不要在本文档的步骤外并发跑编译或第二次推理：此前多次整机崩溃都发生在并行重载时。

## 前置

- Windows + ROCm，AMD 7900 XTX 24GB；
- 真实 checkpoint `.models/qwen3.8-27b`（18 shard，含 `model.visual.*`）；
- release 二进制：`cargo build -p mach-server --release --features hip`；
- Python 3 环境带 transformers 5.16.1 + Pillow（视觉塔 golden 还需要 torch/safetensors）。

## 安全注意

- 一次只加载一个服务；启动前确认没有其它推理进程占用 GPU；
- 27B 必须走 Q4（`MACH_Q4=1`），`MACH_CAPACITY=1`、`MACH_PREFILL_ROWS` 按显存收紧；
- 观察 `server.log` 的 VRAM 预检与运行日志；任何 OOM/驱动错误立即停服并记录。

## 步骤

1) 构建（若尚未构建）：

```powershell
cargo build -p mach-server --release --features hip
```

2) 生成 HF golden（processor 一定能跑；`--tower` 会额外尝试加载 `model.visual.*`）：

```powershell
python tools/vision_c4_golden.py --model-dir .models/qwen3.8-27b `
  --image artifacts/vision-c4/input.png --out artifacts/vision-c4/hf_golden.json --tower
```

3) 跑 mach-server 图片请求（脚本自动生成 data-URL 图片、等待 /healthz、记录响应/耗时并停服）：

```powershell
python tools/vision_c4_e2e.py --binary target/release/mach-server.exe `
  --model-dir .models/qwen3.8-27b --out-dir artifacts/vision-c4 `
  --extra-env MACH_Q4=1 --extra-env MACH_CAPACITY=1
```

4) 负例（同一服务或重启后）：

- 不设 `MACH_VISION` 时图片请求应返回 501 `multimodal_not_implemented`；
- 超 `MACH_VISION_MAX_TOKENS` 的图片应返回 400，不得空完成；
- 纯文本请求行为与开启 vision 前一致。

## 验收

- 服务启动日志出现 vision 配置与权重加载；`/healthz` 200；
- 图片问答请求 200，回答能描述图片内容且 temperature=0 下多次一致；
- `tools/vision_c4_golden.py --tower` 的 `features_*` 与 machserve 视觉塔输出在约定容差内一致（若 HF 侧无法加载视觉塔，记录原因并至少对拍 processor grid/pixel_values）；
- VRAM/TTFT/耗时写入 `artifacts/vision-c4/summary.json` 并回填 Issue #146；
- 文本-only 回归、501/400 负例通过。

## C4 仍待处理

- 高分辨率图片 tiling（当前 vision attention 段上限 `max_seg <= 8192`）；
- EXIF orientation 与 HF `load_image` 对齐；
- Fast/torchvision 与 PIL 归一路径的数值漂移复核；
- vision × decode graph capture；HTTP client 连接池复用；CPU 预处理移入 `spawn_blocking` 或引擎线程；
- 多图/并发请求的总显存预算。
