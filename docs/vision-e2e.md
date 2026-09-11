# Qwen3.8 Vision E2E（C4）Runbook

Stage C 的离线链路已经合入；本文档描述真机 GPU 窗口打开后如何完成 C4 硬门禁。
不要在本文档的步骤外并发跑编译或第二次推理：此前多次整机崩溃都发生在并行重载时。

## 前置

- Windows + ROCm，AMD 7900 XTX 24GB；
- 真实 checkpoint `.models/qwen3.8-27b`（18 shard，含 `model.visual.*`）；
- release 二进制：`cargo build -p mach-server --release --features hip`；
- Python 3 环境带 transformers 5.16.1 + Pillow + numpy；视觉塔 golden 还需要 torch + safetensors；Fast 图像处理器还需要 torchvision（缺失时只能用 PIL 回退参考，必须在 golden 里标注）。

## 安全注意

- 一次只加载一个服务；启动前确认没有其它推理进程占用 GPU；
- 27B 必须走 Q4（`MACH_Q4=1`），`MACH_CAPACITY=1`、`MACH_PREFILL_ROWS` 按显存收紧；
- 观察 `server.log` 的 VRAM 预检与运行日志；任何 OOM/驱动错误立即停服并记录；
- E2E 脚本会在 finally 中停服；若异常退出，手动确认 `mach-server` 进程已结束再重跑。

## 步骤

0) 准备同一张输入图片 `artifacts/vision-c4/input.png`（golden 与 E2E 必须使用它，脚本会记录 SHA-256）。

1) 构建（若尚未构建）：

```powershell
cargo build -p mach-server --release --features hip
```

2) HF golden：processor 恒可输出；`--tower` 只打开含 `model.visual.*` 的 shard 并 dump merged features 到 `.npy`。无 torchvision 时必须显式加 `--allow-pil-fallback` 并接受 PIL 参考：

```powershell
python tools/vision_c4_golden.py --model-dir .models/qwen3.8-27b `
  --image artifacts/vision-c4/input.png --out artifacts/vision-c4/hf_golden.json `
  --tower --allow-pil-fallback
python tools/vision_c4_golden.py --smoke   # 可选：tiny CPU 前向自检 hidden_states/pooler_output
python tools/vision_c4_e2e.py --selftest  # 可选：SSE error/TTFT 解析自检
```

3) mach-server E2E（同一张图、stream TTFT、SHA-256、VRAM 采样；dump MachServe merged features）：

```powershell
python tools/vision_c4_e2e.py --binary target/release/mach-server.exe `
  --model-dir .models/qwen3.8-27b --image artifacts/vision-c4/input.png `
  --out-dir artifacts/vision-c4 --extra-env MACH_Q4=1 --extra-env MACH_CAPACITY=1 `
  --extra-env MACH_VISION_DUMP=artifacts/vision-c4/ms_features
```

4) 数值对拍（HF `hf_golden_features.npy` vs MachServe `ms_features.bin/json`；compare 同时校验 grid 与输入 SHA-256 绑定）：

```powershell
python tools/vision_c4_compare.py `
  --hf-npy artifacts/vision-c4/hf_golden_features.npy `
  --ms-prefix artifacts/vision-c4/ms_features --hf-json artifacts/vision-c4/hf_golden.json `
  --e2e-summary artifacts/vision-c4/summary.json --require-hash --atol 1e-3 --rtol 1e-3
```

5) 负例（重启服务后）：

- 不设 `MACH_VISION` 时图片请求应返回 501 `multimodal_not_implemented`；
- 超 `MACH_VISION_MAX_TOKENS` 的图片应返回 400，不得空完成；
- 纯文本请求行为与开启 vision 前一致。

## 验收

- 服务启动日志出现 vision 配置与权重加载；`/healthz` 200；
- E2E 每次只发一个 vision 请求；`MACH_VISION_DUMP` 会被覆盖，summary 校验 dump 非空且 mtime 新于请求开始，SSE `data: {"error": ...}` 会让脚本返回非 0；
- 图片问答请求 200，回答能描述图片内容；`summary.json` 含 `ttft_seconds`、VRAM 采样、输入 SHA-256；
- `vision_c4_compare.py` 的 `max_abs_diff`/`max_rel_diff` 在约定容差内、`nonfinite == 0`（HF 侧必须是 transformer 视觉塔真实输出；PIL 回退只用于 processor 参考）；
- temperature=0 多次运行输出一致；HF 整模型 greedy token/logits 参考由操作员按现有 HF 环境补充并记录；
- 文本-only 回归、501/400 负例通过；VRAM/TTFT/耗时写入 `artifacts/vision-c4/summary.json` 并回填 Issue #146。

## C4 仍待处理

- 高分辨率图片 tiling（当前 vision attention 段上限 `max_seg <= 8192`）；
- EXIF orientation 与 HF `load_image` 对齐；
- Fast/torchvision 与 PIL 归一路径的数值漂移复核；
- vision × decode graph capture；HTTP client 连接池复用；CPU 预处理移入 `spawn_blocking` 或引擎线程；
- 多图/并发请求的总显存预算。
