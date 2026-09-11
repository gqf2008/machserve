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
- VRAM 采样优先用 `rocm-smi`；Windows ROCm 常不含该命令（本机就没有），此时
  自动回退到 `mach-server doctor`（走 `hip::mem_info()`，并保留 `MACH_HIP_PATH`
  以便定位自定义 ROCm 安装）。summary 的 `vram_before`/`vram_after` 是
  `{source,value,error}`：采样失败只记进 `error`，不会冒充采样值，且
  `vram_ok=false` 会让 E2E 退出码非 0；
- 27B 必须走 Q4（`MACH_Q4=1`），`MACH_CAPACITY=1`、`MACH_PREFILL_ROWS` 按显存收紧；
- 观察 `server.log` 的 VRAM 预检与运行日志；任何 OOM/驱动错误立即停服并记录；
- E2E 脚本会在 finally 中停服；若异常退出，手动确认 `mach-server` 进程已结束再重跑。

## CPU 真权重对拍（无需 GPU，建议先跑）

`crates/mach-model/tests/vision_real_weights.rs` 用同一张图把 MachServe 的
CPU 视觉塔钉在 transformers 上。它是 opt-in 的：两个环境变量都不设时打印 SKIP
并返回，只设其中一个会直接失败（不静默降级）。注意 libtest 默认只显示
`1 passed`，必须用 `--nocapture` 确认看到 SKIP 行或对拍结果行——**SKIP 不算验证**。
128x128 输入（grid `[1,16,16]`）单线程约 11 分钟，所以不放进默认门禁。

```powershell
# 1) 生成 golden + raw f32（CPU，约 10s；--parity-export 产出对拍所需文件）
python tools/vision_c4_golden.py --model-dir .models/qwen3.8-27b `
  --image artifacts/vision-c4/input.png --out artifacts/vision-c4/hf_golden.json `
  --tower --allow-pil-fallback --parity-export
# 2) 跑 CPU 对拍
$env:MACH_VISION_GOLDEN = "artifacts/vision-c4"
$env:MACH_VISION_MODEL  = ".models/qwen3.8-27b"
cargo test -p mach-model --test vision_real_weights -- --nocapture
```

实测（2026-09-11，grid `[1,16,16]`、64x5120 merged features）：容差为
`|got-ref| <= 1e-3 * max(1, |ref|)`，`worst_ratio = 0.4488`（最差元素 index
320103：`-1.2215794` vs `-1.2210314`，允许 `1.22e-3`，即只用到 45%），
`nonfinite = 0`，单线程 687s。GPU 窗口建议以 CPU 参考为中间基准：先 CPU↔HF，
再 GPU↔CPU。

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

3) 开真机前先做一次预检（不启动服务、不加载模型；会做一次轻量 `doctor` 设备
   查询以确认 VRAM 采样可用）：

```powershell
python tools/vision_c4_e2e.py --dry-run `
  --binary target/release/mach-server.exe --model-dir .models/qwen3.8-27b `
  --image artifacts/vision-c4/input.png
```

预检会检查二进制是可执行文件、模型目录有 `config.json` /
`preprocessor_config.json` / `tokenizer.json` 与 safetensors 分片、图片存在、
`--max-patches > 0`、输出目录可写、端口可绑定，并实际探测 VRAM 采样
（`rocm-smi` 缺失时回退 `mach-server doctor`）。`problems` 非空或 VRAM 采样不可用
时退出码为 1，先修再开真机。

4) mach-server E2E（同一张图、stream TTFT、SHA-256、VRAM 采样；dump MachServe merged features）：

```powershell
python tools/vision_c4_e2e.py --binary target/release/mach-server.exe `
  --model-dir .models/qwen3.8-27b --image artifacts/vision-c4/input.png `
  --out-dir artifacts/vision-c4 --max-patches 2048 `
  --extra-env MACH_Q4=1 --extra-env MACH_Q4_DEVICE=2 --extra-env MACH_CAPACITY=1 `
  --extra-env MACH_VISION_DUMP=artifacts/vision-c4/ms_features
```

注意 **`MACH_Q4_DEVICE=2` 不能省**：27B 是 dense，`MACH_Q4=1` 只让 host 权重保持
int4，设备上仍会把 dense 权重解量化成 f16（`doctor` 估算 ~53 GiB，超出 24GB 卡）；
只有 `MACH_Q4_DEVICE=2` 会把 dense 张量也按原始 int4 常驻设备（本次服务 preflight
实测 `estimated need 16.42GiB`）。`estimate_vram` 对 `=1`/`=2` 用同一个系数，所以
这条结论来自 Q4-on-device 的运行时语义与实测 preflight，而不是估算差值。

`--max-patches 2048` 覆盖 `MACH_VISION_MAX_TOKENS`：128x128 图会先按 `min_pixels`
放大到 256x256，得 grid `[1,16,16]` = **256 patch（merge 后 64 个视觉 token）**，
2048 足够且把 vision scratch 压到最小；不设时默认 8192。

5) 数值对拍（HF `hf_golden_features.npy` vs MachServe `ms_features.bin/json`；compare 同时校验 grid 与输入 SHA-256 绑定）：

```powershell
python tools/vision_c4_compare.py `
  --hf-npy artifacts/vision-c4/hf_golden_features.npy `
  --ms-prefix artifacts/vision-c4/ms_features --hf-json artifacts/vision-c4/hf_golden.json `
  --e2e-summary artifacts/vision-c4/summary.json --require-hash --atol 1e-3 --rtol 1e-3
```

6) 负例与确定性（脚本会起停服务两次；需真机）：

```powershell
python tools/vision_c4_negative.py --binary target/release/mach-server.exe `
  --model-dir .models/qwen3.8-27b --image artifacts/vision-c4/input.png `
  --out artifacts/vision-c4/negatives.json `
  --extra-env MACH_Q4=1 --extra-env MACH_Q4_DEVICE=2 --extra-env MACH_CAPACITY=1
```

检查项：图片 200、同图两次 `temperature=0` 逐字一致、超 patch 预算 400、纯文本 200、
不带 `MACH_VISION` 时图片 501 `multimodal_not_implemented`；报告写入
`artifacts/vision-c4/negatives.json`，任一不符退出码非 0。

## 验收

- 服务启动日志出现 vision 配置与权重加载；`/healthz` 200；
- E2E 每次只发一个 vision 请求；`MACH_VISION_DUMP` 会被覆盖，summary 校验 dump 非空且 mtime 新于请求开始，SSE `data: {"error": ...}` 会让脚本返回非 0；
- 图片问答请求 200，回答能描述图片内容；`summary.json` 含 `ttft_seconds`、VRAM 采样（`vram_ok=true`）、输入 SHA-256；
- `vision_c4_compare.py` 的 `pass` 采用 numpy 混合容差（`|d| <= atol + rtol*|ref|`）：`max_abs_diff` 与 `mean_abs_diff` 应远小于 atol，`max_rel_diff` 在近零特征上天然很大，必须结合混合容差与 `nonfinite == 0` 一起读（HF 侧必须是 transformer 视觉塔真实输出；PIL 回退只用于 processor 参考）；
- temperature=0 多次运行输出一致；HF 整模型 greedy token/logits 参考由操作员按现有 HF 环境补充并记录；
- 文本-only 回归、501/400 负例通过；VRAM/TTFT/耗时写入 `artifacts/vision-c4/summary.json` 并回填 Issue #146。

## C4 真机实测（2026-09-11，7900 XTX / ROCm 6.2 / Windows）

本次完成的是**HTTP 端到端多模态链路**与**视觉塔 GPU↔HF 特征对拍**；issue #146 验收里
的「整模型 HF token/logits 对齐」仍未做（见文末待办）。

输入 `artifacts/vision-c4/input.png`（128x128，SHA-256 `4550d45d…`），模型
`.models/qwen3.8-27b`，`MACH_Q4=1 MACH_Q4_DEVICE=2 MACH_CAPACITY=1
MACH_VISION_MAX_TOKENS=2048`（release 二进制）：

- 启动/加载：从进程启动到**首个请求开始**共 **123.3s**（`summary.json` 的
  `load_plus_wait_seconds`，含 healthz 等待与一次 VRAM 采样；纯 healthz 时刻未单独记录）；日志
  `GPU preflight: device_count=2, VRAM free 23.84GiB / 23.98GiB, estimated need 16.42GiB`、
  `vision: depth=27 hidden=1152 merge=2 image_token=248056 max_patches=2048`。
- 图片问答：HTTP **200**，SSE 流式回答
  “A red circle, a white square, and a yellow triangle are arranged on a dark blue
  background.”，**TTFT 3.15s**（含 prefill+vision 前向）；`temperature=0` 连发两次
  输出逐字一致。
- GPU↔HF 对拍（`vision_c4_compare.py --atol 1e-3 --rtol 1e-3 --require-hash`）：
  **pass=true**，`grid_match=true`、`image_hash_match=true`、`nonfinite=0`、
  **max_abs_diff 7.362e-4**、mean_abs 1.626e-6。`pass` 用的是混合容差，所以同一份
  输出里的 `max_rel_diff 22.6`（出现在近零特征上）与 pass 不矛盾；这里主要看
  `max_abs_diff`。
- 负例与确定性（`tools/vision_c4_negative.py`，报告
  `artifacts/vision-c4/negatives.json`）：超过 patch 预算的 2048x2048 图
  （grid 128x128 = 16384 patch）返回 **400**
  `image grid 128x128 = 16384 patches exceeds limit 2048`；不带 `MACH_VISION` 时
  图片请求返回 **501** `multimodal_not_implemented`；同服务纯文本请求 **200**；
  同一张图两次 `temperature=0` 输出逐字一致。
- Fast/PIL 漂移：装上 torchvision 0.29.0+cpu 后用 `AutoProcessor` 走 Fast 路径与
  PIL 路径对比同一张图，grid 相同、`max_abs_diff 5.9e-8`、mean 3.3e-9（ULP 级）；
  且 **PIL 参考与 Rust 生产预处理逐位相同**（`pil vs rust max_abs 0.0`）。
- VRAM 采样注意：`mach-server doctor` 是**另一个进程**，Windows 驱动只报该进程的
  空闲值（服务运行中仍报 23.84 GiB free），所以 `summary.json` 的 `vram_*` 只能当
  背景参考；服务日志的 preflight 行（`VRAM free 23.84GiB … estimated need
  16.42GiB`）也只是**加载前的预算**。本次**没有采到实际占用**——要采需在服务进程内
  前后采样，或使用可用的 `rocm-smi`。

## C4 仍待处理

待办：

- HF 整模型 greedy token/logits 参考：本机 31GB 内存装不下 BF16 27B、且 PyTorch
  没有 Windows ROCm 轮子，必须在 ≥64GB 内存的机器（或 Linux+ROCm torch）上生成
  后再与本实现的输出对比。其余真机项见上面「C4 实测结果」。

纯离线 P3（不阻塞真机）：

- 高分辨率图片 tiling（当前 vision attention 段上限 `max_seg <= 8192`）；
- 多图/并发请求的总显存预算（连接池复用与 CPU 预处理已由 PR #159 覆盖）。

已暂停/不做：vision × decode graph capture（`CLAUDE.md` 记录服务链 graph 零收益、
30B 在 ROCm 6.2/Windows 上有驱动腐化，方向已停）。

已完成（无需在此重复）：EXIF orientation（PR #158）、HTTP client 连接池复用与
CPU 预处理移出 async worker（PR #159）、CPU 真权重视觉塔对拍（PR #160）、
图像解码/预处理/SSRF 用例进入 CPU 测试面（PR #162，`mach-server --lib` 的 CPU
面从 0 个测试变为 29 个）。注意这只覆盖这些用例；仓库其它 hip-only 测试面的同类
假绿已另开批次 issue 跟踪，不在本 runbook 范围。
