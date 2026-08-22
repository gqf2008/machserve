# MachServe 路线图

> 目标:除内核外全部 Rust,端到端性能超越 TokenSpeed。
> 验收:同模型、同权重、同请求分布,对比 TTFT / TPOT / 吞吐 / GPU 利用率。

## 分阶段

| 阶段 | 交付物 | 验收(exit criteria) |
|---|---|---|
| P0 地基 | 工作区骨架 + mach-engine(device/内存/stream/graph 生命周期)+ mach-kernel 边界 + mach-kernel-sys FFI 骨架 + 基准框架 | 全绿构建;软件 graph 捕获/重放测试;注册表调度基准 |
| P1 单模型 decode 链路 | 小模型:静态 KV + CUDA graph 化 decode + eager prefill + safetensors 权重加载 | 输出与参考实现逐 token 一致;TPOT 对标 TokenSpeed 同 kernel 场景 |
| P2 引擎化 | mach-scheduler(复用 ts-scheduler-core)+ 连续批处理 + 采样 + axum OpenAI server | 多请求延迟/吞吐基线 vs TokenSpeed/vLLM |
| P3 性能主力 | MoE + FP8 + MLA(flashinfer)+ spec-decode + AMD(gluon) | 吞吐追上/局部超越;GPU util 达标 |
| P4 分布式 | NCCL + TP/PP/EP,通信与计算重叠 | 多卡扩展效率曲线对比 |
| P5 打磨 | 正确性契约全绿 + 性能矩阵 + 稳定化 | "超越"指标表逐项确认 |

## 关键决策(已确认)

- **不用 burn 运行时**:burn 是训练/通用框架,serving 用不上 autodiff/dispatch;其 graph capture 设计作为参考抄入 `mach-engine`。
- **不用 libtorch / tch-rs**:框架非内核,会把 C++ 运行时拉回。
- **内核 = 第三方优秀实现**(flashinfer/cutlass/trtllm/gluon),通过 `mach-kernel-sys` FFI 接入。
- **CUDA 控制用 cudarc**(stream/graph/内存),`cuda` feature 默认关闭。
- **P1 是 go/no-go 闸门**:TPOT 追不平立即复盘,不盲目继续。

## "超越"的抓手

1. host 侧每 token 开销:无 GIL / 无 Python 解释器,调度采样全编译期确定;
2. CUDA graph 覆盖率更高(整条 decode 链);
3. 内存确定性:自研 caching allocator + graph pool,零碎片;
4. 显式 stream/event 重叠:prefill↔decode、通信↔计算;
5. 单静态二进制部署。

## 正确性

- 契约测试:参考实现输出固化为 golden 文件,每阶段全量跑。
- 数值:FP8 场景有界误差,用 tolerance 断言。

## 平台调整(2026-08-22,已确认)

- **目标 GPU = AMD Radeon RX 7900 XTX(gfx1100,24G,Windows 原生 ROCm 6.2)**。
- 路线改为 **AMD/HIP 优先**:`mach-kernel-sys` 提供 HIP FFI(动态加载 amdhip64_6.dll + hiprtc0602.dll),
  `mach-engine` 提供 `HipMemoryPool` / `HipGraphCapture`(HIP graph 捕获 = AMD 版 CUDA Graph)。
- tokenspeed-kernel-amd 目前只有 gfx950/gfx1250;7900 XTX 的 kernel 走自有 HIP/hiprtc 路径,
  后续可参考 Gluon(gfx1100 支持)补充。
- P1 验收改为:小模型 decode 链路在 7900 XTX 上跑通,TPOT 对标 TokenSpeed(同 kernel 场景)。

## 进度日志

- **P0 完成(2026-08-22)**:workspace 骨架、mach-engine 核心抽象、mach-kernel 内核边界、
  mach-kernel-sys FFI 骨架、mach-bench 基准(host 派发 ~86-90 ns/op)。
- **P0.5 AMD/HIP 地基完成(2026-08-22,真机验证)**:
  - HIP 运行时 FFI(纯 Rust 动态加载,无链接期依赖);
  - hiprtc 运行时编译 HIP kernel + hipModule 加载/启动;
  - `HipMemoryPool`(malloc/free/pin)与 `HipGraphCapture`(严格生命周期 + 捕获/重放);
  - 7900 XTX 实测:`hiprtc_saxpy_runs_on_gpu` / `hip_graph_capture_records_and_replays`
    等 4 个 GPU 测试全部通过。
