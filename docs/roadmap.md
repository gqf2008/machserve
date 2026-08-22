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
