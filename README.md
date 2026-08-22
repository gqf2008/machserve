# MachServe

> 马赫 —— 突破音障,超越 TokenSpeed。

MachServe 是一个**除内核外全部使用 Rust** 编写的高性能 LLM 推理引擎。
目标是超越 [TokenSpeed](https://github.com/lightseekorg/tokenspeed) 的端到端性能:
host 侧(调度/采样/内存/图捕获)用 Rust 做到零 Python 开销,GPU 侧直接引用
业界最优秀的第三方内核(flashinfer / cutlass / trtllm / gluon)。

## 定位

- **只做推理引擎**,不做训练框架。
- **内核边界唯一化**:`mach-kernel` 是运行时唯一内核入口,`mach-kernel-sys` 是唯一 FFI 入口;
  第三方内核代码只存在于 `thirdparty/`,通过 `ops/<family>/<solution>` 引入并注册。
- **不依赖 libtorch / tch-rs / burn 运行时**;burn 仅作为设计参考。
- 目标硬件:**AMD(ROCm / HIP)优先**(当前:7900 XTX / gfx1100),CPU 作参考/开发后端。

## 架构

```
mach-engine      核心:Device / DType / Shape / 内存池 / Stream-Event / CUDA Graph 生命周期
mach-kernel      唯一内核边界:Kernel trait + op 注册表 + ops/<family>/<solution>
mach-kernel-sys  唯一 FFI 边界:thirdparty/ 中 C/CUDA 库的绑定与加载
mach-scheduler   连续批处理调度器(P2,复用 tokenspeed-scheduler-core 能力)
mach-model       模型定义 / MoE / MLA / FP8(P1+)
mach-sampling    采样 / spec-decode(P2+)
mach-distributed NCCL + TP/PP/EP(P4)
mach-server      axum OpenAI 兼容 API(P2)
thirdparty/      flashinfer / cutlass / trtllm / gluon / candle-kernels(全部 pin 版本)
```

## 当前状态

- **P0 + P0.5 + P1a(已完成)**:工作区骨架、mach-engine 核心抽象、mach-kernel 边界、
  mach-kernel-sys **HIP FFI + hiprtc kernel 运行时 + hipBLAS**、mach-bench 基准、
  **mach-model decode 切片**(GQA 小 transformer + 静态 KV + HIP graph 捕获重放)。
  7900 XTX 真机验收:GPU 解码与 CPU 参考逐 token 一致、graph 重放与 eager 一致,
  launch-only 路径 graph 每 token 快 ~2-5x。
- 详细路线图见 [`docs/roadmap.md`](docs/roadmap.md)。

## 构建

```bash
# 默认(CPU,无 CUDA 依赖)
cargo build --workspace
cargo test  --workspace
cargo clippy --workspace --all-targets
cargo fmt --all --check
```

启用 CUDA(需要本机 CUDA 工具链,未默认开启):

```bash
cargo build -p mach-engine --features cuda
```

## License

MIT OR Apache-2.0


