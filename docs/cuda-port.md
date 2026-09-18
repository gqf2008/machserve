# CUDA / NVIDIA 移植计划（honest）

> 状态：**未开始实施**。本机无 NVIDIA 卡，无法真实验证；本文冻结设计，等有 CUDA 环境（或 NVIDIA 硬件决策）后按此执行。

## 1. 现状（已有，可复用）

> 2026-09 后端重构批次 1 后：原 `mach-engine` 的 `MemoryPool` / `Device` / `DType` / `Shape`
> 与 `cuda.rs` 占位已删除（零用户）；服务级后端边界（`mach-model::backend::ServingModel`）
> 在批次 6-8 落地，CUDA 占位后端随批次 8 接入。本文 P1-P5 的"镜像 hiprtc/NVRTC"思路
> 届时按 NVIDIA CUDA Rust 双轨（cutile-rs Tile JIT / cuda-oxide SIMT）重新评估。

- **唯一 FFI 边界**（`mach-kernel-sys`）：HIP 侧是动态加载 amdhip64_6.dll / hiprtc0602.dll / hipblas.dll；CUDA 侧镜像为 cuBLAS / NVRTC（cuBLAS 动态库 + nvrtc64_*.dll）。

## 2. 要做的（按依赖顺序）

| 步 | 内容 | 依赖 |
|---|---|---|
| P1 | `mach-kernel-sys` 加 `cuda` 后端：动态加载 cudart / cuBLAS / NVRTC，镜像现有 hip.rs 的 `Hip` 结构（`Cuda` + `api` 表） | 有 CUDA 工具链的构建机 |
| P2 | ~~`CudaMemoryPool` 真实现（复用 MemoryPool 契约）；`CudaGraphCapture`~~（MemoryPool trait 与 graph 面已随批次 1/3 退役——本行随批次 8 按 ServingModel 边界重定义） | P1 |
| P3 | GEMM 走 cuBLAS（GemmEx，fp16/fp32 累加路径已定义）；注意力/采样/MoE 内核走 NVRTC 编译（镜像 hiprtc 的 `HipKernelModule` → `CudaKernelModule` + KERNEL_CACHE） | P1 |
| P4 | `mach-model` / `mach-server` 按 `feature` 选后端（`hip` / `cuda` 互斥）；Q4/FP8 存储、offload、prefill 缓冲、状态复用全部后端无关，直接复用 | P2/P3 |
| P5 | FP8 **计算**路径（cuBLAS fp8 GEMM 在 NVIDIA 上可用，正是 FreeToken 的论文路径）——这是 AMD（gfx1100 hipBLAS 拒 fp8）做不到的差异化 | P3 |

## 3. 硬件决策（待拍板）

- 本机 7900 XTX 走 AMD/ROCm 继续推进（当前全链路）。
- 若要量化超大模型（284B/753B 级）或 FP8 计算，**NVIDIA 路径现实得多**（FreeToken 的 fp8/flashinfer/marlin 生态都是 CUDA）；AMD 保持 fp16/bf16 边缘 MoE（20-70B offload）。

## 4. 验收

- P1 后 `cargo check --features cuda` 编译过；~~P2 后 MemoryPool/GraphCapture 契约测试（CpuMemoryPool 参考）在 CUDA 后端复跑全绿~~（契约面已退役，批次 8 重定义为 ServingModel 占位后端编译 + `hip,cuda` 互斥 `compile_error!`）；P4 后 qwen3-moe-tiny Q4/FP8 在 NVIDIA 卡上与 HIP 对拍一致。
