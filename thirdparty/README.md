# thirdparty/ —— 优秀第三方内核代码的唯一存放地

本目录是 MachServe 的**唯一外部内核代码来源**。规则:

1. **只放内核/算子级代码**,不放完整框架(libtorch、pytorch 本体禁止进入)。
2. 每个子目录对应一个第三方项目,**固定 pin 到具体 tag / commit**,可复现构建。
3. 禁止运行时直接依赖:`mach-kernel-sys` 是唯一 FFI 边界,`mach-kernel` 是唯一内核入口。
4. 引入新内核走评审:版本、License、维护活跃度、与现有 `ops/<family>/<solution>` 的关系。

## 计划布局(按优先级)

| 目录 | 项目 | 用途 | 平台 |
|---|---|---|---|
| `flashinfer/` | flashinfer | attention decode/prefill、MLA、group gemm、sampling | CUDA |
| `cutlass/` | cutlass + Cute | fp8/bf16 GEMM、epilogue fusion | CUDA |
| `trtllm-kernels/` | TensorRT-LLM kernels | MoE、量化、融合算子 | CUDA |
| `gluon/` | Gluon | AMD kernel DSL 与内核 | ROCm |
| `candle-kernels/` | HuggingFace candle kernels | 基础算子参考/兜底 | CUDA |
| `vllm-kernels/` | vLLM kernels | 采样/quant/page-attention 参考 | CUDA |

## 引入流程

```bash
# 示例:引入 flashinfer 并 pin
git submodule add https://github.com/flashinfer-ai/flashinfer thirdparty/flashinfer
cd thirdparty/flashinfer && git checkout <release-tag>
# 在 mach-kernel-sys 中为所需入口编写绑定,并登记到 mach-kernel/ops/<family>/
```

## License

第三方代码遵循各自项目的 License;MachServe 本体为 MIT OR Apache-2.0。
