# MachServe

> 马赫 —— 突破音障,超越 TokenSpeed。

MachServe 是一个**除内核外全部使用 Rust** 编写的高性能 LLM 推理引擎(HIP/ROCm,
Windows 原生)。host 侧(调度/采样/内存/图捕获/HTTP)全部 Rust,零 Python 开销;
GPU 侧直接调用 AMD hipBLAS/hiprtc 运行时编译的内核。

## 当前战绩(2026-08-23,AMD RX 7900 XTX / gfx1100,Windows 原生 ROCm 6.2)

| 指标 | 结果 |
|---|---|
| **decode 吞吐(B=512)** | **35251 tok/s ≈ llama.cpp Vulkan(643 tok/s)的 55x** |
| decode 吞吐(B=64,短 ctx) | 12887 tok/s(4.97 ms/step) |
| **长 context decode(2048)** | **13.40 ms/step(4778 tok/s/seq,GQA 复用 2.6x)** |
| **长 prompt TTFT** | 512-token 57ms / 2048-token 289ms(分块 prefill) |
| 上下文能力 | 8192 tokens(fp16 KV) |
| **数值正确性** | GPU vs 真 transformers 模型最终 logits 差 **4e-5**,chat 回答正确 |
| **OpenAI API** | completions / chat / SSE / 采样全参数 / top_logprobs / stop / n / usage |
| 模型兼容 | Qwen2.5-0.5B / **1.5B**(F32+F16 双路径,head_dim 128) |

> 测量口径:Qwen2.5-0.5B fp16、capacity 64,`lctx_bench`(不 reset、真长 context)。
> 详细历史见 [docs/roadmap.md](docs/roadmap.md)。

## 定位

- **只做推理引擎**,不做训练框架。
- **除内核外全 Rust**:内核(GEMM/attention/采样)走 hipBLAS + hiprtc 运行时编译;
  burn 仅作设计参考,不依赖 libtorch/tch-rs/burn 运行时。
- 目标硬件:**AMD(ROCm / HIP)优先**,当前 RX 7900 XTX / gfx1100。
- 对标物为本机可运行的 **llama.cpp(Vulkan 后端)**(同 GPU、同模型)。

## 架构(crates)

```
mach-engine        Device / DType / 内存 / Stream / HIP graph 捕获
mach-kernel-sys    唯一 FFI 边界:amdhip64_6.dll / hiprtc0602.dll / hipblas.dll 动态加载
mach-model         模型:config / safetensors 加载 / fp16 / 连续批处理 / 采样 / tokenizer
mach-server        axum OpenAI 兼容 API(completions / chat / SSE 流式)
mach-bench         基准框架
thirdparty/        第三方参考代码(占位)
```

## 能力清单

- **GQA 复用 decode attention**:每 KV head 一个 block、K/V 各读一次跨组复用
  (长 context KV 流量降 groups 倍),2048-token decode **2.6x**(34.8→13.4 ms/step)。
- **fp16 计算**:权重/激活/GEMM 输入 fp16、fp32 累加(hipBLAS `GemmEx_v2`);
  隐藏层 GEMM 输出 fp16 + cast(瘦长形状 c16 比 c32 快 3-4x);**fp16 KV cache**。
- **连续批处理**:序列生命周期(prefill/decode 混合、EOS/max_new、槽位压缩复用)。
- **分块 prefill**:每步消费最多 `capacity` 个 prompt tokens,长 prompt TTFT 降 ~58x。
- **GPU 采样**:top-k / top-p / temperature,确定性 SplitMix64 RNG,CPU 参考可对拍。
- **真实 tokenizer**:字节级 BPE(Qwen2.5/Llama),与 HF tokenizers 逐 token 对拍一致。
- **SSE 流式**:`stream: true` → 逐 token delta + `[DONE]`,增量 UTF-8 跨 token 不分裂。
- **对话模板**:Qwen chat template,`<|im_end|>` 停止。
- **正确性**:GPU vs 独立 fp64 numpy 参考(~1e-4)+ 真 transformers 模型(4e-5)。

## 构建与运行

```bash
# 构建 + 测试(默认 / HIP)
cargo build --workspace
cargo test  --workspace
cargo test  --workspace --features hip -- --test-threads 1   # 需 ROCm + GPU
cargo clippy --workspace --all-targets --features hip
cargo fmt --all --check

# 下载模型(首次)
curl -L -o .models/qwen-0.5b.safetensors \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors
curl -L -o .models/qwen-config.json \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json
curl -L -o .models/tokenizer.json \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json

# 基准(Qwen2.5-0.5B,decode 吞吐 / 批量 / prefill TTFT / 长 context)
cargo run -p mach-model --release --features hip --example qwen_bench

# 真实对话诊断(验证模型输出正确)
cargo run -p mach-model --release --features hip --example chat_check

# 启动 OpenAI 兼容服务器(fp16 默认;MACH_DTYPE=f32 可关)
cargo run -p mach-server --release --features hip
#   环境变量:MACH_MODELS(默认 .models)、MACH_MODEL、MACH_CONFIG、
#   MACH_CAPACITY(默认 64)、MACH_ADDR(默认 127.0.0.1:8080)、MACH_DTYPE(f16/f32)

# 调用示例
curl -s http://127.0.0.1:8080/v1/chat/completions -H "content-type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":40}'
curl -sN http://127.0.0.1:8080/v1/chat/completions -H "content-type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":20,"stream":true}'
```

## 关键决策

- **HIP 优先**(7900 XTX / gfx1100,Windows 原生 ROCm 6.2);CUDA 不在当前路线。
- **fp16 路径**:权重/激活/GEMM fp16 + fp32 累加;隐藏层 GEMM 输出 fp16(c16 比 c32
  快 3-4x,瘦长形状的关键调优);fp16 KV cache(内存减半、attention 带宽减半)。
- **不用 burn / libtorch / tch-rs 运行时**;hipBLAS `GemmEx_v2` 提供 fp16 GEMM。
- 详细路线与踩坑见 [`docs/roadmap.md`](docs/roadmap.md)。

## License

MIT
