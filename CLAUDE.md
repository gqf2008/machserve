# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

MachServe 是**除内核外全部 Rust** 的 LLM 推理引擎,目标硬件为 **AMD GPU(ROCm/HIP,当前 RX 7900 XTX / gfx1100,Windows 原生 ROCm 6.2)**。host 侧(调度/采样/内存/图捕获/HTTP)零 Python;GPU 侧走 hipBLAS GEMM + hiprtc 运行时编译的自有内核。CUDA 不在当前路线(`cuda` feature 仅为占位)。主分支是 `master`。文档与提交描述用中文。

## 常用命令

```bash
# 本地门禁(CI 同款,提交前必跑)
cargo fmt --all --check
cargo clippy --workspace --all-targets --features hip -- -D warnings
cargo check --workspace --all-targets                    # CPU-only 面(必须始终可编译)
cargo check --workspace --all-targets --features hip     # HIP 面 type-check(CI 是 check×2,两面都要过)
cargo test --workspace --lib                             # CPU 单元测试
cargo test -p mach-model --test decode_slice --test fp16 --test load_safetensors --test spec_decode  # CPU 集成测试

# GPU 回归(需本机 ROCm + GPU;必须单线程 —— ROCm Windows 并发 GPU setup 会死锁)
cargo test --workspace --features hip -- --test-threads 1
cargo test -p mach-engine --features hip --lib -- --ignored --test-threads 1   # #[ignore] 的 GPU 测试
cargo test -p mach-server --features hip

# 单个测试示例
cargo test -p mach-model --features hip --test batched batched_paged_decode_matches_static_gpu -- --test-threads 1

# 真实模型测试:设置 MACH_TEST_MODEL 指向 .safetensors 或模型目录(未设置自动 skip,不跑多 GB 加载)
MACH_TEST_MODEL=.models/qwen-0.5b.safetensors cargo test -p mach-model --features hip --test real_model -- --test-threads 1

# 跑服务 / 排障 / 基准(模型放 .models/,已 gitignore)
cargo run -p mach-server --release --features hip          # OpenAI 兼容 API,默认 127.0.0.1:8080
cargo run -p mach-server --release --features hip -- doctor  # 一键排障:GPU/显存/环境/模型检查
cargo run -p mach-model --release --features hip --example qwen_bench    # decode 吞吐/批量/prefill 基准
cargo run -p mach-model --release --features hip --example chat_check    # 真实对话验证
```

运行时环境变量(MACH_MODELS / MACH_MODEL / MACH_CONFIG / MACH_TOKENIZER / MACH_CAPACITY / MACH_PREFILL_ROWS / MACH_MOE_SLOTS / MACH_ADDR / MACH_DTYPE / MACH_Q4 / MACH_FP8 / MACH_SPEC / MACH_SPEC_K / MACH_DRAFT / MACH_DRAFT_CONFIG / MACH_HIP_PATH 等)以 `crates/mach-server/src/main.rs` 中 `env::var` 的读取点为准(顶部文档注释只列了其中一部分)。

## 架构(crates,自底向上)

```
mach-kernel-sys    唯一的 ROCm/BLAS FFI 边界(基准 example 的 psapi 显存计数除外):运行时动态加载
                   amdhip64_6.dll(回退 amdhip64.dll)/ hiprtc0602.dll / hipblas.dll(libloading,无链接期
                   依赖;ROCm bin 目录可用 MACH_HIP_PATH 覆盖;flashinfer/cutlass feature 需 MACH_THIRDPARTY 指向预编译库)
mach-engine        后端无关核心:Device/DType/Shape、内存池(CPU + HipMemoryPool)、
                   stream/event、图捕获(SoftwareGraphCapture + HipGraphCapture)
mach-kernel        Kernel trait / op registry 内核边界抽象(主要为设计骨架 + mach-bench 面)
mach-model         模型层(见下)
mach-server        axum OpenAI 兼容 API(completions/chat/SSE)+ doctor 子命令
mach-bench         host 侧派发开销微基准
```

### mach-model 内部分层(理解本仓库的关键)

- **CPU 参考栈(无需 GPU,CI 可跑)**:`ref_model` / `cpu_engine` / `fp64_ref` / PagedRef(`paged_kv` 的 CPU 全变压器)。它们既是行为规范也是 GPU 接线的蓝图。
- **GPU 路径**:`kernels.rs` 集中放所有 HIP 内核源码(hiprtc 运行时编译,进程内有编译缓存);`model.rs`(GpuModel 单序列)→ `batched.rs`(BatchedModel 批量 decode,分页 KV 经 `with_paged_kv` 接入)→ `continuous.rs`(ContinuousModel 连续批处理:prefill/decode 混合、EOS、槽位压缩);另有 `prefill_buffered.rs`(双缓冲 prefill,下一层权重预取与当前层计算重叠)与 `sampling.rs`(GPU 批量采样:温度/top-k/top-p/惩罚,crate 内最大模块)。
- **调度/复用(TokenSpeed 对齐线)**:`scheduler_fsm`(7 状态 FSM)、`kv_block_pool`(LCM 块池+RAII)、`prefix_cache`(SHA-256 前缀哈希链)、`reuse_planner` / `prefix_kv` / `state_reuse`(跨请求前缀共享与多轮状态复用)、`paged_scheduler`。
- **量化/特化**:`q4`(存储级 int4)/ `fp8`(E4M3 存储,计算级已证伪)/ `moe_backend` + `moe_offload`(LRU 专家缓存 + host RAM offload)/ `adaptive*`(带宽自适应 offload 决策)/ `speculative`(spec-decode,实测净负收益,**暂停投入**)。
- `loader.rs`:纯 Rust safetensors(F32/F16/BF16、多 shard、Llama/Qwen/DeepSeek-MLA/Qwen-MoE/Qwen3.5-GDN 键名映射);`tokenizer.rs`:字节级 BPE。

### 请求流与线程模型

HTTP handler(axum)→ channel → **唯一后台引擎线程**(模型/GPU 状态只在该线程,`mach-server/src/engine.rs`)→ ContinuousModel → BatchedModel → hipBLAS GEMM + hiprtc 内核。HIP graph 捕获重放在 GpuModel 单序列路径(`model.rs` 的 `capture_decode`)与服务链(#103,MACH_GRAPH 实验开关)均已接入;#103 实测服务链 host-bound、graph 零收益,30B 图捕获在 ROCm 6.2/Windows 驱动上有腐化缺陷,该方向暂停。

## 关键约定

1. **双编译面**:`--features hip` 开/关都必须编译通过且 CPU 测试绿(CI 两面都 check)。GPU 代码全部 `#[cfg(feature = "hip")]`,CPU 参考路径不依赖 hip。
2. **离线内核编译门禁**:新增 HIP 内核源码(`kernels.rs` 的 `const`)**必须**同步加入 `kernels.rs` 内 `offline_tests` 模块的 `ALL_KERNELS` 列表(当前 77 个;计数由 `kernel_count_matches_documented_gate` 测试机器校验——改列表须同步更新该断言与本文档计数)——hiprtc 只需 ROCm 运行时、无需 GPU 设备即可离线编译。注意:该测试是 `#[cfg(all(test, feature = "hip"))]`；显式运行它时若 ROCm 运行时不不可用会 fail-loud，不再 skip。CPU CI 不带 hip feature、不执行它（GPU job 仅 workflow_dispatch 手动触发），坏内核靠**本机** `cargo test -p mach-model --features hip --lib offline` 拦下，而不是 CPU CI。
3. **正确性方法论**:每条 GPU 路径都要有 CPU 参考对拍测试(logits 逐位/容差一致);独立 fp64 参考在 `tools/ref_llama.py`。真机对拍记录进 `docs/roadmap.md` 进度日志。
4. **GPU 测试**:`--test-threads 1` 强制;sampling 的 GPU 测试 `#[ignore]`d(`-- --ignored` 显式跑);真实大模型测试用 `MACH_TEST_MODEL` 门控。
5. **文档同步**:里程碑完成后在 `docs/roadmap.md` 追加进度日志条目(含验证命令与结果),README 性能地图与 `docs/tokenspeed-alignment.md` 状态表按需同步;基准方法论在 `docs/benchmark-protocol.md`。
6. **性能声称必须有实测**:每个优化方向记录真 A/B 数据;已证伪方向(split-K、FP8 计算级、spec-decode 等)记入 README 性能地图,不再重复投入。

## 开发流程(issue 驱动)

- 每个工作单元一个 GitHub issue(同类 ≥3 条合并为批次 issue + checklist);基于 `master` 开 worktree 开发,**不直接在 master 上改**;PR 关联 issue。
- 提交信息用 conventional commits(`feat(model):` / `fix:` / `chore:` / `test:` / `docs:` + 中文描述 + issue 号;标题末尾的 `(#N)` 是 squash merge 自动附带的 PR 号,不是手写;更早历史提交为英文,近期起统一中文);PR 审查通过 + CI 绿(fmt / clippy -D warnings / check×2 / CPU 测试)后 squash merge,合并后清理分支与 worktree。
- 开始处理 issue 前在 issue 上标记"处理中"并注明 worktree,避免多 agent 撞车。
