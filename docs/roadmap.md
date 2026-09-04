# MachServe 路线图

> 目标:除内核外全部 Rust,端到端性能对标并超越 **TokenSpeed 与 FreeToken**。
> 验收:同模型、同权重、同请求分布,对比 TTFT / TPOT / 吞吐 / GPU 利用率。
> 平台(2026-09-04 修订):**芯片平台全都要支持** —— 当前已实现 AMD
> (ROCm/HIP,gfx1100);macOS / Windows / Linux 皆为终态,`mach-kernel-sys`
> 唯一 FFI 边界与预留的 `cuda` feature 即为此服务。本机现阶段可复跑的对标
> 是 llama.cpp(Vulkan);TokenSpeed/FreeToken 为跨平台归一的目标对手。
> 节奏:**模型覆盖渐进添加,底子优先** —— 新模型族(如 Qwen3.8 混合线性
> 注意力)以层类型抽象进入,不做单检查点补丁。

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

- **P1a decode 切片完成(2026-08-22,7900 XTX 真机验证)**:
  - `mach-model` crate:小 transformer(GQA attention + RMSNorm + SwiGLU MLP)、静态 KV cache、
    CPU f32 参考实现、HIP 解码路径(embed gather / rms_norm / silu_mul / add / kv_store /
    attn_decode 手写 HIP kernel + hipBLAS GEMM,hiprtc 运行时编译);
  - decode 步拆分为 update_inputs / run_kernels / read_logits,`run_kernels` 可整体捕获为
    HIP graph 并重放(位置/token 通过设备缓冲更新,一张图服务所有位置);
  - **验收测试全绿(真机)**:gpu_matches_cpu_reference(GPU==CPU 对拍)、
    graph_replay_matches_eager(graph 重放==eager)、kv_cache_is_positional;
  - **基准(小模型 512 维/2 层)**:launch-only 路径 graph 重放比 eager 每 token 快
    ~2-5x(本次实测 48.5 vs 94.4 us/token);完整步(含 logits 读回)~1.15x。
  - 踩坑记录:hipblas 列主序 leading dimension、kernel 指针参数传法、
    动态共享内存大小、graph capture 的 stream 所有权——均已修复并有回归测试。

- **P1b 真实权重加载完成(2026-08-22,7900 XTX 真机验证)**:
  - `loader.rs`:纯 Rust safetensors 解析(F32/F16/BF16→f32)+ Llama/Qwen 权重名映射
    (embed/q/k/v/o_proj/gate/up/down + RMSNorm),支持 `tie_word_embeddings`;
  - 模型新增 **RoPE**(真实 Llama 必需,GPU kernel + CPU 参考同步实现)与
    **intermediate_size**(真实 MLP 宽度,不再硬编码 = d_model);
  - **真实模型验证**:加载 `hf-internal-testing/tiny-random-LlamaForCausalLM`
    (4.1MB safetensors),GPU 解码 5 token,与独立 **fp64 Python 参考**
    (`tools/ref_llama.py`)逐元素对比,**相对误差 1.8e-7**(此前 5.9% 的"误差"
    是 Python 参考的 bug——MLP 宽度写错,已修);Rust GPU==Rust CPU f32==f64
    全部自洽;
  - 新增测试:合成 safetensors roundtrip(CPU)、加载权重 GPU 解码对拍 CPU(真机)、
    真实模型有限/确定性 smoke(真机,模型缺失则跳过);
  - 验证:默认 16 + HIP 23 测试全绿,clippy(default+hip)0 告警,fmt 干净。

- **P1c 真实模型验证完成(2026-08-22,7900 XTX 真机)**:
  - 加载真实 **Qwen2.5-0.5B-Instruct**(942MB BF16 safetensors,24 层/896 维/GQA 14:2/
    intermediate 4864/rope_theta 1e6/tie_embeddings),GPU 解码并与**独立 fp64 numpy 参考**
    逐元素对拍,**相对误差 4.4e-6**;
  - 过程中发现并修复一个真 bug:**`embed_gather` kernel 的 grid 固定 [1,1,1]**(只有 256 线程),
    当 d_model > 256(如 Qwen 896)时只写前 256 维,其余是未初始化内存 → 整模型数值错乱、
    且随运行变化(曾误判为 1.949x/0.0028 等假象)。修复为 `ceil(cols/256)` 分块;
    `kv_store` 同样改为 grid-stride 防御;
  - **基准(Qwen2.5-0.5B,7900 XTX)**:launch-only 路径 graph 重放 **0.30-0.47 ms/token**
    vs eager 1.06-1.59 ms/token(**2.3-5.3x**);完整步(含 607KB logits 读回)~7.2-9.8 ms;
  - 新增回归测试:`gpu_matches_cpu_reference_large_dmodel`(d_model=512>256,直接抓 embed grid bug);
  - 工具:`tools/ref_llama.py`(numpy fp64 参考,支持任意 Llama/Qwen config)、
    `examples/qwen_bench.rs`、`examples/kernel_probe.rs`(逐 kernel 隔离验证)、
    `examples/ref_cpu.rs`(Rust f32/f64 参考);
  - 验证:默认 16 + HIP 24 测试全绿,clippy(default+hip)0 告警,fmt 干净。

- **P1 收尾状态(2026-08-22)**:TPOT 与 TokenSpeed 同场景对标所需的运行环境暂不可用——
  WSL2 Ubuntu-24.04 反复启动失败(0x800705aa 系统资源不足),且 WSL 内未装 ROCm/PyTorch;
  已产出 `docs/benchmark-protocol.md`(对标协议 + 模板 + MachServe 侧实测数据),
  环境就绪后按协议填 TokenSpeed 数字即可完成对标。

- **Windows 原生对标可行性实测(2026-08-22)**:
  - MachServe 全程跑在 Windows 原生(ROCm 6.2 HIP SDK),从未依赖 WSL;
  - **TokenSpeed 无法在 Windows 原生运行**(已核实):(a) PyTorch ROCm 官方 wheel
    (rocm6.2/6.3)只有 manylinux,无 win_amd64;(b) TokenSpeed 整个 AMD kernel 栈依赖
    `tokenspeed-triton`(Triton fork),其 PyPI wheel 全部 manylinux,无 Windows 版;
  - llama.cpp HIP 构建实测:Windows+ROCm 6.2 下踩到 perl 缺失(已装)、HIP_PLATFORM、
    MSVC 误编 .cu、ROCm 6.2 缺 `__hip_fp8_e4m3`(已补 fnuz 别名)等,最终 HIP kernel
    未正确链入(ggml-hip.lib 仅 4KB)——Windows HIP 构建的已知痛点;
  - 可选替代:llama.cpp **Vulkan 后端**(Windows 上成熟,7900 XTX 驱动好),但需先装
    Vulkan SDK;未获用户确认前不擅自安装。

- **P1 对标重定义 + Windows 原生 GPU 基线完成(2026-08-22)**:
  - 按用户定位,**对标物从 TokenSpeed 改为本机可运行的 llama.cpp(Vulkan 后端)**;
  - 在 Windows 原生构建 llama.cpp(ggml 2100e59, Vulkan SDK 1.4.357),下载
    Qwen2.5-0.5B-Instruct Q8_0 GGUF,`llama-bench` 实测 **decode 643 tok/s(1.55 ms/tok)**;
  - **MachServe(HIP graph 重放)GPU 解码计算 0.30-0.47 ms/tok,约为 llama.cpp 的 2-5x**;
    但端到端 TPOT(7-10 ms/tok)落后——瓶颈是每 token 全量读回 151936 个 logits(607KB),
    llama.cpp 在 GPU 侧采样;已列为下一步优化(GPU 侧 sampling kernel);
  - 完整对比表见 `docs/benchmark-protocol.md`;Windows HIP 构建 llama.cpp 的坑
    (MSVC-vs-hipcc/空 HIP lib)已记录,不重复踩。

- **P2b 批量 decode 完成(2026-08-22,7900 XTX 真机)**:
  - `batched.rs`:`BatchedModel`——B 序列共享一次前向:批量投影 GEMM(m=B,替换 m=1,
    hipBLAS `OP_T/OP_N` 形式,任意 in/out 维度都满足 leading-dim 约束)、批量
    embed/rope(每序列独立 pos)/kv_store/attention(逐序列 mask)/批量 argmax 采样;
  - **正确性**:batched step 与逐序列单模型逐 token 一致(3 步 × 4 序列)、序列间独立、确定性;
  - **性能(关键)**:批量缩放曲线——step 时间几乎与 B 无关(~12-15ms),每序列 TPOT 按 1/B 线性下降:
    B=1: 11.9ms | B=8: 1.60ms | B=16: 0.84ms | B=32: 0.47ms | **B=64: 0.216ms(4640 tok/s)**;
    对比 llama.cpp Vulkan 1.55ms(643 tok/s):**B=64 时快 7.2x**;
  - 这证实 P2 的判断:瓶颈是 m=1 GEMM,批量(m=B)直接修复,且连续批处理本身就是引擎模型。

- **P2c 连续批处理引擎完成(2026-08-22,7900 XTX 真机)**:
  - `continuous.rs`:`ContinuousModel`——序列生命周期管理:add(带 prompt)/step(每步推进
    prefill 或 decode 一个 token)/finish(EOS 或 max_new);**prefill 与 decode 在同一 batch
    step 中混合**(生产连续批处理语义);稳定 `SeqId` 与 KV 槽位解耦,序列完成后槽位压缩复用;
  - `BatchedModel` 增加 `decode_step_explicit`(显式 lens/活跃数)+ `copy_seq_kv`(槽位搬运);
    修复 `sample()` 按实际活跃数返回(此前按容量返回,导致引擎在活跃<容量时出错);
  - **正确性(真机)**:引擎生成 == 单序列贪婪参考逐 token 一致(多 prompt)、序列完成释放
    槽位且新序列 KV 隔离、EOS 精确停止;
  - 测试:默认 + HIP 31 全绿(新增 continuous 3、batched 2),clippy/default+hip 0 告警。

- **P2d OpenAI 兼容服务器完成(2026-08-22,7900 XTX 真机)**:
  - `mach-server` crate(axum + tokio):后台引擎线程(持有 ContinuousModel,仅该线程碰 GPU)+
    channel 通信;`ServerEngine::submit` 提交请求,引擎循环按容量 admit + step + 完成回传;
  - 路由:`GET /healthz`、`POST /v1/completions`、`POST /v1/chat/completions`;
    prompt 接受 token id 数组或 ASCII 字符串(朴素 byte→token 映射,真 tokenizer 留待后续);
  - **端到端(真机, Qwen2.5-0.5B)**:`/healthz` ok;`/v1/completions` 生成 8 个真实 BPE token;
    2 个并发请求同时完成(连续批处理);chat 接口正常;
  - 集成测试:HTTP 响应 == 直接引擎调用逐 token 一致、healthz + text prompt;
  - 测试:默认 + HIP 33 全绿(新增 server 2),clippy(default+hip)0 告警,fmt 干净。

- **P3a GPU 采样完成(2026-08-22,7900 XTX 真机)**:
  - `sampling.rs` 新增 **`BatchedSampler`**:单个 HIP kernel(每序列一行一个 block)实现
    temperature 缩放 + top-k(k-th largest logit 阈值,二分搜索,含边界并列)+ top-p
    (累计概率阈值,边界整层纳入)+ 单次均匀抽样;**确定性 RNG = SplitMix64**,每序列
    每步恰好推进一次 seed,host 侧保有权威 seed(CPU 参考可精确复现同一 draw);
  - **`SamplingParams`**(temperature/top_k/top_p/seed,默认 greedy)贯穿全链路:
    `BatchedModel::decode_step_explicit` 按序列采样 → `ContinuousModel::add/step`
    维护每序列参数与 seed 推进 → `/v1/completions`、`/v1/chat/completions` 透传
    `temperature`/`top_p`/`top_k`/`seed`(OpenAI 形状,缺省 greedy);
  - **正确性(真机)**:GPU vs CPU 同种子逐 token 一致(峰态分布)、greedy==argmax、
    同 seed 确定、不同 seed 发散、真实 tiny-llama 采样冒烟(同 seed 跨引擎复现)、
    HTTP 采样请求 == 直接引擎同参数同 seed 输出;
  - 测试:默认 + HIP 40 全绿(新增 sampling 4、continuous 1、server 1、real_model 1),
    clippy(default+hip)0 告警,fmt 干净。

- **P3b fp16 计算完成(2026-08-22,7900 XTX 真机)**:
  - `ModelDType::{F32, F16}` 贯穿 Config/BatchedModel/GpuModel;权重与 GEMM 输入 fp16、
    **fp32 累加**(hipBLAS `GemmEx_v2`,A/B=16F、C=32F 或 16F,compute=32F);
  - **关键性能发现**:rocBLAS 瘦长形状(m>>n)fp16 GEMM 输出 fp16 比 fp32 快 3-4x,
    而输出 fp32 反而比 f32 慢——隐藏层 GEMM 输出 fp16 + cast 回 f32,lm_head 保持 fp32
    (采样精度);第一版(c32 输出)在 B=64 无提升,改 c16 后拿到 2.4x;
  - **性能(Qwen2.5-0.5B)**:B=16/32/64 每步 4.84/4.45/5.78ms vs f32 14.18/14.84/13.78ms
    (2.9x/3.3x/2.4x);B=64 每序列 0.090ms = **11074 tok/s**,llama.cpp Vulkan 643 tok/s
    的 **17x**;
  - **正确性(真机)**:fp16 vs fp32 最大 logit 差 5e-5(tiny-llama 真实权重)/2e-3(随机),
    贪心 argmax 一致;新增 fp16 集成测试(single/batched/真实模型)+ hipblas GemmEx 探针;
  - 服务默认 dtype=f16(`MACH_DTYPE=f32` 关闭);测试默认 + HIP 全绿,clippy 0 告警。

- **P3c 真实 tokenizer + SSE 流式完成(2026-08-22,7900 XTX 真机)**:
  - `tokenizer.rs`:纯 Rust 字节级 BPE(Qwen2.5/Llama 风格)——NFC 归一化 + GPT-2 预切分
    正则(fancy-regex)+ ByteLevel(byte↔unicode)+ BPE merge + added/special tokens;
    解析 HuggingFace `tokenizer.json`;**与 HF tokenizers 0.23.1 golden 逐 token 对拍一致**
    (16 组文本 + decode + special token,`tests/data/tok_golden.json` 由
    `tools/_gen_golden.py` 生成);round-trip 中文/emoji/重音全过;
  - `mach-server`:prompt 文本用真实 tokenizer 编码、生成结果用真实 tokenizer 解码
    (naive byte 映射仅作无 tokenizer 时的回退);`/v1/completions`、`/v1/chat/completions`
    支持 `stream: true` → **SSE**(`text/event-stream`,逐 token delta + `[DONE]`);
    增量 UTF-8 解码保证多字节字符跨 token 不分裂;引擎逐 token 推送(跳过 prefill 预测,
    完成时先推最后 token 再关闭流);
  - **真机端到端(Qwen2.5-0.5B + fp16 + 真实 tokenizer)**:SSE 流式返回中文/多语言
    delta 正常拼接,`finish_reason` + `[DONE]` 收尾;非流式文本解码正确;
  - 测试:默认 + HIP 全绿(新增 tokenizer 2、server 2),clippy 0 告警,fmt 干净。

- **P3d lm_head c16 + cast 完成(2026-08-22,7900 XTX 真机)**:
  - lm_head 改用 fp16 输出(c16)+ cast 回 fp32(采样仍需 fp32 logits):rocBLAS 在
    m=151936 瘦长形状 c16 比 c32 快 **3.35x**(0.43 vs 1.43ms);直接复用隐藏层
    `gemm_batched_f16` 路径,`_logits` 变体删除,`yh` 缓冲扩到 batch×vocab;
  - **性能(Qwen2.5-0.5B fp16)**:B=64 每步 5.78 → **4.97ms**,每序列 0.078ms =
    **12885 tok/s**,较上一轮 +16%,是 llama.cpp Vulkan(643 tok/s)的 **20x**;
  - **自研 GEMM 调研(暂缓)**:共享内存分块 fp16 GEMM(向量化 float4 加载)在关键
    形状上仍比 rocBLAS 慢 2-3x(标量 fp16→fp32 累加无法打包成 RDNA3 的 packed FMA,
    需 `v_dot2_f32_f16` 类内建指令);rocBLAS solution-index 调优 API 在 ROCm 6.2
    Windows 未导出;已记录,留待后续内核专项。

- **P3e 分块 prefill 完成(2026-08-22,7900 XTX 真机)**:
  - `ContinuousModel::step` 改为**分块 prefill**:每步消费最多 `capacity` 个 prompt tokens
    (按序列顺序填满行预算,prefill 与 decode 混合在同一 batched forward),彻底摆脱
    "每步每序列 1 个 token"的逐 token prefill(TTFT 灾难);
  - **关键修复**:batched KV 写入/attention 原假定"行号==槽位";分块 prefill 下同一
    序列多行须写同一 KV 槽位,给 `kv_store_batched`/`attn_decode_batched` 增加
    `slots` 数组(每行→槽位),`decode_step_explicit` 透传;
  - **正确性(真机)**:分块 prefill 贪心输出与单 token prefill 逐 token 一致(长 prompt),
    80-token prompt 在 capacity=16 下 <20 步完成;server 流式在引擎简化(去掉
    prefill_left,step 只返回真实 token)后 SSE 与非流式仍精确一致;
  - **性能(Qwen2.5-0.5B fp16, capacity 64)**:512-token prompt **TTFT 61.7ms(8 步)**,
    prefill **8292 tok/s**,较单 token prefill(~3.6s)**~58x**;
  - 测试:默认 + HIP 全绿(新增 continuous 2),clippy 0 告警,fmt 干净。

- **P3f fp16 KV cache + attention 完成(2026-08-22,7900 XTX 真机)**:
  - KV cache 从 f32 改为 **fp16**(最后一块 f32 大缓冲):新增 `kv_store_batched_f16`
    (存 f32→fp16)与 `attn_decode_batched_f16`(读 fp16 K/V,f32 q/输出)两个 kernel,
    `kv_cache` 按 dtype 选择元素宽度;rope/GEMM/采样不动;
  - **效果**:KV 内存减半(B=512 的 KV 12.8GB→6.4GB,总显存 ~10GB 可容纳);attention
    读 KV 带宽减半(长 context 直接受益);
  - **性能(Qwen2.5-0.5B fp16)**:fp16 KV 释放内存后 **B=512 可跑**:step 19.56ms、
    每序列 0.038ms、**26173 tok/s ≈ llama.cpp Vulkan(643 tok/s)的 41x**;
    B=128/256:0.057/0.044 ms/seq-tok(17692/22747 tok/s);
  - **正确性(真机)**:batched fp16 vs fp32 最大 logit 差 0.0024(fp16 KV 舍入,仍极小),
    连续/采样/分块 prefill 测试全绿;
  - 测试:默认 + HIP 全绿,clippy 0 告警,fmt 干净。

- **P3i flash-prefill attention 实验(2026-08-22,结论:暂缓)**:
  - 实现共享 KV 的 `attn_prefill_f16` kernel(每 key 位置读一次、因果掩码复用给 C 行,
    warp shuffle 归约 + 两遍 softmax),正确性通过(与单 token 参考一致);
  - **结论:naive 版本比 decode attention 慢 5.7x**(2048-token prefill 553→3186ms):
    每 chunk 只有 14 个 block(64 行 × 14 heads),GPU 96 CU 只用 15% 占用;且每 key 的
    f16_to_f32(每元素 ~8 指令)与 3 次 warp shuffle 开销大;行分块会牺牲 KV 共享——
    占用与共享存在本质权衡;
  - **当前处置**:run 检测禁用(所有行走 decode attention,TTFT 回到 552ms);
    正确 kernel + run 基础设施 + run_mask 保留供后续"占用感知"重设计;新增 F16
    分块 prefill 正确性测试(`fp16_prefill_attention_matches_single_token`);
  - 长期 target:长 prompt/长 context 的 attention 仍是下一个优化方向,需要
    多 block 共享 KV + 减少每 key 开销的 flash 式设计。

- **P3j decode attention 优化 + 长 context 基准(2026-08-22/23,7900 XTX 真机)**:
  - `attn_decode_batched_f16` 加权求和重写:softmax 指数一次性缓存到共享(`sexp`,
    消除 64x 冗余 `__expf`)+ 全部 256 线程参与(blockDim/head_dim=4 线程/输出维,
    分区键后共享归约),不再只有 head_dim=64 个线程串行;
  - **长 context 实测**:Qwen2.5-0.5B fp16、capacity 64、context 2048 时
    **decode 18.78 ms/token**——分析为 **KV 带宽受限**(896 blocks × 2048 keys ×
    64 dims × 2B × 2(K+V)× 24 层 ≈ 11.3GB/步 @ ~600GB/s);exp 缓存/线程并行是
    计算侧优化,墙钟受内存带宽主导;
  - 正确性:fp16/连续/real_model 全绿,`chat_check` 仍正确回答;短 context decode
    无回归(B=512 19ms、B=64 ~5ms);
  - 结论:长 context decode 瓶颈是 KV 读取带宽,需内存访问模式优化的 attention
    (合并 K/V 读、GQA 复用);已留作后续专项。

- **P3j 修正:decode attention 分区加权求和(2026-08-23,7900 XTX 真机)**:
  - 前次 P3j 的 kernel 改动因脚本中断未落盘(只改了 launch 共享字节,2× 浪费且压
    context 能力);本次真正实现:加权求和改为 **256 线程分区**(blockDim/head_dim
    线程/输出维,每 key 的 exp 只算一次),共享字节回落 `max_seq+256`(context 上限
    恢复 ~16k);
  - **踩坑并修复**:初版分区归约假设 `per=4`(head_dim=64),tiny 模型 head_dim=32 →
    per=8 只加了 4 个部分和,漏掉一半 V 贡献 → F16 多行 forward logits 错 2.05
    (F32/F16 对拍发现);归约改为遍历全部 `per` 部分和后,F32-BATCHED vs
    F16-BATCHED 回到 0.0019(fp16 舍入),argmax 一致;
  - **性能(修复后,Qwen2.5-0.5B fp16, capacity 64)**:2048-token prefill TTFT
    **373ms(-33%)**、长 context(2048)decode **8.27 ms/token(-56%)**——此前误判为
    "带宽受限中性",实际旧 64 线程加权循环是并行度受限;
  - 正确性:fp16/continuous/real_model 全绿,`chat_check` 正确回答;clippy/fmt 干净。

- **P3k fp16 转换改硬件原生指令(2026-08-23,7900 XTX 真机)**:
  - 手写分支式 `f16_to_f32`/`f32_to_f16`(每元素 ~8-10 指令)替换为 **`_Float16`
    原生 cvt 指令 + union 位重解释**(每元素 1-2 指令),覆盖 attention/KV store/
    cast/embed 全部 fp16 kernel;
  - **踩坑**:初版直接 `(float)((_Float16)u16)`/`(unsigned short)((_Float16)f32)`
    是**数值转换不是位重解释**(u16 0x3C00→15360.0),logits 错 1.6——改为 union
    位重解释后回到 0.0019;
  - **性能(Qwen2.5-0.5B fp16,7900 XTX)**:B=512 **35251 tok/s(+35%,= llama.cpp
    Vulkan 的 55x)**;B=64/128/256:13070/21066/29273 tok/s;2048-token prefill
    TTFT 325ms;长 context(2048)decode **6.99 ms/token**(自 P3j 起 18.78→6.99,
    累计 -63%);
  - 正确性:fp16/continuous/real_model 全绿,clippy/fmt 干净。

- **P3m OpenAI `stop` 序列 + 完成原因(2026-08-23,7900 XTX 真机)**:
  - `/v1/completions`、`/v1/chat/completions` 支持 OpenAI `stop`(字符串或数组):
    tokenizer 编码成 token 序列,引擎在生成落到任一 stop 序列时终止(等价 EOS);
  - **finish_reason 区分** `stop`/`length`(此前恒为 length):引擎记录每序列是否
    因 EOS/stop 结束,服务器响应与 SSE 末块透传;
  - 确定性单测:stop=[首个贪心 token] 立即终止、两 token stop 序列精确停止、
    max_new 结束报 length;默认 + HIP 全绿,clippy/fmt 干净。

- **P3n OpenAI `n` 多生成(2026-08-23,7900 XTX 真机)**:
  - `/v1/completions`、`/v1/chat/completions` 支持 `n`(默认 1):n>1 时每个 choice
    用不同 seed 提交独立序列(引擎连续批处理并发跑),响应返回 n 个 choice;
    n=1 行为不变(透传调用方 seed);
  - 服务器测试:n=2 返回 2 个独立 sample 且不同;默认 + HIP 全绿,clippy/fmt 干净。

- **P3o OpenAI `logprobs`(2026-08-23,7900 XTX 真机)**:
  - 采样 kernel 在选中 token 时同步输出其 **log 概率**(`(logit-max)*inv_t - log(total)`),
    贪心为 0;`sample_batched` 返回 (tokens, logprobs);引擎按序列累计
    `generated_logprobs(id)`;
  - `/v1/completions`、`/v1/chat/completions` 支持 `logprobs: true`,响应
    `choices[].logprobs` 返回 `tokens` + `token_logprobs`;
  - 测试:采样层(贪心=0、采样有限且<=0)+ 服务器(返回字段);默认 + HIP 全绿,
    clippy/fmt 干净。

- **P3p OpenAI presence/frequency penalty(2026-08-23,7900 XTX 真机)**:
  - `SamplingParams` 增加 `presence_penalty`/`frequency_penalty`;采样 kernel 在
    softmax 前**预扫描惩罚列表就地改 logits**(host 维护每序列 token 计数,每步上传
    (token,count) 对);CPU 参考 `sample_cpu` 同步实现(可对拍);
  - 引擎按序列累计计数(prefill 不惩罚,decode 起生效),`/v1/completions`、
    `/v1/chat/completions` 解析透传;
  - **对拍**:GPU vs CPU 同 seed 逐 token 一致(含贪心+采样);默认 + HIP 全绿,
    clippy/fmt 干净。

- **P3q OpenAI logit_bias(2026-08-23,7900 XTX 真机)**:
  - kernel 增加 bias 预扫描(就地 `row[tok] += bias`),`sample_batched`/
    `decode_step_explicit`/引擎按行传递静态 (token, bias) 列表;API 解析 OpenAI
    `logit_bias` 对象({token_id: bias});
  - CPU 参考同步(penalty+bias 统一走"改 logits 副本"路径);**GPU vs CPU 同 seed
    逐 token 对拍一致**(含贪心+采样);
  - 至此 OpenAI 采样参数面完整:temperature/top_p/top_k/seed/presence/frequency/
    logit_bias/stop/n/logprobs/stream/finish_reason;默认 + HIP 全绿,clippy/fmt 干净。

- **P3r OpenAI 错误 JSON + 优雅停机(2026-08-23)**:
  - `/v1/completions`、`/v1/chat/completions` 4 处 `503` 空响应改为 **OpenAI 风格
    错误体** `{"error": {"message", "type", "code"}}`:`engine_busy`(容量满)、
    `engine_shutting_down`(停机中拒绝新请求)、`model_error`;
  - **优雅停机**:`ServerEngine` 增加 `shutdown: AtomicBool` + `ShuttingDown` 错误;
    引擎 `run()` 循环收到停机信号后排空**排队 + 在跑**序列再退出(空闲等待用
    condvar `notify_all` 唤醒,不轮询);`main()` 用 `tokio::signal::ctrl_c()` +
    axum `with_graceful_shutdown`,Ctrl-C 后置停机标志 → 引擎排空 → join 引擎线程
    → 打印退出;
  - 测试:busy 引擎返回 OpenAI 错误 JSON 形状(零容量直接拒,无需 GPU)+ 停机后两个
    在跑请求排空完成且引擎线程自行退出;默认 + HIP 全绿,clippy/fmt 干净。

- **P3s OpenAI `top_logprobs`(2026-08-23,7900 XTX 真机)**:
  - 新 GPU 内核 `topk_batched`:每行一个 block(256 线程),max-subtracted softmax +
    per-thread 局部 top-k(≤20)+ 动态共享内存合并,输出每 token 的 top-k
    (token, logprob);`logprob = (logit - max) * inv_t - log(total)`,并列按较小
    token id 排序;`t <= 0`(贪心)按 `inv_t = 1.0` 报告;读 penalty/bias 就地修改
    后的 logits,与采样分布一致;
  - `SamplingParams.top_logprobs`(0 关,≤20)、`sample_batched` 第三返回值
    `SampleOutput`、引擎按序列累计 `generated_top_logprobs`;`/v1/completions`、
    `/v1/chat/completions` 支持 `top_logprobs: n`(需 `logprobs: true`),响应
    `logprobs.top_logprobs` 每个生成位置返回按 logprob 降序的 top-n;
  - **对拍**:GPU vs CPU `topk_cpu` 逐位置 token 完全一致、logprob 差 < 1e-3;
    服务器端到端测试(每位置 n 项、降序、无 logprobs 时忽略);默认 + HIP 全绿,
    clippy/fmt 干净;`chat_check` 真实模型输出不变。

- **P3t OpenAI 请求校验 400(2026-08-23)**:
  - `/v1/completions`、`/v1/chat/completions` 参数校验:非法请求返回 **400
    `invalid_request_error`** JSON(`{"error": {message, type: invalid_request_error,
    code: invalid_request}}`),而不是进入引擎:max_tokens=0、top_logprobs>20、n=0
    或 n>128;`err_response` 泛化为支持任意状态码(503 引擎错误不变);
  - 测试:无需 GPU(校验先于 submit,零容量引擎直接返回 400)的 3 例形状断言;
    默认 + HIP 全绿,clippy/fmt 干净。

- **P3u 尝试 GQA 复用 decode attention(2026-08-23,已回退)**:
  - 目标:长 context decode(KV 带宽受限)按 roadmap P3j 结论做 GQA 复用——block 改为
    每 (seq, kv_head),组内 query head 复用同一份 K/V,预期 KV 全局读流量降 groups 倍
    (Qwen 0.5B = 7x);
  - **踩坑并回退**:新内核用 online softmax + 每线程持一个输出维,把"单维乘积
    q[g][dd]·k[p][dd]"当成 score,而 score 必须是**完整 head-dim 点积**(所有输出维
    共享)——数值对拍误差 0.06→10.8 随 context 涨,旧双遍内核精确(0.000000);已
    `git checkout` 回退旧内核,全量测试恢复全绿;
  - **结论(留作专项)**:GQA 复用的真正难点是完整点积需要跨 head_dim 归约或 K 分块进
    共享(输出维并行只适用于 V 累加),需 flash-decoding 式分块设计,不在一轮里仓促做;
    经验已沉淀 `~/.agents/rules/LESSON_GPU注意力GQA复用内核softmax分数须完整点积.md`。

- **P3v 更大模型加载验证 Qwen2.5-1.5B(2026-08-23,7900 XTX 真机)**:
  - 下载 Qwen2.5-1.5B-Instruct(2.94GB safetensors);`chat_check`/`qwen_bench` 增加
    MACH_MODEL/MACH_CONFIG/MACH_DTYPE 环境变量(默认仍 0.5B),max_seq 改从 config
    读取(封顶 8192);
  - **验证通过**:1.5B(hidden 1536 / 28 层 / 12:2 heads / head_dim=128 / vocab 151936)
    **F32 与 F16 两条路径都端到端跑通**,同一问题回答 "Paris"(0.5B 为
    "The capital of France's Paris");head_dim=128 的 attention/GEMM、28 层加载无回归;
  - **部分基准**(qwen_bench 1.5B f16,用户中断未跑完):单序列 TPOT 24.3 ms/tok;
    批量 scaling batch 1→16:31→492 tok/s(近线性);**batch 32 异常升到 228 ms/step**
    (疑似该基准 f32 权重 + 大 KV 显存压力,非引擎逻辑),fp16 批量段未跑;
  - 结论:引擎对 3 倍参数、head_dim 翻倍的大模型开箱即用;完整 1.5B 基准与
    batch≥32 显存/性能排查留作后续(按用户偏好不跑超长基准)。

- **P3w 1.5B 服务器冒烟 + batch-32 分析(2026-08-23,7900 XTX 真机)**:
  - **1.5B 走完整 serving 路径全部通过**(mach-server f16/capacity 8):chat 补全
    "Paris"(finish_reason=stop)、SSE 流式逐 token + [DONE]、`top_logprobs` 每个
    位置降序 top-2、`max_tokens=0` 返回 400 invalid_request_error JSON;
  - **batch 1→16 恒定 ~32.4 ms/step 的解释**:与 0.5B 的 ~13.8 ms/step 恒定步时
    同模式——小 m 下 hipBLAS GEMM 的固定开销主导(batched decode 每层 7 个 GEMM
    × 28 层 + lm_head,步时与 batch 弱相关、与模型规模成正比),非逐批递增;
  - batch 32 跳升 228 ms/step:疑似 hipBLAS 对该 m/n/k 组合切到病态 kernel,或
    该基准 f32 权重 + 大 KV 的显存压力;非正确性 bug(f16 生产路径不受影响);
    完整排查留待后续专项(不跑超长基准)。

- **P3x GQA 复用 decode attention 内核(2026-08-23,7900 XTX 真机)**:
  - 新 `attn_decode_batched_f16_gqa`:每 (seq, kv_head) 一个 block,分块两阶段——
    阶段 A 线程按位置算**完整 head_dim 点积**(K 行每位置读一次、组内 7 个 query
    head 复用,uint4 向量化读),阶段 B 按 (输出维, lane) online softmax(V 每位置
    读一次跨 group 复用)+ 跨 lane 合并;KV 全局读流量降 groups 倍(0.5B = 7x);
  - **正确性**:CPU 精确参考逐元素对拍 hd=32/64/128、pos 1~2047 全 **0.000000**;
    fp16(F32 vs F16)/continuous/real_model 全绿;旧双遍内核已移除,唯一 f16
    attention 路径为 GQA 内核;
  - **性能(2048-token 长 context, B=64, f16, Qwen 0.5B, 真 A/B)**:旧双遍
    **34.8 ms/step → GQA+uint4 13.4 ms/step(2.6x,1840→4777 tok/s/seq)**;低于理论
    7x 是因为旧内核已靠 L2 吃到部分组间复用,且步时中还含 GEMM 等非 attention
    成本;短 context(B=64, pos<60)为 4.76 ms/step 无回归(GEMM 主导)。
  - **测量纠偏**:初版 lctx_bench 计时前误调 `reset_state()`(lens 清零),把"长
    context"测成了短 context(+15% 是伪影);修正后真实长 context A/B 见上(GQA
    2.6x)。

- **P3z OpenAI `usage` 字段(2026-08-23)**:
  - `/v1/completions`、`/v1/chat/completions` 非流式响应补齐 OpenAI **`usage`**
    (`prompt_tokens` / `completion_tokens` / `total_tokens`;prompt = 编码后长度,
    completion = 各 choice 生成 token 数之和);
  - 测试:tiny 模型端到端断言 usage 计数与 choices 对齐;服务器套件 12 个全绿,
    clippy/fmt 干净。

- **P3aa decode attention 成本分析(2026-08-23,7900 XTX 真机)**:
  - 隔离测量 `attn_decode_batched_f16_gqa`(B=64, pos=2047,f16):**单层 0.33 ms**
    → 24 层共 ~7.9 ms,占长 context decode 步(13.4 ms)的 60%;理论 KV 流量地板
    单层 ~0.067 ms(67MB @ ~1TB/s),现为地板的 ~5x;
  - 尝试 scores 布局转置(消 bank 冲突)与 `#pragma unroll`(防局部数组溢出)均无
    可测收益(0.33 ms 不变),已回退保持内核与已验证版本一致;
  - **瓶颈定位**:单 block 处理 2048 位置即需 ~0.35 ms,且 B=8(16 blocks)比 B=64
    (128 blocks)更慢——受单 block 串行工作/访存延迟主导,非并行度;
  - **下一步(留作专项)**:flash-decoding 式 **split-K**(位置分多个 block + 跨 block
    online-softmax 归并)或共享 K tile 化,目标单层 0.33 → 0.1 ms 级。

- **P3ab split-K 去风险实验(2026-08-23,7900 XTX 真机)**:
  - 临时把 GQA 内核改为 2-split(blockIdx.y 拆两个连续位置半区,每 block 半量工作、
    2x block,不做跨区归并只计时):full 0.345 ms vs half 0.345 ms,**0x 加速**;
  - **结论:split-K 被证伪**——每位置计算/字节总量不变时加 block 无收益,瓶颈不是
    单 block 串行或并行度;
  - 方向修正:瓶颈是**每位置的 compute 或 DRAM 效率**(单层 0.33ms、67MB、~190GB/s,
    约 HBM 峰值的 20%);下一步应减少每位置计算(如两遍 softmax、softmax 权重按
    (g,p) 缓存到共享避免 64x 冗余 exp/online 更新),而非拆位置;
  - 临时变体与 probe 已清理,内核保持与已验证 2.6x 版本一致。

- **P3ac attention 瓶颈归因(2026-08-23,7900 XTX 真机,决定性诊断)**:
  - 控制变量计时(B=64, pos=2047, f16, 单层):full 0.35 ms / **exp 移除 0.31 ms
    (-13%)** / **V 读移除 0.23 ms(-34%)**;
  - **结论:不是计算/exp 主导,是内存访问模式**——单层 66MB(K+V)@ ~190GB/s,仅
    HBM 峰值(~960GB/s)的 20%;K 是逐位置 128B 行的 gather,V 是每位置 64B 分块、
    position 间 1KB 步长,都是小粒度访问;
  - **方向修正(留作专项)**:合并/加宽内存访问——K 共享 tile 化(把 gather 变
    合并流式读)+ V 用更宽(128B+)读;目标单层 0.35 → 0.15 ms 级;
  - 临时变体(noexp/nov/half)与 probe 已清理,内核保持与已验证 2.6x 版本一致。

- **P3ad 内存布局假设证伪 + 剩余线索(2026-08-23,7900 XTX 真机)**:
  - 临时把 GQA 内核改为 `[slot][kv][pos][dim]` 布局(一个 block 读连续 256KB,
    probe 用新布局 KV 直接喂):old 0.337-0.345 vs new 0.333-0.340 ms,**0x**——布局/
    突发效率不是瓶颈;
  - **至此已系统性证伪**:split-K(0x)、计算/exp(-13% 非主导)、内存布局(0x);
  - **唯一有效线索**:移除 V 读 -34%(0.35→0.23 ms)——V 的**标量 2B 加载指令数**
    是真实成本(K 已 uint4 向量化,V 仍是每 (position, dim) 一条 2B 加载);
  - **下一步**:V 加载向量化(每线程读多维)或 V 共享 tile 化(需重排共享预算),
    或上 rocprof 精确定位;临时变体与 probe 已清理,内核保持 2.6x 已验证版。

- **P3ae V 加载向量化假设证伪 + attention 收敛结论(2026-08-23,7900 XTX)**:
  - 临时 v2 变体(每线程持 2 输出维,`ushort2` 4B 读,V 加载指令减半):与 GQA 数值
    完全一致(0.00e0),但性能 0.359 vs 0.346 ms(略慢)——acc2 数组增加的 local
    memory 开销抵消加载节省;**V 加载指令数假设证伪**;
  - **至此 attention 内核(~0.345 ms/层,B=64/pos 2047)全部黑盒假设证伪**:split-K
    (0x)、exp/计算(-13% 非主导)、内存布局(0x)、V 加载(0x);nov(-34%)疑似编译器
    DCE 假象而非真实信号;
  - **结论**:进一步优化需要 rocprof 级剖析或全新算法结构,黑盒实验已到边际;
    2.6x GQA 复用为已落地收益,长 context decode 34.8→13.4 ms/step;
  - v2 临时变体与 probe 已清理,内核保持 2.6x 已验证版。

- **P3ag prefill 行批量解耦 + TTFT 提速(2026-08-23,7900 XTX 真机)**:
  - BatchedModel 解耦 **行容量(rows)与 KV 槽位(slots)**:`with_rows(slots, rows)`;
    ContinuousModel/ServerEngine 加 `with_prefill_rows` / `MACH_PREFILL_ROWS`(服务器
    默认 256);连续引擎 step() 的 prefill 预算用 prefill_rows,长 prompt 每步打包更多
    提示位置、步数变少、GEMM m 更大;
  - **实测(Qwen 0.5B f16, capacity 64, 生成 8 token 的总耗时)**:
    512-token 98.5→**73.9 ms**(-25%)、2048-token 371→**244.6 ms**(-34%);
  - **正确性**:prefill_rows=256 与默认生成结果逐 token 完全一致(parity MATCH +
    committed 测试 `prefill_rows_gives_identical_output`);默认 + HIP 全量回归全绿;
  - 内存影响:行缓冲区按 rows 分配(KV 仍按 slots),0.5B rows=256 约 +250MB,可忽略。

- **P3ah prefill_rows 扫描定标(2026-08-23,7900 XTX)**:
  - 2048-token 扫描(全部 parity MATCH):rows=256 248.1ms → 512 **228.7ms**(-8%)
    → 1024 216.7ms(-5%,递减);512 为甜点(行缓冲 ~500MB 可忽略),服务器默认
    MACH_PREFILL_ROWS 从 256 提到 **512**。

- **P3ai QKV/gateup GEMM 融合尝试(已回退,教训记录)**:
  - 目标:decode 步是 launch 开销主导(B=64 短 ctx 4.76ms ≈ 168 个小 GEMM),融合
    wq/wk/wv(3→1)与 wg/wu(2→1)可省 43% launch;
  - **踩坑**:用"fused 缓冲 + 指针偏移"实现(单次 GEMM 写 `[b, nq+2nkv]`,q/k_buf/
    v_buf 指向其偏移)——**下游内核假设每个张量连续**(kv_store/attention/rope 按
    `s*nkv*hd`/`s*nq` 步长索引行),而 fused 缓冲行步长是 `nq+2nkv` → batch>1 全错
    (n=1 恰好对,连续测试 n=2 起全红);
  - 正确路径(留作专项):hipBLAS 无 strided 输出,需给 5 个下游内核(kv_store/
    attention/rope/add_bias/silu_mul)加行步长参数,或 GEMM 后加转置/拷贝(摊薄
    收益);已回退,内核保持回归全绿;
  - 经验:融合 GEMM 输出布局必须与下游内核的连续步长假设一致,否则指针别名在
    batch>1 静默出错。

- **P3aj decode 步 launch 假设验证(2026-08-23,7900 XTX)**:
  - 层数扫描(tiny F32, B=64):2/4/8 层 = 0.537/0.826/1.094 ms/step——**次线性**
    (每翻倍 1.3-1.5x,非 2x),且 tiny d=128 的 GEMM 远小于真实 0.5B(d=896/
    inter 4864);
  - **结论:decode 步非 launch 主导**(真实模型 GEMM 更大、launch 占比更小),
    QKV/gateup 融合的收益上限很低——P3ai 的融合方向正式关闭,不再投入;
  - 真实 0.5B B=64 短 ctx 4.76ms/step 主要由 GEMM 计算(hipBLAS 内核效率)主导,
    属库层,非本项目可调。

- **P3al speculative decoding(草稿-验证,2026-08-23)**:
  - 新增 `mach_model::speculative::SpeculativeDecoder`(单序列):0.5B 草稿提议 K 个
    token,目标一次 batched 前向验证(K+1 行:位置 L-1 的末 token + K 个草稿),
    **argmax 验收**——最长匹配前缀被接受、下一 token 取目标在拒绝点的 argmax,
    输出与纯贪心**逐 token 一致**;
  - **算法经 Python 仿真验证**(K×验收率全 parity=MATCH,含首 token 与草稿生成的
    两个 off-by-one 修正)+ **tiny 随机模型真机测试** `spec_decode.rs`(k=1/2/4 与
    纯贪心一致,34s 通过);
  - `spec_check.rs` 示例:0.5B 草稿 + 1.5B 目标测速(parity + speedup;可选运行,
    加载双模型约数分钟);
  - **回归**:驱动恢复后全量 HIP 套件全绿(含新 spec_decode 测试),默认套件与
    clippy/fmt 干净;1.5B 收益测量待显式运行。

- **P3am spec-decode 引擎集成设计(2026-08-23,规划)**:
  - 现状:单序列 `SpeculativeDecoder` 已落地且正确性多层验证(仿真 + tiny 全/低
    验收 + 全量回归);1.5B 收益测量工具 `spec_check` 就绪(按用户意愿再跑);
  - **集成设计(连续批处理)**:共享一个草稿模型 + 一个目标模型,每序列独立维护
    (draft_last, len) 上下文;
    1. **草稿阶段**:对所有解码中序列各生成 K 个草稿(草稿模型批量前向,K 轮,
      每轮 m=活跃序列数);
    2. **验证阶段**:拼接所有序列的 `[draft_last, c[0..K-1]]` 行(每序列 K+1 行)
      一次目标批量前向 → 每序列 pred[0..K];
    3. **验收**:每序列独立 argmax 验收(最长前缀 + 拒绝点取目标 argmax),与
      单序列算法一致,输出仍与纯贪心逐 token 一致;
    4. **KV 管理**:目标 KV 已含验证行;被拒尾部由下轮验证行覆盖(与单序列相同);
      草稿 KV 按接受前缀 + next 推进;
  - **工程要点**:capacity 槽位 × (K+1) 行可能超行容量 → 验证行按序列分块或
    扩大行容量;采样/penalty/bias 需在验证时一并应用(当前贪心路径无);EOS/
    stop 序列在验收后按接受 token 判断;
  - 验收:与现引擎输出逐 token 一致(连续批处理测试)+ 吞吐提升;工作量中等偏大,
    作为独立专项实施。

- **P3an 批量 spec-decode 实现 + 验证(2026-08-23,7900 XTX)**:
  - `SpeculativeBatch`(多序列):共享草稿/目标模型,草稿阶段每轮 m=活跃序列数批量
    前向,验证阶段一次 `n*(k+1)` 行批量前向,每序列独立 argmax 验收;目标模型用
    `with_rows(capacity, capacity*(k+1))` 满足验证行容量;
  - prefill 分块化(`prefill_chunked`,按模型行容量)解决长 prompt 超行容量;
    `BatchedModel::row_capacity()` 访问器;
  - **验证**:tiny 模型 3 序列 × k=4 批量 spec-decode 与逐序列纯贪心**逐 token 一致**
    (GPU 测试通过);全量 HIP 回归全绿;clippy/fmt 干净;
  - 引擎集成路径已打通:验证行分块/扩容量、草稿推进按位置逐轮,均为连续批处理
    集成所需的批处理原语。

- **P3ao spec-decode 生成生命周期验证(2026-08-23,7900 XTX)**:
  - 验证 `SpeculativeBatch` 走完整 serving 生命周期:prefill → spec 轮 → 每序列
    EOS/stop/max_new 终止,与纯贪心(EOS/max_new 感知)逐 token 一致;
  - 新增生命周期测试(3 序列、各自 max_new、EOS=77):GPU 通过;全量 HIP 回归全绿,
    clippy/fmt 干净;
  - 引擎集成的生命周期模式已实证(终止判定在验收 token 上进行),下一步是把
    `SpeculativeBatch` 接入 `ContinuousModel`(连续批处理 + 槽位复用)。

- **P3ap SpeculativeBatch 跳过已完成序列(2026-08-23,7900 XTX)**:
  - `SpeculativeBatch` 增加 `active` 状态 + `finish(s)`/`is_active(s)`;
    `step()` 只对活跃序列草稿/验证/推进,返回索引对齐的 `Vec<Option<Vec<u32>>>`
    (None=已完成)——连续批处理集成必需(避免对已完成序列浪费算力,支持槽位复用);
  - 生命周期测试改用 `finish()` + `active()>0` 驱动,验证跳过逻辑正确;
    全量 HIP 回归全绿,clippy/fmt 干净。

- **P3aq FP8 可行性探针(2026-08-23,7900 XTX)**:
  - 加 `gemm_ex_fp8_probe`(E4M3 量化 + `hipblasGemmEx` fp8×fp8→f32):**hipBLAS 拒绝
    fp8**(错误 "profiler already started",fp8 专属;fp16 探针同设置正常);
  - **结论:gfx1100 + Windows ROCm 6.2 下 hipBLAS 原生 fp8 GEMM 不可用**,原生 fp8
    路径需自定义 hiprtc fp8 内核(大投入、收益未验证),或等 ROCm 更新;
  - 探针测试保留为回归证据(`mach-kernel-sys` lib 测试,含 fp8 常量 30/31)。

- **P3ar SpeculativeEngine(服务引擎层,2026-08-23,7900 XTX)**:
  - 新增 `SpeculativeEngine`:包装 `SpeculativeBatch`,提供与 `ContinuousModel` 同
    形状的 API(add → generated / finish_reason / is_done / all_done),每请求
    (prompt/max_new/eos)通过 spec-decode(贪心)服务;
  - **验证**:tiny 模型,与标准 `ContinuousModel`(贪心)逐序列生成结果 + finish_reason
    完全一致(新测试);全量 HIP 回归全绿,clippy/fmt 干净;
  - 这是引擎集成的第一个落地形态;后续可接入服务器(需采样参数/penalties + 流式,
    以及收益测量确认)。

- **P3as spec-decode 服务器集成(greedy 模式,2026-08-23,7900 XTX)**:
  - `ServerEngine` 增加 spec 后端:`with_spec(capacity, k)` + `spawn_spec` +
    `run_spec`(独立路径,不触碰 continuous run);`MACH_SPEC=1`(可配 MACH_DRAFT/
    MACH_SPEC_K)启动;
  - spec 模式为 **greedy-only**:非 greedy 参数/stop/logit_bias 在 submit 时返回
    400 invalid_request(EngineError::InvalidRequest);
  - **验证**:服务器测试——greedy 请求正常生成 max_tokens、非 greedy 返回 400;
    全量 HIP 回归全绿,clippy/fmt 干净;
  - 收益(吞吐加速)仍待 1.5B `spec_check` 测量确认;模式为可选启用,不影响默认路径。

- **P3at MoE 地基(config + 权重 + CPU 参考,2026-08-23)**:
  - `Config` 增 `num_experts`/`num_experts_per_tok`(默认 0=稠密);`LayerWeights`
    增 MoE 张量(`moe_router` [ne,d]、`moe_wg/wu` [ne,inter,d]、`moe_wd` [ne,d,inter],
    稠密为空);`Weights::random` 按配置生成;
  - `ref_model`(CPU)增 MoE 前向:router softmax → top-k 专家 → 概率加权 SwiGLU 和;
  - **验证**:合成 MoE 模型——张量形状正确、CPU 前向有限且确定(新测试 moe.rs);
    全量 HIP 回归全绿,clippy/fmt 干净;
  - 后续切片:loader 读取真实 MoE 张量(`model.layers.N.mlp.experts.M.*` +
    `mlp.gate`)、GPU 路由/专家 GEMM(逐 token top-k 分组)、真实 Qwen2.5-MoE 验证。

- **P3au loader 读取真实 MoE 张量(2026-08-23)**:
  - `load_safetensors` 在 `num_experts > 0` 时读取路由 `model.layers.N.mlp.gate.weight`
    ([ne,d]) 与专家 `model.layers.N.mlp.experts.M.{gate,up,down}_proj.weight`
    (逐专家 [inter,d]/[d,inter]),拼接为与 `Weights::random` 一致的 expert-major 布局;
  - **验证**:`load_safetensors` 测试增 MoE 回归——合成 MoE checkpoint 读写往返与
    `Weights::random(moe_cfg)` 逐元素一致(路由 + 专家 + 稠密字段);全量 HIP 回归全绿,
    clippy/fmt 干净;
  - 后续切片:GPU 路由 + top-k 专家 GEMM(逐 token 分组批量)、与 ref_model CPU 对拍、
    真实 Qwen2.5-MoE 验证。

- **P3av GPU 路由 + top-k 专家 GEMM(2026-08-23,7900 XTX)**:
  - `GpuModel` MoE MLP:路由 GEMM([ne,d]) → `moe_router` 内核(softmax + top-k,
    平局取小下标,与 ref_model 完全一致,输出归一化权重)→ `moe_gather_weights`
    把选中专家权重打包进连续 scratch → gate/up 用 concat-GEMM(各 slot 共享输入 x,
    [topk*inter,d] 一次 sgemm)→ silu_mul → **down 必须逐 slot 各自 GEMM**(各 slot
    有自己的 hidden state,concat 技巧不适用;初版误用单 GEMM 只算了 slot 0,对拍
    抓到 806 级误差后修正)→ `moe_accumulate` 加权残差累加;
  - 全程无 D2H,保持 HIP graph 可捕获;MoE 走 f32 GEMM(fp16 MoE 权重留待后续);
  - **验证**:`moe.rs` 增 GPU vs CPU 对拍(合成 tiny MoE,4 专家/2 活跃,随机权重,
    max diff 满足 2e-3+2e-3*scale);主导路由(不可能翻车)对拍 max diff 2e-6;
    全量 HIP 回归全绿,clippy/fmt 干净;
  - 后续切片:fp16 MoE 权重、batched/continuous 路径的逐 token 分组批量专家 GEMM、
    真实 Qwen2.5-MoE 验证。

- **P3aw batched MoE 分组批量专家 GEMM(2026-08-23,7900 XTX)**:
  - `BatchedModel` MoE MLP:路由 GEMM([B,ne]) → `moe_router_batched`(每 token
    softmax+top-k,与 ref_model 一致)→ `moe_count_experts` 直方图 → 每层单次 D2H
    读回 counts → host 算 prefix offsets → `moe_gather_rows` 按专家打包 token 行
    (atomic 定位)→ 逐专家 `gemm_batched`(gate/up → silu_mul → down,counts 已知
    无额外同步)→ `moe_scatter_add` 加权回填 h_acc → 残差相加;
  - fp16 路径完整支持(逐专家 fp16 权重 + xh_moe/yh_moe scratch);真正的稀疏收益点
    (只算路由到的专家,而非稠密全专家);
  - **验证**:`batched.rs` 增 `batched_moe_matches_single_seq`(F32)与
    `batched_moe_f16_matches_single_seq`(F16,router 放大保证跨精度路由一致):
    与单序列 GpuModel 逐序列对拍,greedy token 一致 + logits 在容差内;全量 HIP
    回归全绿,clippy/fmt 干净;
  - 踩坑记录:MoE 分组计数器初名 `pos_dev` 与既有每序列位置缓冲冲突,rename 为
    `moe_pos_dev` 后 memset 仍误指旧 `pos_dev`,把每层注意力位置清零导致 layer≥1
    注意力全错(对拍 0.28 级误差,路由一致仍炸)→ 已修并沉淀 LESSON;
  - 后续切片:counts 读回改 GPU 侧调度(去每层 D2H)、真实 Qwen2.5-MoE 验证。

- **P3ax MoE 端到端服务验证(2026-08-23,7900 XTX)**:
  - `ContinuousModel` 直接包装 `BatchedModel`,MoE 配置自动走通连续批处理生命周期
    (prefill → decode → 采样 → 完成);新增 server 端到端测试
    `completions_endpoint_moe_matches_direct_engine`:合成 tiny MoE(4 专家/2 活跃)
    权重,HTTP `/v1/completions` 输出与直接引擎逐 token 一致;
  - **验证**:server 套件 14 个测试全绿(含新 MoE 用例);全量 HIP 回归全绿,
    clippy/fmt 干净;
  - 意义:MoE 从权重 → loader → GPU(单序列 + batched 分组 GEMM)→ 连续批处理 →
    OpenAI HTTP 服务的完整链路已闭环;
  - 后续切片:counts 读回改 GPU 侧调度(去每层 D2H)、真实 Qwen2.5-MoE 验证
    (需下载权重)。

- **P3ay 真实形状 MoE 分组对拍(2026-08-23,7900 XTX)**:
  - `batched_moe_small_config_matches_single_seq`:基于 `Config::small` 的真实形状
    (d=512, inter=2048, 8 专家/3 活跃, batch=8,vocab 收敛到 2048 提速)——token 在
    专家间充分分散,分组/打包/逐专家 GEMM 在更接近生产的形状下与单序列 GpuModel
    逐序列对拍(greedy token + logits 容差);
  - **验证**:batched MoE 三个对拍测试(F32 tiny / F16 tiny / F32 真实形状)全绿;
    全量 HIP 回归全绿,clippy/fmt 干净;
  - 后续切片:counts 读回改 GPU 侧调度(去每层 D2H,需自定义 grouped-GEMM 内核)、
    真实 Qwen2.5-MoE 验证(需下载权重)。

- **P3az MoE 专家偏移改设备端 prefix-sum(2026-08-23,7900 XTX)**:
  - 新增 `moe_prefix_sum` 内核(单 block,ne<=256,Hillis-Steele 独占前缀和),把
    gather 偏移计算从 host 移到 GPU:count → prefix-sum → gather 全在设备端流水,
    去掉每层 host 前缀和计算与 offsets H2D 上传;counts 的 D2H 读回保留(hipBLAS
    per-expert batch 计数仍需 host 侧),改为流上异步拷贝 + 一次 sync;
  - **验证**:batched MoE 三个对拍测试(F32 tiny / F16 tiny / F32 真实形状)全绿;
    全量 HIP 回归全绿,clippy/fmt 干净;
  - 后续切片:彻底去 D2H 需自定义 grouped-GEMM 内核(设备端按专家分段调度),
    真实 Qwen2.5-MoE 验证(需下载权重)。

- **P3ba Qwen3 QK-norm(2026-08-23,7900 XTX)**:
  - `Config` 增 `qk_norm` 标志 + per-head RMSNorm 权重;`ref_model`/`GpuModel`/
    `BatchedModel` 在 Q/K 投影后、RoPE 前逐 head 做 RMSNorm;
  - 验证:`qwen3.rs` 增 QK-norm 权重形状 + GPU==CPU 对拍;全量 HIP 回归全绿,
    clippy/fmt 干净。

- **P3bb F16 路径只保留 fp16 权重驻留(2026-08-23,7900 XTX)**:
  - `BatchedModel`/`GpuModel` 的 F16 路径不再同时保留 fp32 权重副本,仅驻留 fp16
    权重以降低显存占用;
  - 验证:`qwen3.rs` 增 F16 vs F32 argmax 对拍;全量 HIP 回归全绿,clippy/fmt 干净。

- **P3bc 多 shard safetensors loader + Qwen3 真实模型 smoke(2026-08-23)**:
  - `load_safetensors` 支持多 shard checkpoint(按张量名聚合);新增 `qwen3_real`
    smoke 测试(真实 Qwen3-8B 比例 + F16,下载 ~16GB,缺 checkpoint 自动 skip);
  - 验证:`load_safetensors` 增 sharded 往返一致测试;全量 HIP 回归全绿,
    clippy/fmt 干净。

- **P3bd Qwen2-1.5B 真实模型 smoke(2026-08-23,7900 XTX)**:
  - `qwen15_real.rs` 用已下载的 Qwen2-1.5B checkpoint(BF16、tie_word_embeddings、
    GQA + rope_theta=1e6、F16 设备路径)做真实模型解码 smoke;
  - 验证:真实模型解码有限且确定(本机 checkpoint 存在,回归中实际运行);
    全量 HIP 回归全绿,clippy/fmt 干净。

- **P3ca MLA 地基(config/weights/loader/CPU 参考,2026-08-23)**:
  - `Config` 增 MLA 字段(q_lora_rank / kv_lora_rank / qk_nope_head_dim /
    qk_rope_head_dim / v_head_dim);`LayerWeights` 增低秩 Q 与压缩 KV 张量;
    loader 读 DeepSeek-V2 风格 MLA 权重;`ref_model` 增 MLA 前向;
  - 验证:`mla.rs` 增张量形状 + CPU 前向有限确定;全量 HIP 回归全绿,clippy/fmt 干净。

- **P3cb MLA 单序列 GPU decode(expanded KV,2026-08-23,7900 XTX)**:
  - `GpuModel` MLA 路径:q_lora/kv_lora 低秩投影 + RMSNorm + q_b/kv_b 展开 +
    手写 MLA assemble/attn 内核,expanded per-head KV cache;
  - 验证:`mla.rs` 增 `mla_gpu_matches_cpu_reference` 对拍;全量 HIP 回归全绿,
    clippy/fmt 干净。

- **P3cc MLA batched decode(expanded KV,2026-08-23,7900 XTX)**:
  - `BatchedModel` MLA 分支(run_kernels 内,decode-only、f32):q_lora/kv_lora
    投影 → RMSNorm → q_b/kv_b 展开 → 5 个 batched 内核(`mla_assemble_q_batched` /
    `mla_extract_kv_lora` / `mla_extract_k_rope` / `mla_assemble_kv_batched` /
    `mla_attn_decode_batched`)→ mla_o 投影;MLA KV 走独立 `mla_kv_cache`
    (expanded per-head f32);
  - 验证:`mla.rs` 增 `mla_batched_matches_cpu_reference`(2 序列逐 token 对拍);
    全量 HIP 回归全绿,clippy/fmt 干净;
  - 已知边界:仅 f32 + decode 步(ContinuousModel 槽位压缩/真实 MLA checkpoint
    尚未接入);后续切片:MLA 连续批处理/服务集成 + 真实 DeepSeek MLA 权重验证。

- **P3cd MLA 连续批处理/服务集成(2026-08-23)**:
  - `BatchedModel::copy_seq_kv` 支持 MLA expanded per-head KV cache 的槽位搬移
    (此前只搬 `kv_cache`,MLA 序列在槽位压缩后 KV 错位)——连续批处理槽位
    复用/压缩的关键修复;
  - `continuous.rs` 增 `engine_matches_single_model_mla`(连续批处理 vs 单序列
    GpuModel 逐 token 一致)与 `slots_compact_keeps_mla_sequence_intact`
    (A 先完成触发 B 槽位压缩,MLA KV 随槽位搬移后 B 输出不变);
    `server.rs` 增 `completions_endpoint_mla_matches_direct_engine`
    (MLA 配置 HTTP `/v1/completions` 与直接引擎一致);
  - 验证:本地门禁全绿(rustfmt / cargo check / clippy / CPU 测试);
    **HIP 验证已跑(2026-08-24)**:mla 4/4 + continuous 11/11 + server 15/15
    + 全量 HIP 回归全绿;
  - 后续切片:真实 DeepSeek MLA checkpoint 数值对拍(需下载权重)。

- **P1 加载安全(2026-08-24,issue #7 / PR #8)**:
  - hiprtc 编译:进程内编译缓存(key=arch+source),同一进程第 2 个模型起复用
    已编译内核(实测 continuous 套件 225s→28s);`MACH_COMPILE_PROGRESS=1`
    打印逐内核进度+总耗时;`HipKernelModule` 改 `Arc<ModuleHandle>` 引用计数,
    最后一个引用(含缓存)drop 才 unload,修复共享句柄 use-after-unload;
  - mach-server 启动预检:HIP 运行时/设备数/`hipMemGetInfo` 显存,估算=权重文件
    +KV+256MiB(spec 含草稿),不足即可读错误退出;TCP 提前 bind 快速失败;
  - `dalloc` 分配失败报字节数;部分分配失败由 Drop 释放(原已具备);
  - 验证:mla 4/4 + continuous 9/9(带缓存,多次 建/drop/重载 无悬垂);
    正向 server 起 qwen-0.5b 预检通过、负向 MACH_CAPACITY=4096 显存不足
    快速失败;全量 HIP 回归全绿;clippy/fmt 干净。

- **P3ce MLA F16 路径(2026-08-24)**:
  - MLA 分支此前 f32-only;现对齐稠密 F16 路径(P3bb 语义):`LayerDevF16` 增 6 个
    MLA 投影矩阵 fp16 权重;F16 模式不再驻留 MLA f32 副本;`run_kernels` MLA 分支
    6 个投影 GEMM 改走共享 `gemm` 闭包(F16→`gemm_batched_f16`,F32→`gemm_batched`);
  - 验证:`mla.rs` 增 `mla_batched_f16_matches_f32`(batched F16 vs F32 logits 差
    < 0.1);本地门禁全绿;**HIP 对拍已实跑(2026-08-24)**:mla 套件 5/5 通过;
  - 后续切片:真实 DeepSeek MLA checkpoint 数值对拍(需下载权重)。

- **P3cf spec-decode 收益测量(2026-08-24)**:
  - 首次 3 次运行触发系统级问题(双模型加载峰值提交内存 → os error 1453
    ERROR_QUOTA_EXCEEDED / 终端退出);PR #13 轻量化 `spec_check`(基准后
    drop 参考引擎、建完双引擎后 drop ~8GB 主机权重、`MACH_MAX_NEW`)后
    峰值内存降 ~10GB,测量成功完成;
  - **实测(0.5B 草稿→1.5B 目标,K=4,greedy,单序列)**:plain greedy
    10.8 ms/tok vs spec-decode 37.6 ms/tok → **speedup 0.29x(慢 ~3.5 倍)**,
    parity=MATCH(8/30 token 两次运行一致);
  - **结论:spec-decode 当前形态为净负收益,暂停投入**。正确性已多层验证,
    但草稿+验证每轮开销盖过收益;若未来在 batched/连续批处理形态重新评估,
    需先解决草稿相对目标不够便宜的问题。

- **P3cg 真实 MoE 权重验证受阻于模型可用性(2026-08-24)**:
  - 目标模型 `Qwen/Qwen2.5-MoE-A3B`(3B,64 专家/8 活跃,Qwen2Moe 张量布局,唯一能塞进
    24GB 卡的小 MoE)经核实**在 HuggingFace 上不存在**(Qwen/QwenLM 各种变体均
    "Repository not found",搜索无官方仓库),ModelScope 亦无(record not found),
    无社区重传;网络已修复(系统代理 + 本机 HF token 可达 HF API),纯属模型下架。
  - 缓解:loader 的 Qwen2Moe 键名(`mlp.gate.weight` + `mlp.experts.{e}.*`)已由
    审查对照 vllm/candle/llama.cpp 交叉验证,真实权重验证的残余风险已大幅降低;
    `tests/moe_real.rs`(PR #14)保持 skippable,模型文件到位即可跑。
  - 若需完成:需从其他渠道获取该模型(或兼容 Qwen2Moe 布局的小模型),放入
    `.models/qwen2.5-moe-a3b/`。

- **TokenSpeed 对齐第一/二/三片(2026-08-25)**:scheduler_fsm(7 状态+12 事件) +
  kv_block_pool(LCM 块池+RAII) + prefix_cache(SHA-256 前缀哈希链,黄金向量逐位一致)
  + reuse_planner(跨请求复用准入) + prefix_kv(CPU 参考路径前缀共享,复用 logits==全算)
  + paged_scheduler(FSM 多请求调度);5 请求共享系统提示 → prompt token 复用 71%。
  剩余:GPU(batched.rs)接线 + 容量驱逐/owning-ref 索引 + 真机 A/B(见
  docs/tokenspeed-alignment.md)。

- **分页 KV GPU 接线(2026-08-26,#52-#60)**:离线 hiprtc 编译门禁(40 内核) +
  分页地基(块表/页分配/attention) + PagedRef 参考变压器(==RefModel 逐位) +
  分页内核真机对拍(7900 XTX vs 连续逐位一致) + batched.rs 分页 decode 接入
  (`with_paged_kv`,GPU 对拍通过) + 共享前缀块表构造器 + f16 分页内核 +
  cpu_engine 背压/有界记录。剩余:共享前缀页接入 decode + f16 分页接入 +
  真机 A/B(TTFT/TPOT)(见 docs/tokenspeed-alignment.md)。

- **分页 KV 服务链闭环 + 真机 A/B(2026-08-27,#78 C1-C7,7900 XTX)**:
  - C1 per-slot 块表(`set_block_table`):共享前缀物理页混叠,reuse logits == 全算
    (64-token 页,B delta-only);
  - C2 chunked prefill 分页:per-row 表偏移刷新,12/8 行两种形态 vs 逐 token 对拍;
  - C3/C4 f16 / MLA 分页接线:运行期编译 4+2 内核(44 内核离线门禁),70 步跨页对拍;
  - C5 服务链:`ContinuousModel::with_paged_prefill_rows`(物化水位门控复用 + 表搬移
    压缩 + reuse 统计)+ `MACH_PAGED=1` HTTP e2e(输出 == 直接引擎,reused==64);
  - C6 真机 A/B(paged_prefix_ab_bench,交错准入 + 热身):提示词 325→69(节省 78.8%
    == 理论值)、端到端 2.84x、后续请求 TTFT 17.4→1.3ms(13.4x)、首请求公平对齐;
  - C7 文档同步:tokenspeed-alignment 状态表/第 4 节勾选、README 性能地图、
    benchmark-results-paged-prefix.md;
  - 剩余:真实模型同口径 A/B、scheduler_fsm 接入(页池驱逐与 paged 量化接线已落地,见 #80)
    continuous.rs(见 tokenspeed-alignment.md 接入计划)。

- **分页 KV 后续(#80 P1-P4,2026-08-27,7900 XTX)**:审查(#79)延迟项兑现——
  - P1 准入哈希链单次计算(plan/register_chain,免物化重哈希);
  - P2 MLA 融合 store 内核(边展开边写页池,scratch 退役,净 -87 行);
  - P3 paged Q4/FP8 接线(with_paged_kv_rows_q4/_fp8,server spawn 分支);
  - P4 页池驱逐(整条目驱逐+引用守卫+池满重试;退休即还 pad 页与
    首写者竞争重复页);
  - 二轮审查(PR #81)修复:复用限整页(部分末页专属,杜绝并发同
    prompt 生成区覆写)、pad 回滚截断(防双释放)、free_plan_pages 用
    精确页数、驱逐后 r 重算、paged_guards 改 Err、Q4/FP8 server 分支
    真正接线(MACH_PAGED 不再静默忽略);
  - 验证:continuous 20/20(含并发部分末页、并发同 prompt 池不泄漏、
    驱逐压力)、batched 14/14、server 17/17(含 Q4 paged HTTP e2e)、
    CPU lib 5 套件、离线门禁 43 内核(R1 时实际清单即为 43,此处
    原记 44 系误记,已在 R3 条目更正)。

- **PR #81 三轮审查修复(2026-08-28,7900 XTX)**:code-review max 15 条
  findings 兑现,正确性优先——
  - 复用边界单一事实源(critical):`r` 改由 plan 的真实别名页数派生
    (cache 恰持物化页,陈旧注册项最多造成 cache miss,不可能跳过未写
    位置);`find_reusable` 整体删除,O(entries×pages×tpp) 准入扫描与
    PagedEntry.prefix/registered 一并退役(准入少一次全 prompt 拷贝);
  - 驱逐正确性:共享 hash 由最后认领者释放(驱逐时逐 hash 查兄弟认领,
    存活链恒可解析),全失效条目按死元数据清扫;retired 上限兜底改
    逐页强驱逐(活跃表别名的页保持注册、经别名表自身条目可恢复,不再
    产生永久不可达的孤儿页);
  - `set_block_table` 失败路径补 free_plan_pages(不再泄漏 plan 新页);
  - 行为回归收敛:Q4/FP8+MLA+MACH_PAGED 恢复降级服务(告警+连续,
    不再启动硬失败);MACH_TPP 校验下沉到真正消费 paged 的分支
    (SPEC/MoE-offload 忽略 MACH_PAGED 时不再被陈旧值 abort,MoE 分支
    补齐忽略告警);
  - 加固/清理:PageAllocator 加 allocated 位图(O(1) 双释放检测;
    free→realloc→再 free 属无 owner 追踪不可检测,文档如实标注)、
    plan 注释与代码矛盾消除(reused_tokens 恒等整页语义)、死 API
    register_plan/build_table_fresh/register_table 删除(收敛为
    plan→register_chain 单路)、权重 Q4/FP8 测试转换 4 份拷贝收敛为
    `From<&Weights>`;
  - 测试覆盖:驱逐+兄弟 retired 条目端到端回归(并发同 prompt→池压
    驱逐→同 prompt 复用,静默垃圾路径钉死)、retired 上限排空后恢复
    复用、paged_kv CPU 纯逻辑测试 6 组(plan 整页语义/first-writer/
    free_plan_pages 别名安全/pad 回滚/驱逐不复用/分配器双释放)、
    paged MLA 对拍 CPU 参考(tpp=8 跨页,重写内核寻址钉到参考实现);
  - 内核计数机器校验:ALL_KERNELS 计数断言入 offline_tests(实际 43,
    融合内核退役 MLA_ASSEMBLE_KV_ROWS 后 R1 条目 44 为误记);主仓
    CLAUDE.md(未跟踪文件)计数需同步为 43;
  - 验证:CPU lib 184/184(mach-model 166 + server/其余 18)、fmt/clippy
    (-D warnings)/check×2 全绿;GPU 面(7900 XTX,--test-threads 1):
    continuous 22/22(含新增驱逐兄弟/retired 上限回归)、batched 14/14、
    mla 5/5(含新增 paged MLA 对拍 CPU 参考)、fp16 6/6、decode_slice
    2/2、server 全绿(含 ignored Q4/FP8 HTTP e2e)、离线门禁 2/2
    (43 内核 + 计数断言)。

- **PR #81 四轮审查修复(R3 复审,2026-08-28)**:复审 15 条兑现——
  - 驱逐保冷(freed>0 即停):evict_one_retired 一次池压事件不再清空
    整个冷缓存(原实现一次排空全部可驱逐条目,复用经济性归零),改
    「逐条目驱逐直到一页归还」;共享 helper evict_entry_at 消除与
    force_evict_oldest 的 20 行重复;
  - 部分末页不再永久钉页:物化只注册整页哈希(PagedEntry.full_pages),
    未注册的部分页在 retire 时随其他未注册内容一并释放(原实现每个
    不同短 prompt 永久占死一个注册但永不可复用的池页);
  - paged 校验前置:MLA/TPP 兼容性判定提升到权重加载之前(量化 MLA
    或 F16-MLA + MACH_PAGED 告警降级、坏 MACH_TPP 快速失败),不再
    出现「加载数分钟后 exit(1)」;MACH_TPP 非数字值告警并回退默认
    (文档与行为一致);SPEC/MoE 忽略 MACH_PAGED 的语义不变;
  - 链克隆消除:PagedTablePlan/PagedEntry.chain 改 Arc<[String]>,
    准入与每次驱逐重试零拷贝共享;
  - 收敛与文档:ContinuousModel 7 处构造尾部收敛为 with_model 单点、
    weights.rs 转换函数移出类型定义区(修 rustdoc 错位)、
    PagedTablePlan.reused_pages 文档对齐恒等式、README 44→43 内核、
    tokenspeed-alignment 状态表驱逐落地/页级 LRU 剩余;
  - 验证:CPU lib 184/184、fmt/clippy(-D warnings)/check×2 全绿;
    GPU 面(7900 XTX,--test-threads 1):continuous 22/22、batched
    14/14、mla 5/5、fp16 6/6、decode_slice 2/2、server 31+2(含
    ignored Q4/FP8 HTTP e2e)、离线门禁 2/2(43 内核+计数断言)。

- **PR #81 五轮审查修复(R4 复审,2026-08-28)**:复审 15 条兑现——
  - **准入驱逐盲区(真缺陷)**:驱逐改逐 hash 收割(整条目 any-引用
    跳过改为「条目内逐 hash 判定」,混有活跃共享 hash 的 retired 条目
    其独占页可被释放),冷页不再被困在共享条目后面;认领计数表一次
    构建 O(n)(原每删一条 O(n) 重建,SipHash 毫秒级引擎线程停顿);
  - **FP8/Q4 From 转换逐专家化(真缺陷)**:from_weights(w, cfg) 按
    num_experts 拆分 MoE 拼接张量逐专家量化(concat 各专家独立
    scale),与 safetensors 加载器逐字节同产;原实现整张量单 scale,
    一个离群专家压扁其余专家精度,测试验证的是真实加载不会产生的
    量化形态;
  - **整前缀命中单 token 回退**:r==len 时 r-=1(原回退整页,tpp 个
    transformer forward 重算已写 KV 且 stats 少计);
  - **reused_tokens 字段删除**:恒等于 reused_pages*tpp,保留即邀请
    误用为 free 边界(字段自文档都在警告的事,直接消灭);引擎与
    build_table 从页数派生;
  - 文档/测试收敛:server paged 测试 post 闭包提取共享 helper;
    full-pages-only 规则三处镜像点互引;roadmap R1 条目计数更正
    (43,原 44 误记);
  - **遗留(记录为后续项)**:页所有权/refcount 下沉 builder 或复用
    kv_block_pool RAII(消除引擎层手工引用扫描);PageAllocator
    世代号 owner 检测(free→realloc→陈旧 free 不可检测残差);
    ALL_KERNELS 声明点宏化(成员关系机器校验,当前计数 pin 只防
    列表↔断言漂移);引擎选择 Option<usize> 构造收敛(5 处 if-let);
    ATTN_DECODE_PAGED_MLA 为有意保留的单序列 paged MLA 预留内核
    (offline-gate-only 注释,43 计数含它)。
  - 验证:CPU lib 184/184、fmt/clippy(-D warnings)/check×2 全绿;
    GPU 面(7900 XTX,--test-threads 1):continuous 22/22、batched
    14/14、mla 5/5、fp16 6/6、decode_slice 2/2、server 31+2(含
    ignored Q4/FP8 HTTP e2e)、离线门禁 2/2(43 内核+计数断言)。

- **PR #81 六轮审查修复(R5 复审,2026-08-28)**:复审 14 条,无正确性
  破坏(驱逐/认领机制、整页复用规则、plan/pad/free 路径、融合内核
  索引、r-=1 回退全部独立追迹通过)——
  - 前置门补 attention-smem 界:max_seq_len>16128 的 MACH_PAGED 配置
    告警降级(原 paged_guards 在多分钟加载后才 Err);
  - 死代码回退删除:force_evict_oldest 在收割式 evict_one_retired 下
    不可达(后者仅在空表时返回 false),删除并让 evict_one_retired
    兼任 retired 上限封顶(空表早退);
  - born-dead 条目不入册:短 prompt(<一整页)退休不再累积死元数据
    (其条目永远无法服务复用也无法释放页面);
  - concat_many:Q4Tensor/Fp8Tensor 单遍预分配拼接,from_weights 逐
    专家组合 O(ne²)→O(n);
  - **量化分页路径 CPU 对拍(规则 3)**:paged-q4/fp8 对照 dequantized
    权重的 CPU 参考,20 token 跨 2 页边界,实测 diff ~2e-3(f16 舍入
    水平,量化误差完全对消);附 f16 控制组与连续 q4 对照;修复
    harness 对有状态 RefModel 的误用(整前缀重复消费致步 1 即 0.7
    假阳性——参考须每步只喂增量 token);
  - 文档/收敛:with_paged_kv 陈旧 follow-up 表述更正(f16/MLA 已接线)、
    roadmap 符号名修正(evict_entry_at/from_weights)、server.rs
    completions 内联块全量收敛至 post_completions、paged_kv 测试重复
    断言清理;
  - 不采纳(记录理由):add() 的 plan/pad Err 防御分支保留(守护池
    尺寸不变量,删除使未来改动静默失护);build_table/plan 测试薄
    委托保留(纯委托生产路径,无独立语义);分支历史英文标题不改写
    (squash merge 落库标题为中文+issue 号);
  - 验证:CPU lib 184/184、fmt/clippy(-D warnings)/check×2 全绿;
    GPU(7900 XTX,--test-threads 1):batched 18/18(含 4 个新对拍)、
    continuous 22/22、mla 5/5、fp16 6/6、decode_slice 2/2、server
    31+2、离线门禁 2/2(43 内核+计数断言)。

- **PR #81 七轮审查修复(R6 复审,2026-08-28)**:最终轮 12 条(无正确性
  破坏;r-=1 共享写回候选经核实为前提有误——旧整页回退重写的正是被
  别名的最后一整页且更宽,严格非回归)——
  - 前置门接权威检查:BatchedModel::check_paged_support 公开(薄包
    paged_guards),main.rs 加载前直接调用并删除手抄 smem/MLA 字面量
    (消除两处同义门禁的漂移面);量化 MLA 的 pre-load 降级单独保留
    (编码 build_q4/fp8 强制 F16 的跨 crate 不变式);
  - parse_paged_tpp 拆出纯校验 validate_paged_tpp(cfg, raw)(无 env/
    exit 副作用,CPU 可测)+ 薄 exit 包装;
  - full_pages 挂上 PagedTablePlan:整页复用规则的三处镜像收敛为
    builder 单一事实源,引擎不再自行整除;
  - 驱逐簿记切片 chain[..full_pages]:永不注册的部分页哈希不再进认领
    表(死键清理,两循环失配 panic 面缩小);
  - 死代码 retain 清扫删除(认领计数不变式下永不触发);
  - spawn_spec 补 MACH_PAGED 守卫(与 spawn_q4/fp8 对齐,API 边界不再
    静默忽略);
  - concat_many 文档限定折叠等价条件(当且仅当全段对齐)+ q4/fp8
    折叠等价性质测试;校验层补测(validate_paged_tpp 5 断言、
    check_paged_support 镜像 paged_guards 四分支);
  - 不采纳(记录理由):claimed_counts 增量维护与页所有权 refcount
    下沉(同一后续架构项);step() 零 prefill 行准入(消除 r-=1 与
    state_reuse 双补丁点,连 scheduler_fsm 接入一起做);量化测试
    Q4/FP8 分支泛型去重(跨无公共 trait 的两类型,闭包体操不值);
  - 验证:CPU lib mach-model 168/mach-server bin+lib 3、fmt/clippy
    (-D warnings)/check×2 全绿;GPU(7900 XTX,--test-threads 1):
    continuous 22/22、batched 18/18、mla 5/5、fp16 6/6、decode_slice
    2/2、server 32+2、离线门禁 2/2(43 内核+计数断言)。

- **批量 MoE 解码 host 串行化消除(#70 P1/P3/P4,2026-08-29,7900 XTX)**:
  - P1 grouped GEMV 解码路径:`moe_grouped_gate_up`(f32/f16)+ `moe_grouped_down`
    (f32/f16)+ `moe_scatter_all`(确定性:row_pos 输入序映射 + 固定 k 序累加,
    无原子、跨运行可复现)+ `moe_gather_rows_tokenmajor`(token 主序,零
    调度,grouped 路径免 memsets/count/prefix)共 6 新内核,gather 增加
    `exp_of_row`/`row_pos` 输出;每 MoE 层每步不再需要 counts D2H + 全流
    sync + host 逐 expert 小 GEMM(原 m=1 hipBLAS ×3ne 次发射);
    decode_only 显式路由(调用方已知 prefill/decode,小 prefill chunk 不
    误走 GEMV),chunked-prefill 大 chunk 保留 hipBLAS(m>1 权重重用);
    `MACH_MOE_GROUPED=0` 回退开关;
  - P3 sampler 参数上传收敛:12 个 H2D memcpy 从独立分配改为按字段 n 行
    从 8 对齐打包块内拷贝(单分配对;整块单次上传曾尝试但按容量缩放,
    n≪capacity 时每步 ~34MB 反而回退,已放弃);
  - P4 router top-k 并行化:thread-0 串行 O(topk²·ne) 选择改为每 k 一轮
    256 线程并行 argmax 归约(平局取小索引,语义与串行扫描严格一致);
  - 测试:moe_grouped_gemv(GEMV 逐内核对拍 CPU,f32 结果先于 f16 覆盖
    取回)、moe_grouped_pipeline(wrapper 全流水线 vs CPU,抓出 scatter
    非原子丢更新 bug)、paged 位级对拍恢复;MoE 对拍 greedy token 改
    near-tie 兼容断言(不同 GEMM 实现的合法翻转);
  - **A/B(moe_batched_bench,d=512/2 层/64 专家/topk8/batch32/F16)**:
    host 路径 4.867 ms/step(6.6K tok/s)→ grouped 路径 **0.124 ms/step
    (258K tok/s),39x**(R4 复核,同方法论);
  - 验证:fmt/clippy(-D warnings)/check×2 全绿;GPU(7900 XTX,
    --test-threads 1)batched 21/21(含新增 CPU 参考与 hipBLAS 回退
    两个对拍 oracle + f16 回退变体)、continuous 22/22、moe 2/2、
    gpu_tests 4/4(含恢复的 paged 对拍、router 直接对拍)、lib
    183/183(+9 ignored)、离线门禁 49 内核+计数断言;
  - P2(单序列 offload/slots 每层同步)为设计取舍,维持排期;已知限制:
    mixed 步(任一 prefill 行)整步走 hipBLAS、buffered 模型的 decode 仍
    逐层等 prefetch 流;f16 激活舍入差异:grouped 内核权重 f16/激活 f32,
    hipBLAS 回退逐层把激活与结果舍入 f16(grouped 反而更准),契约已写入
    内核文档并由 f16 回退对拍(0.1 带)钉住;残余:gather/count/prefix
    的每层调度可进一步收敛为 token-major 恒等布局(已落地 token-major
    gather,count/prefix 仅 prefill 路径使用);MACH_MOE_GROUPED 在
    mach-model 内解析(其余 MACH_* 旋钮在 server main.rs,属已知差异,
    已在 main.rs 环境文档注明)。

- **批量 MoE 残余收尾(#83,2026-08-30,7900 XTX)**:
  - P1 token-major 恒等布局收敛:gate_up/down 直读 x+ids(`t = r/topk`
    取 token 行、`ids[r]` 取专家),scatter_all 内联 `j = t*topk+k` 并直读
    w;删除 MOE_GATHER_ROWS_TOKENMAJOR(离线门禁 49→48)与
    exp_of_row_dev/row_pos_dev 分配;grouped 路径每 MoE 层 5→4 发射;
    A/B(moe_batched_bench,同方法论):0.124 → **0.110 ms/step(258K→290K
    tok/s)**;
  - P2 prefill_buffered 跨步 ping-pong 竞态修复:奇数 MoE 层数时新一步
    prefetch(0) 覆盖上一步末层仍在读的槽位(步内等待链只到
    compute_ev[n-2]);begin() 增加对 compute_ev[last] 的事件等待(覆盖
    所有层数,首次调用按 HIP 语义视为已完成);移除构造时预记录;
    回归测试为**确定性事件序断言**(watch 流等待 prefetch_ev[0]:无修复
    121.9us 即过、有修复等至探针结束),并附 32 步奇数层跨步解码对拍;
    调试发现:此平台已启动内核不可见并发 H2D 拷贝写入(陈旧 L1/L2),
    内容对拍无法区分竞态,必须用事件序;
  - P3 sampler 参数上传重叠:实现上传流 + 事件链(A/B 实测中性:平台
    host 发射受限,12 次小拷贝的 GPU 时间已被后续发射的 host 时间隐藏),
    已撤销,记录为已证伪方向;
  - P4 单序列 offload/slots 每层同步事件化(#70 条目 2 兑现):
    forward_moe_offload/forward_moe_slots 的两次全流 sync 改为 xfer
    拷贝流 + 事件(router_done/gpu_part_done):ids/weights/xn2 D2H 只等
    router,CPU 专家计算与 GPU-resident GEMM 重叠;x D2H 与残差 H2D 等
    accumulate;offload 路径 gpu_n 只依赖 budget,GPU 部分提前入队;
    读回缓冲 pinned 化(hipMemcpyAsync 对非 pinned 内存回退同步拷贝,
    审查 R2 发现后修复);
    A/B(合成 d=512/2 层/16 专家/topk4):slots=8 均衡分割 2.487 →
    **2.119 ms/step(-14.8%,含 pinned 化)**;pinned 化前的中间测量
    2.271(-8.7%),重度 CPU 回退场景当时 +4%(同步开销,已随 pinned
    化消除),slots 大时中性;真机 30B 未测(加载成本,待验证);
  - 验证:fmt/clippy -D warnings/check×2 全绿;GPU(7900 XTX,
    --test-threads 1)全量套件;fp64 参考 5/5(含双放置不变性);离线
    门禁 48 内核+计数断言;

- **Q4 MoE 存储路径(#85,2026-08-30,7900 XTX)**:
  - P1 Q4-on-device 专家池 grouped GEMV:MOE_GROUPED_GATE_UP_Q4 /
    MOE_GROUPED_DOWN_Q4 读原始打包 int4 + 每 group f32 scale(Q4_GROUP=32),
    内核内反量化(精确 f32 scale 乘)——无 f16/f32 设备专家拷贝,30B 类
    检查点 Q4 池 ~16GB 可行(f16 ~50GB 超 24GB VRAM);with_rows_q4_device
    构建,Q4 模式整步走 grouped 路径;离线门禁 48→50;loader MoE 拼接改
    concat_many(O(n²)→O(n));对拍:tiny MoE Q4-on-device vs 反量化 f16
    参考 5e-2 容差;
  - P2 **Qwen3-30B-A3B 真机验证兑现 README 待验证项**:加载 201.6s
    (BF16→Q4,专家池 13.8GB)→构建 25.5s→16 步解码全有限 logits、两遍
    greedy 逐位稳定(run-to-run 确定性);190 ms/step——瓶颈为 attention
    投影 m=1 f16 GEMM(已知问题),非 MoE 路径;
  - P3 MACH_MOE_GROUPED 解析统一:Config.moe_grouped 字段(默认 true),
    库内零 MACH_* env 读取,server config_from_json 旋钮区解析;
  - 验证:fmt/clippy -D warnings/check×2 全绿;GPU(7900 XTX,
    --test-threads 1)全量套件(lib 184+9 含新 GEMV 对拍);离线门禁
    51 内核+计数断言;

- **m=1 GEMV 内核 + grouped 小批量并行重构(#87,2026-08-30,7900 XTX)**:
  - 归因修正:30B Q4-on-device 解码 190 ms/step 并非 attention m=1 GEMM
    单因素——head-to-head 实测 rocBLAS m=1 2048×2048 f16 GEMM 18.8ms vs
    自定义 GEMV 246us(76x);而 **grouped MoE 内核小批量饥饿才是主因**
    (batch=1 时 8 路由行×3 y-tile = 24 blocks×256 线程读 18.9MB,
    实测 ~5GB/s,延迟无法隐藏);
  - GEMV_F16 新内核:每 warp 处理一个(行,输出),32 lane 分摊 d +
    蝴蝶归约(固定序,确定性),共享内存缓存 x 行(d×4B,≤48KB guard),
    f32 直接输出免 f16 中转+两次 cast;GEMV_MAX_M=8 以下走 GEMV;
  - 6 个 grouped 内核(f32/f16/Q4 × gate_up/down)重构为
    block-per-(row, output) + 128 lane k-stride + 共享树归约
    (batch=1 时 786K~1.5M 线程 vs 原 6144);离线门禁 50→51;
  - A/B:**30B 真机 190 → 17.4 ms/step(10.9x,greedy 逐位一致)**;
    batch=32 合成 0.110 → 0.098 ms/step(无回归,更快);head-to-head
    76x;剩余 17ms 与 ~6ms 内存下界的差距为后续 Q4 反量化内核优化空间;
  - 验证:fmt/clippy -D warnings/check×2 全绿;GPU(7900 XTX,
    --test-threads 1)全量套件(lib 184+9 含新 GEMV 对拍);离线门禁
    51 内核+计数断言;

- **30B 分相剖析器(#89 P1,2026-08-30,7900 XTX)**:
  - MACH_STEP_PROFILE 诊断开关(Config.step_profile,server 旋钮区解析):
    batched.rs 每层三事件对(layer_start/attn_done/moe_done)+ 整步事件对
    (含输出头),decode_step 采样同步后打印分相(hip_event_elapsed_time
    新增 FFI);
  - 30B Qwen3-30B-A3B 实测(16 步稳定):**attn=8.1ms moe=7.1ms
    other=1.0ms(lm_head 622MB)总计 ~16.2ms**——两条 GEMV 路径各 ~6x 超
    内存下界:GEMV_F16 每 warp 64B/事务(半行宽)、Q4 grouped 逐元素
    反量化 scale 查找;二者均衡,单点优化上限 ~2x,合并优化(Q4 向量
    化解包 + f16 全行宽载入)目标 ~6ms;
  - 后续:rocprof 细粒度剖析 + Q4 向量化(f32 结果向量化、双 int4 解包);
  - 验证:fmt/clippy -D warnings/check×2 全绿;GPU 全量 batched 22/22;
    server 17+2;

- **GEMV_F16 全行宽载入 + Q4 字节对反量化(#91,2026-08-31,7900 XTX)**:
  - GEMV_F16:每 lane 一次处理 2 个连续 f16(4B 载入,128B 全行宽,
    替换 #87 的 64B 半行宽);
  - Q4 grouped gate_up/down:每 lane 每 iteration 处理 1 字节 = 2 元素
    (双 nibble 解包;32 元素 group = 16 整字节,字节永不跨组,scale 每
    字节一次),iterations 减半;
  - 修复:down_f16 首版 u32 索引把 o 双计(base 已含 (e*d+o)*einter)——
    对拍抓出后以标量对载入修正;
  - A/B(30B,分相):attn 8.1→6.5ms、moe 7.1→5.4ms、总 17.4 →
    **14.4 ms/step(-17%)**;batch=32 合成 0.098→0.100(噪声内);
  - 验证:fmt/clippy -D warnings/check×2 全绿;GPU 全量 batched 22/22;
  - 后续:剩余 ~2.4x 差距需 rocprof 细粒度剖析(occupancy/调度);

- **逐内核形状基准 + QKV 融合发射(#93,2026-08-31,7900 XTX)**:
  - P1 形状基准(gemv_shape_bench,30B 形状,batch=1,每迭代含 sync
    为延迟主导带宽):q GEMV 16.8MB 132μs(127GB/s)、**k/v GEMV
    2.1MB 108μs(19.5GB/s,64 blocks occupancy 饥饿)**、o 8.4MB
    116μs(72GB/s)、gate_up_q4 12MB 138μs(91GB/s)、down_q4 6MB
    150μs(42GB/s)——数据定位 QKV 融合为首选优化;
  - P2 QKV 融合发射:GEMV_F16_QKV 内核单次 launch 覆盖 q+k+v 三投影
    (按输出行选择 wq/wk/wv 权重与 q/k_buf/v_buf 输出缓冲),k/v 行获得
    q 行的 640 blocks 并行度;离线门禁 51→52;batched.rs 标准注意力
    路径小 m(f16 && b<=8)走融合,大 m 与 f32 保持 hipBLAS;修复:
    融合分支最初误置 else-if 链上跳过 rope/kv_store/attn——对拍抓出
    (2.29 diff)后重构进 else 块内;
  - A/B(30B 分相):**attn 6.5→5.55ms、总 14.4 → 13.31 ms/step
    (-8%)**;greedy 序列不变;
  - 验证:fmt/clippy -D warnings/check×2 全绿;GPU 全量 batched 22/22、
    lib 185+9(含 gemv_f16_qkv_matches_cpu_reference 对拍);离线门禁
    52 内核+计数断言;

- **30B 真实模型服务链端到端(#95,2026-08-31,7900 XTX)**:
  - **根因修复:Qwen3 共享 QK-norm**——HF checkpoint 的 q_norm/k_norm
    为共享 [head_dim] 向量,QK-norm 内核按 per-head [n_heads,head_dim]
    索引;f32 loader 已做共享->per-head 广播,**Q4/FP8 loader 缺失**,
    真实权重下 head>=1 读越界垃圾 -> 全部真实 Qwen3 推理退化
    (合成权重测试用 per-head 布局恰好掩盖)。loader 补
    broadcast_qk_norm 后 8B Q4 与 30B Q4-on-device 均生成连贯推理文本;
  - server:config_from_json 的 qk_norm 默认改由 model_type 前缀 qwen3
    推导(HF 无 use_qk_norm 键,此前被静默关闭,显式键仍覆盖);
    estimate_vram 修正 Q4 口径(服务端从 BF16 文件加载,设备 f16 =
    文件同大小,原 x4 高估 4 倍挡掉合法加载)并新增 q4_device 0.3x 档;
  - **30B Q4-on-device 服务链端到端**:mach-server(MACH_Q4=1
    MACH_Q4_DEVICE=1 MACH_CAPACITY=4)+ 真实 tokenizer + chat 模板 +
    HTTP;真实对话输出完整 Qwen3 思考链,答案正确(Paris);
  - 真实负载性能:单流 **68 tok/s**(3.74s / 256 tok,含采样与 HTTP
    开销);engine 侧 13.3 ms/step(attn 5.55/moe 5.5/other 0.75);
  - 验证:fmt/clippy -D warnings/check×2 全绿;GPU 全量 batched 22/22;
    server 13+17;qwen3_q4_real 新增真实文本生成测试;

- **双口径内核基准 + down_q4 u16-lane 优化(#97,2026-09-01,7900 XTX)**:
  - gemv_shape_bench 增加双口径:sync(launch+sync,延迟下界)与
    stream(背靠背 launch,单次尾 sync,in-stream 吞吐);
  - in-stream 实测(30B 形状,batch=1):q 423GB/s、k/v 152-154GB/s、
    o 330GB/s、gate_up_q4 232GB/s、down_q4 170GB/s——**引擎已在内核
    稳态吞吐运行**(sync 口径含 launch+排空,不代表引擎实际);
  - down_q4 优化:每 lane 2 字节(4 元素,u16 载入)替代 1 字节;
    wbase 32 对齐保证 4 元素不跨组(单 scale 查找);迭代减半;
    A/B:in-stream 37→31.2μs(170→202GB/s,-16%);30B 真机 moe
    5.5→5.2ms、总 **13.31 → 12.99 ms/step**;greedy 序列不变;
  - 验证:fmt/clippy -D warnings/check×2 全绿;GPU 全量 batched 22/22;
    离线门禁 52 内核+计数断言;

- **gate_up_q4 u16-lane 优化(#99,2026-09-01,7900 XTX)**:
  - #97 down_q4 同款:每 lane 2 字节 × 2 张量(wg+wu)= 8 元素/迭代,
    迭代减半;wbase 32 对齐保证不跨组(单 scale 查找);
  - A/B(30B 分相):**moe 5.5→4.81ms(-13%)**,总 12.99 →
    **12.72 ms/step**;greedy 序列不变;
  - 验证:fmt/clippy -D warnings/check×2 全绿;GPU 全量 batched 22/22;
    离线门禁 52 内核+计数断言;

- **内核内 clock64/globaltimer 插桩剖析 + 三项微优化(#102,2026-09-02,7900 XTX)**:
  - **背景**:RGP 实测对 hiprtc hipModuleLaunchKernel 路径 trace 即进程
    segfault(AMD 已知 ROCm/rocm-systems#395;runtime API 正常)——外部
    计数器工具不可用,转内核内插桩。工具链发现:RDS 须以
    RadeonDeveloperServiceCLI 提权运行;RDP 驱动层枚举 [0]=核显
    [1]=7900 XTX,与 HIP 层相反;readsteadycounter 单读 ~162ns(sendmsg
    序列化)只能采样,gfx1100 无 s_memrealtime/s_memtime,全局时钟须用
    `__builtin_readsteadycounter()`(sendmsg MSG_RTN_GET_REALTIME,
    实测 100MHz/10ns tick);clock64() = HW_REG_SHADER_CYCLES(32 位
    每 SIMD 周期计数,~4 cycle/读,块内 delta 有效)。
  - **插桩契约**(GEMV_F16 / GEMV_F16_QKV / MOE_GROUPED_GATE_UP_Q4 /
    MOE_GROUPED_DOWN_Q4):可空 prof 出参,每 block thread0 记
    [clock64 entry/loop_done/end] + 每 16 块采样 [globaltimer entry/end];
    prof=null 为生产路径(每 block 一次空判),不增内核(离线门禁仍 52)。
    新增 gemv_prof_bench example:稳态流中插桩一次(中途驻留防降频),
    按采样块校准 cycles→ns,报告 span/busy/块级并行度/loop 占比/块时长
    分位/尾波。
  - **首批数据**(30B 形状,batch=1):GEMV loop 占 97-98%、驻留 ~3
    块/CU(par 269-307,近 96 CU × 3 上限)、访存延迟主导(在途字节不足);
    gate_up_q4 loop 74%(reduce 26%);down_q4 loop 仅 50%(7 级
    syncthreads 树归约吃掉一半块时长)。
  - **优化 1**:Q4 双内核 warp-shfl 归约 + 一轮共享合并(替换 7 级
    syncthreads 树)——down_q4 loop 50→74%、span 58→44μs,in-stream
    201→236 GB/s;gate_up loop 74→89%。
  - **优化 2**:GEMV_F16/QKV uint2 载入(每 lane 4×f16=256B/warp/步,
    在途字节翻倍;d%4==2 行基址仅 4-mod-8 对齐,保 u32 对回退)——
    q 445→488 GB/s、k/v 152→213、o 330→425、qkv 515→565(in-stream)。
  - **优化 3**:Q4 双内核字节对改为单次 u16 载入(每张量每 lane 少 1 条
    load)——A/B 在噪声内,按"更少指令"保留。
  - **30B 真机 A/B**:attn 5.55→5.1ms、moe 4.6→4.37ms、总 12.34(当日
    master 基线)→ **10.2ms;decode 11.33 ms/step(88 tok/s,16 步)**;
    greedy 序列与 master 逐 token 一致(220 220 16 271 ... 198)。
  - 累计:190 → 11.33 ms/step = 16.8x。
  - 验证:fmt/clippy -D warnings/check×2 全绿;GPU 全量;离线门禁 52
    内核+计数断言;gemv_shape_bench 新增 MACH_BENCH_HOLD_SECS 驻留旋钮
    (RGP 复测用)。

- **服务链 HIP graph 捕获接入(#103,2026-09-02,7900 XTX)——实验开关落地**:
  - **摸底**:30B 单步 ~725 次 hiprtc 模块 launch(48 MoE 层 × 15)+ ~5
    输入上传 + 采样(12 参数上传 + kernel + sync + 2 D2H);#102 步时
    11.33ms vs 内核 prof 总和 ~10.2ms → host launch 间隙 ~1.1ms(~10%)。
  - **实现**(mach-engine/hip.rs + batched.rs,全部走 k.stream 单流):
    per-n decode graph 桶(`decode_graphs: HashMap<i32, _>`)+ 贪心
    `decode_step` 专用图(argmax 入图);捕获门控 `graph_capture_ok`
    (f16、n≤8、decode_only、无 prof/prefetch、grouped MoE、专家槽
    足量);host 每步只写 pinned 暂存(tokens/pos/slots/run_mask/表
    偏移 + 采样参数),graph 内 memcpy 节点重放时重读;采样 sync+D2H
    readback 留在图外(图边界)。`hipGraphUpload` 实例化后显式上传。
  - **正确性**:合成 GPU 测试(F16+Q4-device)graph 与 eager 逐位
    一致;30B 贪心路径 12000 次 replay(6000×2 pass)run-to-run 稳定;
    churn 复现 example(qwen3_30b_graph_churn)eager 模式 8 请求
    逐 token 一致。
  - **裸 decode harness 收益**:11.02 → 10.27 ms/step(91→97 tok/s,
    +7%),greedy 序列不变。
  - **30B 真机腐化(驱动级,未解)**:服务链 MACH_GRAPH=1 下重复请求
    数百至数千次 replay 后输出静默腐化(先"满速算错"后整体 no-op,
    ~0.9ms/步,logits 冻结交替旧值;hipGraphLaunch/sync 均返回成功、
    无 fault)。腐化与图内容无关(full/kernels-only/零 memcpy/
    uploads-only 全中)、与 hipBLAS prefill 无关(GEMV prefill 同坏)、
    hipGraphUpload 与周期重捕获均无效(重捕获还暴露 instantiate 数轮
    capture/destroy 后假性 OOM);纯单流重放 12k 次稳定。判为 ROCm
    6.2/Windows 驱动缺陷(与 rocm-systems#395 同类),用户态无解,
    待 ROCm 升级用 churn example 复测。
  - **服务链端到端零收益(前提被否定)**:kernels-only 图稳定运行时
    服务端 512-token 请求 7.55s(eager)== 7.55s(graph)——瓶颈在
    GPU 步之外的 host 簿记(~2.5ms/步),graph 只省 GPU 内 launch
    开销;graph 收益只存在于裸 decode harness。
  - **落地**:MACH_GRAPH=1 实验开关(默认关);churn example 留作
    ROCm 升级复测工具;hip_graph_upload 入 API 表。
  - 验证:fmt/clippy -D warnings/check×2 全绿;GPU 全量
    --test-threads 1;离线门禁 52 内核+计数断言。

- **真实检查点批次:混合 dense+MoE Q4-on-device + DeepSeek MLA(#104,
  2026-09-04,7900 XTX)——RoPE 配对约定根因已修,Q4 遗留另立 #107**:
  - **选模**:DeepSeek-V2-Lite-Chat 一个模型同时覆盖两项 —— layer0 为
    dense(`first_k_dense_replace=1`)+ 26 层 MoE(`n_routed_experts=64`、
    `topk=6`、`n_shared_experts=2`)= 混合检查点;真实 MLA
    (`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、
    `v_head_dim=128`);15.7B 参数,必须走 Q4-on-device 才装得进 24GB。
  - **根因:RoPE 配对约定按 checkpoint 而异(已修)**。半半配对
    (GPT-NeoX / HF `rotate_half`,坐标 `d` 配 `d + head_dim/2`)是 Llama /
    Qwen2 / Qwen3 的约定;DeepSeek 用**相邻对**(坐标 `2d` 配 `2d+1`),
    其 `apply_rotary_pos_emb` 先做 `view(d//2,2).transpose(4,3)` 置换再套
    `rotate_half`,即最终一起旋转的那一对来自**相邻坐标**(transformers 侧对应
    按 checkpoint 选择的 `apply_rotary_pos_emb_interleave`,非默认路径)。
    此前统一按半半配对实现,DeepSeek 每个 `pos>0` 的位置都被破坏。
    - **数值隔离证据**:`inv_freq` 两者一致(rel 1.0e-08,错的不是频率);
      施加 RoPE 后与 HF 的差 —— 相邻对 1.5e-07(吻合),半半配对 ~2.0
      (量级级错误)。
    - **为什么此前没抓到**:两者在 `pos == 0` 输出**完全相同**(cos=1、
      sin=0 让 RoPE 退化为恒等),同理 `pos 0` 时 T=1、softmax 退化为常数,
      softmax_scale 错也不可见。#85 的 30B 验证只查了"logits 全有限 + 两遍
      greedy 逐位稳定"(确定性),未查连贯性,且半半配对对 Qwen3 恰好正确。
    - **修复**:`Config::rope_interleave`(非 MLA 的 4 个构造函数默认 false);
      两个 HIP rope 内核加 `int interleave` 参数;ref_model / fp64_ref 的
      CPU 参考同步;`mach-server` 按 **白名单** `deepseekv2` / `deepseekv3` /
      `deepseekvlv2` 判定(该约定属于 checkpoint 自带建模代码,不是超参,
      无法从配置推导);`kernel_probe` 支持 `MACH_ROPE_INTERLEAVE=1`。
      `Config::mla()` 默认改为 `true` —— MLA 即 DeepSeek 的注意力,默认 false
      会与它要描述的真实检查点相反,且让相邻对分支在模型级零覆盖。
    - **判定必须是白名单,不能用 `starts_with("deepseek")`**:model_type 为
      `"deepseek"` 的是 DeepSeek-**V1** / DeepSeekMoE,其建模代码是 Llama 抄本
      (半半配对、无 permute),且为 dense 形状 —— 加载不报错,却在每个
      `pos > 0` 被静默写坏,且无错误路径可拦。第 2 版实现正是踩了这个坑。
    - **两个族名来源要归一化**:`model_type` 是 snake_case(`deepseek_v2`),
      `architectures[0]` 回退是 PascalCase 类名
      (`DeepseekV2ForCausalLM`);只 `to_lowercase` 得到
      `deepseekv2forcausallm`,永远匹配不上白名单 —— 缺 `model_type` 的检查点
      会静默漏掉修复。归一化为"仅保留 ASCII 字母数字 + 小写"后两者都收敛到
      `deepseekv2`。此缺陷由补写的
      `rope_interleave_falls_back_to_architectures` 当场抓出。
    - **防回归**:新增 `apply_rope_interleave_pairs_adjacent_coordinates`
      与 `rope_conventions_agree_at_pos_zero_and_differ_after`,后者双向
      钉住"pos 0 必须一致(<1e-6)、pos 5 必须不同(>1e-2)",使 pos-0 对拍
      这种无效比较无法再伪装成通过。
    - **顺带修掉两处同源缺陷**:
      - `examples/ref_cpu.rs` 的 `rope()` **硬编码相邻对**,但它自称对拍的
        `tools/ref_llama.py:62-66` 用的是半半配对(注释即 "matching HF
        rotate_half"),且该文件只处理 llama/qwen 配置 —— 于是它在**任何
        `pos > 0`** 与自己的 Python 参考静默不一致。同样因 pos 0 不可见而
        一直没暴露。已改为按 `cfg.rope_interleave` 选约定(仓库内无已提交的
        由它生成的 golden JSON,不影响既有数据)。
      - 内核侧注明"配对约定 ≠ 输出重排":旋转结果写回**读出的同一对槽位**
        (两种模式皆然),HF 的 permute 是被**消去**而非遗漏,照抄会写坏。
  - **对齐/已核对无误**(对照 numpy 独立参考实现):`mla_assemble_kv_batched`
    的 k_nope/v 切分顺序、MLA softmax scale(`192^-0.5 × mscale² =
    0.114721`)、YaRN inv_freq 用 rope 维 64、shared experts 的 SwiGLU 后
    累加、`norm_topk_prob=false` 的路由权重语义、chat 模板与 BOS
    (prompt_tokens=12 与 numpy 的 12 个 id 吻合)、Q4 分组对齐(行长
    2048/512/1408 均为 32 的倍数,无组跨越切分边界)。
  - **加载与显存**(Q4-on-device):need **10.08 GiB** / free 23.84 GiB,
    加载成功。对照:f16 需 30.56 GiB、FP8 需 59.82 GiB、Q4 不开启
    `Q4_DEVICE` 需 30.56 GiB —— **均装不下**,即该 checkpoint 在本卡上只有
    Q4-on-device 一条可跑路径,拿不到未量化基线。
  - **遗留(未解决,已另立 #107)**:Q4-on-device 下输出仍不连贯
    (`' Languages Capital Languages...'`,首 token 60750),而 numpy 参考
    **加上模拟 Q4 之后依然连贯**(`' The capital of France is Paris.'`,
    step 7 正常 EOS)。60750 不在 numpy top-20 内(下限 22.85),偏离约
    7 个 logit 量级。已排除:核内 Q4 反量化 kernel(8 层截断 A/B 下
    `MACH_Q4_DEVICE` 开/关给出**相同首 token** 11763)、f16 前向路径(截断
    模型上取到 numpy 次选,间距仅 0.18,落在 f16 舍入内)、rope 缓冲区
    越界(`k_buf` 实为 `b*3072` float > 所需 `b*1024`)。#85 引入的
    `moe_grouped_gate_up_q4` / `moe_grouped_down_q4` 目前**零测试覆盖**,
    建议优先补单元测试。
  - **性能数据暂缺**:既有吞吐数字(25.73 tok/s @ Q4-on-device、capacity 4)
    是在 RoPE 修复**之前**测的,且对应输出不连贯,不能作为有效数据收录;
    待 #107 解决后重测再补。
  - 验证:fmt/clippy -D warnings/check×2 全绿;CPU 全量(177 passed,含
    新增 2 个 rope 测试);GPU 全量 --test-threads 1;离线门禁 52 内核
    +计数断言。
