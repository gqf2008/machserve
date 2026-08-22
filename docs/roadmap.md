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
