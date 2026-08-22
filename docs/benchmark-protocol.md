# TPOT 对标协议: MachServe vs TokenSpeed(7900 XTX)

> 目的:在 **AMD Radeon RX 7900 XTX(24G, gfx1100)** 上,用同一模型、同一权重、
> 同一请求,对比 MachServe(纯 Rust 调度 + HIP kernel + graph 重放)与
> TokenSpeed 的 decode TPOT。当前状态:**MachServe 侧已就绪,TokenSpeed 侧环境待部署**
> (WSL2 暂无法启动:0x800705aa 资源不足;或由用户在其环境运行后填表)。

## 1. 基准模型

- **模型**:Qwen2.5-0.5B-Instruct(24 层 / hidden 896 / GQA 14:2 / intermediate 4864 /
  rope_theta 1e6 / tie_embeddings)
- **权重**:`model.safetensors`(942MB, BF16)——MachServe 侧已下载在 `.models/`
- **请求**:固定 token 序列(如 `[1, 2, 3, ...]` 或一条真实 prompt 经 tokenizer 编码后)
- **序列长度**:固定(如 200 token decode)

## 2. MachServe 侧(已可运行)

```bash
cargo run -p mach-model --release --features hip --example qwen_bench
```

输出(7900 XTX 实测, 2026-08-22):

| 指标 | MachServe eager | MachServe graph |
|---|---|---|
| launch-only 解码 | ~1.1-1.6 ms/tok | **0.30-0.47 ms/tok** |
| 完整步(含 logits 读回) | ~7.6-10.5 ms/tok | ~7.2-9.8 ms/tok |
| 数值 vs fp64 参考 | 相对误差 **4.4e-6** | 一致 |

## 3. TokenSpeed 侧(待跑)

在用户侧环境(WSL2 ROCm + PyTorch + tokenspeed 包,或自有 Linux 机)运行
TokenSpeed 的对应基准,记录:

| 指标 | TokenSpeed(7900 XTX) |
|---|---|
| decode TPOT(同 200 token) | ____ ms/tok |
| 首 token 延迟 / prefill | ____ ms |
| 备注(走哪个 kernel 栈: Triton 回退? gluon?) | ____ |

**注意**:`tokenspeed-kernel-amd` 目前只有 gfx950/gfx1250 内核,**7900 XTX(gfx1100)
大概率走 Triton 回退路径**。记录时请注明实际 kernel 栈,便于公平解读。

## 4. 对比口径

- **同模型同权重同请求**:必须一致,否则无意义。
- **TPOT 定义**:单个 decode 步(输入 token → 采样所需 logits)的每 token 延迟。
  两侧都应排除/标注"logits 全量读回"与"仅采样 token"的口径差异。
- **环境**:同机(7900 XTX),CPU 频率/电源模式尽量一致。

## 5. 结论模板

| 指标 | MachServe | TokenSpeed | 备注 |
|---|---|---|---|
| decode TPOT(launch-only) | 0.3-0.5 ms/tok | ____ | 纯 kernel+调度 |
| decode TPOT(完整步) | 7-10 ms/tok | ____ | 含 logits 读回 |
| 数值正确性 | 4.4e-6(vs fp64) | ____ | |
