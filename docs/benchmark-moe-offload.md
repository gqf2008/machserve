# 可复现基准：MoE offload 引擎（TTFT / TPOT）

> 目的：用一种**可复现**、可对比的方式，测出 MachServe MoE offload 引擎在「全驻留 / 有界槽位 offload / 带宽自适应 q*」三种放置下的 TTFT 与 TPOT，让「快没快、快在哪」有据可查。

## 环境

- GPU / OS：与 `docs/benchmark-protocol.md` 一致（AMD RX 7900 XTX，Windows 原生 ROCm 6.2）。
- 模型：一个 **MoE** 开放权重 checkpoint（HF safetensors）。需提前下载到 `.models/`，并设置以下环境变量：

```bash
export MACH_MODEL=<model.safetensors>      # 例如 qwen3-moe-35b.safetensors
export MACH_CONFIG=<config.json>           # 含 num_experts / num_experts_per_tok
export MACH_MOE_SLOTS=2                    # 有界槽位/自适应模式下的 GPU 专家槽位数
export MACH_BENCH_TOKENS=32                # 生成 token 数（TTFT=第1个，TPOT=其余平均）
```

> 真实模型测试/基准默认不加载，需显式设 `MACH_TEST_MODEL` 或不通过测试路径，直接用下面的 example。

## 运行

```bash
cargo run -p mach-model --release --features hip --example moe_offload_bench
```

输出示例：

```
=== MoE offload benchmark ===
model: qwen3-moe-35b.safetensors | d_model=4096 layers=48 experts=128 topk=8 | tokens=32
mode           | TTFT(ms) | TPOT(ms/tok) | tok/s
full          |   210.00 |      14.00 |   71.4
slots=2       |   240.00 |      16.00 |   62.5
adaptive      |   228.00 |      15.20 |   65.8
note: TTFT/TPOT include the offload path syncs/D2H; placement is
      invariance-agnostic, so any diff vs full is scheduling, not accuracy.
```

## 指标口径

- **TTFT**：首个 decode step 的端到端耗时（含 offload 路径的 `self.k.sync()` + D2H/H2D）。
- **TPOT**：`MACH_BENCH_TOKENS` 个 token 的平均/每 token 耗时（`decode_step` 带 logits 回读）。
- **tok/s** = 1000 / TPOT。
- **放置无关性**：三档输出的数学应当一致（`moe_gpu_slot_offload_matches_full` / `moe_gpu_adaptive_offload_matches_full` 已证）；因此三者 TTFT/TPOT 的差异反映的是**调度/同步开销**，不是精度差异。

## 结果记录

每次跑完，把「模型名 / slots / 三项 TTFT·TPOT」记到 `docs/benchmark-results-moe-offload.md`（若不存在则创建），并注明：

- GPU / 驱动版本
- checkpoint 的 HF ID（e.g. `Qwen/Qwen3-30B-A3B`）
- `MACH_MOE_SLOTS` / `MACH_BENCH_TOKENS`
- 是否开启 `--release`

> 备注：这是**单序列** offload 基准（`GpuModel`）。批量 serving 的基准见 `docs/benchmark-protocol.md` 的 batched decode 部分；把本引擎纳入该协议是后续项。

## 与 FreeToken 对标口径（可选）

- FreeToken 论文报的是「长 context TTFT 降 42–58%」「Agent 多轮 TTFT 降 65–80%」，口径是**相对传统 offload**。
- 若要对标，用同一个 checkpoint、同一 prompt，测：全驻留 vs offload（`slots=2`），并记录 TTFT 降幅。注意 FreeToken 是 Python+CUDA，本引擎是 Rust+HIP，绝对数字须同机、同模型、同请求分布。

