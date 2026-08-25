# fp64 三路对拍：GPU(f32) == CPU(f32) == fp64 参考（issue #22）

> 目的：用 **fp64 独立参考**做三路对拍，证明 MoE offload（full / slots / adaptive、批量 cpu-backend）的误差在 **f32 舍入量级**，且**放置无关性在 fp64 下也成立**——offload 只改变调度，不改变数学。

## 方法

- 新增 `crates/mach-model/src/fp64_ref.rs`：**独立 f64 实现**（不复用 f32 路径），覆盖
  - `matvec_t` / `expert_mlp`（gate/up/down SwiGLU）
  - `moe_route` / `moe_layer`（router softmax + top-k 加权和）
  - `moe_batch_cpu_residual`（批量 CPU 残差）
  - `Fp64RefModel`（完整非-MLA forward，供 logits 级三路对拍）
- f32 → f64 转换是精确的，因此同权重下 fp64 与 f32 路径的差就是 f32 路径的**纯舍入误差**。
- 对拍对象：
  - GPU(f32 logits)：`GpuModel`（full / slots / adaptive）、`BatchedModel`（full / cpu-backend）。
  - CPU(f32 logits / 残差)：`RefModel` / `moe_offload` 现有 f32 路径。
  - fp64 参考：`fp64_ref`。

## 容差与依据

- **MoE 层残差**：`tol = 8 · dots · √n · ε32 · scale`（`ε32 = 2⁻²³ ≈ 1.19e-7`，`dots` = 链式点积数 ≈ `3·topk+1`，`n = max(d, inter)`）。依据：单个长度-n f32 点积典型相对误差 ≈ √n·ε32（最坏 n·ε32），8 倍余量。
- **完整 forward（CPU f32 vs fp64）**：`tol = 8 · 7 · layers · √n · ε32 · scale`（每层约 7 个点积：attention q/k/v/o + MLP g/u/d，误差随层数近似线性累积），8 倍余量。
- **GPU 对拍**：沿用仓库既有 GPU-vs-CPU 边界（`tests/moe.rs` 的 `2e-3 + 2e-3·scale`）——GPU GEMM 用分块顺序 + `__expf`，允许 2e-3 相对 + 绝对余量。
- **argmax 一致性**：三路 logits 的 argmax 必须完全相同。

## 运行

```bash
# CPU 单测（非 GPU，lib 测试）
cargo test -p mach-model --lib

# GPU 对拍（#[ignore]、串行，每次 1 个）
cargo test -p mach-model --features hip --test fp64_parity gpu_full_slots_adaptive_three_way_fp64_parity -- --ignored --test-threads=1
cargo test -p mach-model --features hip --test fp64_parity batched_cpu_backend_three_way_fp64_parity -- --ignored --test-threads=1
cargo test -p mach-model --features hip --test fp64_parity moe_real_three_way_fp64_parity -- --ignored --test-threads=1
```

## 结果（本机，AMD RX 7900 XTX 无头卡，Windows ROCm 6.2，2026-08-25）

### CPU 单测（`cargo test -p mach-model --lib`，5 项 fp64 测试全绿）

| 对比 | 形状/种子 | max abs diff | 相对误差（diff/scale） |
|---|---|---|---|
| expert_mlp f32 vs fp64 | d=64..256, inter=128..512, 4 seeds | 2e-8 … 1.4e-4 | ≈ 5e-7 |
| moe_layer（router+topk）f32 vs fp64 | ne=4..16, topk=2..4, 4 seeds | 2e-8 … 1.5e-4 | ≈ 5e-7 |
| 批量 CPU 残差 f32 vs fp64 | b=2, 同上 | 2e-8 … 1.0e-4 | ≈ 5e-7 |
| fp64 vs 两个 f32 放置（resident / cpu-overflow） | 同上 | 与 moe_layer 相同 | ≈ 5e-7 |
| 完整 forward f32 vs fp64 | 4 形状 × 4 seeds（含 qk_norm） | ≈ 1e-6（scale≈2） | ≈ 5e-7 |

路由稳定：所有种子/形状下 f32 与 fp64 的 router top-k **完全相同**（无近并列翻转）。

### GPU 三路对拍（合成 MoE：d=128, ne=4, topk=2, tokens=[5,9,3,200]）

| 模式 | scale | GPU vs CPU(f32) | GPU vs fp64 | CPU vs fp64 | argmax |
|---|---|---|---|---|---|
| full | 1.864 | 1.31e-6 (7.0e-7) | 9.13e-7 (4.9e-7) | 7.16e-7 (3.8e-7) | 649=649=649 |
| slots=1 | 1.864 | 1.55e-6 (8.3e-7) | 1.12e-6 (6.0e-7) | 7.16e-7 (3.8e-7) | 649=649=649 |
| adaptive | 1.864 | 1.31e-6 (7.0e-7) | 9.13e-7 (4.9e-7) | 7.16e-7 (3.8e-7) | 649=649=649 |

### 批量 cpu-backend 三路对拍（batch=2, tokens=[5,9]）

| 行 | scale | GPU vs CPU(f32) | GPU vs fp64 | CPU vs fp64 | argmax |
|---|---|---|---|---|---|
| row0 full | 2.133 | 1.13e-6 (5.3e-7) | 9.93e-7 (4.7e-7) | 1.12e-6 (5.3e-7) | 324=324=324 |
| row0 cpu-backend | 2.133 | 9.54e-7 (4.5e-7) | 8.49e-7 (4.0e-7) | 1.12e-6 (5.3e-7) | 324=324=324 |
| row1 full | 1.780 | 1.21e-6 (6.8e-7) | 1.16e-6 (6.5e-7) | 1.05e-6 (5.9e-7) | 672=672=672 |
| row1 cpu-backend | 1.780 | 1.31e-6 (7.4e-7) | 9.20e-7 (5.2e-7) | 1.05e-6 (5.9e-7) | 672=672=672 |

批量放置无关性：full vs cpu-backend max diff = 6.26e-7（rel 2.9e-7）。

### 真实模型（qwen3-moe-tiny：d=1024, 24 层, 16 专家, topk=4, tokens=[1,100,200,300]）

| 模式 | scale | GPU vs CPU(f32) | GPU vs fp64 | CPU vs fp64 | argmax |
|---|---|---|---|---|---|
| full | 4.915 | 9.54e-6 (1.94e-6) | 9.04e-6 (1.84e-6) | 8.73e-6 (1.78e-6) | 18256=18256=18256 |
| slots=4 | 4.915 | 9.54e-6 (1.94e-6) | 9.04e-6 (1.84e-6) | 8.73e-6 (1.78e-6) | 18256=18256=18256 |
| adaptive | 4.915 | 9.54e-6 (1.94e-6) | 9.04e-6 (1.84e-6) | 8.73e-6 (1.78e-6) | 18256=18256=18256 |

括号内为相对误差（diff / max|logit|）。

## 结论

- **误差在 f32 舍入量级**：合成模型三路最大绝对误差 ≈ 1e-6（相对 ≈ 7e-7）；真实 qwen3-moe-tiny 三路最大绝对误差 ≈ 9.5e-6（相对 ≈ 1.9e-6），均远低于 f32 舍入累积上界（`√n·ε32` 量级 × 层数）。
- **argmax 三路全一致**：合成与真实模型、所有 offload 模式（full / slots / adaptive / 批量 cpu-backend）的 logits argmax 完全相同。
- **放置无关性在 fp64 下成立**：GPU full / slots / adaptive 与 CPU 参考在 fp64 参考下落在同一误差带；批量 full vs cpu-backend max diff = 6.3e-7（rel 2.9e-7）。fp64 残差本身对「resident」与「cpu-overflow」两种 f32 放置的偏差完全相同（≈5e-7 相对）——offload 只改变调度，不改变数学。
