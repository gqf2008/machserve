# 分页 KV 前缀复用基准结果（真机实测）

> 记录每次真机基准，含环境、模型、命令与结果，供复现与对比。

## 2026-08-27 · paged vs contiguous 前缀复用 A/B（#78 C6）

- **GPU / 驱动**：AMD RX 7900 XTX / gfx1100，Windows 原生 ROCm 6.2，HIP 后端，`--release`。
- **模型**：合成 tiny（d_model=128，2 层，4 heads / 2 kv-heads，head_dim=32，vocab=1024，max_seq=256），`Weights::random(seed=42)`。
- **工作负载**：5 请求共享 64-token「系统提示」页 + 各自唯一尾 token（prompt_len=65，max_new=8），**交错准入**（请求 i+1 在请求 i 首 token 后准入，真实服务到达形态）；先热身后计时（hiprtc 缓存 + hipBLAS/驱动惰性初始化先行支付）。
- **命令**：
  ```bash
  cargo run -p mach-model --release --features hip --example paged_prefix_ab_bench
  ```

### 结果（同一 GPU、同权重、同请求分布；连续引擎 vs 分页引擎）

| 指标 | 连续（全量重算） | 分页（页池复用） | 倍率 |
|---|---|---|---|
| 提示词计算量（prompt tokens computed） | 325 | **69** | 节省 78.8% |
| 端到端 wall（5 请求 × 8 token） | 93.2 ms | 32.8 ms | **2.84x** |
| 首请求 TTFT（各自 65-token prefill） | 17.0 ms | 17.4 ms | 0.98x（公平对齐） |
| 后续请求 TTFT（复用后 delta-only） | 17.4 ms | 1.3 ms | **13.4x** |
| TPOT（含小批 decode） | 2.33 ms/tok | 0.82 ms/tok | 2.8x（批变小，如实报告） |

### 解读

- **节省率精确命中理论值**：`savings_pct=78.8` == `(N-1)·page/(N·prompt_len) = 64·4/325`；第二请求起只算尾 token（1/65），KV 前缀页由首请求物化后经块表混叠读取，输出与全量重算逐 token 一致（正确性由 `shared_prefix_paged_reuse_matches_full_compute` 与连续引擎套件保证）。
- **后续请求 TTFT 13.4x**：delta-only prefill 只算 1 个 token 的 embed→LM-head 全链，主成本从 65-token prefill 降到 1-token decode。
- **首请求两引擎对齐（17.0 vs 17.4 ms）**：确认 paged 路径无固有开销差；早期 7x 差是首跑引擎承担的一次性 hipBLAS/驱动初始化，热身已消除（A/B 口径见 `docs/benchmark-protocol.md`）。
- **TPOT 差异是批形状产物**：分页引擎交错准入使后续 decode 多为单行小批；不做加速声称，真实多并发 decode 吞吐需 `lctx_bench`/服务端口径另行测量。
- **结论 / 下一步**：跨请求前缀共享的 GPU 服务链已闭环（`MACH_PAGED=1`）。后续：真实模型（Qwen3-8B）同口径 A/B、共享前缀页接入后的页池驱逐/LRU（当前池容量=capacity×max_pages，装满即拒）、`scheduler_fsm` 替换隐式生命周期。

## 真实模型同口径 A/B（harness 已就绪，待机器窗口）

合成 tiny 模型上的数字（上一节）不能替代真实 checkpoint 的口径。真实模型 A/B 用
`tools/paged_prefix_ab_real.py`，一个 arm 一次调用：

```bash
# arm 1（分页）
python tools/paged_prefix_ab_real.py --arm paged
# 留 >=10 分钟观察窗口，确认 doctor 的 device_count 没变，再做 arm 2
python tools/paged_prefix_ab_real.py --arm contiguous
```

- 工作负载：5 个请求共享同一段长 system 块（`--shared-repeats 22` ≈ 1937 token），
  只有末尾 user turn 不同，`max_new=8`、`temperature=0`、SSE 逐帧计时。
- 分页 arm 会自动带 `MACH_PAGED_DEBUG=1`，服务日志里每个请求一行
  `paged: admit id=… prompt_tokens=… full_pages=… reused_pages=…`，预填物化时一行
  `paged: register slot=… full_pages=…`。**该复用而 `reused_pages=0` 就是缓存未命中的直接证据。**
- arm 若因为模型/形状不受支持而静默退化成 contiguous，harness 会检查日志里的
  `MACH_PAGED is unsupported` 并直接失败，不会输出误导性数字。
- **安全约束**：两个 arm 必须拆到不同时间窗口，中间留 ≥10 分钟观察窗口。本机曾两次
  在重 GPU arm 结束的同一秒级窗口内出现显示驱动 TDR（Event 4101），随后 7900 XTX
  掉出 HIP 设备枚举（`doctor` 只剩核显），必须重启才能复位。每 arm 结束都查一次
  4101 并比对 `mach-server doctor` 的 `device_count`。