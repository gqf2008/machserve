# TokenSpeed 对齐现状与路线

> 生成：2026-08-25。本文件是 machserve 与上游 [lightseekorg/tokenspeed] 的
> 对齐基线：哪些概念已落地、哪些是缺口、按什么顺序补齐。目标是让
> 「除内核外全 Rust、超越 TokenSpeed」的声称有可核对的验收清单。

[lightseekorg/tokenspeed]: https://github.com/lightseekorg/tokenspeed

## 1. 关系与背景

- **上游 TokenSpeed**：C++ 调度器控制面 + Python 执行面 + 可插拔内核（Blackwell
  MLA / gfx950 / gfx1250），面向 agentic 工作负载，活跃推进中。
- **wt-rust-port 实验（gqf2008 自己的分支）**：把 C++ 调度器控制面移植成 Rust
  （`ts-scheduler-core` + `ts-scheduler-pyo3`，~11k 行，`#![forbid(unsafe_code)]`），
  验证可行（110 cargo test + 58/58 pytest 契约），但上游 PR #1193 被维护者关闭
  （"You can maintain your fork"），实验已收尾，不再开发。
- **machserve**：这条「Rust 化 TokenSpeed」路线的实际载体。与 wt-rust-port 不同，
  machserve 是 clean-room 引擎，**不走「相同 Python API 替换」的 cutover**，而是
  原生实现同样的调度语义（FSM / 分页 KV / 前缀缓存），host 侧全 Rust、无 Python。

因此「对齐上游」的定义 = **把 rust-port.md 里确立的设计契约（FSM 状态机、前缀
哈希链、LCM 块池、KV 事件）原生实现进 machserve**，而不是同步上游 commit 或
通过上游的 pytest。

## 2. 对齐状态表

| TokenSpeed 能力 | machserve 现状 | 状态 |
|---|---|---|
| MLA（Blackwell 内核） | MLA 单序列/批量/连续批处理 + F16 decode，与 CPU 参考对拍 | ✅ 概念对齐（合成权重；真实 checkpoint 受显存限制待验证） |
| MoE（Kimi-K3 / flashinfer tactics） | MoE 全链路：router + 分组 GEMM + 连续批处理 + HTTP | ✅ 已对齐 |
| 连续批处理 / 请求生命周期 | `continuous.rs`（prefill/decode 混合、EOS、槽位压缩） | 🟡 功能对齐，无显式 FSM |
| Agentic state reuse | `state_reuse.rs`（token 边界锚点 + 增量 prefill，dense/moe/mla 对拍） | 🟡 同会话多轮复用；**不跨请求** |
| **分页 KV + 块表 + LCM 块池** | `kv_block_pool.rs` 已移植（LCM 放置 + 块表 + RAII）；静态槽位尚未替换 | ❌ **架构缺口（地基已备，待接入）** |
| **全局前缀缓存（SHA-256 前缀哈希链 + matcher）** | `prefix_cache.rs` + `reuse_planner.rs` + `prefix_kv.rs`：CPU 参考路径已实现**跨请求前缀共享**（共享前缀的请求只算 delta，复用 logits 与全算逐位一致） | 🟡 已打通 CPU 路径；GPU（batched.rs）接线待做 |
| **调度器 FSM（7 状态 + 事件 + Retracted/WriteBack/LoadBack）** | `scheduler_fsm.rs` 已移植（7 状态 + 12 事件，非法迁移 panic） | 🟡 已实现，未接入 `continuous.rs` |
| KV 缓存事件（PD 跨节点） | 无（单节点） | ⏸ 单卡目标暂不需要 |
| spec-decode | 已实现；实测 0.29x 净负，暂停 | ⏸ 需更便宜草稿/批量形态 |
| 存储量化 | Q4 int4 真机验证；FP8 E4M3 存储路径已合入 | ✅ 已对齐（存储级） |
| 内核栈 | hipBLAS + hiprtc（gfx1100 / 7900 XTX） | 🟡 目标硬件不同（上游 gfx950/gfx1250），内核无法直接对齐 |

## 3. 中期对齐路线（本分支 `mid/paged-kv-prefix`）

按依赖顺序分三片，全部 CPU-only、可独立单测，最后再接入 decode 路径：

1. **`scheduler_fsm.rs`**：7 状态 + 12 事件的请求生命周期 FSM（移植上游契约，
   非法迁移 panic），带 `on_transition` 钩子供未来接副作用。✅ 已实现（13 单测）。
2. **`kv_block_pool.rs`**：LCM 物理块池 + 逻辑块表 + RAII 块引用（CacheBlockRef
   最后 owner drop 归还槽位），page 0 保留 null。✅ 已实现（19 单测）。
3. **`prefix_cache.rs`**：SHA-256 前缀哈希链（framing 与上游逐字节一致，含黄金
   向量）+ 前缀索引 + Full-attention 连续命中 matcher。✅ 已实现（18 单测）。
4. **`reuse_planner.rs`**：跨请求前缀复用准入规划器（组合 1–3，探测共享前缀 →
   全有或全无分配尾部新块 → 维护索引）。✅ 已实现（10 单测）。
5. **`prefix_kv.rs`**：CPU 参考路径的跨请求前缀共享——按 plan 用 Anchor 恢复
   复用前缀的 KV、只算 delta 并逐页快照缓存。✅ 已实现（6 单测：
   `shared_prefix_reuses_and_matches_full_recompute` 等；10-token 请求共享 8-token
   前缀时只算 2 个 token，复用 logits 与全算逐位一致）。
6. **`paged_scheduler.rs`**：CPU 侧分页调度器——FSM（Submitted→PrefillDone→
   Decoding→Finished）+ 复用规划 + 前缀 KV 驱动参考模型的多请求调度。✅ 已实现
   （2 单测：5 个共享 8-token 系统提示的请求，45 个 prompt token 复用 32 个、
   只算 13 个——**节省 71%**，贪心解码与全算逐位一致；对标 FreeToken 多轮
   TTFT -65..-80% 目标）。
7. **`cpu_engine.rs`**：CPU 连续批处理引擎——队列 + 槽位复用 + 交错 prefill/decode
   + 跨请求前缀复用 + FSM 生命周期（镜像 `continuous.rs` 的 serving 语义，
   是 GPU 接线的硬件无关参考实现）。✅ 已实现（4 单测：5 请求共享系统提示交错
   运行全部完成、贪心解码与全算逐位一致、prompt 复用 71%；容量 2 的引擎槽位
   复用跑完 5 个排队请求）。

### 接入计划（后续批次，非本分支）

- 把 `BlockPool`/`BlockTable` 接到 `batched.rs` 的 KV 布局：静态槽位 → 分页表，
  长上下文内存按需分配、可驱逐。
- 用 `prefix_cache` 做跨请求前缀共享：系统提示/工具定义被并发请求复用，多轮
  TTFT 对标 FreeToken 报告的 65–80% 降幅。
- 用 `scheduler_fsm` 替换 `continuous.rs` 的隐式生命周期，补 Retracted
  （WriteBack/LoadBack 容量驱逐），与 MoE offload（`moe_backend`）协同。

## 4. 验收清单（对齐门禁）

- [ ] `scheduler_fsm`：上游全部合法迁移 + 非法迁移 panic，单测覆盖
- [ ] `kv_block_pool`：acquire/release 不变量、RAII 归还、满池拒绝，单测覆盖
- [ ] `prefix_cache`：黄金向量逐字节一致（空页/链式/extra/负 token）、索引与
      matcher 连续 run 语义
- [ ] 三模块 CPU-only：`cargo test -p mach-model --lib` 全绿 + `clippy -D warnings`
      干净（不依赖 GPU）
- [x] CPU 参考路径：跨请求前缀共享（复用 logits == 全算，逐位一致；delta-only 计算）
- [x] CPU 分页调度器：FSM 生命周期 + 前缀共享多请求调度（5 请求共享系统提示 → 71% prompt token 复用）
- [x] CPU 连续批处理引擎：队列/槽位复用/交错 prefill+decode + 前缀复用（GPU 接线参考）
- [ ] GPU（batched.rs）接线：静态 KV 槽位 → 分页表 + 前缀共享（需真机 A/B：TTFT/TPOT）
  - 地基已备：`paged_kv.rs`（块表 + 分页 attention CPU 参考，与连续参考逐位一致）、
    `ATTN_DECODE_PAGED` 内核（已进离线 hiprtc 编译门禁，37 内核全过）；
    剩余：接入 batched.rs + 数值对拍 + 真机 A/B
- [x] owning-ref 索引 + 容量驱逐：`PrefixCacheIndex` 持 `CacheBlockRef`（块被索引钉住，
      释放请求 plan 不会误释放仍被复用的块；上游 `prefix_index` 语义）；`PrefixKvCache`
      池满时 LRU 驱逐最冷页（索引 + 主机页 + 释放块），缓存有界
- [ ] README/roadmap 同步（对齐状态表更新，不再把「静态 KV」当长期设计）