# 本机 GPU 硬件稳定性说明（7900 XTX + Windows ROCm）

> 重要：在**当前这台机器**上跑持续 GPU 负载，可能触发**显示器驱动 TDR（Event 4101）**，严重时**整机无响应 → 黑屏 → 只能硬件复位**。运行任何 GPU 测试/基准前请先读本页。

## 已观察到的根因（2026-08-24）

1. **驱动 TDR（Event 4101）**：`System` 日志在 2026-08-23 17:51 / 20:29 各出现一次 `Display` 4101（- display driver amdkmdag stopped responding and has recovered）。这是 GPU 操作超时被驱动看门狗复位；在持续推理/长 decode/JIT 编译下更易触发。
2. **虚拟显示适配器处于 Error 状态**：`vdev 虚拟显示器` 与 `Todesk Virtual Display Adapter` 在 `Win32_VideoController` 里 `Status=Error`。虚拟显示器叠加在物理 7900 XTX 上，会恶化显示栈稳定性。
3. **Windows 原生 ROCm 6.2 对 RDNA3 仍属实验性**，长任务稳定性不足。

> 结论：**本机不适合做 GPU 运行期验证/压测**；offload 引擎代码已完成并可编译（`cargo check --features hip` 不执行 GPU），运行期数据需换**稳定 GPU 环境**（另一台机器、远程/CI）。

## 安全运行 GPU 测试/基准的协议

- **GPU 测试默认 `#[ignore]`**：`cargo test --features hip` 不会跑它们。要跑需显式：

```bash
cargo test -p mach-model --features hip --test moe -- --ignored --test-threads=1
```

- **单条跑 + `--test-threads=1`**（避免多测试并发争 GPU）。
- **硬超时护栏**（超时就停，不重试）：

```powershell
$j = Start-Job { cargo test -p mach-model --features hip --test moe -- --ignored moe_gpu_slot_offload_matches_full -- --test-threads=1 }
if (Wait-Job $j -Timeout 180) { Receive-Job $j } else { Stop-Job $j; "超时，疑似卡死，已中止" }
Remove-Job $j -Force
```

- 真实模型测试已门控：需 `MACH_TEST_MODEL` 指向 `.safetensors`（默认跳过，不加载大模型）。
- 基准：`cargo run -p mach-model --release --features hip --example moe_offload_bench`（需 MoE checkpoint，见 `docs/benchmark-moe-offload.md`）。

## 系统性修复建议（需管理员，按需执行——不在本仓库代跑）

1. **放大 TDR 超时**（避免长时间 GPU 操作被看门狗当作超时）：在 `HKCU\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`（或 HKLM）新增 `TdrDelay`=60、`TdrDdiDelay`=60（DWORD），重启。
2. **移除/禁用 Error 状态的虚拟显示器**（vdev、Todesk Virtual Display Adapter），只保留物理 7900 XTX。
3. 更新 AMD 驱动到与 ROCm 6.2 匹配的稳定版；必要时降级/调整驱动。
4. **优先换稳定 GPU 环境做运行期验证**，本机不再做持续 GPU 压测。

