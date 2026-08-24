# GPU 验证清单（7900 XTX + Windows ROCm 6.2）

> 本机（7900 XTX）在持续 GPU 负载下会触发驱动 TDR（Event 4101），严重时整机无响应需硬复位。**先做系统级修复，再用「单条 + 超时」跑验证。**

## 第 0 步：系统级修复（需管理员，我当前会话非管理员无法代改）

1. **放大 TDR 超时**（避免长 GPU 操作被误复位）：
   - 用管理员身份导入 `docs/raise-tdr-timeout.reg`，或管理员 PowerShell 执行：

```powershell
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 60 /f
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDdiDelay /t REG_DWORD /d 60 /f
```
   - **重启生效。**
2. **禁用 Error 状态虚拟显示器**（可选，强烈建议；注意：vdev/Todesk 虚拟显示器禁用可能**断开远程会话**，确认后再做）：设备管理器 → 显示适配器 → `vdev 虚拟显示器`/`Todesk Virtual Display Adapter` → 禁用。
3. 确认 AMD 驱动（32.0.31035.1003）与 ROCm 6.2 匹配；若不稳，用已对齐的稳定版。

## 第 1 步：HIP 冒烟（最轻，~1s/条，先确认 GPU/HIP 活着）

```powershell
cargo test -p mach-engine --features hip --lib hip::tests:: -- --test-threads=1
```

- **预期**：`hip_device_is_visible` / `hiprtc_saxpy_runs_on_gpu` / `hip_graph_capture_records_and_replays` / `hip_graph_lifecycle_is_strict` → 4 passed。
- **若这步就硬锁**：别继续，先做第 0 步修复或换稳定机器；把 `System` 日志的 4101 时间发我。

## 第 2 步：MoE offload 对拍（小随机模型，非真实大模型；每条 ~20-30s）

```powershell
$t = @("moe_gpu_forward_matches_cpu_reference","moe_gpu_offload_placement_invariant","moe_gpu_slot_offload_matches_full","moe_gpu_adaptive_offload_matches_full")
foreach ($k in $t) {
  $j = Start-Job { param($k) cargo test -p mach-model --features hip --test moe -- --ignored $k -- --test-threads=1 } -ArgumentList $k
  if (Wait-Job $j -Timeout 180) { Receive-Job $j } else { Stop-Job $j; "TIMEOUT on $k - likely a hang; STOP" }
  Remove-Job $j -Force
}
```

- **预期**：4 条全过；尤其 `moe_gpu_offload_placement_invariant` / `moe_gpu_slot_offload_matches_full` / `moe_gpu_adaptive_offload_matches_full`，它们验证**放置无关性**（offload/槽位/自适应与全驻留一致，diff≈0）。
- **任一超时/硬锁**：说明该内核路径触发 TDR，立即停并记录；换稳定机器或继续修驱动。

## 第 3 步：真实 MoE 基准（TTFT / TPOT）

```powershell
$env:MACH_MODEL="qwen3-moe-35b.safetensors"; $env:MACH_CONFIG="config.json"; $env:MACH_MOE_SLOTS="2"; $env:MACH_BENCH_TOKENS="32"
cargo run -p mach-model --release --features hip --example moe_offload_bench
```

- **预期输出**：打印 `mode | TTFT(ms) | TPOT(ms/tok) | tok/s` 三行（full / slots / adaptive）。
- **判定**：`slot`/`adaptive` 的 TTFT/TPOT 应**接近** `full`（offload 主要是调度/同步开销，不是精度差异）；差距明显过大说明 offload 路径同步开销是主要成本。若卡住/超时：换稳定机器，把`docs/benchmark-moe-offload.md`的现象记录给我。

## 判定与上报

- 把第 1/2/3 步的**实际输出 + 是否超时/硬锁**贴给我。
- 若第 2 步全过、第 3 步能跑：offload 引擎在本机可运行；否则按 `docs/gpu-hardware-note.md` 换环境。

## 备注
- 真实模型测试已门控 `MACH_TEST_MODEL`；GPU 测试已 `#[ignore]`，必须 `-- --ignored` 才跑。
- 本机只做**单条、时间盒**验证；不要连续跑整套 GPU 套件。

