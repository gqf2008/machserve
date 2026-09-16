# ROCm/Windows 重 GPU 负载整机硬断电：事故记录与排查交接

> 交接对象：接手排查的会话
> 记录人：2026-09-15 会话 + 当晚深夜第二次复核（新增 §12，并修订 §1/§3.2/§6/§11）
> 机器：MAXSUN MS-eSport B850ITX / Ryzen 5 9600X / 32GB / RX 7900 XTX (gfx1100) + 核显 / Windows 10 19045 / ROCm 6.2
> 状态：**未定案，但范围已收窄**。软件路径已排除（MachServe 与 llama.cpp 两个独立栈都触发同一形态）；
> 供电线 daisy-chain 与"瓦数不足"已由用户答复排除（**两根独立 8pin / PSU 确认 850W+**，见 §12.4）。
> 剩余嫌疑：**PSU 个体老化/故障、显卡侧 VRM、市电/电路、温度**——都是硬件面，不再是代码面。

---

## 0. 一页速览（先读这段）

**现象**：整机**瞬间断电**——不是蓝屏、不是驱动恢复。事件日志只有 `Kernel-Power 41` + `EventLog 6008`，
**零 WHEA、零 BugCheck、零 minidump**。典型电源保护闩锁行为：刚上电会在 7~8 秒内再次断电，静置约 1 分钟后才能正常启动。

> **§12 复核修订**：`Kernel-Power 41` 的 `BugcheckCode` 字段 **30 天内逐条核实全为 0** ⇒ 这些断电**一次都不是蓝屏**。
> 并且机器**断电后会自己起来**（08-24 连着 20:31 → 20:39 → 20:42 → 20:48 四次，间隔 3~8 分钟，没人能每分钟手动按一次电源键），
> 说明 BIOS 的"断电恢复后自动开机"是开着的——**"必须人工上电"不能再当判据**。

**2026-09-15 一天内两次**：

| 时刻 | 先兆 | 结果 |
|---|---|---|
| 17:24:40 | Display 4101（amduw23g TDR，"已成功恢复"） | 随后整机异常关机；20:33:52 冷复位后 XTX 一度掉出 HIP 枚举 |
| ~22:04–22:07 | **无 4101、无任何日志先兆** | 整机瞬断，22:07:21 开机 |

**崩机密度（§12 新增，30 天全量）**：`Kernel-Power 41` 共 **30 次**（最早 08-20 21:09，最新 09-15 22:07），
`Display 4101` 共 21 次，WHEA **0** 次，真正的蓝屏只有 08-23 14:53 一次（有 WER 1001）。
分布成簇：08-23 晚 3 次、08-24 晚 4 次（18 分钟内）、09-06 下午 3 次（8 分钟内）、09-12 一天 6 次。
**这个密度与"某条软件路径有 bug"不相容**——这一个月 MachServe 换了几十个 PR，崩机密度没有变化。

**关键新事实（改变了此前的结论）**：第二次的直接触发是 **llama.cpp（HIP）加载并推理 Qwen3.8-27B Q4_K_M**，
**不是 MachServe**。同一 build、同一模型：

- `llama-bench -p 64 -n 8 -r 1 -ngl 99` → **12.2s 正常跑完**（pp64 537.47 t/s、tg8 36.70 t/s）
- 紧接着的 `llama-cli -c 2048 -n 48 -ngl 99` → **整机硬断电**

因此此前"llama.cpp 稳定 ⇒ 问题在 MachServe"的推理**不成立**，至少作为定论不成立
（"同样的负载有时通过、有时死"本身就是硬件层随机性的典型特征）。

**建议的第一刀**（§12 后更新）：线材与瓦数已排除，剩下的是**装功耗温度监控 + 限功耗判别实验**；
零 GPU 风险的动作优先（§8 的协议已由 `tools/gpu_guard.ps1` 自动化）。**不要回 MachServe 代码里找 bug**——
那会继承一个已经被证伪的前提；只有**带 4101 的 A 类**才回代码面。

---

## 1. 环境（本次已核实，含查询命令）

| 项目 | 值 |
|---|---|
| 主板 | MAXSUN **MS-eSport B850ITX**（ITX 版型，风道/供电余量天然受限） |
| CPU | AMD Ryzen 5 9600X，6C/12T |
| 内存 | 32 GB（33398587392 B） |
| dGPU | **AMD Radeon RX 7900 XTX**，`PCI\VEN_1002&DEV_744C&SUBSYS_2422148C&REV_C8`，PCI 总线 3 / 设备 0 / 功能 0 |
| dGPU 链路 | **Gen4 x16 满速**（CurrentLinkWidth=16 / CurrentLinkSpeed=4，与 MaxLinkWidth/MaxLinkSpeed 相同） |
| dGPU 驱动 | 32.0.31041.1004（2026-08-17），内核驱动名 `amduw23g` |
| iGPU | AMD Radeon(TM) Graphics（DEV_13C0），**正在驱动 3840x2160 显示器** |
| 显示拓扑 | **XTX 无头**（不接显示器）；显示器在核显上 |
| 虚拟显示驱动 | `vdev 虚拟显示器`(OK)、`ToDesk Virtual Display Adapter`(**Status: Error**)、`ToDesk(R) Virtual Display Device`(OK) |
| 磁盘 | ZHITAI TiPlus7100 4TB NVMe / Samsung 850 EVO 500G / Kingston SV300 240G / SanDisk Extreme Pro 4TB(USB) |
| 电源方案 | 高性能（`8c5e7fda-...`；ToDesk 会周期性切换它，属正常噪音） |
| TDR 注册表 | `TdrLevel/TdrDelay/TdrLimitTime/TdrLimitCount` **均未设置** → 走 Windows 默认（TdrDelay=2s、60s 内 5 次上限）。已排除"有人把 TDR 关掉了" |
| ROCm | 6.2 + clang 19，gfx1100/gfx1036；该版本只有 `__hip_fp8_e4m3_fnuz` |
| 已安装工具 | **无** amd-smi / rocm-smi / HWiNFO / clpeak / gpu-burn。有 GPU-Z 2.70（需提权 + 只能 GUI，无 CLI 日志）。ROCm 6.2 Windows 不带 amd-smi。~~`C:\Program Files\AMD` 下没有 Adrenalin GUI（CNext 不存在，驱动-only 装法）~~ **2026-09-15 22:51 已装上完整版 Adrenalin 26.8.1（同一驱动版本，GUI 26.10.41.01）**，提供 Performance → Tuning 功率/频率上限 + Metrics 功耗温度日志；根因与装法见 §13 |
| 电源（PSU） | **850W 以上**（用户 2026-09-15 深夜确认，见 §12.4）。**真实型号/年龄仍未知**——需要现场看铭牌 |
| dGPU 供电线 | **两根独立 8pin**（用户确认，不是 daisy-chain）⇒ H1 的经典踩坑点排除；接头有无发黄/变形仍需现场看 |

查询命令：

```powershell
Get-CimInstance Win32_BaseBoard | Select Manufacturer,Product
Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors
Get-CimInstance Win32_VideoController | Select Name,DriverVersion,PNPDeviceID,CurrentHorizontalResolution
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' | Select Tdr*
$d = Get-PnpDevice -Class Display | ? FriendlyName -like '*7900*'
Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName DEVPKEY_PciDevice_CurrentLinkWidth,DEVPKEY_PciDevice_CurrentLinkSpeed
powercfg /getactivescheme
```

> 注：`Win32_VideoController.AdapterRAM` 对 XTX 报 4GB，是 WMI 的已知 32 位字段溢出，不是真实显存。
> 真实值看 `mach-server doctor`（23.84 GiB free / 23.98 GiB total）。

---

## 2. 时间线（2026-09-15）

| 时间 | 事件 | 证据来源 |
|---|---|---|
| 17:23–17:25 | `tools/paged_prefix_ab_real.py` 跑 Qwen3-8B / `MACH_KV=f16` / `MACH_PAGED=1` / capacity 1 / prefill_rows 16 / ~420-token 共享前缀 + 3 短 completion | 上一会话记录 / benchmark-results-paged-prefix.md |
| 17:24:40 | **Display 4101**：`amduw23g-203304-e29338d6 已停止响应，并且已成功恢复` | System 日志 |
| 17:2x–17:3x | 整机异常关机（6008 记"上次关机 17:09:25"，与 4101 时间矛盾，见 §3.2） | EventLog 6008 |
| 20:33:52 | 冷复位成功 | 开机时间 |
| 20:33–21:55 | 构建 llama.cpp HIP：`ggml-hip.dll` 21:37、`llama-cli-impl.dll` 21:55 | build 目录 mtime |
| 21:59:39 | BF16 GGUF 转换完成（53.8 GB） | convert.log |
| 22:04:02 | **Q4_K_M 量化完成**（16,547,400,192 B = 15.40 GiB，4.92 BPW，851 个张量） | quantize.log |
| ~22:04–22:05 | `llama-bench` 跑 27B Q4_K_M 成功，**12.2s**，无新增 4101/41 | 上一会话输出（**未落盘**） |
| ~22:05–22:07 | `llama-cli` 跑同一 27B Q4_K_M → **整机瞬断**（工具侧只看到会话 `aborted`，无任何输出） | 本次事故 |
| 22:07:21 | 开机 | Kernel-General 12 |
| 22:07:23 | **Kernel-Power 41**；22:07:31 **EventLog 6008**（报"上次关机 21:35:51"） | System 日志 |
| 22:10+ | 复核取证：**4101 最后一次仍是 17:24:40**，本次事故零先兆事件 | 本次复核 |

---

## 3. 证据

### 3.1 取证命令（可原样复现）

```powershell
# 崩溃/断电
Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Power';Id=41} -MaxEvents 5 |
  Select TimeCreated, Id, Message | Format-List
Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='EventLog';Id=6008} -MaxEvents 5 |
  Select TimeCreated, Message | Format-List
# 驱动 TDR
Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Display'} -MaxEvents 5 |
  Select TimeCreated, Id, Message | Format-List
# 硬件错误 / 蓝屏
Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-WHEA-Logger'} -MaxEvents 10
Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-WER-SystemErrorReporting';Id=1001} -MaxEvents 5
Get-ChildItem C:\Windows\Minidump
Get-ChildItem C:\Windows\LiveKernelReports -Recurse -File
# 上次会话的收尾（看死前最后一条日志）
Get-WinEvent -FilterHashtable @{LogName='System';EndTime=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime} -MaxEvents 30 |
  Sort TimeCreated
```

### 3.2 本次取证的确定结论

- `Display` 4101：**最后一次是 17:24:40**，本次（~22:06）**没有**新增 4101。
- WHEA-Logger：**0 条**（provider 确实存在，不是查询写错名字）。
- WER 1001（BugCheck）：最近一次是 2026-08-23，本次无。
- `C:\Windows\Minidump`：**空**。`LiveKernelReports`：**空**。
  > **§12 更正**：这台机器的 dump 是**重定向过的**——`HKLM\SYSTEM\CurrentControlSet\Control\CrashControl` 里
  > `DumpFile=E:\Dumps\MEMORY.DMP`、`MinidumpDir=E:\Dumps\Minidump`，且 `AutoReboot=0`。
  > 因此"`C:\Windows\Minidump` 为空"**不能**当作"没有蓝屏"的证据；正确口径是查 `E:\Dumps`
  > （当前只有 2 个用户态 dump，**没有任何内核 dump** ⇒ 与硬断电一致）。
- `Kernel-Power 41` 的 `BugcheckCode` 字段：**30 天逐条核实全为 0** ⇒ 排除"其实是蓝屏被自动重启掩盖"（见 §12.1）。
- Application 日志 21:00–22:08 的 Error/Critical：**0 条**（连应用层报错都没有）。
- 死前最后一条系统日志：**22:03:06**（ToDesk 切电源方案，属正常噪音），之后到开机之间**什么都没有**。

### 3.3 时间戳陷阱（很重要，别被带偏）

6008 报"上次关机时间 21:35:51"，但：

- `convert.log` mtime **21:59:39**
- `quantize.log` mtime **22:04:03**
- `qwen3.8-27b-q4_k_m.gguf` mtime **22:04:02**

即**有文件写入发生在 6008 声称的关机时刻之后**。6008 的时间来自 `LastAliveStamp` 一类的周期性落盘标记，
在这台机器上系统性偏早（20:33 那次同样报 17:09:25，而 4101 在 17:24:40）。

**结论：6008 的时间戳不能用作精确崩溃时刻。**崩溃窗口取 `[最后一条系统日志 22:03:06, 下次开机 22:07:21]`。
教训文件里"6008 记录的上次关机时间可能比实际死亡时刻早数分钟——以最后落盘为准"这条已再次被验证。

### 3.4 缺失的证据（接手后要补）

- **无 GPU 功耗 / 温度 / 时钟日志**（本次事故最关键的缺口）
- **llama-bench 的结果没有落盘**，只有会话内输出；重跑时必须 `Tee-Object` 到文件
- 无 PSU 型号 / 瓦数 / 年龄 / PCIe 供电线数量
- 无是否有 PCIe riser、机箱型号、风道信息
- 未确认系统 dump 配置（硬断电不会有 dump，但要先排除"其实是 BSOD 被自动重启掩盖"）

---

## 4. 故障分型：A 类 vs B 类（排查的分水岭）

| 分型 | 特征 | 已知实例 | 首疑方向 |
|---|---|---|---|
| **A 类** | 先出 4101（驱动 TDR，事件自称"已成功恢复"），随后掉卡或关机；`doctor` 的 `device_count` 从 2 掉到 1 | 09-12 00:06、09-15 17:24 | 驱动/内核路径：hiprtc JIT 内核、paged attention/page table、Q4-on-device GEMV、server teardown |
| **B 类** | **零先兆瞬断**：只有 41 + 6008，无 4101/WHEA/dump；需人工上电，且短时间内再上电会立即再断 | 09-06 两次、09-15 ~22:06 | 硬件层：供电瞬态（OCP/OVP/12V 跌落）、GPU 硬挂、温度、PCIe 链路 |

**两个必须记住的判读陷阱**：

1. **"没有 4101"不能证明 GPU 没挂**。XTX 现在是无头计算卡（显示在核显），而 4101 是显示驱动的 TDR 上报；
   历史记录显示"显示器直连 7900 时重负载必报 4101 + 硬锁；改无头后重负载直接硬挂"。
   所以 B 类只排除了"驱动成功恢复"这条路径，**没有排除 GPU 侧原因**。
2. **A 类和 B 类可能同源**。A 类是驱动还活着时的上报，B 类是驱动连上报都来不及。不能因为它们形态不同就当成两个独立问题。

---

## 5. 结论修正（避免继承错误前提）

之前 LESSON 里"对照更正：2026-09-15 llama.cpp HIP 在同一机器稳定"一节，把 llama.cpp 当成了稳定基线，
并据此把故障收敛到"MachServe 特有路径"。**该结论现在需要降级**：

仍然成立的部分：

- "ROCm/Windows 上 llama.cpp 完全不能用" → **错**，bench 通过实测。
- "模型规模（8B/27B）本身必然崩" → **不成立**。
- MachServe 的 paged 路径确实能触发 **A 类** TDR（有 4101 的实证）→ 这条仍成立。

必须修正的部分：

- **"只有 MachServe 会崩" → 不成立**：llama.cpp 27B Q4 CLI 也把整机干掉了。
- llama.cpp 不是"稳定基线"，只能当作**同后端负载对照**（而且它自己也崩过一次）。

现在能谨慎表述的：本机在**满载 7900 XTX** 时存在**硬件级随机崩机**；MachServe 的 paged/JIT 路径
**额外**能触发 A 类 TDR。这两件事需要分开验证，不能互相解释。

---

## 6. 假设清单与判别实验（按性价比排序）

### H1 · 供电瞬态（PSU OCP/OVP 或 12V 瞬态跌落）——**当前最优先**

- 预测：崩机与"峰值功耗/瞬态"相关，而非与某个特定软件路径相关；降载或限功耗后不崩；
  崩机多发生在负载启动/切换的数十秒内；断电后短时间内再上电立即再断（保护闩锁）。
- 已被 §12 排除的子项：~~8pin daisy-chain~~（用户确认**两根独立线**）、~~瓦数不足~~（用户确认 **850W+**）。
- 剩余子项（按可疑度排序）：
  1. **PSU 个体问题**：老化（电容退化 ⇒ 瞬态响应变差）、个体故障、或小厂虚标瓦数——铭牌型号/年龄仍未知；
  2. **显卡侧 VRM/供电级退化**：卡自身供电模块劣化同样会拉出瞬时过流、触发 PSU 的 OCP；
  3. **市电/电路**：§12.3 的"晚间聚集"现象——若与家中其他大负载共线、或电压偏低，瞬态余量会被吃掉。
- 判别：
  1. 记录 GPU 功耗峰值（需要 HWiNFO64 或 AMD 驱动日志）；
  2. 装 AMD Adrenalin（当前只有驱动、没有 GUI，见 §1）后用 Performance → Tuning 把功耗上限降到 **300W / 250W**，重跑同一命令；
  3. 看 PSU 铭牌型号/瓦数/年龄；用**另一路插座 / 加 UPS** 做市电对照；
  4. 换一颗已知良好的 PSU 复跑（最贵但最决定性）。
- 判定：限功耗后稳定 ⇒ 供电/功耗结论；同一负载随机通过/失败也支持此假设。

### H2 · GPU 硬挂（内核/固件/驱动）导致整机锁死

- 预测：崩机与特定 kernel 序列相关；无头导致没有 4101 上报。
- 判别：用**完全不加载 MachServe 代码**的 HIP 压测复现——llama-bench 多轮、clpeak、
  或自己写一个纯 hipBLAS GEMM 循环。若纯压测也能瞬断 ⇒ 与 MachServe 无关。
- 判定：纯压测瞬断 ⇒ H2/H1 都成立，需靠功耗监控区分二者。

### H3 · 温度（GPU hotspot / VRAM junction / VRM / CPU）

- 预测：崩机与持续时长正相关；ITX 风道差会放大。
- 判别：HWiNFO64 记录 GPU hotspot、VRAM junction、edge、VRM、CPU package power/Tctl，
  看崩机前最后 60s 曲线。
- 判定：远未到 100°C+ 就死 ⇒ 弱化温度假设。

### H4 · PCIe 链路 / 接触 / riser

- 预测：链路训练正常但高负载出现 fatal AER；WHEA 通常应有记录（但硬断电可能来不及落盘）。
- 判别：查 PCIe/WHEA AER 事件；跑 PCIe 压力（反复大块 H2D/D2H 拷贝或 3DMark PCIe 测试）；
  确认是否用 riser、插槽是否插到底。
- 判定：有 AER/链路重训事件 ⇒ 链路问题。

### H5 · MachServe 特有路径（仅在 H1–H4 排除后才做）

- 候选点：hiprtc 运行时内核生命周期、paged attention/page table、Q4-on-device GEMV、
  server teardown/同步、HIP graph 重放（ROCm 6.2/Windows 已知腐化缺陷）。
- 判别：`HIP_LAUNCH_BLOCKING=1`、`AMD_SERIALIZE_KERNEL=3`、`GPU_MAX_HW_QUEUES=1` 逐步缩小；
  以及 contiguous vs paged、f16 vs q4 KV 的单变量矩阵（先 0.5B/tiny 级，**不要**直接上 8B paged）。

---

## 7. 27B llama.cpp 复现手册（本次崩机的完整配方）

### 7.1 环境准备

```powershell
$env:PATH="$env:HIP_PATH\bin;E:\Users\gxh\Documents\GitHub\llama.cpp\build-hip\bin;$env:PATH"
$env:HIP_VISIBLE_DEVICES='0'          # 只暴露 XTX
```

### 7.2 构建（ROCm 6.2 需要 fp8 fnuz 兼容宏，未改上游源码）

```powershell
cd E:\Users\gxh\Documents\GitHub\llama.cpp
$env:PATH="$env:HIP_PATH\bin;$env:PATH"
cmake -S . -B build-hip -G Ninja -DGPU_TARGETS=gfx1100 -DGGML_HIP=ON `
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release `
  '-DCMAKE_CXX_FLAGS=-D__hip_fp8_e4m3=__hip_fp8_e4m3_fnuz'
cmake --build build-hip --target llama-bench llama-cli llama-quantize -j 8
```

> 坑：`CMAKE_HIP_FLAGS` 不生效，必须用 `CMAKE_CXX_FLAGS`。
> `build-hip/CMakeCache.txt` 关键项：`GGML_HIP=ON`、`GPU_TARGETS=gfx1100`、
> `AMDGPU_TARGETS=gfx1100;gfx1036`、`GGML_HIP_GRAPHS=ON`、`GGML_HIP_NO_VMM=ON`。
> llama.cpp 版本：`6011c34`。
> 产物是 exe + dll 成对（`llama-cli.exe` 只有 9.7KB，实际逻辑在 `llama-cli-impl.dll`），
> **运行时必须带全 `build-hip\bin` 到 PATH**，否则找不到 dll。

### 7.3 模型转换与量化（已完成，产物可复用）

```powershell
# 源：.models\qwen3.8-27b\（18 shards / 51.75 GiB / Qwen3_5ForConditionalGeneration / qwen3_5）
$env:PYTHONPATH="E:\Users\gxh\Documents\GitHub\llama.cpp\gguf-py"
python convert_hf_to_gguf.py .models\qwen3.8-27b --outtype bf16 `
  --outfile .models\qwen3.8-27b\gguf\qwen3.8-27b-bf16.gguf --no-nextn

E:\Users\gxh\Documents\GitHub\llama.cpp\build-hip\bin\llama-quantize.exe `
  .models\qwen3.8-27b\gguf\qwen3.8-27b-bf16.gguf `
  .models\qwen3.8-27b\gguf\qwen3.8-27b-q4_k_m.gguf Q4_K_M
```

| 产物 | 路径 | 大小 |
|---|---|---|
| BF16 | `.models\qwen3.8-27b\gguf\qwen3.8-27b-bf16.gguf` | 53,808,282,112 B（50.11 GiB） |
| Q4_K_M | `.models\qwen3.8-27b\gguf\qwen3.8-27b-q4_k_m.gguf` | 16,547,400,192 B（15.40 GiB） |

量化报告：851 张量、`model size 51305.09 MiB (16.00 BPW)` → `quant size 15770.35 MiB (4.92 BPW)`、
耗时 256.6s。日志：`convert.log` / `quantize.log`（同目录）。

### 7.4 通过的命令（bench）

```powershell
llama-bench.exe -m .models\qwen3.8-27b\gguf\qwen3.8-27b-q4_k_m.gguf -p 64 -n 8 -r 1 -ngl 99
# 结果：pp64 537.47 t/s，tg8 36.70 t/s，12.2s 完成，跑后无新增 4101/41，device_count=2
```

### 7.5 崩机的命令（cli）

```powershell
llama-cli.exe -m .models\qwen3.8-27b\gguf\qwen3.8-27b-q4_k_m.gguf `
  -ngl 99 -c 2048 -n 48 -st -p "用三句话说明为什么GPU上的矩阵乘法比CPU快。" --no-warmup
# 结果：整机瞬断。工具侧只看到会话 aborted，无任何输出
#       ⇒ 不知道崩在权重上传 / KV 分配 / 首次 forward / 生成哪一步
```

> 注意：本版本 llama.cpp **没有 `-no-cnv`**（会报 `error: invalid argument: -no-cnv` 并立刻退出），
> 非交互单轮用 `-st/--single-turn`，且 `-p` 预置首轮时不会进交互。

### 7.6 bench 与 cli 的差异（若怀疑"软件特定路径"，这就是变量表）

| 维度 | llama-bench | llama-cli |
|---|---|---|
| 上下文 | 模型默认 | `-c 2048`（显式分配 KV） |
| prompt | 合成随机 token，`-p 64` | 中文真实 prompt + chat template（带 system 段） |
| 生成 | `-n 8` | `-n 48` |
| 采样 | 无采样（只测速） | 默认采样链（top-k/top-p/temp/min-p） |
| warmup | bench 自身流程 | **`--no-warmup`（首次真实 forward 即大 batch）** |
| 持续时间 | ~12.2s | 更长（模型加载 mmap + prefill + 48 token） |
| 退出 | 正常退出 | 未走到退出（整机断电） |

其中"没有 warmup"和"持续更久"这两条与**功耗/温度**假设直接相关，是最该先怀疑的两条。

---

## 8. 真机安全协议（强制；违反可能赔掉一整天甚至损坏硬件）

> **已自动化（2026-09-15 深夜）**：`tools/gpu_guard.ps1` 把下面的协议变成机器检查——
> `pre`（doctor 校验 device_count / XTX 是否在枚举里 + 距上次硬断电的静置窗口，通过才写下时间戳标记）
> → 跑 arm → `post`（对比标记以来的 4101 / 41 / 6008 / WHEA + 二次 doctor 比对 device_count，异常退出码 1）。
> 状态与日志写 `target/gpu-guard/`。**每条 GPU arm 前后都必须跑，不要靠人记。**
> 当晚实测：`pre` 正确拒绝（距最近一次硬断电 20 分钟 < 60 分钟门槛）。

1. **单负载串行**：GPU 推理/基准/模型加载期间，禁止并发 `cargo build`/clippy/第二个推理/大文件拷贝。
   （历史：满载 GPU + 并行编译叠加触发硬断电。）
2. **每条 arm 前**：`mach-server doctor` 必须 `device_count=2` 且 `gpu[0]` 是 RX 7900 XTX；
   否则**禁止任何 GPU 负载**。
3. **每条 arm 后**：立刻查 4101 / Kernel-Power 41 / 6008，并再跑一次 doctor 比对 `device_count`。
   `device_count=1` ⇒ 判定 dGPU 不可用，当天停止所有 GPU 负载。
4. **arm 之间 ≥10 分钟观察窗**；任何异常（TDR/掉卡/断电）当天剩余时间一律停 GPU。
5. **硬断电后**：静置 ≥1 分钟再上电；恢复后 **≥1 小时内的 GPU 计算结果不可信**（历史曾出现乱码 + 再挂死）。
6. **优雅退出**：不要用脚本 `terminate()` 杀 server（历史多次崩在 teardown）。
7. **每条 arm 前先写下**"这条 arm 要判别的假设 + 预测结果"，避免跑完才找解释。
8. 远程管理（ToDesk）的机器硬断电 = 彻底失联，满载操作前按最保守口径决策。

---

## 9. 建议的排查顺序

**第一小时（零/低风险）**

1. 问清硬件：PSU 型号/瓦数/年龄、PCIe 供电线是几根、是否用 riser、机箱与风道、是否有 UPS。
2. 装 **HWiNFO64**（免费）开传感器日志：CSV、1–2s 采样，记录
   `GPU Power / GPU Hotspot / GPU Memory Junction / GPU Edge / VRM / CPU Package Power / Clocks / 风扇转速`。
3. 顺带确认没有第二个人/会话在用这台机器（ToDesk 远程面板、其他 agent 会话）。

**第一天（受控阶梯负载）**

4. 复跑 `llama-bench` 27B 3 轮（每轮之间 ≥10 分钟观察窗），**全程落盘日志**
   （`... 2>&1 | Tee-Object bench-<n>.log`）。
5. 若复现 → 立刻查崩机前最后 60s 的功耗/温度曲线，记录峰值。
6. 把 GPU 功耗上限降到 **250W** 重跑同一命令；仍崩再降到 200W。**找阈值**。
7. 若限功耗后稳定 ⇒ 供电/功耗结论，转长期策略（限功耗常开 / 换 PSU / 换独立供电线）。
8. 若限功耗后仍崩，且**纯 HIP 压测（不含 MachServe）也崩** ⇒ 转 H2/H3/H4。

**只有在上面全部指向"非硬件"之后**

9. 才回到 MachServe 的 A 类路径定位（单变量矩阵、`HIP_LAUNCH_BLOCKING=1` 等），
   并且**从 0.5B/tiny 起步**，不再直接跑 8B paged 性能矩阵。

---

## 10. 相关记录索引

仓库文档：

- `docs/benchmark-results-paged-prefix.md` —— 已有的 arm 安全约束（≥10 分钟观察窗、查 4101/device_count）
- `docs/roadmap.md` §「Q4 KV 真机 parity（#192，2026-09-15）」—— 已有的真机验证记录
- `docs/moe-offload.md` §「环境注意」—— "显示器直连 7900 时持续 GPU 负载会触发 TDR" 的历史观察

经验文件（`~/.agents/rules/`）：

- `LESSON_满载GPU推理与并行编译叠加触发整机硬断电须单负载串行.md`（含 09-06 实例、硬断电后处置）
- `LESSON_重GPU真机测试完成后仍可能触发整机崩溃需留观察窗口.md`（含 09-12、09-15 实例；**§对照更正需要按本文 §5 修订**）
- `LESSON_ROCm6.2Windows的HIP大图重放数千次后静默腐化.md`

issue：`#192`（Q4 KV 性能矩阵，仍 open）、`#168`。

llama.cpp：`E:\Users\gxh\Documents\GitHub\llama.cpp` @ `6011c34`，`build-hip`（配置见 §7.2）。

---

## 11. 需要用户确认的问题（接手者可直接问）

**已答复（2026-09-15 深夜，见 §12.4）**：

1. ~~PSU 的型号、瓦数、购买/使用年限？~~ → **≥850W**（型号与年龄仍需现场看铭牌）
2. ~~7900 XTX 的 2 个 8pin 是两根独立线，还是一根 daisy-chain？~~ → **两根独立线**
6. ~~之前"850W+"这个数字是否可靠？~~ → **可靠，用户确认 850W 以上**

**仍待确认**：

3. 机箱型号？是否用 PCIe riser？显卡插槽是否插到底？8pin 接头有没有发黄/变形？
4. 是否允许安装 HWiNFO64 做功耗/温度日志？（当晚用户选择：**先做零 GPU 风险项，暂不装**）
5. 这台机器当前是否有其他会话/人正在使用（ToDesk、其他 agent）？
7. **PSU 铭牌**：品牌、型号、额定瓦数、+12V 联合输出、购买年份？
8. 这台机器是否与家中其他大功率电器**共线**？晚间电压是否偏低？（对应 §12.3 的傍晚聚集）
9. 能否借到一颗**已知良好的 PSU** 做对照实验？
10. 主板 BIOS 里 "Restore on AC Power Loss / 断电恢复" 设成了什么？（§12.2 推断为**开启**）

---

## 12. 2026-09-15 深夜第二次复核（本次新增）

> 触发：当晚 ~22:06 又一次整机瞬断（当月第 30 次）。
> 本次复核**只做零 GPU 风险的取证**：没有跑任何 GPU 负载、没有改注册表、没有重启。
> §12.5 的命令可原样复现全部数字。

### 12.1 决定性证据：30 次断电没有一次是蓝屏

`Kernel-Power 41` 事件带 `BugcheckCode` 字段，是区分"硬断电"与"蓝屏"的关键：

| 时间 | BugcheckCode | PowerButtonTimestamp |
|---|---|---|
| 09-15 22:07:23 | 0 | 非 0（粘滞值，见下） |
| 09-15 20:33:55 | 0 | 0 |
| 09-13 22:29:07 | 0 | 0 |
| 09-12 18:08 / 12:09 / 11:16 / 10:40 / 10:28 | 均 0 | 均 0 |
| 08-20 ～ 09-09 共 22 条 | 均 0 | 均 0 |

- **30 条全为 0** ⇒ 没有一次是蓝屏/内核崩溃。配合 `AutoReboot=0`（真蓝屏会停在蓝屏界面等人看），
  "其实是 BSOD 被自动重启掩盖"这条可能性**彻底关闭**。
- 30 天里唯一的真蓝屏是 **08-23 14:53**（`WER-SystemErrorReporting 1001` 与 41 同秒级出现），与其余 29 次不同源。
- `PowerButtonTimestamp` 几乎全为 0 ⇒ 这些关机**不是人按电源键强关的**。
  （09-15 22:07 那条非 0，按 FILETIME 换算约等于 20:56，即 20:55:43 那次重启之前按的键；该字段语义上是
  "最近一次电源键事件"，**会粘滞**，不能用来判定当次是谁关的机。）

### 12.2 机器断电后会自己起来（此前记录的"必须人工上电"有误）

把"开机"（`Kernel-General 12`）与"41"对齐（41 在**下次开机**时才记录，描述的是**上一次**关机）：

- 08-24：开机 20:31:33 → 20:39:04 → 20:42:44 → 20:48:59，间隔 **7.5 / 3.7 / 6.2 分钟**，每次开机都配一条 41。
- 09-06：开机 13:18:51 → 13:25:57 → 13:26:43，间隔 **7.1 / 0.8 分钟**。
- 09-12：一天之内 9 次开机（08:00 ～ 18:08）。

没有人能每隔 4 分钟手动按一次电源键，何况这是一台 ToDesk 远程管理的机器 ⇒
**BIOS 的"断电恢复后自动开机"是开着的**：机器崩完会自己回来、然后很快再崩。两个操作后果：

1. **"必须人工上电"不能再当判据**（原文 §0 与 §8.5 的措辞已按此修订）。
2. **崩机后自动上电再崩是常态**，不能当成"保护闩锁需要静置"的独立证据——两种解释都成立。

### 12.3 时间分布明显偏向傍晚/夜间

30 次里 **16 次**落在 20:00–23:30（20:13、20:14、20:20、20:31、20:32、20:39、20:42、20:49、
21:09、21:24、21:34、21:39、20:33、22:07、22:57、23:23）。两种解释目前**无法区分**：

- (a) 用户本来就晚上跑 GPU（工作习惯）；
- (b) **市电/共线负载**在晚间吃掉瞬态余量。

区分需要 §6 H1 判别里的"换一路插座 / 加 UPS"对照实验，并记录崩机时刻家里其他大功率电器的状态。

### 12.4 用户答复（缩小了假设空间）

| 问题 | 答复 | 后果 |
|---|---|---|
| 8pin 接法 | **两根独立线** | H1 的 daisy-chain 子项**排除** |
| PSU 瓦数 | **确认 850W 以上** | H1 的"瓦数不足"子项**排除**（个体老化/故障仍未排除） |
| 下一步 | **先做零 GPU 风险项，不动 GPU** | 本次不装 HWiNFO / 不装 Adrenalin / 不跑复现 |

### 12.5 本次复核新增的取证命令

```powershell
# BugcheckCode（区分硬断电 vs 蓝屏的关键字段）
Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Power';Id=41} -MaxEvents 8 |
  ForEach-Object { $x=[xml]$_.ToXml(); $d=@{}; $x.Event.EventData.Data | ForEach-Object { $d[$_.Name]=$_.'#text' }
    [PSCustomObject]@{ Time=$_.TimeCreated; Bugcheck=$d['BugcheckCode']; PwrBtn=$d['PowerButtonTimestamp'] } }
# dump 的真实落点（本机被重定向到 E:\Dumps）
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl' | Select CrashDumpEnabled,DumpFile,MinidumpDir,AutoReboot
Get-ChildItem E:\Dumps -Recurse
# 开机/断电时间轴（41 在下次开机时才记录）
Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-General';Id=12;StartTime=(Get-Date).AddDays(-30)} |
  Select -Expand TimeCreated | Sort-Object
# 一键状态 + 门禁（本仓库）
pwsh -File tools/gpu_guard.ps1 status
pwsh -File tools/gpu_guard.ps1 pre      # 跑 GPU arm 前
pwsh -File tools/gpu_guard.ps1 post     # 跑完后
```

### 12.6 本次复核排除/弱化的东西（省下后续时间）

- **蓝屏**：§12.1，排除。
- **内存/CPU 不稳（EXPO 之类）**：30 天应用崩溃只有 7 条，全是 explorer / WindowsTerminal / powershell 的零散
  `c0000005`，**没有**"随机不同进程成片崩溃"这种内存不稳的典型形态 ⇒ 弱化，不作主线。
- **软件路径**：§0 的密度论证——一个月 30 次，不随代码版本变化 ⇒ 排除。
- **daisy-chain / 瓦数不足**：§12.4，排除。

### 12.7 复核后的状态与处置

- `tools/gpu_guard.ps1` 已落地（§8 协议的自动化）。当晚实测 `pre` 正确拒绝：距最近一次硬断电 20 分钟 < 60 分钟门槛。
- GPU 当前状态：`device_count=2`、`gpu[0]=RX 7900 XTX`、23.84 GiB free —— 本次崩机后 XTX **没有**
  像 17:24 那次掉出 HIP 枚举。
- 未跑 GPU 负载、未改注册表、未重启。
- 下一步取决于 §11 剩余问题的答复；**硬件面结论出来之前，不再用这台机器做任何性能矩阵。**

---

## 13. Adrenalin 完整安装落地 + "仅驱动之谜"的根因（2026-09-15 22:51）

**目的**：装 GUI **不是为了升驱动**——装的是**同一个 26.8.1**，只为拿到 §6 H1 判别实验需要的两件工具：
Performance → Tuning 的功率/频率上限，和 Metrics 的功耗/温度带时间戳日志（§3.4 最大证据缺口）。

### 13.1 上次（09-15 22:42）那次"装好了"其实什么都没装

安装器启动后按 **install type 4 = 仅驱动程序** 走：`CUIManager::isDriverOnlyInstall :: isDriverOnlyInstall::1`（`C:\Program Files\AMD\CIM\Log\Install.log`）。
该类型下包清单被过滤成**只剩驱动**（Install Manager MSI + Display INF + HDMI/Streaming Audio + SoundWire ×2），
**RSX 整包被丢弃**，日志收尾 `InstallMan::isRSXInstalled : RSX package not installed`。驱动版本没变 ⇒ 等于零。

而 `4` 来自注册表 `HKLM\SOFTWARE\AMD\CN\6CD6F123-9AE3-4A7D-A182-D90051B6B327 = 4`——
安装器二进制里就有 `GetRSXDefaultViewFromINF`（`HKR,,6CD6F123-…,.*,(.*)`），它就是**记住的"上次选的安装类型"**。
08-25 那次选过「仅驱动程序」后就一直被继承。INF 自带的默认值是 `1`（显示类键 `0005`/`0006` 里都是 1）。
值含义：**1=完整安装 / 2=最小安装 / 4=仅驱动程序**（二进制里还有对应开关 `-View:1/2/4`、`-DefView=1/2/4`、`-DriverOnly`、`-AllowedView`）。
包本身是全量的：`adrenalin-26.8.1-win11-b.exe` 内 `Packages\Drivers\Display\WT6A_INF\B026373\ccc2_install.exe`（242 MB，CCC2=RSX 本体）。

### 13.2 修法（可复现，无需碰驱动）

`.scratch/install_adrenalin_full.ps1`：把上面那个值从 4 改回 1（先备份，`-Revert` 可回滚），再提权启动安装器。
改完后日志变成 `isDriverOnlyInstall::0`，包清单里出现 `[AMD Settings] ccc-next64.msi`（GUI 本体）、Branding64、DVR64、WVR64、RyzenMasterSDK、AUEP64。
中文界面：若「安装类型」仍是「仅驱动程序」，点开**「附加选项」**→ **「安装类型」** 改 **「完整安装」**（会弹"切换安装类型可能会导致…"警告，确认继续）。

### 13.3 装后核实（22:51，重启前）

- `InstallMan::isRSXInstalled : RSX package installed`（×3）；`C:\Program Files\AMD\CNext\CNext\RadeonSoftware.exe` 存在（29 MB）。
- Radeon Software 26.10.41.01，与驱动内部版本同 build。
- **驱动版本没动**：XTX 32.0.31041.1004、核显 32.0.21045.5002；卸载列表 `AMD Software 26.8.1`。
- `PendingFileRenameOperations` 16 → 28（驱动文件排队替换）⇒ **需重启**（无 GPU 负载，安全）。
- 预期坑：7900 XTX（TBP 355 W）Adrenalin 功率上限滑条消费级区间通常只有 −10% ~ +15%（≈320 W 起），
  可能压不到 250 W；届时改用同一页的**「最大频率」上限**做降载（效果等价，同样安全）。
  更深的限功耗要走 MorePowerTool 改 PowerPlay 表（改 vBIOS 层，风险等级不符，不主动动）。

---

## 14. 装完 GUI 当天 XTX 静默掉卡（2026-09-15 23:02，A 类无 4101 变体）

**时间线**：22:54:22 重启完成 → 22:56:46 `gpu_guard pre` **device_count=2**、XTX 23.84 GiB free（门禁拒绝仅因距 22:07 硬断电 49 分钟 < 60）→ 22:57 首次启动 RadeonSoftware.exe，用户进入界面 → **~23:00 用户在 GUI 里看到 GPU1 掉了** → 23:02:40 `gpu_guard status` device_count=1（只剩核显）。

**取证**：
- 掉卡窗口（22:57–23:02）**零事件**：无 4101、无 41、无 WHEA、无任何 Kernel-PnP surprise-removal——比 A 类还安静。
- 设备管理器：XTX `ProblemCode 31`（`STATUS_UNSUCCESSFUL` 0xC0000001，驱动启动失败）；PCIe 链路仍 Gen4 x16 满速 ⇒ 卡在总线上，是**驱动初始化失败**，不是掉链路。
- 恢复尝试：关 GUI + `pnputil /restart-device` → "Device restarted successfully"、ProblemCode 归 0、WDDM 正常，**但 HIP 仍只枚举核显**——WDDM 活、计算通道死，驱动级重启救不回。

**定性**：GUI 首次启动（很可能在 Tuning 页触发驱动重载）在这张病卡上把驱动重初始化打挂了——与 §4 记录的"09-15 20:33 冷复位后一度掉出 HIP 枚举"同族。不是 Radeon Software 本身有问题，是**卡已经经不起驱动重载**——这本身又是一条指向 GPU 侧硬件退化的证据。

**处置**：按 §8.3（device_count=1 ⇒ 当天停止一切 GPU 负载），当晚的限功耗实验**取消**；恢复路径 = 完整关机 → 静置 ≥60s → 冷启动（09-15 20:33 验证过的路径），回来后仍只做只读核实。

**恢复结果（09-16 00:17）**：关机静置 60s+ 冷启动后 `gpu_guard status` **device_count=2**、XTX 23.84 GiB free，恢复成功（与 09-15 20:33 同路径）。**细化一处推断**：本次开机 RadeonSoftware.exe 随登录自启（00:15:41）而卡无恙 ⇒ "GUI 启动本身"不致死；23:00 的掉卡更可能由"进入 Tuning/性能页触发的驱动重载"或病卡自发性掉卡导致。后续进 Tuning 页前先跑门禁、先开指标日志，把它当一条 arm 对待。

**09-16 上午侦察（零事故，截图存档 .scratch\tuning-*.png）**：
- Tuning 页结构：Manual Tuning → Custom → GPU Tuning Enabled → Advanced Control Enabled（警告 ×2）。
  滑条出厂值：**Min Freq 500 / Max Freq 3065 MHz / Voltage 1150 mV**；Power Tuning 展开后有 **Power Limit (%)**，默认 0。
  **未点 Apply Changes**（点之前这些都不生效）。
- **Adrenalin 指标日志的致命缺陷（实测）**：`记录指标` 的 CSV **不是边采边写**——启动时一次性回填内存缓冲
  （实测回填了约 85 分钟），之后不追加；新会话要等约 80 秒才落第一行。Arm 0 基线（llama-bench，19s 通过，
  pp64 617 t/s）恰好落在这个启动空窗里，**功耗曲线没被采到**。⇒ **整机断电时 Adrenalin 的 CSV 不会有死亡时刻的数据**，
  崩机现场取证不能用它。改用 **GPU-Z 的 Sensors → Log to file**（每秒落盘）做崩溃现场记录，Adrenalin CSV 只当存活 arm 的补充。
- 待读数：Power Limit 滑条最低刻度（拖到头看，先不 Apply）。
- CSV 列序（Hardware.*.CSV）：GPU 1 = 核显（PWR 列实为 APU package 功耗 48-85W 空闲），**GPU 2 = XTX**
  （BRD PWR 空闲 20-21W、edge 47°C、hotspot 54°C、VRAM 结温 69-70°C 空闲偏热、电压 ~515mV 空闲、风扇 0 RPM）。

---

## 15. 第一份崩机现场功耗曲线 + 限功率首次存活（2026-09-16 晚）

> 本月第 31 次崩机（21:40）第一次被逐秒记录。记录工具 = GPU-Z Sensors → Log to file（每秒落盘；
> Adrenalin CSV 已证不能用于崩溃现场，§14）。日志：`.scratch/amd-metrics/gpuz-xtx-20260916-2018-to-2145.log`。

### 15.1 21:40 致死 run（另一 agent 的 GPU benchmark，stock 无限制；门禁 pre 21:24:35 通过）

> 用户 09-16 深夜确认：21:38-21:40 那段持续 75s+ 的满载是**另一个 agent（codex）在跑显卡 benchmark**，
> 不是人工操作。20:36/20:56/21:24 的几次短时 bench 尖峰也归属同一 agent。

爬升段（27B Q4_K_M，显存 16.7 GB 常驻）：

```
21:39:00  2402MHz 0.831V 332W  load=100  memT=74°C   ← 持续满载 330-365W
21:39:06  2396MHz 0.875V 365W                        ← 电压/功率最高点
21:39:12  → 瞬时归零(clk 0 / 0.070V / 81W)            ← 第 1 次坍缩,21:39:18 恢复(1242MHz/0.704V/40W)
21:39:22  1896MHz 0.806V 383W  load=100              ← 恢复后立刻再冲,峰值 383W
21:39:23  → 再次归零                                ← 第 2 次坍缩,又恢复
21:40:16  2330MHz 0.814V 326W  load=100  memT=80°C   ← 第三次回到满载
21:40:17  2331MHz 0.811V 336W            memT=82°C
21:40:18  → 最终坍缩(79MHz/0.070V),之后再没活过来
21:41:42~21:45:07  传感器全 0(GPU 彻底无响应),系统僵死(4.5 分钟只采到 4 个样本)
事件:Display 4101 ×2(21:45:00/21:45:34,系统卡顿导致晚写入,对应两次恢复尝试) → 整机断电,21:58 自启
```

**判读**：
- **温度彻底排除**：死时 edge 54°C / hotspot 67°C / VRAM 82°C，离降阈值（95-100°C）很远。
- 坍缩形态 = **2400MHz/0.87V 满载 → 1 秒内归零**：不是驱动 TDR 超时（驱动两次成功复位），是**卡的供电/SMC 在满载下崩**。
  峰值 383W、持续 365W、电压 0.875V——全在 stock 规格内（TBP 355W），卡"在规格内运行也会死"。
- 850W PSU 带 383W GPU + ~150W 其余 = ~550W，瓦数层面无恙 ⇒ 指向 **PSU 瞬态响应退化（OCP 误触发）或卡侧 VRM 故障**，
  与 §6 H1 排序一致。1 秒采样看不到微秒级瞬态尖峰，这是当前观测极限。

### 15.2 22:54 Arm A：限功率下杀手命令首次存活（1/1）

用户应用（截图 tuning-4.png 确认）：**Max Frequency 2200MHz（帽）+ Power Limit −10%（下限）**。
重跑 09-15 22:04 的杀手命令（`llama-cli 27B Q4_K_M --no-warmup`）：**全程跑完正常退出**（9s，pp 233.8 t/s，tg 38.1 t/s）。
遥测确认约束生效：峰值 **2181MHz / 0.785V / 310W**（stock 是 2413MHz / 0.875V / 365W）——**频率帽是主要约束**（压频率 → 电压点自动降）。

**但证据强度有限**：本次 GPU 满载仅 ~2s（cli 的 prefill+48 token），21:40 致死 run 是 75s+ 持续满载。
**存活 1 次 ≠ 限功率有效**（崩机本来就是随机的）。明日阶梯：① 同设置连跑 5 次 cli；② 同设置跑持续负载（llama-bench -r 3+）；
③ 都稳则逐档抬帽（2400→stock）找死亡阈值。

### 15.3 流程事故（记录，已改）

22:54 的 pre 实际判决是 **REJECTED**（距 21:58 硬断电 56 分钟 < 60），但我把 `pre 2>&1 | tail -4` 用 `&&` 链了下一条命令，
管道把退出码 1 吞成了 tail 的 0，负载在拒绝判决下跑了。**教训：门禁判决绝不允许过管道，必须直读 `$?`/exit code。**
门禁本身按设计工作了（22:54 拒、post 正确报 ANOMALY）。按 §8.3，21:45 出现 4101 + 硬断电后**当天已停止 GPU 负载**。

### 15.4 待用户确认

1. ~~21:38-21:40 那段持续 75s+ 的满载跑的是什么？~~ 已确认：另一 agent（codex）的显卡 benchmark。
2. ~~限功率是 22:07 重启后才 Apply 的？~~ 已确认：Max Freq 2200 + PL −10% 由 codex 应用（用户 09-16 深夜：
   "我只让它把显卡的功率降下来，现在正在验证"）。21:40 致死 run 的 2400MHz/365W = 降功率**之前**的 stock 基线。

### 15.5 暴露的协议漏洞：多 agent 并发无协调

今晚两个 agent 在同一台机器上各自跑 GPU 负载：本 agent 的每条 arm 走 gpu_guard pre/post，
但 codex 的 benchmark **不经过门禁**（pre 21:24:35 是本 agent 的标记，codex 的负载落在它的窗口里）。
§8"单负载串行"目前只约束单个 agent 自己，**没有任何机制阻止另一个 agent 同时起 GPU 负载**——
门禁状态文件 `target/gpu-guard/state.json` 是"最后写赢"，两边互不可见。

处置（09-16 深夜，用户确认）：**codex 不在本仓库干活**（任务是降功率 + 验证，不走 gpu_guard，锁也拦不住它）⇒
跨 agent 协调走**人工**：codex 验证期间本 agent 停止一切 GPU 负载，用户验证完后通知再恢复实验阶梯。
仓内 gpu_guard 的跨进程锁仅对"未来在本仓干活的第二个 agent"有价值，暂不实现。
**在硬件面结论出来前，这台机器上任何 agent 都不应跑 GPU benchmark**（§8 性能矩阵禁令本就生效）——
codex 当前的验证性 benchmark 是用户特批的唯一例外。

---

## 16. 限功率被证伪：2200MHz/−10% 下再次掉卡（2026-09-16 23:17，第 32 次）

> 本次调查到此**搁置**（用户 09-16 深夜决定）。本节是搁置前的最后一份现场。

codex 降功率验证第二轮 soak（llama-bench pp4096/tg128 20s 与 llama-cli 9s 交替循环；
launcher 日志显示它**自觉遵守了 ≥60 分钟静置窗**，23:00:28 才启动）：

- 23:00 启动 → 连跑 **58 段 / 17.5 分钟**全部 exit=0（pp ~893 t/s，tg ~38.9 t/s，性能与 stock 基本持平）
- **23:17:50-52 死亡**（GPU-Z `gpuz-xtx.log` 3550-3556 行）：
  - 死前 10 秒持续满载 **2161-2198MHz（2200 帽顶格）/ 0.76-0.78V / 240-312W**，edge 59°C / hotspot 71°C / VRAM 86°C
  - 23:17:50 的 312W → 23:17:52 归零（76MHz / 0.655V / 32W → 0.080V）→ 23:17:59 起传感器全零，**无恢复**
  - SEG 59 cli 在 GPU 消失后挣扎 125s 以 exit=-1073741819（0xC0000005）死亡；launcher 还在向死卡发 SEG 60
- **零事件**：无 4101、无 41、无 6008 —— 纯 B 类静默掉卡（与 09-15 23:02 同款），系统存活
- PnP 状态：`CM_PROB_DISABLED`（code 22；昨晚是 code 31，形态不同）

**结论（假设矩阵收敛）**：
1. **限功率不能防止故障**。死亡点在 312W/0.78V/2198MHz（−10% + 频率帽全程生效，遥测证实），
   比昨晚 stock 死亡点（365W/0.87V/2400MHz）低了一档还是死了 ⇒ "功率量级/频率"假设**降级**，
   22:54 的存活 1/1 只是基率运气（崩机本来就是随机的）。
2. 两次死亡曲线形态一致（满载 → 1-2 秒内电压归零），与功率帽无关 ⇒ 矛头集中指向
   **卡侧 VRM/SMC 退化**（或 PSU 微秒级瞬态响应，1s 采样看不到，无法区分）。
3. 温度三次排除（54-60°C edge / 67-71°C hotspot / 82-86°C VRAM）。
4. 恢复仍只能冷启动；当前 device_count=1。

**搁置状态**：GPU 实验阶梯（§15.2 的 ①②③）未执行即终止；待办硬件检查（§11 PSU 铭牌/8pin 变色/共线负载/借 PSU）仍然有效但暂停。
恢复调查时从本节与 §15 读起即可。

**误诊脚注**：手动跑 doctor 时 `MACH_HIP_PATH` 必须指到 ROCm 的 **bin 目录**（`C:\Program Files\AMD\ROCm\6.2\bin`）；漏了 `\bin` 会报 `hiprtc0602.dll: LoadLibraryExW failed`，看起来像"HIP 整个挂了"。`tools/gpu_guard.ps1` 的 `Initialize-HipPath` 已内置正确路径，统一走它。



---

## 15. 2026-09-16 晚：Adrenalin 限功耗实测（限功耗判别实验第一刀）

> 触发：接手会话按 §8 协议做"限功耗判别实验"的第一刀——先把 Power Limit 压到最低并**实测确认它真的降了功耗**，再谈稳定性。
> 本次**没有改注册表、没有重启整机**；两个 arm 都是 llama.cpp HIP（27B Q4_K_M，-ngl 99），gpu_guard pre/post 全绿。

### 15.1 结论（一句话）

**Adrenalin 的 Power Limit 能生效，且 -10% 实测把满载板卡功耗从 ~418W 峰值 / ~350-400W 持续，压到 ~352W 峰值 / ~318-352W 持续（≈ -10~-15%），吞吐只掉 2.4%。** 但 -10% 是滑条下限，约 350W 仍偏高。

### 15.2 A/B 实测（同一命令 `llama-bench -m qwen3.8-27b-q4_k_m -ngl 99 -p 4096 -n 128 -r 1`，GPU-Z 1Hz 落盘）

| 项目 | 不限功耗（0%） | 限功耗（-10%） | 变化 |
|---|---|---|---|
| 峰值 Board Power | **418 W** | **352 W** | -16% |
| 满载持续 Board Power | 353~398 W | 318~352 W | ≈ -11% |
| 峰值 GPU 时钟 | 2511 MHz | 2423 MHz | -3.5% |
| 峰值 Hotspot | 74 °C | 70 °C | -4 °C |
| 峰值 显存结温 | 76 °C | 76 °C | 持平 |
| pp4096 吞吐 | 1007 t/s | 983 t/s | **-2.4%** |
| tg128 吞吐 | 40.08 t/s | 39.57 t/s | -1.3% |
| 结果 | 存活 | 存活 | — |

- 两次 arm 各 ~18s（模型加载 ~11s + 计算 ~8s），`gpu_guard.ps1 post` 报 `no 4101/41/6008/WHEA`、`device_count` 2→2。
- 原始日志：`.scratch/amd-metrics/gpuz-xtx.log`（每秒），arm 输出 `.scratch/amd-metrics/ab-uncapped.log`、`ab-capped.log`。
- 滑条可调范围实测：**-10% .. +15%**（VK_LEFT/RIGHT 逐格驱动、读编辑框回读；最小值夹在 -10 不再下降）。

### 15.3 三个必须记住的坑（下一步自动化直接用）

1. **"点了 Apply Changes 却静默回滚"= 旧 Adrenalin 实例状态已脏。**
   症状：点 Apply 后 Power Tuning 自己变回 Off、编辑框回 0、**注册表零变化、无任何弹窗/事件日志**。
   本次第一次尝试就踩到（该实例从当天早些时候一直在跑，且当天 23:00 在这张卡上出过一次静默掉卡）。
   **处置：`RadeonSoftware.exe` 重启出一个干净实例后，同样操作一次成功。**
2. **判定"Apply 成功"的可靠信号**（不依赖截图/注册表）：
   - ✅ 成功：Apply 后 `Power Tuning` 仍为 On、编辑框仍等于设定值、且 **`Apply Changes` / `Discard Changes` 两个按钮同时消失**（表示没有待应用改动）。
   - ❌ 失败：Apply 后 toggle 被复位成 Off / 编辑框回 0。
3. **Adrenalin 的 UIA ClassName 后缀每次进程重启都会变**（本次从 `MainDesktopWindow_QMLTYPE_46` / `NavigationButton_QMLTYPE_106` 变成 `_QMLTYPE_19` / `_QMLTYPE_91`）。
   脚本**不要写死 `_QMLTYPE_N`**，按 `Name` + class 前缀（`NavigationButton*` / `TextIconButton*` / `Qt*QWindowIcon`）匹配；主窗口用 `EnumWindows` 按标题 `AMD Software*` 重新发现。

### 15.4 数据缺口与下一步

- **-10% 只到 ~350W**：若真因是供电瞬态（OCP/12V 跌落），这个幅度可能不够。§13 设想的第二把刀是同一页 **GPU Tuning → Max Freq 上限**（比如 2000-2200 MHz）做更狠的降载，本次**未做**（需先过告警弹窗）。
- **单次 arm 存活不能证明稳定性**：本机历史上"同样的负载有时通过、有时死"。要有意义必须做**长时 soak**（例如连续 30-60 分钟满载）或多轮 arm，并盯 `.scratch` 里 GPU-Z 日志的死亡点。
- 本次应用后的状态已留在机器上：**Power Tuning = On、Power Limit = -10%、已 Apply（无待应用改动）**。
