#Requires -Version 5.1
<#
.SYNOPSIS
    GPU 真机负载安全门禁：把 docs/rocm-windows-hard-poweroff-investigation.md §8 的人工安全协议
    变成机器可执行的检查，避免"跑完才找解释"。

.DESCRIPTION
    子命令：
      pre     跑 GPU 负载前：doctor 校验 device_count / XTX 是否在枚举里，并检查距上次硬断电的静置时间；
              通过后写下时间戳标记
      post    跑完后：对比标记以来的 Display 4101 / Kernel-Power 41 / EventLog 6008 / WHEA，
              再查一次 doctor 并比对 device_count，异常时退出码 1
      status  随时：打印当前 doctor 状态 + 最近崩溃清单（不写标记）

    退出码：0 = 允许 / 无异常；1 = 拒绝 / 发现异常。
    状态与日志写在 target/gpu-guard/（target 已 gitignore）。

.EXAMPLE
    pwsh -File tools/gpu_guard.ps1 pre
    # ... 跑一条 GPU arm（期间禁止并发编译 / 第二个推理 / 大文件拷贝）...
    pwsh -File tools/gpu_guard.ps1 post
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('pre', 'post', 'status')]
    [string]$Phase,

    # mach-server 可执行文件；默认 target\release\mach-server.exe
    [string]$ServerExe = '',

    # 硬断电后的静置要求：距最近一次 Kernel-Power 41 不足这么多分钟则拒绝跑 GPU
    # （doc §8.5：恢复后 ≥1 小时内 GPU 计算结果不可信）
    [int]$QuietMinutes = 60,

    # post 阶段提示的观察窗口
    [int]$ObserveMinutes = 10,

    # 只做日志比对、不调用 doctor（用于零 GPU 触碰的演练）
    [switch]$SkipDoctor
)

$ErrorActionPreference = 'Stop'

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$StateDir  = Join-Path $RepoRoot 'target\gpu-guard'
$StatePath = Join-Path $StateDir 'state.json'
$LogPath   = Join-Path $StateDir 'guard.log'

function Write-Head([string]$t) { Write-Host ""; Write-Host "== $t" -ForegroundColor Cyan }
function Write-Ok([string]$t)   { Write-Host "  [OK]   $t" -ForegroundColor Green }
function Write-Bad([string]$t)  { Write-Host "  [FAIL] $t" -ForegroundColor Red }
function Write-Note([string]$t) { Write-Host "  [WARN] $t" -ForegroundColor Yellow }
function Write-Info([string]$t) { Write-Host "  $t" }

function Resolve-ServerExe {
    if ($ServerExe) {
        if (-not (Test-Path $ServerExe)) { throw "mach-server not found: $ServerExe" }
        return (Resolve-Path $ServerExe).Path
    }
    $cand = Join-Path $RepoRoot 'target\release\mach-server.exe'
    if (Test-Path $cand) { return $cand }
    throw "missing $cand -- run: cargo build -p mach-server --release --features hip"
}

function Initialize-HipPath {
    if ($env:MACH_HIP_PATH) { return }
    $hip = 'C:\Program Files\AMD\ROCm\6.2\bin'
    if (Test-Path $hip) { $env:MACH_HIP_PATH = $hip }
}

# 调 mach-server doctor，解析 device_count / gpu 列表 / vram
function Get-DoctorState {
    if ($SkipDoctor) { return $null }
    Initialize-HipPath
    $exe = Resolve-ServerExe
    $out = (& $exe doctor 2>&1 | Out-String)
    $count = $null
    if ($out -match 'device_count:\s*(\d+)') { $count = [int]$Matches[1] }
    $gpus = @()
    foreach ($m in [regex]::Matches($out, 'gpu\[(\d+)\]:\s*(.+)')) { $gpus += $m.Groups[2].Value.Trim() }
    $vram = ''
    if ($out -match 'vram:\s*(.+)') { $vram = $Matches[1].Trim() }
    [PSCustomObject]@{ DeviceCount = $count; Gpus = $gpus; Vram = $vram; Raw = $out }
}

function Get-GpuEvents {
    param([datetime]$Since, [int]$Max = 50)
    $specs = @(
        @{ Provider = 'Display';                        Id = 4101;  Kind = 'TDR(4101)' },
        @{ Provider = 'Microsoft-Windows-Kernel-Power'; Id = 41;    Kind = 'HARDOFF(41)' },
        @{ Provider = 'EventLog';                       Id = 6008;  Kind = 'DIRTY(6008)' },
        @{ Provider = 'Microsoft-Windows-WHEA-Logger';  Id = $null; Kind = 'WHEA' }
    )
    $res = @()
    foreach ($s in $specs) {
        $ht = @{ LogName = 'System'; StartTime = $Since }
        if ($s.Provider) { $ht.ProviderName = $s.Provider }
        if ($s.Id) { $ht.Id = $s.Id }
        try   { $ev = @(Get-WinEvent -FilterHashtable $ht -MaxEvents $Max -ErrorAction Stop) }
        catch { $ev = @() }
        foreach ($e in $ev) {
            $res += [PSCustomObject]@{ Time = $e.TimeCreated; Kind = $s.Kind; Id = $e.Id }
        }
    }
    # 逗号包装：空数组也要以"一个数组对象"返回,否则调用方拿到 $null,
    # 下游 @($null).Count=1 会把空窗口误判成有事件
    return , @($res | Sort-Object Time)
}

function Write-EventTable($events) {
    $list = @($events) | Where-Object { $null -ne $_ }
    if ($list.Count -eq 0) { Write-Ok 'no 4101 / 41 / 6008 / WHEA in window'; return }
    foreach ($e in $list) {
        $line = '  {0}  {1}  id={2}' -f $e.Time.ToString('MM-dd HH:mm:ss'), $e.Kind, $e.Id
        Write-Host $line -ForegroundColor Red
    }
}

# 返回 $true 表示发现问题（拒绝）
function Test-DoctorGate($doc) {
    if ($null -eq $doc) { Write-Note 'doctor skipped (-SkipDoctor) / not available'; return $false }
    if ($null -eq $doc.DeviceCount) { Write-Bad 'cannot parse device_count from doctor output'; return $true }
    $bad = $false
    if ($doc.DeviceCount -lt 2) {
        Write-Bad "device_count=$($doc.DeviceCount): dGPU not in HIP enumeration -- no GPU load today"
        $bad = $true
    } else {
        Write-Ok "device_count=$($doc.DeviceCount)"
    }
    if ($doc.Gpus.Count -gt 0 -and $doc.Gpus[0] -notmatch '7900 XTX') {
        Write-Bad "gpu[0]=$($doc.Gpus[0]) is not RX 7900 XTX"
        $bad = $true
    } elseif ($doc.Gpus.Count -gt 0) {
        Write-Ok "gpu[0]=$($doc.Gpus[0])  vram=$($doc.Vram)"
    }
    return $bad
}

function Save-State($state) {
    if (-not (Test-Path $StateDir)) { New-Item -ItemType Directory -Path $StateDir -Force | Out-Null }
    $state | ConvertTo-Json -Depth 4 | Set-Content -Path $StatePath -Encoding UTF8
}

function Read-State {
    if (-not (Test-Path $StatePath)) { return $null }
    try { return (Get-Content $StatePath -Raw | ConvertFrom-Json) } catch { return $null }
}

function Add-LogLine([string]$line) {
    if (-not (Test-Path $StateDir)) { New-Item -ItemType Directory -Path $StateDir -Force | Out-Null }
    ('{0}  {1}' -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'), $line) | Add-Content -Path $LogPath -Encoding UTF8
}

# ---------------------------------------------------------------- pre
function Invoke-Pre {
    Write-Head 'GPU guard / pre'
    Write-Info "repo: $RepoRoot"
    $bad = $false

    $doc = Get-DoctorState
    if (Test-DoctorGate $doc) { $bad = $true }

    $ev = Get-GpuEvents -Since (Get-Date).AddDays(-7)
    $last41 = @($ev | Where-Object { $_.Kind -eq 'HARDOFF(41)' })
    if ($last41.Count -gt 0) {
        $mins = [int]((Get-Date) - $last41[-1].Time).TotalMinutes
        if ($mins -lt $QuietMinutes) {
            Write-Bad "last hard power-off was $mins min ago (need >= $QuietMinutes) -- doc 8.5"
            $bad = $true
        } else {
            Write-Ok "last hard power-off was $mins min ago"
        }
    } else {
        Write-Ok 'no hard power-off in the last 7 days'
    }

    if ($bad) {
        Write-Host ''
        Write-Bad 'REJECTED: do not start GPU work'
        Add-LogLine "pre  REJECT"
        exit 1
    }

    $state = [PSCustomObject]@{
        Phase       = 'pre'
        Marker      = (Get-Date).ToString('o')
        DeviceCount = $(if ($doc) { $doc.DeviceCount } else { $null })
        Gpus        = @($(if ($doc) { $doc.Gpus } else { @() }))
        Vram        = $(if ($doc) { $doc.Vram } else { '' })
    }
    Save-State $state
    Write-Head 'PASSED'
    Write-Info 'rule: one GPU load at a time (no cargo build / clippy / 2nd inference / big file copy)'
    Write-Info 'after the run: pwsh -File tools/gpu_guard.ps1 post'
    Add-LogLine "pre  OK  device_count=$($state.DeviceCount)"
    exit 0
}

# ---------------------------------------------------------------- post
function Invoke-Post {
    Write-Head 'GPU guard / post'
    $state = Read-State
    if ($state) {
        $marker = [datetime]::Parse($state.Marker)
        Write-Info ("baseline marker: {0} (pre device_count={1})" -f $marker.ToString('MM-dd HH:mm:ss'), $state.DeviceCount)
    } else {
        $marker = (Get-Date).AddHours(-3)
        Write-Note ("no pre marker, falling back to a 3h window from {0}" -f $marker.ToString('MM-dd HH:mm:ss'))
    }

    $ev = Get-GpuEvents -Since $marker
    Write-Head 'crash / TDR events in window'
    Write-EventTable $ev

    Write-Head 'device comparison'
    $doc = Get-DoctorState
    $bad = $false
    if (Test-DoctorGate $doc) { $bad = $true }
    if ($doc -and $state -and $null -ne $state.DeviceCount -and $null -ne $doc.DeviceCount) {
        if ($doc.DeviceCount -ne [int]$state.DeviceCount) {
            Write-Bad "device_count changed: $($state.DeviceCount) -> $($doc.DeviceCount)"
            $bad = $true
        } else {
            Write-Ok "device_count unchanged ($($doc.DeviceCount))"
        }
    }
    if (@($ev).Count -gt 0) { $bad = $true }

    Write-Host ''
    if ($bad) {
        Write-Bad 'ANOMALY: stop all GPU work for the rest of the day; forensics in doc 3.1'
        Add-LogLine ("post ANOMALY events={0}" -f @($ev).Count)
        exit 1
    }
    Write-Ok "clean. keep >= $ObserveMinutes min observation window before the next arm"
    Add-LogLine 'post OK'
    exit 0
}

# ---------------------------------------------------------------- status
function Invoke-Status {
    Write-Head 'GPU guard / status'
    $doc = Get-DoctorState
    if ($doc -and $null -ne $doc.DeviceCount) {
        Write-Info "device_count=$($doc.DeviceCount)  vram=$($doc.Vram)"
        foreach ($g in $doc.Gpus) { Write-Info "  gpu: $g" }
    }

    Write-Head 'last 10 crash / TDR events (30d)'
    $ev = Get-GpuEvents -Since (Get-Date).AddDays(-30)
    Write-EventTable @($ev | Select-Object -Last 10)

    $c41  = @($ev | Where-Object { $_.Kind -eq 'HARDOFF(41)' }).Count
    $c4101 = @($ev | Where-Object { $_.Kind -eq 'TDR(4101)' }).Count
    $cwhea = @($ev | Where-Object { $_.Kind -eq 'WHEA' }).Count
    Write-Head '30d counts (per-query cap 50)'
    Write-Info "Kernel-Power 41 (hard power-off): $c41"
    Write-Info "Display 4101 (driver TDR):        $c4101"
    Write-Info "WHEA:                             $cwhea"

    Write-Head 'uptime'
    $os = Get-CimInstance Win32_OperatingSystem
    Write-Info ("last boot: {0}   up {1} min" -f $os.LastBootUpTime.ToString('MM-dd HH:mm:ss'), [int]((Get-Date) - $os.LastBootUpTime).TotalMinutes)
    exit 0
}

switch ($Phase) {
    'pre'    { Invoke-Pre }
    'post'   { Invoke-Post }
    'status' { Invoke-Status }
}
