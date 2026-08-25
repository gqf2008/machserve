# MachServe 一键安装（Windows + AMD ROCm）
# 用法: powershell -ExecutionPolicy Bypass -File scripts/install.ps1
# 步骤: 环境自检 -> release 构建 -> 下载 starter MoE 模型(hf-mirror 回退) -> 冒烟基准 -> 指引
param(
    [string]$ModelId = "PrimeIntellect/qwen3-moe-tiny",
    [string]$ModelsDir = ".models",
    [switch]$SkipBuild,
    [switch]$SkipModel,
    [switch]$SkipSmoke
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [ok] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [!!] $msg" -ForegroundColor Yellow }

Write-Step "MachServe 安装自检"
# 1. rust
$rust = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $rust) { Write-Warn "未找到 cargo，请先安装 Rust: https://rustup.rs"; exit 1 }
Write-Ok ("cargo " + (cargo --version).Split(" ")[1])

# 2. git + curl
foreach ($t in @("git", "curl")) {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { Write-Warn "未找到 $t"; exit 1 }
}
Write-Ok "git + curl 就绪"

# 3. GPU 探测
$gpus = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name)
if (-not $gpus) { Write-Warn "未探测到显卡（Win32_VideoController 为空）"; $gpus = @("unknown") }
$gpus | ForEach-Object { Write-Ok ("GPU: " + $_) }
$hasAmd = (($gpus -join " ") -match "AMD|Radeon")
if (-not $hasAmd) { Write-Warn "未检测到 AMD 显卡；MachServe 目前面向 AMD ROCm，NVIDIA 支持尚未就绪" }

# 4. ROCm/HIP 探测: MACH_HIP_PATH 或常见安装路径
$hipPath = $env:MACH_HIP_PATH
if (-not $hipPath) {
    $cand = @("$env:ProgramFiles\AMD\ROCm\*\bin", "C:\Program Files\AMD\ROCm\*\bin")
    $found = $cand | ForEach-Object { Get-ChildItem $_ -ErrorAction SilentlyContinue } | Select-Object -First 1
    if ($found) { $hipPath = $found.FullName }
}
if ($hipPath) { Write-Ok ("ROCm bin: " + $hipPath); $env:MACH_HIP_PATH = $hipPath }
else { Write-Warn "未找到 ROCm/HIP；若已安装请设置 MACH_HIP_PATH 指向 bin 目录" }

# 5. 构建
if (-not $SkipBuild) {
    Write-Step "release 构建 (--features hip)"
    cargo build --release --features hip
    if ($LASTEXITCODE -ne 0) { Write-Warn "构建失败"; exit 1 }
    Write-Ok "构建完成"
}

# 6. 模型下载（hf-mirror 回退）
if (-not $SkipModel) {
    Write-Step ("下载 starter 模型 " + $ModelId + " -> " + $ModelsDir)
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
    $files = @("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")
    foreach ($f in $files) {
        $dst = Join-Path $ModelsDir $f
        if (Test-Path $dst) { Write-Ok ($f + " 已存在，跳过"); continue }
        $urls = @(
            ("https://huggingface.co/" + $ModelId + "/resolve/main/" + $f),
            ("https://hf-mirror.com/" + $ModelId + "/resolve/main/" + $f)
        )
        $ok = $false
        foreach ($u in $urls) {
            Write-Host ("    下载 " + $f)
            curl.exe -L --retry 3 --retry-all-errors -o $dst $u | Out-Null
            if ($LASTEXITCODE -eq 0 -and (Test-Path $dst) -and (Get-Item $dst).Length -gt 0) { $ok = $true; break }
            Remove-Item $dst -ErrorAction SilentlyContinue
        }
        if (-not $ok) { Write-Warn ($f + " 下载失败（网络不稳可重试）"); exit 1 }
        Write-Ok ($f + " 就绪 (" + [math]::Round((Get-Item $dst).Length / 1MB, 1) + " MB)")
    }
}

# 7. 冒烟: MoE offload 基准（加载 + 三种放置对拍）
if (-not $SkipSmoke) {
    Write-Step "冒烟: moe_offload_bench（真实模型 + 放置无关性）"
    $env:MACH_MODELS = (Resolve-Path $ModelsDir).Path
    $env:MACH_MODEL = "model.safetensors"
    $env:MACH_CONFIG = "config.json"
    $env:MACH_MOE_SLOTS = "2"
    $env:MACH_BENCH_TOKENS = "16"
    cargo run -p mach-model --release --features hip --example moe_offload_bench
    if ($LASTEXITCODE -ne 0) { Write-Warn "冒烟失败，请运行: cargo run -p mach-server --features hip -- doctor"; exit 1 }
    Write-Ok "冒烟通过"
}

Write-Step "完成"
Write-Host "下一步:"
Write-Host "  cargo run -p mach-server --release --features hip -- doctor   # 排障 / 环境检查"
Write-Host "  cargo run -p mach-server --release --features hip            # 启动 OpenAI 兼容服务"
Write-Host "  支持矩阵 / 模型选择见 docs/support-matrix.md"
