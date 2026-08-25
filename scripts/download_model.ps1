# MachServe 模型下载器（PowerShell 7）：多 shard（30B+）支持，resume + 并行 + hf-mirror 回退 + 尺寸校验。
# 用法:
#   pwsh -File scripts/download_model.ps1 -ModelId Qwen/Qwen3-30B-A3B -OutDir .models/qwen3-30b-a3b -Parallel 4
param(
    [string]$ModelId = "PrimeIntellect/qwen3-moe-tiny",
    [string]$OutDir = ".models",
    [int]$Parallel = 4,
    [switch]$MirrorOnly
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Discover the file list from the (mirror) API; fall back to a fixed shard list.
$files = @()
foreach ($api in @("https://hf-mirror.com/api/models/$ModelId", "https://huggingface.co/api/models/$ModelId")) {
    try {
        $j = (Invoke-RestMethod -Uri $api -TimeoutSec 30 -ErrorAction Stop)
        $files = @($j.siblings | ForEach-Object { $_.rfilename } | Where-Object { $_ -like "*.safetensors" -or $_ -like "*.json" -or $_ -like "*tokenizer*" -or $_ -like "*.jinja" })
        if ($files.Count -gt 0) { Write-Host "discovered $($files.Count) files via $api"; break }
    } catch { }
}
if ($files.Count -eq 0) {
    Write-Warning "API discovery failed; assuming config.json + model-00001..00016 shards"
    $files = @("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    for ($i = 1; $i -le 16; $i++) { $files += ("model-{0:D5}-of-00016.safetensors" -f $i) }
}

function Get-Len([string]$url) {
    try {
        $r = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 30 -ErrorAction Stop
        return [long]$r.Headers['Content-Length']
    } catch { return -1L }
}

$results = $files | ForEach-Object -Parallel {
    $f = $_
    $OutDir = $using:OutDir; $ModelId = $using:ModelId; $MirrorOnly = $using:MirrorOnly
    $dst = Join-Path $OutDir $f
    function Get-Len([string]$url) {
        try {
            $r = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 8 -ErrorAction Stop
            return [long]$r.Headers['Content-Length']
        } catch { return -1L }
    }
    $mirror = "https://hf-mirror.com/$ModelId/resolve/main/$f"
    $primary = "https://huggingface.co/$ModelId/resolve/main/$f"
    $expect = Get-Len $mirror
    if ($expect -gt 0 -and (Test-Path $dst) -and (Get-Item $dst).Length -eq $expect) {
        return "OK(already):$f"
    }
    $urls = @()
    if (-not $MirrorOnly) { $urls += $mirror }
    $urls += $primary
    foreach ($u in $urls) {
        for ($try = 1; $try -le 20; $try++) {
            & curl.exe -L -C - --retry 5 --retry-all-errors --max-time 1200 -sS -o $dst $u 2>$null
            $code = $LASTEXITCODE
            $cur = if (Test-Path $dst) { (Get-Item $dst).Length } else { 0 }
            if ($code -eq 33 -and $cur -gt 0) { return ("OK(complete):{0}:{1}" -f $f, $cur) }
            if ($code -eq 0 -and $cur -gt 0 -and ($expect -le 0 -or $cur -eq $expect)) {
                return ("OK:{0}:{1}" -f $f, $cur)
            }
            if ($try -lt 20) { Start-Sleep -Seconds 3 }
        }
    }
    "FAIL:$f"
} -ThrottleLimit $Parallel

$ok = @($results | Where-Object { $_ -like "OK*" }).Count
$fail = @($results | Where-Object { $_ -like "FAIL*" }).Count
$results | ForEach-Object { Write-Host "[done] $_" }
Write-Host "download complete: ok=$ok fail=$fail"
if ($fail -gt 0) { exit 1 }
