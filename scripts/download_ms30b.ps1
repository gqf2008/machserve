param(
    [string]$Records = "",
    [string]$OutDir = ".models/qwen3-30b-a3b",
    [string]$ModelId = "Qwen/Qwen3-30B-A3B"
)
$ErrorActionPreference = "Continue"
if ($Records -eq "") { $Records = Join-Path $env:TEMP "q3-shards.txt" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$lines = Get-Content $Records
if (-not $lines) { Write-Host "no records at $Records"; exit 1 }
$base = "https://modelscope.cn/models/$ModelId/resolve/master"
foreach ($line in $lines) {
    $parts = $line.Split("`t")
    $name = $parts[0]
    $want = [long]$parts[1]
    $sha = $parts[2]
    $dst = Join-Path $OutDir $name
    $attempt = 0
    while ($true) {
        $cur = if (Test-Path $dst) { (Get-Item $dst).Length } else { 0 }
        if ($cur -ge $want) {
            $h = (Get-FileHash $dst -Algorithm SHA256).Hash.ToLower()
            if ($h -eq $sha) { Write-Host ("OK {0} ({1:N1} GB) sha ok" -f $name, ($want/1GB)); break }
            Write-Host ("sha MISMATCH {0}; redownload" -f $name)
            Remove-Item $dst -Force -ErrorAction SilentlyContinue
            continue
        }
        $attempt = $attempt + 1
        if ($attempt -gt 60) { Write-Host ("GAVE UP {0}" -f $name); exit 1 }
        $u = $base + "/" + $name
        & curl.exe -L -C - --retry 5 --retry-all-errors --max-time 300 -sS -o $dst $u
        if ($LASTEXITCODE -eq 33) { Remove-Item $dst -Force -ErrorAction SilentlyContinue }
        $c2 = if (Test-Path $dst) { (Get-Item $dst).Length } else { 0 }
        if ($c2 -lt $want) { Start-Sleep -Seconds 3 }
    }
}
Write-Host "ALL SHARDS DONE + sha verified"
