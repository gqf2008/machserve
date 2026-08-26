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
$mirror = "https://hf-mirror.com/$ModelId/resolve/main"
$primary = "https://huggingface.co/$ModelId/resolve/main"
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
            Write-Host ("sha MISMATCH {0} got {1}.. want {2}..; redownload" -f $name, $h.Substring(0,8), $sha.Substring(0,8))
            Remove-Item $dst -Force -ErrorAction SilentlyContinue
            continue
        }
        $attempt = $attempt + 1
        if ($attempt -gt 60) { Write-Host ("GAVE UP {0}" -f $name); exit 1 }
        $urls = @($mirror, $primary)
        foreach ($u in $urls) {
            & curl.exe -L -C - --retry 5 --retry-all-errors --max-time 300 -sS -o $dst $u
            if ($LASTEXITCODE -eq 33) { Remove-Item $dst -Force -ErrorAction SilentlyContinue }
            $c2 = if (Test-Path $dst) { (Get-Item $dst).Length } else { 0 }
            if ($c2 -ge $want) { break }
        }
        Start-Sleep -Seconds 3
    }
}
Write-Host "ALL SHARDS DONE + sha verified"


