<#
.SYNOPSIS
  把 walgit 主仓的 refs/heads/* 与 refs/tags/* 镜像到 GitHub(唯一的发版通道)。

.DESCRIPTION
  MachServe 的主仓在自建 walgit,GitHub 只作镜像与发版(其 Issues/Wiki/Projects/Discussions 已关闭)。
  脚本用一个裸仓做缓冲,每个 tick:
    1. 从 walgit fetch heads + tags(带 --prune,分支删除也会同步);
    2. 把 heads(带 --prune)推给 GitHub,再把 tags 推给 GitHub(不 --prune,避免误删 GitHub 上的 tag)。
  refs/collab/*(walgit 的 issue/PR/评审条目)**永不**同步:只推上面两个 refspec,推前还会显式校验。
  仓库里的 refs/remotes/*、notes、pull/* 同样不会出现在这两个 refspec 里。

  GitHub 侧认证走本机 git 凭据(gh 登录 / Git Credential Manager),脚本里不存 token。
  默认只跑一个 tick;常驻用 -Loop(Windows 上可配合计划任务,不要把 -Loop 塞进登录脚本了事)。

.EXAMPLE
  pwsh -File tools/mirror_to_github.ps1 -Once     # 按需镜像一次(发版前)
  pwsh -File tools/mirror_to_github.ps1 -Loop     # 常驻:默认每 60s 一个 tick
#>
[CmdletBinding()]
param(
  [string]$WalgitUrl = "http://127.0.0.1:8081/gqf2008/machserve.git",
  [string]$GithubUrl = "https://github.com/gqf2008/machserve.git",
  [string]$Buffer = (Join-Path $env:USERPROFILE ".walgit\machserve-github-mirror.git"),
  [int]$IntervalSeconds = 60,
  [switch]$Loop
)

$ErrorActionPreference = "Stop"
$script:ExitCode = 0

function Write-Log([string]$Message) {
  Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

function Invoke-Git {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  $out = & git @Args 2>&1
  $code = $LASTEXITCODE
  if ($out) { $out | ForEach-Object { "  $_" } | Write-Host }
  if ($code -ne 0) { throw ("git {0} failed (exit {1})" -f ($Args -join " "), $code) }
}

function Assert-RefspecIsSafe([string]$Refspec) {
  if ($Refspec -match "collab|remotes|notes|pull/") {
    throw "unsafe refspec (never mirror collaboration or remote refs): $Refspec"
  }
}

if ($WalgitUrl -eq $GithubUrl) { throw "WalgitUrl and GithubUrl are the same: $WalgitUrl" }
if ($WalgitUrl -match "github\.com") { throw "WalgitUrl looks like GitHub: $WalgitUrl" }

$HeadsRefspec = "+refs/heads/*:refs/heads/*"
$TagsRefspec = "+refs/tags/*:refs/tags/*"
Assert-RefspecIsSafe $HeadsRefspec
Assert-RefspecIsSafe $TagsRefspec

if (-not (Test-Path (Join-Path $Buffer "HEAD"))) {
  Write-Log "creating buffer repository: $Buffer"
  Invoke-Git init --bare --quiet $Buffer
}

function Sync-Tick {
  Write-Log "fetch  $WalgitUrl -> $Buffer"
  Invoke-Git -C $Buffer fetch --prune --quiet $WalgitUrl $HeadsRefspec $TagsRefspec

  # 双保险:缓冲仓里绝不允许出现 collab 引用被顺带推走。
  $stray = & git -C $Buffer for-each-ref --format='%(refname)' refs/collab 2>$null
  if ($stray) { throw "buffer has refs/collab/* (will not mirror): $stray" }

  Write-Log "push   heads -> $GithubUrl"
  Invoke-Git -C $Buffer push --prune --quiet $GithubUrl $HeadsRefspec

  Write-Log "push   tags  -> $GithubUrl"
  Invoke-Git -C $Buffer push --quiet $GithubUrl $TagsRefspec

  $head = & git -C $Buffer rev-parse refs/heads/master 2>$null
  Write-Log ("ok: master={0}" -f $head)
}

if ($Loop) {
  Write-Log ("mirror loop started (every {0}s) — Ctrl-C to stop" -f $IntervalSeconds)
  while ($true) {
    try { Sync-Tick } catch { Write-Warning $_; $script:ExitCode = 1 }
    Start-Sleep -Seconds $IntervalSeconds
  }
} else {
  Sync-Tick
}

exit $script:ExitCode
