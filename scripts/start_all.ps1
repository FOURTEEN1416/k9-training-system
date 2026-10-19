#Requires -Version 5.1
<#
.SYNOPSIS
    K9 Training Vision System 一键启动编排（P0-2）

.DESCRIPTION
    启动顺序: 预检(.env/密钥占位) → Redis(缺失则拉起) → PostgreSQL 探活 → Alembic 迁移
              → Celery Worker → FastAPI(uvicorn) → 前端 Vite（可选）
    失败时打印缺失项（端口/权重/密钥）与日志路径，最终聚合 /health 组件状态表。

.PARAMETER SkipFrontend
    跳过前端启动（仅后端联调时用）。

.PARAMETER SkipCelery
    跳过 Celery Worker（纯 API 调试时用；视频推理任务将无人消费）。

.PARAMETER HealthTimeoutSec
    /health 全绿轮询超时（默认 90 秒）。

.EXAMPLE
    pwsh scripts/start_all.ps1
    pwsh scripts/start_all.ps1 -SkipFrontend
#>
[CmdletBinding()]
param(
    [switch]$SkipFrontend,
    [switch]$SkipCelery,
    [int]$HealthTimeoutSec = 90
)

$ErrorActionPreference = 'Continue'   # 组件失败需逐个记录，不能全局中止
$Root = Split-Path -Parent $PSScriptRoot          # 仓库根目录
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$LogDir = Join-Path $Root 'logs'
$ApiPort = 8000
$RedisPort = 6379
$PgPort = 5433

# 运行账本: name → @{ ok=$true/$false; note=... }
$script:ledger = [ordered]@{}

# ---------------------------------------------------------------
# 工具函数（须先于调用定义）
# ---------------------------------------------------------------
function Write-Step([string]$msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Fail([string]$msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red }
function Write-Warn([string]$msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

function Save-Ledger([string]$name, [bool]$ok, [string]$note) {
    $script:ledger[$name] = @{ ok = $ok; note = $note }
}

function Show-Summary {
    Write-Step '组件汇总'
    $script:ledger.GetEnumerator() | ForEach-Object {
        $mark = if ($_.Value.ok) { '[OK]  ' } else { '[FAIL]' }
        $color = if ($_.Value.ok) { 'Green' } else { 'Red' }
        Write-Host ("  {0} {1,-10} {2}" -f $mark, $_.Key, $_.Value.note) -ForegroundColor $color
    }
    Write-Host ("日志目录: {0}" -f $LogDir)
}

function Test-RedisUp {
    try {
        $r = redis-cli -p $RedisPort ping 2>$null
        return ($r -match 'PONG')
    } catch { return $false }
}

function Exit-WithSummary([int]$code) {
    Show-Summary
    exit $code
}

# ---------------------------------------------------------------
# 0. 预检：venv / .env / 密钥占位
# ---------------------------------------------------------------
Write-Step '0. 预检（venv / .env / 密钥占位）'

if (-not (Test-Path $VenvPython)) {
    Write-Fail "未找到 venv: $VenvPython —— 先按 README 建虚拟环境"
    Save-Ledger 'preflight' $false 'venv missing'
    Exit-WithSummary 1
}
Write-Ok 'venv 就绪'

$EnvFile = Join-Path $Root '.env'
if (-not (Test-Path $EnvFile)) {
    Write-Fail "缺少 .env —— 复制 backend/.env.example 为 .env 并填真实口令（DATABASE_URL / AUTH_SECRET_KEY）"
    Save-Ledger 'preflight' $false '.env missing'
    Exit-WithSummary 1
}

# 密钥占位扫描（CHANGE_ME 泄漏进 .env 是最常见的起不来原因）
$placeholders = Select-String -Path $EnvFile -Pattern 'CHANGE_ME' |
    ForEach-Object { ($_.Line -split '=', 2)[0] }
if ($placeholders) {
    Write-Fail ".env 仍含占位符: $($placeholders -join ', ') —— 逐项替换后再启动"
    Save-Ledger 'preflight' $false "placeholder keys: $($placeholders -join ', ')"
    Exit-WithSummary 1
}
Write-Ok '.env 就绪（无 CHANGE_ME 占位）'
Save-Ledger 'preflight' $true 'ok'

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ---------------------------------------------------------------
# 1. Redis（未运行则拉起）
# ---------------------------------------------------------------
Write-Step "1. Redis（127.0.0.1:$RedisPort）"

if (Test-RedisUp) {
    Write-Ok 'Redis 已在运行'
    Save-Ledger 'redis' $true 'already running'
} else {
    $redisExe = Get-Command redis-server -ErrorAction SilentlyContinue
    if (-not $redisExe) {
        Write-Fail 'redis-server 不在 PATH（可 choco install redis-64）'
        Save-Ledger 'redis' $false 'redis-server not in PATH'
    } else {
        Start-Process -FilePath $redisExe.Source -ArgumentList '--port', $RedisPort `
            -WindowStyle Hidden -RedirectStandardOutput (Join-Path $LogDir 'redis.log')
        $deadline = (Get-Date).AddSeconds(10)
        while (-not (Test-RedisUp) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 300 }
        if (Test-RedisUp) {
            Write-Ok 'Redis 已拉起'
            Save-Ledger 'redis' $true 'started'
        } else {
            Write-Fail "Redis 拉起失败（10s 无 PONG）—— 见 $(Join-Path $LogDir 'redis.log')"
            Save-Ledger 'redis' $false 'start timeout'
        }
    }
}

# ---------------------------------------------------------------
# 2. PostgreSQL 探活
# ---------------------------------------------------------------
Write-Step "2. PostgreSQL（127.0.0.1:$PgPort）"

$pgUp = Test-NetConnection -ComputerName 127.0.0.1 -Port $PgPort -WarningAction SilentlyContinue |
    Select-Object -ExpandProperty TcpTestSucceeded
if (-not $pgUp) {
    Write-Fail "PG 端口 $PgPort 不通 —— 先启动 PostgreSQL 17 服务（端口须与 .env 一致）"
    Save-Ledger 'postgres' $false "port $PgPort closed"
    Exit-WithSummary 1
}

Push-Location $Root
$pgProbe = & $VenvPython -c "from backend.app.core.config import settings; import psycopg2; c=psycopg2.connect(settings.pg_dsn, connect_timeout=5); c.close(); print('AUTH_OK')" 2>&1
Pop-Location
if (($pgProbe | Out-String) -notmatch 'AUTH_OK') {
    Write-Fail "PG 端口通但认证失败 —— .env 口令与库内实际口令不一致（P0-1 轮换后须同步 ALTER USER）"
    Save-Ledger 'postgres' $false 'auth failed'
    Exit-WithSummary 1
}
Write-Ok 'PostgreSQL 连通且认证通过'
Save-Ledger 'postgres' $true 'ok'

# ---------------------------------------------------------------
# 3. Alembic 迁移
# ---------------------------------------------------------------
Write-Step '3. Alembic 迁移（upgrade head）'

Push-Location (Join-Path $Root 'backend')
$alembicOut = & $VenvPython -m alembic upgrade head 2>&1
$alembicCode = $LASTEXITCODE
Pop-Location
if ($alembicCode -ne 0) {
    $firstErr = ($alembicOut | Select-Object -First 1)
    Write-Fail "Alembic 失败 (exit=$alembicCode) —— 首行输出: $firstErr"
    Save-Ledger 'alembic' $false "exit=$alembicCode"
} else {
    Write-Ok '迁移已是最新'
    Save-Ledger 'alembic' $true 'ok'
}

# ---------------------------------------------------------------
# 4. Celery Worker（可选）
# ---------------------------------------------------------------
if (-not $SkipCelery) {
    Write-Step '4. Celery Worker（pool=solo，Windows）'
    $celeryLog = Join-Path $LogDir 'celery.log'
    # 捕获 PID 做存活判断（对抗验收发现：原实现按命令行 grep，清屏环境下查不到 = 假阴性）
    $celeryProc = Start-Process -FilePath $VenvPython -PassThru `
        -ArgumentList '-m', 'celery', '-A', 'backend.workers.celery_app', 'worker', '--loglevel=info', '--pool=solo' `
        -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $celeryLog -RedirectStandardError $celeryLog
    Start-Sleep -Seconds 4
    $celeryAlive = $celeryProc -and -not $celeryProc.HasExited
    if ($celeryAlive) {
        Write-Ok "Celery 已启动（PID=$($celeryProc.Id)，日志: $celeryLog）"
        Save-Ledger 'celery' $true "started pid=$($celeryProc.Id)"
    } else {
        $tail = (Get-Content $celeryLog -Tail 3 -ErrorAction SilentlyContinue) -join ' | '
        Write-Warn "Celery 进程启动后 4s 内退出 —— 见 $celeryLog；尾行: $tail"
        Save-Ledger 'celery' $false "exited early: $tail"
    }
} else {
    Write-Warn '按参数跳过 Celery'
    Save-Ledger 'celery' $true 'skipped'
}

# ---------------------------------------------------------------
# 5. FastAPI（uvicorn）
# ---------------------------------------------------------------
Write-Step "5. FastAPI（127.0.0.1:$ApiPort）"

$apiLog = Join-Path $LogDir 'api.log'
Start-Process -FilePath $VenvPython `
    -ArgumentList '-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', "$ApiPort" `
    -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $apiLog -RedirectStandardError $apiLog

# ---------------------------------------------------------------
# 6. /health 轮询聚合
# ---------------------------------------------------------------
Write-Step '6. 等待 /health 聚合'

$health = $null
$deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
while ((Get-Date) -lt $deadline) {
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 3
        if ($resp.status -eq 'ok') { $health = $resp; break }
        $health = $resp   # degraded 也记录，继续等到超时看是否恢复
    } catch { }
    Start-Sleep -Seconds 2
}

if ($null -eq $health) {
    Write-Fail "$HealthTimeoutSec`s 内 /health 无响应 —— 见 $apiLog"
    Save-Ledger 'api' $false 'no /health response'
} else {
    Save-Ledger 'api' ($health.status -eq 'ok') "status=$($health.status)"
    Write-Host ("API version={0} env={1}" -f $health.version, $health.environment)
    $health.checks.PSObject.Properties | ForEach-Object {
        $mark = if ($_.Value.status -eq 'ok') { '[OK]  ' } elseif ($_.Value.status -eq 'degraded') { '[DEGR]' } else { '[DOWN]' }
        $color = if ($_.Value.status -eq 'ok') { 'Green' } elseif ($_.Value.status -eq 'degraded') { 'Yellow' } else { 'Red' }
        Write-Host ("  {0} {1,-10} {2}" -f $mark, $_.Name, $_.Value.detail) -ForegroundColor $color
    }
}

# ---------------------------------------------------------------
# 7. 前端（可选）
# ---------------------------------------------------------------
if (-not $SkipFrontend) {
    Write-Step '7. 前端（Vite dev）'
    $feDir = Join-Path $Root 'frontend'
    if (-not (Test-Path (Join-Path $feDir 'node_modules'))) {
        Write-Warn 'frontend/node_modules 缺失 —— 先 npm install'
        Save-Ledger 'frontend' $false 'node_modules missing'
    } else {
        $feLog = Join-Path $LogDir 'frontend.log'
        Start-Process -FilePath 'npm.cmd' -ArgumentList 'run', 'dev' `
            -WorkingDirectory $feDir -WindowStyle Hidden -RedirectStandardOutput $feLog -RedirectStandardError $feLog
        Start-Sleep -Seconds 5
        $feUp = Test-NetConnection -ComputerName 127.0.0.1 -Port 5173 -WarningAction SilentlyContinue |
            Select-Object -ExpandProperty TcpTestSucceeded
        if ($feUp) {
            Write-Ok '前端已启动: http://127.0.0.1:5173'
            Save-Ledger 'frontend' $true 'started'
        } else {
            Write-Warn "前端 5s 内未监听 5173 —— 见 $feLog"
            Save-Ledger 'frontend' $false 'port 5173 not listening'
        }
    }
} else {
    Save-Ledger 'frontend' $true 'skipped'
}

# ---------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------
Show-Summary

$failed = $script:ledger.GetEnumerator() | Where-Object { -not $_.Value.ok }
if ($failed) {
    $failedNames = @($failed) | ForEach-Object { $_.Key }
    Write-Host "`n存在失败组件: $($failedNames -join ', ')。修复后重跑本脚本即可。" -ForegroundColor Red
    exit 1
}
Write-Host "`n全部组件就绪。" -ForegroundColor Green
exit 0
