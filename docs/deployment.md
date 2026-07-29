# 部署指南（Windows 本地部署）

> 工作犬训练机器视觉识别系统 — Phase 1 MVP
> 适用版本: v0.1.0
> 目标系统: Windows 11（开发/单机部署）
> 部署模式: 单机本地化（无云依赖，数据不出本机）

## 1. 系统要求

### 1.1 硬件

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 4 核 Intel/AMD x64 | 8 核+ |
| 内存 | 8 GB | 16 GB+ |
| 存储 | 20 GB SSD | 50 GB+ SSD（含视频/模型权重） |
| GPU | NVIDIA RTX 30/40/50 系列 8GB VRAM | RTX 4060/5060+ 12GB |
| 显存 | 8 GB | 12 GB+ |

### 1.2 软件

| 组件 | 版本 | 用途 |
|------|------|------|
| Windows | 11 / Server 2022+ | 操作系统 |
| Python | 3.12.x（CPython） | 后端运行时 |
| Node.js | 20.x LTS | 前端构建 |
| PostgreSQL | 16.x（端口 5433） | 主数据库 |
| Redis | 7.x+（端口 6379） | Celery 消息队列 + 结果后端 |
| NVIDIA Driver | 560.x+（Blackwell 架构需 565+） | GPU 驱动 |
| CUDA Toolkit | 12.8+（Blackwell 兼容） | PyTorch GPU 后端 |
| Git | 2.40+ | 代码版本控制 |

## 2. 软件预安装

### 2.1 Python 3.12

```powershell
# 通过 uv 安装（推荐）
winget install astral-sh.uv
uv python install 3.12

# 或通过官方安装包: https://www.python.org/downloads/release/python-3120/
# 安装时勾选 "Add Python to PATH"
```

### 2.2 Node.js 20 LTS

```powershell
winget install OpenJS.NodeJS.LTS
# 验证
node --version  # 应输出 v20.x.x
npm --version
```

### 2.3 PostgreSQL 16（端口 5433）

```powershell
# 通过 Chocolatey 安装
choco install postgresql16 -y

# 或下载 EnterpriseDB 安装包: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
# 安装时端口改为 5433（避免与默认 5432 冲突），superuser 密码自定
```

**创建数据库用户与库**：

```powershell
# 以 postgres 超级用户登录
psql -U postgres -p 5433

# 在 psql 提示符中执行
CREATE USER k9system WITH PASSWORD 'K9System2026!';
CREATE DATABASE k9system OWNER k9system ENCODING 'UTF8';
\q
```

### 2.4 Redis（端口 6379）

```powershell
# 通过 Chocolatey 安装
choco install redis-64 -y

# 启动 Redis（默认 6379）
redis-server --port 6379

# 或注册为 Windows 服务（推荐生产环境）
redis-server --service-install redis.windows.conf --service-name redis
Start-Service redis
```

### 2.5 CUDA Toolkit 12.8（Blackwell GPU 必需）

```powershell
# 下载地址: https://developer.nvidia.com/cuda-12-8-0-download-archive
# 安装时选择 "Custom" → 仅勾选 CUDA Toolkit（不装驱动）
# 验证
nvcc --version  # 应输出 12.8
nvidia-smi      # 显示 GPU 信息 + 驱动版本
```

## 3. 项目部署

### 3.1 克隆代码

```powershell
cd D:\Desktop
git clone <repository-url> k9-training-system
cd k9-training-system
```

### 3.2 创建 Python 虚拟环境

```powershell
# 使用 uv（推荐，速度快）
uv venv .venv --python 3.12
.venv\Scripts\activate

# 或使用标准 venv
python -m venv .venv
.venv\Scripts\activate
```

### 3.3 安装 PyTorch（GPU 版本）

```powershell
# Blackwell 架构（RTX 50 系）必须用 cu128
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# 验证 GPU 可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### 3.4 安装后端依赖

```powershell
pip install -r backend\requirements.txt

# 验证关键包
python -c "import fastapi, celery, sqlalchemy, ultralytics, onnxruntime; print('OK')"
```

### 3.5 配置环境变量

```powershell
# 复制模板
Copy-Item backend\.env.example .env

# 编辑 .env 文件，按需修改：
#   DATABASE_URL / PG_DSN: 数据库连接串（密码与 §2.3 一致）
#   REDIS_URL / CELERY_BROKER_URL / CELERY_RESULT_BACKEND
#   APP_PORT: API 端口（默认 8000）
#   CORS_ORIGINS: 前端访问源
```

### 3.6 初始化数据库 Schema

```powershell
# Alembic 迁移到最新版本
.venv\Scripts\alembic.exe upgrade head

# 验证表结构
psql -U k9system -d k9system -p 5433 -c "\dt"
# 应看到 dogs / handlers / videos / keypoints / behaviors / scores / ml_models 等表
```

### 3.7 准备模型权重

```powershell
# 将以下文件放置到指定路径（不入 Git）：
#   runs\train-2\weights\best.pt   — YOLO26-pose 微调权重（24 关键点）
#   runs\train-2\weights\best.onnx — ONNX 导出版本（生产推荐）

# 若无 best.pt，系统会自动下载 yolo26n-pose.pt 作为兜底（精度较低）
```

### 3.8 构建前端

```powershell
cd frontend
npm install
npm run build  # 产物输出到 frontend\dist\
cd ..
```

## 4. 服务启动

### 4.1 启动顺序

按以下顺序启动服务（建议每个开一个独立终端窗口）：

```
PostgreSQL → Redis → Celery Worker → FastAPI → Frontend
```

### 4.2 启动 PostgreSQL

```powershell
# 通常安装时已注册为服务，开机自启
Get-Service postgresql* | Start-Service

# 验证
psql -U k9system -d k9system -p 5433 -c "SELECT version();"
```

### 4.3 启动 Redis

```powershell
# 方式 A: 注册为服务（推荐生产）
Start-Service redis  # 若已注册

# 方式 B: 前台运行（开发期）
redis-server --port 6379

# 验证
redis-cli ping  # 应返回 PONG
```

### 4.4 启动 Celery Worker

```powershell
# 在项目根目录
.venv\Scripts\activate
celery -A backend.workers.celery_app worker -l info --pool solo -n k9_worker@%h
```

**参数说明**：
- `--pool solo`: Windows 不支持 prefork，必须用 solo 池
- `-n k9_worker@%h`: 指定 worker 名称（多机部署时区分）
- `-l info`: 日志级别（生产用 `-l warning`）

**验证**：终端输出 `k9_worker@<hostname> ready.` 即启动成功。

### 4.5 启动 FastAPI 后端

```powershell
# 开发模式（热重载，仅开发用）
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

# 生产模式（稳定，推荐生产用）
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

**验证**：
```powershell
curl http://127.0.0.1:8000/health
# 应返回 {"status":"ok","version":"0.1.0","environment":"development"}
```

**API 文档**：浏览器访问 http://127.0.0.1:8000/docs

### 4.6 启动前端

```powershell
# 开发模式（Vite 热更新）
cd frontend
npm run dev  # 默认 http://127.0.0.1:5173

# 生产模式（静态文件服务）
# 先构建（见 §3.8），然后用任意静态服务器托管 frontend\dist\
npx serve frontend\dist -l 5173
```

**Vite 代理配置**（`frontend/vite.config.ts`）：
- `/api` → `http://127.0.0.1:8000`（后端 API）
- `/health` → 后端健康检查
- `/docs` → 后端 Swagger UI

若后端端口非 8000，编辑 `frontend/.env`：
```
VITE_BACKEND_HOST=127.0.0.1
VITE_BACKEND_PORT=8001
```

## 5. 端到端验证

### 5.1 健康检查

```powershell
# 后端
curl http://127.0.0.1:8000/health

# Celery（通过 API 间接验证）
curl http://127.0.0.1:8000/docs  # 找到 /api/health 相关接口
```

### 5.2 端到端冒烟测试

```powershell
# 运行 Phase 1.7 端到端测试脚本（自动上传合成视频 → 推理 → PDF 报告）
.venv\Scripts\python.exe scripts\phase1_7_e2e_test.py
```

**预期输出**：
```
[0] 环境准备:
  [PASS] 健康检查 /health
  [PASS] 创建测试犬只
  [PASS] 模型预热
[1] 1.7a 科目测评端到端:
  [PASS] 科目测评视频 → PDF 报告
[2] 1.7b 选育场景端到端:
  [PASS] 选育视频 → 9 信号 → PDF 报告
[3] 1.7c YAML 评分卡动态修改:
  [PASS] PUT 评分卡 → 热加载 → 评分对比
[4] 1.7d 端到端延迟验证:
  [PASS] 延迟 ≤ 1 min / min 视频
结果: 7 passed, 0 failed, 0 skipped
```

### 5.3 前端验证

1. 浏览器访问 http://127.0.0.1:5173
2. 上传页：拖入一段 mp4 视频，选择场景（科目测评/幼犬选育），点击上传
3. 报告页：等待处理完成，查看 7 维评分 + PDF 预览
4. 历史页：查看视频列表与状态
5. 管理页：切换 Tab 查看模型/评分卡/犬只/系统信息

## 6. 生产环境部署

### 6.1 注册 Windows 服务（推荐）

使用 NSSM（Non-Sucking Service Manager）将 FastAPI 和 Celery 注册为 Windows 服务：

```powershell
# 安装 NSSM
choco install nssm -y

# 注册 FastAPI 服务
nssm install k9_api "D:\Desktop\k9-training-system\.venv\Scripts\python.exe" `
  "-m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
nssm set k9_api AppDirectory "D:\Desktop\k9-training-system"
nssm set k9_api AppEnvironmentExtra "PYTHONUNBUFFERED=1"
nssm set k9_api Start SERVICE_AUTO_START
Start-Service k9_api

# 注册 Celery Worker 服务
nssm install k9_worker "D:\Desktop\k9-training-system\.venv\Scripts\celery.exe" `
  "-A backend.workers.celery_app worker -l info --pool solo -n k9_worker@%h"
nssm set k9_worker AppDirectory "D:\Desktop\k9-training-system"
nssm set k9_worker Start SERVICE_AUTO_START
Start-Service k9_worker

# 注册前端静态服务（用 serve 或 IIS）
nssm install k9_frontend "npx" "serve D:\Desktop\k9-training-system\frontend\dist -l 5173"
nssm set k9_frontend AppDirectory "D:\Desktop\k9-training-system\frontend"
Start-Service k9_frontend
```

### 6.2 开机自启

将 PostgreSQL、Redis、k9_api、k9_worker、k9_frontend 全部设为 `SERVICE_AUTO_START`：

```powershell
Set-Service postgresql-x64-16 -StartupType Automatic
Set-Service redis -StartupType Automatic
Set-Service k9_api -StartupType Automatic
Set-Service k9_worker -StartupType Automatic
Set-Service k9_frontend -StartupType Automatic
```

### 6.3 日志管理

```powershell
# NSSM 服务日志（默认位置）
# C:\Program Files\NSSM\k9_api\stdout.log
# C:\Program Files\NSSM\k9_api\stderr.log

# 自定义日志路径
nssm set k9_api AppStdout "D:\Desktop\k9-training-system\logs\api.log"
nssm set k9_api AppStderr "D:\Desktop\k9-training-system\logs\api_error.log"
nssm set k9_api AppRotateFiles 1
nssm set k9_api AppRotateBytes 10485760  # 10 MB 轮转
```

### 6.4 数据备份

```powershell
# PostgreSQL 备份（每日）
$backup = "D:\backup\k9system_$(Get-Date -Format 'yyyyMMdd').sql"
pg_dump -U k9system -p 5433 k9system > $backup

# 视频文件备份（按需）
Copy-Item D:\Desktop\k9-training-system\data\uploads\* D:\backup\uploads\ -Recurse

# 评分卡 YAML 备份（纳入 Git 版本控制即可）
```

## 7. 故障排查

### 7.1 GPU 不可用

**症状**：`torch.cuda.is_available()` 返回 `False`，或推理速度异常缓慢。

**排查**：
```powershell
nvidia-smi  # 检查 GPU 驱动
nvcc --version  # 检查 CUDA Toolkit 版本

# Blackwell GPU（RTX 50 系）必须 CUDA 12.8+
# 重装 PyTorch:
pip uninstall torch -y
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

### 7.2 Celery 任务卡住

**症状**：视频上传后状态长时间停在 `processing`。

**排查**：
```powershell
# 检查 worker 是否就绪
celery -A backend.workers.celery_app inspect ping

# 检查 Redis 队列
redis-cli LLEN celery  # 队列长度

# 查看 worker 日志（前台运行时直接看终端）
# NSSM 服务模式下查看 logs\k9_worker.log
```

**常见原因**：
- Redis 未启动 → 启动 Redis
- Worker 未启动 → 启动 Celery Worker
- 模型权重缺失 → 检查 `runs/train-2/weights/best.onnx`
- GPU 显存不足 → 关闭其他 GPU 进程，或减小 `imgsz`

### 7.3 端口冲突

**症状**：FastAPI 或前端启动失败，提示端口占用。

```powershell
# 查看占用进程
netstat -ano | findstr ":8000.*LISTENING"
# 替换 8000 为冲突端口

# 终止进程（PID 从上一条命令获取）
Stop-Process -Id <PID> -Force

# 或修改 .env 中 APP_PORT，同步修改 frontend/.env 中 VITE_BACKEND_PORT
```

### 7.4 数据库连接失败

**症状**：FastAPI 启动时报 `OperationalError: could not connect to server`。

**排查**：
```powershell
# 检查 PostgreSQL 服务
Get-Service postgresql*

# 检查端口
netstat -ano | findstr ":5433.*LISTENING"

# 测试连接
psql -U k9system -d k9system -p 5433 -c "SELECT 1;"

# 密码错误时重置
psql -U postgres -p 5433 -c "ALTER USER k9system WITH PASSWORD 'K9System2026!';"
```

### 7.5 PDF 报告生成失败

**症状**：视频处理完成但 `report_path` 为空，或下载 PDF 返回 500。

**排查**：
```powershell
# 检查 reports 目录权限
icacls D:\Desktop\k9-training-system\reports

# 检查 reportlab 安装
python -c "import reportlab; print(reportlab.Version)"

# 查看后端日志中的栈跟踪
```

### 7.6 前端无法访问 API

**症状**：前端页面加载正常，但所有 API 请求 404 或 CORS 错误。

**排查**：
```powershell
# 1. 确认后端启动
curl http://127.0.0.1:8000/health

# 2. 确认 Vite 代理配置（frontend/vite.config.ts）
#    target 应指向后端实际端口

# 3. 确认 CORS（.env 中 CORS_ORIGINS 包含前端源）
#    默认: ["http://127.0.0.1:5173","http://localhost:5173"]
```

## 8. 卸载

```powershell
# 停止服务
Stop-Service k9_api, k9_worker, k9_frontend, redis, postgresql-x64-16

# 删除 NSSM 服务
nssm remove k9_api confirm
nssm remove k9_worker confirm
nssm remove k9_frontend confirm

# 删除项目目录
Remove-Item D:\Desktop\k9-training-system -Recurse -Force

# 删除数据库（可选）
psql -U postgres -p 5433 -c "DROP DATABASE k9system;"
psql -U postgres -p 5433 -c "DROP USER k9system;"
```

## 9. 附录

### 9.1 端口清单

| 服务 | 端口 | 说明 |
|------|------|------|
| PostgreSQL | 5433 | 主数据库 |
| Redis | 6379 | Celery broker/result |
| FastAPI | 8000（默认） / 8001 | 后端 API |
| Vite Dev | 5173 | 前端开发服务器 |
| Flower | 5555（可选） | Celery 监控面板 |

### 9.2 关键路径

| 路径 | 说明 |
|------|------|
| `data/uploads/` | 上传视频存储 |
| `data/generated/` | 推理产物（关键点 JSON 等） |
| `data/models_weights/` | 模型权重存储 |
| `reports/` | 生成的 PDF 评分报告 |
| `runs/train-2/weights/` | YOLO26-pose 微调权重 |
| `backend/ml/scoring/configs/` | 评分卡 YAML 配置 |
| `logs/` | 服务日志（生产模式） |

### 9.3 环境变量速查

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATABASE_URL` | `postgresql+asyncpg://k9system:***@127.0.0.1:5433/k9system` | 异步数据库 URL |
| `PG_DSN` | `postgresql://k9system:***@127.0.0.1:5433/k9system` | 同步 DSN（Alembic） |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis 连接 |
| `CELERY_BROKER_URL` | `redis://127.0.0.1:6379/1` | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://127.0.0.1:6379/2` | Celery 结果后端 |
| `APP_PORT` | `8000` | API 端口 |
| `CORS_ORIGINS` | `["http://127.0.0.1:5173",...]` | 允许的前端源 |
| `VITE_BACKEND_HOST` | `127.0.0.1` | 前端代理目标 |
| `VITE_BACKEND_PORT` | `8000` | 前端代理目标端口 |
