# Runtime Baseline — 运行时基线 Truth

> Truth source: Phase 0 启动时实测形成；ADR 0002 修正后基线
> 状态: ✅ Phase 0 基线已建立（2026-07-26）
> Owner: Phase 0 基础设施
> 修改触发: 运行时栈任何组件版本变更、硬件变更、服务端口变更

## 1. 硬件基线

| 组件 | 型号 | 关键参数 | 状态 |
|------|------|---------|------|
| GPU | NVIDIA RTX 5060 Laptop | Blackwell sm_120, 8GB VRAM | ✅ 已验证 |
| 驱动 | NVIDIA Driver 573.24 | ≥550.90 满足 Blackwell 要求 | ✅ 已验证 |
| CUDA Runtime | 12.8 | Blackwell 必需 | ✅ 已验证 |
| OS | Windows 11 | x64 | ✅ |

## 2. Python 运行时

| 项 | 值 | 验证命令 |
|----|----|---------|
| 系统级 Python | 3.10.11（不用于本项目） | `python --version` |
| 项目 venv Python | **3.12.12** | `.venv\Scripts\python.exe --version` |
| venv 路径 | `d:\Desktop\k9-training-system\.venv\` | - |
| pip | 25.0.1 | `.venv\Scripts\python.exe -m pip --version` |

## 3. 核心依赖（已验证）

| 包 | 版本 | 用途 | 验证状态 |
|----|------|------|---------|
| torch | 2.11.0+cu128 | PyTorch + CUDA 12.8 | ✅ `torch.cuda.is_available()=True`, `device_cap=(12,0)` |
| numpy | 2.2.1 | 数值计算 | ⏳ 随 pip install 安装 |
| fastapi | 0.115.6 | Web 框架 | ⏳ Phase 0 验收验证 |
| uvicorn[standard] | 0.34.0 | ASGI 服务器 | ⏳ Phase 0 验收验证 |
| SQLAlchemy[asyncio] | 2.0.36 | ORM | ⏳ Phase 0 验收验证 |
| asyncpg | 0.30.0 | 异步 PG 驱动 | ⏳ Phase 0 验收验证 |
| alembic | 1.14.0 | 数据库迁移 | ⏳ Phase 0 验收验证 |
| celery[redis] | 5.4.0 | 异步任务队列 | ⏳ Phase 0 验收验证 |
| redis | 5.2.1 | Redis 客户端 | ⏳ Phase 0 验收验证 |
| ultralytics | 8.3.55 | YOLO26-pose | ⏳ Phase 0 验收验证（仅 import，权重在 Phase 1） |

> 完整版本清单见 `backend/requirements.txt`

## 4. 外部服务

### 4.1 PostgreSQL 17

| 项 | 值 |
|----|----|
| 版本 | PostgreSQL 17.10 |
| 安装路径 | `C:\Program Files\PostgreSQL\17\` |
| 数据目录 | `C:\Program Files\PostgreSQL\17\data\` |
| 端口 | **5433**（避免与旧 PG15 的 5432 冲突） |
| Windows 服务名 | `postgresql-17` |
| 启动类型 | Automatic |
| 认证 | scram-sha-256 |
| pgvector 版本 | 0.8.0（源码编译，2026-07-26） |

**应用数据库与用户**：

| 项 | 值 |
|----|----|
| 数据库名 | `k9system` |
| 用户名 | `k9system` |
| 密码存放 | `.env`（不入 Git） |
| 权限 | `GRANT ALL PRIVILEGES ON DATABASE k9system TO k9system` |
| 已启用扩展 | `vector` (pgvector 0.8.0) |

**连接字符串模板**（写入 `.env`）：
```
DATABASE_URL=postgresql+asyncpg://k9system:<PASSWORD>@127.0.0.1:5433/k9system
```

**旧 PostgreSQL 15**（系统残留）：
- 端口 5432，服务名 `postgresql-15`，本项目不使用
- 可保留或停用，不影响本项目

### 4.2 Redis

| 项 | 值 |
|----|----|
| 版本 | 8.6.3 |
| 端口 | 6379（默认） |
| 用途 | Celery Broker + 缓存 |

**连接字符串模板**：
```
REDIS_URL=redis://127.0.0.1:6379/0
CELERY_BROKER_URL=redis://127.0.0.1:6379/1
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/2
```

### 4.3 Node.js

| 项 | 值 |
|----|----|
| 版本 | v22.16.0 |
| 用途 | 前端构建 |

## 5. 验证证据（2026-07-26）

### 5.1 PyTorch + Blackwell

```python
>>> import torch
>>> torch.__version__
'2.11.0+cu128'
>>> torch.cuda.is_available()
True
>>> torch.version.cuda
'12.8'
>>> torch.cuda.get_device_capability()
(12, 0)  # sm_120 = Blackwell
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce RTX 5060 Laptop GPU'
```

### 5.2 PostgreSQL + pgvector

```sql
-- 在 k9system 数据库中执行
k9system=# SELECT extname, extversion FROM pg_extension WHERE extname='vector';
 extname | extversion
---------+------------
 vector  | 0.8.0

k9system=# CREATE TABLE _t (id serial PRIMARY KEY, embedding vector(3));
k9system=# INSERT INTO _t (embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[7,8,9]');
k9system=# SELECT * FROM _t ORDER BY embedding <-> '[3,1,2]' LIMIT 2;
 id | embedding
----+-----------
  1 | [1,2,3]
  2 | [4,5,6]
-- KNN 向量检索成功
```

### 5.3 PG17 服务状态

```
postgresql-17   Running   Automatic   ✅
postgresql-15   Running   Automatic   （本项目不使用）
```

## 6. 环境变量约定（`.env`）

`.env` 文件位于项目根目录，被 `.gitignore` 排除。模板：

```env
# === 数据库 ===
DATABASE_URL=postgresql+asyncpg://k9system:K9System2026!@127.0.0.1:5433/k9system
PG_DSN=postgresql://k9system:K9System2026!@127.0.0.1:5433/k9system

# === Redis / Celery ===
REDIS_URL=redis://127.0.0.1:6379/0
CELERY_BROKER_URL=redis://127.0.0.1:6379/1
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/2

# === 应用 ===
APP_ENV=development
APP_HOST=127.0.0.1
APP_PORT=8000
APP_LOG_LEVEL=INFO

# === 路径 ===
DATA_DIR=d:/Desktop/k9-training-system/data
UPLOAD_DIR=d:/Desktop/k9-training-system/data/uploads
GENERATED_DIR=d:/Desktop/k9-training-system/data/generated
MODELS_DIR=d:/Desktop/k9-training-system/data/models_weights
```

> **注**：实际 `.env` 中密码为真实值；此处仅为示例。`.env` 永不入 Git。

## 7. 未验证项（Phase 0 验收前需验证）

- ⏳ FastAPI 启动正常（`uvicorn backend.app.main:app`）
- ⏳ Celery Worker 启动正常（Windows 兼容性）
- ⏳ SQLAlchemy 异步连接 PG17 + pgvector 操作正常
- ⏳ Alembic 迁移初始化正常
- ⏳ ultralytics 8.3.55 在 PyTorch 2.11+cu128 + sm_120 上 import 正常
- ⏳ 前端 Vite + Vue3 + Naive UI 启动正常

## 8. 已知问题

### 8.1 Celery 5.x 在 Windows 的稳定性

Celery 5.x 在 Windows 上不支持 `--pool prefork`（默认池），需使用 `--pool solo`（开发期）或 `--pool gevent`（生产期）。

**Phase 0 启动命令**：
```bash
celery -A backend.workers.celery_app worker -l info --pool solo
```

### 8.2 PG15 与 PG17 共存

系统原有 PG15 占用 5432，本项目 PG17 使用 5433。如需释放 5432：
1. 停止 `postgresql-15` 服务
2. 修改 PG17 端口为 5432（修改 `postgresql.conf` 的 `port=5432`）
3. 重启 PG17

当前保留双服务运行，不影响本项目。

### 8.3 mmaction2 / TensorRT 延后

按 `technical-selection.md` §7：
- mmaction2 1.2.x + PyTorch 2.11 + Blackwell 兼容性 → Phase 1 验证
- TensorRT 10.x for CUDA 12.8 Windows 安装 → Phase 1 验证（仅 import 验证，TRT 模型转换在 Phase 1）

Phase 0 不安装 mmaction2 / TensorRT，避免环境复杂化。

## 9. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-26 | Phase 0 基线建立：Python 3.12 + PyTorch 2.11+cu128 + PG17.10 + pgvector 0.8.0 + Redis 8.6.3 |
