# Phase 0 基础设施 — 验收报告

> 阶段: Phase 0 基础设施
> 状态: ✅ 通过
> 验收日期: 2026-07-26
> Owner: Phase 0 基础设施
> 入口依据: 立项完成（5 份 truth 文档 + AGENTS.md + ADR 0001/0002）
> 出口依据: `dev-docs/stages/phase-0.md` §6 验收清单
> Git: 分支 `main`，commit `c2e5893`

## 1. 验收范围

本报告覆盖 `dev-docs/stages/phase-0.md` §6 全部出口条件，按 §6.1-§6.5 五个维度组织证据。所有验证命令于 2026-07-26 当日执行，证据为本日新鲜验证（非历史日志）。

## 2. §6.1 环境基线

| 项目 | 期望 | 实测 | 状态 |
|------|------|------|------|
| Python venv | `.venv/` 存在，Python 3.12 | `.venv/` 存在 | ✅ |
| PyTorch | 2.11+cu128 | `2.11.0+cu128` | ✅ |
| CUDA 可用 | True | `True` | ✅ |
| GPU 计算能力 | sm_120 (Blackwell) | `(12, 0)` | ✅ |
| GPU 名称 | RTX 5060 Laptop | `NVIDIA GeForce RTX 5060 Laptop GPU` | ✅ |
| PostgreSQL | 17.x @ 5433 | `PostgreSQL 17.10 on x86_64-windows` | ✅ |
| pgvector | 0.8.0 已安装 | `vector 0.8.0` | ✅ |
| Redis | 8.6.3 @ 6379 | `Redis server v=8.6.3` | ✅ |
| Node.js | v22.16.0 | `v22.16.0` | ✅ |
| npm | - | `10.9.4` | ✅ |

**验证命令**：

```powershell
# PyTorch + Blackwell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# 输出: 2.11.0+cu128 True (12, 0)

# PostgreSQL + pgvector
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h 127.0.0.1 -p 5433 -U k9system -d k9system -c "SELECT version();"
# 输出: PostgreSQL 17.10 on x86_64-windows, compiled by msvc-19.44.35227, 64-bit

# Redis
& "C:\ProgramData\chocolatey\lib\redis\tools\redis-cli.exe" -h 127.0.0.1 -p 6379 ping
# 输出: PONG
```

## 3. §6.2 后端骨架

| 项目 | 期望 | 实测 | 状态 |
|------|------|------|------|
| 目录结构 | backend/ 齐备 | 8 个子模块（api/core/models/schemas/services + workers + alembic） | ✅ |
| 依赖安装 | pip install 成功 | requirements.txt 已安装到 .venv | ✅ |
| FastAPI 启动 | uvicorn 可启动 | 8000 端口 LISTENING | ✅ |
| GET /health | 200 + ok | `200 {"status":"ok","version":"0.1.0","environment":"development"}` | ✅ |
| GET /docs | Swagger UI | `200`，HTML 长度 949 字节 | ✅ |
| OpenAPI | 路由齐备 | `/health / /api/dogs /api/dogs/{dog_id} /api/videos /api/videos/{video_id} /api/videos/upload /api/scores /api/scores/{score_id} /api/models /api/models/{model_id}` | ✅ |
| Celery worker | --pool solo 启动 | `celery@Fourteen ready.` | ✅ |
| Celery 任务消费 | health.check 成功 | `Task health.check[...] succeeded in 0.0s: {'status': 'ok', 'worker': 'celery'}` | ✅ |
| Alembic 应用 | upgrade head 成功 | `version_num = 5d7105cb28a9` | ✅ |
| 9 张业务表 | 全部存在 | handlers / dogs / training_sessions / videos / behaviors / behavior_vectors / scores / keypoints / ml_models | ✅ |

**验证命令**：

```powershell
# FastAPI health
Invoke-WebRequest http://127.0.0.1:8000/health
# 200 {"status":"ok","version":"0.1.0","environment":"development"}

# OpenAPI 路由
(Invoke-WebRequest http://127.0.0.1:8000/openapi.json).Content | ConvertFrom-Json | % { $_.paths.PSObject.Properties.Name }
# /health, /, /api/dogs, /api/dogs/{dog_id}, /api/videos, /api/videos/{video_id},
# /api/videos/upload, /api/scores, /api/scores/{score_id}, /api/models, /api/models/{model_id}

# Celery worker
celery -A backend.workers.celery_app worker -l info --pool solo
# Connected to redis://127.0.0.1:6379/1
# celery@Fourteen ready.

# Celery 任务
python -c "from backend.workers.celery_app import health_check; r = health_check.delay(); print(r.get(timeout=10))"
# {'status': 'ok', 'worker': 'celery'}
```

## 4. §6.3 前端骨架

| 项目 | 期望 | 实测 | 状态 |
|------|------|------|------|
| 目录结构 | frontend/ 齐备 | src/{api,layouts,router,stores,views} + 3 配置文件 | ✅ |
| npm install | 成功 | `added 110 packages in 52s` | ✅ |
| 类型检查 | vue-tsc 通过 | `npm run type-check` 退出 0，无错误 | ✅ |
| 生产构建 | vite build 通过 | `built in 11.14s`，4197 modules transformed | ✅ |
| npm run dev | 5173 启动 | `VITE v6.4.3 ready in 1811 ms` @ 127.0.0.1:5173 | ✅ |
| 4 个页面占位 | 可访问 | UploadView / ReportView / HistoryView / AdminView 路由注册 | ✅ |
| Naive UI 渲染 | 组件正常 | MainLayout 使用 NLayout/NMenu/NTag/NButton 等组件 | ✅ |
| dev proxy | → backend | `http://127.0.0.1:5173/health` 返回后端响应 | ✅ |

**验证命令**：

```powershell
cd frontend
npm install           # added 110 packages
npm run type-check    # vue-tsc --noEmit, exit 0
npm run build         # 4197 modules, built in 11.14s
npm run dev           # VITE v6.4.3 ready @ 127.0.0.1:5173

# Dev proxy
Invoke-WebRequest http://127.0.0.1:5173/health
# 200 {"status":"ok","version":"0.1.0","environment":"development"}
```

## 5. §6.4 数据库 Schema

| 项目 | 期望 | 实测 | 状态 |
|------|------|------|------|
| alembic_version 表 | 存在 | `tablename = alembic_version` | ✅ |
| 9 张业务表 | 全部存在 | handlers / dogs / training_sessions / videos / behaviors / behavior_vectors / scores / keypoints / ml_models | ✅ |
| Alembic 版本 | 与 migration 头一致 | `5d7105cb28a9` | ✅ |
| behavior_vectors.embedding | vector 类型 | `data_type=USER-DEFINED, udt_name=vector` | ✅ |
| 插入测试 | handlers/dogs 可插入 | `INSERT 0 1`（已回滚） | ✅ |
| Vector 字面量 | 可 cast | `'[0.1, 0.2, 0.3]'::vector` → `[0.1,0.2,0.3]` | ✅ |

**验证命令**：

```sql
-- 表清单
SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename;
-- alembic_version, behavior_vectors, behaviors, dogs, handlers,
-- keypoints, ml_models, scores, training_sessions, videos

-- Alembic 版本
SELECT version_num FROM alembic_version;
-- 5d7105cb28a9

-- Vector 字段类型
SELECT column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_name='behavior_vectors' AND column_name='embedding';
-- embedding | USER-DEFINED | vector

-- 插入/Vector 字面量（事务回滚）
BEGIN;
INSERT INTO handlers (name, role, is_active) VALUES ('测试训导员', 'HANDLER', true) RETURNING id;  -- id=1
INSERT INTO dogs (name, breed, handler_id, training_stage) VALUES ('测试犬', '马犬', 1, 'P0') RETURNING id;  -- id=2
SELECT '[0.1, 0.2, 0.3]'::vector AS v;  -- [0.1,0.2,0.3]
ROLLBACK;
```

## 6. §6.5 文档

| 文档 | 路径 | 状态 |
|------|------|------|
| 运行时基线 | `dev-docs/runtime.md` | ✅ 已创建 |
| 阶段总计划 | `dev-docs/stage-plan.md` | ✅ 已创建 |
| Phase 0 计划 | `dev-docs/stages/phase-0.md` | ✅ 已创建 |
| ADR 0001（中文路径迁移） | `dev-docs/decisions/0001-chinese-path-migration-plan.md` | ✅ 已存在 |
| ADR 0002（Blackwell 运行时栈修正） | `dev-docs/decisions/0002-runtime-stack-revision-blackwell.md` | ✅ 已创建 |
| 验收报告 | `reports/phase-0-validation.md`（本文档） | ✅ 已创建 |

## 7. 工程化产物

| 项目 | 实测 | 状态 |
|------|------|------|
| Git 主分支 | `main`（从 `master` 重命名） | ✅ |
| Phase 0 骨架 commit | `c2e5893`（64 文件，+5272 行） | ✅ |
| .gitignore | 已覆盖 .venv / .env / __pycache__ / node_modules / dist / 模型权重 / pg_hba_* 等 | ✅ |
| 临时排错脚本清理 | pg_hba_*.{conf,bat} 4 个文件已删除 | ✅ |
| .env.example | backend + frontend 均已提供 | ✅ |

## 8. 已知未验证项（不在 Phase 0 范围内）

以下项目按 `dev-docs/technical-selection.md` §7 计划在后续阶段验证，Phase 0 不要求：

- ⏳ YOLO26-pose 在工作犬品种（马犬/昆明犬）的精度（Phase 2）
- ⏳ PoseC3D 在俯视/低光/运动模糊场景的表现（Phase 2）
- ⏳ TensorRT 10.x for CUDA 12.8 在 Windows + RTX 5060 Laptop 的安装路径（Phase 1）
- ⏳ MMAction2 1.2.x + PyTorch 2.11 + Blackwell sm_120 兼容性（Phase 1）
- ⏳ Celery 5.x 在 Windows 长跑稳定性（开发期观察）
- ⏳ 前端 E2E 测试（Phase 1+）

## 9. 出口决策建议

Phase 0 全部出口条件已满足，建议升级到 Phase 1（MVP）。

按 `dev-docs/stages/phase-0.md` §7 要求，升级决策需用户确认。升级决策记录应归档为 `dev-docs/decisions/0003-phase-0-to-phase-1.md`。

## 10. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-26 | Phase 0 验收报告创建，全部出口条件通过 |
