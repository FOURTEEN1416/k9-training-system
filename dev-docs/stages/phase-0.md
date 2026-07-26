# Phase 0 — 基础设施阶段计划

> 阶段: Phase 0 基础设施
> 状态: ✅ 完成（2026-07-26 验收通过，见 `reports/phase-0-validation.md`）
> Owner: Phase 0 基础设施
> 入口条件: 立项完成（5 份 truth 文档 + 宪法 + ADR 0001/0002）✅
> 出口条件: 见 §6 验收清单（全部通过）

## 1. 阶段目标

**让开发者能够在新机器上从零启动整套开发环境**，并建立可演进的前后端骨架与数据库 schema。本阶段不交付任何业务功能。

## 2. 范围

### 2.1 包含

- Python 3.12 venv + PyTorch 2.11+cu128（Blackwell sm_120 验证）
- PostgreSQL 17 + pgvector 0.8.0
- Redis 8.6.3
- Node.js v22.16.0 + 前端构建链
- backend/ 骨架（FastAPI + Celery + SQLAlchemy + Alembic）
- frontend/ 骨架（Vue3 + Vite + TS + Naive UI + Pinia + ECharts）
- DB schema 初版（dogs / videos / keypoints / behaviors / scores / models / handlers / training_sessions）
- `.env` 模板与 gitignore
- runtime.md truth 文档

### 2.2 不包含

- 任何业务功能（视频上传/推理/评分等）
- YOLO26-pose 模型权重下载（Phase 1）
- mmaction2 / TensorRT 安装（Phase 1）
- 单元测试 / E2E 测试编写（Phase 1）
- PDF 报告模板（Phase 1）

## 3. 任务分解

### B1 环境基线 ✅

- **B1a** Python 3.12 venv + PyTorch 2.11+cu128 + sm_120 验证 ✅
- **B1b** PostgreSQL 17 + pgvector 0.8.0 ✅
- **B1c** Redis 8.6.3 ✅
- **B1d** Node.js v22.16.0 ✅

### B2 Truth 文档

- **B2a** runtime.md（运行时基线）✅
- **B2b** stage-plan.md + stages/phase-0.md ✅（本文档）

### B3 后端骨架

- **B3a** backend/ 目录结构
- **B3b** backend/requirements.txt
- **B3c** backend/app/core/config.py（Pydantic Settings + .env）
- **B3d** backend/app/core/database.py（SQLAlchemy async engine）
- **B3e** backend/app/models/*.py（SQLAlchemy 模型，含 pgvector）
- **B3f** backend/app/api/*.py（FastAPI 路由占位）
- **B3g** backend/app/main.py（FastAPI 入口 + CORS + 路由注册）
- **B3h** backend/workers/celery_app.py（Celery 实例）
- **B3i** backend/alembic/（Alembic 迁移初始化 + env.py + 首个 migration）
- **B3j** backend/app/schemas/*.py（Pydantic schema 占位）

### B4 前端骨架

- **B4a** frontend/ 目录结构（Vite + Vue3 + TS 脚手架）
- **B4b** frontend/package.json（依赖锁定）
- **B4c** Naive UI + Pinia + Vue Router + ECharts 集成
- **B4d** frontend/src/api/（HTTP 客户端封装）
- **B4e** frontend/src/views/（4 个页面占位：Upload/Report/History/Admin）
- **B4f** frontend/src/router/（路由配置）
- **B4g** frontend/src/stores/（Pinia store 占位）
- **B4h** frontend/vite.config.ts（dev proxy → backend）

### B5 Git 工程化

- **B5a** 切换主分支 master → main
- **B5b** .gitignore 完善（已存在，按骨架实际产物补全）
- **B5c** 提交 Phase 0 骨架

### B6 数据库 Schema

- **B6a** SQLAlchemy 模型定义（9 张核心表）
- **B6b** Alembic 首个 migration 生成
- **B6c** migration 应用到 k9system 数据库
- **B6d** pgvector 类型字段验证（behavior_vectors 表）

## 4. 技术决策

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| Python 版本 | 3.10 / 3.11 / 3.12 | **3.12** | PyTorch 2.11 兼容、mmaction2 推荐 |
| venv 位置 | 项目根 / 用户目录 | **项目根 `.venv/`** | 隔离 + 易迁移 |
| 包管理 | pip / poetry / uv | **pip + requirements.txt** | Phase 0 简单；Phase 2+ 评估 pyproject.toml |
| PG 端口 | 5432 / 5433 | **5433** | 避免与系统旧 PG15 冲突 |
| Celery pool | prefork / solo / gevent | **solo**（开发期） | Windows 不支持 prefork |
| 前端脚手架 | Vue CLI / Vite | **Vite** | 官方推荐、速度快 |
| API 风格 | REST / GraphQL | **REST** | 简单、FastAPI 原生支持 |

## 5. 验证命令

```bash
# 后端
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload  # 应在 127.0.0.1:8000 启动
.venv\Scripts\celery.exe -A backend.workers.celery_app worker -l info --pool solo  # 应启动 worker

# 前端
cd frontend && npm run dev  # 应在 127.0.0.1:5173 启动

# 数据库
.venv\Scripts\alembic.exe upgrade head  # 应成功创建所有表

# 健康检查
curl http://127.0.0.1:8000/health  # 应返回 {"status":"ok"}
```

## 6. 验收清单（出口条件）

### 6.1 环境

- [x] Python 3.12 venv 存在
- [x] PyTorch 2.11+cu128 可用，sm_120 验证通过
- [x] PostgreSQL 17.10 运行在 5433
- [x] pgvector 0.8.0 已安装且可 CREATE EXTENSION
- [x] Redis 8.6.3 运行在 6379
- [x] Node.js v22.16.0 可用

### 6.2 后端

- [ ] backend/ 目录结构齐备
- [ ] `pip install -r backend/requirements.txt` 成功
- [ ] `uvicorn backend.app.main:app` 启动成功
- [ ] `GET /health` 返回 200
- [ ] `GET /docs` 返回 Swagger UI
- [ ] Celery worker 启动成功（--pool solo）
- [ ] Alembic 迁移应用成功
- [ ] 9 张核心表在 k9system 数据库中可见

### 6.3 前端

- [ ] frontend/ 目录结构齐备
- [ ] `npm install` 成功
- [ ] `npm run dev` 启动成功
- [ ] 4 个页面占位可访问
- [ ] Naive UI 组件渲染正常

### 6.4 数据库

- [ ] Alembic 版本表存在
- [ ] 9 张业务表存在
- [ ] behavior_vectors 表包含 vector 类型字段
- [ ] 插入/查询测试通过

### 6.5 文档

- [x] runtime.md 已创建
- [x] stage-plan.md 已创建
- [x] stages/phase-0.md（本文档）已创建
- [ ] reports/phase-0-validation.md 验收报告

## 7. 出口决策

Phase 0 完成后，由用户判断是否升级到 Phase 1。升级决策记录到 `dev-docs/decisions/0003-phase-0-to-phase-1.md`（待创建）。

## 8. 不可逆操作清单

本阶段涉及以下不可逆操作，需用户确认：

- ✅ PostgreSQL 17 安装（已执行）
- ✅ pgvector 编译安装（已执行）
- ✅ k9system 数据库 + 用户创建（已执行）
- ✅ Alembic migration 应用到 k9system 数据库（已执行，version=5d7105cb28a9）
- ✅ Git master → main 切换（已执行）

## 9. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-26 | Phase 0 计划创建 |
| v1.1 | 2026-07-26 | Phase 0 验收通过，状态更新为 ✅ 完成 |
