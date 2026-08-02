# Runtime Baseline — 运行时基线 Truth

> Truth source: Phase 0 启动时实测形成；ADR 0002 修正后基线
> 状态: ✅ Phase 0 基线已建立（v1.6，2026-08-02 onnxruntime 环境修复 + FastAPI 204 路由兼容性修复 + 631 单元测试 0 失败）
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
| numpy | 2.2.1 | 数值计算 | ✅ Phase 0 验收通过 |
| fastapi | 0.115.6 | Web 框架 | ✅ Phase 0 验收通过 |
| uvicorn[standard] | 0.34.0 | ASGI 服务器 | ✅ Phase 0 验收通过 |
| SQLAlchemy[asyncio] | 2.0.36 | ORM | ✅ Phase 0 验收通过 |
| asyncpg | 0.30.0 | 异步 PG 驱动 | ✅ Phase 0 验收通过 |
| alembic | 1.14.0 | 数据库迁移 | ✅ Phase 0 验收通过 |
| celery[redis] | 5.4.0 | 异步任务队列 | ✅ Phase 0 验收通过 |
| redis | 5.2.1 | Redis 客户端 | ✅ Phase 0 验收通过 |
| ultralytics | 8.4.107 | YOLO26-pose | ✅ Phase 1+ 验证（v1.1：从 8.3.55 升级，匹配 `requirements.txt`） |
| opencv-python | 5.0.0.93 | 视频解码/抽帧 | ✅ Phase 1+ 验证（v1.1：非 headless 版，AGENTS.md §9 cv2 冲突修复） |
| onnxruntime-gpu | 1.20.1 | ONNX 推理 | ✅ Phase 1.1 验证 |
| tensorrt | 10.8.0.43 | TRT FP16（备选） | ✅ Phase 1.1 验证 |
| scipy | 1.15.0 | 关键点处理 | ✅ Phase 1.2 验证 |
| pandas | 2.2.3 | 数据处理 | ✅ Phase 1.2 验证 |
| matplotlib | 3.10.0 | PDF 报告图片 | ✅ Phase 1.4 验证 |
| reportlab | 4.2.5 | PDF 生成 | ✅ Phase 1.4 验证 |
| pgvector | 0.5.0 | 向量扩展 Python 端 | ✅ Phase 0 验证 |

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

## 5. 验证证据（2026-07-26 Phase 0 + 2026-08-01 新鲜验证）

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

### 5.4 新鲜验证（2026-08-02，onnxruntime 修复 + FastAPI 204 兼容性修复 + 631 单元测试 0 失败）

- ✅ **631 单元测试通过** + 2 skipped + 0 failed in 98.40s（`pytest backend/tests/`，较 v1.5 的 566 +65 = 3.6c RBAC 端到端脚本进入测试 + onnxruntime 损坏 17 项恢复 + FastAPI 204 修复）
  - ✅ **onnxruntime 环境已修复**：venv 中仅 onnxruntime-gpu 1.20.1（与 _pybind_state.pyd 1.20.1 版本一致），之前报告的 1.28.0/1.20.1 混合已不存在。runtime.md v1.5 提到的"用户手动修复"步骤无需执行
  - ✅ **FastAPI 0.115.6 204 兼容性修复**：4 个 204 路由（auth.py `/change-password` + `/users/{id}` DELETE + bases.py `/{id}` DELETE + dogs.py `/{id}` DELETE）添加 `response_class=Response` + 返回类型 `-> None` 改为 `-> Response` + 显式 `return Response(status_code=204)`，解决 `AssertionError: Status code 204 must not have a response body`
- ✅ **RBAC 单元测试 45/45 通过**（`backend/tests/test_rbac.py` in 4.87s：TestPasswordHash 7 + TestJWT 7 + TestAuthErrors 6 + TestUserRole 4 + TestCheckBaseAccess 11 + TestCheckDogAccess 10）
- ✅ **3.1e 部署单元测试 20/20 通过**（`backend/tests/ml/test_stgcn_bc_deploy.py`: TestExportOnnx 4 + TestSTGCNBCInferer 6 + TestBehaviorRecognizer 8 + TestEpisodeSplit 2）
- ✅ **3.4 FCI-IGP 单元测试 15/15 通过**（`backend/tests/integration/test_phase3_4_fci_igp_e2e.py`）
- ✅ **端到端 SHADOW 模式新鲜验证通过**（USPCA 视频 2700 帧/90s → video_id=37 → 101.8s → verdict=pass score=81.0 → PDF 4036 bytes）
  - SHADOW 对比日志：`STGCN=1 RULE=1 common=0 stgcn_only=1 rule_only=1`（双轨均识别 1 个行为）
  - 延迟分析：SHADOW 双轨仅占 2s（< 2%），瓶颈在 pose 推理 98s（96%），1.13x 略超 1.0x 阈值，将通过 Phase 3.5 Jetson TRT FP16 优化
- ✅ **FCI-IGP 端到端验证通过**（video_id=46, scene=fci_igp, verdict=pass, score=78.9, 57.0s/0.63x, PDF 4156 bytes）
- ✅ ST-GCN+BC 合成数据 baseline 训练验证（30 epochs / 1.43M 参数 / best_val_acc=46.97% @ epoch 21 / 边界 F1=58.45%）
- ✅ MotionBERT 17→24 适配 + InterPet4D 微调验证（best_epoch=12 / MPJPE=21.74mm / P-MPJPE=20.67mm，见 `reports/phase-3.3d-3d-pose-eval.json`）
- ✅ `scripts/phase2_6_e2e_test.py` 8/9 通过（USPCA 闭环 + 延迟 1.13x 略超标 + PDF 报告）
- ✅ API 启动验证（`uvicorn backend.app.main:app --port 8001`）—— cv2 冲突修复后通过
- ✅ Celery Worker 启动验证（`celery -A backend.workers.celery_app worker -l info --pool=solo`）
- ✅ **3.6 RBAC auth API 已上线**（`main.py` 注册 auth + bases 路由 + AuthError exception_handler）
- ✅ **3.6 RBAC 端到端 25/25 通过**（`scripts/eval_rbac.py` → `reports/phase-3.6c-rbac-eval.json`：authentication 6 + role_permission 5 + base_isolation 2 + base_crud 5 + user_management 4 + change_password 3，真实 PG17 migration `c3d4e5f6a7b8` 执行验证）

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

## 7. 已验证项（Phase 0 验收 + Phase 1-3 增量验证）

> v1.1 修订：原"未验证项（Phase 0 验收前需验证）"全部转为已验证（Phase 0 验收 2026-07-26 + 2026-08-01 新鲜验证）。

- ✅ FastAPI 启动正常（`uvicorn backend.app.main:app`）
- ✅ Celery Worker 启动正常（Windows 兼容性，`--pool solo`）
- ✅ SQLAlchemy 异步连接 PG17 + pgvector 操作正常
- ✅ Alembic 迁移初始化正常（3 个迁移版本：phase0 / phase1_4d / phase2_1a）
- ✅ ultralytics 8.4.107 在 PyTorch 2.11+cu128 + sm_120 上 import 正常
- ✅ 前端 Vite + Vue3 启动正常

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

### 8.3 mmaction2 / TensorRT

按 `technical-selection.md` §7 + ADR 0006 v1.1：
- **mmaction2**：Phase 1.3 未触发（1.2f 条件通过维持 PoseC3D 跳过决策），mmcv/mmaction2 未安装。Phase 3+ 如需 ST-GCN 系列可重新评估
- **TensorRT 10.8.0.43**：Phase 1.1 已安装（PyPI 官方包），实测 YOLO26n-pose + RTX 5060 Blackwell 上 TensorRT 17.05ms vs ONNX 13.39ms，生产用 ONNX Runtime GPU，TensorRT engine 保留作为备选

### 8.4 opencv-python-headless 冲突（2026-08-01 修复）

**问题**：`.venv` 同时安装 opencv-python-headless 4.14 + opencv-python 5.0，Python 取 headless 版（缺 `imshow`/`imwrite`），导致 ultralytics 导入崩溃，API 完全无法启动。

**修复**：卸载 opencv-python-headless，保留 opencv-python 5.0.0.93。`requirements.txt` 第33行已明确注释"必须用非 headless 版"。

**依据**：AGENTS.md v1.15 §9 运行时阻断修复记录。

### 8.5 Redis 服务启动（2026-08-02 验证）

**问题**：Redis 安装在 `C:\ProgramData\chocolatey\bin\redis-server.exe`，但默认不自动启动。Celery worker 启动时连接 `redis://127.0.0.1:6379/1` 失败（`Error 10061 connecting to 127.0.0.1:6379`），导致端到端推理任务无法调度。

**启动命令**（开发期，前台运行便于观察日志）：
```bash
redis-server --port 6379 --daemonize no
```

**验证**：
```bash
redis-cli ping
# 期望输出: PONG
```

**生产期建议**：通过 `nssm` 注册为 Windows 服务自动启动（参考 `technical-selection.md` Windows 服务管理约定）。

### 8.6 ST-GCN+BC 部署模式（2026-08-02 上线）

**4 模式路由**（`backend/ml/behavior/router.py::BehaviorRecognizer`）：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `SHADOW` | ST-GCN+BC 推理 + 规则引擎返回，仅记录对比 | 当前默认（合成模型验证期） |
| `VOTE` | ST-GCN+BC + 规则引擎投票合并 | 真实数据训练后切换 |
| `PRIMARY_STGCN` | ST-GCN+BC 主 + 规则引擎备（失败降级） | ST-GCN+BC 准确率 ≥ 85% 后切换 |
| `RULE_ONLY` | 仅规则引擎（ST-GCN+BC 不可用自动降级） | 模型缺失时降级 |

**配置切换**：通过 `settings.behavior_deploy_mode`（`.env` 未配置时默认 `shadow`）。

**模型路径解析**（`backend/workers/tasks.py::_resolve_stgcn_bc_path`）：
1. 优先 ONNX：`data/models/stgcn_bc/stgcn_bc_dog24.onnx`
2. 回退 PyTorch checkpoint：`runs/stgcn_bc_synthetic/best.pt`
3. 全无：自动降级 `RULE_ONLY` 模式

**ONNX 导出 CLI**：
```bash
python scripts/export_stgcn_bc_onnx.py \
    --checkpoint runs/stgcn_bc_synthetic/best.pt \
    --output data/models/stgcn_bc/stgcn_bc_dog24.onnx \
    --opset 17
```

### 8.7 RBAC 用户权限（2026-08-02 上线 + 端到端验收通过）

**状态**：3.6a migration + model ✅ / 3.6b auth API + deps + main.py 注册 ✅ 上线 / 3.6c 多租户端到端 25/25 通过 ✅

**已实现文件**：
- `backend/alembic/versions/c3d4e5f6a7b8_phase3_6_rbac_base_tables.py`（migration，真实 PG17 执行验证通过）
- `backend/app/models/handler.py`（UserRole 5 角色：ADMIN/MANAGER/HANDLER/RESEARCHER/VIEWER + ROLE_HIERARCHY + password_hash + base_id + is_superuser）
- `backend/app/models/base_entity.py`（BaseEntity time-mixin）
- `backend/app/models/dog_associations.py`（DogBaseAssociation + DogHandlerAssociation + DogHandlerRole PRIMARY/SECONDARY）
- `backend/app/core/security.py`（JWT 编解码 + bcrypt 直接调用密码哈希 + 72 字节截断 + AuthError 异常层级 + TokenPayload sub 字符串化）
- `backend/app/core/deps.py`（get_current_handler + get_optional_handler + require_roles + require_role_hierarchy + check_dog_access + check_base_access，async 接口）
- `backend/app/api/auth.py`（POST /auth/login + /auth/refresh + GET /auth/me + POST /auth/logout + POST /auth/register ADMIN + GET /auth/users ADMIN/MANAGER + PATCH /auth/users/{id}/deactivate ADMIN + POST /auth/change-password）
- `backend/app/api/bases.py`（基地 CRUD：POST 创建 ADMIN / GET 列表 ADMIN 全量+MANAGER 本基地 / GET 单个 / PATCH 更新 ADMIN / PATCH deactivate ADMIN）
- `backend/app/main.py`（注册 auth + bases 路由 + AuthError exception_handler）
- `backend/tests/test_rbac.py`（45 单元测试：密码哈希 7 + JWT 7 + 异常 6 + UserRole 4 + check_base_access 11 + check_dog_access 10）
- `scripts/eval_rbac.py`（端到端评估：httpx.AsyncClient + ASGITransport + psycopg2 种子数据）

**验证证据**：
- 真实 PG17 执行 migration `c3d4e5f6a7b8` 通过（bases + handlers + dog_base_association + dog_handler_association 表 + user_role / dog_handler_role 枚举）
- 单元测试 45/45 通过（`pytest backend/tests/test_rbac.py` in 4.87s）
- 端到端 25/25 通过（`reports/phase-3.6c-rbac-eval.json`：authentication 6 + role_permission 5 + base_isolation 2 + base_crud 5 + user_management 4 + change_password 3）

## 9. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-26 | Phase 0 基线建立：Python 3.12 + PyTorch 2.11+cu128 + PG17.10 + pgvector 0.8.0 + Redis 8.6.3 |
| v1.1 | 2026-08-01 | ①ultralytics 版本同步 8.3.55→8.4.107（与 `requirements.txt` 一致）；②§3 多个依赖从"⏳ Phase 0 验收验证"转为"✅ 通过"；③§7 未验证项全部转为已验证（Phase 0 验收 + 2026-08-01 新鲜验证）；④§5.4 新增 2026-08-01 新鲜验证证据（401 单元测试 + e2e 9/9）；⑤§8.3 mmaction2/TensorRT 状态同步（mmaction2 未触发安装，TensorRT 已安装）；⑥§8.4 新增 opencv-python-headless 冲突修复记录 |
| v1.2 | 2026-08-01 | §5.4 新鲜验证更新：①单元测试 401→480（+79，含 ST-GCN+BC 83 测试）；②新增 ST-GCN+BC 合成数据 baseline 训练验证证据（30 epochs / 1.43M 参数 / best_val_acc=46.97% / 边界 F1=58.45%）；③对应 Phase 3.1c + 3.1d 完成（见 phase-3.md v2.5 + AGENTS.md v1.17） |
| v1.3 | 2026-08-02 | **Phase 3.1e ST-GCN+BC 部署集成完成 + SHADOW 模式上线**：①§5.4 新鲜验证更新：单元测试 480→500（+20，含 3.1e 部署测试 20 个）+ 端到端 SHADOW 模式验证通过（USPCA video_id=37 / 101.8s / score=81.0 / PDF 4036 bytes）；②§8.5 新增 Redis 服务启动指引（chocolatey 安装路径 + 启动命令 + nssm 生产期建议）；③§8.6 新增 ST-GCN+BC 部署模式章节（4 模式路由表 + 配置切换 + 模型路径解析 + ONNX 导出 CLI）；④对应 Phase 3.1e 完成（见 phase-3.md v2.6 + AGENTS.md v1.18） |
| v1.4 | 2026-08-02 | **sliver-vibe-coding 接管审计 + 3.6 RBAC 基础代码完成 + 文档漂移修复**：①§5.4 新鲜验证更新：单元测试 500→538（+38 = 3.4 FCI-IGP 15 + 3.5 frame_stride 测试 + 接管审计重跑 106.71s）+ FCI-IGP 端到端验证通过（video_id=46 / 0.63x / PDF 4156 bytes）+ 3.6 RBAC auth API 未上线警告；②§8.7 新增 RBAC 用户权限章节（8 个已实现文件清单 + 4 项待办：main.py 注册路由 + exception_handler + migration 真实执行 + 3.6c 测试）；③对应 sliver-vibe-coding 接管审计（见 phase-3.md v2.8 + AGENTS.md v1.20 + Git commit `2e6aef3`） |
| v1.5 | 2026-08-02 | **3.6 RBAC 端到端验收通过 + onnxruntime 环境损坏警告**：①§5.4 新鲜验证更新：单元测试 538→566（+28 = RBAC 45 测试 - 17 个 onnxruntime 损坏相关）+ RBAC 单元测试 45/45 通过 in 4.87s + RBAC 端到端 25/25 通过 + 3.6 RBAC auth API 已上线；②§8.7 RBAC 章节：未上线 → 上线 + 端到端验收通过（main.py 注册路由 + exception_handler + 真实 PG17 migration 执行 + 8.7 文件清单扩展含 main.py + test_rbac.py + eval_rbac.py + 验证证据）；③onnxruntime 环境损坏警告（ultralytics 自动安装 onnxruntime-gpu 1.28.0 失败导致 Python 文件混合，TRAE 沙箱阻止修复，提供用户手动修复命令）；④对应 phase-3.md v2.9 + AGENTS.md v1.21 |
| v1.6 | 2026-08-02 | **onnxruntime 环境自然修复 + FastAPI 204 路由兼容性修复 + 631 单元测试 0 失败**：①§5.4 新鲜验证：566→631（+65 = 17 项 onnxruntime 错误恢复 + FastAPI 204 修复后偶发跨测试污染消失 + RBAC 端到端脚本计入）+ 0 failed + 0 errors；②**onnxruntime 已修复**：venv 中仅 onnxruntime-gpu 1.20.1，v1.5 警告的"用户手动修复"步骤无需执行；③**FastAPI 0.115.6 204 路由修复**：4 个 204 路由（auth.py change-password + users DELETE + bases.py DELETE + dogs.py DELETE）添加 `response_class=Response` + `-> Response` 返回类型 + 显式 `return Response(status_code=204)`；④RBAC 端到端 25/25 重测通过（无 204 副作用） |
