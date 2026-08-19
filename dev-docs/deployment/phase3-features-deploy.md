# Phase 3 新功能部署指南

> 阶段: Phase 3.8c
> Owner: 后端开发（`backend/`）+ ML 开发（`backend/ml/`）
> 依据: [phase-3.md §3.8c](../stages/phase-3.md) + [runtime.md](../runtime.md) + [jetson-deploy.md](jetson-deploy.md)
> 状态: v1.0（2026-08-02）
> 适用: 本地部署（Windows 优先，Phase 1-3 严格本地化）

本文档覆盖 Phase 3 四项新功能的服务端部署配置。Jetson 边缘部署见 [jetson-deploy.md](jetson-deploy.md)。

## 0. 前置条件

| 项 | 要求 | 验证 |
|----|------|------|
| Python venv | 3.12.12（`.venv`） | `.venv\Scripts\python.exe --version` |
| PostgreSQL | 17（端口 5433） | `psql -p 5433 -U k9system -d k9system` |
| Redis | ≥ 5.0（端口 6379） | `redis-cli ping` → PONG |
| 依赖 | `requirements.txt` 已安装 | `.venv\Scripts\pip.exe check` |

## 1. ST-GCN+BC 双轨行为识别（SHADOW 模式）

### 1.1 部署模式

[router.py](../../backend/ml/behavior/router.py) `DeployMode` 四模式：

| 模式 | 值 | 说明 | 当前状态 |
|------|----|------|---------|
| SHADOW | `shadow` | ST-GCN+BC 影子推理 + 规则引擎返回（仅记录对比） | ✅ 默认，已上线 |
| VOTE | `vote` | 双轨投票融合（高置信度优先） | ⏳ 待真实数据训练后启用 |
| PRIMARY_STGCN | `primary_stgcn` | ST-GCN+BC 主 + 规则引擎备（失败降级） | ⏳ 待真实数据训练 |
| RULE_ONLY | `rule_only` | 仅规则引擎（ST-GCN+BC 不可用时降级） | ✅ 自动降级路径 |

### 1.2 配置方式

通过环境变量 `BEHAVIOR_DEPLOY_MODE` 切换模式（[tasks.py](../../backend/workers/tasks.py) `_behavior_recognizer_singleton`）：

```bash
# 默认 SHADOW 模式（合成模型 46.97% 准确率，仅对比不接管）
set BEHAVIOR_DEPLOY_MODE=shadow

# 切换 RULE_ONLY（ST-GCN+BC 不可用时）
set BEHAVIOR_DEPLOY_MODE=rule_only
```

### 1.3 模型文件

| 模型 | 路径 | 来源 |
|------|------|------|
| ST-GCN+BC ONNX | `data/models/stgcn_bc/stgcn_bc_dog24.onnx` | `scripts/export_stgcn_bc_onnx.py` 导出 |
| ST-GCN+BC checkpoint | `runs/stgcn_bc_synthetic/best.pt` | `scripts/train_stgcn_bc.py` 训练 |

**路径解析**：`_resolve_stgcn_bc_path()` 优先 ONNX，回退 checkpoint，均不存在则 `RULE_ONLY`。

### 1.4 升级路径

```
SHADOW（当前）→ 真实数据训练 ≥ 85% → VOTE（置信度校准）→ PRIMARY_STGCN（接管主算法）→ 退役规则引擎
```

## 2. FCI-IGP 评分卡

### 2.1 配置文件

[backend/ml/scoring/configs/fci_igp.yaml](../../backend/ml/scoring/configs/fci_igp.yaml)：

| 字段 | 值 | 说明 |
|------|----|------|
| `scene` | `fci_igp` | 场景标识（`Video.VALID_SCENES` 已注册） |
| `igp_level` | `IGP1` / `IGP2` / `IGP3` | IGP 等级（影响行为权重） |
| 7 维权重 | 准确度 0.25 / 延迟 0.15 / 保持 0.15 / 搜索效率 0.15 / 注意力 0.10 / 胆量 0.10 / 步态 0.10 | 归一化 |
| `disqualifications` | 枪怯等硬约束 | 触发即 DQ（评分归零） |

### 2.2 激活方式

上传视频时指定 `scene=fci_igp`，推理 Worker 自动加载对应评分卡：

```
POST /api/videos/upload  scene=fci_igp  →  推理任务  →  评分引擎加载 fci_igp.yaml
```

### 2.3 验证

```bash
.venv\Scripts\python.exe scripts/eval_fci_igp.py
# 4 档全通过: Excellent 96.0 + Borderline 70.0 + Failing 30.0 + DQ 3/3
# 报告: reports/phase-3.4c-fci-igp-eval.json
```

## 3. RBAC 用户权限 + 多租户

### 3.1 数据库迁移

```bash
# 执行 RBAC migration（真实 PG17 已验证）
.venv\Scripts\python.exe -m alembic upgrade head
# migration: c3d4e5f6a7b8（handlers 表扩展 password_hash/base_id/is_superuser + base/dog_associations 表）
```

### 3.2 角色体系

[handler.py](../../backend/app/models/handler.py) `UserRole` 五角色（权限等级递增）：

| 角色 | 等级 | 权限 |
|------|------|------|
| VIEWER | 10 | 全基地只读（Phase 3.6a 新增） |
| HANDLER | 20 | 训导员：仅自有犬只/视频读写 |
| RESEARCHER | 30 | 科研人员：本基地只读 + 标注 |
| MANAGER | 40 | 基地管理员：本基地全部读写 |
| ADMIN | 100 | 系统管理员：全局读写 |

### 3.3 鉴权配置

[config.py](../../backend/app/core/config.py) RBAC 相关设置（通过环境变量或 `.env`）：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `AUTH_SECRET_KEY` | dev-secret（**生产必须覆盖**） | JWT 签名密钥（≥ 32 字符） |
| `AUTH_ALGORITHM` | HS256 | JWT 算法 |
| `AUTH_ACCESS_TOKEN_EXPIRE_MINUTES` | 480 | Token 有效期（8 小时工作日） |
| `AUTH_BCRYPT_ROUNDS` | 12 | bcrypt cost factor（10-14 推荐） |

### 3.4 API 端点

[main.py](../../backend/app/main.py) 已注册路由：

| 路由前缀 | 端点 | 鉴权 |
|---------|------|------|
| `/api/auth` | login / register(ADMIN) / users(ADMIN,MANAGER) / deactivate(ADMIN) / change-password | login 公开，其余需对应角色 |
| `/api/bases` | 基地 CRUD | MANAGER+ |
| `/api/dogs` | 犬只 CRUD | HANDLER+ |
| `/api/videos` | 视频上传/查询 | HANDLER+ |
| `/api/scores` | 评分查询 + `/compare` 对比 | HANDLER+ |
| `/api/scoring` | 评分卡管理 | MANAGER+ |

### 3.5 验证

```bash
.venv\Scripts\python.exe scripts/eval_rbac.py
# 45 单元测试 + 25 端到端测试（authentication 6 + role_permission 5 + base_isolation 2 + base_crud 5 + user_management 4 + change_password 3）
# 报告: reports/phase-3.6c-rbac-eval.json
```

## 4. 训练历史对比可视化

### 4.1 后端 API

[scores.py](../../backend/app/api/scores.py) `GET /api/scores/compare`：

| 参数 | 类型 | 说明 |
|------|------|------|
| `dog_ids` | string | 逗号分隔犬只 ID（1-10 只） |
| `date_from` | datetime | 起始时间（ISO 8601，可选） |
| `date_to` | datetime | 结束时间（ISO 8601，可选） |
| `standard` | string | 评分标准过滤: GA-T / USPCA / FCI-IGP / CUSTOM |
| `limit_per_dog` | int | 每犬最大数据点（1-100，默认 100） |

响应：`ScoreCompareResponse`（dogs + series + stats + dimension_labels）。

### 4.2 前端

| 文件 | 用途 |
|------|------|
| [CompareView.vue](../../frontend/src/views/CompareView.vue) | 对比页面（ECharts 折线趋势 + 雷达图 7 维 + 统计卡片 + 对比表） |
| [router/index.ts](../../frontend/src/router/index.ts) | `/compare` 路由 |
| [MainLayout.vue](../../frontend/src/layouts/MainLayout.vue) | 侧栏「训练对比」菜单（StatsChartOutline 图标） |

### 4.3 验证

```bash
.venv\Scripts\python.exe scripts/eval_scores_compare.py
# 41 端到端测试通过
# 报告: reports/phase-3.7c-scores-compare-eval.json
```

## 5. 服务启动顺序

```bash
# 1. PostgreSQL + Redis 已运行
# 2. 后端 API
.venv\Scripts\uvicorn.exe backend.app.main:app --host 127.0.0.1 --port 8000

# 3. Celery Worker（推理 + 评分）
.venv\Scripts\celery.exe -A backend.workers worker -l info -P solo

# 4. 前端
cd frontend && npm run dev
```

## 6. 完整验证命令

| 验证目标 | 命令 | 频率 |
|---------|------|------|
| 单元测试全集 | `.venv\Scripts\python.exe -m pytest backend/tests/ -q` | 每次提交 |
| 前端类型检查 | `cd frontend && npm run type-check` | 每次提交 |
| 项目守卫 | `.venv\Scripts\python.exe scripts/check_project_guardrails.py . --mode bootstrap` | Truth 变更时 |
| RBAC 端到端 | `.venv\Scripts\python.exe scripts/eval_rbac.py` | 权限变更时 |
| FCI-IGP 评分卡 | `.venv\Scripts\python.exe scripts/eval_fci_igp.py` | 评分卡变更时 |
| 训练对比 | `.venv\Scripts\python.exe scripts/eval_scores_compare.py` | API 变更时 |

## 7. 已知限制

- **ST-GCN+BC 合成 baseline**：当前 46.97% 准确率，仅 SHADOW 对比，不接管主算法（待真实数据训练 ≥ 85%）
- **多犬追踪 ReID**：OSNet 权重下载阻塞（Clash 代理），自动降级 `with_reid=False`
- **Jetson 部署**：硬件待采购，抽帧策略 + TRT 转换脚本已就绪（见 [jetson-deploy.md](jetson-deploy.md)）
- **延迟**：当前 1.13x（SHADOW 双轨 +2s，瓶颈在 pose 推理 96%），Jetson TRT FP16 + 抽帧优化至 ≤ 0.5x
