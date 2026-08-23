# Code Wiki — 工作犬训练机器视觉识别系统

> 代码级参考文档。当前主线为 P0 数据解阻；Phase 4 保留为并行实施工作。
> Phase 0-3 已完成工程关闭，但真实行为数据、Jetson 实机和多犬 GT 仍有明确验证边界。

## 1. 文档定位

本 Wiki 描述当前代码结构、运行入口、模块责任和验证边界。阶段决策以 `dev-docs/` 为准，历史验收证据以 `reports/` 为准。

能力状态使用以下五级标记：

| 标记 | 含义 |
|------|------|
| 已实现 | 代码和接口存在，并有针对性验证 |
| 合成验证 | 仅证明合成数据管线可运行，不代表真实泛化 |
| 公开真实验证 | 在公开真实数据上完成验证 |
| 实验性/SHADOW | 仅并行对比或实验使用，不接管主算法 |
| 硬件未验证 | 代码或脚本存在，但目标硬件尚未实测 |

## 2. 当前研究与产品边界

研究路线：

```text
规则种子 → AimCLR → SMQ → 姚青 JIA 方法迁移 → TCL
→ 主动学习 100-200 片段 → ST-GCN+BC / Mamba 验证
```

产品系统是研究方法的验证平台，主要闭环为：

```text
视频上传 → YOLO26-pose → 行为识别 → 评分卡 → PDF 报告
```

当前不应把以下结果混写：

- ST-GCN+BC 合成基线：46.97%，实验性/SHADOW；
- Mamba 合成基线：85.61%，尚未完成真实数据验证；
- 规则引擎合成结果：只能作为规则管线基线；
- FCI-IGP 0.63x：单次端到端验证结果，不等于所有场景实时保证；
- SHADOW 1.13x：未达到 1.0x 当前阈值，Jetson 仍未实测。

## 3. 运行入口

| 服务 | 命令 | 默认地址 |
|------|------|----------|
| FastAPI | `.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001` | `http://127.0.0.1:8001` |
| Celery Worker | `.venv\Scripts\python.exe -m celery -A backend.workers.celery_app worker --loglevel=info --pool=solo` | Redis DB 1/2 |
| 前端开发服务器 | `cd frontend; npm run dev` | Vite 默认端口 |
| PostgreSQL | Windows 服务 `postgresql-17` | `127.0.0.1:5433` |
| Redis | 本地服务 | `127.0.0.1:6379` |

运行时基线：Python 3.12.12、Node.js 22.16.0（20+ 兼容范围）、PostgreSQL 17、Redis 服务端 8.6.3、Python Redis 客户端 5.2.1。完整版本以 `dev-docs/runtime.md` 和 `backend/requirements.txt` 为准。

## 4. 后端模块

| 模块 | Owner | 入口/职责 |
|------|-------|-----------|
| API | `backend/app/api/` | FastAPI 路由、认证、基地、视频、评分、模型 |
| 服务 | `backend/app/services/` | PDF 报告和业务服务 |
| 推理 Worker | `backend/workers/tasks.py` | 视频摄取、姿态、行为、评分闭环 |
| 姿态检测 | `backend/ml/pose/` | YOLO26-pose、关键点处理、3D lifting |
| 行为识别 | `backend/ml/behavior/` | 规则引擎、ST-GCN+BC、Mamba、路由 |
| 评分引擎 | `backend/ml/scoring/` | YAML 评分卡、条件求值、标准映射 |
| 多犬追踪 | `backend/ml/tracking/` | BoxMOT、OSNet ReID、轨迹指标 |
| 数据库 | `backend/app/models/` + `backend/alembic/` | ORM、迁移、租户与权限模型 |

## 5. 当前 API 入口

| 路由 | 用途 |
|------|------|
| `/health` | 健康检查 |
| `/api/auth/*` | 登录、注册、用户管理、密码修改 |
| `/api/bases/*` | 基地 CRUD 与隔离 |
| `/api/videos/*` | 视频上传、状态轮询、报告下载 |
| `/api/dogs/*` | 犬只档案 |
| `/api/scores/*` | 评分查询、历史对比 |
| `/api/scoring/*` | YAML 评分卡读取、修改、预览 |
| `/api/models/*` | 模型注册与激活 |
| `/api/llm/explain` | Phase 4 实验室行为解释器，外部 API 仅临时豁免 |

## 6. 标注与数据飞轮

YouTube 自标数据源和原 Label Studio 专用服务已废弃。当前流程不绑定某个视频来源：

1. 核心无标签池：InterPet4D、Animal Kingdom、APTv2；
2. 可选扩展：SyDog-Video、samtwl 等；
3. 规则种子、AimCLR、SMQ、TCL 生成候选片段；
4. 主动学习按不确定性抽取 100-200 片段；
5. Label Studio 可作为通用人工复核工具，输入输出协议需由当前 P0 管线提供；
6. 训练并按合成、公开真实、真实 K9 三层数据域报告指标。

## 7. 验证边界

- Phase 3 工程验收报告：`reports/phase-3.8a-e2e-eval.json`；其中真实行为数据、Jetson、多犬 GT 明确为 SKIP。
- 延迟证据：`reports/phase-3.8b-latency-eval.json`。
- Mamba 合成基线：`reports/phase-4.2-mamba-baseline.md`。
- 当前证据总索引：`reports/documentation-evidence-index.md`。
- 冲突登记：`reports/documentation-conflict-register.md`。

## 8. 前端结构

| 路径 | 用途 |
|------|------|
| `frontend/src/views/UploadView.vue` | 视频上传与场景选择 |
| `frontend/src/views/ReportView.vue` | 报告查看与 PDF 下载 |
| `frontend/src/views/HistoryView.vue` | 历史记录 |
| `frontend/src/views/CompareView.vue` | 训练历史趋势、雷达与统计对比 |
| `frontend/src/views/AdminView.vue` | 模型、评分卡与管理 |
| `frontend/src/router/index.ts` | 路由注册 |
| `frontend/src/api/` | API 封装与 TypeScript 类型 |

## 9. 验证命令

```powershell
python -m pytest backend/tests/ -q --ignore=backend/tests/integration
python scripts/check_project_guardrails.py . --mode bootstrap
```

前端类型检查和端到端验证命令见 `docs/deployment.md`。不要使用本 Wiki 中未列出的旧 Label Studio 服务启动命令作为当前流程。
