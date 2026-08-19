# K9 Training Vision System（工作犬训练机器视觉识别系统）

用于分析工作犬训练视频的机器视觉识别系统：自动检测 24 个犬只关键点、识别 22 种训练行为，并按 FCI-IGP / USPCA / GA-T 标准生成端到端评分报告，无需人工干预。

> **数据隐私**：公安/海关场景，全流程本地部署，数据不出本地（无云依赖）。

---

## 核心能力

| 模块 | 说明 |
|------|------|
| 姿态检测 | YOLO26-pose，24 个犬只关键点，APTv2 犬科微调 Box mAP50 = 96.1% |
| 行为识别 | 22 类行为（P0 基础 8 / P1 训练 8 / P2 高级 6），ST-GCN+BC + Mamba 双轨 |
| 多犬追踪 | BoxMOT + OSNet ReID，APTv2 24 视频评估，最高 MOTA 93.33% / IDF1 96.55% |
| 评分引擎 | FCI-IGP 7 维评分 + USPCA + GA-T，5 级评级 + DQ 硬约束 |
| LLM 行为解释器 | 中文行为分析报告（行为概率 + 7 维评分 → 自然语言报告） |
| 端到端闭环 | 视频上传 → 关键点 → 行为 episodes → 评分 → PDF 报告 |

## 技术栈

- **后端**：Python 3.12 / FastAPI / SQLAlchemy / PostgreSQL 17 / Redis + Celery
- **深度学习**：PyTorch 2.11+cu128 / YOLO26-pose / ST-GCN+BC / Mamba (mamba_ssm) / OSNet ReID
- **前端**：Vue 3 + TypeScript + ECharts（训练历史对比可视化）
- **权限**：JWT + bcrypt，5 角色 RBAC（ADMIN/MANAGER/TRAINER/VIEWER/AUDITOR）+ 基地隔离

## 系统流程

```
视频上传 → YOLO26-pose 关键点检测 (T, 24, 3)
        → 行为识别（规则引擎 / ST-GCN+BC / Mamba 多轨路由）
        → 行为 episodes → 评分信号 → FCI-IGP / USPCA / GA-T 评分
        → PDF 评分报告 + LLM 中文行为分析
```

## 快速开始

### 环境要求

- Windows 10/11（目标用户环境优先）
- Python 3.12+、Node.js 18+、PostgreSQL 17、Redis

### 启动服务

```bash
# 1. Redis
redis-server --port 6379

# 2. PostgreSQL 17（127.0.0.1:5433）

# 3. Celery Worker
.venv\Scripts\python.exe -m celery -A backend.workers.celery_app worker --loglevel=info --pool=solo

# 4. FastAPI
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001

# 5. 前端（开发模式）
cd frontend && npm run dev
```

环境变量参考 `backend/.env.example`（数据库、Redis、LLM API 等配置）。

### 运行测试

```bash
python -m pytest backend/tests/ -q --ignore=backend/tests/integration   # 全量单元测试（605 passed）
python scripts/check_project_guardrails.py . --mode bootstrap           # 项目守卫
```

## 项目结构

```
backend/
  app/          # FastAPI 应用（API、auth、RBAC、配置）
  ml/           # 机器学习模块（pose / behavior / tracking / scoring）
  workers/      # Celery 推理任务（tasks.py 生产 pipeline）
  tests/        # 单元测试
frontend/       # Vue 3 + TypeScript 前端
dev-docs/       # 项目 truth 文档（阶段计划、ADR 决策、调研）
docs/           # 用户手册、代码维基
scripts/        # 训练 / 评估 / 验证脚本
reports/        # 阶段验收报告与评估结果
external/       # 第三方源码（VideoMamba 等，独立 git 仓库）
data/           # 数据集与上传视频（不入库，见 .gitignore）
runs/           # 模型权重与训练输出（不入库，见 .gitignore）
```

## 模型权重说明

模型权重**不进 Git 仓库**（`.gitignore` 已配置），需按阶段验证报告中的路径单独获取或训练：

| 模型 | 说明 |
|------|------|
| YOLO26-pose（APTv2 微调） | 犬只姿态检测，Box mAP50 = 96.1% |
| ST-GCN+BC（合成基线） | 22 类行为，best_val_acc = 46.97%（真实数据训练待标注数据） |
| Mamba 基线 | 21K 参数，best_val_acc = 85.61% |
| Mamba+BC（MS-Temba） | 多尺度膨胀 SSM + 边界检测（合成训练中） |

## 文档索引

- `dev-docs/HANDOVER.md` — 项目交接文档（最新状态、配置、测试命令、未完成任务）
- `dev-docs/AGENTS.md` — 项目宪法（Agent 行为规范）
- `dev-docs/stages/` — 阶段计划（phase-1 ~ phase-4）
- `dev-docs/decisions/` — ADR 决策记录（0001-0011）
- `docs/user-guide.md` — 用户手册
- `docs/CODE_WIKI.md` — 代码维基

## 已知限制与路线图

- **Phase 4 进行中**：LLM 行为解释器 ✅ / Transformer-Mamba 基线 ✅ / RL 评分优化 ⏳
- 真实标注数据训练（YouTube 自标 682 片段标注后训练 ST-GCN+BC ≥85%）
- YOLO26（AGPL 许可）商用许可确认（Phase 5 前解决）

## 许可证

本项目许可证**待定**（涉及 YOLO26 AGPL 及第三方模型许可评估，见 `dev-docs/decisions/`）。在许可确认前，仅限内部研究使用。