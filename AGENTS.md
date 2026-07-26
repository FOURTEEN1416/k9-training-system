# AGENTS.md — Agent Constitution

> 工作犬训练机器视觉识别系统的 Agent 行为宪法
> 基于 sliver-vibe-coding agent-constitution 框架
> 状态: ✅ v1.0
> 日期: 2026-07-26

## 0. 项目身份

**项目**：工作犬训练机器视觉识别系统（K9 Training Vision System）
**当前阶段**：立项完成，准备进入 Phase 0 基础设施
**Truth root**：`dev-docs/`
**主语言**：中文（代码注释遵循用户最新消息语言）

## 1. 顶层原则

### 1.1 四大技术原则（用户确认）

1. **广泛调研优先**：每个新模块启动前必须完成多方案调研（论文/GitHub/工业界），结果记录到 `dev-docs/research/`
2. **必要时自研**：当现有方案无法满足需求时，允许自主开发专用算法（DL/RL/Transformer/GNN/LLM 等）
3. **不择手段达成目标**：允许使用任意方法，不受预设技术栈限制
4. **深度优先而非浅层包装**：追求模型预测精度与评分质量，不满足于"能跑通"

### 1.2 sliver-vibe-coding 执行法则

- 路由先于行动：先选路由，再行动
- Owner 先于补丁：先确定责任模块，再修改代码
- 契约先于跨 owner 实现：先定接口契约，再跨模块实现
- 无新鲜验证，无完成声明：没有当前验证证据，不能声称完成
- 不引入兜底层、兼容 shim、重复 owner、生成文件编辑、投机抽象

### 1.3 用户偏好（来自 user_profile）

- 一次性解决问题，反对反复试错
- 透明操作，反对伪装
- 亲自完成，反对 subagent（除非必要）
- 质量优先于速度
- 清理临时脚本和死代码
- 同步文档以改进目标与约束

## 2. Owner Map（责任模块）

### 2.1 Truth 文档 Owner

| Truth 文档 | Owner | 修改触发 |
|-----------|-------|---------|
| `dev-docs/project-brief.md` | 立项阶段 | 产品方向变更 |
| `dev-docs/function-list.md` | 立项阶段 | 功能范围变更 |
| `dev-docs/technical-selection.md` | 技术选型阶段 | 技术栈变更 |
| `dev-docs/architecture.md` | 架构阶段 | 架构变更 |
| `dev-docs/database-design.md` | Phase 0 | Schema 变更 |
| `dev-docs/backend-boundary.md` | Phase 0 | API 边界变更 |
| `dev-docs/stages/*.md` | 阶段计划 | 阶段启动/关闭 |
| `dev-docs/decisions/*.md` | 任意阶段 | 重要决策 |

### 2.2 代码 Owner（Phase 0 建立后）

| 模块 | Owner | 路径 |
|------|-------|------|
| 前端 | 前端开发 | `frontend/` |
| 后端 API | 后端开发 | `backend/app/api/` |
| 后端服务 | 后端开发 | `backend/app/services/` |
| 推理 Worker | ML 开发 | `backend/workers/` |
| 姿态检测 | ML 开发 | `backend/ml/pose/` |
| 行为识别 | ML 开发 | `backend/ml/behavior/` |
| 评分引擎 | ML 开发 | `backend/ml/scoring/` |
| 数据库 | 后端开发 | `backend/alembic/` + `backend/app/models/` |

## 3. 项目硬约束（不可违反）

> 来源：`research/project-brief-source.md`（PRD v2.0）

- **本地部署**：Phase 1-3 严格本地化，无云依赖
- **三大标准兼容**：GA-T / USPCA / FCI-IGP
- **数据隐私**：公安/海关场景，数据不出本地
- **Windows 优先**：目标用户环境
- **22 种行为分类**：P0 基础 8 种 / P1 训练 8 种 / P2 高级 6 种
- **7 维评分**：准确度/延迟/保持/搜索效率/注意力/胆量/步态
- **端到端闭环**：视频 → 评分报告，无需人工干预

## 4. 工程约定

### 4.1 代码风格

- Python：PEP 8 + 类型注解（Python 3.12+）
- Vue：Composition API + `<script setup>` + TypeScript
- 提交信息：Conventional Commits（中文描述）
- 注释：遵循用户最新消息语言

### 4.2 依赖管理

- Python：`requirements.txt`（Phase 0）→ `pyproject.toml`（Phase 2+）
- Node.js：`package.json` + lockfile
- 模型权重：不进 Git（`.gitignore` 已配置）

### 4.3 测试约定

> 详细测试策略见 sliver-vibe-coding `testing-strategy.md`

- 推理模块：单元测试 + 集成测试（至少 1 段真实视频）
- API：FastAPI TestClient 单元测试
- 前端：Vitest 单元测试 + Playwright E2E（Phase 2+）
- 验证报告：归档到 `reports/phase-{N}-validation.md`

### 4.4 Git 约定

- 主分支：`main`
- 开发分支：`dev`
- 功能分支：`feature/<phase>-<module>`
- 不强制 PR（单人开发），但关键变更需 commit message 清晰

## 5. 决策流程

### 5.1 技术决策

1. **调研前置**：多方案调研，记录到 `dev-docs/research/`
2. **决策记录**：重要决策记录到 `dev-docs/decisions/NNNN-*.md`
3. **Truth 同步**：决策影响 truth 文档时同步更新

### 5.2 自研触发

> 用户确认：由用户逐案决策，spec 不设统一触发条件

1. 发现现有方案不理想
2. 用户决策是否启动自研
3. 记录到 `dev-docs/decisions/`
4. 允许使用任意方法（DL/RL/Transformer/GNN/LLM 等）

### 5.3 阶段升级

> 用户确认：不设硬性精度阈值，由用户判断是否升级

1. 阶段完成 → 验证报告归档 `reports/phase-{N}-validation.md`
2. 用户判断是否升级到下一阶段
3. 升级决策记录到 `dev-docs/decisions/`

## 6. 验证命令映射

| 验证目标 | 命令 | 频率 |
|---------|------|------|
| Python 单元测试 | `pytest backend/tests/` | 每次提交 |
| 前端单元测试 | `cd frontend && npm test` | 每次提交 |
| API 启动验证 | `uvicorn backend.app.main:app` | 开发期 |
| Celery 启动验证 | `celery -A backend.workers worker -l info` | 开发期 |
| 推理端到端 | `python -m backend.workers.inference --video <path>` | 阶段验证 |
| 项目守卫 | `python scripts/check_project_guardrails.py . --mode bootstrap` | Truth 变更时 |

## 7. 阶段映射

| 阶段 | 状态 | 关键交付 |
|------|------|---------|
| 立项 | ✅ 完成 | project-brief / function-list / technical-selection / architecture / constitution |
| Phase 0 基础设施 | ✅ 完成（2026-07-26 验收） | 环境 + DB schema + 前后端骨架（见 reports/phase-0-validation.md） |
| Phase 1 MVP | ⏳ 待启动（待用户升级决策） | YOLO26-pose + 规则 + PoseC3D + 评分 + 前后端 |
| Phase 2 核心 | ⏳ | 数据飞轮 + 16 行为 + USPCA |
| Phase 3 专业 | ⏳ | ST-GCN+BC + 多犬 + 3D + FCI-IGP |
| Phase 4 前沿 | ⏳ 按需 | LLM / Transformer-Mamba / RL |

## 8. 防漂移规则

以下情况触发漂移检查：
- 当前计划与 truth 文档冲突
- AI 提议当前阶段外的功能
- 请求影响技术栈/schema/权限/部署
- 变更无明确 owner 放置
- 验证证据与声称完成矛盾

漂移处理：
1. 记录变更内容
2. 识别受影响 truth 文档
3. 评估是否触及 foundation
4. 选择：拒绝 / 更新 truth / 重新设计阶段 / 停止等用户决策

## 9. 未解决问题

- ⏳ 中文路径迁移时机（见 `decisions/0001-chinese-path-migration-plan.md`）
- ⏳ YOLO26 AGPL 商用许可（Phase 5 前解决）
- ⏳ 工作犬数据采集方案（Phase 2 启动前调研）
- ⏳ MMAction2 Windows 兼容性（Phase 0 验证）

## 10. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-26 | 初始版本，基于立项阶段产出 |
| v1.1 | 2026-07-26 | Phase 0 验收通过，阶段映射 §7 状态更新 |
