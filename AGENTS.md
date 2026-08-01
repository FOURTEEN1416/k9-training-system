# AGENTS.md — Agent Constitution

> 工作犬训练机器视觉识别系统的 Agent 行为宪法
> 基于 sliver-vibe-coding agent-constitution 框架
> 状态: ✅ v1.9
> 日期: 2026-07-29

## 0. 项目身份

**项目**：工作犬训练机器视觉识别系统（K9 Training Vision System）
**当前阶段**：Phase 3 启动中（2026-07-30 ADR 0010 确认升级，Phase 2 核心 ✅ 完成 + 1.2f 数据源问题彻底解决）
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
- 调研搜索强制 GitHub-First（零容忍硬规则）：本项目**完全禁止**使用 `WebSearch` 工具进行技术调研——没有「最后补充」、没有「仅官方域名」、没有任何例外。所有调研**必须**按以下顺序进行：① 先调用 `github-search-strategy` skill，按 Directory-First 流程（`sindresorhus/awesome` → `awesome-<topic>` → 从目录发现具体仓库 → 逐项验证活跃度/Issues）；② 需要深度抓取页面内容时调用 `browser-automation` skill 的爬虫（Crawl4AI / ScrapeGraphAI / Playwright）抓取 GitHub repo/issues、arxiv、官方文档、HuggingFace Discussions、Reddit r/computervision 等权威源。违反此规则（即对技术调研任务调用 `WebSearch`）立即触发 §8 漂移处理流程。唯一例外：用户对特定查询的明确书面许可

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
| Phase 1 MVP | ✅ 完成（2026-07-28 验收，带条件） | YOLO26-pose + 规则引擎 + 评分 + 前后端 + 端到端闭环（见 reports/phase-1-validation.md v1.2 + ADR 0005 v1.1） |
| Phase 2 核心 | ✅ 完成（2026-07-30 验收，见 reports/phase-2-validation.md + ADR 0010） | 数据飞轮 + 16 行为 + USPCA + 1.2f 补强 |
| Phase 3 专业 | ⏳ 启动中（2026-07-30 ADR 0010 确认升级） | ST-GCN+BC + 多犬 + 3D + FCI-IGP + Jetson + 用户权限 |
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

- ✅ 中文路径迁移（见 `decisions/0001-chinese-path-migration-plan.md`，已完成迁移至 `D:\Desktop\k9-training-system`）
- ⏳ YOLO26 AGPL 商用许可（Phase 5 前解决）
- ✅ 工作犬数据采集方案（Phase 1 末改用 Dog-Pose 微调 + 公开数据集评估，未触发 300 张采集，见 ADR 0003 §2.4）
- ⏳ MMAction2 Windows 兼容性（Phase 1.3 未触发，1.2f 条件通过维持 PoseC3D 跳过决策，见 ADR 0006 v1.1）
- ⏳ 项目守卫脚本 `scripts/check_project_guardrails.py` 未实现（Phase 0 遗留，Phase 2 内补建）
- ✅ 1.6d 真实序列验证（InterPet4D kp_world 226/226 + 9/9 姿态指标变异，2026-07-28 通过，见 `reports/phase-2-prereq-1.6d-validation.md`）
- ✅ 1.2f 真实数据复核（InterPet4D v1 无视频/标签，采用三层降级验证: 合成 92.9% + kp_world 管线 100% + 真实视频延后，2026-07-28 条件通过，见 `reports/phase-2-prereq-1.2f-validation.md` v1.1 + ADR 0006 v1.1）
- ⏳ YouTube 玩球视频物体检测验证（1.6d 补充，Phase 2.0c 内推进）
- ✅ 1.2f 真实视频补强（双轨验证通过 2026-07-30: 轨道 A dog-pose val mAP50=92.2% + 部署一致性 0.19% + 轨道 B 合成行为 96.3%；AK 数据域不匹配已证实 YOLO 检测率 0%-19%，image.tar.gz 42GB 取消下载；YouTube 5 视频 + LS project id=1 待人工标注为后台并行非阻塞项，见 ADR 0008 v1.6 + reports/phase-2.0c-1_2f-dogpose-val-map.md）
- ✅ Phase 2 范围收敛（2026-07-29 sliver-vibe-coding 项目体检）：6 子阶段 → 3 核心（2.1 数据飞轮 + 2.2 16 行为 + 2.3 USPCA）+ 1 后台（2.0 数据补强）+ 1 收尾（2.6 系统集成）；2.4 多租户延后 Phase 3；2.5 训练历史合并 2.6；时间约束 6 周 → 3-4 周，见 phase-2.md v2.0
- ✅ 用户决策升级 Phase 2（2026-07-28 ADR 0007 确认升级，Case A + Case B 并行）

## 10. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-26 | 初始版本，基于立项阶段产出 |
| v1.1 | 2026-07-26 | Phase 0 验收通过，阶段映射 §7 状态更新 |
| v1.2 | 2026-07-26 | Phase 1 升级决策（ADR 0003）确认，阶段映射 §7 状态更新 |
| v1.3 | 2026-07-26 | 项目体检（PROJECT_HEALTH_CHECK_2026-07-26）后修正 §0 当前阶段 + §9 未解决问题（4 处过时字段） |
| v1.4 | 2026-07-28 | §1.3 用户偏好新增「调研搜索强制 GitHub-First（零容忍硬规则）」：完全禁止 WebSearch 用于技术调研（无任何例外/漏洞），强制 github-search-strategy + browser-automation 流程，违反立即触发 §8 漂移处理 |
| v1.5 | 2026-07-28 | Phase 1 MVP 验收通过（带条件），§7 阶段映射更新：Phase 1 ✅ 完成（2026-07-28 验收，见 reports/phase-1-validation.md + ADR 0005），Phase 2 ⏳ 启动条件待满足（1.6d + 1.2f 真实验证）。§9 未解决问题：工作犬数据采集方案 ✅ 解决（未触发 300 张采集），新增 1.6d/1.2f Phase 2 启动前置条件 |
| v1.6 | 2026-07-28 | 1.6d/1.2f 验证完成: §9 未解决问题更新 — 1.6d ✅ 通过（226/226 + 9/9 姿态指标变异），1.2f ✅ 条件通过（数据限制: InterPet4D v1 无视频/标签，三层降级验证）。新增 YouTube 物体检测 + 真实视频准确率作为 Phase 2 内推进项。Phase 2 启动条件达成，待用户决策（ADR 0005 v1.1 + ADR 0006 v1.1） |
| v1.7 | 2026-07-28 | Phase 2 启动: §0 当前阶段 + §7 阶段映射更新（Phase 2 ✅ 启动中，ADR 0007）。§9 未解决问题: 用户决策升级 ✅ 解决，新增 1.2f 补强（ADR 0008: Animal Kingdom + YouTube 自标 + Label Studio）。Case A + Case B 并行推进 |
| v1.8 | 2026-07-28 | Animal Kingdom 数据集已获取（用户提供 Google Drive 链接，无需申请）。§9 未解决问题更新: 1.2f 补强策略调整 — AK 犬类样本严重不足（219/30100，sit=2/down=0/stand=0/come=0），YouTube 自标升级为主路径。pose_estimation/dataset.tar.gz (2.46GB) 已下载验证，video.tar.gz (15.6GB) 因 Google Drive 限流待重试。见 ADR 0008 v1.1 + reports/phase-2-ak-canine-mapping.json |
| v1.9 | 2026-07-29 | §9 未解决问题更新: 1.2f 补强 — Label Studio 部署完成（v1.23，session auth，project id=1，5 视频 + 5 预标注任务全部 200 OK），待人工标注。见 ADR 0008 v1.3 §2.3.2 部署详情 |
| v1.10 | 2026-07-29 | §9 未解决问题更新: 1.2f 补强 — Animal Kingdom video.tar.gz 下载完成（15.59 GB，gzip 完整性验证通过，curl + Clash 代理断点续传）。见 ADR 0008 v1.4 |
| v1.11 | 2026-07-29 | §9 未解决问题更新: 1.2f 补强 — video.tar.gz 犬类视频提取完成（211 个视频，111.2 MB，25 种行为: Wolf 142/Wild Dog 35/Dog 31/Dingo Dog 2/African Painted Dog 1）。image.tar.gz 后台下载启动（42GB，Google Drive 配额限制，每小时重试最多 24h）。见 ADR 0008 v1.5 |
| v1.12 | 2026-07-29 | **sliver-vibe-coding 项目体检收敛**：①清理漂移（_browser_profile 删除 + reports/_ls_*.png 移到 docs/screenshots/ + scripts/_setup_ls_project.py 重命名为 setup_label_studio.py + image.tar.gz 后台下载停止）；②Phase 2 范围收敛（phase-2.md v2.0）：6 子阶段 → 3 核心 + 1 后台 + 1 收尾，2.0 降级为后台并行任务（不阻塞 2.1/2.2/2.3 主线），2.4 多租户延后 Phase 3，2.5 训练历史合并 2.6，时间约束 6 周 → 3-4 周；③§9 新增 Phase 2 范围收敛 ✅ 解决项 |
| v1.13 | 2026-07-30 | **1.2f 数据源问题彻底解决**：①**AK 数据域不匹配诊断**（YOLO26-pose 在 AK 野生动物 Wolf/Wild Dog 上检测率 0%-19%，不适合项目模型行为级验证）；②**image.tar.gz 42GB 正式取消下载**（同源野生动物图像无价值，用户确认）；③**1.2f 验证数据源切换**（dog-pose val 1703 张同域数据）；④**双轨验证通过**（轨道 A 姿态级 mAP50=92.2% + 部署一致性 0.19% + 轨道 B 行为级 96.3%）；⑤§9 未解决问题: 1.2f 真实视频补强 ⏳→✅。见 ADR 0008 v1.6 + reports/phase-2.0c-1_2f-dogpose-val-map.md + phase-2.md v2.6 |
| v1.14 | 2026-07-30 | **Phase 2 正式关闭 + Phase 3 启动**：①**Phase 2 验收报告归档** `reports/phase-2-validation.md` v1.0（§6.1-§6.8 全部证据 + v2.0 范围收敛说明 + 1.2f 双轨验证）；②**ADR 0010 创建**（Phase 2 → Phase 3 升级决策，含 v2.0 收敛说明：用户权限延后 Phase 3 + 训练历史合并 2.6 + 1.2f 数据源彻底解决）；③§0 当前阶段 + §7 阶段映射更新（Phase 2 ✅ 完成，Phase 3 ⏳ 启动中）；④phase-2.md v2.7 + stage-plan.md 同步。Phase 3 范围：ST-GCN+BC + 多犬 + 3D + FCI-IGP + Jetson + 用户权限 + 训练历史对比 |
