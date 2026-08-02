# AGENTS.md — Agent Constitution

> 工作犬训练机器视觉识别系统的 Agent 行为宪法
> 基于 sliver-vibe-coding agent-constitution 框架
> 状态: ✅ v1.22
> 日期: 2026-08-02

## 0. 项目身份

**项目**：工作犬训练机器视觉识别系统（K9 Training Vision System）
**当前阶段**：Phase 3 实施中（2026-07-30 ADR 0010 确认升级 + 启动；2026-07-30 完成 3.1b pyskl+K9Graph / 3.2b BoxMOT 多犬追踪 / 3.2c ReID 身份关联 / 3.3b InterPet4D 配对管线；2026-08-01 sliver-vibe-coding 接管审计修复运行时阻断；2026-08-01 完成 3.1c BC 头实现 + 3.1d ST-GCN+BC 训练评估管线（合成数据 baseline 40.91% / 边界 F1 58.45% / 1.43M 参数 / 30 epochs）+ 3.3c MotionBERT 17→24 适配 + 3.3d 3D 姿态评估（MPJPE=21.74mm ≤ 50mm 通过）；2026-08-02 完成 3.1e ST-GCN+BC 部署集成（双轨并行 SHADOW 模式 + ONNX 导出 + 4 模式路由层 + FCI-IGP pipeline）；2026-08-02 完成 3.4b FCI-IGP 评分卡 YAML 扩展 + 3.4c 评分卡验证（4 档全通过 + 端到端 0.63x）+ 3.5c 部分完成（抽帧策略 + TRT FP16 转换脚本就绪，Jetson 硬件待采购）；2026-08-02 sliver-vibe-coding 第二次接管审计：Git 检查点 commit `2e6aef3` 保护 58 文件 +12340 行 + 3.6 RBAC 文档漂移修复（3.6a migration+model ✅ / 3.6b auth API 代码 ✅ 但 main.py 未注册路由未上线 / 3.6c 未启动），新鲜验证通过：538 单元测试 + 2 skipped + 0 failed in 106.71s + FCI-IGP 端到端 9/9 通过；2026-08-02 完成 3.6b 收尾（main.py 注册 auth+bases 路由 + AuthError exception_handler + passlib→bcrypt 直接调用 + JWT sub 字符串化）+ 3.6c 多租户测试（scripts/eval_rbac.py + backend/tests/test_rbac.py 45 测试 + 真实 PG17 migration 执行 + 端到端 25/25 通过）；2026-08-02 完成 3.7 训练历史对比可视化（3.7a /api/scores/compare API + 3.7b CompareView 前端 + 3.7c 单元测试 48/48 + 端到端 41/41 通过）+ onnxruntime 自然修复 + FastAPI 0.115.6 204 路由兼容性修复（4 个 204 路由 response_class=Response），新鲜验证：631 单元测试 + 2 skipped + 0 failed in 98.40s）
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
| Phase 3 专业 | 🔄 实施中（2026-07-30 启动，3.1b+3.1c+3.1d+3.1e+3.2b+3.2c+3.3b+3.3c+3.3d+3.4b+3.4c+3.5c(部分)+3.6a+3.6b+3.6c+3.7a+3.7b+3.7c 已完成，Git 检查点 `2e6aef3`，见 `dev-docs/stages/phase-3.md` v3.0） | ST-GCN+BC + 多犬 + 3D + FCI-IGP + Jetson + 用户权限 + 训练历史对比 |
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
- ✅ 运行时阻断修复（2026-08-01 sliver-vibe-coding 接管审计）：①cv2 冲突——`.venv` 同时安装 opencv-python-headless 4.14 + opencv-python 5.0，Python 取 headless 版缺 imshow/imwrite，导致 ultralytics 导入崩溃 → API 完全无法启动，修复：卸载 headless 保留 opencv-python，同步 requirements.txt；②annotation 列名不匹配——模型 `source` 属性未映射到 DB 列 `annotation_source`（迁移 b2c3d4e5f6a7），查询报列不存在，修复：`mapped_column("annotation_source", ...)` 显式映射；新鲜验证：401 单元测试通过 + phase2_6_e2e_test 9/9 通过（USPCA 闭环 + 延迟 0.88x + PDF 报告）
- ✅ 3.6 RBAC 文档漂移修复（2026-08-02 sliver-vibe-coding 第二次接管审计）：代码已存在但 truth 标记 ⏳ 未启动——①3.6a migration `c3d4e5f6a7b8` + handler.py UserRole 5 角色 + base_entity.py + dog_associations.py 代码完成（migration 未在真实 PG17 验证）；②3.6b auth.py 4 端点 + core/security.py JWT+bcrypt + core/deps.py 5 依赖项代码完成，**main.py 未注册 auth/bases 路由未上线**；③3.6c 多租户测试未启动；修复：phase-3.md §3.6 状态修正 + §6.6 验收清单 ⏳→🔄 + v2.8 修订历史 + AGENTS.md/runtime.md/dev-docs README.md/stage-plan.md 同步；Git 检查点 commit `2e6aef3` 保护 58 文件 +12340 行；新鲜验证：538 单元测试 + 2 skipped + 0 failed in 106.71s
- ✅ 3.6 RBAC 端到端验收通过（2026-08-02）：①3.6b 收尾——main.py 注册 auth+bases 路由 + AuthError exception_handler + passlib→bcrypt 直接调用（解决 passlib 1.7.4 与 bcrypt 5.0.0 不兼容 `AttributeError: module 'bcrypt' has no attribute '__about__'`）+ JWT sub 字符串化（PyJWT 2.x 规范）+ auth.py 端点扩展（register/users/deactivate/change-password）；②3.6c 多租户测试——scripts/eval_rbac.py（httpx.AsyncClient+ASGITransport+psycopg2 种子数据）+ backend/tests/test_rbac.py 45 单元测试 + 真实 PG17 执行 migration `c3d4e5f6a7b8` 验证通过；③新鲜验证：566 单元测试 + 2 skipped + 5 failed + 12 errors（onnxruntime 环境损坏预存问题，与 RBAC 无关，TRAE 沙箱阻止修复）+ RBAC 单元测试 45/45 + RBAC 端到端 25/25（authentication 6 + role_permission 5 + base_isolation 2 + base_crud 5 + user_management 4 + change_password 3，见 reports/phase-3.6c-rbac-eval.json）

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
| v1.15 | 2026-08-01 | **sliver-vibe-coding 接管审计 + 运行时阻断修复**：①**接管只读审计**（不依赖现有报告，基于实际代码/运行时证据）：发现 API 无法启动（ultralytics/cv2 冲突）+ 30+ 文件未提交无备份 + 文档漂移；②**Git 检查点保护**（commit 3d49f54，102 文件 +22302 行，external/ 加入 .gitignore）；③**运行时阻断修复**（commit 956f8e6）：cv2 冲突（卸载 opencv-python-headless）+ annotation 列名不匹配（mapped_column 显式映射）；④**新鲜验证通过**：401 单元测试 + 2 skipped + 0 failed + phase2_6_e2e_test 9/9 通过（USPCA 闭环 + 延迟 0.88x + PDF 报告）；⑤§0 当前阶段 + §9 新增运行时阻断修复记录 |
| v1.16 | 2026-08-01 | **文档审计修复（Phase 3 状态同步）**：①**文档全面审计**：对照 `dev-docs/stages/phase-3.md` v2.4 + `stage-plan.md` v1.1 + `architecture.md` v1.1 + `runtime.md` v1.1 + `reports/phase-2-validation.md` 发现 AGENTS.md §0/§7 Phase 3 状态描述滞后（仍为"⏳ 启动中"，实际已完成 3.1b+3.2b+3.2c+3.3b 四个子任务）；②**§0 当前阶段**：`启动中` → `实施中`，补全子任务完成清单（3.1b pyskl+K9Graph / 3.2b BoxMOT 多犬追踪 / 3.2c ReID 身份关联 / 3.3b InterPet4D 配对管线）；③**§7 阶段映射**：Phase 3 行 `⏳ 启动中` → `🔄 实施中`，补全子任务完成引用；④同步 `architecture.md` v1.1（PostgreSQL 17 + 模块结构修正 + tracking 模块）+ `runtime.md` v1.1（ultralytics 8.4.107 + opencv 冲突修复记录 + 401 单元测试新鲜验证）+ `dev-docs/README.md`（ADR 0001-0010 完整索引 + 阶段状态同步） |
| v1.17 | 2026-08-01 | **Phase 3.1c + 3.1d ST-GCN+BC 训练评估管线完成**：①**3.1c BC 头实现**（2026-07-30 已完成）：`bc_head.py::BCHead` + `loss.py::STGCNBCLoss` + `model.py::STGCNBC` + `stgcn.py` ST-GCN++ 主干（UnitGCN + MSTCN + STGCNBlock × 10），38 个 3.1c 单元测试覆盖；②**3.1d 训练评估管线**：`dataset.py::STGCNBCDataset`（pyskl pickle + 内存双模式 + 数据增强 + withers 归一化）+ `make_synthetic_dataset`（22 类合成数据）+ `trainer.py::STGCNBCTrainer`（AdamW + Cosine + warmup + AMP + 早停 + 检查点）+ `scripts/train_stgcn_bc.py` + `scripts/eval_stgcn_bc.py`（准确率 + 22 类 P/R/F1 + 混淆矩阵 + P0/P1/P2 分层 + IGP A/B/C 分层 + 边界 F1 + JSON 报告）；③**合成数据 baseline 验证通过**：30 epochs / 1.43M 参数 / best_val_acc=46.97% @ epoch 21 / 边界 F1=58.45% / 22 类基线 4.5% × 9 倍提升；④**新鲜单元测试验证**：480 passed + 2 skipped + 0 failed（含 ST-GCN+BC 83 测试，较 v1.15 的 401 增加 79 个）；⑤**评估脚本 bug 修复**：boundary_logits 时间维度下采样导致与 boundary_labels 形状不匹配，新增最近邻上采样对齐逻辑；⑥§0 当前阶段 + §7 阶段映射 + phase-3.md v2.5（3.1c/3.1d 状态 ⏳→✅ + §6.1 验收清单状态更新）同步 |
| v1.18 | 2026-08-02 | **Phase 3.1e ST-GCN+BC 部署集成完成（双轨并行 SHADOW 模式上线）**：①**ONNX 导出**（`backend/ml/behavior/stgcn_bc/export_onnx.py::export_onnx`：动态 batch+time 轴 + opset 17 + 一致性验证 1e-3 阈值兼容 MSTCN 膨胀卷积浮点误差）；②**双后端推理器**（`inference.py::STGCNBCInferer`：PyTorch / ONNX Runtime + 滑动窗口 + 边界检测 episode 分割 + softmax/sigmoid 数值稳定性 clip[-50,50]）；③**双轨路由层**（`router.py::BehaviorRecognizer`：4 模式 SHADOW 影子对比 / VOTE 投票 / PRIMARY_STGCN 主+规则备降级 / RULE_ONLY 仅规则引擎）；④**tasks.py 集成**（BehaviorRecognizer 单例 + `_resolve_stgcn_bc_path()` 优先 ONNX 回退 checkpoint + FCI-IGP pipeline `_run_fci_igp_pipeline()` + 22 类行为枚举映射扩展）；⑤**CLI 工具** `scripts/export_stgcn_bc_onnx.py`；⑥**ONNX 模型已导出**至 `data/models/stgcn_bc/stgcn_bc_dog24.onnx`（来源 `runs/stgcn_bc_synthetic/best.pt` epoch 21 best_val_acc=46.97%）；⑦**单元测试 20/20 通过**（`backend/tests/ml/test_stgcn_bc_deploy.py`: TestExportOnnx 4 + TestSTGCNBCInferer 6 + TestBehaviorRecognizer 8 + TestEpisodeSplit 2）；⑧**端到端 SHADOW 模式新鲜验证通过**（USPCA 视频 2700 帧/90s → video_id=37 → 101.8s → verdict=pass score=81.0 → PDF 4036 bytes；SHADOW 对比日志 `STGCN=1 RULE=1 common=0 stgcn_only=1 rule_only=1`）；⑨**新鲜单元测试全集**：500 passed + 2 skipped + 0 failed in 100.76s（较 v1.17 的 480 +20 = 3.1e 部署测试）；⑩**延迟分析**：SHADOW 双轨仅占 2s（< 2%），瓶颈在 pose 推理 98s（96%），1.13x 略超 1.0x 阈值，将通过 Phase 3.5 Jetson TRT FP16 + 抽帧策略优化至 ≤ 0.5x；⑪§0 当前阶段 + §7 阶段映射 + phase-3.md v2.6（3.1e 状态 ⏳→✅ + §6.1 部署集成标记 + §6.3 3D 姿态 MPJPE 21.74mm ✅ + 修订历史 v2.5 补录 3.1c+3.1d+3.3c+3.3d + v2.6 3.1e 完成）同步 |
| v1.19 | 2026-08-02 | **Phase 3.4 FCI-IGP 评分卡 + 3.5c 部分完成 + 三线并行推进**：①**3.4b FCI-IGP 评分卡 YAML 扩展**（新增 `backend/ml/scoring/configs/fci_igp.yaml`：7 维权重 准确度 0.25 + 延迟 0.15 + 保持 0.15 + 搜索效率 0.15 + 注意力 0.10 + 胆量 0.10 + 步态 0.10，22 行为 100% 覆盖 IGP A/B/C 三阶段，DQ 硬约束块 + 5 级评分 Excellent/Very Good/Good/Satisfactory/Insufficient，Schema 扩展 `ScoringCardSpec.disqualifications` + `igp_level`，`Video.VALID_SCENES` 注册 fci_igp 场景，`fci_igp_signals.py` 独立模块消除 celery 依赖）；②**3.4c FCI-IGP 评分卡验证**（`scripts/eval_fci_igp.py` 4 档全通过: Excellent 96.0 + Borderline 70.0 + Failing 30.0 + DQ 3/3，报告 `reports/phase-3.4c-fci-igp-eval.json`；单元测试 15/15 通过 `backend/tests/integration/test_phase3_4_fci_igp_e2e.py`；端到端视频验证 video_id=46 scene=fci_igp verdict=pass score=78.9 57.0s/0.63x PDF 4156 bytes）；③**3.5c 部分完成**（抽帧策略 `backend/ml/pose/frame_stride.py` 线性/最近邻插值 + 自适应 stride 推荐 + `SPEEDUP_TOLERANCE=0.05` 边界处理；TRT FP16 转换脚本 `scripts/convert_trt_fp16.py` ONNX 解析 + builder 配置 + 动态 batch；Jetson 硬件待采购后实际 engine 转换 + 延迟测试）；④**ST-GCN+BC inference 空输入 NaN 修复**（`inference.py` T=0 早返回避免 `_normalize` 空切片均值 NaN）；⑤**TrainingStage 枚举校验**（test 脚本 P3→P2 修正）；⑥§0 当前阶段 + §7 阶段映射（Phase 3 子任务清单补 3.4b/3.4c/3.5c）+ phase-3.md v2.7（3.4b/3.4c ✅ + 3.5c 🔄 部分完成 + §6.4 FCI-IGP 通过 + v2.7 修订历史）同步 |
| v1.20 | 2026-08-02 | **sliver-vibe-coding 第二次接管审计 + Git 检查点 + 3.6 RBAC 文档漂移修复**：①**接管只读首检**（路由 `接管项目`，只读审计 Git/Truth/运行时/AI 债务，发现 30+ 未提交文件 + 3.6 RBAC 代码已存在但 truth 标记 ⏳）；②**Git 检查点保护**（commit `2e6aef3`，58 文件 +12340 行，保护 22 已修改 + 30 未跟踪文件，防止 Phase 3 已完成工作丢失）；③**3.6 RBAC 文档漂移修复**：§3.6a ⏳→✅（migration `c3d4e5f6a7b8` + handler.py UserRole 5 角色 + base_entity.py + dog_associations.py 代码完成，migration 未在真实 PG17 验证）；§3.6b ⏳→🔄（auth.py 4 端点 + core/security.py JWT+bcrypt + core/deps.py 5 依赖项代码完成，**main.py 未注册 auth/bases 路由未上线**）；§3.6c ⏳ 维持（未启动）；④**§6.6 验收清单** ⏳→🔄 部分完成；⑤**文档同步**：phase-3.md v2.7→v2.8 + runtime.md v1.3→v1.4（§5.4 538 测试 + §8.7 RBAC 章节）+ dev-docs/README.md（阶段映射 v2.8）+ stage-plan.md v1.1→v1.2；⑥**新鲜验证**：pytest 538 passed + 2 skipped + 0 failed in 106.71s（较 v1.19 的 500 +38 = 3.4 FCI-IGP + 3.5 frame_stride 测试 + 接管审计重跑） |
| v1.21 | 2026-08-02 | **3.6 RBAC 端到端验收通过（3.6b 收尾 + 3.6c 多租户测试）**：①**3.6b 收尾**——main.py 注册 auth+bases 路由 + `@app.exception_handler(AuthError)` JSONResponse + passlib→bcrypt 直接调用（解决 passlib 1.7.4 与 bcrypt 5.0.0 不兼容 `AttributeError: module 'bcrypt' has no attribute '__about__'`）+ JWT sub 字符串化（PyJWT 2.x 规范）+ auth.py 端点扩展（register ADMIN + users ADMIN/MANAGER + deactivate ADMIN + change-password）；②**3.6c 多租户测试**——scripts/eval_rbac.py（httpx.AsyncClient+ASGITransport 单 event loop + psycopg2 同步种子数据 + bcrypt 4 rounds 加速）+ backend/tests/test_rbac.py 45 单元测试（密码哈希 7 + JWT 7 + 异常 6 + UserRole 4 + check_base_access 11 + check_dog_access 10）+ 真实 PG17 执行 migration `c3d4e5f6a7b8` 验证通过；③**§9 新增 3.6 RBAC 端到端验收通过记录**；④**文档同步**：phase-3.md v2.8→v2.9（§3.6b 🔄→✅ + §3.6c ⏳→✅ + §6.6 ✅ 通过 + v2.9 修订历史）+ runtime.md v1.4→v1.5（§5.4 566 测试 + §8.7 RBAC 上线 + onnxruntime 损坏警告）+ dev-docs/README.md + stage-plan.md v1.2→v1.3；⑤**新鲜验证**：pytest 566 passed + 2 skipped + 5 failed + 12 errors（onnxruntime 环境损坏预存问题，与 RBAC 无关，TRAE 沙箱阻止修复，提供用户手动修复命令 `pip uninstall onnxruntime -y && pip install onnxruntime==1.20.1`）+ RBAC 单元测试 45/45 + RBAC 端到端 25/25（authentication 6 + role_permission 5 + base_isolation 2 + base_crud 5 + user_management 4 + change_password 3，见 reports/phase-3.6c-rbac-eval.json） |
| v1.22 | 2026-08-02 | **3.7 训练历史对比可视化完成 + onnxruntime 修复 + FastAPI 204 路由兼容性修复**：①**3.7a 历史评分查询 API 扩展**（`backend/app/api/scores.py` 新增 `GET /api/scores/compare` 端点：dog_ids 逗号分隔（1-10 只）+ date_from/date_to + standard 别名解析 GA-T/USPCA/FCI-IGP/CUSTOM + limit_per_dog + 去重保序 + 参数校验 400 + 资源校验 404；`backend/app/schemas/common.py` 新增 DogBrief/ScorePoint/DogScoreSeries/DogScoreStats/ScoreCompareResponse；辅助函数 `_compute_trend_slope` 线性回归斜率 分/天 + `_avg_dimensions` 维度平均 + `_parse_standard` 别名解析）；②**3.7b 对比可视化前端**（`frontend/src/views/CompareView.vue` 犬只多选 + 日期范围 + 标准过滤 + ECharts 折线趋势图 + 雷达图 7 维对比 + 统计卡片 + 对比表；`frontend/src/api/index.ts` TS 类型；`frontend/src/router/index.ts` `/compare` 路由；`frontend/src/layouts/MainLayout.vue` 导航入口 StatsChartOutline 图标）；③**3.7c 端到端测试**：`backend/tests/test_scores_compare.py` 48/48 单元测试通过（路由注册 4 + trend_slope 7 + avg_dimensions 4 + score_dimensions 3 + parse_standard 15 + 参数校验 6 + 业务逻辑 9）+ `scripts/eval_scores_compare.py` 41/41 端到端通过（参数校验 6 + 资源存在性 1 + 单犬时序 11 + 多犬对比 6 + 标准过滤 4 + 日期范围 4 + 空结果 3 + 去重保序 2 + 维度标签 1 + 部分维度 3，见 `reports/phase-3.7c-scores-compare-eval.json`）；④**onnxruntime 自然修复**（venv 中仅 onnxruntime-gpu 1.20.1，v2.9 报告的 1.28.0/1.20.1 混合已不存在）；⑤**FastAPI 0.115.6 204 路由兼容性修复**（4 个 204 路由: auth.py /change-password + /users/{id} DELETE + bases.py /{id} DELETE + dogs.py /{id} DELETE 添加 `response_class=Response` + `-> Response` 返回类型 + 显式 `return Response(status_code=204)`，依据 fastapi/routing.py:467 Response 子类跳过 response_model 检查）；⑥**§6.7 验收清单** ⏳→✅ 通过；⑦**新鲜单元测试全集**：631 passed + 2 skipped + 0 failed in 98.40s（较 v1.21 的 566 +65 = 17 项 onnxruntime 恢复 + FastAPI 204 修复消除偶发跨测试污染 + 48 scores_compare 测试）；⑧**文档同步**：phase-3.md v2.9→v3.0 + §0 当前阶段 + §7 阶段映射（Phase 3 子任务清单补 3.7a/3.7b/3.7c）+ §10 修订历史 v1.22；§3.7a/3.7b/3.7c 标记 ✅ 完成 |
