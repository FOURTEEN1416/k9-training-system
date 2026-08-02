# dev-docs/ — 内部开发 truth root

本目录是工作犬训练系统的内部开发 truth root，存放 AI 与项目团队的规则、决策、架构、阶段计划等 truth 文档。

**公开文档与内部 truth 分离**：本目录内容不应直接推送到公开仓库（已在 `.gitignore` 中排除 `.trae/`，但 `dev-docs/` 默认入库；如需私有化，参见 `project-flow.md` 的 dual-repo 管理策略）。

> **修订说明（2026-08-02）**：本次修订同步至 Phase 3 v2.8 实施中状态（sliver-vibe-coding 接管审计后）。原文档多处理于"⏳ Phase 0"过时状态——实际 Phase 0/1/2 已完成验收（见 `reports/phase-{0,1,2}-validation.md` + AGENTS.md §7），Phase 3 已实施至 3.6b 部分（见 `stages/phase-3.md` v2.8：3.1b/c/d/e + 3.2b/c + 3.3b/c/d + 3.4b/c + 3.5c 部分 + 3.6a + 3.6b 部分完成）。`design/` / `quality/` / `acceptance/` / `database-design.md` / `backend-boundary.md` 等 truth 文档**未建立**（占位条目移除，避免误导）。

## 文档索引

### 项目基础 truth
| 文档 | 用途 | 状态 |
|------|------|------|
| `project-brief.md` | 项目立项 brief（用户、场景、MVP、非目标） | ✅ |
| `function-list.md` | 功能清单与复杂功能文档索引 | ✅ |
| `technical-selection.md` | 技术选型 truth（栈+框架+架构组合） | ✅ |
| `architecture.md` | 项目架构（拓扑、owner map、模块边界） | ✅（部分漂移，待同步 Phase 2/3 实际结构） |
| `runtime.md` | 运行时基线（包管理器、版本、启动命令） | ✅ Phase 0 已建立（v1.4，538 单元测试新鲜验证） |
| `stage-plan.md` | 大阶段计划（Phase 0-4） | ✅（v1.2，Phase 3 v2.8 状态同步） |

### 阶段 truth
| 文档 | 用途 | 状态 |
|------|------|------|
| `stages/phase-0.md` | Phase 0 基础设施计划 | ✅ 完成（2026-07-26 验收） |
| `stages/phase-1.md` | Phase 1 MVP 计划 | ✅ 完成（2026-07-28 验收，带条件） |
| `stages/phase-2.md` | Phase 2 核心计划 | ✅ 完成（2026-07-30 验收，v2.7） |
| `stages/phase-3.md` | Phase 3 专业计划 | 🔄 实施中（v2.8：3.1b/c/d/e + 3.2b/c + 3.3b/c/d + 3.4b/c + 3.5c 部分 + 3.6a + 3.6b 部分完成） |

### 设计/质量/验收 truth
> `design/` / `quality/` / `acceptance/` 目录**尚未建立**。原 README 列为占位条目但实际未创建——Phase 0-2 期间验收证据统一归档到 `reports/phase-{N}-validation.md`。如需建立独立 truth 目录，Phase 3+ 启动时按需创建。

### 决策记录
| 文档 | 用途 | 状态 |
|------|------|------|
| `decisions/0001-chinese-path-migration-plan.md` | 中文路径迁移计划 | ✅ 已完成 |
| `decisions/0002-runtime-stack-revision-blackwell.md` | 运行时栈修正（PyTorch 2.11+cu128） | ✅ 已确认 |
| `decisions/0003-phase-0-to-phase-1.md` | Phase 0 → Phase 1 升级 | ✅ 已确认 |
| `decisions/0005-phase-1-to-phase-2.md` | Phase 1 → Phase 2 升级 | ✅ 已确认 |
| `decisions/0006-dogmo-open-alternative.md` | DogMo 替代方案（InterPet4D） | ✅ 已确认 |
| `decisions/0007-phase-2-start.md` | Phase 2 启动 | ✅ 已确认 |
| `decisions/0008-1.2f-supplement-plan.md` | 1.2f 数据补强计划 | ✅ 已确认（v1.6） |
| `decisions/0009-data-strategy-four-paths.md` | 数据策略四路径 | ✅ 已确认 |
| `decisions/0010-phase-2-to-phase-3.md` | Phase 2 → Phase 3 升级 | ✅ 已确认 |

> **ADR 0004 说明**：编号 0004 预留给"PoseC3D 触发但失败时记录"（见 `phase-1.md` §6 + `0003-phase-0-to-phase-1.md` §55）。AGENTS.md §9 已确认 PoseC3D 跳过决策（1.2f 条件通过维持跳过），故 ADR 0004 **未触发创建**——这是设计预期，非缺失。

### 调研资料（只读参考）
| 文档 | 来源 |
|------|------|
| `research/RESEARCH_LITERATURE.md` | 论文综述 |
| `research/RESEARCH_OPENSOURCE.md` | 开源项目分析 |
| `research/RESEARCH_TECH_COMPARISON.md` | 技术对比矩阵 |
| `research/RESEARCH_IMPLEMENTATION.md` | 实现细节 |
| `research/RESEARCH_STANDARDS.md` | 标准映射 |
| `research/RESEARCH_COMMERCIAL.md` | 商业分析 |
| `research/RESEARCH_YOLO26_DEPLOY_DEEP.md` | YOLO26 部署深度分析 |
| `research/RESEARCH_PHASE1_STACK_DEEP.md` | Phase 1 技术栈深度分析 |
| `research/RESEARCH_STGCN_BC.md` | ST-GCN+BC 复现分析（Phase 3） |
| `research/RESEARCH_MULTI_DOG_TRACKING.md` | 多犬追踪调研（Phase 3） |
| `research/RESEARCH_3D_POSE_RECONSTRUCTION.md` | 3D 姿态重建调研（Phase 3） |
| `research/RESEARCH_FCI_IGP_STANDARD.md` | FCI-IGP 标准调研（Phase 3） |
| `research/RESEARCH_JETSON_DEPLOYMENT.md` | Jetson 边缘部署调研（Phase 3） |
| `research/RESEARCH_PUPPY_SELECTION_BEHAVIOR.md` | 幼犬选育行为调研 |
| `research/RESEARCH_PUBLIC_WORKING_DOG_VIDEOS.md` | 公开工作犬视频调研 |
| `research/RESEARCH_SCORING_RULE_ENGINE.md` | 评分规则引擎调研 |
| `research/RESEARCH_1.2F_SUPPLEMENT_DATASETS.md` | 1.2f 补强数据集调研 |
| `research/PHASE2_DATA_STRATEGY.md` | Phase 2 数据策略 |
| `research/PROJECT_HEALTH_CHECK_2026-07-26.md` | 2026-07-26 项目体检报告 |
| `research/BCST-GCN_ST-GCN_深度分析报告.md` | BCST-GCN 复现分析 |
| `research/工作犬标准自动化深度调研报告.md` | GA/T 标准详分 |
| `research/project-brief-source.md` | 原 PROJECT_PRD.md v2.0（brief 源） |

## 文档约定

- **truth 单一性**：每个概念只有一个 owner 文档，禁止重复 truth
- **中英文**：truth 文档使用中文，代码注释遵循用户最新消息语言
- **未验证标记**：无法确认的内容标记 `未验证`，不假装已知
- **决策可追溯**：重要决策记录到 `decisions/`，含依据与候选方案

## 阶段映射

本项目遵循 sliver-vibe-coding Main Sequence，当前处于 Phase 3 实施中（2026-08-02 sliver-vibe-coding 接管审计后）：
- ✅ 项目空间/Git/dev-docs 已建立
- ✅ Project brief / Function list / Technical selection / Architecture 已创建
- ✅ Agent constitution 已创建（`AGENTS.md` v1.20）
- ✅ ADR 0001-0010 已建立（0004 占位未触发）
- ✅ Runtime baseline 已建立（Phase 0 验收 + v1.4 新鲜验证 538 测试，见 `runtime.md`）
- ✅ Frontend/Database/Backend skeleton 已建立（Phase 0 验收）
- ✅ Phase 0 基础设施完成（2026-07-26 验收）
- ✅ Phase 1 MVP 完成（2026-07-28 验收，带条件）
- ✅ Phase 2 核心完成（2026-07-30 验收，v2.7 + ADR 0010）
- 🔄 Phase 3 专业实施中（v2.8：3.1b/c/d/e ST-GCN+BC + 3.2b/c 多犬追踪+ReID + 3.3b/c/d 3D 姿态 + 3.4b/c FCI-IGP + 3.5c 抽帧部分 + 3.6a/3.6b RBAC 基础部分完成，Git 检查点 `2e6aef3`）
- ⏳ Phase 4 前沿按需
