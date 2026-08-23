# dev-docs/ — 内部开发 truth root

本目录是工作犬训练系统的内部开发 truth root，存放 AI 与项目团队的规则、决策、架构、阶段计划等 truth 文档。

**公开文档与内部 truth 分离**：本目录内容不应直接推送到公开仓库（已在 `.gitignore` 中排除 `.trae/`，但 `dev-docs/` 默认入库；如需私有化，参见 `project-flow.md` 的 dual-repo 管理策略）。

> **修订说明（2026-08-20）**：当前主线为 P0 数据解阻；Phase 4 保留为并行实施工作。Phase 0-3 工程阶段已关闭，但真实数据、Jetson 与多犬 GT 验证边界单独记录。历史方案允许保留，但必须标注其当前状态，避免把 YouTube 专用流程当作现行路径。

## 文档索引

### 项目基础 truth
| 文档 | 用途 | 状态 |
|------|------|------|
| `project-brief.md` | 项目立项 brief（用户、场景、MVP、非目标） | ✅ v2.1（2026-08-19 漂移修正） |
| `function-list.md` | 功能清单与复杂功能文档索引 | ✅ |
| `technical-selection.md` | 技术选型 truth（栈+框架+架构组合） | ✅ |
| `architecture.md` | 项目架构（拓扑、owner map、模块边界） | ✅ v1.1（2026-08-01 同步 Phase 2/3 实际结构） |
| `runtime.md` | 运行时基线（包管理器、版本、启动命令） | ✅ Phase 0 已建立（v1.6，631 单元测试新鲜验证 + onnxruntime 修复 + FastAPI 204 兼容性修复 + 3.7 训练历史对比端到端 41/41 通过） |
| `stage-plan.md` | 大阶段计划（Phase 0-4 + P0） | ✅（当前 Truth） |

### 阶段 truth
| 文档 | 用途 | 状态 |
|------|------|------|
| `stages/phase-0.md` | Phase 0 基础设施计划 | ✅ 完成（2026-07-26 验收） |
| `stages/phase-1.md` | Phase 1 MVP 计划 | ✅ 完成（2026-07-28 验收，带条件） |
| `stages/phase-2.md` | Phase 2 核心计划 | ✅ 完成（2026-07-30 验收，v2.7） |
| `stages/phase-3.md` | Phase 3 专业计划 | ✅ 已关闭（v3.3：2026-08-17 正式关闭；3.1b/c/d/e + 3.2b/c/d + 3.3b/c/d + 3.4b/c + 3.5c + 3.6a/b/c + 3.7a/b/c + 3.8a/b/c/d 完成） |
| `stages/phase-4.md` | Phase 4 前沿计划 | 🔄 实施中（v1.1：2026-08-19 LLM 双阶段 + Mamba 授权纳入） |
| `stages/phase-4-transformer-mamba.md` | Transformer-Mamba 基线 | ✅ 已授权纳入（2026-08-19 用户认可） |

### 设计/质量/验收 truth
> `design/` / `quality/` / `acceptance/` 目录**尚未建立**。原 README 列为占位条目但实际未创建——Phase 0-2 期间验收证据统一归档到 `reports/phase-{N}-validation.md`。如需建立独立 truth 目录，Phase 3+ 启动时按需创建。

### 决策记录
| 文档 | 用途 | 状态 |
|------|------|------|
| `decisions/0001-chinese-path-migration-plan.md` | 中文路径迁移计划 | ✅ 已完成 |
| `decisions/0002-runtime-stack-revision-blackwell.md` | 运行时栈修正（PyTorch 2.11+cu128） | ✅ 已确认 |
| `decisions/0003-phase-0-to-phase-1.md` | Phase 0 → Phase 1 升级 | ✅ 已确认 |
| `decisions/0005-phase-1-to-phase-2.md` | Phase 1 → Phase 2 升级 | ✅ 已确认 |
| `decisions/0006-dogmo-open-alternative.md` | DogMo 替代方案（InterPet4D，YouTube 路径已废弃 2026-08-19） | ✅ 历史决策 |
| `decisions/0007-phase-2-start.md` | Phase 2 启动 | ✅ 已确认 |
| `decisions/0008-1.2f-supplement-plan.md` | 1.2f 数据补强计划 | ⚠️ 历史决策（YouTube 路径已废弃，见 v1.7） |
| `decisions/0009-data-strategy-four-paths.md` | 数据策略四路径 | ⚠️ 历史决策（被 ADR 0012 替代） |
| `decisions/0010-phase-2-to-phase-3.md` | Phase 2 → Phase 3 升级 | ✅ 已确认 |
| `decisions/0011-phase-4-llm-explainer-adr.md` | Phase 4 LLM 行为解释器 | ✅ 已实施（v1.1：实验室 Agnes API + 实用期本地模型） |
| `decisions/0012-foundation-system-direction.md` | 基座系统方向决策 | ✅ 已确认（物理-语义解耦 + 小样本自监督数据飞轮） |

> **ADR 0004 说明**：编号 0004 预留给"PoseC3D 触发但失败时记录"（见 `phase-1.md` §6 + `0003-phase-0-to-phase-1.md` §55）。AGENTS.md §9 已确认 PoseC3D 跳过决策（1.2f 条件通过维持跳过），故 ADR 0004 **未触发创建**——这是设计预期，非缺失。

### 调研资料（只读参考）
| 文档 | 状态 | 说明 |
|------|------|------|
| `research/RESEARCH_FOUNDATION_SYSTEM_VISION.md` | ✅ 当前支撑 | 基座系统方向 + 7 方向 37 仓库调研 |
| `research/RESEARCH_DATA_BLOCKADE_SOLUTION.md` | ✅ 当前支撑 | 数据阻塞诊断 + 四路线组合方案 |
| `research/PAPER_POSITIONING.md` | ✅ 当前支撑 | 论文定位：Pattern Recognition / IJCV |
| `research/RESEARCH_TRANSFORMER_MAMBA.md` | ✅ Phase 4 参考 | Mamba/Transformer 基线实验 |
| `research/RESEARCH_LITERATURE.md` | ✅ 方法背景 | 学术论文综述（BCST-GCN 等 12+ 篇） |
| `research/archive/` | 📦 归档 | Phase 1-3 已完成阶段的调研产物，详见 [research/README.md](research/README.md) |

## 文档约定

- **truth 单一性**：每个概念只有一个 owner 文档，禁止重复 truth
- **中英文**：truth 文档使用中文，代码注释遵循用户最新消息语言
- **未验证标记**：无法确认的内容标记 `未验证`，不假装已知
- **决策可追溯**：重要决策记录到 `decisions/`，含依据与候选方案

## 阶段映射

本项目遵循 sliver-vibe-coding Main Sequence，当前主线为 P0 数据解阻，Phase 4 并行实施中：
- ✅ 项目空间/Git/dev-docs 已建立
- ✅ Project brief / Function list / Technical selection / Architecture 已创建
- ✅ Agent constitution 已创建（`AGENTS.md` v1.21）
- ✅ ADR 0001-0010 已建立（0004 占位未触发）
- ✅ Runtime baseline 已建立（Phase 0 验收 + v1.5 新鲜验证 566 测试，见 `runtime.md`）
- ✅ Frontend/Database/Backend skeleton 已建立（Phase 0 验收）
- ✅ Phase 0 基础设施完成（2026-07-26 验收）
- ✅ Phase 1 MVP 完成（2026-07-28 验收，带条件）
- ✅ Phase 2 核心完成（2026-07-30 验收，v2.7 + ADR 0010）
- ✅ Phase 3 专业完成（2026-08-17 正式关闭，v3.3：3.1b/c/d/e + 3.2b/c/d + 3.3b/c/d + 3.4b/c + 3.5c + 3.6a/b/c + 3.7a/b/c + 3.8a/b/c/d 完成，Git 检查点 `2e6aef3`）
- 🔄 Phase 4 并行实施中（LLM 行为解释器 + Transformer-Mamba；实验与产品化后置于 P0 数据解阻主线）
- 🔄 P0 数据解阻为当前主线（AimCLR + SMQ + 姚青 JIA 迁移 + TCL + 主动学习）
