# dev-docs/ — 内部开发 truth root

本目录是工作犬训练系统的内部开发 truth root，存放 AI 与项目团队的规则、决策、架构、阶段计划等 truth 文档。

**公开文档与内部 truth 分离**：本目录内容不应直接推送到公开仓库（已在 `.gitignore` 中排除 `.trae/`，但 `dev-docs/` 默认入库；如需私有化，参见 `project-flow.md` 的 dual-repo 管理策略）。

## 文档索引

### 项目基础 truth
| 文档 | 用途 | 状态 |
|------|------|------|
| `project-brief.md` | 项目立项 brief（用户、场景、MVP、非目标） | ✅ |
| `function-list.md` | 功能清单与复杂功能文档索引 | ✅ |
| `technical-selection.md` | 技术选型 truth（栈+框架+架构组合） | ✅ |
| `architecture.md` | 项目架构（拓扑、owner map、模块边界） | ✅ |
| `runtime.md` | 运行时基线（包管理器、版本、启动命令） | ⏳ Phase 0 |

### 阶段 truth
| 文档 | 用途 | 状态 |
|------|------|------|
| `stages/` | 每个大阶段一个文件（含 stage 控制契约） | ⏳ |
| `stage-plan.md` | 大阶段计划（Phase 0-4） | ⏳ |

### 设计 truth
| 文档 | 用途 | 状态 |
|------|------|------|
| `design/` | UI 原型生命周期 truth | ⏳ |

### 基础设施 truth
| 文档 | 用途 | 状态 |
|------|------|------|
| `database-design.md` | 数据库 schema 设计 | ⏳ Phase 0 |
| `backend-boundary.md` | 后端职责与 API 边界 | ⏳ Phase 0 |
| `backend-architecture.md` | 后端架构 truth | ⏳ Phase 0 |
| `frontend-architecture.md` | 前端架构 truth | ⏳ Phase 0 |
| `security-boundary.md` | 安全边界 | ⏳ Phase 0 |

### 质量与验收
| 文档 | 用途 | 状态 |
|------|------|------|
| `quality/` | 阶段/功能质量验证 | ⏳ |
| `acceptance/` | 用户验收记录 | ⏳ |

### 决策记录
| 文档 | 用途 | 状态 |
|------|------|------|
| `decisions/` | 自研/升级/重要技术决策记录 | ✅ |
| `decisions/0001-chinese-path-migration-plan.md` | 中文路径迁移计划 | ✅ |

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
| `research/BCST-GCN_ST-GCN_深度分析报告.md` | BCST-GCN 复现分析 |
| `research/工作犬标准自动化深度调研报告.md` | GA/T 标准详分 |
| `research/project-brief-source.md` | 原 PROJECT_PRD.md v2.0（brief 源） |

## 文档约定

- **truth 单一性**：每个概念只有一个 owner 文档，禁止重复 truth
- **中英文**：truth 文档使用中文，代码注释遵循用户最新消息语言
- **未验证标记**：无法确认的内容标记 `未验证`，不假装已知
- **决策可追溯**：重要决策记录到 `decisions/`，含依据与候选方案

## 阶段映射

本项目遵循 sliver-vibe-coding Main Sequence，当前处于：
- ✅ 项目空间/Git/dev-docs 已建立
- ✅ Project brief 已创建
- ✅ Function list 已创建
- ✅ Technical selection 已创建
- ✅ Architecture 已创建
- ⏳ Agent constitution 待创建
- ⏳ Runtime baseline 待建立（Phase 0）
- ⏳ Frontend/Database/Backend skeleton 待建立（Phase 0）
