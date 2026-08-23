# research/ — 研究资料索引

> 本目录存放支撑当前 P0 数据解阻主线与 Phase 4 并行工作的研究资料。
> **当前执行 Truth**：`../stage-plan.md` + `../stages/data-unblocking.md` + `../decisions/0012-foundation-system-direction.md`
> **最后清洗**：2026-08-20（全量文档治理，过时调研归档至 `archive/`）

---

## 状态规则

| 状态 | 含义 | 处置 |
|------|------|------|
| **当前支撑** | 仍被 P0/Phase 4 主线的 ADR 或阶段文档直接引用，结论作为路线依据 | 保留在根目录 |
| **历史参考** | Phase 1-3 已完成或降级，内容已融入 ADR/阶段文档，仅保留方法学参考价值 | 归档至 `archive/` |
| **已清除** | 临时产物或已被明确废弃的中间文件 | 已删除 |

---

## 当前支撑资料（根目录）

### P0 数据解阻核心

| 文档 | 用途 | 关联 Truth |
|------|------|-----------|
| [`RESEARCH_FOUNDATION_SYSTEM_VISION.md`](./RESEARCH_FOUNDATION_SYSTEM_VISION.md) | 基座系统方向：物理-语义解耦 + 小样本自监督数据飞轮，7 方向 37 仓库调研结论 | ADR 0012 + data-unblocking.md §2 |
| [`RESEARCH_DATA_BLOCKADE_SOLUTION.md`](./RESEARCH_DATA_BLOCKADE_SOLUTION.md) | 数据阻塞诊断 + 四路线组合方案（规则种子→主动学习→半监督→验证闭环） | ADR 0012 + data-unblocking.md §3 |
| [`PAPER_POSITIONING.md`](./PAPER_POSITIONING.md) | 论文定位：Pattern Recognition / IJCV，核心创新（姚青 JIA 迁移），方法管线 | project-brief.md §1 + stage-plan.md §2.7 |

### Phase 4 并行参考

| 文档 | 用途 | 关联 Truth |
|------|------|-----------|
| [`RESEARCH_TRANSFORMER_MAMBA.md`](./RESEARCH_TRANSFORMER_MAMBA.md) | Mamba/Transformer 基线实验结果 + 集成计划 | phase-4.md §4.2 + phase-4-transformer-mamba.md |

### 方法论背景

| 文档 | 用途 | 关联 Truth |
|------|------|-----------|
| [`RESEARCH_LITERATURE.md`](./RESEARCH_LITERATURE.md) | 学术论文综述（BCST-GCN/ASBAR/PoseC3D/SLEAP 等 12+ 篇），定位本工作与已有研究的差距 | technical-selection.md §2.2 + PAPER_POSITIONING.md §2 |

---

## 归档资料（`archive/`）

以下文档属于 Phase 1-3 完成阶段的调研产物，内容已融入对应的 ADR 或阶段文档。保留归档以防未来回溯，但**不作为当前执行入口**。

| 文档 | 归档原因 | 对应 Truth |
|------|---------|-----------|
| `RESEARCH_YOLO26_DEPLOY_DEEP.md` | YOLO 部署策略已固化到 runtime.md + technical-selection.md | runtime.md §3 |
| `RESEARCH_MULTI_DOG_TRACKING.md` | Phase 3.2 已实现，BoxMOT 选型决策见 phase-3.md §3.2 | phase-3.md §3.2 |
| `RESEARCH_3D_POSE_RECONSTRUCTION.md` | Phase 3.3 已实现，MotionBERT-Lite 选型见 phase-3.md §3.3 | phase-3.md §3.3 |
| `RESEARCH_FCI_IGP_STANDARD.md` | Phase 3.4 已实现，评分卡 YAML 见 `backend/ml/scoring/configs/fci_igp.yaml` | phase-3.md §3.4 |
| `RESEARCH_JETSON_DEPLOYMENT.md` | Phase 3.5 已放弃（RTX 5060 软件优化足够） | phase-3.md §3.5 |
| `RESEARCH_STANDARDS.md` | GA-T/USPCA/FCI-IGP 标准映射已固化为 YAML 评分卡 | backend/ml/scoring/configs/ |
| `RESEARCH_STGCN_BC.md` | Phase 3.1 已实现，BC 头自研决策见 phase-3.md §3.1 | phase-3.md §3.1 |
| `BCST-GCN_ST-GCN_深度分析报告.md` | BCST-GCN 方法参考已融入 RESEARCH_LITERATURE.md | RESEARCH_LITERATURE.md §1.1 |
| `工作犬标准自动化深度调研报告.md` | GA-T/USPCA/FCI-IGP 标准映射已固化为 YAML 评分卡 | backend/ml/scoring/configs/ |
| `RESEARCH_PUPPY_SELECTION_BEHAVIOR.md` | 选育信号提取器已在 Phase 1.6 实现 | phase-1.md §1.6 |
| `RESEARCH_SCORING_RULE_ENGINE.md` | YAML 评分引擎已完成，实现见 `backend/ml/scoring/` | backend/ml/scoring/ |
| `RESEARCH_TECH_COMPARISON.md` | 技术选型对比已固化为 technical-selection.md | technical-selection.md |
| `RESEARCH_OPENSOURCE.md` | 开源项目分析已融入各阶段调研 | 分散在各阶段文档 |
| `RESEARCH_COMMERCIAL.md` | 商业分析已非当前重点 | — |
| `RESEARCH_IMPLEMENTATION.md` | 实现细节已分散到各阶段文档 | 分散到 stages/ |
| `RESEARCH_PHASE1_STACK_DEEP.md` | Phase 1 调研，已被后续 ADR 覆盖 | ADR 0003-0008 |
| `RESEARCH_PUBLIC_WORKING_DOG_VIDEOS.md` | YouTube/公开视频路径已废弃，数据源见 data-unblocking.md §3 | data-unblocking.md §3 |
| `RESEARCH_1.2F_SUPPLEMENT_DATASETS.md` | 1.2f 数据补强方案已被 ADR 0012 替代 | ADR 0012 |
| `PHASE2_DATA_STRATEGY.md` | Phase 2 数据策略已被 ADR 0012 替代 | ADR 0012 |
| `MULTI_DOG_TRACKING_DATA.md` | 多犬数据方案已被 Phase 3.2 实现替代 | phase-3.md §3.2 |
| `PROJECT_HEALTH_CHECK_2026-07-26.md` | 一次性体检报告，结论已落实到后续 ADR | ADR 0003 + AGENTS.md |
| `PHASE4_EXECUTION_SUMMARY.md` | Phase 4 执行记录含废弃 YouTube 路径引用，需清理 | — |
| `initial-notes.txt` | 项目初始笔记，无执行价值 | — |

---

## 文档间交叉验证

| 方法组件 | 论文出处 | 本项目状态 | 验证依据 |
|---------|---------|-----------|---------|
| AimCLR | AAAI 2022, MIT 许可 | ⏳ P0.1 待实施 | [RESEARCH_FOUNDATION_SYSTEM_VISION.md](./RESEARCH_FOUNDATION_SYSTEM_VISION.md) §发现 2 |
| SMQ | ICCV 2025 | ⏳ P0.2 待实施 | [RESEARCH_FOUNDATION_SYSTEM_VISION.md](./RESEARCH_FOUNDATION_SYSTEM_VISION.md) §发现 1 |
| TCL | CVPR 2021 | ⏳ P0.4 待实施 | [RESEARCH_FOUNDATION_SYSTEM_VISION.md](./RESEARCH_FOUNDATION_SYSTEM_VISION.md) §发现 3 |
| 姚青 JIA 迁移 | 图像域（VFM+锚点+聚类+伪标签） | 🔥 P0.3 核心创新 | [RESEARCH_FOUNDATION_SYSTEM_VISION.md](./RESEARCH_FOUNDATION_SYSTEM_VISION.md) §发现 4 |
| ST-GCN+BC | BCST-GCN 2026 (Front. Vet. Sci.) | ✅ Phase 3 已实现 | [RESEARCH_LITERATURE.md](./RESEARCH_LITERATURE.md) §1.1 |
| Mamba 基线 | VideoMamba ECCV 2024 + MS-Temba CVPR 2026 | ✅ Phase 4 已授权 | [RESEARCH_TRANSFORMER_MAMBA.md](./RESEARCH_TRANSFORMER_MAMBA.md) |
| FCI-IGP 7 维 | FCI-IGP 2025 官方规则 | ✅ Phase 3.4 已实现 | `backend/ml/scoring/configs/fci_igp.yaml` |
| keypoint-MoSeq | Nature Methods 2024 | ⏸️ 存档废弃 | [RESEARCH_DATA_BLOCKADE_SOLUTION.md](./RESEARCH_DATA_BLOCKADE_SOLUTION.md) §1.1 |

---

## 已清除资料

- `project-brief-source.md` — 原始 PRD v2.0，已被 `project-brief.md` v2.3 取代
- `canine_action_temp.pdf` — 临时抓取文件（已删除）

> 历史资料中的"当前""推荐""待执行"均以文件当时日期为准；若与最新 Truth 冲突，以 ADR 0012 和 P0 阶段计划为准。
