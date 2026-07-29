# 1.2f 补强数据集调研报告

> 调研日期: 2026-07-28
> Owner: ML 开发（见 AGENTS.md §2.2）
> 依据: ADR 0006 v1.1 + ADR 0007 + reports/phase-2-prereq-1.2f-validation.md v1.1
> 调研流程: GitHub-First（AGENTS.md §1.3 零容忍硬规则）

## 1. 调研背景

### 1.1 问题

InterPet4D v1 无视频文件 + 无行为标签，1.2f 原设计（视频→YOLO26-pose→准确率）无法执行。需寻找含视频+标签的数据集补强 1.2f，或采用 YouTube 自标方案。

### 1.2 调研目标

1. 寻找含视频+行为标签的犬类（或动物）行为识别数据集
2. 评估 YouTube 自标方案（标注工具 + 工作流）
3. 为 Phase 2.0a（1.2f 补强数据集选择）提供决策依据

## 2. 调研结果

### 2.1 Animal Kingdom 数据集（CVPR 2022）⭐ 推荐

**来源**: [GitHub](https://github.com/sutdcv/Animal-Kingdom) + [官网](https://sutdcv.github.io/Animal-Kingdom/) + [arXiv 2204.08129](https://arxiv.org/abs/2204.08129)

**核心数据**:
- **30K 视频序列**（动作识别任务，含视频 + 行为标签）
- **140 行为描述**（细粒度多标签动作识别）
- **850+ 动物物种**，6 大类别（哺乳动物/两栖动物/爬行动物/鸟类/鱼类）— **犬类属于哺乳动物**
- **33K 帧**（姿态估计任务）
- **50 小时标注视频**（视频接地任务）

**适用性评估**:
| 维度 | 评估 | 说明 |
|------|------|------|
| 视频可用 | ✅ | 30K 视频序列，有 RGB 视频 |
| 行为标签 | ✅ | 140 行为描述，细粒度多标签 |
| 犬类覆盖 | ✅ | 850 物种含犬类（哺乳动物类别） |
| 关键点 | ⚠️ | 33K 帧姿态估计，但非 Dog-Pose 24 关键点格式 |
| 许可证 | ⚠️ | 学术研究用，需 Google Form 申请 |
| 活跃度 | ✅ | GitHub 2024-12-08 最新更新，104 commits |

**申请流程**:
- Google Form: https://forms.gle/NipvmReDKaD5zUEw6
- 申请周期: 待确认（通常 1-2 周）

**风险**:
- 行为标签与本项目 P0 8 类（sit/down/stand/come/stay/...）映射需调研
- 哺乳动物类别包含多种物种，犬类占比待确认
- 关键点格式与 Dog-Pose 24 关键点不一致，需适配或仅用视频+行为标签

### 2.2 YouTube 自标方案

#### 2.2a CVAT（专业视频标注）⭐ 推荐

**来源**: [GitHub cvat-ai/cvat](https://github.com/cvat-ai/cvat)

**核心特性**:
- 2026-07-27 最新更新，6233 commits，非常活跃
- 专业视频标注工具，支持:
  - 关键点标注（适合 Dog-Pose 24 关键点）
  - 边界框标注（适合物体检测）
  - 时间区间标注（适合行为识别）
  - 多人协作
- Docker 部署，支持本地化（符合项目硬约束）

**适用性**: YouTube 玩球视频标注 + 1.2f 真实视频标注

#### 2.2b Label Studio（ML 集成标注）⭐ 推荐

**来源**: [GitHub heartexlabs/label-studio](https://github.com/heartexlabs/label-studio)

**核心特性**:
- 支持视频分类 + 多数据类型
- **ML 集成**（关键优势）:
  - Pre-labeling: YOLO26-pose 预标注 → 人工修正
  - Autolabeling: 自动标注
  - Active Learning: 主动学习（选择最复杂样本标注）
  - Online Learning: 在线学习（标注同时训练）
- Docker 部署，支持本地化

**适用性**: YouTube 自标 + 数据飞轮标注工具（Phase 2.1）

### 2.3 其他数据集（待评估）

按 GitHub-First 流程，以下数据集需进一步验证:

| 数据集 | 来源 | 状态 |
|--------|------|------|
| DogMo | arxiv 2510.24117 | ❌ 付费，已弃（ADR 0006） |
| InterPet4D | HuggingFace ohicarip/interpet4d | ✅ 已下载，但无视频/标签（ADR 0006 v1.1） |
| Animal Kingdom | CVPR 2022 | ⭐ 待申请 |
| YouTube 自标 | - | ⭐ 待执行 |

## 3. 推荐方案

### 3.1 主方案: Animal Kingdom 申请 + YouTube 自标并行

**路径 A（Animal Kingdom）**:
1. 立即提交 Google Form 申请
2. 申请通过后下载，筛选犬类视频
3. 行为标签映射到本项目 P0 8 类
4. 执行 1.2f 真实视频准确率验证

**路径 B（YouTube 自标）**:
1. 下载 3-5 段 YouTube 玩球视频（30-60 秒/段）
2. 用 Label Studio + YOLO26-pose 预标注
3. 人工修正关键点 + 行为标签
4. 执行 1.2f 真实视频准确率验证

### 3.2 标注工具选择

| 工具 | 优势 | 劣势 | 推荐场景 |
|------|------|------|---------|
| **CVAT** | 专业视频标注，关键点/边界框/时间区间 | 无 ML 集成 | 纯人工标注 |
| **Label Studio** | ML 集成（pre-labeling + active learning） | 配置复杂 | ML 辅助标注（推荐） |

**推荐**: Label Studio（ML 集成减少人工成本，适配 Phase 2.1 数据飞轮）

### 3.3 Phase 2.0a 执行计划

1. **Week 1**: 提交 Animal Kingdom 申请 + 下载 YouTube 视频 + Label Studio 部署
2. **Week 2**: YouTube 自标（YOLO26-pose 预标注 + 人工修正）+ Animal Kingdom 申请跟进
3. **Week 3**: 1.2f 真实视频准确率验证（YouTube 自标数据）
4. **Week 4**: Animal Kingdom 数据到达后补充验证（如申请通过）

## 4. 风险与缓解

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| Animal Kingdom 申请被拒 | 20% | 中 | YouTube 自标方案独立推进 |
| Animal Kingdom 犬类占比低 | 40% | 中 | 筛选犬类视频 + YouTube 自标补充 |
| YouTube 自标成本高 | 60% | 中 | Label Studio ML 辅助 + 主动学习 |
| 行为标签映射困难 | 30% | 中 | 仅映射覆盖的 4-5 类，其余延后 |

## 5. 决策依据

### 5.1 AGENTS.md 原则对齐

- ✅ §1.1.1 **广泛调研优先**: 多方案调研（Animal Kingdom + CVAT + Label Studio）
- ✅ §1.3 **GitHub-First 零容忍硬规则**: 全程使用 GitHub-First 流程，未使用 WebSearch
- ✅ §1.1.4 **深度优先**: 选择有视频+标签的数据集，确保验证深度

### 5.2 ADR 0006 v1.1 对齐

- InterPet4D v1 无视频/标签 → 需替代方案
- Animal Kingdom 作为 InterPet4D 的补强（有视频+标签）
- YouTube 自标作为快速验证路径

## 6. 引用

- **Animal Kingdom 论文**: Ng et al., "Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding", CVPR 2022
- **Animal Kingdom 数据**: https://sutdcv.github.io/Animal-Kingdom/
- **Animal Kingdom GitHub**: https://github.com/sutdcv/Animal-Kingdom
- **Animal Kingdom 申请**: https://forms.gle/NipvmReDKaD5zUEw6
- **CVAT**: https://github.com/cvat-ai/cvat
- **Label Studio**: https://github.com/heartexlabs/label-studio

## 7. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-28 | 初始版本，调研 Animal Kingdom + CVAT + Label Studio |
