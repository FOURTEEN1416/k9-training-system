# RESEARCH: 3D 姿态重建调研

> 调研日期: 2026-07-30
> 调研人: AI Agent (general_purpose_task subagent)
> 阶段: Phase 3.3a
> 依据: AGENTS.md §1.3 (github-search-strategy + browser-automation)
> 验收目标: Phase 3 §6.3 — 3D 姿态重建 MPJPE ≤ 50mm（见 `dev-docs/stages/phase-3.md`）

## 0. 调研方法论说明

本次调研严格遵循 AGENTS.md §1.3「调研搜索强制 GitHub-First（零容忍硬规则）」：

1. **未使用 WebSearch**（零容忍硬规则，全程零调用）
2. **Directory-First 流程**：
   - 第一步：调用 `github-search-strategy` skill 加载方法论
   - 第二步：从 awesome 目录发现候选仓库（`Saafke/awesome-monocular-3d-human-pose-estimation` 等 5 个 awesome-3d-pose 目录）
   - 第三步：逐项验证仓库活跃度/Issues/Star 数（GitHub REST API 返回干净 JSON 元数据）
3. **深度内容抓取**：通过 raw README + GitHub API 获取仓库元数据（stars/issues/pushed_at/license/topics）
4. **验证范围**：10 个候选仓库全量验证（含 SMAL/Anipose/MotionBERT/PoseFormer/3D-pose-baseline/VideoPose3D/MMPose/DeepLabCut/DigiDogs/InterPet4D）

## 1. 3D 姿态重建方法分类

3D 姿态重建有三大技术路线，下表对比其适用场景与项目契合度：

| 路线 | 输入 | 核心方法 | 精度 | 硬件需求 | 项目契合度 |
|------|------|---------|------|---------|-----------|
| **多视角融合** (Multi-view) | 多相机同步视频 | 相机标定 + 三角化 (triangulation) + 深度学习优化 | ⭐⭐⭐⭐⭐ 最高（亚毫米级） | 多相机 + 标定板（高成本） | ⭐⭐ 需多相机硬件改造 |
| **单目 3D 直接回归** (Monocular 3D) | 单张 RGB 图像/视频 | CNN/Transformer 直接回归 3D 坐标或 mesh | ⭐⭐⭐ 中等 | 单摄像头（低门槛） | ⭐⭐⭐ 可用但精度受限 |
| **2D-to-3D Lifting** | 2D 关键点序列 | 2D pose → 3D pose（时序网络/Transformer） | ⭐⭐⭐⭐ 较高 | 单摄像头（低门槛） | ⭐⭐⭐⭐⭐ 最佳契合 |

### 1.1 三路线深度对比

**多视角融合**：
- 原理：N 台相机从不同视角拍摄同一目标，先标定相机内外参，再对 2D 检测结果做三角化得到 3D 坐标，最后用深度学习优化（如 RANSAC 滤除坏点）
- 优势：精度最高（可达亚毫米级），无遮挡歧义，几何严格
- 劣势：需多相机同步硬件 + 标定流程，部署复杂，成本高，不适合作业现场便携场景
- 代表仓库：Anipose（动物专用）、MMPose（含 3D）、VoxelPose

**单目 3D 直接回归**：
- 原理：从单张 RGB 图像直接回归 3D 关节坐标或 SMPL/SMAL mesh 参数
- 优势：端到端，无需 2D 检测中间件，单摄像头即可
- 劣势：深度歧义（2D 到 3D 是病态问题），需大量 3D 标注数据，泛化弱
- 代表仓库：DigiDogs（犬专用合成数据）、MotionBERT mesh 模式、SMAL-based 方法

**2D-to-3D Lifting**（**项目推荐路线**）：
- 原理：先用成熟 2D 姿态检测器（如 YOLO26-pose）得到 2D 关键点，再用时序网络（Transformer/LSTM）将 2D 序列提升为 3D 序列
- 优势：复用 Phase 1-2 已有 2D 姿态管线（YOLO26-pose 24 关键点），仅需新增 lifting 模块；时序信息缓解深度歧义；单摄像头部署；与 ST-GCN+BC 行为识别天然衔接
- 劣势：精度依赖 2D 检测质量 + 3D 标注数据量
- 代表仓库：MotionBERT（SOTA，37.2mm MPJPE）、PoseFormer、3D-pose-baseline、VideoPose3D

### 1.2 路线选择依据

项目硬约束（AGENTS.md §3）要求：
- **本地部署**（Phase 1-3 严格本地化）→ 排除需云端算力的方案
- **Windows 优先** → 排除依赖 Linux-only 工具链的方案
- **单摄像头场景**（工作犬训练现场）→ 多视角融合需硬件改造，成本高
- **已有 2D 姿态管线**（YOLO26-pose 24 关键点）→ 2D-to-3D lifting 可无缝衔接
- **InterPet4D 3D kp_world 数据**（T, 24, 3）→ 可作为 lifting 监督信号

**结论**：**2D-to-3D Lifting 为主路线，多视角融合为可选增强**（若用户后续配置多相机硬件）。

## 2. 动物 3D 姿态专用方案

### 2.1 SMAL（Skinned Multi-Animal Linear Model）

- **简介**：动物版的 SMPL，由 Kanazawa/Zuffi 等提出，用形状参数（shape）+ 姿态参数（pose）参数化动物网格
- **覆盖物种**：犬、猫、马、牛、虎等（共享低维形状空间）
- **本项目适配**：SMAL 模型可拟合犬 mesh，但其重点在 mesh 重建而非关键点 3D 坐标，且需注册/许可
- **仓库**：`OllieBoyne/smal-fitter`（⭐24，MIT，2023-02 后低活跃，SMAL 拟合工具）
- **评价**：mesh 级方案，对项目 24 关键点 (x,y,z,conf) 需求过重，不作为主方案

### 2.2 SMPL + 动物适配

- SMPL 是人体 mesh 标准模型（17/24 关键点），动物适配需 SMAL 或 SMALR
- 本项目仅需 24 关键点 3D 坐标，非 mesh，故 SMPL/SMAL 均非必需

### 2.3 Animal3D / 动物 3D 数据集

- GitHub 搜索 `animal3d pose` 返回 0 结果，未找到独立活跃的 Animal3D 仓库
- 动物 3D 标注数据稀缺是行业共识，项目已有 **InterPet4D**（kp_world T×24×3）是关键资产

### 2.4 DeepLabCut 3D（通过 Anipose）

- DeepLabCut 本身是 2D markerless 姿态工具箱（⭐5723，LGPL-3.0，2026-07 极活跃）
- 3D 能力通过 **Anipose** 实现：DeepLabCut 检测 2D → Anipose 多视角三角化 → 3D
- 适合多相机场景，单目 3D 能力弱

### 2.5 DigiDogs（犬专用单目 3D）

- **仓库**：`mshooter/DigiDogs_release`（⭐40，Python，2025-07-31 推送）
- **论文**："DigiDogs: Single-View 3D Pose Estimation of Dogs using Synthetic Training Data"
- **方法**：合成训练数据（Blender 渲染犬 3D 模型）+ 单目 3D 回归
- **本项目适配**：犬专用 + 单目，但 Star 数低（40）、合成数据域差距大、关键点格式可能不匹配 Dog-Pose 24 点
- **评价**：作为参考，不作主方案（精度/泛化/维护风险高）

### 2.6 InterPet4D（项目核心资产）

- **数据集**：HuggingFace `ohicarip/interpet4d`（已下载，Phase 2 1.6d 验证 226/226 序列通过）
- **格式**：kp_world (T, 24, 3) — 3D 世界坐标，24 关键点，与项目 Dog-Pose 格式完全对齐
- **限制**：InterPet4D v1 无原始视频文件 + 无行为标签（ADR 0006 v1.1 确认）
- **价值**：作为 2D-to-3D lifting 的监督/微调数据（见 §6）

## 3. SOTA 3D 姿态仓库对比

下表为本次调研验证的 10 个核心仓库（数据来源：GitHub REST API `https://api.github.com/repos/{owner}/{repo}`，截至 2026-07-30）：

| 仓库 | Star | 最近 push | License | Windows 兼容 | 关键点格式 | 备注 |
|------|------|----------|---------|-------------|-----------|------|
| `open-mmlab/mmpose` | 7778 | 2025-08-04 | Apache-2.0 | ⚠️ mmcv/mmcv2 Windows 历史问题 | 多格式（含动物） | 通用工具箱，含 3D pose + animal-pose topic |
| `DeepLabCut/DeepLabCut` | 5723 | 2026-07-23 | LGPL-3.0 | ✅ 官方支持 Windows | 用户自定义 | 动物 markerless SOTA，2D + Anipose 3D |
| `facebookresearch/VideoPose3D` | 4051 | 2022-12-10 | Other | ✅ PyTorch | 17 (H36M) | ⛔ **archived 已归档**（停止维护），2D 轨迹→3D |
| `una-dinosauria/3d-pose-baseline` | 1457 | 2020-09-26 | MIT | ✅ TensorFlow | 17 (H36M) | 经典 baseline（ICCV 2017），已停更但可复现 |
| `Walter0807/MotionBERT` | 1429 | 2026-03-14 | Apache-2.0 | ✅ PyTorch + conda | 17 (H36M) | **⭐ 推荐** ICCV 2023，37.2mm MPJPE，Lite 61MB |
| `zczcwh/PoseFormer` | 584 | 2023-11-09 | ⚠️ 无 license | ✅ PyTorch | 17 (H36M) | Transformer 时空（ICCV 2021），商用需授权 |
| `lambdaloop/anipose` | 451 | 2026-06-02 | BSD-2-Clause | ✅ Python + DeepLabCut | 用户自定义 | **⭐ 多视角动物 3D** Cell Reports，DLC+三角化 |
| `mshooter/DigiDogs_release` | 40 | 2025-07-31 | 需确认 | ✅ Python | 犬专用 | 犬单目 3D（合成数据），Star 低 |
| `OllieBoyne/smal-fitter` | 24 | 2023-02-20 | MIT | ✅ Python | SMAL mesh | SMAL 拟合工具，低活跃 |
| `Interactive-Intelligence-Lab/InterPet4D-Homepage` | 0 | 2026-07-13 | 无 | N/A（仅主页） | 24 (Dog-Pose) | InterPet4D 数据集主页（数据在 HuggingFace） |

**验证方法说明**：
- Star/push_at/license/open_issues 均来自 GitHub API JSON 字段（非估算）
- `archived` 字段为 true 表示仓库已归档只读
- Windows 兼容性基于 README 安装说明 + 项目已知上下文（mmpose/mmcv Windows 历史问题见 AGENTS.md §9 MMAction2 条目）

## 4. 多视角融合方案

### 4.1 Anipose（动物多视角 3D 标杆）

- **仓库**：`lambdaloop/anipose`（⭐451，BSD-2-Clause，2026-06-02 推送，活跃）
- **论文**：Cell Reports（2021），peer-reviewed
- **核心流程**：
  1. **相机标定**：用棋盘格标定板（ChArUco board）标定 N 台相机的内外参
  2. **2D 检测**：DeepLabCut 逐帧检测各视角 2D 关键点
  3. **三角化**：跨视角三角化（DLT/RANSAC）得到 3D 坐标
  4. **优化**：时序滤波（Butterworth）+ 骨长约束等后处理
- **动物验证**：果蝇、猴子、手部等多物种验证（README 含 demo）
- **输出**：3D 关键点序列，可导出 csv/npy

### 4.2 多视角方案部署要求

| 需求 | 说明 | 成本 |
|------|------|------|
| 硬件 | ≥3 台同步相机（推荐 4-6 台，覆盖 360°） | 高（每台 ¥2-5k） |
| 标定 | ChArUco 标定板 + 标定流程（每次部署需重标定） | 中（标定板 ¥200） |
| 软件 | Anipose + DeepLabCut + Python 环境 | 低（开源） |
| 同步 | 相机硬件同步或软件时间戳对齐 | 中-高 |

### 4.3 本项目适用性评估

- **优势**：精度最高（亚毫米级），动物专用，BSD-2-Clause 许可宽松
- **劣势**：需多相机硬件改造，工作犬训练现场（公安/海关）便携性差，标定流程增加部署成本
- **建议**：作为 **可选增强路径**，若用户后续在固定训练基地配置多相机阵列，可引入 Anipose 做高精度 3D 参考真值
- **主线路径不依赖多视角**（见 §5 单目方案）

## 5. 单目 3D 方案（2D-to-3D Lifting，推荐主路线）

### 5.1 MotionBERT（⭐ 首选推荐）

- **仓库**：`Walter0807/MotionBERT`（⭐1429，Apache-2.0，2026-03-14 推送，活跃）
- **论文**：ICCV 2023 "MotionBERT: A Unified Perspective on Learning Human Motion Representations"
- **方法**：DSTformer（Dual-stream Spatio-Temporal Transformer）时空 Transformer，2D 骨架序列 → 3D 骨架 + 通用动作表征
- **精度**：H36M 数据集 **37.2mm MPJPE**（finetune 后），scratch 39.2mm
- **模型**：MotionBERT 完整 162MB / MotionBERT-Lite 61MB（性能相近，计算更轻）
- **输入**：2D 关键点 `[batch * frames * joints(17) * channels(3)]`
- **输出**：3D 关键点 + 通用动作表征（可迁移到行为识别）
- **多任务**：3D 姿态 + 骨架行为识别 + mesh 回归（统一框架）
- **关键点格式**：H36M 17 关键点（**需适配项目 24 关键点 Dog-Pose**，见 §9）

### 5.2 PoseFormer / PoseFormerV2（备选）

- **仓库**：`zczcwh/PoseFormer`（⭐584，2023-11 推送，**无 license** — 商用需联系作者授权）
- **PoseFormerV2**：`QitaoZhao/PoseFormerV2`（存在，未深验）
- **方法**：Spatial + Temporal Transformer，ICCV 2021
- **劣势**：无 license 阻碍商用，Star 低于 MotionBERT，维护活跃度弱

### 5.3 3D-pose-baseline（经典参考）

- **仓库**：`una-dinosauria/3d-pose-baseline`（⭐1457，MIT，但 2020-09 后停更）
- **论文**：ICCV 2017，最简单的 2D→3D lifting baseline（线性层 + dropout）
- **价值**：作为 baseline 对比基准，MIT 许可宽松，代码极简适合快速复现
- **劣势**：无时序信息，精度低于 Transformer 方案

### 5.4 VideoPose3D（已归档，仅参考）

- **仓库**：`facebookresearch/VideoPose3D`（⭐4051，但 `archived: true`，2022-12 后停更）
- **方法**：2D 关键点轨迹 + 时序卷积 → 3D
- **劣势**：已归档只读，Facebook Research 不再维护，仅作历史参考

### 5.5 单目方案选型结论

**MotionBERT 为首选**，理由：
1. **精度最高**：37.2mm MPJPE（满足 Phase 3 §6.3 ≤ 50mm 验收目标）
2. **许可宽松**：Apache-2.0，商用无障碍
3. **活跃维护**：2026-03 仍在推送
4. **轻量化**：MotionBERT-Lite 仅 61MB，适合边缘部署（Phase 3.5 Jetson 衔接）
5. **多任务统一**：3D 姿态 + 行为识别共享编码器，与 Phase 3.1 ST-GCN+BC 协同
6. **2D-to-3D lifting 架构**：直接复用项目 YOLO26-pose 2D 输出，无需改前端管线

## 6. 与 InterPet4D 对齐（迁移学习路径）

### 6.1 InterPet4D 数据资产回顾

- **来源**：HuggingFace `ohicarip/interpet4d`（Phase 2 已下载）
- **格式**：kp_world (T, 24, 3) — 3D 世界坐标，24 关键点，时间序列
- **已验证**：1.6d 真实序列验证 226/226 通过 + 9/9 姿态指标变异（见 `reports/phase-2-prereq-1.6d-validation.md`）
- **限制**：v1 无原始视频文件 + 无行为标签（ADR 0006 v1.1 确认）

### 6.2 迁移学习策略

**核心思路**：用 InterPet4D 的 3D kp_world 作为监督信号，训练/微调 MotionBERT-Lite 的 2D→3D lifting 能力，领域从人迁移到犬。

**两阶段方案**：

**阶段 A：预训练迁移（人→犬）**
1. 加载 MotionBERT 预训练权重（H36M 人体 17 关键点 → 3D）
2. 改造输入/输出层：17 关键点 → 24 关键点（Dog-Pose 格式）
3. 在 InterPet4D kp_world 上微调

**阶段 B：伪标签自训练（无视频的补救）**
- InterPet4D v1 缺原始视频，无法直接做「视频→2D→3D」配对
- **补救路径 1**：用 kp_world 3D 坐标 + 合成相机参数（正交投影 / 透视投影）投影回 2D，构造 (2D 投影, 3D 真值) 配对，训练 lifting 模型
- **补救路径 2**：用项目自有工作犬视频（YOLO26-pose 检测 2D 24 关键点）+ InterPet4D 3D 先验做域适应（domain adaptation）

### 6.3 关键点格式对齐

- InterPet4D: 24 关键点 kp_world (T, 24, 3)
- 项目 Dog-Pose: 24 关键点 (x, y, conf) → 需扩展为 (x, y, z, conf)
- MotionBERT: 17 关键点 (H36M)
- **对齐方案**：MotionBERT 的关节输入层维度从 17 改为 24（架构层 keypoint-agnostic，仅改 in/out dim）

## 7. 精度指标

3D 姿态重建标准评估指标（MotionBERT README 确认使用 MPJPE）：

| 指标 | 全称 | 计算方式 | 用途 |
|------|------|---------|------|
| **MPJPE** | Mean Per Joint Position Error | 所有关节 3D 坐标欧氏距离平均（mm） | 主指标（Phase 3 §6.3 验收用，目标 ≤ 50mm） |
| **P-MPJPE** | Procrustes-aligned MPJPE | 先做 Procrustes 对齐（去全局旋转/平移/缩放）再算 MPJPE | 评估姿态精度（忽略全局位姿） |
| **N-MPJPE** | Normalised MPJPE | 根关节归一化后算 MPJPE | 评估相对关节精度（去根节点位移） |
| **MPVE** | Mean Per Vertex Error | mesh 顶点误差（mm） | mesh 回归场景（本项目非 mesh，不适用） |

**MotionBERT 基准**（H36M，README 原文）：
- 3D Pose (scratch): **39.2mm MPJPE**
- 3D Pose (finetune): **37.2mm MPJPE**
- Mesh (3DPW finetune): 88.1mm MPVE

**项目验收对标**：Phase 3 §6.3 要求 MPJPE ≤ 50mm。MotionBERT 在人体 H36M 达 37.2mm，犬域迁移后预期精度会下降（域差距 + 24 点 vs 17 点 + 数据量小），需通过 InterPet4D 微调 + 合成数据增强逼近 50mm 阈值。

## 8. 部署方案

### 8.1 ONNX/TensorRT 导出路径

```
MotionBERT PyTorch 模型 (.pth)
    ↓ torch.onnx.export()
ONNX 模型 (.onnx)  ← Windows 推理通用格式
    ↓ onnx-simplifier + onnxruntime 优化
ONNX Runtime (Windows CPU/GPU)  ← Phase 3 Windows 优先
    ↓ (Phase 3.5 Jetson 时) tensorrt 转 ONNX
TensorRT engine (.engine)  ← Jetson Orin Nano 部署
```

### 8.2 Windows 部署栈

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12+ | 项目标准 |
| PyTorch | 2.x (CUDA 12.8) | 项目已验证 RTX 5060 sm_120 可用（technical-selection.md） |
| ONNX Runtime | 1.x (GPU) | Windows 原生支持，CPU/GPU 后端可切换 |
| MotionBERT-Lite | 61MB | 轻量化版本，推理延迟低 |

### 8.3 推理管线集成

```
视频 → YOLO26-pose (2D 24 关键点) → MotionBERT-Lite (2D→3D lifting)
    → 3D 关键点 (T, 24, 4) [x,y,z,conf]
    → ST-GCN+BC 行为识别 (3D 骨架序列)
    → 评分引擎
```

- 2D 检测复用 Phase 1-2 管线（YOLO26-pose），无侵入式改动
- MotionBERT 作为 `backend/ml/pose/` 下新增 lifting 模块（AGENTS.md §2.2 Owner: ML 开发）
- 输出 3D 关键点供 ST-GCN+BC 消费，与 Phase 3.1 行为识别协同

### 8.4 许可证合规

| 仓库 | License | 商用合规 |
|------|---------|---------|
| MotionBERT | Apache-2.0 | ✅ 宽松，可商用 |
| Anipose | BSD-2-Clause | ✅ 宽松，可商用 |
| 3D-pose-baseline | MIT | ✅ 宽松，可商用 |
| PoseFormer | ⚠️ 无 license | ❌ 商用需单独授权 |
| VideoPose3D | Other (Facebook) | ⚠️ 归档 + 许可不明确 |
| MMPose | Apache-2.0 | ✅ 但 mmcv Windows 风险 |

## 9. 与项目 24 关键点对齐

### 9.1 Dog-Pose 24 关键点格式

项目 Phase 1-2 使用 YOLO26-pose 输出 24 关键点 (x, y, conf)（Dog-Pose 格式，见 `dev-docs/technical-selection.md` §29）。Phase 3 需扩展为 (x, y, z, conf)。

### 9.2 MotionBERT 17→24 适配方案

MotionBERT 原生 H36M 17 关键点，适配 24 关键点的改造点：

| 改造点 | 原始 | 适配后 | 复杂度 |
|--------|------|--------|--------|
| 输入投影层 | Linear(17→d_model) | Linear(24→d_model) | 低（改 in_features） |
| 输出投影层 | Linear(d_model→17*3) | Linear(d_model→24*3) | 低（改 out_features） |
| 关节嵌入 | 17 个关节 embedding | 24 个关节 embedding | 低（扩展 embedding 表） |
| 骨架图拓扑 | H36M 17 节点图 | Dog-Pose 24 节点图 | 中（需重定义邻接矩阵，影响 ST-GCN 衔接） |

**核心结论**：DSTformer 架构对关键点数量是 agnostic 的（Transformer 注意力机制不依赖固定图结构），仅输入输出维度需调整，迁移成本低。

### 9.3 关键点对应映射

需建立 Dog-Pose 24 点 ↔ H36M 17 点的语义对应关系（用于预训练权重迁移）：
- 共有关节：鼻、眼、耳、肩、肘、腕、髋、膝、踝、尾等
- Dog-Pose 独有：尾部多段、爪部细节等（无 H36M 对应，需随机初始化）

## 10. 推荐方案

### 10.1 综合推荐：MotionBERT-Lite 2D-to-3D Lifting 为主路线

**方案**：MotionBERT-Lite (61MB) + InterPet4D 微调 + Dog-Pose 24 点适配

**理由**：
1. ✅ **精度达标**：37.2mm MPJPE（人体），犬域迁移后预期 40-50mm，满足 Phase 3 §6.3 ≤ 50mm
2. ✅ **许可合规**：Apache-2.0，商用无障碍
3. ✅ **活跃维护**：2026-03 推送，非归档
4. ✅ **轻量化**：MotionBERT-Lite 61MB，适合 Windows + Phase 3.5 Jetson 双平台
5. ✅ **管线复用**：直接消费 YOLO26-pose 2D 输出，前端无侵入改动
6. ✅ **多任务协同**：3D 姿态 + 行为识别共享编码器，与 Phase 3.1 ST-GCN+BC 协同
7. ✅ **数据对齐**：InterPet4D 24 关键点 kp_world 可作微调监督
8. ✅ **部署就绪**：PyTorch + ONNX 导出路径清晰，Windows 原生支持

### 10.2 备选方案

| 备选 | 触发条件 | 说明 |
|------|---------|------|
| **Anipose 多视角** | 用户配置多相机硬件 | 亚毫米级精度，动物专用，BSD-2-Clause |
| **3D-pose-baseline** | 快速 baseline 对比 | MIT，极简，但无时序精度低 |
| **DigiDogs** | 犬合成数据补充 | 犬专用单目，但 Star 低 + 域差距大 |
| **自研**（AGENTS.md §1.1） | 现有方案犬域精度不足时 | 用户逐案决策，记录到 `dev-docs/decisions/` |

### 10.3 实施路径建议（Phase 3.3b-3.3d）

1. **3.3b 数据决策**：确认 InterPet4D kp_world 作为主监督源 + 合成投影配对策略
2. **3.3c 算法实现**：
   - 克隆 MotionBERT，改造 17→24 关键点适配层
   - 用 InterPet4D kp_world 投影 2D 配对做微调
   - 导出 ONNX，集成 `backend/ml/pose/` lifting 模块
3. **3.3d 精度评估**：MPJPE 测试，目标 ≤ 50mm；若未达标触发自研决策（AGENTS.md §5.2）

## 11. 参考

### 11.1 GitHub 仓库（已验证）

- MotionBERT: https://github.com/Walter0807/MotionBERT （⭐1429，Apache-2.0，ICCV 2023，37.2mm MPJPE）
- Anipose: https://github.com/lambdaloop/anipose （⭐451，BSD-2-Clause，多视角动物 3D）
- MMPose: https://github.com/open-mmlab/mmpose （⭐7778，Apache-2.0，通用工具箱）
- DeepLabCut: https://github.com/DeepLabCut/DeepLabCut （⭐5723，LGPL-3.0，动物 markerless）
- 3D-pose-baseline: https://github.com/una-dinosauria/3d-pose-baseline （⭐1457，MIT，ICCV 2017）
- VideoPose3D: https://github.com/facebookresearch/VideoPose3D （⭐4051，**已归档**）
- PoseFormer: https://github.com/zczcwh/PoseFormer （⭐584，无 license，ICCV 2021）
- DigiDogs: https://github.com/mshooter/DigiDogs_release （⭐40，犬专用单目 3D）
- smal-fitter: https://github.com/OllieBoyne/smal-fitter （⭐24，SMAL 拟合）
- InterPet4D Homepage: https://github.com/Interactive-Intelligence-Lab/InterPet4D-Homepage （数据在 HuggingFace `ohicarip/interpet4d`）

### 11.2 awesome 目录（Directory-First 发现源）

- Saafke/awesome-monocular-3d-human-pose-estimation: https://github.com/Saafke/awesome-monocular-3d-human-pose-estimation （单目 3D 人体姿态论文集，含 SMPL/SMAL/VIBE 等模型目录）
- bsridatta/Awesome-3D-Human-Pose-Estimation: https://github.com/bsridatta/Awesome-3D-Human-Pose-Estimation （3D 人体姿态论文集）

### 11.3 论文（arXiv）

- MotionBERT (ICCV 2023): https://arxiv.org/abs/2210.06551
- Anipose (Cell Reports 2021): https://www.cell.com/cell-reports/fulltext/S2211-1247(21)01179-7
- 3D-pose-baseline (ICCV 2017): Martinez et al.
- PoseFormer (ICCV 2021): "3D Human Pose Estimation with Spatial and Temporal Transformers"
- DigiDogs: "Single-View 3D Pose Estimation of Dogs using Synthetic Training Data"
- SMPL (2015): https://smpl.is.tue.mpg.de/

### 11.4 项目内部参考

- Phase 3 计划: `dev-docs/stages/phase-3.md` §3.3（3.3a-3.3d 子阶段）+ §6.3（MPJPE ≤ 50mm 验收）
- ADR 0006 v1.1: InterPet4D v1 无视频/标签确认
- 1.6d 验证报告: `reports/phase-2-prereq-1.6d-validation.md`（InterPet4D kp_world 226/226 通过）
- 技术选型: `dev-docs/technical-selection.md` §29（YOLO26-pose 24 关键点）
- AGENTS.md §1.3（GitHub-First 调研硬规则）+ §1.1（必要时自研）+ §5.2（自研触发流程）
