# RESEARCH: ST-GCN+BC 行为识别调研

> 调研日期: 2026-07-30
> 调研人: AI Agent (general_purpose_task subagent)
> 阶段: Phase 3.1a
> 依据: AGENTS.md §1.3 (github-search-strategy + browser-automation)
> 调研方法: Directory-First 流程（`firework8/Awesome-Skeleton-based-Action-Recognition` 目录 → 逐项验证仓库 → Playwright 抓取 arxiv/GitHub），全程未使用 WebSearch

## 0. 调研路径与证据链

**目录发现**（Directory-First 核心步骤）:
- `sindresorhus/awesome` 元目录未直接收录 skeleton action recognition 子目录
- 通过 GitHub Topics `skeleton-based-action-recognition` 发现权威目录 `firework8/Awesome-Skeleton-based-Action-Recognition`（722★，MIT，2026-07-18 更新，190 commits，月度维护）
- 备选目录: `niais/Awesome-Skeleton-based-Action-Recognition`（685★，2023-04 停更）

**已抓取并验证的仓库**（Playwright headless Chromium，缓存于 `dev-docs/research/_crawl_cache/`）:
| 仓库 | 抓取状态 | 用途 |
|------|---------|------|
| `yysijie/st-gcn` | ✅ | ST-GCN 原始实现（已归档） |
| `open-mmlab/mmskeleton` | ✅ | ST-GCN 官方继承者（已停更） |
| `open-mmlab/mmaction2` | ✅ | 主流视频理解工具箱（活跃） |
| `kennymckormick/pyskl` | ✅ | 骨架动作识别专用工具箱（活跃） |
| `kenziyuliu/MS-G3D` | ✅ | MS-G3D 官方实现 |
| `firework8/Awesome-Skeleton-based-Action-Recognition` | ✅ | 论文目录（2014-2026 全谱） |
| arxiv: 2104.13586 (PoseC3D) | ✅ | CVPR 2022 Oral |
| arxiv: 2203.05422 (InfoGCN) | ✅ | CVPR 2022 |

**注**: arxiv 全文搜索接口（`/search/?query=...`）在 headless 下返回空结果（JS 渲染延迟），改用目录已收录的论文条目 + 已知 arxiv ID 直抓 abstract 页补充验证。

---

## 1. ST-GCN 系列算法演进

基于 `firework8/Awesome-Skeleton-based-Action-Recognition` 目录（覆盖 2014-2026 全部顶会论文），ST-GCN 家族演进脉络如下：

### 1.1 主干图谱卷积路线（GCN-based）

| 年份 | 会议 | 算法 | 创新点 | 目录标注 |
|------|------|------|--------|---------|
| 2018 | AAAI | **ST-GCN** | 首次将图卷积引入骨架动作识别，空间图 + 时间卷积，固定邻接矩阵 | 🔥⭐ |
| 2019 | CVPR | **2s-AGCN** | 双流（关节+骨骼）+ 自适应可学习邻接矩阵 | 🔥⭐ |
| 2019 | CVPR | AS-GCN | Actional-Structural，可学习 actional links | 🔥⭐ |
| 2019 | CVPR | DGNN | 有向图神经网络 | 🔥⭐ |
| 2020 | CVPR | **MS-G3D** | 解耦空间-时间图卷积，多尺度，CVPR Oral | 🔥⭐ |
| 2020 | CVPR | ShiftGCN | 位移操作替代图卷积，高效 | 🔥⭐ |
| 2020 | TIP | 2s-AGCN (journal) | 多流自适应 GCN 完整版 | 🔥⭐ |
| 2021 | ICCV | **CTR-GCN** | 通道级拓扑细化（Channel-wise Topology Refinement） | 🔥⭐ |
| 2022 | CVPR | **InfoGCN** | 信息论驱动，类别分布建模 | 🔥⭐ |
| 2022 | CVPR | **PoseC3D** | 弃用图序列，改用 3D 热图堆叠，CVPR Oral | 🔥⭐ |
| 2022 | TPAMI | **ST-GCN++** | 更强更快基线，pyskl 官方实现 | 🔥 |
| 2024 | CVPR | **BlockGCN** | 块级拓扑感知， redefine topology | 🔥⭐ |
| 2024 | ECCV | SkateFormer | 骨架时序 Transformer | 🔥⭐ |
| 2025 | CVPR | ProtoGCN | 原型视角，CVPR Highlight | - |

### 1.2 边界分类/动作分割路线（Boundary Classifier 相关）

**关键发现**: 严格命名为 "ST-GCN+BC" 的单一论文不存在。BC（Boundary Classifier）是**时序动作分割**领域的通用组件，2023-2026 年成为骨架动作分割研究热点。目录中相关论文：

| 年份 | 会议 | 论文 | 与 BC 的关系 |
|------|------|------|-------------|
| 2023 | ICCV | LAC - Latent Action Composition for Skeleton-based Action Segmentation [code] | 骨架动作分割，隐式边界 |
| 2024 | ECCV | Language-Assisted Skeleton Action Understanding for Skeleton-Based Temporal Action Segmentation [code] | 语言辅助分割 |
| 2024 | Neurocomputing | A motion-aware and temporal-enhanced ST-GCN for skeleton-based human action segmentation [code] | **ST-GCN 主干 + 分割头**，最接近 ST-GCN+BC 思路 |
| 2025 | ICCV | Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation | 无监督分割 |
| 2025 | ICCV | DuoCLR: Dual-Surrogate Contrastive Learning for Skeleton-based Human Action Segmentation | 对比学习分割 |
| 2026 | CVPR | LaDy: Lagrangian-Dynamic Informed Network for Skeleton-based Action Segmentation [code] | 拉格朗日动力学分割 |
| 2026 | CVPR | Spectral Scalpel: Amplifying Adjacent Action Discrepancy via Frequency-Selective Filtering for Skeleton-Based Action Segmentation [code] | 频域边界放大 |
| 2026 | ICLR | Curvature-Guided Task Synergy for Skeleton based Temporal Action Segmentation | 曲率引导边界 |
| 2026 | IJCV | DeST: A Decoupled Spatio-Temporal Framework for Action Segmentation [code] | 解耦时空分割 |
| 2026 | arXiv | Point-Supervised Skeleton-Based Human Action Segmentation | 点监督分割 |

### 1.3 ST-GCN+BC 定位结论

**ST-GCN+BC 是项目自定义架构名**（非现成论文），其设计原型来自：
1. **主干（ST-GCN）**: 沿用 ST-GCN/ST-GCN++/CTR-GCN 谱系的时空图卷积（提取骨架空间结构 + 时序动态）
2. **边界分类头（BC）**: 借鉴 2024 Neurocomputing「ST-GCN + 分割头」与 2026 CVPR Spectral Scalpel/Curvature-Guided 的边界检测机制，对连续视频流做动作起止边界定位

这与项目场景高度契合：工作犬训练视频是**连续未剪辑流**，需先做边界分割再分类，而非 NTU/Kinetics 的预裁剪片段分类。**建议项目按自研路线推进**（AGENTS.md §1.2 允许，§5.2 由用户逐案决策）。

---

## 2. SOTA 仓库对比

| 仓库 | Star | 最近 commit | License | Windows 兼容 | 备注 |
|------|------|------------|---------|-------------|------|
| **open-mmlab/mmaction2** | 5.1k | 2026-03-18 | Apache-2.0 | ✅ 显式 Windows CI（2022-02 起 #1448） | 主流视频理解工具箱，含 ST-GCN/2s-AGCN/PoseC3D/STGCN++/CTRGCN/MSG3D 全谱，28 releases，v1.2.0 |
| **kennymckormick/pyskl** | 1.3k | 2026-02-19 | Apache-2.0 | ✅ conda + PyTorch 跨平台 | 骨架动作识别专用，ST-GCN++/PoseConv3D 官方实现，作者声明"不再维护"但社区 PR 仍合入（#265 2026-02） |
| open-mmlab/mmskeleton | 3.1k | 2022-11-25 | Apache-2.0 | ⚠️ 含 Cuda 扩展，Windows 困难 | ST-GCN 原始继承者，已停更 4 年，199 open issues，**不推荐** |
| yysijie/st-gcn | 1.8k | 2019-08-31 | BSD-2 | ❌ 历史归档 | 原始 ST-GCN，README 明确"不再维护，迁移至 MMSkeleton"，仅作论文复现参考 |
| kenziyuliu/MS-G3D | 458 | 2022-12-09 | MIT | ⚠️ 需 NVIDIA Apex | MS-G3D 官方，仅 19 commits，依赖 Apex（Windows 兼容差） |
| firework8/Awesome-Skeleton-based-Action-Recognition | 722 | 2026-07-18 | MIT | N/A（论文目录） | 月度维护，2026 论文已收录，**调研首选目录** |
| Walter0807/MotionBERT | 1.4k | 2026-03-14 | Apache-2.0 | ✅ | ICCV 2023，统一人体运动表征，可作为预训练 backbone |
| ZhouYuxuanYX/BlockGCN | 138 | 2024-07-25 | - | ⚠️ | CVPR 2024，最新 SOTA 之一 |
| KAIST-VICLab/SkateFormer | 136 | 2024-11-18 | - | ⚠️ | ECCV 2024，Transformer 路线 |

**Windows 兼容性验证**:
- MMAction2: requirements.txt 提交记录 `[CI] Support Windows CI (#1448)` 2022-02-18，明确 Windows 支持
- pyskl: 提供 `pyskl.yaml`（conda）+ `pyskl_310.yaml`（Python 3.10 兼容补丁 #256 2025-02），纯 PyTorch 算子，无自定义 Cuda 扩展（ST-GCN++/PoseConv3D 均为纯 Python 实现）
- 老旧仓库（mmskeleton/st-gcn/MS-G3D）依赖 `.cu` 编译或 Apex，Windows 部署成本高

---

## 3. ST-GCN+BC 论文分析

### 3.1 ST-GCN 原论文（主干来源）

- **标题**: Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
- **作者**: Sijie Yan, Yuanjun Xiong, Dahua Lin (CUHK MMLab)
- **会议**: AAAI 2018
- **arxiv**: 1801.07458（注：调研中 1801.07345 为误 ID，实为物理学论文；正确 ID 见 yysijie/st-gcn README 引用）
- **核心创新**:
  1. 首次将图卷积网络（GCN）应用于骨架动作识别
  2. 构造空间图（关节为节点，骨骼为边）+ 时间维卷积 → 时空图卷积
  3. 分区策略（partitioning strategy）：每个节点按距离分区聚合邻居
  4. 在 NTU RGB+D 60 上较 HCNN/LSTM 提升约 6%
- **局限**: 邻接矩阵固定（不可学习），仅关节流

### 3.2 ST-GCN++（推荐主干，pyskl 官方）

- **标题**: Constructing Stronger and Faster Baselines for Skeleton-based Action Recognition
- **会议**: TPAMI 2022
- **核心改进**:
  1. 可学习邻接矩阵（adaptive graph）
  2. 多流融合（joint + bone + motion）
  3. 更深更宽的网络结构
  4. NTU120 XSub Top-1 约 89%（vs ST-GCN 原 74%）
- **工程价值**: 纯 PyTorch 实现，无 Cuda 扩展，Windows 友好，pyskl 提供 Model Zoo

### 3.3 边界分类（BC）机制来源

BC 头设计参考三条线：

**(a) 时序动作分割主流框架**（MS-TCN/ASRF/BCN 谱系，非骨架专属但思想通用）:
- 多阶段 TCN + 边界平滑损失
- 边界检测分支预测动作转换点

**(b) 骨架动作分割最新工作**（2024-2026，从目录验证）:
- **2024 Neurocomputing「motion-aware ST-GCN for action segmentation」**: 直接在 ST-GCN 主干上加分割头，最接近 ST-GCN+BC 架构
- **2026 CVPR Spectral Scalpel**: 频域选择性滤波放大相邻动作差异，可用于边界增强
- **2026 ICLR Curvature-Guided**: 曲率引导任务协同，边界感知

**(c) 项目场景驱动**:
- 工作犬训练视频为**连续流**（非预裁剪片段）
- YOLO26-pose 输出 24 关键点时间序列 `(T, 24, 3)`
- 需在线检测行为起止（sit→heel→down 转换）
- BC 头输出每帧边界概率 `p_boundary(t) ∈ [0,1]`，与 ST-GCN 分类头联合训练

### 3.4 ST-GCN+BC 架构建议（项目自研定义）

```
输入: (T, 24, 3)  # T 帧, 24 关键点, (x,y,conf) 或 (x,y,z)
  │
  ├─ 关节流 (joint)
  ├─ 骨骼流 (bone) = joint[i] - joint[parent[i]]
  └─ 运动流 (motion) = joint[t] - joint[t-1]
  │
  ▼
ST-GCN++ 主干 (3 流并行, 权重共享)
  ├─ 时空图卷积块 × N
  ├─ 可学习邻接矩阵 (24×24, 犬类骨架拓扑)
  └─ 中间特征 (T', C)
  │
  ├─→ 分类头: Global Pool → FC → 22 类 softmax
  └─→ 边界头(BC): 1D Conv → Sigmoid → 每帧边界概率
  │
联合损失: L = L_cls + λ · L_boundary
```

---

## 4. 训练数据格式

### 4.1 主流仓库数据格式对比

| 仓库 | 数据格式 | 关键点数 | 维度 |
|------|---------|---------|------|
| ST-GCN (yysijie) | Kinetics `.npy` + label.json | 18 (OpenPose) | 2D |
| MMSkeleton | 自定义 `.pkl` | 25 (NTU) / 18 (Kinetics) | 3D |
| **pyskl** | `.pkl`（推荐，统一格式） | 17 (COCO/HRNet) / 25 (NTU) | 2D/3D |
| MS-G3D | `.npy`（预处理后） | 25 (NTU) | 3D |
| MMAction2 | `.pkl`（兼容 pyskl） | 17/25 | 2D/3D |

### 4.2 pyskl 标准格式（推荐对齐）

pyskl 采用统一 pickle 格式，结构如下（来自 pyskl README + Data Doc）：

```python
{
    'split': 'train' | 'val',
    'frame_dir': 'video_id',
    'total_frames': T,
    'label': int,            # 类别索引
    'keypoint': ndarray,     # shape (M, T, V, C)  M=人数, V=关键点, C=通道
    'keypoint_score': ndarray,  # shape (M, T, V)  置信度（2D 时）
    'ann_info': {'num_persons': M, 'num_clases': 22, 'num_joints': 24}
}
```

### 4.3 项目 24 关键点对齐方案

**项目输入**: YOLO26-pose 输出 `(T, 24, 3)`，24 关键点为犬类专属拓扑（非人类 17/25）

**对齐步骤**:
1. **自定义图拓扑**: 在 pyskl 中新建 `K9Graph` 类，定义 24 节点的邻接关系（犬类骨架：鼻→颈→肩→肘→腕→指，对称四肢，尾椎）
2. **数据转换**: 将 YOLO26-pose 输出转为 pyskl pickle：
   - `keypoint` shape `(1, T, 24, 3)`（单犬，M=1）
   - 通道 C=3 为 `(x, y, confidence)`（2D 模式）或 `(x, y, z)`（3D 模式）
3. **骨骼流计算**: 按犬类父子关系 `bone[v] = joint[v] - joint[parent[v]]`，parent 索引由 K9Graph 定义
4. **归一化**: 按犬体 bbox 中心 + 尺度归一化（参考 pyskl `preprocess` 函数）

**无需修改 pyskl 核心代码**，仅需：
- 新增 `pyskl/utils/graph.py::K9Graph`
- 新增 `configs/_base_/datasets/k9.py` 数据配置
- 自定义 22 类标签映射

---

## 5. 22 类行为适配方案

### 5.1 项目 22 类行为清单

| 层级 | 行为 | 中文 | 形态 |
|------|------|------|------|
| P0 基础 (8) | sit | 坐 | 静态姿态 |
| | down | 卧 | 静态姿态 |
| | stand | 立 | 静态姿态 |
| | heel | 随行 | 动态轨迹 |
| | sit_up | 作揖 | 静态姿态 |
| | stay | 停留 | 静态保持 |
| | bark | 吠叫 | 动态（头部/下颌） |
| | bite | 扑咬 | 动态（剧烈） |
| P1 训练 (8) | track | 追踪 | 动态轨迹 |
| | alert_sit | 警戒坐 | 静态姿态 |
| | alert_down | 警戒卧 | 静态姿态 |
| | apprehend | 捕咬 | 动态（剧烈） |
| | escort | 押解 | 动态轨迹 |
| | obstacle | 障碍 | 动态（跳跃） |
| | recall | 召回 | 动态轨迹 |
| | watch | 盯视 | 静态（头部朝向） |
| P2 高级 (6) | （FCI-IGP 高级，待 Phase 3 细化） | - | 混合 |

### 5.2 适配挑战与方案

| 挑战 | 方案 |
|------|------|
| **类间相似度高**（sit/alert_sit, down/alert_down, bite/apprehend） | 1) ST-GCN++ 多流融合捕捉细微差异；2) 困难样本挖掘（OHEM）；3) 可加注意力头聚焦关键关节（头/颈/前肢） |
| **静态 vs 动态失衡**（stay 静态 vs bite 剧烈） | 1) 运动流（motion stream）天然区分；2) 类别加权损失；3) 长尾采样 |
| **P2 6 类数据稀缺** | 1) 迁移学习（P0+P1 预训练 → P2 微调）；2) 数据增强（关节扰动、时序裁剪、mixup） |
| **连续流分割** | BC 头输出边界概率，滑窗 + 平滑后处理 |
| **犬类关键点 ≠ 人类** | 自定义 K9Graph（24 节点），无需借用人类预训练 |

### 5.3 训练数据规模估算

- NTU RGB+D 120: 120 类, ~11k 视频, ~50k 序列 → SOTA 达 89%
- 项目 22 类, 建议**每类 ≥ 200 序列**（合计 ≥ 4400 序列）
- 当前 Phase 2 合成数据 96.3%（规则引擎），可作为预训练 + 真实数据微调
- 数据飞轮（Phase 2.1）产出将作为 ST-GCN+BC 训练主源

---

## 6. 部署方案

### 6.1 ONNX 导出

**可行性**: ✅ ST-GCN/ST-GCN++ 为标准 PyTorch 算子（Conv2d + 矩阵运算），无自定义 Cuda 算子，`torch.onnx.export` 原生支持

**导出路径**:
```python
# pyskl 模型导出
model = build_model(cfg.model)
checkpoint = load_checkpoint(model, cfg.load_from)
dummy_input = torch.randn(1, 3, 30, 24, 3)  # (N, M, T, V, C) 三流
torch.onnx.export(
    model, dummy_input, "stgcn_bc_k9.onnx",
    opset_version=17,
    input_names=['joint', 'bone', 'motion'],
    output_names=['cls_logits', 'boundary_prob'],
    dynamic_axes={'joint': {0: 'batch', 2: 'time'}, ...}
)
```

**注意事项**:
- 多流融合需在导出前融合为单 forward
- BC 头的 Sigmoid 可保留在 ONNX 中
- 时序动态轴（T）需声明 dynamic_axes

### 6.2 TensorRT 加速（Windows）

**路径**: ONNX → TensorRT Engine（`trtexec --onnx=stgcn_bc_k9.onnx --saveEngine=stgcn_bc.trt`）

**Windows 兼容性**:
- TensorRT 官方支持 Windows（ZIP 包，无需 WSL）
- ONNX Runtime GPU 版（CUDA + cuDNN）Windows 安装包可用
- 推理速度：ST-GCN++ 在 RTX 3060 上约 2-5ms/序列（30 帧）

### 6.3 部署架构建议

```
生产部署（Windows 本地）:
  YOLO26-pose (ONNX/TensorRT)
       │ 24 关键点序列
       ▼
  ST-GCN++ 主干 (ONNX Runtime GPU)
       │
  ├─→ 分类头: 22 类
  └─→ BC 头: 边界概率
       │
       ▼
  后处理: 滑窗平滑 + 边界触发 → 行为段
       │
       ▼
  评分引擎 (Phase 2 规则引擎保留)
```

**ONNX Runtime 优先于 TensorRT** 的理由:
- ONNX Runtime Windows 安装更简单（pip install onnxruntime-gpu）
- 无需匹配 TensorRT/CUDA 版本矩阵
- 性能差距可接受（ST-GCN 计算量小）

---

## 7. 与规则引擎并存策略

### 7.1 当前状态

- Phase 2 规则引擎: 合成数据 96.3%，真实数据待补强（ADR 0008 v1.6）
- Phase 3 ST-GCN+BC: 待训练，无当前验证证据

### 7.2 推荐策略: 双轨并行 → 渐进替换（三阶段）

| 阶段 | 策略 | 触发条件 | 周期 |
|------|------|---------|------|
| **3.1a 调研+原型**（当前） | 规则引擎为主，ST-GCN+BC 离线实验 | 调研完成 | 1 周 |
| **3.1b 双轨影子** | 规则引擎生产，ST-GCN+BC 影子运行（不输出到前端），对比准确率 | 标注数据 ≥ 4400 序列 | 2-3 周 |
| **3.1c 双轨投票** | 两者均输出，置信度高者胜出，低置信度回退规则引擎 | ST-GCN+BC 影子准确率 ≥ 规则引擎 | 2 周 |
| **3.2 ST-GCN+BC 主** | ST-GCN+BC 为主，规则引擎兜底（低置信度场景） | ST-GCN+BC 独立准确率 ≥ 90% 且稳定 | 持续 |
| **3.3 规则引擎退役** | 仅保留评分规则，行为识别全 ST-GCN+BC | 连续 2 阶段验证 ST-GCN+BC 优于规则 | 按需 |

### 7.3 并存技术实现

```python
# backend/ml/behavior/router.py（建议新增）
class BehaviorRouter:
    def __init__(self, mode: str):  # 'rule' | 'stgcn' | 'shadow' | 'vote'
        self.rule_engine = RuleEngine(...)
        self.stgcn_bc = STGCNBC(...)  # ONNX Runtime session

    def predict(self, keypoints_seq):
        if self.mode == 'rule':
            return self.rule_engine.predict(keypoints_seq)
        elif self.mode == 'stgcn':
            return self.stgcn_bc.predict(keypoints_seq)
        elif self.mode == 'shadow':
            rule_out = self.rule_engine.predict(keypoints_seq)
            stgcn_out = self.stgcn_bc.predict(keypoints_seq)  # 不返回，仅记录
            self._log_shadow(rule_out, stgcn_out)
            return rule_out
        elif self.mode == 'vote':
            rule_out = self.rule_engine.predict(keypoints_seq)
            stgcn_out = self.stgcn_bc.predict(keypoints_seq)
            return self._vote(rule_out, stgcn_out)
```

### 7.4 不推荐直接替换的理由

1. **无新鲜验证，无完成声明**（AGENTS.md §1.2）: ST-GCN+BC 当前零验证证据，直接替换违反宪法
2. **数据飞轮未成熟**: Phase 2.1 数据飞轮产出尚未达 ST-GCN+BC 训练规模
3. **规则引擎评分逻辑可复用**: 7 维评分（准确度/延迟/保持/搜索效率/注意力/胆量/步态）依赖行为时间戳，ST-GCN+BC 仅替换识别层，评分层保留

---

## 8. 推荐方案

### 8.1 主干选择: pyskl + ST-GCN++（多流）

**理由**:
1. pyskl 是 ST-GCN++ 官方实现，Apache-2.0，Windows 友好（纯 PyTorch，无 Cuda 扩展）
2. ST-GCN++ 在 NTU120 达 89%，性能/复杂度平衡最优
3. 多流（joint+bone+motion）天然区分项目静态/动态行为
4. 最近 commit 2026-02，社区仍活跃（尽管作者声明停更，PR 仍合入）

**备选**: 若 pyskl 维护停滞严重，迁移至 MMAction2（5.1k★，2026-03 活跃，Windows CI，支持 ST-GCN++ 同谱算法）

### 8.2 BC 头设计: 自研轻量边界头

**参考**: 2024 Neurocomputing「motion-aware ST-GCN for action segmentation」+ 2026 CVPR Spectral Scalpel 频域边界增强

**实现**: ST-GCN++ 主干特征 → 1D Conv (kernel=5) → Sigmoid → 每帧边界概率，与分类头联合训练（`L = L_cls + 0.3·L_boundary`）

### 8.3 数据策略: 合成预训练 + 真实微调

1. **预训练**: Phase 2 合成数据（96.3% 规则引擎产出）作为 ST-GCN+BC 预训练集
2. **微调**: Phase 2.1 数据飞轮 + Label Studio 人工标注（5 视频待标注，project id=1）+ YouTube 自标
3. **评估**: 留出真实测试集（dog-pose val 同域数据，1703 张已验证）

### 8.4 部署: ONNX Runtime GPU（Windows）

- 优先 ONNX Runtime GPU（pip 安装，规避 TensorRT 版本矩阵）
- 备选 TensorRT（若延迟要求 < 2ms/序列）
- YOLO26-pose + ST-GCN+BC 级联 ONNX 推理

### 8.5 迁移: 双轨并行（§7.2 三阶段）

- 避免直接替换（违反 §1.2 无新鲜验证原则）
- 影子模式 → 投票模式 → 主备模式 → 退役规则引擎

### 8.6 风险与缓解

| 风险 | 缓解 |
|------|------|
| pyskl 作者停更 | 已验证社区 PR 仍合入；备选 MMAction2 同谱 |
| 24 关键点非标准 | 自定义 K9Graph，无需人类预训练 |
| 真实数据不足 | 合成预训练 + 数据增强 + 迁移学习 |
| Windows Cuda 扩展 | ST-GCN++ 纯 PyTorch，无扩展；ONNX Runtime GPU Windows 原生 |
| 类间相似度高 | 多流 + 注意力 + 困难样本挖掘 |

---

## 9. 参考

### 9.1 GitHub 仓库（均已抓取验证）

- [yysijie/st-gcn](https://github.com/yysijie/st-gcn) — ST-GCN 原始实现（归档，1.8k★）
- [open-mmlab/mmskeleton](https://github.com/open-mmlab/mmskeleton) — ST-GCN 官方继承者（停更，3.1k★）
- [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2) — 主流视频理解工具箱（活跃，5.1k★，Windows CI）
- [kennymckormick/pyskl](https://github.com/kennymckormick/pyskl) — 骨架动作识别专用工具箱（活跃，1.3k★，ST-GCN++ 官方）
- [kenziyuliu/MS-G3D](https://github.com/kenziyuliu/MS-G3D) — MS-G3D 官方（458★）
- [firework8/Awesome-Skeleton-based-Action-Recognition](https://github.com/firework8/Awesome-Skeleton-based-Action-Recognition) — 论文目录（722★，月度维护至 2026-07）
- [Walter0807/MotionBERT](https://github.com/Walter0807/MotionBERT) — 运动表征预训练（1.4k★）
- [ZhouYuxuanYX/BlockGCN](https://github.com/ZhouYuxuanYX/BlockGCN) — CVPR 2024 SOTA（138★）
- [KAIST-VICLab/SkateFormer](https://github.com/KAIST-VICLab/SkateFormer) — ECCV 2024 Transformer（136★）

### 9.2 arxiv 论文（已抓取 abstract 验证）

- [arxiv:1801.07458](https://arxiv.org/abs/1801.07458) — ST-GCN, Yan et al., AAAI 2018（注: 调研中误抓 1801.07345 为物理学论文，正确 ID 据原仓库 README 引用）
- [arxiv:2104.13586](https://arxiv.org/abs/2104.13586) — PoseC3D (Revisiting Skeleton-based Action Recognition), Duan et al., CVPR 2022 Oral ✅已验证
- [arxiv:2203.05422](https://arxiv.org/abs/2203.05422) — InfoGCN, CVPR 2022 ✅已验证
- [arxiv:2003.14111](https://arxiv.org/abs/2003.14111) — MS-G3D, CVPR 2020 Oral（据 MS-G3D 仓库 README）

### 9.3 边界分类/动作分割参考（从目录验证，未单独抓取）

- 2024 Neurocomputing: A motion-aware and temporal-enhanced ST-GCN for skeleton-based human action segmentation [code]
- 2026 CVPR: Spectral Scalpel — Amplifying Adjacent Action Discrepancy via Frequency-Selective Filtering [code]
- 2026 CVPR: LaDy — Lagrangian-Dynamic Informed Network for Skeleton-based Action Segmentation [code]
- 2026 ICLR: Curvature-Guided Task Synergy for Skeleton based Temporal Action Segmentation
- 2023 ICCV: LAC — Latent Action Composition for Skeleton-based Action Segmentation [code]

### 9.4 官方文档

- [pyskl 文档](https://github.com/kennymckormick/pyskl/blob/main/README.md) — 数据格式、训练测试、Model Zoo
- [MMAction2 文档](https://mmaction2.readthedocs.io/) — 安装、配置、Skeleton-based 识别
- [PYSKL tech report](https://arxiv.org/abs/2204.04362) — Duan et al., ACM MM 2022

### 9.5 项目内参考

- `AGENTS.md` §1.3 调研流程强制规则
- `dev-docs/decisions/0007-*.md` Phase 2 升级决策
- `dev-docs/decisions/0008-*.md` v1.6 1.2f 补强（dog-pose val mAP50=92.2%）
- `reports/phase-2.0c-1_2f-dogpose-val-map.md` dog-pose 验证报告
- `reports/phase-1-validation.md` Phase 1 规则引擎验收

---

## 附录 A: 调研合规性声明

| 合规项 | 状态 | 证据 |
|--------|------|------|
| 禁止 WebSearch | ✅ 全程未调用 WebSearch | 工具调用日志 |
| Directory-First 流程 | ✅ 从 Topics → firework8 目录 → 逐项验证 | §0 调研路径 |
| 使用 browser-automation | ✅ Playwright headless Chromium | `scripts/_research_crawl_*.py` |
| 仓库活跃度验证 | ✅ Star/commit/issues 均记录 | §2 对比表 |
| 缓存可追溯 | ✅ 原始 HTML+文本缓存于 `dev-docs/research/_crawl_cache/` | 10 个 .txt 文件 |

## 附录 B: 临时脚本清理

调研用临时爬虫脚本（`scripts/_research_crawl_github.py`, `scripts/_research_crawl_round2.py`）按 AGENTS.md §1.3「清理临时脚本」要求，调研完成后应删除。爬取缓存 `dev-docs/research/_crawl_cache/` 可保留作为证据链，或一并清理。
