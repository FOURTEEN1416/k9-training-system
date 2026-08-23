# BCST-GCN + ST-GCN 架构深度分析报告

> 生成日期: 2026-07-01 | 基于 Frontiers in Veterinary Science 2026 + GitHub 开源代码分析
> 警犬训练项目背景

---

## 1. ST-GCN 开源实现全景分析

### 1.1 原始仓库

#### yysijie/st-gcn (1.8k stars)
- **地址**: https://github.com/yysijie/st-gcn
- **状态**: **已归档** — 作者已转移至 MMSkeleton
- Input: (N, C, T, V, M) — Batch, Channels, Frames, Joints, Persons

#### open-mmlab/mmskeleton (已不再维护)
- ST-GCN 作者实验室官方工具箱, 已全职迁移至 MMAction2

### 1.2 MMAction2（当前最成熟实现）

MMAction2 是骨架行为识别的**业界标准工具箱**, 集成多种 SOTA 模型。

#### ST-GCN 实现 (`backbones/stgcn.py`)

| 组件 | 实现 | 细节 |
|------|------|------|
| Graph 构造 | `Graph` 类 (`utils/graph.py`) | 支持 openpose(18)/nturgb+d(25)/coco(17) + 自定义 dict |
| 邻接矩阵 | 3 子集归一化 | 根节点 + 向心 + 离心 |
| 空间卷积 | `unit_gcn` (`utils/gcn_utils.py`) | 1x1 Conv2d + einsum 图卷积 |
| 时间卷积 | `unit_tcn` / `mstcn` | kernel=9 TCN / 多尺度 MS-TCN |
| 残差连接 | 标准 residual | channel 变化时 1x1 Conv 投影 |

**关键代码 — einsum 图卷积**:
```python
if self.conv_pos == '"'"'pre'"'"':
    x = self.conv(x)  # 1x1 Conv: C_in -> C_out * num_subsets
    x = x.view(n, self.num_subsets, -1, t, v)
    x = torch.einsum('"'"'nkctv,kvw->nctw'"'"', (x, A))  # 邻接加权聚合
elif self.conv_pos == '"'"'post'"'"':
    x = torch.einsum('"'"'nctv,kvw->nkctw'"'"', (x, A))
    x = x.view(n, -1, t, v)
    x = self.conv(x)
```

**ST-GCN Pipeline**:
```
Input: (N, M, T, V, C)
  -> permute: (N, M, V, C, T)
  -> data_bn: BatchNorm
  -> reshape: (N*M, C, T, V)
  -> [STGCNBlock x 10]
       -> unit_gcn(x):    einsum 图卷积 + 1x1 Conv
       -> unit_tcn(x):    时间卷积 (kernel=9)
       -> residual + ReLU
  -> reshape: (N, M, C'"'"', T'"'"', V'"'"')
Output: (N, M, 256, T/4, V)
```

**默认超参数**:
```python
graph_cfg = dict(layout='"'"'openpose'"'"', mode='"'"'stgcn_spatial'"'"')
base_channels = 64
ch_ratio = 2
num_stages = 10
inflate_stages = [5, 8]    # 通道扩张
down_stages = [5, 8]       # 时间下采样
```

**通道路径**: Stage 1: 3->64 | 2-4: 64 | 5: 64->128(s2) | 6-7: 128 | 8: 128->256(s2) | 9-10: 256

### 1.3 ST-GCN++ (PYSKL, arxiv 2205.09443)

改进: 自适应图卷积(adaptive='"'"'init'"'"'), 残差GCN, 多尺度TCN, 四流融合

**NTU60 X-Sub 精度**:
| 方法 | 2D joint | 2D 四流 | 3D 四流 |
|------|----------|---------|---------|
| ST-GCN | 86.14 | 90.69 | 87.31 |
| ST-GCN++ | 89.39 | 91.87 | 91.87 |
| 2s-AGCN | 88.13 | 91.11 | 88.98 |
| PoseC3D | 85.42 | 92.96 | — |

### 1.4 其他实现

- **CTR-GCN** (700+ stars) — ICCV2021, 通道级拓扑精炼, NTU60 92.4
- **PoseC3D** — 3D heatmap 堆叠替代图序列 (详见第3节)

---

## 2. BCST-GCN 论文深度解析与可重现性分析

### 2.1 论文概要

| 属性 | 内容 |
|------|------|
| 论文 | BCST-GCN: a skeleton-based spatiotemporal GCN with bidirectional cross-attention for pig behavior recognition |
| 期刊 | Frontiers in Veterinary Science, April 2026 |
| DOI | 10.3389/fvets.2026.1782396 |
| 总精度 | **94.43%** (baseline +6.94%, precision +5.61%, recall +6.88%) |
| 开源 | **未公开代码** |
| 环境 | Python 3.8, PyTorch, RTX 3060, DeepLabCut v2.3.7 |
| 行为 | 4 类: feeding, walking, lying, dog-sitting |

### 2.2 方法论解析

论文提出三步核心设计:

**1. DLC 姿势估计**: 用 DeepLabCut 提取猪骨架关键点, 构建拓扑结构

**2. 轻量化 ST-GCN**: "streamline the ST-GCN by removing redundant network layers"
- 推测: 从原始的 10 stages 减少到 6-8 stages
- 可能仅使用 joint 单流 (而非四流)

**3. 双向交叉注意力 BC 模块**:

BC 模块包含两条互补路径:

- **全局自注意力 (Global Self-Attention)**: 动态构建关节间潜在依赖拓扑, 突破静态邻接矩阵约束, 捕获非物理连接关节的长距离关系
- **局部自注意力 (Local Self-Attention)**: 基于1-hop邻接 mask, 量化物理连接强度, 实现细粒度局部动态感知

```
# 推测的 BC+STGCNBlock 实现 (最可能方案)
def bc_stgcn_block(x, A_static):
    x_gcn = unit_gcn(x, A_static)                # 原始 GCN
    x_bc_global = global_self_attn(x)             # 全局注意力
    x_bc_local = local_self_attn(x, A_static)      # 局部注意力 (masked)
    x = unit_tcn(x_gcn + x_bc_global + x_bc_local)  # 融合 + 时间卷积
    return relu(x + residual(x))
```

**与 lucidrains/bidirectional-cross-attention 关联**:
- 共享 Query/Key 注意力机制
- 同时更新全局和局部表示
- 该库 197 stars, 可直接 pip 安装

### 2.3 推测超参数

| 参数 | 推测值 | 依据 |
|------|--------|------|
| backbone base_channels | 64 | ST-GCN 默认 |
| 总 stages | 6-8 | 轻量化 |
| attention heads | 4-8 | 常见设置 |
| 时间窗口 | 20-50 帧 | 6s@25fps |
| 关键点数 | 12-17 | 猪 DLC 常见配置 |

### 2.4 可重现性评估

**结论**: 可行, 工作量 3-5 天

**路径**:
1. Fork MMAction2 `backbones/stgcn.py`
2. 在 `STGCNBlock.forward()` 中添加 BC 模块
3. 减少 stages (10 -> 6-8)
4. 添加狗骨架 Graph 定义

---

## 3. PoseConv3D (PoseC3D) 分析

### 3.1 核心思想

用 **3D heatmap 堆叠** 替代图序列:
```
ST-GCN:  坐标序列 -> 图拓扑 -> GCN
PoseC3D: 坐标序列 -> 3D heatmap 体积 -> 3D ConvNet
```

### 3.2 优劣对比

| 维度 | ST-GCN | PoseC3D |
|------|--------|---------|
| 抗噪性 | 对抖动敏感 | **强** (高斯 heatmap) |
| 多动物 | 需 M 维度 | **天然支持** |
| 多模态融合 | 困难 | **容易** |
| 参数量 | ~1.4M | ~3.9M |
| 推理速度 | 快 | 较慢 |

### 3.3 对狗场景的适用性

| 因素 | ST-GCN | PoseC3D |
|------|--------|---------|
| 骨架差异大 | 需自定义 Graph | 仅改 in_channels |
| 关节抖动 | 敏感 | **天然抗噪** |
| 实时性 | **小计算量** | 3D Conv 较慢 |
| 小样本 | 需大量数据 | 可用 Kinetics 预训练 |

---

## 4. ASBAR 代码结构深度分析

### 4.1 仓库概况

| 属性 | 内容 |
|------|------|
| 地址 | https://github.com/MitchFuchs/asbar |
| Stars | 17 | MIT 许可证 |
| 框架 | DeepLabCut + MMAction2 PoseC3D |
| 应用 | 大猿 9 类行为, 74.98% Top-1 |

### 4.2 目录结构

```
asbar/
  main.py              # 入口: DEEPLABCUT / POSEC3D 路由
  conf.yaml            # DLC + MM 配置
  requirements.yaml    # conda env (Python 3.8, PyTorch 1.7)
  gui.sh               # 终端 GUI
  tools/
    dlc_utils.py       # DLC: 创建/训练/评估
    mmaction_utils.py  # DLC -> PoseC3D 数据转换
    mm_distr_train.py  # 分布式训练 (torch.distributed)
  data/omc/            # 姿势数据集
  models/mmaction2/
    configs/skeleton/posec3d/asbar.py  # PoseC3D 完整配置
```

### 4.3 关键数据流

```
视频 -> DLC 推理 -> *_full.pickle (关键点+置信度)
  -> mmaction_utils.py (sample_interval=20, seq_len=20)
  -> train.pkl (PoseC3D 格式)
  -> PoseC3D SlowOnly R50 (in_channels=17, 48 frames, 64x64)
  -> 9 类行为
```

### 4.4 PoseC3D 配置 (ASBAR)

```python
backbone = dict(depth=50, in_channels=17, base_channels=32,
                num_stages=3, stage_blocks=(4, 6, 3))
# Pipeline: 48帧, 64x64 heatmap, sigma=0.6
# SGD lr=0.025, CosineAnnealing, 2 epochs
```

### 4.5 替换狗 24 关键点工作量

| 步骤 | 工作量 |
|------|--------|
| DLC 关键点标注 | 3-5 天 |
| MMAction2 Graph 定义 | 2-3 小时 |
| 改 in_channels + num_classes | 30 分钟 |
| 训练 | 1-2 周 |
| BC 模块 (可选) | 3-5 天 |
| **总计** | **2-4 周** |

---

## 5. 警犬场景架构选型建议

### 5.1 方案打分

| 方案 | 精度 | 效率 | 泛化 | 可定制 | 实施难度 | 总分 |
|------|------|------|------|--------|---------|------|
| **PoseC3D (ASBAR 路径)** | 9 | 6 | 9 | 7 | 7 | **38/50** |
| MMAction2 ST-GCN | 7 | 9 | 6 | 8 | 6 | 36/50 |
| ST-GCN++ | 8 | 8 | 7 | 8 | 5 | 36/50 |
| BCST-GCN (复现) | 8 | 8 | 7 | 9 | 4 | 36/50 |

### 5.2 推荐: 两阶段渐入

#### 阶段一 (Quick Win, 1-2 周): ASBAR + PoseC3D
- ASBAR 提供完整 DLC->PoseC3D pipeline
- 狗关键点抖动多, PoseC3D heatmap 天然抗噪
- 仅需改 in_channels 和 num_classes
- 可复用 Kinetics400 预训练

#### 阶段二 (进阶, 3-4 周): 自定义 ST-GCN + BC
- 需要实时推理时升级
- MMAction2 内注册狗骨架 + Graph 定义
- 添加 BC 模块 (双向交叉注意力)

### 5.3 警犬行为分类建议

| 行为 | 难度 | 关注关节 |
|------|------|---------|
| 坐 | 中 | 后腿关节角度 |
| 卧 | 中 | 前腿+躯干 |
| 立 | 低 | 前腿+头颈 |
| 警戒 | 高 | 耳朵+尾巴+躯干 |
| 扑咬 | 高 | 嘴+前腿 |
| 嗅探 | 中 | 鼻子+颈+尾巴 |

### 5.4 决策树

```
有标注数据? -> 否 -> DLC 标注
实时推理? -> 是 -> ST-GCN (MMAction2)
跨品种泛化? -> 是 -> PoseC3D (ASBAR)
精度 >95%? -> 是 -> PoseC3D + RGB 多模态
需关节注意力? -> 是 -> BCST-GCN
```

---

## 附录: 资源索引

| 资源 | 链接 |
|------|------|
| BCST-GCN 论文 | https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2026.1782396 |
| yysijie/st-gcn (1.8k) | https://github.com/yysijie/st-gcn |
| MMAction2 | https://github.com/open-mmlab/mmaction2 |
| ASBAR | https://github.com/MitchFuchs/asbar |
| PYSKL (ST-GCN++) | https://arxiv.org/abs/2205.09443 |
| CTR-GCN | https://github.com/Uason-Chen/CTR-GCN |
| 双向交叉注意力 | https://github.com/lucidrains/bidirectional-cross-attention |
| DeepLabCut | https://github.com/DeepLabCut/DeepLabCut |
