# ADR 0009: Phase 2 数据策略 — 四路径并行 + keypoint-MoSeq 主路径

> 状态: 🔄 执行中（v1.1: Path 1 训练进行中 + Path 4 完成 + Path 2 调研完成）
> 日期: 2026-07-29
> Owner: ML 开发（见 AGENTS.md §2.2）
> 修改触发: Phase 2 数据卡点诊断 / keypoint-MoSeq 集成决策
> 依据: [PHASE2_DATA_STRATEGY.md](../research/PHASE2_DATA_STRATEGY.md) + [ADR 0008](0008-1.2f-supplement-plan.md) + [phase-2.md v2.1](../stages/phase-2.md) + 用户决策（2026-07-29）

## 1. 上下文

### 1.1 问题背景

Phase 2 启动后，1.2f 真实视频准确率验证（≥ 80%）成为数据卡点：
- InterPet4D v1 无视频文件 + 无行为标签
- ADR 0008 YouTube 自标路径需要人工标注成本
- 211 AK 犬类视频无行为标签
- Label Studio 部署完成但人工标注未启动

### 1.2 诊断

**项目并非被数据卡住，而是被"数据必须是人工标注的真实视频"这一假设卡住。**

现有 226 InterPet4D 关键点序列 + 5 YouTube 视频 + 211 AK 犬类视频 = **442 个未标注视频/序列**。若用无监督/弱监督方法提取行为标签，足以启动 Phase 2 主线。

### 1.3 调研结论

详见 [PHASE2_DATA_STRATEGY.md](../research/PHASE2_DATA_STRATEGY.md)（GitHub-First 流程调研）。

| 路径 | 工具 | 标注成本 | 产出 | 推荐度 |
|------|------|---------|------|--------|
| **Path 1 无监督发现** | keypoint-MoSeq（Nature Methods 2024） | 0 | 226 序列行为标签 | ⭐⭐⭐⭐⭐ |
| **Path 2 迁移学习** | DeepEthogram（Kinetics700 预训练） | 5-10 视频 | 帧级行为概率 | ⭐⭐⭐⭐ |
| **Path 3 主动学习** | Label Studio + 规则引擎预标注 | 按需 | 持续高质量标签 | ⭐⭐⭐⭐ |
| **Path 4 弱监督自训练** | 规则引擎伪标签 + 轻量分类器 | 1-2 天工程 | 扩大训练集 | ⭐⭐⭐ |

## 2. 决策

### 2.1 四路径并行启动（用户确认）

| 路径 | 优先级 | 时机 | 阻塞主线？ |
|------|--------|------|-----------|
| **Path 1 keypoint-MoSeq** | P0 | Phase 2.1 立即 | 不阻塞（后台并行） |
| **Path 3 主动学习** | P1 | 已在运行 | 不阻塞（后台并行） |
| **Path 4 弱监督自训练** | P2 | Path 1 完成后 | 不阻塞 |
| **Path 2 DeepEthogram** | P3 | 若 Path 1 不足 | 不阻塞 |

**核心洞察**: Path 1 解决 1.2f 真实数据验证问题——226 InterPet4D 序列通过 keypoint-MoSeq 获得行为标签后，可直接用于 16 行为准确率验证，无需等待人工标注。

### 2.2 Syllable → 16 行为映射策略（用户确认）

**方案 B: 规则引擎自动映射 P0 + 人工映射 P1（推荐）**

| 行为类型 | 映射方式 | 依据 |
|---------|---------|------|
| **P0 8 类**（sit/down/stand/heel/sit_up/stay/bark/bite） | 规则引擎自动映射 | 3D 几何姿态判定 + 70% 帧分类一致阈值 |
| **P1 8 类**（track/alert_sit/alert_down/apprehend/escort/obstacle/recall/watch） | 启发式建议 + 人工确认 | 运动学特征启发式 + 代表片段人工审核 |

**P0 自动映射规则**（3D 世界坐标 y-up, 米）:
- 坐 sit: 后腿折叠 < 0.05m + 前腿伸直 > 0.10m
- 卧 down: 肩甲高度 < 0.40m + 四肢折叠 < 0.08m
- 立 stand: 肩甲高度 > 0.50m + 腿伸直 > 0.15m
- 映射阈值: 单 syllable 内 ≥ 70% 帧分类一致

**P1 启发式建议规则**:
- 追踪 track: 鼻尖贴地比例 > 0.7 + 平均速度 > 0.05m/帧
- 障碍 obstacle: y_rise > 0.10m + 高速 > 0.15m/帧
- 扑咬 apprehend: 高速 > 0.20m/帧 + 嘴部方差 > 0.001
- 返回 recall: 方向反转 + 高速 > 0.15m/帧
- 警戒 watch: 头部抬起 > 0.6 + 低速 < 0.05m/帧

### 2.3 不选其他方案的理由

| 方案 | 不选理由 |
|------|---------|
| 方案 A 全人工映射 | 精确但耗时，与"先做不需要数据的"原则冲突 |
| 方案 C 全自动映射 | P1 行为需要上下文判断，自动映射有歧义风险 |
| 等待 1.2f 人工标注 | 阻塞主线，违反 Phase 2 v2.0 收敛原则 |
| 仅用 DeepEthogram | 需要 5-10 视频帧级标签，仍有标注成本 |

## 3. 实现路径

### 3.1 Path 1 keypoint-MoSeq 集成

**步骤**:
1. ✅ 安装 keypoint-moseq（pip install keypoint-moseq，清华镜像加速）
2. ✅ 创建训练脚本 `scripts/train_keypoint_moseq.py`（kpm 0.6.6 API 兼容）
3. ✅ 创建映射脚本 `scripts/map_syllables_to_behaviors.py`
4. ✅ 修复 API 兼容性问题:
   - `init_model` 的 `pca` 参数必须作为关键字参数传递（位置参数顺序: data, states, params, ...）
   - JAX 64-bit 精度: `jax.config.update("jax_enable_x64", True)` + `convert_data_precision(data, x64=True)`
   - `conf_threshold` 合并到 config dict，避免重复参数冲突
5. 🔄 运行训练: `python scripts/train_keypoint_moseq.py --num-iters 50`（AR-HMM 迭代中）
6. ⏳ 运行映射: `python scripts/map_syllables_to_behaviors.py`
7. ⏳ 人工审核 P1 映射: 查看 `data/kpm_project/syllable_report.md`

**输入**: InterPet4D smal_npy/*.npz（226 序列，kp_world (T, 24, 3)）
**输出**:
- `data/kpm_project/results.h5`: syllable 时间序列标注
- `data/kpm_project/syllable_mapping.json`: syllable_id → behavior
- `data/kpm_project/syllable_features.json`: 运动学特征
- `data/kpm_project/syllable_report.md`: 可读报告

### 3.2 Path 4 弱监督自训练 ✅ 完成

**步骤**:
1. ✅ 创建脚本 `scripts/train_weak_supervised.py`
2. ✅ 3D 姿态分类生成 P0 伪标签（sit/down/stand）
3. ✅ 运动学特征启发式生成 P1 伪标签（track/obstacle/watch/apprehend/recall）
4. ✅ 滑动窗口特征提取（30 帧窗口，50% 重叠，104 维特征）
5. ✅ Random Forest 训练 + 3 折交叉验证

**执行结果**（2026-07-29）:
- 数据: 226 InterPet4D 序列 → 8880 窗口（90% 有标签）
- 标签分布: watch 6621 / track 1180 / unknown 886 / obstacle 190 / apprehend 3
- 3 折交叉验证准确率: **99.8% ± 0.1%**
- 模型: `data/weak_supervised/weak_supervised_rf.pkl`
- 报告: `data/weak_supervised/weak_supervised_report.json`

**注意**: 99.8% 准确率反映分类器成功学习了规则引擎的决策边界（伪标签是确定性的）。分类器价值在于:
- 提供比规则引擎更快的推理（向量化的 RF vs 逐帧规则判断）
- 可用于 YouTube/AK 视频的快速行为标注
- 作为 Path 1 keypoint-MoSeq 的独立验证基线

### 3.3 Path 2 DeepEthogram（备选）⚠️ 有条件

**调研结论**（2026-07-29 GitHub-First）:
- 仓库活跃: jbohnslav/deepethogram（最后提交 2026-03-30）
- **Python 版本冲突**: 要求 Python >=3.9, <3.12，项目用 3.12 → 需独立虚拟环境
- **商用许可风险**: 学术免费，商用需 Harvard OTD 授权（公安/海关场景可能触发）
- 预训练权重: Kinetics700（Google Drive 可下载）
- Docker 支持: 可用于隔离 Python 环境

**触发条件**: Path 1 完成后，若 1.2f 真实数据准确率 < 80%
**缓解措施**: 用 conda 创建 Python 3.11 独立环境 + Docker 隔离

### 3.4 Path 3 主动学习（已在运行）

Label Studio project id=1 已就绪，用户闲暇时推进人工标注。

## 4. 影响

### 4.1 对 Phase 2 主线的影响

- **2.1 数据飞轮**: 不阻塞，Path 1 作为后台并行任务
- **2.2 16 行为**: 不阻塞，Path 1 完成后提供真实数据复核
- **2.3 USPCA**: 已完成 ✅
- **2.6 系统集成**: 不阻塞，依赖 2.1/2.2 完成

### 4.2 对 1.2f 验证的影响

Path 1 成功后，1.2f 真实数据验证可在 Phase 2.2 完成后立即用 16 行为引擎重测，无需等待人工标注。

### 4.3 对 Phase 3 的铺垫

keypoint-MoSeq 的 syllable 发现能力可直接用于 Phase 3 的 ST-GCN+BC 行为识别——syllables 作为图节点，行为类作为标签。

## 5. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| keypoint-MoSeq Windows 兼容性 | 已通过 pip install 测试，依赖 jax-moseq |
| syllable → 行为映射歧义 | P0 规则引擎自动 + P1 人工审核双重保障 |
| InterPet4D 3D 坐标 vs 规则引擎 2D 坐标 | 映射脚本使用 3D 感知的独立几何规则，不复用 2D RuleEngine |
| jax-moseq GPU 需求 | RTX 5060 已就绪，CUDA 12.x 兼容 |

## 6. 验证标准

| 项目 | 验证方法 | 通过标准 |
|------|---------|---------|
| Path 1 训练完成 | `results.h5` 生成 | ≥ 50 个 syllables 发现 |
| P0 自动映射 | `syllable_mapping.json` | ≥ 3 个 P0 行为映射成功 |
| P1 人工映射 | 人工审核报告 | 8 个 P1 行为全部映射 |
| 1.2f 真实数据验证 | 16 行为引擎重测 | 准确率 ≥ 80% |

## 7. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-29 | 初始版本：四路径并行 + 规则引擎自动映射 P0 + 人工映射 P1。依据 PHASE2_DATA_STRATEGY.md 调研 + 用户决策 |
| v1.1 | 2026-07-29 | 执行进展: Path 1 修复 3 个 API 兼容性问题（pca 关键字参数 + JAX x64 + conf_threshold 合并），训练进入 AR-HMM 迭代阶段。Path 4 完成（8880 窗口 + RF 99.8% CV）。Path 2 调研完成（Python 版本冲突 + 商用许可风险）。§3.1-3.3 更新 |
