# Phase 2 数据问题调研报告：工作犬行为识别数据策略

> 调研日期: 2026-07-29
> 调研方法: GitHub-First（AGENTS.md §1.3 强制规则）
> Owner: ML 开发
> 状态: 调研完成，待用户决策

## 1. 问题陈述

### 1.1 当前数据资产

| 数据源 | 规模 | 标签状态 | 可用性 |
|--------|------|---------|--------|
| InterPet4D kp_world | 226 clips / 13 dogs | 无行为标签，有 3D 关键点 (T, 24, 3) | ✅ 已验证 |
| YouTube 玩球视频 | 5 视频 | YOLO26-pose 预标注（待人工修正） | ✅ 已上传 Label Studio |
| Animal Kingdom 犬类视频 | 211 视频 / 111.2 MB | 无行为标签（25 种动物类别） | ⚠️ 跨物种，非犬类专属 |
| Dog-Pose 图像 | 6773 张 | 姿态标签（非行为） | ✅ 姿态训练用 |
| 合成数据 | 16 行为 × N 样本 | 完美标签 | ✅ 规则引擎验证用 |

### 1.2 核心痛点

1. **1.2f 真实视频准确率验证**：需要 ≥ 80% 准确率的 16 行为真实视频标签
2. **Phase 2.2 真实数据复核**：合成 97.5% 已通过，但真实视频未验证
3. **数据飞轮冷启动**：Label Studio 已部署，但人工标注成本高、速度慢

### 1.3 错误诊断

**项目并非被数据卡住，而是被"数据必须是人工标注的真实视频"这一假设卡住。**

现有 226 InterPet4D 关键点序列 + 5 YouTube 视频 + 211 AK 犬类视频 = **442 个未标注视频/序列**，若用无监督/弱监督方法提取行为标签，足以启动 Phase 2 主线。

---

## 2. GitHub-First 调研结果

### 2.1 已验证仓库

| 仓库 | 链接 | 活跃度 | 关键特性 | 许可证 |
|------|------|--------|---------|--------|
| **keypoint-MoSeq** | [dattalab/keypoint-moseq](https://github.com/dattalab/keypoint-moseq) | ✅ 2026-05-05, 903 commits, 44 releases, 28 issues | **无监督行为发现**，从关键点序列自动发现刻板运动模式（syllables），Nature Methods 2024 | 学术免费/商用联系 Harvard |
| **DeepEthogram** | [jbohnslav/deepethogram](https://github.com/jbohnslav/deepethogram) | ✅ 2026-03-30, 437 commits, 58 issues | **帧级行为分类**，Kinetics700 预训练，有标注数据集 + GUI | 学术免费/商用联系 Harvard |
| **awesome-action-recognition** | [jinwchoi/awesome-action-recognition](https://github.com/jinwchoi/awesome-action-recognition) | ⚠️ 2023-05-14, 296 commits | 动作识别资源目录（MMAction2/PySlowFast/3D-ResNets） | MIT |

### 2.2 keypoint-MoSeq 深度分析（推荐主路径）

**论文**: Weinreb et al., "Keypoint-MoSeq: parsing behavior by linking postural dynamics to neural activity", Nature Methods, 2024

**核心价值**:
- **无需标签**：从关键点序列无监督发现刻板运动模式
- **输入兼容**：接受任意关键点格式 (T, n_keypoints, n_dims)，我们的 Dog-Pose 24 关键点 (T, 24, 3) 直接可用
- **输出**：行为 syllables（刻板运动单元）+ 时间序列标注
- **活跃维护**：2026 年 5 月仍更新，有 Slack 社区

**工作流**:
```
YOLO26-pose → 24 关键点序列 (T, 24, 3)
    ↓
keypoint-MoSeq 无监督训练
    ↓
发现 N 个 syllables（如 50-100 个刻板运动模式）
    ↓
人工映射 syllables → 16 行为类（sit/down/stand/...）
    ↓
自动标注全部 226 InterPet4D + 5 YouTube 序列
```

**映射策略**:
- 每个 syllable 可视化代表性片段
- 通过几何规则（坐/卧/立的姿态判定）自动映射 P0 行为
- 通过运动模式（追踪/扑咬/障碍的时序特征）半自动映射 P1 行为
- 映射完成后，全部序列获得行为标签 → 可用于训练/验证

### 2.3 DeepEthogram 深度分析（备选路径）

**核心价值**:
- **Kinetics700 预训练**：迁移学习，小数据集也能收敛
- **帧级分类**：每帧输出行为概率，与我们的评分引擎（延迟/保持维度）天然兼容
- **有标注数据**：作者发布了标注数据集（Dropbox），含多种动物行为
- **GUI 标注工具**：内置视频标注界面

**适用场景**:
- keypoint-MoSeq 无监督发现不够精确时
- 需要视频帧级行为概率（而非关键点级）时
- 迁移学习到犬类视频

### 2.4 awesome-action-recognition 资源索引

| 工具 | 用途 | Phase 适用性 |
|------|------|-------------|
| MMAction2 | 通用动作识别框架 | Phase 3（ST-GCN+BC） |
| PySlowFast | 视频识别（SlowFast） | Phase 3 |
| 3D-ResNets-PyTorch | 3D CNN 视频分类 | Phase 3 |
| Kinetics700 | 大规模预训练数据 | 迁移学习特征提取器 |

---

## 3. 数据解决方案：四路径并行策略

### Path 1: 无监督行为发现（keypoint-MoSeq）— 主路径 ⭐

**目标**: 从现有 226 InterPet4D + 5 YouTube 关键点序列无监督发现行为标签

**步骤**:
1. 将 YOLO26-pose 输出转换为 keypoint-MoSeq 格式 (T, 24, 3)
2. 在 InterPet4D kp_world（226 序列）上训练 keypoint-MoSeq 模型
3. 发现 50-100 个 syllables
4. 用规则引擎几何判定自动映射 P0 8 行为（sit/down/stand/heel/stay/sit_up/bark/bite）
5. 人工映射 P1 8 行为（track/alert_sit/alert_down/apprehend/escort/obstacle/recall/watch）
6. 输出：226 序列 × 16 行为标签

**成本**: 0 标注成本（无监督）+ 1-2 天工程集成
**风险**: syllable → 行为类的映射可能有歧义（通过规则引擎 + 人工审核缓解）
**产出**: 226 个带标签序列 → 1.2f 真实数据验证 + Phase 2.2 真实数据复核

### Path 2: 迁移学习（DeepEthogram + Kinetics700）— 备选路径

**目标**: 用 Kinetics700 预训练特征提取器 + 少量犬类数据微调

**步骤**:
1. 下载 DeepEthogram Kinetics700 预训练权重
2. 用 5 YouTube 视频 + 211 AK 犬类视频微调
3. 帧级行为分类 → 输出每帧 16 行为概率
4. 与 keypoint-MoSeq 结果交叉验证

**成本**: 需少量标注（5-10 视频帧级标签）+ 2-3 天工程
**风险**: Kinetics700 是人类动作数据集，跨域迁移效果待验证
**产出**: 帧级行为概率 → 评分引擎延迟/保持维度信号

### Path 3: 主动学习循环（Label Studio + Rule Engine）— 持续优化

**目标**: 已部署的 Label Studio + 规则引擎预标注 → 人工修正 → 模型改进

**步骤**:
1. 规则引擎对新视频预标注（已有 63.1% 检测率）
2. Label Studio 人工修正（闲暇时推进，不阻塞主线）
3. 标注数据回流 → 微调 YOLO26-pose + 改进规则引擎阈值
4. 迭代循环

**成本**: 人工标注成本（按需，不阻塞）+ 0 额外工程（已部署）
**风险**: 人工标注速度慢（已在 Phase 2.0 后台并行）
**产出**: 持续增长的高质量标注数据集

### Path 4: 弱监督自训练（Rule Engine + Self-Training）— 数据增强

**目标**: 用规则引擎作为弱标注器，在无标注视频上生成伪标签，训练 ML 模型

**步骤**:
1. 规则引擎在 211 AK 犬类视频上生成伪标签
2. 过滤低置信度样本
3. 训练轻量级行为分类器（如 1D-CNN on 关键点时序）
4. 用分类器反向校准规则引擎阈值

**成本**: 1-2 天工程
**风险**: 伪标签噪声（通过置信度过滤缓解）
**产出**: 扩大训练集 + 规则引擎阈值校准

---

## 4. 推荐执行优先级

| 优先级 | 路径 | 时机 | 产出 | 阻塞主线？ |
|--------|------|------|------|-----------|
| **P0** | Path 1 (keypoint-MoSeq) | Phase 2.1 立即 | 226 序列行为标签 | 不阻塞（后台并行） |
| **P1** | Path 3 (主动学习) | 已在运行 | 持续高质量标签 | 不阻塞（后台并行） |
| **P2** | Path 4 (弱监督) | Path 1 完成后 | 扩大训练集 | 不阻塞 |
| **P3** | Path 2 (DeepEthogram) | 若 Path 1 不足 | 帧级概率 | 不阻塞 |

**核心洞察**: Path 1 解决 1.2f 真实数据验证问题——226 InterPet4D 序列通过 keypoint-MoSeq 获得行为标签后，可直接用于 16 行为准确率验证，无需等待人工标注。

---

## 5. 与 Phase 2 计划的关系

### 5.1 不阻塞主线

按 phase-2.md v2.1，2.1/2.2/2.3 主线均不依赖 1.2f 真实视频准确率。Path 1 作为**后台并行任务**推进，与 2.0 数据补强同优先级。

### 5.2 加速 1.2f 验证

若 Path 1 成功，1.2f 真实数据验证可在 Phase 2.2 完成后立即用 16 行为引擎重测，无需等待人工标注完成。

### 5.3 Phase 3 铺路

keypoint-MoSeq 的 syllable 发现能力可直接用于 Phase 3 的 ST-GCN+BC 行为识别——syllables 作为图节点，行为类作为标签，形成端到端的图神经网络行为识别管线。

---

## 6. 待决策项

1. **是否启动 Path 1（keypoint-MoSeq 集成）？**
   - 优势：0 标注成本，解决 1.2f 数据问题
   - 成本：1-2 天工程集成 + 许可证合规（学术免费，商用待 Phase 5 解决）

2. **keypoint-MoSeq syllable → 16 行为映射策略？**
   - 方案 A：全人工映射（精确，但耗时）
   - 方案 B：规则引擎自动映射 P0 + 人工映射 P1（推荐，平衡精度与效率）
   - 方案 C：全自动映射（快速，但有歧义风险）

3. **DeepEthogram 预训练权重是否现在下载？**
   - 文件大小：约 200-500 MB（Kinetics700 预训练模型）
   - 用途：Path 2 备选路径 + Phase 3 迁移学习
   - 建议：先跑 Path 1，根据结果决定是否需要 Path 2

---

## 7. 参考链接

- keypoint-MoSeq: https://github.com/dattalab/keypoint-moseq
- keypoint-MoSeq 论文: https://www.nature.com/articles/s41592-024-02318-2
- keypoint-MoSeq 文档: https://keypoint-moseq.readthedocs.io/
- DeepEthogram: https://github.com/jbohnslav/deepethogram
- awesome-action-recognition: https://github.com/jinwchoi/awesome-action-recognition
- Phase 2 计划: dev-docs/stages/phase-2.md v2.1
- 1.2f 补强 ADR: dev-docs/decisions/0008-1.2f-supplement-plan.md v1.5
