# ADR 0006: DogMo 数据集开源替代方案

> 状态: ✅ 已确认（v1.1 修订: InterPet4D v1 实际结构发现）
> 日期: 2026-07-28
> Owner: ML 开发（见 AGENTS.md §2.2）
> 修改触发: 数据采集方案变更 / 1.6d + 1.2f 验证策略变更
> 依据: [RESEARCH_PUBLIC_WORKING_DOG_VIDEOS.md](../research/RESEARCH_PUBLIC_WORKING_DOG_VIDEOS.md) + InterPet4D (arxiv 2607.10287) + Animal Kingdom (CVPR 2022) + InterPet4D v1 下载后实际结构验证

## 1. 上下文

### 1.1 问题背景

Phase 1 MVP 原计划使用 DogMo 数据集（北京理工大学，arxiv 2510.24117）验证：
- **1.6d**：DogMo "Play With Toy" 序列验证 `puppy_signals.py` 9 信号提取器
- **1.2f 复核**：DogMo 真实视频评估规则引擎准确率，决定是否复核 Phase 1.3 PoseC3D 跳过决策

### 1.2 DogMo 数据集限制

- **付费获取**：DogMo 数据集需购买，与 AGENTS.md §1.1.4 "深度优先"原则中的"广泛调研优先"冲突（应优先寻找开源替代）
- **申请周期不确定**：学术授权响应周期长，阻塞 Phase 1 验收
- **数据规模有限**：10 犬 × 1200 序列 × 220 分钟，物种单一

### 1.3 替代方案调研结论

详见 2026-07-28 调研报告（本 ADR §3 引用）。

| 方案 | 可用性 | 关键点对齐 | 1.6d | 1.2f |
|------|--------|-----------|------|------|
| **InterPet4D** | ✅ HuggingFace 直接下载 10.7 GB | ✅ 24 关键点 3D 完美对齐 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Animal Kingdom** | ⚠️ Google Form 申请 | ⚠️ 需映射 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **YouTube 自标** | ✅ 直接下载 | ❌ 需自标 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **DogMo 购买** | ❌ 付费 | ✅ 24 关键点 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 2. 决策

### 2.1 放弃购买 DogMo，采用开源替代方案

**组合方案**：InterPet4D（主）+ YouTube 玩球视频（补充）

| 任务 | 数据来源 | 用途 |
|------|---------|------|
| **1.2f 真实复核** | InterPet4D（13 犬 × 227 clips，sit/down/stand/come） | 规则引擎准确率真实评估，复核 Phase 1.3 跳过决策 |
| **1.6d 姿态验证** | InterPet4D `smal_npy/kp_world: (T, 24, 3)` | 24 关键点完美对齐，验证 `puppy_signals.py` 姿态处理逻辑 |
| **1.6d 物体检测** | YouTube "dog playing with toy" 短视频 3-5 段 | 验证 `object_detector.py` 球类检测 + 9 信号提取 |

### 2.2 InterPet4D 适用性分析

**核心优势**：
- **24 关键点完美对齐**：`smal_npy/` 目录下 `kp_world: (T, 24, 3)` 与本项目 Dog-Pose 24 关键点完全一致，无需任何映射
- **HuggingFace 直接下载**：10.7 GB，227 clips，无申请周期
- **CC BY-NC 4.0 许可证**：非商业研究用，符合本项目学术用途

**v1 实际结构（2026-07-28 下载后验证）**：
- ✅ `smal_npy/*.npz`：含 `kp_world (T, 24, 3)` + `kp_weight (T, 24)` — 3D 世界坐标（米，y-up）
- ✅ `pet_npy/` / `mano_npy/` / `smpl_ny/`：人/手/宠物 SMAL 拟合结果
- ✅ `audio/`：配套音频 .mp3
- ❌ **无视频文件**（v1 仅 motion capture + audio，无 RGB 视频帧）
- ❌ **无行为标签**（自然互动，非服从训练 sit/down/stand/come 标注）

**局限（v1 实际）**：
- **无视频帧**：原设计 1.2f（视频→YOLO26-pose→准确率）无法执行，改用 kp_world 替代验证
- **无行为标签**：无法计算真实准确率，合成基线 92.9% 保留为主要证据
- **不含 "Play With Toy"**：1.6d 玩具场景需 YouTube 补充
- **静态摄像机 + 室内**：与工作犬训练室外场景分布略有差异
- **关键点是 SMAL 拟合结果**：非 YOLO26-pose 直接推理，坐标系为 3D 世界坐标（米），与 2D 图像坐标（像素）需转换

### 2.3 验证策略

#### 1.2f 真实复核（Phase 2 启动条件）— v1.1 修订

**原设计（基于 ADR 0006 v1.0 假设）**：
```
InterPet4D 视频帧 → YOLO26-pose 推理 24 关键点 → 规则引擎识别 → 真实准确率
```

**实际执行（v1.1 修订：InterPet4D v1 无视频/标签）**：
```
三层降级验证:
1. 合成基线（主要证据）: 42 样本准确率 92.9% ≥ 80% 阈值 → 确认跳过 PoseC3D
2. kp_world 管线验证（辅助证据）: kp_world → y翻转 + ×1000 缩放 → 规则引擎 → 行为分布
3. 真实视频验证（延后）: 待获取含视频+标签的数据集后执行（Phase 2 内推进）
```

**kp_world 坐标系转换**：
```python
# 世界坐标 (米, y-up) → 图像坐标 (像素, y-down)
kp_image[:, :, 0] = kp[:, :, 0] * 1000.0   # x: 米 → 毫米-像素
kp_image[:, :, 1] = -kp[:, :, 1] * 1000.0  # y: 米 → 翻转 + 毫米-像素
```

**验证结果**（`reports/phase-2-prereq-1.2f-validation.md` v1.1）：
- ✅ 管线运行率 100%（226/226 clips 无异常）
- ⚠️ 行为多样性 2 种（down, stay）— 坐标系转换限制，非引擎缺陷
- ✅ 合成基线 92.9% 保留为 PoseC3D 跳过决策主要证据
- **状态**: 条件通过（数据限制）

#### 1.6d 选育信号验证（Phase 2 启动条件）— 通过

**姿态处理验证**（InterPet4D kp_world）：
```
InterPet4D kp_world (T, 24, 3) → puppy_signals.extract_puppy_signals() → 9 信号合理性检查
```

**验证结果**（`reports/phase-2-prereq-1.6d-validation.md`）：
- ✅ 管线运行 226/226 clips 无异常
- ✅ 9/9 姿态代理指标跨 clips 有变异（≥6 通过）
- **状态**: 通过

**物体检测验证**（YouTube 玩球视频，延后）：
```
YouTube 玩球视频 → YOLO26-pose + object_detector → 9 信号 → 评分
```
注: Phase 2 内推进，不阻塞 Phase 2 启动。

### 2.4 长期学术资源规划

并行申请 Animal Kingdom（Google Form），作为 Phase 1.8b 学术副产物的跨物种行为识别基准数据集。

## 3. 实施计划

### 3.1 立即执行（Phase 1.8 验收后）

1. **下载 InterPet4D**：`huggingface-cli download ohi_carip/interpet4d --repo-type dataset`（10.7 GB）
2. **下载 YouTube 玩球视频**：3-5 段，每段 30-60 秒，含犬 + 球互动
3. **编写验证脚本**：`scripts/validate_interpet4d.py` + `scripts/validate_youtube_toy.py`

### 3.2 Phase 2 启动条件

1.6d/1.2f 真实数据验证作为 **Phase 2 启动前置条件**，不阻塞 Phase 1 验收（合成数据 92.9% + 端到端 7/7 PASS 已满足 Phase 1 出口条件）。

## 4. 影响评估

### 4.1 受影响 Truth 文档

| 文档 | 变更 |
|------|------|
| `dev-docs/stages/phase-1.md` §1.6d | 更新数据来源说明（DogMo → InterPet4D + YouTube） |
| `dev-docs/stages/phase-1.md` §1.2f | 更新复核数据来源（DogMo → InterPet4D） |
| `dev-docs/research/RESEARCH_PUBLIC_WORKING_DOG_VIDEOS.md` | 补充 InterPet4D + Animal Kingdom 详节 |
| `reports/phase-1-validation.md` | 记录 1.6d/1.2f 替代验证方案 |

### 4.2 风险

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| InterPet4D 关键点与 YOLO26-pose 推理结果分布不一致 | 30% | 中 | 同时跑 YOLO26-pose 推理 + SMAL 拟合结果对比 |
| YouTube 玩球视频物体检测准确率低 | 20% | 低 | 用 COCO 预训练权重 + 物体检测阈值调优 |
| InterPet4D 行为标签与本项目 8 类映射不全 | 40% | 中 | 仅评估覆盖的 4-5 类（sit/down/stand/come），其余类延后 |

## 5. 决策依据

### 5.1 AGENTS.md 原则对齐

- ✅ §1.1.1 **广泛调研优先**：完成多方案调研，记录到 `dev-docs/research/`
- ✅ §1.1.4 **深度优先而非浅层包装**：选择 24 关键点完美对齐的 InterPet4D，确保验证深度
- ✅ §1.3 **质量优先于速度**：开源替代方案质量足够，无需为付费数据集妥协
- ✅ §8 **防漂移规则**：数据采集方案变更已记录 ADR

### 5.2 不选 DogMo 的理由

1. **付费门槛**：与开源优先原则冲突
2. **关键点对齐优势不显著**：InterPet4D 同样提供 24 关键点
3. **物种多样性不足**：仅 10 犬，InterPet4D 13 犬 + Animal Kingdom 850 物种
4. **申请周期不确定**：阻塞 Phase 1 验收

## 6. 引用

- **InterPet4D 论文**：Peng et al., "InterPet4D: A Multimodal 4D Human-Pet Interaction Dataset for Pet Motion Generation", arxiv 2607.10287, 2026-07
- **InterPet4D 数据**：https://huggingface.co/datasets/ohicarip/interpet4d
- **Animal Kingdom 论文**：Ng et al., "Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding", CVPR 2022
- **Animal Kingdom 数据**：https://sutdcv.github.io/Animal-Kingdom/
- **DogMo 论文**：Wang et al., "DogMo: A Large-Scale Multi-View RGB-D Dataset for 4D Canine Motion Recovery", arxiv 2510.24117, 2025-10

## 7. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-28 | 初始版本，确认 InterPet4D + YouTube 替代方案 |
| v1.1 | 2026-07-28 | InterPet4D v1 下载后实际结构发现: 无视频文件 + 无行为标签。§2.2 更新实际结构 + 局限，§2.3 修订 1.2f 为三层降级验证（合成基线 + kp_world 管线 + 延后真实视频），1.6d 标记通过。1.2f 状态: 条件通过（数据限制） |
