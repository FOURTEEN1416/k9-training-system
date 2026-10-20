# P1 合成生产线路 — 实施与评测报告（2026-10-20）

> 结论先行：**上游四个论文项目没有一个能在本机形成可训练/可复现基线**；本轮交付的是「类条件 + 骨长约束 + 词法文本条件」的合成生产线路、唯一可下载的真实犬文本-运动数据归档（OmniMotionGPT dog 子集），以及一套可复验的诚实评测。所有数字属**合成层自证**，不得与公开真实层/真实 K9 指标混排。

---

## 1. 可获得性实测

| 目标 | 可得 | 实测证据 | 处置 |
|------|------|----------|------|
| AniMo (CVPR2025) | code only | git clone 成功（60 files, 736KB）；utils/paramUtil.py 实测 30 行 offsets；LICENSE 存在=False；仓库内 checkpoint 文件=False | 仅作设计参考（359 维布局 / RVQ→base→residual 结构），不训练 |
| X-MoGen (AAAI2026) | no | 侦察实测：code/xmogen/README.md = 0 字节；仓库 13 条文件、无权重、无 LICENSE | 零投入 |
| OmniMotionGPT (CVPR2024) | code + data (MIT) | codeload 81,414,958 B 解包成功；LICENSE 首行='MIT License'；dog 子集 65 clip / 192 caption / 2624 帧实测可读；ylib 存在=False；checkpoint=[] | 数据归档为 omni_dog/；不训练（无 ckpt + 缺 ylib + 硬编码 23*6） |
| Kirin / AiM3D | partial (action only) | 侦察实测：代码未发布；dog.zip 24.5GB 中 npy 仅 ~123MB 压缩，但全部 caption 入口 404 | 本轮不下载（磁盘预算与文本缺口），仅记录为未来 Range 抽取候选 |
| syn_v2 (psd-framework) | yes (in-repo) | 本轮实测：fit_from_reference 返回 keys 无类条件字段；同 seed 下 track/jump 输出 max|Δ|=0.0 | 作为对照基线；本线路单独实现类条件生成器，不修改对方仓库 |

## 2. 本轮实现

- `scripts/eval_synth_motion_line.py`：拟合/生成/评测一体（本文件）。
- 生成器：**逐类** PCA（解释 ≥90% 方差，上限 8 维）建模「相对质心姿态」；clip 级静态偏移与 clip 内时序动态分开建模；时序动态 = 逐维 AR(1) + **真实残差行的块 bootstrap 新息**（block=5，保留跨维相关/重尾/短程自相关）；14 根骨长逐帧投影；逐类尺度抖动；逐关节经验置信度池。
- 文本条件：**词法映射**（模板关键词 → 类），不是 CLIP/学习到的文本编码器。全部 280 条 prompt 均通过自身词法映射回环校验。
- 基线：psd-framework `synth_stgcn_v2`（只读调用，同域、同训练集、同 seed）。
- 真实参考：`full12v2_T30.pkl`（352 clip / 30 帧 / 24 关节；视频级 train/val 无重叠，本轮实测）。
- 归一化口径：与 `backend.ml.behavior.stgcn_bc.dataset.STGCNBCDataset._normalize` 逐元素一致（最大差 0.00e+00）。

## 3. 结果

### 3.1 类区域命中率（最近类质心；非生成质量指标）

| 方法 | 命中率 | n | 随机 |
|------|--------|---|------|
| 真实留出验证集（描述子上限） | 0.1915 | 94 | 0.1429 |
| 本线路 k9_synth_v3 | 0.5786 | 280 | 0.1429 |
| syn_v2 基线 | 0.1429 | 280 | 0.1429 |

> 生成样本按构造拟合训练类统计，命中率高是预期；它只证明"类条件生效"，不证明真实感。

### 3.2 保真度 A（主口径：2 折交叉拟合，验证集不参与）

| 指标 | 本线路 | 同折真实对半（分辨率下限） |
|------|--------|---------------------------|
| 速度 KS 均值 | 0.2768 | 0.1692 |
| 速度直方图 L1 | 0.5171 | 0.2792 |
| 速度均值比（生成/真实） | 1.3359x | 0.7769x |
| 姿态均值相对误差（质心归一） | 3.7717 | 1.9495 |
| 骨长 CV（生成/真实） | 0.4549 / 0.6115 | 0.6050 / 0.6105 |

> 拟合折 A → 合成 → 对留出折 B 算指标，交换再算一次，逐类二次平均。这是**主口径**：生成侧与被比较侧都限定在同一训练分布内。同折真实对半 = 同一切分下把 A 折再对半，代表真实数据在同样本数下的可分辨率。

### 3.2b 保真度 B（次口径：生成 vs 留出真实验证集）

| 指标 | 本线路 | syn_v2 | 真实折半参照 |
|------|--------|--------|--------------|
| 速度 KS 均值 | 0.3677 | 0.5144 | 0.1313 |
| 速度直方图 L1 | 0.6887 | 1.1263 | 0.2641 |
| 速度均值比（生成/真实） | 1.3362x | 1.7279x | 0.9745x |
| 姿态均值相对误差（质心归一） | 4.6077 | 3.9210 | 1.2851 |
| 骨长 CV（生成侧） | 0.4762 | 0.5326 | 0.6010 |

> 次口径存在**已知混杂**：验证集与训练集的速度水平本身就不同（实测 track 训练 0.166 vs 验证 0.083；片段中位长度 52 帧被均匀抽成 30 帧，步长 ~1.7）。本线路在该口径 0.37 已明显低于 syn_v2 的 0.51，但仍远高于同量纲参照 0.13。此处只用于与 syn_v2 的相对比较，不单独作为质量证据。

### 3.3 逐类保真度（本线路）

| 类 | n_gen | n_real | 速度 KS | 速度 L1 | 速度比 | 姿态误差 | 骨长 CV |
|----|-------|--------|---------|---------|--------|----------|---------|
| stay | 40 | 9 | 0.4230 | 0.5240 | 1.11× | 2.0919 | 0.4428 |
| bite | 40 | 10 | 0.3643 | 0.7032 | 0.49× | 7.8844 | 0.4739 |
| track | 40 | 33 | 0.4195 | 0.8098 | 2.66× | 3.5530 | 0.4928 |
| apprehend | 40 | 6 | 0.4687 | 0.9500 | 1.56× | 6.8696 | 0.4884 |
| watch | 40 | 21 | 0.4095 | 0.7336 | 1.44× | 0.9288 | 0.5091 |
| retrieve | 40 | 4 | 0.2174 | 0.6237 | 0.68× | 5.9581 | 0.4053 |
| jump | 40 | 11 | 0.2712 | 0.4768 | 1.41× | 4.9684 | 0.5210 |

### 3.4 文本词法

- 生成 prompt：280 条，回环一致 280 条。
- OmniMotionGPT dog 真实人工描述 192 条：词表命中 39 条，未覆盖（超出 PSD 7 类动作空间）153 条。
- 按类命中：{'stay': 9, 'bite': 0, 'track': 0, 'apprehend': 11, 'watch': 0, 'retrieve': 0, 'jump': 19}

## 4. 产物

- `D:\Desktop\PR论文\datasets\synth_motion_samples\k9_synth_v3\generator_params.npz` — 1103416 B, sha256 `c2abdae12b5e17fd…`
- `D:\Desktop\PR论文\datasets\synth_motion_samples\k9_synth_v3\manifest.json` — 2266 B, sha256 `e25bb86d0249584b…`
- `D:\Desktop\PR论文\datasets\synth_motion_samples\k9_synth_v3\samples.npz` — 1774129 B, sha256 `70e8725407ce6773…`
- `D:\Desktop\PR论文\datasets\synth_motion_samples\k9_synth_v3\text_lexicon.json` — 2465 B, sha256 `832cd26789b82f26…`
- `D:\Desktop\PR论文\datasets\synth_motion_samples\k9_synth_v2_baseline\manifest.json` — 374 B, sha256 `7a2f542243b3cb63…`
- `D:\Desktop\PR论文\datasets\synth_motion_samples\k9_synth_v2_baseline\samples.npz` — 1806023 B, sha256 `5edc5a7d81d65ac0…`
- `D:\Desktop\PR论文\datasets\synth_motion_samples\omni_dog\omni_dog_captions.json` — 13325 B, sha256 `384ebf7c8a147c2b…`
- `D:\Desktop\PR论文\datasets\synth_motion_samples\omni_dog\omni_dog_manifest.json` — 1117 B, sha256 `6262b83d33086144…`
- `D:\Desktop\PR论文\datasets\synth_motion_samples\omni_dog\omni_dog_motions.npz` — 2930342 B, sha256 `fbefdd310935dbba…`

## 5. 限制与未完成（如实）

- 文本条件是词法映射，不是学习到的文本-运动对齐；不能声称具备文本生成能力。
- 生成器是类条件 PCA + 经验增量 bootstrap 的统计模型（无神经网络、无训练损失），不是扩散/Transformer，也不是对 AniMo/X-MoGen/Omni 的复现。
- 仅 7 个参考类可拟合；sit 只有 2 条且全在 val（训练 0 条），已排除。
- 所有指标为合成层自证：真实留出集仅 96 条 clip、7 类，最近质心命中率的置信区间很宽。
- 骨长逐帧投影保证骨骼刚性，但也会把真实检测抖动一并规整；骨长 CV 与真实仍有差距（见 3.2 表）。
- OmniMotionGPT 无法形成可训练基线：无预训练权重、无 ylib、motion_clip.py 硬编码 23*6；本轮只归档数据。
- AniMo 无 LICENSE、无权重、无 AniMo4D 数据，且 BaseTransformer 存在 _num_tokens NameError；未训练。
- Kirin 未下载：dog.zip 24.5GB（npy-only 需 Range 抽取）且 captions 全部 404；本轮未投入。
- X-MoGen 为空壳仓库，零可用产物。
- 生成器新息直接取自真实训练残差行（block bootstrap），因此生成样本在动态统计上是训练片段的「重组」，无法外推训练集未出现的运动模式；这不是可泛化的生成模型。
- 主口径速度均值仍为真实的 1.34 倍、姿态误差为真实折半下限的 1.9 倍——即本线路尚未达到"与真实数据不可区分"的保真度，只能作为受控合成层使用。
- 脚本依赖本机路径 D:/Desktop/PR论文/psd-framework（参考集与 syn_v2 基线）与 external/omnimotiongpt（Omni dog 数据）；缺任一项时脚本会报错而非静默降级。
- 白名单偏差：OmniMotionGPT 解包目录为 external/omnimotiongpt/（80MB，已被 .gitignore 的 external/ 规则覆盖），非白名单列出的 external/animo/；因 Omni 数据是唯一可用真实犬文本-运动来源且 omni_dog 归档需要可复现地重建，故保留，特此声明。

## 5b. AniMo 359 维字段表（仅作设计参考，未使用其数据）

实测 `external/animo/utils/paramUtil.py` 的 `animo_raw_offsets` 为 30 行 → 30 关节；`train_vq.py:58-59` 设 `opt.joints_num=30, dim_pose=359`。该 359 维是 HumanML3D 风格向量向 30 关节的扩展：

| 字段 | 宽度 | 计算 |
|------|------|------|
| 根旋转速度 | 1 | 固定 |
| 根线速度 | 2 | xy |
| 根高 | 1 | y |
| RIC（相对初始姿态） | 87 | (J-1)×3 |
| 6D 旋转 | 174 | (J-1)×6 |
| 局部速度 | 90 | J×3 |
| foot contact | 4 | 固定 |
| **合计** | **359** | 4+29×3+29×6+30×3+4 |

这是**重定向后的骨架特征向量**，不是 SMPL/SMAL 参数直出；与我们 24 关节 COCO 域不可直接拼接。本线路未采用该编码（生成物保持 `(30,24,3)` 的 backend 原生域）。

## 6. 复现

```bash
"D:/Desktop/PR论文/k9-training-system/.venv/Scripts/python.exe" scripts/eval_synth_motion_line.py
"D:/Desktop/PR论文/k9-training-system/.venv/Scripts/python.exe" scripts/verify_synth_line.py
```
