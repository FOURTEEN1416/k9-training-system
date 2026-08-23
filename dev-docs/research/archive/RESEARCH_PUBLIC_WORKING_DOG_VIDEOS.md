# 调研：公开工作犬视频资源

> Owner: 立项阶段调研
> 日期: 2026-07-27
> 依据: 用户确认"拿不到基地视频，用公开数据"
> 状态: ✅ 完成

## 0. 调研目标

回答三个问题：
1. 哪些公开视频/数据集可用于工作犬行为识别训练与评估？
2. 这些资源的标注级别（raw 视频 / 关键点 / 行为标签）如何？
3. 3-4 周内如何利用这些资源做模型训练 + 评估？

## 1. 学术数据集（带标注）

### 1.1 DogMo（2025 北理工，顶级资源）

来源：[selectdataset.com/DogMo](https://www.selectdataset.com/dataset/8b639bf7008b000e8c2e248f9ca2ddc6)、[arXiv:2510.24117](https://arxiv.org/abs/2510.24117)

- **规模**：10 只犬 × 1200 序列 × 220+ 分钟
- **采集**：5 台 ORBBEC Astra 2 RGB-D 相机环形 2.5m 半径
- **模态**：RGB 1920×1080 + 深度 1600×1200 + 掩码 + 关键点
- **动作类别**（11 类）：
  - Sit（坐）
  - Stand Up（站立）
  - Get On Chair（上椅子）
  - Play With Toy（玩玩具）⭐ 选育相关
  - Play With Human（与人互动）⭐ 选育相关
  - 等共 11 类
- **基准**：4 种运动恢复设定（单目/多视角 × RGB/RGB-D）
- **可用性**：✅ 公开下载，3-4 周内可用

**对本项目价值**：
- "玩玩具"和"与人互动"类目直接对应选育场景的玩具欲望测试
- 24+ 关键点标注可微调 YOLO26-pose
- 多视角数据可做姿态鲁棒性训练

### 1.2 InterPet4D（2026 arxiv，人犬互动 4D）

来源：[arXiv:2607.10287](https://arxiv.org/abs/2607.10287v1)

- **规模**：13 只犬（11 品种）× 23 人 × 6.8M 帧
- **采集**：同步多视角 + egocentric（犬载第一视角）
- **标注**：多视角视频 + 分割 + 2D/3D 关键点 + mesh + 音频
- **任务**：人犬服从任务（obedience tasks）⭐ 科目测评相关
- **基准**：InterPetMoGen 框架（FID 11.21）

**对本项目价值**：
- 服从任务直接对应科目测评（坐/卧/立/前来）
- 3D 关键点可做姿态精化
- 第一视角数据可做训练场景泛化

### 1.3 Working dogs in dynamic on-duty environments（Figshare 2022）

来源：[selectdataset.com](https://www.selectdataset.com/dataset/c99ce31bd6463936bcf24b0e85693c2c/working-dogs-in-dynamic-on-duty-environments)、[Figshare 21717776](https://figshare.com/articles/dataset/Working_dogs_in_dynamic_on-duty_environments/21717776)

- **数据**：工作犬在光声干扰下的任务延迟/持续时间/准确率
- **场景**：3 种干扰（光/声/光声组合）下的训练
- **可用性**：✅ Figshare 公开

**对本项目价值**：
- 直接对应"抗干扰"测试（TEST2）
- 评分规则可参考（任务延迟 + 准确率）

### 1.4 Accelerometer + annotation（Figshare 2022）

来源：[selectdataset.com](https://www.selectdataset.com/dataset/8b6c3d8bea992e699795167514934006/Accelerometer%20and%20annotation%20d)

- **规模**：11 只受训援助犬 × 10 天 × 12.5Hz 加速度计
- **标注**：视频时间戳标注（sit/begup/spin 等指令动作）
- **可用性**：✅ 公开

**对本项目价值**：
- 可作科目测评的行为标签来源
- 加速度计可作多模态融合参考（Phase 2）

### 1.5 CNN-BiLSTM 犬类行为识别（浙理工 2026）

来源：[浙江理工大学学报](https://xuebao.zstu.edu.cn/oa/darticle.aspx?id=20260508&type=view)

- **方法**：CNN-BiLSTM + 鲁棒标准化 + PCA
- **行为类别**：9 种（含"玩耍"复杂行为，F1 提升 26%）
- **输入**：可穿戴传感器数据

**对本项目价值**：
- 行为类别设计可参考
- 可穿戴 vs 视觉的差异认知

## 2. 公开视频资源（无标注）

### 2.1 优酷"国外训犬视频"播单

来源：[list.youku.com/albumlist/show/id_26559038.html](https://list.youku.com/albumlist/show/id_26559038.html)

- **上传者**：小刀ceo（K9 训犬精英账号，5277 粉丝，播放 2403 万）
- **内容**：
  - 米奇左侧靠随行训练
  - KNPV 训练
  - Police K9 patrol dog grip training（扑咬）
  - 防卫挑逗
  - How to Train a dog to Lie Down
- **可用性**：⚠️ 需手动下载 + 自标注

### 2.2 YouTube KCSO K-9 Training

来源：[King County Sheriff's Office YouTube](https://youtu.be/E1WHnw9XGEM)

- **内容**：美国警察 K9 训练官方视频
- **可用性**：⚠️ YouTube 访问受限

### 2.3 dogsacademy.org K9 训练指南

来源：[dogsacademy.org/k9-police-dog-training/](https://dogsacademy.org/k9-police-dog-training/)

- **内容**：K9 训练步骤（基础/社交/吠叫/搜毒/追踪）
- **可用性**：✅ 文字 + 部分视频

## 3. 国内基地公开宣传视频

### 3.1 德州市公安局天衢新区分局

来源：[中国警察网](http://m.toutiao.com/group/7363486706357813812/)

- **内容**：警犬扑咬科目训练、马里努阿犬"二炮"训练
- **场景**：基础科目（坐/卧/立/前来/随行/坐立/站立）+ 障碍物

### 3.2 山东省公安厅警犬训练基地

来源：[齐鲁晚报](http://www.sd.xinhuanet.com/20240315/7ae30a1ed28d4fbab31b484088a39d2a/c.html)

- **内容**：2024 年第一批 36 头幼犬配发仪式 + 幼训
- **场景**：气味感知/声音/触摸测试 + 采食/捕猎/胆量/抗压

### 3.3 南宁铁路公安局警犬繁育训练基地

来源：[中国新闻网](https://m.yunnan.cn/system/2023/02/15/032468941.shtml)

- **内容**：114 头工作犬 + 37 头 2-3 月龄幼犬
- **场景**：幼训（食物/猎捕/搜索动力）+ 服从性训练

## 4. 资源对比与推荐

| 资源 | 标注 | 场景匹配 | 可用性 | 推荐用途 |
|------|------|---------|--------|---------|
| DogMo | ✅ 关键点+分割 | 玩玩具/与人互动 | ✅ 直接下载 | 主训练集 |
| InterPet4D | ✅ 3D 关键点+mesh | 服从任务 | ✅ 申请 | 微调+评估 |
| Working dogs dynamic | ⚠️ 任务指标 | 抗干扰 | ✅ Figshare | 评分参考 |
| 优酷播单 | ❌ 无 | 训练全场景 | ⚠️ 手动 | 测试集 |
| 国内基地视频 | ❌ 无 | 幼训+服从 | ⚠️ 手动 | 演示素材 |

## 5. 3-4 周数据策略

### 5.1 训练数据

**主训练集**：DogMo（1200 序列，11 类动作）
- 11 类动作中至少 6 类直接对应科目测评（坐/站/上椅子/玩玩具/与人互动 + 其他）
- 24+ 关键点标注可用于微调 YOLO26-pose 工作犬版

**辅助训练集**：InterPet4D（6.8M 帧）
- 申请访问权限（学术授权）
- 用于行为识别时序模型训练

### 5.2 评估数据

**内部评估**：
- DogMo 测试集划分
- InterPet4D 测试集

**外部评估**（无标注）：
- 优酷播单：下载 ≥3 段 K9 训练视频做定性评估
- 国内基地视频：下载 ≥3 段幼训视频做演示

### 5.3 选育场景数据

**问题**：DogMo/InterPet4D 都没有"食物欲望测试"标签

**方案**：
1. 从 DogMo "Play With Toy" 序列中提取行为信号（速度/距离/朝向）
2. 用规则引擎自动打伪标签（食物欲望高/中/低）
3. 人工抽检 50 段校准阈值
4. 形成 ≥200 段带伪标签的选育测试集

## 6. 与既有 truth 文档的差异

| 既有文档假设 | 调研结论 | 需更新 |
|------------|---------|--------|
| phase-1.md §1.0f 要求"用户提供 ≥3 段工作犬视频" | 用户拿不到，改用公开数据 | ✅ 需重写 |
| 数据采集方案依赖基地 | 可用 DogMo/InterPet4D 替代 | ✅ 需重写 |
| 未提及 DogMo/InterPet4D | 顶级资源可用 | ✅ 需补充 |

## 7. 引用

- [DogMo - selectdataset](https://www.selectdataset.com/dataset/8b639bf7008b000e8c2e248f9ca2ddc6)
- [DogMo arXiv:2510.24117](https://arxiv.org/abs/2510.24117)
- [InterPet4D arXiv:2607.10287](https://arxiv.org/abs/2607.10287v1)
- [Working dogs in dynamic on-duty environments - Figshare](https://figshare.com/articles/dataset/Working_dogs_in_dynamic_on-duty_environments/21717776)
- [Accelerometer + annotation data - Figshare](https://www.selectdataset.com/dataset/8b6c3d8bea992e699795167514934006/Accelerometer%20and%20annotation%20d)
- [CNN-BiLSTM 犬类行为识别 - 浙理工学报](https://xuebao.zstu.edu.cn/oa/darticle.aspx?id=20260508&type=view)
- [优酷 K9 训犬视频播单](https://list.youku.com/albumlist/show/id_26559038.html)
- [KCSO K-9 Training YouTube](https://youtu.be/E1WHnw9XGEM)
- [dogsacademy K9 训练指南](https://dogsacademy.org/k9-police-dog-training/)
