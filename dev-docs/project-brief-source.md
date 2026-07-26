# 工作犬训练机器视觉识别系统 — 产品需求与技术方案 v2.0

> 版本: v2.0 | 日期: 2026-07-01 | 状态: 调研完成（深度）
> 调研覆盖: 4轮多智能体协同调研 | 12+篇论文全文 | 15+开源仓库代码级分析 | 3大国际标准详细分解 | 8份专项研究报告

---

## 一、项目背景

### 1.1 问题定义

工作犬（搜爆、搜救、巡逻等方向）训练、幼犬选育和科目测评过程中，行为判断依赖人工观察和经验记录，存在以下痛点：

| 痛点 | 具体表现 | 量化数据 |
|------|---------|---------|
| 主观性强 | 同一动作不同训导员评分差异大，标准不统一 | 失败率>50%（全球工作犬数据），每犬训练成本>$12,000 |
| 复盘困难 | 人工看视频、手动记录时间点，效率低下 | 无自动化评估工具可用 |
| 数据难沉淀 | 标注数据难以形成闭环，模型训练不易管理 | 行业无公开工作犬行为数据集 |
| 人才瓶颈 | 专家依赖度高，工作犬培养难以规模化 | 警犬基地逐年萎缩 |

### 1.2 目标

构建一个**本地部署 + 机器视觉 + AI 行为识别 + 自动评分**的系统，实现：
- 视频输入 → 犬只姿态检测 → 动作行为识别 → 标准化评分报告 的端到端闭环
- 训练数据积累 → 模型迭代优化 的数据飞轮
- 兼容中国 GA/T、USPCA、FCI-IGP 三大国际标准

### 1.3 目标用户

- 公安警犬基地、海关缉私犬队、消防搜救犬队
- 工作犬训练机构、犬行为科研单位
- 工作犬训导员、犬只繁育选育人员

---

## 二、核心调研结论

### 2.1 调研总览

| 调研维度 | 源数量 | 方法 | 报告文档 |
|---------|--------|------|---------|
| 学术论文综述 | 7篇核心 + 12+篇相关 | 全文阅读 + 交叉验证 | RESEARCH_LITERATURE.md |
| 开源代码分析 | 15+仓库 | GitHub API + 代码深度阅读 | RESEARCH_OPENSOURCE.md |
| 技术选型对比 | 4子方向 | 深度对比矩阵 | RESEARCH_TECH_COMPARISON.md, RESEARCH_YOLO26_DEPLOY_DEEP.md |
| 实现细节 | 全栈 | 架构设计 + Schema + 代码 | RESEARCH_IMPLEMENTATION.md |
| 工作犬标准 | 3大国际标准 | GA/T全文分解 + USPCA/FCI-IGP | RESEARCH_STANDARDS.md, 工作犬标准自动化深度调研报告.md |
| 商业竞品 | 10+产品 | 全球扫描 + 差距分析 | RESEARCH_COMMERCIAL.md |
| BCST-GCN/ST-GCN | 论文 + 代码 | 架构重现分析 | BCST-GCN_ST-GCN_深度分析报告.md |

### 2.2 关键发现

#### 发现 1：已有成熟狗姿态数据集 + 完整微调管道

**Ultralytics Dog-Pose Dataset**（24 关键点）：
- 训练集 6,773 张，测试集 1,703 张
- 格式：YOLO Pose 格式，`(x, y, visibility)` 三元组
- 下载：`yolo train data=dog-pose.yaml` 自动拉取

**对工作犬场景的局限性**（已评估）:
- 缺少工作犬品种（马犬/昆明犬）：→ 迁移学习 + 200-500 张工作犬微调即可
- 缺少俯视/低光/运动模糊场景：→ 数据增强 + 合成数据(SyDog-Video)
- **不影响 Phase 1 MVP**：基础坐/卧/立/走已完全覆盖

#### 发现 2：学术界已验证方向可行性

| 论文 | 出处 | 年度 | 核心结论 | 对本系统价值 |
|------|------|------|---------|-------------|
| NC State (Science 2026) | Science | 2026 | IMU+AI 量化工作犬行为，已在导盲犬学校部署 10 年 | ✅ 方向可行性验证 |
| BCST-GCN (Front. Vet. Sci.) | Frontiers | 2026 | ST-GCN+双向注意力 → 94.43% 四足动物行为分类 | ✅ 最直接技术参考 |
| ASBAR (eLife) | eLife | 2024 | 首个动物骨架行为识别框架，开源代码 | ✅ **最可复用的代码基础** |
| SLEAP (Nature Methods) | Nature | 2022 | 多动物姿态追踪，被引 1000+ | ✅ 多犬场景参考 |
| PoseC3D (CVPR) | CVPR | 2022 | 3D热图堆叠行为分类，MMAction2原生 | ✅ **Phase 1 推荐架构** |

#### 发现 3：行为分类推荐路径更新

| 阶段 | 原方案 | **更新后方案** | 理由 |
|------|--------|---------------|------|
| Phase 1 | 规则引擎 | **规则引擎 + PoseC3D (ASBAR路径)** | PoseC3D 对数据量要求低(<500序列即可训练)，抗噪强 |
| Phase 2 | LSTM | **PoseC3D + ST-GCN 对比** | 数据量到1000+序列后，对比两者择优 |
| Phase 3 | ST-GCN | **ST-GCN + BC 模块** | 1500+序列时 ST-GCN 可解释性+精度更优 |

**关键洞察**: PoseC3D 用 3D heatmap 堆叠替代图序列，天然抗噪、泛化性强，对狗场景（关节抖动多、需跨品种泛化）反而比 ST-GCN 更适合 Phase 1-2。

#### 发现 4：YOLO26-pose 相比 YOLO11 的关键提升

| 特性 | YOLOv8/v11 | YOLO26 | 对狗姿态的意义 |
|------|-----------|--------|--------------|
| NMS | 需要后处理 | **端到端 NMS-Free** | 简化部署，降低延迟 |
| 后端 | 单头 | **双头设计** | 原生端到端推理 |
| CPU推理速度 | 基准 | **提升 43%** | 无 GPU 时也能用 |
| 人体骨架假设 | 有 | **移除** | 更适应狗骨架自定义关键点 |

#### 发现 5：工作犬评估标准已完整映射

| 标准 | 可自动化比例 | 核心可自动化科目 |
|------|------------|----------------|
| 中国 GA/T 系列 | **~55%** | 服从全部（坐/卧/立/随行/远距离指挥）, 搜索路径 70% |
| USPCA PDI 700分 | **~45%** | 服从 60%（120分）, 搜索 70%（140分） |
| FCI-IGP 300分 | **~50%** | 服从 70%（100分）, 追踪 40% |

**核心结论**: 服从科目自动化率 60-70% 可立即实现，搜索 50-60%，防卫 20-30%。

#### 发现 6：全球市场空白确认

**全球范围内不存在完整的全科目自动警犬评估系统。** 现有产品要么是消费级宠物监测（Traini/Fi/Whistle），要么是实验室动物研究工具（EthoVision $15k+/SLEAP），要么是纯管理 SaaS（DogBase）。

**本系统是全球首个从视频输入到标准化评分报告闭环的警犬AI评估系统。**

---

## 三、技术方案

### 3.1 系统架构图

```mermaid
flowchart TB
    subgraph 前端层["前端层 (Vue 3 + Naive UI)"]
        UPLOAD[视频上传页] --> PREVIEW[实时预览/Canvas标注]
        PREVIEW --> REPORT[评分报告页]
        REPORT --> HISTORY[训练历史/数据对比]
        HISTORY --> TRAIN[模型训练管理]
    end

    subgraph 后端层["后端层 (FastAPI + Celery)"]
        API[API 路由] --> ASYNC[异步任务队列 Celery+Redis]
        API --> REPO[报告生成/数据导出]
    end

    subgraph 推理层["AI 推理层"]
        POSE[YOLO26-pose<br/>姿态估计 (TensorRT)]
        POSE --> ACTION[行为分类<br/>PoseC3D/ST-GCN + BC]
        ACTION --> SCORE[评分引擎<br/>支持 GA-T / USPCA / FCI-IGP]
    end

    subgraph 存储层["存储层"]
        DB[(PostgreSQL 15<br/>+ pgvector)]
        FS[文件系统<br/>模型/视频/缓存]
    end

    前端层 <-->|REST API| 后端层
    后端层 --> 推理层
    推理层 --> 存储层
    后端层 --> 存储层
```

### 3.2 模块设计

#### 模块 A：视频采集与预处理

| 功能 | 技术 | 参数 | 说明 |
|------|------|------|------|
| 视频输入 | OpenCV | 支持 RTSP/本地文件/图片序列 | 摄像头推荐 USB 1080p 30fps |
| 帧提取 | OpenCV | 25-30fps 原始，可配置抽帧 | 行为分类 2-5fps 足够 |
| 目标检测 | YOLO26n (detect mode) | 定位犬只，输出 bounding box | 单犬场景，多犬 Phase 3 |
| 图像预处理 | Letterbox resize | 640×640 → TensorRT INT8 输入 | 低端设备可用 416×416 |
| 数据增强 | albumentations | 训练阶段 | 旋转/缩放/亮度/模糊 |

#### 模块 B：姿态估计（核心）

| 参数 | 值 | 理由 |
|------|----|------|
| 模型 | **YOLO26n-pose** (首选) / YOLO26m-pose (高精度) | NMS-Free, 双头, 43% CPU加速 |
| 关键点 | **24** (Dog-Pose 定义) | 覆盖训练全面部/四肢/躯干 |
| 预训练 | yolo26n-pose.pt → Dog-Pose 微调 → 工作域适应 | 三阶段迁移学习 |
| 推理后端 | **TensorRT FP16** (首选) / ONNX Runtime / OpenVINO | 2-3x加速, <0.5%精度损失 |
| 推理速度 | **60+ FPS** (Jetson Orin Nano Super, TensorRT) | 满足实时训练需求 |
| 极限速度 | **219 FPS** (YOLO26n + TensorRT INT8) | 多路同时推理可行 |
| 输出 | 24 × (x, y, confidence) | 连续帧组成时序序列 |

#### 模块 C：动作行为分类（三阶段递进）

| 阶段 | 方案 | 架构选择 | 数据需求 | 预期精度 | 交付时间 |
|------|------|---------|---------|---------|---------|
| **Phase 1** | 规则引擎 + **PoseC3D** (ASBAR路径) | MMAction2 PoseC3D + MobileNet 3D | 200-500时序 | 75-85% | 4-6 周 |
| **Phase 2** | PoseC3D vs ST-GCN 对比择优 | MMAction2 两方案同时训练 | 500-1500时序 | 80-88% | +4-6 周 |
| **Phase 3** | **ST-GCN + BC 模块** | BCST-GCN 架构复现 + 狗骨架自定义 | 1500+时序 | 88-94% | +8-12 周 |

**Phase 1 关键代码参考**:
```python
# ASBAR 路径：YOLO26-pose 24关键点 → PoseC3D
# 1. 用 YOLO26-pose 提取每帧 24 关键点
# 2. 编码为 3D 热图体积 (C=24, T=32, H=56, W=56)
# 3. 输入 PoseC3D (3D MobileNet) 分类行为
```

#### 模块 D：评分引擎

工作犬行为分类体系（**22 种动作，分三级**）:

**P0 基础姿态 (8种, Phase 1)**:
| 编码 | 动作 | 判定逻辑 | 所属科目 |
|------|------|---------|---------|
| B01 | 坐 Sit | 后膝角<90° + 前肢直立 | 服从 |
| B02 | 卧 Down | 肩隆Y—后爪Y < 身高30% | 服从 |
| B03 | 立 Stand | 四肢延展 + 肩隆基线高度 | 服从 |
| B04 | 行走 Walk | 四足交替运动时序 | 服从/搜索 |
| B05 | 小跑 Trot | 对角腿同步 + 速度模式 | 搜索 |
| B06 | 搜索 Search | 鼻尖低于肩部 + 持续移动 | 搜毒/搜爆 |
| B07 | 随行 Heel | 犬首维持在人腿侧±阈值 | 服从 |
| B08 | 吠叫 Bark | 鼻尖—下巴距离时序变化 | 服从/警戒 |

**P1 训练专项 (8种, Phase 2)**:
追踪/示警坐/示警卧/扑咬/押解/障碍穿越/返回/警戒

**P2 高级评估 (6种, Phase 3)**:
食物欲望/胆量测试/注意力测试/搜索效率/冲动控制/步态分析

**7 维评分体系**:

| 维度 | 权重 | 量化方法 | 可自动化 |
|------|------|---------|---------|
| 动作准确度 | 20% | cosine_sim(实测姿态, 标准姿态) | ✅ 高 |
| 响应延迟 | 15% | t_指令发出 → t_动作开始 | ✅ 高 |
| 保持时长 | 15% | 连续满足姿态条件帧数 | ✅ 高 |
| 搜索效率 | 20% | 检出数 / 搜索耗时 | ✅ 高 |
| 注意力集中度 | 10% | 头部方向与训导员的偏差角均值 | ✅ 中 |
| 胆量/欲望 | 10% | 接近速度曲线 / 最小接近距离 | ✅ 中 |
| 步态质量 | 10% | 步幅变异系数 | ✅ 高 |

**兼容标准映射**:
- 中国 GA/T: 100分制 → 自动映射（60及格/80良好/90优秀）
- USPCA PDI: 700分制 → 按扣分规则自动评分
- FCI-IGP: 三科各100分 → 分科评分汇聚

#### 模块 E：前端页面设计

| 页面 | 功能 | 核心组件 |
|------|------|---------|
| 视频上传/录制 | 文件上传 + 实时 RTSP 预览 | 上传组件、Canvas 播放器 |
| 姿态标注预览 | 实时 24 关键点覆盖层 + 连线 | Canvas 2D 渲染 |
| 行为分析 | 时间轴标注 + 动作概率曲线 | ECharts + 时间轴组件 |
| 评分报告 | 多标准兼容(GA-T/USPCA/FCI)雷达图+总分+PDF | ECharts + html2pdf |
| 训练历史 | 趋势对比 + 排行榜 | ECharts 折线图 |
| 模型管理 | 标注→训练→评估流程 | 模板 + 进度条 |

---

## 四、技术选型清单

### 4.1 最终技术栈

| 层 | 组件 | 版本 | 安装方式 | 备注 |
|----|------|------|---------|------|
| 姿态引擎 | ultralytics | ≥8.6 | pip install | YOLO26-pose, NMS-Free |
| 推理加速 | TensorRT / onnxruntime-gpu | ≥10.0 / ≥1.18 | NVIDIA SDK / pip | Jetson优先TensorRT |
| 行为分类 | mmaction2 + pytorch | ≥2.0 | pip install mmaction2 | PoseC3D / ST-GCN |
| 后端框架 | fastapi + uvicorn | ≥0.115 | pip | 异步原生 |
| 数据库 | PostgreSQL 15 + pgvector | 15+ | 本地安装 / Docker | 关系+向量混合 |
| ORM | sqlalchemy + asyncpg | ≥2.0 | pip | 异步 ORM |
| 任务队列 | celery + redis | ≥5.4 | pip | 异步推理 |
| 前端框架 | Vue 3 + Vite | ≥5 | bun create vue@latest | 中国生态 |
| UI 库 | Naive UI | ≥2.40 | bun add | 轻量中文 |
| 视频处理 | opencv-python + ffmpeg | ≥4.9 | pip / choco | 帧提取 |
| 可视化 | echarts | ≥5.5 | bun add | 雷达图/时间轴 |
| 行为分类 | PyTorch | ≥2.4 | pip install torch | GPU 训练/推理 |

### 4.2 关键决策理由（v2.0 更新）

| 决策点 | v1.0 | **v2.0 更新** | 理由 |
|--------|------|---------------|------|
| 姿态模型 | YOLO11-pose | **YOLO26-pose** | NMS-Free, 双头, 43% CPU加速, 移除人体假设 |
| 行为分类 | 规则→LSTM→ST-GCN | **规则→PoseC3D→ST-GCN+BC** | PoseC3D 抗噪强/泛化好, ASBAR 代码可直接复用 |
| 推理加速 | ONNX Runtime | **TensorRT FP16 (首选)** | 2-3x加速, Jetson 原生支持 |
| 边缘硬件 | — | **Jetson Orin Nano Super ($249, 67 TOPS)** | 60-80 FPS YOLO26n, 最佳性价比 |
| 评分兼容 | 7维度 | **7维度 + GA-T/USPCA/FCI-IGP 映射** | 工作犬场景实际需求 |

### 4.3 部署方案

**推荐方案**: Jetson Orin Nano Super ($249) + USB 摄像头 ($30-80)
- BOM 成本: ~$350-500
- 性能: 60+ FPS YOLO26n-pose TensorRT
- 支持 2-3 路同时推理

**Windows 降级方案**:
- GPU (RTX 3060+): 30+ FPS, TensorRT/ONNX Runtime GPU
- CPU 纯算: 5-15 FPS, ONNX Runtime CPU / OpenVINO

---

## 五、实施路线图

### Phase 1 — MVP（4-6 周, 验证最小闭环）

**目标**: 上传视频 → YOLO26-pose 24点 → 规则+PoseC3D 8类行为 → 评分报告

- [ ] 环境搭建（Python venv + ultralytics + mmaction2 + 模型下载）
- [ ] YOLO26-pose 姿态检测模块（Dog-Pose 预训练 + TensorRT）
- [ ] ASBAR 路径 PoseC3D 行为分类（24关键点→3D热图→行为类别）
- [ ] 规则引擎动作分类（坐/卧/立/走 四类基础兜底）
- [ ] 评分引擎（GA-T 标准 + 延迟+准确度+保持时长 三维度）
- [ ] FastAPI 后端搭建（上传+推理+结果 API）
- [ ] PostgreSQL 15 数据库 + pgvector 初始化
- [ ] Vue 3 前端原型（视频上传 + Canvas 24点标注 + 评分报告）
- [ ] 端到端验证（1 段训练视频 → 评分报告 PDF）

### Phase 2 — 核心功能（+4-6 周）

- [ ] 工作犬数据采集（200-500 张关键点 + 500 行为时序）
- [ ] Dog-Pose + 工作犬数据联合微调 YOLO
- [ ] PoseC3D vs ST-GCN 对比实验
- [ ] 扩展动作库至 P0+P1 = 16 种行为
- [ ] 完善评分体系（7 维度 + USPCA 映射）
- [ ] 训练历史记录与对比分析 + PDF/CSV 导出

### Phase 3 — 专业功能（+8-12 周）

- [ ] ST-GCN + BC 模块复现（参考 BCST-GCN 2026）
- [ ] 多犬同时追踪（SLEAP 参考）
- [ ] 3D 姿态重建（多摄像头方案）
- [ ] FCI-IGP 标准适配
- [ ] 幼犬选育评估模块
- [ ] 数据闭环（训练数据→模型迭代）
- [ ] 批量视频处理 + 模型定制训练服务

---

## 六、风险与应对

| 风险 | 等级 | 应对方案 |
|------|------|---------|
| 无专门工作犬数据集 | ⚠️ 中 | Dog-Pose 6,773 张 + 自建 200-500 张工作犬场景 (Phase 2) |
| PoseC3D/MMAction2 环境配置 | ⚠️ 中 | Docker 环境预配置 + pip 一键安装脚本 |
| 吠叫检测精度 | ⚠️ 中 | 额外训练 2 个嘴部关键点 + 时序开合分析 |
| AGPL-3.0 许可证限制 | ⚠️ 中 | 商用买 Ultralytics 许可；或替换 YOLO11n (旧版 MIT) |
| Jetson 采购周期 | ⚠️ 低 | 可先用 Windows GPU 开发 |
| 训导员遮挡狗姿态 | ⚠️ 中 | 多摄像头 + 单帧关键点插值填充 |
| 不同品种体型差异 | ⚠️ 低 | 数据增强 + 关键点归一化 |
| 复杂训练场背景 | ⚠️ 低 | 背景增强 + 关键点方法天然抗背景变化 |

---

## 七、参考资源索引

### 调研文档（本目录）

| 文档 | 大小 | 核心内容 |
|------|------|---------|
| RESEARCH_LITERATURE.md | 15 KB | 学术论文综述（BCST-GCN/ASBAR/PoseC3D/SLEAP, 12+篇论文） |
| RESEARCH_STANDARDS.md | 12 KB | 工作犬标准（GA-T/USPCA/FCI-IGP/NC State, 22种行为分类） |
| RESEARCH_IMPLEMENTATION.md | 20 KB | 实现细节（YOLO微调/TensorRT部署/Database Schema/代码） |
| RESEARCH_OPENSOURCE.md | 9 KB | 开源项目分析（Ultralytics/DLC/SLEAP/SimBA, 11仓库） |
| RESEARCH_TECH_COMPARISON.md | 8 KB | 技术选型对比矩阵（YOLO26/DLC/SLEAP, PoseC3D/LSTM/ST-GCN） |
| RESEARCH_COMMERCIAL.md | 8 KB | 商业竞品分析（Traini/Fi/DogBase, 市场空白确认） |
| RESEARCH_YOLO26_DEPLOY_DEEP.md | 25 KB | YOLO26深度部署分析（微调参数/边缘硬件/ONNX/TensorRT） |
| BCST-GCN_ST-GCN_深度分析报告.md | 10 KB | BCST-GCN论文架构重现 + ST-GCN代码分析 |
| 工作犬标准自动化深度调研报告.md | 16 KB | GA-T 2104详细分解 + 37种动作自动化可行性矩阵 |

### 开源项目

| 项目 | Stars | 用途 | 复用方式 |
|------|-------|------|---------|
| ultralytics/ultralytics | 59,005 | YOLO 姿态引擎 | 核心引擎，pip 安装 |
| MMAction2 (OpenMMLab) | 4,000+ | 行为分类框架 | PoseC3D/ST-GCN 实现 |
| MitchFuchs/asbar | ⭐ | 动物骨架行为识别 | **最直接代码参考** |
| DeepLabCut/DeepLabCut | 5,701 | 自定义姿态训练 | 标注工具备用 |
| talmolab/sleap | 598 | 多动物追踪 | Phase 3 多犬参考 |
| sgoldenlab/simba | 487 | 行为分类管道 | Phase 1 特征工程参考 |

### 学术论文

| 论文 | 来源 | 年度 | 与本系统关系 |
|------|------|------|-------------|
| BCST-GCN (pig behavior) | Front. Vet. Sci. | 2026 | **最直接技术参考**, ST-GCN+BC |
| NC State Working Dog | Science | 2026 | 方向可行性验证 |
| ASBAR (Animal Skeleton Action) | eLife | 2024 | **最可复用代码** |
| SLEAP (Multi-animal pose) | Nature Methods | 2022 | 多犬追踪参考 |
| Hierarchical Dog Behavior | NeurIPS | 2025 | 单目3D狗姿态 |
| PoseC3D | CVPR | 2022 | **Phase 1 行为分类架构** |
| Dog-Pose Dataset | Ultralytics | 2025 | 24关键点数据集 |

### 数据集

| 数据集 | 大小 | 关键点 | 用途 |
|--------|------|--------|------|
| Dog-Pose (Ultralytics) | 8,476 张 | 24 | **首选训练数据** |
| AP-10K | 10,000+ 图 | 45物种 | 额外品种增强 |
| Animal Kingdom | 50+物种 | 姿态+行为 | 跨物种泛化参考 |
| SyDog-Video (合成) | 500 视频 | 时序姿势 | 合成数据增强（Phase 2） |
| StanfordExtra | 12,000 实例 | 20 | 狗品种补充 |
| 3DDogs-Lab/Wild | MoCap+RGBD+IMU | 3D姿态 | 3D重建参考 |
| K9Bench (HF) | YouTube 视频 | 视频问答 | 行为评估 benchmark |
