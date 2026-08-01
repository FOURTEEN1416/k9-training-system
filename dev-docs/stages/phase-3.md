# Phase 3 — 专业阶段计划

> 阶段: Phase 3 专业
> 状态: 🔄 实施中（v2.4: 3.1b + 3.2b + 3.2c + 3.3b 完成，核心追踪+ReID+3D 配对就绪）
> Owner: Phase 3 专业
> 入口条件: Phase 2 验收通过 ✅（2026-07-30，见 [reports/phase-2-validation.md](../../reports/phase-2-validation.md) + [ADR 0010](../decisions/0010-phase-2-to-phase-3.md)）
> 出口条件: 见 §6 验收清单
> 时间约束: 8-12 周单人全职
> 依据: [ADR 0010](../decisions/0010-phase-2-to-phase-3.md) + [stage-plan.md](../stage-plan.md) + [function-list.md](../function-list.md) + 5 份调研报告

## 1. 阶段目标

**专业级算法升级 + 多犬场景 + 国际标准 + 边缘部署**：

- **3.1 ST-GCN+BC 行为识别**：22 类（P0 8 + P1 8 + P2 6 高级），替代规则引擎作为主算法
- **3.2 多犬追踪 + ID 关联**：BoxMOT + OccluBoost + ReID，支持多犬同时测评
- **3.3 3D 姿态重建**：MotionBERT-Lite 2D-to-3D lifting，提升行为识别精度
- **3.4 FCI-IGP 标准映射**：7 维评分卡（+胆量+步态），国际工作犬标准
- **3.5 Jetson 边缘部署**：Jetson Orin Nano Super + TensorRT FP16
- **3.6 用户权限 + 多租户**（从 Phase 2 延后）：RBAC + 基地管理员角色
- **3.7 训练历史对比可视化**（从 Phase 2 延后）：历史评分查询 + 对比图表

**关键技术验证**：
- ST-GCN+BC 22 类行为准确率（真实数据）≥ 85%
- 多犬追踪 MOTA ≥ 70%
- 3D 姿态重建 MPJPE ≤ 50mm
- Jetson 边缘部署延迟 ≤ 0.5 min/min 视频
- RBAC 多角色验证

## 2. 范围

### 2.1 包含

- **ST-GCN+BC**：pyskl + ST-GCN++ 主干 + 自研 BC 头 + K9Graph 24 节点拓扑
- **多犬追踪**：BoxMOT (8257★) + OccluBoost + YOLO26-pose 集成
- **3D 姿态重建**：MotionBERT-Lite 2D-to-3D lifting（主路线）+ Anipose 多视角（可选增强）
- **FCI-IGP 标准映射**：7 维评分卡 YAML（准确度 0.25 + 延迟 0.15 + 保持 0.15 + 搜索效率 0.15 + 注意力 0.10 + 胆量 0.10 + 步态 0.10）
- **Jetson 边缘部署**：Jetson Orin Nano Super + JetPack 6.1 + TensorRT FP16 + 抽帧策略
- **用户权限 + 多租户**：RBAC + 基地管理员角色（从 Phase 2 延后）
- **训练历史对比可视化**：历史评分查询 + 对比图表（从 Phase 2 延后）

### 2.2 不包含（延后 Phase 4）

- LLM 行为解释器（Phase 4）
- Transformer-Mamba 长序列行为分析（Phase 4）
- RL 评分优化（Phase 4）

### 2.3 调研前置（✅ 已完成 2026-07-30）

按 AGENTS.md §1.1「广泛调研优先」+ §1.3「调研搜索强制 GitHub-First」原则，Phase 3 启动前调研已全部完成（**全程零 WebSearch**）：

| 调研项 | 优先级 | 输出报告 | 推荐方案 |
|--------|--------|---------|---------|
| ST-GCN+BC | P0 | [RESEARCH_STGCN_BC.md](../research/RESEARCH_STGCN_BC.md) (476 行) | pyskl + ST-GCN++ + 自研 BC 头 + K9Graph 24 节点拓扑 |
| 多犬追踪 | P0 | [RESEARCH_MULTI_DOG_TRACKING.md](../research/RESEARCH_MULTI_DOG_TRACKING.md) | BoxMOT (8257★) + OccluBoost 追踪器 + YOLO26-pose 集成 |
| 3D 姿态重建 | P1 | [RESEARCH_3D_POSE_RECONSTRUCTION.md](../research/RESEARCH_3D_POSE_RECONSTRUCTION.md) (391 行) | MotionBERT-Lite 2D-to-3D lifting（37.2mm MPJPE） |
| FCI-IGP 标准 | P1 | [RESEARCH_FCI_IGP_STANDARD.md](../research/RESEARCH_FCI_IGP_STANDARD.md) | 官方 2025 PDF 解析，7 维评分，22 行为 100% 覆盖 |
| Jetson 部署 | P2 | [RESEARCH_JETSON_DEPLOYMENT.md](../research/RESEARCH_JETSON_DEPLOYMENT.md) (803 行) | Orin Nano Super ($249) + JetPack 6.1 + TRT FP16 抽帧 0.399x |

## 3. 子阶段任务分解（v2.0 调研后细化）

### Phase 3.1 ST-GCN+BC 行为识别（P0，主线）

**Owner**: ML 开发（`backend/ml/behavior/`）
**依据**: [RESEARCH_STGCN_BC.md](../research/RESEARCH_STGCN_BC.md)
**选型**: pyskl (1.3k★, Apache-2.0, 2026-02 活跃) + ST-GCN++ 主干 + 自研 BC 头

- ✅ **3.1a** ST-GCN+BC 调研完成（pyskl + ST-GCN++ + 自研 BC 头）
- ✅ **3.1b** pyskl 集成 + K9Graph 24 节点拓扑定义（2026-07-30）
  - ✅ `backend/ml/behavior/stgcn_bc/` 模块新建（项目自有，不依赖 pyskl 安装即可运行）
  - ✅ `k9_graph.py::K9Graph` — 24 节点犬类拓扑（根=withers, 23 条边, 邻接矩阵对称+自环, parent 数组, 骨骼流定义, 空间分区标签）
  - ✅ `labels.py` — 22 类标签映射（P0 8 idx 0-7 + P1 8 idx 8-15 + P2 6 idx 16-21 + FCI-IGP A/B/C 阶段覆盖: A=4/B=12/C=6）
  - ✅ `data_adapter.py` — YOLO26-pose (T,24,3) ↔ pyskl 格式双向转换 + 骨骼流 + 运动流 + 归一化 + 批量标注构建
  - ✅ 单元测试 41/41 通过（`backend/tests/ml/test_stgcn_bc.py`）+ Phase 1/2 规则引擎回归测试 36/36 通过
  - ✅ mmcv-lite 2.2.0 安装成功（清华源直连，无需 Clash 代理）
  - ⏳ pyskl GitHub 克隆阻塞（Clash 代理未运行 + gitclone.com 502），ST-GCN++ 主干集成 + 训练管线作为 3.1c 前置子任务，待 Clash 恢复后 `pip install git+https://github.com/kennymckormick/pyskl.git`
  - **核心交付完成**：K9Graph + 数据适配器 + 22 类标签映射（pyskl 集成不阻塞）
- ⏳ **3.1c** 自研 BC 头实现（1D Conv + Sigmoid，联合训练 L = L_cls + 0.3·L_boundary）
  - BC 头设计：边界检测 + 分类联合优化
  - 训练数据：Phase 2 合成数据（96.3%）预训练 + 数据飞轮真实数据微调
- ⏳ **3.1d** ST-GCN+BC 训练 + 评估（22 类，真实数据）
  - 目标：22 类准确率 ≥ 85%（真实数据）
  - 评估脚本：`scripts/eval_stgcn_bc.py`
- ⏳ **3.1e** ST-GCN+BC 部署集成（双轨并行，非直接替换）
  - 迁移期双轨：ST-GCN+BC（主）+ 规则引擎（备）
  - 三阶段：影子 → 投票 → 主备 → 退役规则引擎
  - ONNX 导出 + Windows 部署

**自研触发**（按 AGENTS.md §5.2 用户逐案决策）:
- ST-GCN+BC 是否按自研路线推进（BC 头为自研组件）
- BC 头具体设计（频域 vs 曲率 vs 简单 1D Conv）

### Phase 3.2 多犬追踪 + ID 关联（P0，主线）

**Owner**: ML 开发（`backend/ml/tracking/`）
**依据**: [RESEARCH_MULTI_DOG_TRACKING.md](../research/RESEARCH_MULTI_DOG_TRACKING.md)
**选型**: BoxMOT (8257★, AGPL-3.0, 2026-07-28 活跃) + OccluBoost 追踪器

- ✅ **3.2a** 多犬追踪调研完成（BoxMOT + OccluBoost）
- ✅ **3.2b** BoxMOT 集成 + YOLO26-pose 多犬追踪（2026-07-30）
  - ✅ `backend/ml/tracking/` 模块新建（types.py + multi_dog_tracker.py + __init__.py）
  - ✅ `types.py` — DogTrackFrame / DogTrack / MultiDogTrackingResult 数据结构（每犬独立轨迹 + 关键点序列填充 + bbox 序列 + 覆盖率摘要）
  - ✅ `multi_dog_tracker.py` — MultiDogTracker（YOLO26-pose 检测 + BoxMOT OccluBoost 追踪 + det_ind 关键点关联 + 单帧增量/整段视频双模式 + 单犬场景向后兼容 + tracker_kwargs 透传）
  - ✅ TrackerBackend 枚举（OCCLUBOOST 默认 / BOTSORT / BYTETRACK / STRONGSORT，Phase 3.2b 仅实现 OccluBoost）
  - ✅ 单元测试 26/26 通过（`backend/tests/ml/test_multi_dog_tracking.py`：数据类型 + YOLO 检测解析 + det_ind 关键点关联 + 两犬交叉场景模拟 + 单犬退化兼容）
  - ✅ `_init_tracker` 后端校验前置（未安装 boxmot 也可测试后端校验逻辑）
  - ✅ BoxMOT 19.0.0 安装成功（清华源 + 禁用 Clash 代理）+ TrackResults ndarray 子类列属性访问适配
  - ✅ `_tracks_to_frames` 列访问重构（兼容 TrackResults 属性访问 + 普通 2D ndarray 列切片）
  - ✅ **端到端验证通过**（`reports/phase-3.2b-tracking-e2e-warmup.json`）：warmup.mp4 30 帧 → 1 犬 track_id=0 → 25 帧 83.3% 覆盖 → 关键点 (30,24,3) 正确关联
  - ✅ 评估脚本 `scripts/eval_multi_dog_tracking.py` 创建（定性指标: 轨迹数/覆盖率/ID switch 估计 + 定量指标: MOTA/IDF1/precision/recall via motmetrics）
  - ⏳ BoxMOT ReID `with_reid=False` 待排查（OSNet 模型已加载但未启用，3.2c 推进）
  - ⏳ 多犬场景端到端验证（待 3.2d，需多犬视频 + GT 标注）
- ✅ **3.2c** ReID 动物身份关联（2026-07-30）
  - ✅ **with_reid=False bug 修复**: `multi_dog_tracker.py::_init_tracker` 显式构造 BoxMOT `ReID(weights, device, half).model` 对象，并设置 `with_reid=True`（OccluBoost 继承 BoostTrack 默认 with_reid=False，YAML 配置仅 CLI 路径生效，编程式实例化必须显式传入）
  - ✅ **降级机制**: ReID 模型加载失败时（如 OSNet 权重下载阻塞）自动降级为 with_reid=False，不抛异常继续追踪
  - ✅ **ReIDExtractor 模块**（`reid_extractor.py`）: 封装 BoxMOT ReID 运行时，提供 `extract_from_boxes` / `extract_from_crops` / `aggregate_track`（mean/max/median + L2 归一化）接口
  - ✅ **DogIdentityGallery 类**: 跨视频身份匹配（cosine_similarity + 阈值过滤）+ 微调数据收集（`export_crops` 导出裁剪图像）
  - ✅ **IDSwitchMonitor 模块**（`id_switch_monitor.py`）: IoU ≥ 0.5 + 重叠 ≥ 3 帧检测 ID switch 事件，`should_trigger_reid_finetune` 阈值 0.1 作为自研触发判断依据
  - ✅ **CanineReIDDataset 模块**（`reid_finetune_dataset.py`）: 犬只 ReID 微调数据接口 — `collect_from_tracking_result`（identity_mapping 跨视频身份合并）+ `integrity_report`（MIN_CROPS_PER_IDENTITY=20 + MIN_IDENTITIES=2）+ `export_boxmot_format`（market1501 兼容）+ `generate_boxmot_train_command`（用户决策后生成训练命令）
  - ✅ **单元测试 44/44 通过**（`backend/tests/ml/test_reid_id_switch.py`）：cosine_similarity 5 + ReIDExtractor 10 + DogIdentityGallery 10 + IDSwitchMonitor 7 + ReID 启用验证 4 + CanineReIDDataset 8
  - ✅ **回归测试全通过**（111/111：3.2b + 3.1b + 3.2c）
  - ✅ **端到端验证**（`reports/phase-3.2c-reid-enabled-warmup.json`）：warmup.mp4 30 帧单犬追踪稳定，with_reid 降级机制工作正常（Clash 代理未运行，OSNet 权重 Google Drive 下载阻塞，代码层 with_reid=True 启用机制通过单元测试验证）
  - ⏳ **OSNet 权重下载**: 28MB Google Drive（id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M），待 Clash 代理恢复后重跑 `scripts/verify_phase_3_2c_reid_enabled.py` 对比 ReID 真正启用前后效果
  - ⏳ **自研触发条件**: 默认 OSNet（MSMT17 预训练），当多犬场景 ID switch rate ≥ 0.1 时由用户决策是否启动犬只 OSNet 微调（5-7 天 + 标注）
- ⏳ **3.2d** 多犬场景端到端测试
  - 目标：MOTA ≥ 70%，IDF1 ≥ 80%
  - 测试脚本：`scripts/eval_multi_dog_tracking.py`

### Phase 3.3 3D 姿态重建（P1，主线）

**Owner**: ML 开发（`backend/ml/pose/`）
**依据**: [RESEARCH_3D_POSE_RECONSTRUCTION.md](../research/RESEARCH_3D_POSE_RECONSTRUCTION.md)
**选型**: MotionBERT-Lite (1429★, Apache-2.0, 2026-03 活跃) 2D-to-3D lifting

- ✅ **3.3a** 3D 姿态重建调研完成（MotionBERT-Lite 2D-to-3D lifting）
- ✅ **3.3b** 数据源决策 + 配对管线实现（2026-07-30）
  - ✅ **主监督源确认**: InterPet4D kp_world (T, 24, 3) 226 clips（与项目 Dog-Pose 24 点对齐）
  - ✅ **配对策略**: kp_world 3D 坐标 + 合成相机参数投影回 2D → (2D 投影, 3D 真值) 微调
  - ✅ **多视角融合**（Anipose）作为可选增强（需多相机硬件，Phase 3 内不阻塞）
  - ✅ **`backend/ml/pose/interpet4d_loader.py`** — InterPet4D SMAL 数据加载器
    - `InterPet4DClip` 数据类（kp_world / kp_weight / frame_idx + 可选 R/t/s_world）
    - `load_clip` / `list_clips` / `load_all_clips`（含 min_frames + min_kp_weight 过滤）
    - `get_dataset_statistics`（clip 数 / 帧数 / 犬只数统计）
    - `parse_clip_id`（interpet_dog{DD}_p{PP}_take{TT}_ego_{NNN} 命名解析）
  - ✅ **`backend/ml/pose/camera_projection.py`** — 合成相机 + 3D→2D 投影
    - `SyntheticCamera`（look-at 相机 + view/projection 矩阵）
    - `project_3d_to_2d`（正交/透视投影 + 归一化到 [-1,1]）
    - `generate_synthetic_cameras`（球面采样 + 随机扰动 + 可复现）
    - `project_clip_to_2d`（批量多相机投影）
  - ✅ **`backend/ml/pose/lifting_pairing.py`** — 2D-3D 配对构建
    - `LiftingSample`（keypoints_2d / keypoints_3d / confidence + 元数据）
    - `normalize_3d_keypoints`（根关节 withers=idx=22 中心化 + bone_length/bbox/none 尺度归一化）
    - `denormalize_3d_keypoints`（反归一化 + 根位置恢复）
    - `slice_windows`（MotionBERT 27 帧窗口 + 50% 重叠 + 短序列 padding）
    - `build_pairs_from_clip` / `build_dataset`（clip × cameras × windows 样本生成）
    - `train_val_split`（按 clip_id 划分，避免数据泄漏）
  - ✅ **`backend/ml/pose/__init__.py`** 导出三个模块公共接口
  - ✅ **单元测试 79/79 通过**（`backend/tests/ml/test_3d_pose_pairing.py`）
    - TestParseClipId 4 + TestInterPet4DClipDataclass 4 + TestInterPet4DLoaderReal 6 + TestInterPet4DLoaderMocked 9
    - TestSyntheticCamera 8 + TestProject3DTo2D 7 + TestGenerateSyntheticCameras 6 + TestProjectClipTo2D 2
    - TestNormalize3DKeypoints 6 + TestDenormalize3DKeypoints 3 + TestSliceWindows 4
    - TestBuildPairsFromClip 6 + TestBuildDataset 2 + TestComputeDatasetStatistics 2 + TestTrainValSplit 4
    - TestIntegrationSynthetic 2 + TestIntegrationReal 3（含真实 226 clips 数据集构建）
  - ✅ **回归测试全通过**（339 passed + 2 skipped，3.1b + 3.2b + 3.2c + 3.3b 全模块）
  - ✅ **端到端验证**（`reports/phase-3.3b-lifting-pairing-validation.json`，6 阶段全 PASS，耗时 2.29s）
    - Stage 1 数据加载: 226 clips + sample clip 326 帧 + kp_world (326, 24, 3)
    - Stage 2 相机投影: 8 相机 + 正交范围 [-0.94, 1.58] + 透视范围 [-1.24, 0.68] + 批量/单独一致性
    - Stage 3 归一化: bone_length scale=0.2112 + 根关节中心化误差 0.0 + 往返误差 2.98e-08
    - Stage 4 配对构建: 单 clip 96 样本（24 windows × 4 cameras）+ 形状全通过
    - Stage 5 数据集构建: 10 clips × 4 cameras = 1824 样本 + train/val 1540:284 无泄漏
    - Stage 6 可复现性: 两次构建 identical=True, max_diff=0.0
- ✅ **3.3c** MotionBERT 17→24 关键点适配（2026-08-01）
  - DSTformer 架构对关键点数量 agnostic，仅改输入/输出投影层 + 关节 embedding
  - `model.py` — DSTformerWrapper + 17→24 权重迁移（259/260 层匹配，仅 pos_embed 丢弃）
  - `train.py` — InterPet4D 微调（225 clips × 8 cameras = 82008 样本，Epoch 12 最佳）
  - `inference.py` — MotionBERTLifter（PyTorch + ONNX 双后端 + 滑动窗口推理）
  - `export_onnx.py` — ONNX 导出 + 一致性验证（max_diff=1.76e-05）
  - 微调权重：`data/models/motionbert_dog24/best_epoch.bin` (61MB)
  - ONNX：`data/models/motionbert_dog24/motionbert_dog24.onnx` (61.31MB)
- ✅ **3.3d** 3D 姿态重建精度评估（2026-08-01）
  - MPJPE = **21.74mm**（阈值 ≤ 50mm ✅，H36M 基线 37.2mm）
  - P-MPJPE = 20.67mm
  - 评估脚本：`scripts/eval_3d_pose.py`（与训练相同数据路径 build_datasets）
  - 验证集 2072 样本（30 clips），评估 MPJPE(norm)=0.0994（训练日志 0.1512）
  - 报告：`reports/phase-3.3d-3d-pose-eval.json`

### Phase 3.4 FCI-IGP 标准映射（P1，主线）

**Owner**: ML 开发（`backend/ml/scoring/`）
**依据**: [RESEARCH_FCI_IGP_STANDARD.md](../research/RESEARCH_FCI_IGP_STANDARD.md)
**选型**: FCI-IGP 2025 官方规则 + 7 维评分卡

- ✅ **3.4a** FCI-IGP 标准调研完成（官方 2025 PDF 解析）
- ⏳ **3.4b** FCI-IGP 评分卡 YAML 扩展
  - 新增 `fci_igp.yaml`：7 维（准确度 0.25 + 延迟 0.15 + 保持 0.15 + 搜索效率 0.15 + 注意力 0.10 + 胆量 0.10 + 步态 0.10）
  - 22 行为 100% 覆盖 IGP 三阶段（A 追踪 / B 服从 / C 护卫）
  - DQ 硬约束块（枪怯/不放口/衔取不吐 = 取消资格）
  - 5 级评分（Excellent 96%+ / Very Good 90%+ / Good 80%+ / Satisfactory 70%+ / Insufficient <70%）
- ⏳ **3.4c** FCI-IGP 评分卡验证
  - 目标：合成数据评分合理性 + 三档验证（excellent/failing/borderline）
  - 评估脚本：`scripts/eval_fci_igp.py`

### Phase 3.5 Jetson 边缘部署（P2，主线）

**Owner**: 后端开发（`backend/`）
**依据**: [RESEARCH_JETSON_DEPLOYMENT.md](../research/RESEARCH_JETSON_DEPLOYMENT.md)
**选型**: Jetson Orin Nano Super ($249) + JetPack 6.1 + TensorRT FP16

- ✅ **3.5a** Jetson 部署调研完成（Orin Nano Super + TRT FP16）
- ⏳ **3.5b** Jetson 部署环境配置
  - 硬件采购：Jetson Orin Nano Super + NVMe SSD + 散热风扇 + 电源（~$309）
  - JetPack 6.1 安装 + TensorRT 10.3 + PyTorch 2.10.0 aarch64 wheel
  - Docker 容器：`ultralytics/ultralytics:latest-jetson-jetpack6`
- ⏳ **3.5c** ONNX → TensorRT 引擎转换 + FP16 量化
  - **关键陷阱**：INT8 校准必须在 Jetson 上执行（不可跨平台）
  - engine 文件不可跨平台（Windows `.engine` ≠ Jetson `.engine`）
  - 抽帧策略（每 2 帧处理 1 帧）：0.399x ≤ 0.5 阈值 ✅
- ⏳ **3.5d** 平台抽象层设计
  - InferenceBackend 抽象类 + AutoBackend（统一 ONNX/TRT 接口）
  - Windows ↔ Jetson 双平台维护（Docker 容器化）
- ⏳ **3.5e** Jetson 边缘部署延迟测试
  - 目标：延迟 ≤ 0.5 min/min 视频
  - 监控：jetson-stats (jtop)

### Phase 3.6 用户权限 + 多租户（P1，从 Phase 2 延后）

**Owner**: 后端开发（`backend/app/`）

- ⏳ **3.6a** RBAC + 基地管理员角色 DB migration
  - 用户表 + 角色表 + 权限表 + 基地表
  - 角色：超管 / 基地管理员 / 训导员 / 查看者
- ⏳ **3.6b** RBAC 权限验证
  - API 中间件 + 装饰器
  - 资源级权限（犬只 / 评分 / 模型）
- ⏳ **3.6c** 多租户场景测试
  - 基地隔离 + 数据权限
  - 测试脚本：`scripts/eval_rbac.py`

### Phase 3.7 训练历史对比可视化（P2，从 Phase 2 延后）

**Owner**: 前端 + 后端

- ⏳ **3.7a** 历史评分查询 API 扩展（对比查询）
  - `GET /api/scores/compare?dog_ids=1,2&date_from=...&date_to=...`
  - 趋势统计 + 对比统计
- ⏳ **3.7b** 对比可视化前端
  - Vue + ECharts 折线图 / 雷达图 / 对比表
  - `frontend/src/views/scores/compare.vue`
- ⏳ **3.7c** 训练历史对比端到端测试

### Phase 3.8 系统集成 + 端到端（P1，收尾）

**Owner**: 全栈

- ⏳ **3.8a** 端到端测试（22 行为 + 多犬 + 3D + FCI-IGP + Jetson + 用户权限）
  - 测试脚本：`scripts/phase3_8_e2e_test.py`
- ⏳ **3.8b** 延迟验证（≤ 1 min/min 视频，维持）
- ⏳ **3.8c** 部署文档更新（Phase 3 新功能 + Jetson 部署）
- ⏳ **3.8d** 用户手册更新（FCI-IGP 场景 + 多犬 + 训练历史对比）

## 4. 周计划（v2.0 调研后细化）

### Week 1-2（调研已完成，进入实施启动）

- ✅ 5 项调研完成（ST-GCN+BC / 多犬追踪 / 3D / FCI-IGP / Jetson）
- 3.1b pyskl 集成 + K9Graph 24 节点拓扑
- 3.2b BoxMOT 集成 + YOLO26-pose 多犬追踪
- 3.6a RBAC DB migration 启动

### Week 3-4（ST-GCN+BC + 多犬追踪主线）

- 3.1c 自研 BC 头实现 + 联合训练
- 3.1d ST-GCN+BC 训练 + 评估（22 类）
- 3.2c ReID 动物身份关联
- 3.2d 多犬场景端到端测试

### Week 5-6（3D 姿态重建 + FCI-IGP）

- 3.3b MotionBERT 数据源决策 + InterPet4D 配对
- 3.3c MotionBERT 17→24 适配 + 微调
- 3.3d 3D 姿态重建精度评估
- 3.4b FCI-IGP 评分卡 YAML 扩展
- 3.4c FCI-IGP 评分卡验证

### Week 7-8（Jetson 部署 + 用户权限）

- 3.5b Jetson 硬件采购 + JetPack 6.1 配置
- 3.5c ONNX → TensorRT 转换 + FP16 量化
- 3.5d 平台抽象层设计
- 3.6b RBAC 权限验证
- 3.6c 多租户场景测试

### Week 9-10（ST-GCN+BC 部署 + 训练历史对比）

- 3.1e ST-GCN+BC 部署集成（双轨并行）
- 3.5e Jetson 边缘部署延迟测试
- 3.7a 历史评分查询 API 扩展
- 3.7b 对比可视化前端

### Week 11-12（系统集成 + 验收）

- 3.7c 训练历史对比端到端测试
- 3.8a 端到端测试（22 行为 + 多犬 + 3D + FCI-IGP + Jetson + 用户权限）
- 3.8b 延迟验证
- 3.8c 部署文档更新
- 3.8d 用户手册更新
- Phase 3 验收报告归档

## 5. 依赖关系（v2.0 调研后细化）

```
调研前置 ✅ ──┐
              ├── 3.1 ST-GCN+BC ──┐
3.3 3D ───────┤                    ├── 3.8 系统集成 ── Phase 3 验收
              ├── 3.2 多犬追踪 ───┤
3.4 FCI-IGP ──┤                    │
              ├── 3.5 Jetson ─────┤
3.6 RBAC ─────┤                    │
              ├── 3.7 训练历史 ───┘
```

**关键依赖**:
- 3.1 ST-GCN+BC 依赖 3.3 3D 姿态（3D 骨架序列作为 ST-GCN+BC 输入）
- 3.2 多犬追踪独立于 3.1/3.3，可并行
- 3.5 Jetson 依赖 3.1e ST-GCN+BC ONNX 导出
- 3.8 系统集成依赖全部子阶段完成

## 6. 验收清单（v2.0 调研后细化）

| 编号 | 项目 | 期望 | 验证方法 | 状态 |
|------|------|------|---------|------|
| §6.1 | ST-GCN+BC | 22 类准确率 ≥ 85%（真实数据） | `scripts/eval_stgcn_bc.py` | ⏳ |
| §6.2 | 多犬追踪 | MOTA ≥ 70% / IDF1 ≥ 80% | `scripts/eval_multi_dog_tracking.py` | ⏳ |
| §6.3 | 3D 姿态重建 | MPJPE ≤ 50mm | `scripts/eval_3d_pose.py` | ⏳ |
| §6.4 | FCI-IGP | 评分卡验证通过 | `scripts/eval_fci_igp.py` | ⏳ |
| §6.5 | Jetson 部署 | 延迟 ≤ 0.5 min/min 视频 | Jetson jtop 监控 | ⏳ |
| §6.6 | 用户权限 | 多角色验证通过 | `scripts/eval_rbac.py` | ⏳ |
| §6.7 | 训练历史 | 对比可视化 | 端到端测试 | ⏳ |
| §6.8 | 端到端 | 全流程跑通 | `scripts/phase3_8_e2e_test.py` | ⏳ |
| §6.9 | 延迟 | ≤ 1 min/min 视频（维持） | 延迟测试 | ⏳ |
| §6.10 | 文档 | 部署 + 用户手册 | 文档审查 | ⏳ |

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| ST-GCN+BC 训练数据不足 | 60% | 高 | 数据飞轮 + keypoint-MoSeq + 弱监督多路径补充 |
| 多犬追踪 ID switch 高 | 50% | 中 | ReID 微调 + 时序平滑 + 单犬场景降级 |
| 3D 姿态重建精度不达标 | 30% | 高 | MotionBERT-Lite 37.2mm 已有裕量 + 多视角融合降级 + Phase 4 RL 优化 |
| FCI-IGP DQ 判定需人工兜底 | 70% | 中 | 自动化 70% + 人工兜底 30%（护卫阶段 20% 自动化） |
| Jetson 边缘部署兼容性 | 40% | 中 | TensorRT 优先 + ONNX Runtime 降级 + Docker 容器化 |
| Windows ↔ Jetson 双平台维护成本 | 50% | 中 | Docker 容器化 + 平台抽象层（InferenceBackend） |
| Jetson 硬件采购周期 | 30% | 中 | Week 7 开始，提前下单 |

## 8. 不可逆操作清单

Phase 3 涉及以下不可逆操作，需在执行前再次确认：

- ⏳ Phase 3 ST-GCN+BC 模型权重下载/训练（pyskl 预训练权重）
- ⏳ Phase 3 多犬追踪算法集成（影响 `inference.py` 接口，单犬 → 多犬）
- ⏳ Phase 3 3D 姿态重建（MotionBERT 17→24 适配 + InterPet4D 微调）
- ⏳ Phase 3 FCI-IGP 评分卡 YAML（新增 `fci_igp.yaml`）
- ⏳ Phase 3 Jetson 部署环境配置（硬件采购 + JetPack 6.1 + Docker）
- ⏳ Phase 3 用户权限 + 多租户 DB migration（RBAC + 基地角色表）

## 9. 未解决问题

- ⏳ Phase 3 详细子阶段执行顺序优化（v2.0 已给周计划，执行中可能调整）
- ⏳ ST-GCN+BC vs 规则引擎并存策略（v2.0 决策双轨并行，三阶段迁移）
- ✅ 3D 姿态重建数据源（v2.4 完成：InterPet4D kp_world (T, 24, 3) 226 clips 主监督源 + 合成相机投影 2D 配对，见 `backend/ml/pose/` 三模块 + `reports/phase-3.3b-lifting-pairing-validation.json`）
- ⏳ Jetson 边缘部署优先级（Phase 3 内必达，Week 7 启动）
- ⏳ 2.0b Label Studio 人工标注（后台并行，Phase 3 期间用户闲暇推进）
- ⏳ ReID 微调触发条件（3.2c 已建立监测机制）：默认 OSNet（MSMT17 预训练），IDSwitchMonitor `should_trigger_reid_finetune`（switch_rate ≥ 0.1）作为判断依据，最终由用户决策是否启动犬只 OSNet 微调
- ⏳ OSNet 权重下载阻塞（3.2c v2.3 新增）: 28MB Google Drive（id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M），Clash 代理未运行导致下载失败，MultiDogTracker 自动降级为 with_reid=False 继续追踪。待 Clash 恢复后重跑 `scripts/verify_phase_3_2c_reid_enabled.py` 对比 ReID 启用前后效果
- ⏳ pyskl GitHub 克隆阻塞（v2.1 新增）: Clash 代理未运行 + gitclone.com 502，待用户启动 Clash 后通过 `pip install git+https://github.com/kennymckormick/pyskl.git` 集成 ST-GCN++ 主干，3.1c 自研 BC 头实现的前置依赖

## 10. 出口决策

Phase 3 验收通过后，依据 ADR（待创建）决策是否升级 Phase 4。

## 11. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-30 | 初始版本，Phase 3 启动（ADR 0010），框架结构 + 调研前置说明。详细子阶段任务分解 + 周计划待调研完成后在 v2.0 补充 |
| v2.0 | 2026-07-30 | **调研完成 + 子阶段任务分解细化**：①**5 项调研全部完成**（ST-GCN+BC / 多犬追踪 / 3D 姿态重建 / FCI-IGP / Jetson，全程零 WebSearch）；②**§2.3 调研前置** ✅ 完成 + 推荐方案汇总；③**§3 子阶段任务分解**细化（含具体仓库选型 + 实施路径 + 自研触发点）：3.1 pyskl+ST-GCN+++自研BC头 / 3.2 BoxMOT+OccluBoost / 3.3 MotionBERT-Lite 2D-to-3D lifting / 3.4 FCI-IGP 7 维评分 / 3.5 Orin Nano Super+TRT FP16 / 3.6 RBAC / 3.7 训练历史对比；④**§4 周计划**细化（12 周分解）；⑤**§5 依赖关系**更新（3.1 依赖 3.3 3D 骨架）；⑥**§6 验收清单**细化（含具体评估脚本）；⑦**§7 风险与缓解** + **§8 不可逆操作清单** + **§9 未解决问题**新增。Phase 3 进入实施阶段 |
| v2.1 | 2026-07-30 | **3.1b pyskl 集成 + K9Graph 24 节点拓扑定义完成**：①`backend/ml/behavior/stgcn_bc/` 模块新建（项目自有，pyskl 安装不阻塞核心代码）；②`k9_graph.py::K9Graph`（24 节点 / 23 边 / 根=withers / 邻接矩阵对称+自环 / parent 数组 / 骨骼流定义 / 空间分区标签）；③`labels.py`（22 类标签映射: P0 8 idx 0-7 + P1 8 idx 8-15 + P2 6 idx 16-21 + FCI-IGP A/B/C 阶段覆盖 A=4/B=12/C=6）；④`data_adapter.py`（YOLO26-pose (T,24,3) ↔ pyskl 格式双向转换 + 骨骼流 + 运动流 + 归一化 + 批量标注构建）；⑤`constants.py` 扩展 P2 6 类（guard/release/retrieve/jump/scale/search_blind）；⑥单元测试 41/41 通过 + Phase 1/2 规则引擎回归测试 36/36 通过；⑦mmcv-lite 2.2.0 安装成功（清华源直连）；⑧pyskl GitHub 克隆阻塞（Clash 代理未运行 + gitclone.com 502），ST-GCN++ 主干集成作为 3.1c 前置子任务待 Clash 恢复。§3.1b 标记 ✅ 完成，§9 新增 pyskl 克隆阻塞未解决问题 |
| v2.2 | 2026-07-30 | **3.2b BoxMOT 集成 + YOLO26-pose 多犬追踪完成**：①`backend/ml/tracking/` 模块新建（types.py + multi_dog_tracker.py + __init__.py）；②`types.py` DogTrackFrame/DogTrack/MultiDogTrackingResult 数据结构（每犬独立轨迹 + 关键点序列填充 + bbox 序列 + 覆盖率摘要）；③`multi_dog_tracker.py` MultiDogTracker（YOLO26-pose 检测 + BoxMOT OccluBoost 追踪 + det_ind 关键点关联 + 单帧增量/整段视频双模式 + 单犬场景向后兼容）；④TrackerBackend 枚举（OCCLUBOOST 默认/BOTSORT/BYTETRACK/STRONGSORT）；⑤单元测试 26/26 通过（数据类型 + YOLO 检测解析 + det_ind 关键点关联 + 两犬交叉场景模拟 + 单犬退化兼容）；⑥`_init_tracker` 后端校验前置（未安装 boxmot 也可测试）；⑦mock 修复（.cpu().numpy() 调用链 + 普通二维 ndarray 匹配真实 BoxMOT 输出格式）。§3.2b 标记 ✅ 完成。3.2c ReID 犬只身份关联 + 3.2d 多犬场景端到端测试待推进 |
| v2.3 | 2026-07-30 | **3.2c ReID 犬只身份关联完成**：①**with_reid=False bug 修复**（OccluBoost 继承 BoostTrack 默认 with_reid=False，YAML 配置仅 CLI 路径生效；编程式实例化必须显式构造 `ReID(weights, device, half).model` 对象 + `with_reid=True`）；②**降级机制**（ReID 模型加载失败时自动降级为 with_reid=False，不抛异常继续追踪）；③**ReIDExtractor 模块**（`reid_extractor.py`：extract_from_boxes/extract_from_crops/aggregate_track mean-max-median + L2 归一化）；④**DogIdentityGallery 类**（跨视频身份匹配 + cosine_similarity + 阈值过滤 + export_crops 微调数据收集）；⑤**IDSwitchMonitor 模块**（`id_switch_monitor.py`：IoU ≥ 0.5 + 重叠 ≥ 3 帧检测 + should_trigger_reid_finetune 阈值 0.1 自研触发判断）；⑥**CanineReIDDataset 模块**（`reid_finetune_dataset.py`：collect_from_tracking_result + identity_mapping 跨视频身份合并 + integrity_report + export_boxmot_format market1501 兼容 + generate_boxmot_train_command）；⑦**单元测试 44/44 通过**（cosine 5 + ReIDExtractor 10 + Gallery 10 + IDSwitchMonitor 7 + ReID 启用验证 4 + CanineReIDDataset 8）；⑧**回归测试 111/111 全通过**（3.1b + 3.2b + 3.2c）；⑨**端到端验证**（`reports/phase-3.2c-reid-enabled-warmup.json`：warmup.mp4 30 帧单犬追踪稳定 + 降级机制工作正常）；⑩**OSNet 权重下载阻塞**（28MB Google Drive，Clash 代理未运行，待恢复后重跑 `scripts/verify_phase_3_2c_reid_enabled.py` 对比 ReID 启用前后效果）。§3.2c 标记 ✅ 完成，§9 未解决问题: ReID 微调触发条件 + OSNet 权重下载阻塞 + pyskl 克隆阻塞 |
| v2.4 | 2026-07-30 | **3.3b 3D 姿态重建数据源决策 + 配对管线实现完成**：①**`backend/ml/pose/interpet4d_loader.py`**（InterPet4D SMAL 数据加载器：InterPet4DClip 数据类 + load_clip/list_clips/load_all_clips + min_frames/min_kp_weight 过滤 + get_dataset_statistics + parse_clip_id 命名解析）；②**`backend/ml/pose/camera_projection.py`**（合成相机 + 3D→2D 投影：SyntheticCamera look-at 相机 + view/projection 矩阵 + project_3d_to_2d 正交/透视投影 + generate_synthetic_cameras 球面采样+随机扰动+可复现 + project_clip_to_2d 批量多相机投影；修复 homogeneous 输入 reshape 不一致 bug）；③**`backend/ml/pose/lifting_pairing.py`**（2D-3D 配对构建：LiftingSample + normalize_3d_keypoints 根关节 withers=idx=22 中心化+bone_length/bbox/none 尺度归一化 + denormalize_3d_keypoints 反归一化 + slice_windows MotionBERT 27 帧窗口+50% 重叠+短序列 padding + build_pairs_from_clip/build_dataset clip×cameras×windows 样本生成 + train_val_split 按 clip_id 划分无泄漏）；④**`backend/ml/pose/__init__.py`** 导出三个模块公共接口；⑤**单元测试 79/79 通过**（`backend/tests/ml/test_3d_pose_pairing.py`：parse_clip_id 4 + Clip 数据类 4 + 真实数据加载 6 + 合成 mock 9 + SyntheticCamera 8 + Project3DTo2D 7 + GenerateCameras 6 + ProjectClip 2 + Normalize 6 + Denormalize 3 + SliceWindows 4 + BuildPairs 6 + BuildDataset 2 + ComputeStats 2 + TrainValSplit 4 + Integration 合成 2 + Integration 真实 3）；⑥**回归测试 339 passed + 2 skipped 全通过**（3.1b + 3.2b + 3.2c + 3.3b 全模块）；⑦**端到端验证**（`reports/phase-3.3b-lifting-pairing-validation.json`：6 阶段全 PASS，耗时 2.29s — Stage 1 数据加载 226 clips + Stage 2 相机投影 8 相机一致性 + Stage 3 归一化 root 中心化误差 0.0+往返误差 2.98e-08 + Stage 4 配对 96 样本 + Stage 5 数据集 1824 样本 train/val 1540:284 无泄漏 + Stage 6 可复现性 identical=True）。§3.3b 标记 ✅ 完成，3.3c MotionBERT 17→24 关键点适配待推进 |
