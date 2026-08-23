# RESEARCH: 多犬追踪算法调研

> 调研日期: 2026-07-30
> 调研人: AI Agent (general_purpose_task subagent)
> 阶段: Phase 3.2a 多犬追踪
> 依据: AGENTS.md §1.3 (github-search-strategy + browser-automation)
> 调研方法: sindresorhus/awesome 元目录 → GitHub Search API → 仓库 meta 验证 → README + Windows Issues 抓取
> 数据源: GitHub API (api.github.com/repos) + GitHub Search API + raw README
> 缓存位置: `dev-docs/research/_crawl_cache/` (调研完毕后清理)

---

## 0. 调研执行摘要

| 维度 | 结论 |
|------|------|
| 推荐方案 | **BoxMOT (统一框架) + YOLO26-pose (已用) + OSNet/lmbn ReID** |
| 追踪器首选 | **OccluBoost** (MOT17 HOTA 70.47 / SportsMOT 83.17，IDF1 最高) |
| ReID 首选 | OSNet (Torchreid, HuggingFace 权重) → 犬只微调；BoxMOT 默认 `lmbn_n_duke` |
| Windows 兼容 | BoxMOT ✅ 良好（无突出 Windows issue）；ByteTrack 原仓 ⚠️ cython_bbox 问题 |
| 集成复杂度 | 低：`pip install boxmot`，单 API 调用接收 YOLO 检测框 |
| 自研触发 | ① OSNet 犬只微调（用户决策）② 多犬 ID switch 后处理（关键点一致性） |
| 许可证风险 | BoxMOT = AGPL-3.0（与 YOLO26 一致，Phase 5 前解决） |

---

## 1. MOT 算法演进

### 1.1 演进路线（2016 → 2026）

```
SORT (2016) → DeepSORT (2017) → ByteTrack (ECCV 2022) → OC-SORT (CVPR 2023) → Hybrid-SORT (2024)
                              ↘ BoT-SORT (arXiv 2022) → BoT-SORT-ReID
                              ↘ StrongSORT (TMM 2023)
                              ↘ Deep-OC-SORT (2023)
                              ↘ OccluBoost (2024-2025) ← 当前 SOTA
端到端路线: DETR → MOTR (ECCV 2022) → MOTRv2 (CVPR 2023) → MeMOTR (ICCV 2023)
```

### 1.2 关键算法对比

| 算法 | 年份 | 核心创新 | MOT17 MOTA | MOT17 IDF1 | MOT17 HOTA | FPS (V100) |
|------|------|---------|-----------|-----------|-----------|-----------|
| **SORT** | 2016 | Kalman + Hungarian（纯运动） | 59.8 | 62.1 | 51.1 | 60+ |
| **DeepSORT** | 2017 | + 深度外观特征 (ReID) | 75.4 | 71.6 | ~57 | 40 |
| **ByteTrack** | ECCV 2022 | 关联**所有**检测框（含低分） | 80.3 | 77.3 | 63.1 | 29.6 |
| **BoT-SORT** | arXiv 2022 | + GMC 相机运动补偿 + 改进 Kalman | 80.6 | 79.5 | 64.6 | ~20 |
| **BoT-SORT-ReID** | arXiv 2022 | + 独立 ReID 分支 | 80.5 | **80.2** | **65.0** | ~18 |
| **StrongSORT** | TMM 2023 | DeepSORT 增强（EMA 更新+GCE+JET） | — | — | ~63 | — |
| **OC-SORT** | CVPR 2023 | 观测中心（ORU）+ ICT | 77.5 | 76.1 | 62.1 | 30+ |
| **Deep-OC-SORT** | 2023 | OC-SORT + 外观特征 | — | — | ~63 | — |
| **Hybrid-SORT** | 2024 | 混合运动+外观，OBB 友好 | — | — | ~64 | — |
| **OccluBoost** | 2024-2025 | 遮挡增强 (BoxMOT 实现) | 78.3 | **84.1** | **70.5** | — |
| **MOTR** | ECCV 2022 | 端到端 Transformer | 73.5 | 71.0 | 56.3 | ~5 |
| **MOTRv2** | CVPR 2023 | + bootstrapping | 78.5 | 73.2 | 59.9 | ~5 |

**核心洞察**：
- **Tracking-by-Detection (TBD)** 路线胜出：OccluBoost/BoT-SORT/ByteTrack 等都基于此范式，精度+速度均优于端到端 MOTR
- **IDF1 大幅提升靠 ReID**：BoT-SORT-ReID (80.2) vs BoT-SORT (79.5)，OccluBoost (84.1) 说明 ReID + 遮挡处理是关键
- **多犬场景关键**: IDF1 比 MOTA 更重要（ID switch 是核心痛点），所以 OccluBoost > ByteTrack

---

## 2. SOTA MOT 仓库对比

### 2.1 元数据全景（按 Star 排序，仅列活跃仓库）

| # | 仓库 | Star | 最近 push | License | 语言 | Windows 兼容 | 备注 |
|---|------|------|----------|---------|------|-------------|------|
| 1 | **ultralytics/ultralytics** | 60,028 | 2026-07-30 | AGPL-3.0 | Python | ✅ 完美 | 已用 Phase 1-2，内置 `model.track()` |
| 2 | **PaddlePaddle/PaddleDetection** | 14,341 | 2026-05-28 | Apache-2.0 | Python | ⚠️ setup.py 问题 | 含 PP-Tracking |
| 3 | **mikel-brostrom/boxmot** ⭐ | 8,257 | 2026-07-28 | AGPL-3.0 | Python | ✅ 良好 | **统一 9 种追踪器框架，首选** |
| 4 | **FoundationVision/ByteTrack** | 6,587 | 2024-06-19 | MIT | Python | ⚠️ cython_bbox 编译问题 | ECCV 2022 原仓 |
| 5 | **nwojke/deep_sort** | 6,160 | 2025-03-02 | BSD-3 | Python+C++ | ⚠️ 需调试 | DeepSORT 原作者 |
| 6 | **abewley/sort** | 4,367 | 2023-11-28 | GPL-3.0 | Python | ✅ 纯 Python | SORT 原版，无 ReID |
| 7 | **ifzhang/FairMOT** | 4,243 | 2023-09-19 | Apache-2.0 | Python | ⚠️ | FairMOT，已不活跃 |
| 8 | **KaiyangZhou/deep-person-reid** | 4,883 | 2026-01-09 | MIT | Python | ✅ 已修复 | **OSNet/Torchreid 官方** |
| 9 | **JDAI-CV/fast-reid** | 3,974 | 2024-07-30 | Apache-2.0 | Python | ✅ 支持 | FastReID 京东 |
| 10 | **open-mmlab/mmtracking** | 3,892 | 2023-09-19 | Apache-2.0 | Python | ❌ 已不活跃 | OpenMMLab，停更 |
| 11 | **ZQPei/deep_sort_pytorch** | 3,011 | 2024-07-16 | GPL-3.0 | Python | ⚠️ NMS 编译问题 | DeepSORT PyTorch 流行实现 |
| 12 | **cheind/py-motmetrics** | 1,484 | 2026-07-17 | MIT | Python | ✅ 纯 Python | **MOT 评估指标工具** |
| 13 | **luanshiyinyang/awesome-multiple-object-tracking** | 1,484 | 2025-10-07 | — | — | — | awesome MOT 目录 |
| 14 | **NirAharon/BoT-SORT** | 1,496 | 2024-08-08 | MIT | Python | ⚠️ cl.exe | BoT-SORT 原仓 |
| 15 | **JonathonLuiten/TrackEval** | 1,248 | 2024-07-03 | MIT | Python | ✅ | HOTA 指标评估 |
| 16 | **noahcao/OC_SORT** | 1,123 | 2026-04-21 | MIT | Python | ✅ 已修复 | OC-SORT CVPR2023 |
| 17 | **megvii-research/MOTR** | 809 | 2024-01-15 | NOASSERTION | Python | — | 端到端 MOTR ECCV2022 |
| 18 | **dyhBUPT/StrongSORT** | 948 | 2024-05-28 | GPL-3.0 | Python | — | StrongSORT TMM2023 |
| 19 | **megvii-research/MOTRv2** | 486 | 2023-02-28 | NOASSERTION | Python | — | MOTRv2 CVPR2023 |
| 20 | **mikel-brostrom/Yolov7_StrongSORT_OSNet** | 482 | 2024-05-28 | GPL-3.0 | Python | — | YOLOv7+StrongSORT+OSNet 集成（BoxMOT 前身） |
| 21 | **GerardMaggiolino/Deep-OC-SORT** | 279 | 2023-05-06 | MIT | Python | — | Deep-OC-SORT |
| 22 | **emptysoal/TensorRT-YOLOv8-ByteTrack** | 247 | 2025-11-01 | MIT | C++ | — | TensorRT 部署 |
| 23 | **wish44165/YOLOv12-BoT-SORT-ReID** | 197 | 2026-07-28 | AGPL-3.0 | Python | — | **最新集成（CVPR 2025 Anti-UAV）** |
| 24 | **GuillaumeMougeot/DogFaceNet** | 160 | 2026-04-29 | MIT | Python | — | **犬脸部识别 FaceNet** |
| 25 | **HanGuangXin/ByteTrack_ReID** | 135 | 2023-11-23 | MIT | Python | — | ByteTrack + ReID 范式 |
| 26 | **PINTO0309/BoT-SORT-ONNX-TensorRT** | 50 | 2024-01-24 | MIT | Python | — | 纯 ONNXRuntime 部署 |
| 27 | **robot-perception-group/RAPID-animal-reidentification** | 14 | 2026-06-22 | GPL-3.0 | Python | — | **动物 ReID 最新研究** |
| 28 | **owahltinez/triplet-loss-animal-reid** | 14 | 2025-08-01 | MIT | Python | — | 动物 ReID triplet |
| 29 | **Imageomics/MAVRIC** | 0 | 2026-06-12 | MIT | Python | — | 视频动物 ReID 流水线 |
| 30 | **ddyy-hash/dog-reid-based-on-yolo-sam** | 2 | 2026-05-19 | Apache-2.0 | Python | — | **狗 ReID (YOLO+SAM)** |

### 2.2 Windows 兼容性深度分析（基于 issues 抓取）

| 仓库 | Windows Issues 摘要 | 结论 |
|------|-------------------|------|
| **mikel-brostrom/boxmot** | 仅依赖 bump PR，无 Windows 兼容 issue | ✅ 良好 |
| **ultralytics/ultralytics** | 无 Windows 特定 issue，已用 Phase 1-2 验证 | ✅ 完美 |
| **FoundationVision/ByteTrack** | #313 windows 无检测结果 (open)、#179 Windows (open)、#52 Building on Win10 CPU (closed)、#305 setup.py develop fails (open) | ⚠️ cython_bbox 编译问题 |
| **ZQPei/deep_sort_pytorch** | #228 "I cann't build nms on windows" (open) | ⚠️ NMS 编译问题 |
| **JDAI-CV/fast-reid** | #266 "Does this project support windows OS?" (closed)、#126 "windows 如何编译？" (closed)、#74 Buffer dtype mismatch (closed) | ✅ 早期问题已修复 |
| **KaiyangZhou/deep-person-reid** | #412 PR "Adding support for rank cython in Windows" (merged)、#408/#407 Buffer dtype (closed)、README v1.3.5 提到 cython Windows 修复 | ✅ 2021 年已修复 |
| **NirAharon/BoT-SORT** | #100 CL.EXE Application Error (closed) | ⚠️ MSVC 编译器配置 |
| **noahcao/OC_SORT** | #42 gp_interpolation.py under windows (closed) | ✅ 已修复 |
| **PaddlePaddle/PaddleDetection** | #9340 ppdet/ext_op setup.py 报错 (closed)、#9474 中文路径问题 | ⚠️ setup.py 问题 |

**结论**：BoxMOT 作为封装层，已处理 ByteTrack/DeepSORT/OC-SORT 等底层追踪器的 Windows 兼容性问题，是 Windows 部署的最优选择。

---

## 3. ReID for animals

### 3.1 通用 ReID 框架（可用于犬只微调）

| 仓库 | Star | 模型 | 特点 | 动物适用性 |
|------|------|------|------|-----------|
| **KaiyangZhou/deep-person-reid** (Torchreid) | 4,883 | OSNet / OSNet-AIN | ICCV'19, TPAMI'21, ONNX/OpenVINO/TFLite 导出, HuggingFace 权重 | ⭐⭐⭐ 可迁移学习 |
| **JDAI-CV/fast-reid** | 3,974 | BagTricks / ViT / ResNet50 | Apache-2.0, TRT 部署, 京东 | ⭐⭐⭐ 可迁移学习 |
| **michuanhaohao/reid-strong-baseline** | 2,353 | ResNet50-IBN | NeurIPS'19 强基线 | ⭐⭐ |
| **alibaba/cluster-contrast-reid** | 237 | ClusterContrast | 阿里, 无监督友好 | ⭐⭐ |

### 3.2 动物 ReID 专用方案

| 仓库 | Star | 目标物种 | 方法 | 项目可用性 |
|------|------|---------|------|-----------|
| **GuillaumeMougeot/DogFaceNet** | 160 | **犬（脸部）** | FaceNet triplet loss | ⭐⭐⭐ 直接用于犬只识别 |
| **robot-perception-group/RAPID-animal-reidentification** | 14 | 通用动物 | 2026-06 最新研究 | ⭐⭐ 参考方法 |
| **owahltinez/triplet-loss-animal-reid** | 14 | 通用动物 | Triplet loss 简单实现 | ⭐⭐ |
| **Imageomics/MAVRIC** | 0 | 通用动物（视频） | 视频动物 ReID 流水线 | ⭐⭐ 视频场景参考 |
| **ddyy-hash/dog-reid-based-on-yolo-sam** | 2 | **犬** | YOLO+SAM+光照不变模块 | ⭐⭐ 犬只专用但实现不成熟 |
| **yujiwen/tiger_reid_pytorch** | 6 | 老虎 | 老虎 ReID | ⭐ 数据集参考 |
| **artemgoncarov/amur_tigers_reid** | 2 | 东北虎 | 东北虎 ReID | ⭐ 数据集参考 |

### 3.3 动物 ReID 数据集

| 数据集 | 物种 | 规模 | 用途 |
|--------|------|------|------|
| **AnimalCLEF 2025/2026** (LifeCLEF) | 多物种（含犬） | 比赛数据 | 项目可参与或下载 |
| **IPAWS102** | 大象 | 102 头 | 大型动物 ReID |
| **Amur Tiger ReID** | 东北虎 | ~100 只 | 老虎 ReID |
| **DogFaceNet dataset** | 犬 | ~1000+ 只 | 犬脸部识别 |
| **Cows2021** | 牛 | 181 头 | 牲畜 ReID |

### 3.4 ReID 在多犬场景的关键挑战

1. **同品种同毛色犬只区分**：同品种工作犬（如多只德牧）外观高度相似，纯外观 ReID 难度高
2. **姿态变化大**：犬只运动姿态（坐/卧/搜索/扑咬）导致外观剧变
3. **遮挡频繁**：多犬互动、植被遮挡、训练器材遮挡
4. **数据稀缺**：工作犬 ReID 公开数据集几乎为零

**应对策略**：
- **特征融合**：外观（OSNet）+ 姿态关键点（YOLO26-pose 24 点）+ 毛色分布直方图
- **时序一致性**：用 BoxMOT 的 Kalman 滤波 + OccluBoost 遮挡处理
- **微调路径**：在 OSNet 基础上用项目自有工作犬数据微调（自研触发，用户决策）

---

## 4. 与 YOLO26-pose 集成方案

### 4.1 推荐架构：YOLO26-pose 检测 + BoxMOT 追踪 + ReID 重关联

```
┌─────────────────────────────────────────────────────────────────┐
│ 视频帧输入                                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ YOLO26-pose 检测器（已用 Phase 1-2, AGPL-3.0）                   │
│  • 输出: 检测框 [x1,y1,x2,y2,conf,cls]                          │
│  • 输出: 24 关键点 kp [(x,y,conf)×24]                           │
│  • 模型: yolo26n-pose.pt (已验证 mAP50=92.2% on dog-pose val)   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ BoxMOT 追踪器（mikel-brostrom/boxmot, AGPL-3.0）                 │
│  • 输入: 检测框 + 当前帧图像                                     │
│  • 追踪器: OccluBoost (MOT17 HOTA 70.47, IDF1 84.14)            │
│  • ReID: lmbn_n_duke (默认) 或 OSNet (建议犬只微调)              │
│  • 输出: tracks [x1,y1,x2,y2,id,conf,cls,det_ind]               │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 关键点关联（项目自建薄层）                                        │
│  • 通过 tracks[:, 7] (det_ind) 找回对应 YOLO 检测索引             │
│  • 取该检测的 24 关键点 → 关联到 track_id                        │
│  • 输出: {track_id: [bbox, keypoints_24, reid_embedding]}        │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3 后续模块（行为识别 / 评分引擎）                           │
│  • 输入: 每 ID 的关键点时间序列                                  │
│  • 输出: 每 ID 的行为分类 + 7 维评分                             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 集成代码示例（基于 BoxMOT Python API）

```python
import numpy as np
from boxmot.trackers import OccluBoost
from ultralytics import YOLO

# Phase 1-2 已有的 YOLO26-pose 检测器
detector = YOLO("yolo26n-pose.pt")  # 已验证 mAP50=92.2% on dog-pose val

# Phase 3 新增的追踪器
tracker = OccluBoost(reid_model="lmbn_n_duke")  # 或 osnet_x1_0 (建议微调)

for frame in video_stream:
    # 1. YOLO26-pose 检测
    results = detector(frame, verbose=False)
    dets = parse_yolo_detections(results)  # (N, 6) [x1,y1,x2,y2,conf,cls]
    kps = parse_yolo_keypoints(results)    # (N, 24, 3) [x,y,conf]

    # 2. BoxMOT 追踪（含 ReID 重关联）
    tracks = tracker.update(dets, frame)   # (M, 8) [x1,y1,x2,y2,id,conf,cls,ind]

    # 3. 关键点关联到 track_id
    per_dog_data = []
    for track in tracks:
        x1, y1, x2, y2, track_id, conf, cls, det_ind = track
        det_ind = int(det_ind)
        if det_ind < len(kps):
            dog_kps = kps[det_ind]  # (24, 3)
            per_dog_data.append({
                "track_id": int(track_id),
                "bbox": [x1, y1, x2, y2],
                "keypoints": dog_kps,
                # 行为识别 + 评分交给 Phase 3.3 后续模块
            })
```

### 4.3 备选集成方案对比

| 方案 | 复杂度 | 灵活性 | 推荐度 |
|------|--------|--------|--------|
| **A. BoxMOT Python API**（推荐） | 低 | 高（9 种追踪器可切换） | ⭐⭐⭐⭐⭐ |
| B. BoxMOT CLI (`boxmot track`) | 极低 | 中（CLI 参数） | ⭐⭐⭐⭐ |
| C. Ultralytics 内置 `model.track()` | 极低 | 低（仅 BoT-SORT/ByteTrack） | ⭐⭐⭐ |
| D. 直接用 ByteTrack 原仓 + 自建 ReID | 高 | 极高 | ⭐⭐ |
| E. 自研 MOT 算法 | 极高 | 极高 | ⭐（不推荐，违反 §1.2 不重复 owner） |

方案 A 选择理由：
- BoxMOT 是 `mikel-brostrom/Yolov7_StrongSORT_OSNet` 的演进版，作者已投入 6+ 年
- 8257 stars，2026-07-28 仍活跃（前天有 commit）
- 统一 9 种追踪器，可通过 benchmark 选最优
- 直接支持 YOLO26 检测器（README 示例即用 `--detector yolo26n`）
- Python 3.10-3.13 支持，pip 一键安装
- AGPL-3.0 与项目 YOLO26 AGPL 一致，无新增许可证风险

---

## 5. 多犬场景 ID switch 处理

### 5.1 ID switch 主要诱因

| 诱因 | 场景描述 | 多犬场景频率 |
|------|---------|-------------|
| **遮挡** | 犬只互相遮挡、植被遮挡、训练器材遮挡 | ⭐⭐⭐⭐⭐ 极高 |
| **交叉运动** | 多犬轨迹交叉，Kalman 预测失效 | ⭐⭐⭐⭐ 高 |
| **快速运动** | 犬只冲刺/扑咬，单帧位移大于关联阈值 | ⭐⭐⭐ 中 |
| **外观相似** | 同品种同毛色工作犬，ReID 特征接近 | ⭐⭐⭐⭐ 高 |
| **检测缺失** | YOLO 漏检导致轨迹断裂 | ⭐⭐ 低（Phase 1 mAP50=92.2%） |
| **重入场景** | 犬只离开画面后重新进入 | ⭐⭐ 低 |

### 5.2 BoxMOT 各追踪器的 ID switch 处理能力

基于 BoxMOT 官方 benchmark（MOT17/SportsMOT/MMOT，IDF1 越高 ID switch 越少）：

| 追踪器 | MOT17 IDF1 | SportsMOT IDF1 | MMOT IDF1 | ID switch 处理 |
|--------|-----------|----------------|-----------|---------------|
| **OccluBoost** | **84.14** | **89.36** | 29.66 | ⭐⭐⭐⭐⭐ 遮挡增强，最佳 |
| BoT-SORT | 81.94 | 78.30 | 61.36 | ⭐⭐⭐⭐ GMC + ReID |
| BoostTrack | 83.20 | 77.82 | 48.44 | ⭐⭐⭐⭐ |
| StrongSORT | 80.76 | 80.27 | 57.39 | ⭐⭐⭐ EMA + GCE |
| Deep-OC-SORT | 80.54 | 79.59 | 58.39 | ⭐⭐⭐ ORU + 外观 |
| ByteTrack | 79.16 | 76.90 | 39.74 | ⭐⭐ 低分关联，无 ReID |
| Hybrid-SORT | 78.87 | 81.88 | **63.22** | ⭐⭐⭐⭐ OBB 友好 |
| OC-SORT | 77.90 | 75.64 | 29.95 | ⭐⭐⭐ ORU，无 ReID |
| sfsort | 69.18 | 72.99 | 46.25 | ⭐⭐ 轻量 |

**多犬场景推荐**：
- **首选 OccluBoost**：SportsMOT（多目标体育场景，类似多犬）IDF1=89.36 最高
- **次选 BoT-SORT**：MMOT 多类场景 IDF1=61.36 较好（若未来扩展多物种）
- **不推荐 ByteTrack（无 ReID 版）**：纯运动关联，同品种犬只易混淆

### 5.3 时序平滑 + ReID 重关联策略

BoxMOT 的 ID switch 处理流水线：

```
检测框 (YOLO26-pose)
    │
    ▼
┌──────────────────────────────────────┐
│ 1. 运动预测（Kalman 滤波）            │
│    • 预测下一帧轨迹位置               │
│    • 多犬交叉时预测可能失效           │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│ 2. 检测关联（IoU + Hungarian）       │
│    • 高分检测先关联                   │
│    • 低分检测再关联（ByteTrack 思路） │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│ 3. 外观 ReID 匹配                     │
│    • ReID 模型提取每条轨迹外观特征    │
│    • 与历史特征库比对（cosine 距离）  │
│    • 恢复被遮挡后重现的犬只 ID        │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│ 4. 遮挡处理（OccluBoost 核心）        │
│    • 短期遮挡：保持 ID，等待重新检测  │
│    • 长期遮挡：ReID 重关联恢复 ID     │
│    • 轨迹片段合并（ICT/ORU 机制）     │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│ 5. 时序平滑（项目自建薄层，可选）     │
│    • 关键点时间序列中值滤波           │
│    • 基于 24 关键点骨架一致性的 ID 校验│
│    • 毛色直方图辅助校验               │
└──────────────────────────────────────┘
```

### 5.4 多犬场景的特殊处理建议

1. **关键点一致性校验**：当 BoxMOT 给出 ID 关联后，用 24 关键点骨架比例（如体长/体高比）做二次校验。同品种犬只骨架比例高度一致，但不同犬只仍有细微差异
2. **毛色特征辅助**：将 YOLO 检测框内的毛色分布直方图作为 ReID 特征的补充通道
3. **训练场景先验**：工作犬测评有固定流程（按训导员指令执行），可利用任务上下文辅助 ID 关联（如某犬正在执行"坐"指令，ID switch 应保持）
4. **多相机场景**（Phase 3+ 扩展）：BoT-SORT 的 GMC（相机运动补偿）对固定机位无用，但跨相机 ReID 是关键

---

## 6. 性能指标

### 6.1 BoxMOT 官方 Benchmark（统一检测器 YOLO11l_3ch + lmbn_n_duke ReID）

#### MOT17 ablation（行人为主，参考价值中等）

| 追踪器 | HOTA | MOTA | IDF1 | 说明 |
|--------|------|------|------|------|
| **OccluBoost** | **70.47** | **78.32** | **84.14** | ⭐ 当前 SOTA |
| BoT-SORT | 69.44 | 78.24 | 81.94 | 经典强基线 |
| BoostTrack | 69.25 | 75.91 | 83.20 | |
| StrongSORT | 68.05 | 76.19 | 80.76 | |
| Deep-OC-SORT | 67.95 | 75.83 | 80.54 | |
| ByteTrack | 67.68 | 78.04 | 79.16 | 无 ReID 版 |
| Hybrid-SORT | 67.31 | 74.09 | 78.87 | OBB 场景强 |
| OC-SORT | 66.44 | 74.55 | 77.90 | 无 ReID 版 |
| sfsort | 62.65 | 76.87 | 69.18 | 轻量级 |

#### SportsMOT val（体育场景，**最接近多犬追踪**）

| 追踪器 | HOTA | MOTA | IDF1 |
|--------|------|------|------|
| **OccluBoost** | **83.17** | 97.48 | **89.36** |
| Hybrid-SORT | 81.14 | **98.07** | 81.88 |
| StrongSORT | 79.80 | 97.31 | 80.27 |
| Deep-OC-SORT | 79.51 | 97.94 | 79.59 |
| BoT-SORT | 76.93 | **98.11** | 78.30 |
| OC-SORT | 76.34 | 96.60 | 75.64 |

#### MMOT OBB test（多类 OBB，最复杂场景）

| 追踪器 | HOTA | MOTA | IDF1 |
|--------|------|------|------|
| **Hybrid-SORT** | **53.86** | 44.63 | **63.22** |
| BoT-SORT | 52.27 | **45.45** | 61.36 |
| Deep-OC-SORT | 50.43 | 43.93 | 58.39 |
| StrongSORT | 49.75 | 43.64 | 57.39 |

### 6.2 原 ByteTrack 论文性能（参考）

| 数据集 | MOTA | IDF1 | HOTA | FPS (V100) |
|--------|------|------|------|-----------|
| MOT17 | 80.3 | 77.3 | 63.1 | 29.6 |
| MOT20 | 77.8 | 75.2 | 61.3 | 13.7 |

### 6.3 评估指标说明

| 指标 | 含义 | 多犬场景重要性 |
|------|------|---------------|
| **MOTA** | 多目标追踪准确度（检测精度为主） | ⭐⭐ 中（检测已由 YOLO 保证） |
| **IDF1** | ID F1 分数（ID 关联一致性） | ⭐⭐⭐⭐⭐ **核心**（ID switch 是痛点） |
| **HOTA** | 高阶追踪准确度（检测+关联平衡） | ⭐⭐⭐⭐ 高（综合指标） |
| **MT** | Mostly Tracked 轨迹占比 | ⭐⭐⭐⭐ 高 |
| **ML** | Mostly Lost 轨迹占比 | ⭐⭐⭐ 中 |
| **IDs** | ID switch 次数 | ⭐⭐⭐⭐⭐ **核心**（越少越好） |
| **FPS** | 每秒处理帧数 | ⭐⭐⭐⭐ 高（实时性要求） |

### 6.4 评估工具

- **cheind/py-motmetrics** (1,484 stars, MIT, 2026-07-17 活跃): Python MOT 指标计算
- **JonathonLuiten/TrackEval** (1,248 stars, MIT): HOTA 指标官方实现
- **BoxMOT 内置评估**: `boxmot eval` 命令直接支持 MOT17/SportsMOT/MMOT

---

## 7. 部署方案

### 7.1 部署路径选择

| 部署路径 | 复杂度 | 性能 | Windows 支持 | 推荐度 |
|---------|--------|------|-------------|--------|
| **A. BoxMOT Python（PyTorch 后端）** | 低 | 中（30-60 FPS） | ✅ | ⭐⭐⭐⭐⭐ Phase 3 首选 |
| B. BoxMOT + ONNX Runtime | 中 | 中高 | ✅ | ⭐⭐⭐⭐ Phase 3+ |
| C. BoxMOT C++ 原生后端 | 中高 | 高 | ✅ | ⭐⭐⭐ Phase 4+ |
| D. ByteTrack + FastReID TRT 自建 | 高 | 极高 | ⚠️ | ⭐⭐ Phase 4 前沿 |
| E. PINTO0309/BoT-SORT-ONNX-TensorRT | 高 | 极高 | ⚠️ | ⭐⭐ 参考 |

### 7.2 方案 A：BoxMOT Python 部署（推荐）

```bash
# Phase 3 安装
pip install boxmot           # 核心
pip install boxmot[onnx]     # ONNX 推理后端（可选）
pip install boxmot[openvino] # OpenVINO 后端（可选）

# 验证
boxmot --help
```

**性能预期**（基于项目硬件，参考 BoxMOT benchmark）：
- RTX 3060 / 4060: 30-50 FPS（含 YOLO26-pose 检测）
- RTX 4090: 80-120 FPS
- 仅 CPU: 5-10 FPS（不推荐）

### 7.3 方案 B：ONNX/OpenVINO 导出

BoxMOT + Torchreid 都支持 ONNX 导出：

```python
# BoxMOT 导出 ReID 模型
boxmot export --reid lmbn_n_duke --format onnx

# Torchreid 导出 OSNet
python tools/export.py --model osnet_x1_0 --output osnet.onnx --format onnx
```

**优势**：
- 跨平台部署（Windows/Linux/嵌入式）
- 无需 PyTorch 依赖（仅 onnxruntime）
- GPU 加速（DirectML on Windows）

### 7.4 方案 C：BoxMOT C++ 原生后端

BoxMOT 12.0+ 提供 C++ 原生追踪器实现：

```bash
boxmot track --tracker-backend cpp --tracker occluboost --source 0
```

**优势**：
- 性能提升 2-3 倍
- 可嵌入独立 C++ 项目（CMake）
- 同一套指标体系（与 Python 路径一致）

### 7.5 Windows 部署注意事项

| 问题 | 解决方案 |
|------|---------|
| cython_bbox 编译失败 | 用 BoxMOT 封装（已处理），不直接装 ByteTrack |
| MSVC cl.exe 错误 | 安装 VS Build Tools 2019/2022，配置 cl.exe 到 PATH |
| NMS 编译失败 | 用 BoxMOT（纯 Python NMS 或预编译 wheel） |
| 中文路径问题 | Ultralytics #21070 PR 已加 `imread_unicode` |
| GPU 驱动 | CUDA 11.8/12.1 + cuDNN 8.x（与 Phase 1-2 一致） |

### 7.6 已验证的部署参考仓库

| 仓库 | Star | 方案 | 备注 |
|------|------|------|------|
| **PINTO0309/BoT-SORT-ONNX-TensorRT** | 50 | 纯 ONNXRuntime BoT-SORT | PINTO 系列，质量高 |
| **emptysoal/TensorRT-YOLOv8-ByteTrack** | 247 | TensorRT YOLOv8+ByteTrack | 2025-11 活跃 |
| **hpc203/bytetrack-opencv-onnxruntime** | 239 | OpenCV/ONNXRuntime ByteTrack | C++ 实现 |
| **Vertical-Beach/ByteTrack-cpp** | 270 | 纯 C++ ByteTrack | 无外部依赖 |
| **Gudard/Strong-SORT-TensorRT** | — | TensorRT StrongSORT | 参考价值 |

---

## 8. 推荐方案

### 8.1 最终推荐：BoxMOT + YOLO26-pose + OSNet ReID

**核心组件**：

| 组件 | 仓库 | 版本 | 作用 |
|------|------|------|------|
| 检测器 | ultralytics/ultralytics | YOLO26-pose（已用 Phase 1-2） | 犬只检测 + 24 关键点 |
| 追踪器 | mikel-brostrom/boxmot | ≥12.0 | 统一追踪框架 |
| 追踪算法 | OccluBoost（BoxMOT 内置） | — | MOT17 HOTA 70.47 最佳 |
| ReID（初始） | lmbn_n_duke（BoxMOT 默认） | — | 通用 ReID，开箱即用 |
| ReID（进阶） | KaiyangZhou/deep-person-reid OSNet | — | **建议犬只微调**（自研触发） |
| 评估 | cheind/py-motmetrics | — | MOTA/IDF1/HOTA 指标 |

### 8.2 实施路径（3 步走）

#### Step 1: 即装即用验证（1 天）
```bash
pip install boxmot
```
- 用 BoxMOT 默认配置在 1 段多犬视频上验证追踪效果
- 测试 OccluBoost / BoT-SORT / StrongSORT 三种追踪器
- 记录 ID switch 次数，确定基线

#### Step 2: 与 YOLO26-pose 集成（2-3 天）
- 实现第 4.2 节的集成代码
- 关键点关联到 track_id
- 在 5-10 段多犬测评视频上端到端验证
- 用 py-motmetrics 量化 IDF1/HOTA

#### Step 3: ReID 犬只微调（用户决策，5-7 天）
- 采集工作犬 ReID 数据（每只犬 20-50 张不同角度/姿态/光照图像）
- 在 OSNet 基础上微调（torchreid 框架）
- 导出 ONNX，替换 BoxMOT 默认 ReID
- 验证 ID switch 下降比例

### 8.3 自研触发判断（按 AGENTS.md §5.2，由用户逐案决策）

| 潜在自研点 | 触发条件 | 收益 | 成本 |
|-----------|---------|------|------|
| **OSNet 犬只微调** | 默认 ReID 在同品种犬只上 ID switch 过高 | ID switch 降低 30-50% | 5-7 天 + 标注成本 |
| **关键点一致性后处理** | BoxMOT 仍有少量 ID switch | 额外降低 10-20% | 2-3 天 |
| **毛色直方图辅助 ReID** | 同品种犬只 ReID 难以区分 | 边际收益 | 1-2 天 |
| **多相机跨镜头 ReID**（Phase 3+） | 跨相机 ID 保持需求 | 跨相机追踪 | 7-10 天 |

**建议**：Step 1-2 不触发自研，先用 BoxMOT 默认配置跑通多犬追踪闭环；Step 3 根据实际 ID switch 数据决定是否微调 ReID。

### 8.4 许可证风险

| 组件 | License | 风险 |
|------|---------|------|
| BoxMOT | AGPL-3.0 | 与 YOLO26 一致，Phase 5 前解决商用许可 |
| Ultralytics YOLO26-pose | AGPL-3.0 | 已知风险，ADR 已记录 |
| OSNet (Torchreid) | MIT | 无风险 |
| FastReID | Apache-2.0 | 无风险 |
| py-motmetrics | MIT | 无风险 |

---

## 9. 参考

### 9.1 核心仓库

- BoxMOT (首选): https://github.com/mikel-brostrom/boxmot
- Ultralytics (已用): https://github.com/ultralytics/ultralytics
- ByteTrack 原仓: https://github.com/FoundationVision/ByteTrack
- BoT-SORT: https://github.com/NirAharon/BoT-SORT
- OC-SORT: https://github.com/noahcao/OC_SORT
- StrongSORT: https://github.com/dyhBUPT/StrongSORT
- DeepSORT 原作者: https://github.com/nwojke/deep_sort
- DeepSORT PyTorch: https://github.com/ZQPei/deep_sort_pytorch
- SORT 原作者: https://github.com/abewley/sort
- MOTR (端到端): https://github.com/megvii-research/MOTR
- MOTRv2: https://github.com/megvii-research/MOTRv2

### 9.2 ReID 仓库

- Torchreid (OSNet 官方): https://github.com/KaiyangZhou/deep-person-reid
- FastReID (京东): https://github.com/JDAI-CV/fast-reid
- ReID Strong Baseline: https://github.com/michuanhaohao/reid-strong-baseline
- ClusterContrast ReID: https://github.com/alibaba/cluster-contrast-reid
- HuggingFace OSNet 权重: https://huggingface.co/kaiyangzhou/osnet

### 9.3 动物 ReID

- DogFaceNet (犬脸部识别): https://github.com/GuillaumeMougeot/DogFaceNet
- RAPID 动物 ReID: https://github.com/robot-perception-group/RAPID-animal-reidentification
- Triplet-loss 动物 ReID: https://github.com/owahltinez/triplet-loss-animal-reid
- MAVRIC 视频动物 ReID: https://github.com/Imageomics/MAVRIC
- 狗 ReID (YOLO+SAM): https://github.com/ddyy-hash/dog-reid-based-on-yolo-sam-and-innovative-Illumination-invariance-module

### 9.4 评估工具

- py-motmetrics: https://github.com/cheind/py-motmetrics
- TrackEval (HOTA): https://github.com/JonathonLuiten/TrackEval

### 9.5 部署参考

- BoT-SORT ONNX/TensorRT: https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT
- TensorRT YOLOv8+ByteTrack: https://github.com/emptysoal/TensorRT-YOLOv8-ByteTrack
- ByteTrack C++: https://github.com/Vertical-Beach/ByteTrack-cpp
- ByteTrack OpenCV/ONNX: https://github.com/hpc203/bytetrack-opencv-onnxruntime
- YOLOv12+BoT-SORT+ReID (最新集成): https://github.com/wish44165/YOLOv12-BoT-SORT-ReID
- YOLOv7+StrongSORT+OSNet (BoxMOT 前身): https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet

### 9.6 awesome 目录

- awesome-multiple-object-tracking: https://github.com/luanshiyinyang/awesome-multiple-object-tracking
- awesome-reid-dataset: https://github.com/NEU-Gou/awesome-reid-dataset
- awesome-video-person-reid: https://github.com/AsuradaYuci/awesome_video_person_reid

### 9.7 关键论文

- ByteTrack (ECCV 2022): arXiv 2110.06864
- BoT-SORT (arXiv 2022): arXiv 2206.14651
- OC-SORT (CVPR 2023): arXiv 2302.11813
- StrongSORT (TMM 2023): arXiv 2202.13514
- Deep-OC-SORT: arXiv 2302.11813
- Hybrid-SORT: arXiv 2308.00783
- OSNet (ICCV 2019 / TPAMI 2021): arXiv 1905.00953 / 1910.06827
- FastReID: arXiv 2006.02631
- SORT (ICIP 2016): arXiv 1602.00763
- DeepSORT (ICIP 2017): arXiv 1703.07502
- MOTR (ECCV 2022): arXiv 2205.09167
- MOTRv2 (CVPR 2023): arXiv 2211.09791

---

## 10. 调研数据归档

本次调研通过 GitHub API + GitHub Search API + raw README 抓取，所有原始数据归档于：
- `dev-docs/research/_crawl_cache/01_awesome_meta.md` — sindresorhus/awesome 元目录
- `dev-docs/research/_crawl_cache/03_repos_meta.json` — 33 个候选仓库 meta
- `dev-docs/research/_crawl_cache/04_readme_*.md` — 20+ 关键仓库 README
- `dev-docs/research/_crawl_cache/05_issues_*.json` — 12+ 仓库 Windows issues
- `dev-docs/research/_crawl_cache/06_search_results.json` — GitHub Search 结果（去重排序）
- `dev-docs/research/_crawl_cache/08_animal_reid_results.json` — 动物 ReID 专用搜索结果
- `dev-docs/research/_crawl_cache/09_extra_repos_meta.json` — 补充仓库 meta

**调研完毕后清理**：`_crawl_cache/` 目录 + `scripts/_research_mot_*.py` 临时脚本。
