"""多犬追踪模块（Phase 3.2）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.2
依据: dev-docs/stages/phase-3.md §3.2 + dev-docs/research/RESEARCH_MULTI_DOG_TRACKING.md

子模块:
    - types: 多犬追踪数据类型（DogTrackFrame / MultiDogTrackingResult）
    - multi_dog_tracker: BoxMOT OccluBoost 集成 + YOLO26-pose 关键点关联
    - reid_extractor: ReID 犬只身份关联（3.2c）— 特征提取 + 身份 gallery + 微调数据收集
    - id_switch_monitor: ID switch 监测（3.2c）— 自研触发的判断依据

架构:
    YOLO26-pose 检测 (N, [xyxy, conf, cls] + 24 kpts)
        │
        ▼
    BoxMOT OccluBoost 追踪（含 ReID 重关联，with_reid=True）
        │ tracks (M, [xyxy, id, conf, cls, det_ind])
        ▼
    关键点关联层（项目自建薄层）
        │ 通过 det_ind 找回对应 YOLO 检测的关键点
        ▼
    每犬独立序列: {track_id: [(bbox, keypoints_24), ...]}
        │
        ▼
    ReIDExtractor（3.2c）— 从裁剪提取 OSNet embedding
        │
        ▼
    DogIdentityGallery（3.2c）— 跨视频身份匹配 + 微调数据收集

注意:
    - BoxMOT 19.0.0 (AGPL-3.0，与 YOLO26 一致)
    - 默认追踪器: OccluBoost (MOT17 IDF1=84.14, SportsMOT IDF1=89.36)
    - 默认 ReID: OSNet x1.0 MSMT17（犬只微调为 3.2c 自研触发项）
"""
from backend.ml.tracking.types import (
    DogTrackFrame,
    MultiDogTrackingResult,
    DogTrack,
)
from backend.ml.tracking.multi_dog_tracker import (
    MultiDogTracker,
    TrackerBackend,
)
from backend.ml.tracking.reid_extractor import (
    ReIDExtractor,
    DogIdentityGallery,
    cosine_similarity,
)
from backend.ml.tracking.id_switch_monitor import (
    IDSwitchEvent,
    IDSwitchReport,
    IDSwitchMonitor,
)
from backend.ml.tracking.reid_finetune_dataset import (
    CanineReIDSample,
    CanineReIDDataset,
)

__all__ = [
    "DogTrackFrame",
    "MultiDogTrackingResult",
    "DogTrack",
    "MultiDogTracker",
    "TrackerBackend",
    "ReIDExtractor",
    "DogIdentityGallery",
    "cosine_similarity",
    "IDSwitchEvent",
    "IDSwitchReport",
    "IDSwitchMonitor",
    "CanineReIDSample",
    "CanineReIDDataset",
]
