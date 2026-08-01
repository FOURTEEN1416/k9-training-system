"""ID switch 监测器（Phase 3.2c）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.2c
依据: dev-docs/research/RESEARCH_MULTI_DOG_TRACKING.md §4.4 + dev-docs/stages/phase-3.md §3.2c

设计目标:
    1. 在线检测 BoxMOT 输出的轨迹中的 ID switch 事件
    2. 评估 ID switch 严重程度，作为 ReID 微调自研触发的判断依据
    3. 评估时同时考虑 IoU + ReID 相似度，避免误判

ID switch 定义（业界共识 + 本项目约束）:
    - 真正的 ID switch: 同一物理犬只的轨迹被分裂为多个 track_id
    - 形式化: 在时空重叠区域内，两条轨迹有高 IoU 重叠但 ID 不同
    - 阈值: IoU ≥ 0.5 视为可能 ID switch（参考 MOTChallenge 评测）

自研触发条件（AGENTS.md §5.2 用户逐案决策）:
    - 默认 OSNet 不微调
    - 当 ID switch rate ≥ 0.1 (每 10 帧 ≥ 1 次) 时，建议用户考虑微调
    - 最终是否微调由用户决策

不引入兜底层:
    - 不自动调用微调流程
    - 不在 ID switch 高时降级为 IoU-only 追踪
    - 仅提供评估数据供用户决策
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.ml.tracking.types import DogTrack, MultiDogTrackingResult

logger = logging.getLogger(__name__)


# 默认 ID switch 检测阈值
DEFAULT_IOU_THRESHOLD = 0.5  # IoU ≥ 0.5 视为可能 ID switch
DEFAULT_OVERLAP_FRAMES = 3   # 至少重叠 3 帧才考虑（避免单帧噪声）
DEFAULT_SWITCH_RATE_TRIGGER = 0.1  # ID switch rate ≥ 0.1 触发自研评估


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """计算两个 bbox 的 IoU.

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        float — IoU 值 [0, 1]
    """
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union_area = area_a + area_b - inter_area

    if union_area <= 1e-12:
        return 0.0
    return inter_area / union_area


@dataclass
class IDSwitchEvent:
    """单次 ID switch 事件."""

    frame_idx: int
    track_id_a: int
    track_id_a_conf: float
    track_id_b: int
    track_id_b_conf: float
    iou: float
    reid_similarity: Optional[float] = None  # 若提供 ReID 特征，则填充

    @property
    def is_confirmed(self) -> bool:
        """确认 ID switch（有 ReID 相似度支撑且 ≥ 0.7）."""
        return self.reid_similarity is not None and self.reid_similarity >= 0.7


@dataclass
class IDSwitchReport:
    """ID switch 评估报告."""

    total_frames: int = 0
    total_events: int = 0
    confirmed_events: int = 0  # 有 ReID 相似度支撑的事件数
    events: List[IDSwitchEvent] = field(default_factory=list)
    switch_rate: float = 0.0  # 总 ID switch / 总帧数
    confirmed_switch_rate: float = 0.0  # 确认 ID switch / 总帧数
    unique_track_pairs: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def should_trigger_reid_finetune(self) -> bool:
        """是否建议触发 ReID 微调（用户最终决策）."""
        return self.switch_rate >= DEFAULT_SWITCH_RATE_TRIGGER

    def summary(self) -> Dict:
        """返回评估摘要."""
        return {
            "total_frames": self.total_frames,
            "total_events": self.total_events,
            "confirmed_events": self.confirmed_events,
            "switch_rate": round(self.switch_rate, 4),
            "confirmed_switch_rate": round(self.confirmed_switch_rate, 4),
            "unique_track_pairs": self.unique_track_pairs,
            "should_trigger_reid_finetune": self.should_trigger_reid_finetune,
            "trigger_threshold": DEFAULT_SWITCH_RATE_TRIGGER,
        }


class IDSwitchMonitor:
    """ID switch 在线监测器.

    用法:
        # 后处理评估（基于追踪结果）
        monitor = IDSwitchMonitor()
        report = monitor.evaluate(result)
        print(report.summary())

        # 带上 ReID 特征做确认
        from backend.ml.tracking.reid_extractor import ReIDExtractor
        extractor = ReIDExtractor()
        report = monitor.evaluate(result, reid_extractor=extractor)

    Args:
        iou_threshold: IoU 阈值（默认 0.5）
        min_overlap_frames: 最小重叠帧数（默认 3）
    """

    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        min_overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.min_overlap_frames = int(min_overlap_frames)

    def evaluate(
        self,
        result: MultiDogTrackingResult,
        reid_extractor: Optional["ReIDExtractor"] = None,
        video_frames: Optional[List[np.ndarray]] = None,
    ) -> IDSwitchReport:
        """评估追踪结果中的 ID switch.

        Args:
            result: 多犬追踪结果
            reid_extractor: 可选 ReID 提取器，用于确认 ID switch
            video_frames: 可选视频帧列表（用于 ReID 特征提取）；
                          若 reid_extractor 提供但 video_frames 为 None，
                          则跳过 ReID 确认

        Returns:
            IDSwitchReport
        """
        report = IDSwitchReport(total_frames=result.total_frames)

        if result.num_dogs < 2:
            # 单犬场景无 ID switch 可能
            return report

        # 1. 构建 frame_idx -> [(track_id, bbox, conf)] 索引
        frame_index: Dict[int, List[Tuple[int, np.ndarray, float]]] = {}
        for tid, track in result.tracks.items():
            for f in track.frames:
                frame_index.setdefault(f.frame_idx, []).append(
                    (tid, f.bbox, f.conf)
                )

        # 2. 逐帧检测 ID switch（同帧多犬 IoU ≥ 阈值）
        seen_pairs: set = set()
        for frame_idx in sorted(frame_index.keys()):
            entries = frame_index[frame_idx]
            if len(entries) < 2:
                continue

            # 两两比较
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    tid_a, bbox_a, conf_a = entries[i]
                    tid_b, bbox_b, conf_b = entries[j]
                    if tid_a == tid_b:
                        continue

                    iou = _bbox_iou(bbox_a, bbox_b)
                    if iou < self.iou_threshold:
                        continue

                    # 检查是否在 min_overlap_frames 帧内持续重叠
                    pair = tuple(sorted([tid_a, tid_b]))
                    if pair in seen_pairs:
                        continue  # 已记录

                    overlap_count = self._count_overlap_frames(
                        result.tracks[tid_a],
                        result.tracks[tid_b],
                        self.iou_threshold,
                    )
                    if overlap_count < self.min_overlap_frames:
                        continue

                    seen_pairs.add(pair)
                    event = IDSwitchEvent(
                        frame_idx=frame_idx,
                        track_id_a=int(tid_a),
                        track_id_a_conf=float(conf_a),
                        track_id_b=int(tid_b),
                        track_id_b_conf=float(conf_b),
                        iou=float(iou),
                    )
                    report.events.append(event)
                    report.total_events += 1
                    report.unique_track_pairs.append(pair)

        # 3. ReID 确认（可选）
        if reid_extractor is not None and video_frames is not None:
            for event in report.events:
                sim = self._compute_event_reid_similarity(
                    event, result, video_frames, reid_extractor
                )
                event.reid_similarity = sim
                if event.is_confirmed:
                    report.confirmed_events += 1

        # 4. 计算速率
        if report.total_frames > 0:
            report.switch_rate = report.total_events / report.total_frames
            report.confirmed_switch_rate = (
                report.confirmed_events / report.total_frames
            )

        return report

    @staticmethod
    def _count_overlap_frames(
        track_a: DogTrack,
        track_b: DogTrack,
        iou_threshold: float,
    ) -> int:
        """统计两条轨迹的高 IoU 重叠帧数."""
        # 构建 frame_idx -> bbox 索引
        bboxes_a: Dict[int, np.ndarray] = {f.frame_idx: f.bbox for f in track_a.frames}
        bboxes_b: Dict[int, np.ndarray] = {f.frame_idx: f.bbox for f in track_b.frames}

        common_frames = set(bboxes_a.keys()) & set(bboxes_b.keys())
        overlap_count = 0
        for fidx in common_frames:
            iou = _bbox_iou(bboxes_a[fidx], bboxes_b[fidx])
            if iou >= iou_threshold:
                overlap_count += 1
        return overlap_count

    @staticmethod
    def _compute_event_reid_similarity(
        event: IDSwitchEvent,
        result: MultiDogTrackingResult,
        video_frames: List[np.ndarray],
        reid_extractor: "ReIDExtractor",
    ) -> Optional[float]:
        """计算 ID switch 事件中两条轨迹的 ReID 相似度.

        取事件帧附近的裁剪，分别提取 ReID 特征，计算余弦相似度。
        """
        from backend.ml.tracking.reid_extractor import cosine_similarity

        if event.frame_idx >= len(video_frames):
            return None

        frame = video_frames[event.frame_idx]
        track_a = result.tracks.get(event.track_id_a)
        track_b = result.tracks.get(event.track_id_b)
        if track_a is None or track_b is None:
            return None

        # 从事件帧附近 ±5 帧取裁剪
        window = 5
        crops_a: List[np.ndarray] = []
        crops_b: List[np.ndarray] = []
        for offset in range(-window, window + 1):
            fidx = event.frame_idx + offset
            if fidx < 0 or fidx >= len(video_frames):
                continue
            f = video_frames[fidx]
            # 找到该帧对应犬的 bbox
            for fr in track_a.frames:
                if fr.frame_idx == fidx:
                    x1, y1, x2, y2 = fr.bbox.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(f.shape[1], x2), min(f.shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        crops_a.append(f[y1:y2, x1:x2].copy())
                    break
            for fr in track_b.frames:
                if fr.frame_idx == fidx:
                    x1, y1, x2, y2 = fr.bbox.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(f.shape[1], x2), min(f.shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        crops_b.append(f[y1:y2, x1:x2].copy())
                    break

        if not crops_a or not crops_b:
            return None

        try:
            feats_a = reid_extractor.extract_from_crops(crops_a)  # (N_a, dim)
            feats_b = reid_extractor.extract_from_crops(crops_b)  # (N_b, dim)
            if feats_a.size == 0 or feats_b.size == 0:
                return None
            # 轨迹级聚合
            track_feat_a = reid_extractor.aggregate_track(feats_a)  # (dim,)
            track_feat_b = reid_extractor.aggregate_track(feats_b)  # (dim,)
            sim = float(cosine_similarity(
                track_feat_a.reshape(1, -1), track_feat_b.reshape(1, -1)
            )[0, 0])
            return sim
        except Exception as e:
            logger.warning(f"[IDSwitchMonitor] ReID 相似度计算失败: {e}")
            return None


__all__ = [
    "IDSwitchEvent",
    "IDSwitchReport",
    "IDSwitchMonitor",
    "DEFAULT_IOU_THRESHOLD",
    "DEFAULT_OVERLAP_FRAMES",
    "DEFAULT_SWITCH_RATE_TRIGGER",
]
