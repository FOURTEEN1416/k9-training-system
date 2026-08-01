"""多犬追踪数据类型.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.2b
依据: dev-docs/research/RESEARCH_MULTI_DOG_TRACKING.md §4.1

数据流:
    视频帧 → YOLO26-pose 检测 → BoxMOT 追踪 → 每犬独立序列

核心类型:
    DogTrackFrame: 单帧单犬的追踪结果（bbox + keypoints + track_id）
    DogTrack: 单犬的完整轨迹（track_id + 帧序列）
    MultiDogTrackingResult: 整段视频的多犬追踪结果
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from backend.ml.behavior.constants import NUM_KEYPOINTS


@dataclass
class DogTrackFrame:
    """单帧单犬的追踪结果.

    Attributes:
        frame_idx: 帧索引
        track_id: 追踪 ID（BoxMOT 分配，跨帧唯一）
        bbox: 检测框 [x1, y1, x2, y2]（像素坐标）
        keypoints: 24 关键点 (24, 3) [x, y, conf]
        conf: 检测置信度 [0, 1]
        det_ind: YOLO 检测索引（用于关键点关联调试）
    """
    frame_idx: int
    track_id: int
    bbox: np.ndarray  # (4,) [x1, y1, x2, y2]
    keypoints: np.ndarray  # (24, 3)
    conf: float = 0.0
    det_ind: int = -1

    def __post_init__(self) -> None:
        if self.bbox is not None and not isinstance(self.bbox, np.ndarray):
            self.bbox = np.asarray(self.bbox, dtype=np.float32)
        if self.keypoints is not None and not isinstance(self.keypoints, np.ndarray):
            self.keypoints = np.asarray(self.keypoints, dtype=np.float32)
        if self.keypoints is not None and self.keypoints.shape != (NUM_KEYPOINTS, 3):
            # 兜底：pad 或截断
            if self.keypoints.shape[0] < NUM_KEYPOINTS:
                pad = np.zeros((NUM_KEYPOINTS - self.keypoints.shape[0], 3), dtype=np.float32)
                self.keypoints = np.concatenate([self.keypoints, pad], axis=0)
            else:
                self.keypoints = self.keypoints[:NUM_KEYPOINTS]


@dataclass
class DogTrack:
    """单犬的完整轨迹.

    Attributes:
        track_id: 追踪 ID
        frames: 该犬出现的所有帧数据
    """
    track_id: int
    frames: List[DogTrackFrame] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def frame_indices(self) -> List[int]:
        """该犬出现的帧索引列表。"""
        return [f.frame_idx for f in self.frames]

    def get_keypoints_sequence(self, total_frames: Optional[int] = None) -> np.ndarray:
        """获取该犬的关键点序列.

        Args:
            total_frames: 若指定，未出现的帧用 0 填充

        Returns:
            np.ndarray, shape=(T, 24, 3)
        """
        if total_frames is None:
            # 仅返回出现的帧
            return np.stack([f.keypoints for f in self.frames], axis=0)

        # 填充模式：未出现的帧用 0
        seq = np.zeros((total_frames, NUM_KEYPOINTS, 3), dtype=np.float32)
        for f in self.frames:
            if 0 <= f.frame_idx < total_frames:
                seq[f.frame_idx] = f.keypoints
        return seq

    def get_bbox_sequence(self, total_frames: Optional[int] = None) -> np.ndarray:
        """获取该犬的检测框序列.

        Args:
            total_frames: 若指定，未出现的帧用 0 填充

        Returns:
            np.ndarray, shape=(T, 4) [x1, y1, x2, y2]
        """
        if total_frames is None:
            return np.stack([f.bbox for f in self.frames], axis=0)

        seq = np.zeros((total_frames, 4), dtype=np.float32)
        for f in self.frames:
            if 0 <= f.frame_idx < total_frames:
                seq[f.frame_idx] = f.bbox
        return seq


@dataclass
class MultiDogTrackingResult:
    """整段视频的多犬追踪结果.

    Attributes:
        tracks: {track_id: DogTrack} 每犬独立轨迹
        total_frames: 视频总帧数
        fps: 视频帧率
        meta: 元数据（视频路径、模型路径等）
    """
    tracks: Dict[int, DogTrack] = field(default_factory=dict)
    total_frames: int = 0
    fps: float = 30.0
    meta: dict = field(default_factory=dict)

    @property
    def num_dogs(self) -> int:
        """追踪到的不同犬只数量。"""
        return len(self.tracks)

    @property
    def track_ids(self) -> List[int]:
        """所有追踪 ID 列表。"""
        return sorted(self.tracks.keys())

    def get_track(self, track_id: int) -> Optional[DogTrack]:
        """获取指定 ID 的轨迹。"""
        return self.tracks.get(track_id)

    def get_all_keypoints_sequences(self) -> Dict[int, np.ndarray]:
        """获取所有犬只的关键点序列（仅出现帧）.

        Returns:
            {track_id: np.ndarray (T_i, 24, 3)}
        """
        return {tid: track.get_keypoints_sequence() for tid, track in self.tracks.items()}

    def get_padded_keypoints_sequences(self) -> Dict[int, np.ndarray]:
        """获取所有犬只的关键点序列（填充到 total_frames）.

        Returns:
            {track_id: np.ndarray (total_frames, 24, 3)}
        """
        return {
            tid: track.get_keypoints_sequence(self.total_frames)
            for tid, track in self.tracks.items()
        }

    def summary(self) -> str:
        """结果摘要（调试用）."""
        lines = [
            f"MultiDogTrackingResult",
            f"  Total frames: {self.total_frames}",
            f"  FPS: {self.fps}",
            f"  Unique dogs (track_ids): {self.num_dogs}",
            f"  Track IDs: {self.track_ids}",
        ]
        for tid in self.track_ids:
            track = self.tracks[tid]
            coverage = track.num_frames / max(self.total_frames, 1) * 100
            lines.append(
                f"  - track_id={tid}: {track.num_frames} frames "
                f"({coverage:.1f}% coverage), frames {track.frame_indices[0]}-{track.frame_indices[-1]}"
                if track.num_frames > 0 else f"  - track_id={tid}: empty"
            )
        return "\n".join(lines)
