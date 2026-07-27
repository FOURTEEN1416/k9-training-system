"""科目规则引擎 P0 8 类行为识别.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.2
依据: dev-docs/stages/phase-1.md §1.2

8 类行为（P0 基础）:
    - sit (坐): 后腿折叠 + 前腿直立
    - down (卧): 躯干低 + 四肢折叠
    - stand (立): 四腿伸直 + 躯干高
    - heel (随行): 持续移动 + 站立姿态（Phase 1.5 加 person 检测后完善）
    - sit_up (坐立): 坐姿 + 头部抬起
    - stay (停留): 关键点位置稳定 ≥ N 帧
    - bark (叫): nose-chin 距离周期性变化
    - bite (咬): 嘴部动作 + 前爪快速移动

输入:
    keypoints_sequence: np.ndarray, shape=(T, 24, 3), 每行 [x, y, conf]
    boxes: optional list[np.ndarray], 每帧犬检测框 [x1, y1, x2, y2]（用于 heel）
    fps: float, 视频帧率（用于 stay 时长判断）

输出:
    list[BehaviorEpisode]: [(behavior, start_frame, end_frame, confidence), ...]

坐标系统:
    图像坐标，y 轴向下（y 大 = 位置低/接近地面）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.ml.behavior.constants import (
    ALL_KNEES, ALL_PAWS, BARK, BITE, CHIN, DOWN,
    FRONT_ELBOWS, FRONT_KNEES, FRONT_PAWS,
    HEEL, NOSE, P0_BEHAVIORS, REAR_ELBOWS, REAR_KNEES, REAR_PAWS,
    SIT, SIT_UP, STAND, STAY, WITHERS,
)

# 默认阈值（可被 YAML 评分卡覆盖，Phase 1.4 集成）
DEFAULT_CONF_THRESHOLD = 0.3      # 关键点置信度低于此值视为不可靠
DEFAULT_FOLD_THRESHOLD = 30.0     # 腿折叠：|paw.y - knee.y| < 此值视为折叠（像素）
DEFAULT_EXTEND_THRESHOLD = 20.0   # 腿伸直：paw.y - elbow.y > 此值视为伸直（像素）
DEFAULT_MID_THRESHOLD = 10.0      # 腿伸直中段：knee.y - elbow.y > 此值
DEFAULT_GROUND_RATIO_LOW = 0.75   # 卧：withers.y > ground_y * 此值
DEFAULT_GROUND_RATIO_HIGH = 0.50  # 立：withers.y < ground_y * 此值
DEFAULT_STAY_FRAMES = 15          # 停留：连续 N 帧不动视为停留（约 0.5s @ 30fps）
DEFAULT_STAY_MOTION = 5.0         # 停留：帧间关键点位移 < 此值视为不动（像素）
DEFAULT_HEEL_MOTION = 3.0         # 随行：帧间 withers.x 变化 > 此值视为移动（像素）
DEFAULT_BARK_STD = 3.0            # 叫：nose-chin 距离序列标准差 > 此值
DEFAULT_BITE_MOTION = 20.0        # 咬：前爪帧间位移 > 此值
DEFAULT_BITE_MOUTH_VAR = 5.0      # 咬：nose-chin 距离方差 > 此值


@dataclass
class BehaviorEpisode:
    """单次行为 episodes。"""
    behavior: str          # 行为类别（SIT/DOWN/...）
    start_frame: int       # 起始帧（含）
    end_frame: int         # 结束帧（含）
    confidence: float      # 置信度 [0, 1]
    metadata: dict = None  # 额外信息（如规则命中细节）

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> dict:
        return {
            "behavior": self.behavior,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class RuleEngine:
    """基于几何规则的 8 类行为识别引擎。

    用法:
        engine = RuleEngine()
        episodes = engine.recognize(keypoints_sequence, fps=30.0)
        for ep in episodes:
            print(f"{ep.behavior}: 帧 {ep.start_frame}-{ep.end_frame} conf={ep.confidence:.2f}")
    """

    def __init__(
        self,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        fold_threshold: float = DEFAULT_FOLD_THRESHOLD,
        extend_threshold: float = DEFAULT_EXTEND_THRESHOLD,
        mid_threshold: float = DEFAULT_MID_THRESHOLD,
        ground_ratio_low: float = DEFAULT_GROUND_RATIO_LOW,
        ground_ratio_high: float = DEFAULT_GROUND_RATIO_HIGH,
        stay_frames: int = DEFAULT_STAY_FRAMES,
        stay_motion: float = DEFAULT_STAY_MOTION,
        heel_motion: float = DEFAULT_HEEL_MOTION,
        bark_std: float = DEFAULT_BARK_STD,
        bite_motion: float = DEFAULT_BITE_MOTION,
        bite_mouth_var: float = DEFAULT_BITE_MOUTH_VAR,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.fold_threshold = fold_threshold
        self.extend_threshold = extend_threshold
        self.mid_threshold = mid_threshold
        self.ground_ratio_low = ground_ratio_low
        self.ground_ratio_high = ground_ratio_high
        self.stay_frames = stay_frames
        self.stay_motion = stay_motion
        self.heel_motion = heel_motion
        self.bark_std = bark_std
        self.bite_motion = bite_motion
        self.bite_mouth_var = bite_mouth_var

    def recognize(
        self,
        keypoints_sequence: np.ndarray,
        boxes: Optional[list] = None,
        fps: float = 30.0,
    ) -> list[BehaviorEpisode]:
        """识别 8 类行为。

        Args:
            keypoints_sequence: shape=(T, 24, 3), [x, y, conf]
            boxes: 每帧犬检测框 [x1,y1,x2,y2]（可选，用于 heel）
            fps: 帧率

        Returns:
            list[BehaviorEpisode]
        """
        kpts = np.asarray(keypoints_sequence, dtype=np.float32)
        if kpts.ndim != 3 or kpts.shape[1] != 24 or kpts.shape[2] != 3:
            raise ValueError(
                f"keypoints_sequence shape 应为 (T, 24, 3)，实际 {kpts.shape}"
            )
        T = kpts.shape[0]
        if T == 0:
            return []

        episodes: list[BehaviorEpisode] = []

        # 1. 逐帧姿态分类（sit / down / stand）
        posture_labels = self._classify_posture_per_frame(kpts)

        # 2. 聚合连续相同姿态 → episodes
        episodes.extend(self._aggregate_posture(posture_labels, kpts))

        # 3. sit_up: 坐姿 + 头部抬起
        episodes.extend(self._detect_sit_up(kpts, posture_labels))

        # 4. stay: 关键点稳定 ≥ N 帧
        episodes.extend(self._detect_stay(kpts))

        # 5. heel: 持续移动 + 站立（简化版，Phase 1.5 加 person 检测）
        episodes.extend(self._detect_heel(kpts, posture_labels, boxes))

        # 6. bark: nose-chin 周期性变化
        episodes.extend(self._detect_bark(kpts))

        # 7. bite: 嘴部动作 + 前爪快速移动
        episodes.extend(self._detect_bite(kpts))

        # 按起始帧排序
        episodes.sort(key=lambda e: (e.start_frame, e.behavior))
        return episodes

    # ===== 内部方法 =====

    def _valid_mask(self, kpts_frame: np.ndarray) -> np.ndarray:
        """单帧关键点置信度 mask（conf >= threshold）。"""
        return kpts_frame[:, 2] >= self.conf_threshold

    def _estimate_ground_y(self, kpts_frame: np.ndarray) -> float:
        """估计地面 y 坐标（取所有有效关键点的最大 y）。"""
        mask = self._valid_mask(kpts_frame)
        if not mask.any():
            return 0.0
        return float(kpts_frame[mask, 1].max())

    def _mean_y(self, kpts_frame: np.ndarray, indices: tuple) -> Optional[float]:
        """取指定关键点的平均 y（仅有效点）。"""
        mask = self._valid_mask(kpts_frame)
        valid_indices = [i for i in indices if mask[i]]
        if not valid_indices:
            return None
        return float(np.mean(kpts_frame[valid_indices, 1]))

    def _mean_diff_y(self, kpts_frame: np.ndarray, idx_a: tuple, idx_b: tuple) -> Optional[float]:
        """两组关键点的平均 y 差（a.y - b.y）。"""
        mask = self._valid_mask(kpts_frame)
        valid_pairs = [(i, j) for i, j in zip(idx_a, idx_b) if mask[i] and mask[j]]
        if not valid_pairs:
            return None
        diffs = [kpts_frame[i, 1] - kpts_frame[j, 1] for i, j in valid_pairs]
        return float(np.mean(diffs))

    def _mean_abs_diff_y(self, kpts_frame: np.ndarray, idx_a: tuple, idx_b: tuple) -> Optional[float]:
        """两组关键点的平均 |y 差|。"""
        mask = self._valid_mask(kpts_frame)
        valid_pairs = [(i, j) for i, j in zip(idx_a, idx_b) if mask[i] and mask[j]]
        if not valid_pairs:
            return None
        diffs = [abs(kpts_frame[i, 1] - kpts_frame[j, 1]) for i, j in valid_pairs]
        return float(np.mean(diffs))

    def _classify_posture_per_frame(self, kpts: np.ndarray) -> list[str]:
        """逐帧分类姿态: sit / down / stand / unknown。"""
        T = kpts.shape[0]
        labels = ["unknown"] * T
        for t in range(T):
            frame = kpts[t]
            labels[t] = self._classify_single_posture(frame)
        return labels

    def _classify_single_posture(self, frame: np.ndarray) -> str:
        """单帧姿态分类。"""
        ground_y = self._estimate_ground_y(frame)
        if ground_y <= 0:
            return "unknown"

        # 后腿折叠度：rear_paw.y - rear_knee.y 的绝对差
        rear_fold = self._mean_abs_diff_y(frame, REAR_PAWS, REAR_KNEES)
        # 前腿伸直度：front_paw.y - front_elbow.y（正值表示 paw 在 elbow 下方）
        front_extend = self._mean_diff_y(frame, FRONT_PAWS, FRONT_ELBOWS)
        # 四肢折叠度
        all_fold = self._mean_abs_diff_y(frame, ALL_PAWS, ALL_KNEES)
        # withers 高度
        withers_y = self._mean_y(frame, (WITHERS,))

        if rear_fold is None or front_extend is None or all_fold is None or withers_y is None:
            return "unknown"

        # 卧：躯干低 + 四肢折叠
        if withers_y > ground_y * self.ground_ratio_low and all_fold < self.fold_threshold:
            return DOWN

        # 坐：后腿折叠 + 前腿伸直
        if rear_fold < self.fold_threshold and front_extend > self.extend_threshold:
            return SIT

        # 立：四腿伸直 + 躯干高
        front_mid = self._mean_diff_y(frame, FRONT_KNEES, FRONT_ELBOWS)
        rear_extend = self._mean_diff_y(frame, REAR_PAWS, REAR_ELBOWS)
        if (front_extend > self.extend_threshold
                and rear_extend is not None and rear_extend > self.extend_threshold
                and withers_y < ground_y * self.ground_ratio_high):
            return STAND

        return "unknown"

    def _aggregate_posture(
        self, labels: list[str], kpts: np.ndarray
    ) -> list[BehaviorEpisode]:
        """聚合连续相同姿态 → episodes（仅 sit/down/stand）。"""
        episodes = []
        T = len(labels)
        i = 0
        min_frames = 3  # 至少 3 帧才算一个 episode
        while i < T:
            label = labels[i]
            if label not in (SIT, DOWN, STAND):
                i += 1
                continue
            j = i
            while j < T and labels[j] == label:
                j += 1
            if j - i >= min_frames:
                # 置信度：该区间内有效关键点比例
                conf = self._episode_confidence(kpts[i:j])
                episodes.append(BehaviorEpisode(
                    behavior=label,
                    start_frame=i,
                    end_frame=j - 1,
                    confidence=conf,
                    metadata={"duration_frames": j - i},
                ))
            i = j
        return episodes

    def _episode_confidence(self, kpts_clip: np.ndarray) -> float:
        """计算 episode 置信度（有效关键点比例）。"""
        mask = kpts_clip[:, :, 2] >= self.conf_threshold
        return float(mask.mean())

    def _detect_sit_up(
        self, kpts: np.ndarray, posture_labels: list[str]
    ) -> list[BehaviorEpisode]:
        """坐立：坐姿 + 头部抬起（nose.y < withers.y）。"""
        episodes = []
        T = kpts.shape[0]
        i = 0
        min_frames = 3
        while i < T:
            if posture_labels[i] != SIT:
                i += 1
                continue
            j = i
            while j < T and posture_labels[j] == SIT:
                j += 1
            # 在坐姿区间内检查头部是否抬起
            head_up_frames = 0
            for t in range(i, j):
                frame = kpts[t]
                mask = self._valid_mask(frame)
                if mask[NOSE] and mask[WITHERS]:
                    if frame[NOSE, 1] < frame[WITHERS, 1]:  # nose 在 withers 上方
                        head_up_frames += 1
            if head_up_frames >= min_frames:
                conf = head_up_frames / (j - i)
                episodes.append(BehaviorEpisode(
                    behavior=SIT_UP,
                    start_frame=i,
                    end_frame=j - 1,
                    confidence=conf,
                    metadata={"head_up_frames": head_up_frames, "sit_frames": j - i},
                ))
            i = j
        return episodes

    def _detect_stay(self, kpts: np.ndarray) -> list[BehaviorEpisode]:
        """停留：连续 N 帧关键点位移 < stay_motion。"""
        T = kpts.shape[0]
        if T < self.stay_frames:
            return []
        episodes = []
        # 计算帧间位移（仅有效关键点）
        motion = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            mask = (kpts[t, :, 2] >= self.conf_threshold) & \
                   (kpts[t-1, :, 2] >= self.conf_threshold)
            if mask.any():
                dy = kpts[t, mask, 0] - kpts[t-1, mask, 0]
                dx = kpts[t, mask, 1] - kpts[t-1, mask, 1]
                motion[t] = float(np.sqrt(dx*dx + dy*dy).mean())

        # 找连续 stay_frames 帧位移都 < stay_motion 的区间
        i = 0
        while i <= T - self.stay_frames:
            window = motion[i+1:i+self.stay_frames]
            if (window < self.stay_motion).all():
                # 扩展区间
                end = i + self.stay_frames
                while end < T and motion[end] < self.stay_motion:
                    end += 1
                episodes.append(BehaviorEpisode(
                    behavior=STAY,
                    start_frame=i,
                    end_frame=end - 1,
                    confidence=1.0 - float(window.mean()) / max(self.stay_motion, 1e-6),
                    metadata={"duration_frames": end - i, "mean_motion": float(window.mean())},
                ))
                i = end
            else:
                i += 1
        return episodes

    def _detect_heel(
        self,
        kpts: np.ndarray,
        posture_labels: list[str],
        boxes: Optional[list] = None,
    ) -> list[BehaviorEpisode]:
        """随行：持续移动 + 站立姿态（简化版）。

        Phase 1.5 集成 person 检测后将完善：犬与 person 相对位置稳定 + 同步移动。
        """
        T = kpts.shape[0]
        if T < self.stay_frames:
            return []

        # 计算帧间 withers.x 位移
        motion_x = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            if (kpts[t, WITHERS, 2] >= self.conf_threshold and
                    kpts[t-1, WITHERS, 2] >= self.conf_threshold):
                motion_x[t] = abs(kpts[t, WITHERS, 0] - kpts[t-1, WITHERS, 0])

        episodes = []
        i = 0
        min_frames = self.stay_frames
        while i <= T - min_frames:
            # 检查区间内是否持续移动 + 站立
            window = motion_x[i+1:i+min_frames]
            stand_ratio = sum(1 for t in range(i, i+min_frames)
                              if posture_labels[t] == STAND) / min_frames
            if (window > self.heel_motion).all() and stand_ratio > 0.5:
                # 扩展区间
                end = i + min_frames
                while (end < T and motion_x[end] > self.heel_motion
                       and posture_labels[end] == STAND):
                    end += 1
                episodes.append(BehaviorEpisode(
                    behavior=HEEL,
                    start_frame=i,
                    end_frame=end - 1,
                    confidence=float(window.mean() / (self.heel_motion * 2)),
                    metadata={
                        "duration_frames": end - i,
                        "mean_motion_x": float(window.mean()),
                        "stand_ratio": stand_ratio,
                        "note": "简化版，Phase 1.5 加 person 检测后完善",
                    },
                ))
                i = end
            else:
                i += 1
        return episodes

    def _detect_bark(self, kpts: np.ndarray) -> list[BehaviorEpisode]:
        """叫：nose-chin 距离序列标准差 > bark_std。"""
        T = kpts.shape[0]
        if T < self.stay_frames:
            return []

        # 计算 nose-chin 距离序列
        dist = np.zeros(T, dtype=np.float32)
        for t in range(T):
            if (kpts[t, NOSE, 2] >= self.conf_threshold and
                    kpts[t, CHIN, 2] >= self.conf_threshold):
                dx = kpts[t, NOSE, 0] - kpts[t, CHIN, 0]
                dy = kpts[t, NOSE, 1] - kpts[t, CHIN, 1]
                dist[t] = float(np.sqrt(dx*dx + dy*dy))

        # 滑动窗口检测高方差区间
        episodes = []
        window = self.stay_frames
        i = 0
        while i <= T - window:
            clip = dist[i:i+window]
            if clip.std() > self.bark_std:
                # 扩展区间
                end = i + window
                while end < T and dist[end-window:end].std() > self.bark_std:
                    end += 1
                episodes.append(BehaviorEpisode(
                    behavior=BARK,
                    start_frame=i,
                    end_frame=end - 1,
                    confidence=min(1.0, float(clip.std()) / (self.bark_std * 2)),
                    metadata={
                        "duration_frames": end - i,
                        "dist_std": float(clip.std()),
                        "dist_mean": float(clip.mean()),
                    },
                ))
                i = end
            else:
                i += 1
        return episodes

    def _detect_bite(self, kpts: np.ndarray) -> list[BehaviorEpisode]:
        """咬：嘴部动作 + 前爪快速移动。"""
        T = kpts.shape[0]
        if T < self.stay_frames:
            return []

        # nose-chin 距离变化
        mouth_dist = np.zeros(T, dtype=np.float32)
        for t in range(T):
            if (kpts[t, NOSE, 2] >= self.conf_threshold and
                    kpts[t, CHIN, 2] >= self.conf_threshold):
                dx = kpts[t, NOSE, 0] - kpts[t, CHIN, 0]
                dy = kpts[t, NOSE, 1] - kpts[t, CHIN, 1]
                mouth_dist[t] = float(np.sqrt(dx*dx + dy*dy))

        # 前爪速度
        paw_motion = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            mask = (kpts[t, FRONT_PAWS, 2] >= self.conf_threshold) & \
                   (kpts[t-1, FRONT_PAWS, 2] >= self.conf_threshold)
            if mask.any():
                dx = kpts[t, FRONT_PAWS, 0] - kpts[t-1, FRONT_PAWS, 0]
                dy = kpts[t, FRONT_PAWS, 1] - kpts[t-1, FRONT_PAWS, 1]
                paw_motion[t] = float(np.sqrt(dx[mask]**2 + dy[mask]**2).mean())

        episodes = []
        window = self.stay_frames
        i = 0
        while i <= T - window:
            mouth_clip = mouth_dist[i:i+window]
            paw_clip = paw_motion[i+1:i+window]
            # 嘴部活跃 + 前爪快速移动
            if (mouth_clip.var() > self.bite_mouth_var and
                    paw_clip.mean() > self.bite_motion):
                end = i + window
                while (end < T and
                       mouth_dist[end-window:end].var() > self.bite_mouth_var and
                       paw_motion[end] > self.bite_motion):
                    end += 1
                episodes.append(BehaviorEpisode(
                    behavior=BITE,
                    start_frame=i,
                    end_frame=end - 1,
                    confidence=min(1.0, float(paw_clip.mean()) / (self.bite_motion * 2)),
                    metadata={
                        "duration_frames": end - i,
                        "mouth_var": float(mouth_clip.var()),
                        "paw_mean_motion": float(paw_clip.mean()),
                    },
                ))
                i = end
            else:
                i += 1
        return episodes


def recognize_behaviors(
    keypoints_sequence: np.ndarray,
    boxes: Optional[list] = None,
    fps: float = 30.0,
    **kwargs,
) -> list[BehaviorEpisode]:
    """便捷函数：识别行为。"""
    engine = RuleEngine(**kwargs)
    return engine.recognize(keypoints_sequence, boxes=boxes, fps=fps)
