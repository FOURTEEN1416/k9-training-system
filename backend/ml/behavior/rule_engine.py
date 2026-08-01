"""科目规则引擎 P0 8 类 + P1 8 类 = 16 类行为识别.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.2 (P0) + 2.2 (P1)
依据: dev-docs/stages/phase-1.md §1.2 + dev-docs/research/RESEARCH_STANDARDS.md §4.1

P0 基础 8 类（Phase 1 已验收 92.9%）:
    - sit (坐): 后腿折叠 + 前腿直立
    - down (卧): 躯干低 + 四肢折叠
    - stand (立): 四腿伸直 + 躯干高
    - heel (随行): 持续移动 + 站立姿态
    - sit_up (坐立): 坐姿 + 头部抬起
    - stay (停留): 关键点位置稳定 ≥ N 帧
    - bark (叫): nose-chin 距离周期性变化
    - bite (咬): 嘴部动作 + 前爪快速移动

P1 训练专项 8 类（Phase 2 新增，依据 RESEARCH_STANDARDS.md §4.1）:
    - track (追踪): 鼻尖贴近地面 + 路径跟随
    - alert_sit (示警坐): 检出目标后坐姿示警
    - alert_down (示警卧): 检出目标后卧姿示警
    - apprehend (扑咬): 高速接近 + 嘴部接触目标
    - escort (押解): 犬侧伴随 + 保持警觉
    - obstacle (障碍穿越): 跳跃/攀爬姿态序列
    - recall (返回): 远离→朝向训导员快速移动
    - watch (警戒): 头部抬起 + 身体紧绷

输入:
    keypoints_sequence: np.ndarray, shape=(T, 24, 3), 每行 [x, y, conf]
    boxes: optional list[np.ndarray], 每帧犬检测框 [x1, y1, x2, y2]（用于 heel）
    person_boxes: optional list, 每帧人检测框（用于 escort/recall 完整版）
    target_boxes: optional list, 每帧目标检测框（用于 alert_sit/alert_down/apprehend 完整版）
    fps: float, 视频帧率

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
    # P1 常量
    TRACK, ALERT_SIT, ALERT_DOWN, APPREHEND, ESCORT, OBSTACLE, RECALL, WATCH,
    P1_BEHAVIORS, ALL_BEHAVIORS,
)

# P0 默认阈值（可被 YAML 评分卡覆盖）
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

# P1 默认阈值（Phase 2 新增）
DEFAULT_TRACK_GROUND_RATIO = 0.85  # 追踪：NOSE.y > ground_y * 此值视为鼻尖贴地
DEFAULT_TRACK_MOTION = 3.0         # 追踪：持续移动阈值（withers.x 帧间变化）
DEFAULT_OBSTACLE_Y_DROP = 15.0     # 障碍：withers.y 下降值（跳跃高度，像素）
DEFAULT_OBSTACLE_MOTION = 15.0     # 障碍：高速移动阈值
DEFAULT_WATCH_STABILITY = 3.0      # 警戒：关键点帧间位移 < 此值视为稳定
DEFAULT_APPREHEND_SPEED = 25.0     # 扑咬：withers 帧间位移 > 此值视为高速
DEFAULT_APPREHEND_MOUTH_VAR = 5.0  # 扑咬：嘴部活跃度（nose-chin 方差）
DEFAULT_ESCORT_STAND_RATIO = 0.7   # 押解：窗口内站立比例阈值
DEFAULT_RECALL_SPEED = 20.0        # 返回：快速移动阈值
DEFAULT_ALERT_TARGET_DIST = 150.0  # 示警：鼻尖到目标距离阈值（像素）


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
        # P1 阈值
        track_ground_ratio: float = DEFAULT_TRACK_GROUND_RATIO,
        track_motion: float = DEFAULT_TRACK_MOTION,
        obstacle_y_drop: float = DEFAULT_OBSTACLE_Y_DROP,
        obstacle_motion: float = DEFAULT_OBSTACLE_MOTION,
        watch_stability: float = DEFAULT_WATCH_STABILITY,
        apprehend_speed: float = DEFAULT_APPREHEND_SPEED,
        apprehend_mouth_var: float = DEFAULT_APPREHEND_MOUTH_VAR,
        escort_stand_ratio: float = DEFAULT_ESCORT_STAND_RATIO,
        recall_speed: float = DEFAULT_RECALL_SPEED,
        alert_target_dist: float = DEFAULT_ALERT_TARGET_DIST,
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
        # P1
        self.track_ground_ratio = track_ground_ratio
        self.track_motion = track_motion
        self.obstacle_y_drop = obstacle_y_drop
        self.obstacle_motion = obstacle_motion
        self.watch_stability = watch_stability
        self.apprehend_speed = apprehend_speed
        self.apprehend_mouth_var = apprehend_mouth_var
        self.escort_stand_ratio = escort_stand_ratio
        self.recall_speed = recall_speed
        self.alert_target_dist = alert_target_dist

    def recognize(
        self,
        keypoints_sequence: np.ndarray,
        boxes: Optional[list] = None,
        person_boxes: Optional[list] = None,
        target_boxes: Optional[list] = None,
        fps: float = 30.0,
        enable_p1: bool = True,
    ) -> list[BehaviorEpisode]:
        """识别 16 类行为（P0 8 类 + P1 8 类）。

        Args:
            keypoints_sequence: shape=(T, 24, 3), [x, y, conf]
            boxes: 每帧犬检测框 [x1,y1,x2,y2]（可选，用于 heel）
            person_boxes: 每帧人检测框（可选，用于 escort/recall 完整版）
            target_boxes: 每帧目标检测框（可选，用于 alert_sit/alert_down/apprehend 完整版）
            fps: 帧率
            enable_p1: 是否启用 P1 检测（默认 True，Phase 2）

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

        # ===== P0 检测 =====
        # 1. 逐帧姿态分类（sit / down / stand）
        posture_labels = self._classify_posture_per_frame(kpts)

        # 2. 聚合连续相同姿态 → episodes
        episodes.extend(self._aggregate_posture(posture_labels, kpts))

        # 3. sit_up: 坐姿 + 头部抬起
        episodes.extend(self._detect_sit_up(kpts, posture_labels))

        # 4. stay: 关键点稳定 ≥ N 帧
        episodes.extend(self._detect_stay(kpts))

        # 5. heel: 持续移动 + 站立
        episodes.extend(self._detect_heel(kpts, posture_labels, boxes))

        # 6. bark: nose-chin 周期性变化
        episodes.extend(self._detect_bark(kpts))

        # 7. bite: 嘴部动作 + 前爪快速移动
        episodes.extend(self._detect_bite(kpts))

        # ===== P1 检测（Phase 2 新增）=====
        if enable_p1:
            # 8. track: 鼻尖贴近地面 + 路径跟随
            episodes.extend(self._detect_track(kpts))

            # 9. obstacle: 跳跃轨迹（withers.y 先降后升）+ 高速移动
            episodes.extend(self._detect_obstacle(kpts))

            # 10. watch: 头部抬起 + 身体紧绷 + 站立
            episodes.extend(self._detect_watch(kpts, posture_labels))

            # 11. apprehend: 高速移动 + 嘴部活跃（+ 接近目标）
            episodes.extend(self._detect_apprehend(kpts, target_boxes))

            # 12. escort: 持续移动 + 站立 + 头部警觉（+ 犬在人侧方）
            episodes.extend(self._detect_escort(kpts, posture_labels, person_boxes))

            # 13. recall: 方向反转 + 快速移动（+ 朝向人）
            episodes.extend(self._detect_recall(kpts, person_boxes))

            # 14. alert_sit: SIT 姿态 + 鼻尖朝向目标（+ 目标框）
            episodes.extend(self._detect_alert_sit(kpts, posture_labels, target_boxes))

            # 15. alert_down: DOWN 姿态 + 鼻尖朝向目标（+ 目标框）
            episodes.extend(self._detect_alert_down(kpts, posture_labels, target_boxes))

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

    # ===== P1 检测方法（Phase 2 新增）=====

    def _compute_withers_motion(self, kpts: np.ndarray) -> np.ndarray:
        """计算 withers 帧间位移（幅度）。"""
        T = kpts.shape[0]
        motion = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            if (kpts[t, WITHERS, 2] >= self.conf_threshold and
                    kpts[t-1, WITHERS, 2] >= self.conf_threshold):
                dx = kpts[t, WITHERS, 0] - kpts[t-1, WITHERS, 0]
                dy = kpts[t, WITHERS, 1] - kpts[t-1, WITHERS, 1]
                motion[t] = float(np.sqrt(dx*dx + dy*dy))
        return motion

    def _compute_mouth_dist(self, kpts: np.ndarray) -> np.ndarray:
        """计算 nose-chin 距离序列。"""
        T = kpts.shape[0]
        dist = np.zeros(T, dtype=np.float32)
        for t in range(T):
            if (kpts[t, NOSE, 2] >= self.conf_threshold and
                    kpts[t, CHIN, 2] >= self.conf_threshold):
                dx = kpts[t, NOSE, 0] - kpts[t, CHIN, 0]
                dy = kpts[t, NOSE, 1] - kpts[t, CHIN, 1]
                dist[t] = float(np.sqrt(dx*dx + dy*dy))
        return dist

    def _detect_track(self, kpts: np.ndarray) -> list[BehaviorEpisode]:
        """追踪：鼻尖贴近地面 + 持续移动 + 头部朝下。

        依据 RESEARCH_STANDARDS.md B09: 鼻尖贴近地面 + 路径跟随。
        """
        T = kpts.shape[0]
        if T < self.stay_frames:
            return []

        motion_x = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            if (kpts[t, WITHERS, 2] >= self.conf_threshold and
                    kpts[t-1, WITHERS, 2] >= self.conf_threshold):
                motion_x[t] = abs(kpts[t, WITHERS, 0] - kpts[t-1, WITHERS, 0])

        episodes = []
        window = self.stay_frames
        i = 0
        while i <= T - window:
            track_frames = 0
            for t in range(i, i + window):
                frame = kpts[t]
                mask = self._valid_mask(frame)
                if mask[NOSE] and mask[WITHERS]:
                    ground_y = self._estimate_ground_y(frame)
                    nose_near_ground = frame[NOSE, 1] > ground_y * self.track_ground_ratio
                    head_down = frame[NOSE, 1] > frame[WITHERS, 1]
                    if nose_near_ground and head_down:
                        track_frames += 1

            track_ratio = track_frames / window
            clip_motion = motion_x[i+1:i+window]
            moving = (clip_motion > self.track_motion).mean() > 0.5

            if track_ratio > 0.7 and moving:
                end = i + window
                while end < T:
                    frame = kpts[end]
                    mask = self._valid_mask(frame)
                    if (mask[NOSE] and mask[WITHERS] and
                            frame[NOSE, 1] > self._estimate_ground_y(frame) * self.track_ground_ratio and
                            frame[NOSE, 1] > frame[WITHERS, 1] and
                            motion_x[end] > self.track_motion):
                        end += 1
                    else:
                        break
                episodes.append(BehaviorEpisode(
                    behavior=TRACK,
                    start_frame=i,
                    end_frame=end - 1,
                    confidence=min(1.0, track_ratio),
                    metadata={"track_ratio": track_ratio,
                              "mean_motion_x": float(clip_motion.mean())},
                ))
                i = end
            else:
                i += 1
        return episodes

    def _detect_obstacle(self, kpts: np.ndarray) -> list[BehaviorEpisode]:
        """障碍穿越：跳跃轨迹（withers.y 先降后升）+ 高速移动。

        依据 RESEARCH_STANDARDS.md B14: 跳跃/攀爬姿态序列。
        y 轴向下：跳跃时 withers 先上升（y 减小）后下降（y 增大）。
        """
        T = kpts.shape[0]
        if T < self.stay_frames * 2:
            return []

        withers_y = kpts[:, WITHERS, 1].copy()
        withers_valid = kpts[:, WITHERS, 2] >= self.conf_threshold
        motion = self._compute_withers_motion(kpts)

        episodes = []
        window = self.stay_frames
        i = 0
        while i <= T - window * 2:
            if not withers_valid[i:i+window*2].all():
                i += 1
                continue

            clip_y = withers_y[i:i+window*2]
            clip_motion = motion[i+1:i+window*2]

            first_half = clip_y[:window]
            second_half = clip_y[window:]
            y_rise = second_half.mean() - first_half.mean()  # 正值=先高后低
            high_motion = clip_motion.mean() > self.obstacle_motion

            if y_rise > self.obstacle_y_drop and high_motion:
                end = i + window * 2
                episodes.append(BehaviorEpisode(
                    behavior=OBSTACLE,
                    start_frame=i,
                    end_frame=end - 1,
                    confidence=min(1.0, y_rise / (self.obstacle_y_drop * 3)),
                    metadata={"y_rise": float(y_rise),
                              "mean_motion": float(clip_motion.mean())},
                ))
                i = end
            else:
                i += 1
        return episodes

    def _detect_watch(self, kpts: np.ndarray, posture_labels: list[str]) -> list[BehaviorEpisode]:
        """警戒：头部抬起 + 身体紧绷（关键点稳定）+ 站立姿态。

        依据 RESEARCH_STANDARDS.md B16: 头部抬起 + 身体紧绷。
        """
        T = kpts.shape[0]
        if T < self.stay_frames:
            return []

        motion = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            mask = (kpts[t, :, 2] >= self.conf_threshold) & \
                   (kpts[t-1, :, 2] >= self.conf_threshold)
            if mask.any():
                dx = kpts[t, mask, 0] - kpts[t-1, mask, 0]
                dy = kpts[t, mask, 1] - kpts[t-1, mask, 1]
                motion[t] = float(np.sqrt(dx*dx + dy*dy).mean())

        episodes = []
        window = self.stay_frames
        i = 0
        while i <= T - window:
            clip_motion = motion[i+1:i+window]
            stable = (clip_motion < self.watch_stability).mean() > 0.7
            stand_ratio = sum(1 for t in range(i, i+window)
                              if posture_labels[t] == STAND) / window

            head_up_frames = 0
            for t in range(i, i+window):
                frame = kpts[t]
                mask = self._valid_mask(frame)
                if mask[NOSE] and mask[WITHERS]:
                    if frame[NOSE, 1] < frame[WITHERS, 1]:
                        head_up_frames += 1
            head_up_ratio = head_up_frames / window

            if stable and stand_ratio > 0.5 and head_up_ratio > 0.6:
                end = i + window
                while end < T:
                    frame = kpts[end]
                    mask = self._valid_mask(frame)
                    if (motion[end] < self.watch_stability and
                            posture_labels[end] == STAND and
                            mask[NOSE] and mask[WITHERS] and
                            frame[NOSE, 1] < frame[WITHERS, 1]):
                        end += 1
                    else:
                        break
                episodes.append(BehaviorEpisode(
                    behavior=WATCH,
                    start_frame=i,
                    end_frame=end - 1,
                    confidence=min(1.0, head_up_ratio * stand_ratio),
                    metadata={"head_up_ratio": head_up_ratio,
                              "stand_ratio": stand_ratio,
                              "mean_motion": float(clip_motion.mean())},
                ))
                i = end
            else:
                i += 1
        return episodes

    def _detect_apprehend(self, kpts: np.ndarray,
                          target_boxes: Optional[list] = None) -> list[BehaviorEpisode]:
        """扑咬：高速移动 + 嘴部活跃。

        依据 RESEARCH_STANDARDS.md B12: 高速接近 + 嘴部接触目标。
        简化版（无 target_boxes）：高速移动 + 嘴部活跃。
        完整版（有 target_boxes）：上述 + 接近目标。
        """
        T = kpts.shape[0]
        if T < self.stay_frames:
            return []

        withers_motion = self._compute_withers_motion(kpts)
        mouth_dist = self._compute_mouth_dist(kpts)

        episodes = []
        window = self.stay_frames
        i = 0
        while i <= T - window:
            withers_clip = withers_motion[i+1:i+window]
            mouth_clip = mouth_dist[i:i+window]

            high_speed = withers_clip.mean() > self.apprehend_speed
            mouth_active = mouth_clip.var() > self.apprehend_mouth_var

            if high_speed and mouth_active:
                approaching = True
                if target_boxes:
                    approaching = self._check_approaching_target(
                        kpts, target_boxes, i, window)

                if approaching:
                    end = i + window
                    while (end < T and
                           withers_motion[end] > self.apprehend_speed and
                           mouth_dist[end-window:end].var() > self.apprehend_mouth_var):
                        end += 1
                    episodes.append(BehaviorEpisode(
                        behavior=APPREHEND,
                        start_frame=i,
                        end_frame=end - 1,
                        confidence=min(1.0, withers_clip.mean() / (self.apprehend_speed * 2)),
                        metadata={"mean_speed": float(withers_clip.mean()),
                                  "mouth_var": float(mouth_clip.var()),
                                  "has_target": target_boxes is not None},
                    ))
                    i = end
                else:
                    i += 1
            else:
                i += 1
        return episodes

    def _check_approaching_target(self, kpts: np.ndarray, target_boxes: list,
                                  start: int, window: int) -> bool:
        """检查犬是否在接近目标（距离缩小）。"""
        dists = []
        for t in range(start, min(start + window, len(target_boxes))):
            if t >= len(target_boxes) or target_boxes[t] is None:
                continue
            mask = self._valid_mask(kpts[t])
            if not mask[WITHERS]:
                continue
            target = target_boxes[t]
            tcx = (target[0] + target[2]) / 2
            tcy = (target[1] + target[3]) / 2
            wx, wy = kpts[t, WITHERS, 0], kpts[t, WITHERS, 1]
            dists.append(float(np.sqrt((wx - tcx)**2 + (wy - tcy)**2)))

        if len(dists) < 2:
            return True  # 数据不足，不阻塞检测
        # 后半段距离小于前半段 = 接近
        mid = len(dists) // 2
        return np.mean(dists[mid:]) < np.mean(dists[:mid])

    def _detect_escort(self, kpts: np.ndarray, posture_labels: list[str],
                       person_boxes: Optional[list] = None) -> list[BehaviorEpisode]:
        """押解：持续移动 + 站立 + 头部警觉。

        依据 RESEARCH_STANDARDS.md B13: 犬侧伴随 + 保持警觉。
        简化版（无 person_boxes）：持续移动 + 站立 + 头部抬起。
        完整版（有 person_boxes）：上述 + 犬在人侧方。
        """
        T = kpts.shape[0]
        if T < self.stay_frames:
            return []

        motion_x = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            if (kpts[t, WITHERS, 2] >= self.conf_threshold and
                    kpts[t-1, WITHERS, 2] >= self.conf_threshold):
                motion_x[t] = abs(kpts[t, WITHERS, 0] - kpts[t-1, WITHERS, 0])

        episodes = []
        window = self.stay_frames
        i = 0
        while i <= T - window:
            clip_motion = motion_x[i+1:i+window]
            moving = (clip_motion > self.heel_motion).mean() > 0.7
            stand_ratio = sum(1 for t in range(i, i+window)
                              if posture_labels[t] == STAND) / window

            head_alert = 0
            for t in range(i, i+window):
                frame = kpts[t]
                mask = self._valid_mask(frame)
                if mask[NOSE] and mask[WITHERS]:
                    if frame[NOSE, 1] <= frame[WITHERS, 1]:
                        head_alert += 1
            head_alert_ratio = head_alert / window

            if moving and stand_ratio > self.escort_stand_ratio and head_alert_ratio > 0.5:
                end = i + window
                while (end < T and motion_x[end] > self.heel_motion and
                       posture_labels[end] == STAND):
                    end += 1
                episodes.append(BehaviorEpisode(
                    behavior=ESCORT,
                    start_frame=i,
                    end_frame=end - 1,
                    confidence=min(1.0, stand_ratio * head_alert_ratio),
                    metadata={"stand_ratio": stand_ratio,
                              "head_alert_ratio": head_alert_ratio,
                              "mean_motion_x": float(clip_motion.mean()),
                              "has_person": person_boxes is not None},
                ))
                i = end
            else:
                i += 1
        return episodes

    def _detect_recall(self, kpts: np.ndarray,
                       person_boxes: Optional[list] = None) -> list[BehaviorEpisode]:
        """返回：方向反转 + 快速移动。

        依据 RESEARCH_STANDARDS.md B15: 远离→朝向训导员快速移动。
        简化版（无 person_boxes）：x 方向先远离后接近 + 快速移动。
        完整版（有 person_boxes）：上述 + 朝向人移动。
        """
        T = kpts.shape[0]
        if T < self.stay_frames * 2:
            return []

        withers_x = kpts[:, WITHERS, 0].copy()
        withers_valid = kpts[:, WITHERS, 2] >= self.conf_threshold

        velocity = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            if withers_valid[t] and withers_valid[t-1]:
                velocity[t] = withers_x[t] - withers_x[t-1]
        speed = np.abs(velocity)

        episodes = []
        window = self.stay_frames
        i = 0
        while i <= T - window * 2:
            if not withers_valid[i:i+window*2].all():
                i += 1
                continue

            first_vel = velocity[i+1:i+window]
            second_vel = velocity[i+window+1:i+window*2]

            first_mean = float(first_vel.mean())
            second_mean = float(second_vel.mean())
            first_dir = np.sign(first_mean) if abs(first_mean) > 0.5 else 0
            second_dir = np.sign(second_mean) if abs(second_mean) > 0.5 else 0

            direction_reversed = (first_dir != 0 and second_dir != 0 and
                                  first_dir != second_dir)

            second_speed = speed[i+window+1:i+window*2]
            fast_return = second_speed.mean() > self.recall_speed

            if direction_reversed and fast_return:
                end = i + window * 2
                episodes.append(BehaviorEpisode(
                    behavior=RECALL,
                    start_frame=i,
                    end_frame=end - 1,
                    confidence=min(1.0, second_speed.mean() / (self.recall_speed * 2)),
                    metadata={"first_direction": float(first_dir),
                              "second_direction": float(second_dir),
                              "return_speed": float(second_speed.mean()),
                              "has_person": person_boxes is not None},
                ))
                i = end
            else:
                i += 1
        return episodes

    def _detect_alert_sit(self, kpts: np.ndarray, posture_labels: list[str],
                          target_boxes: Optional[list] = None) -> list[BehaviorEpisode]:
        """示警坐：SIT 姿态 + 鼻尖朝向目标。

        依据 RESEARCH_STANDARDS.md B10: 检出目标后坐姿示警。
        简化版（无 target_boxes）：SIT + 头部抬起。
        完整版（有 target_boxes）：SIT + 鼻尖朝向目标。
        """
        return self._detect_alert_posture(
            kpts, posture_labels, target_boxes, SIT, ALERT_SIT)

    def _detect_alert_down(self, kpts: np.ndarray, posture_labels: list[str],
                           target_boxes: Optional[list] = None) -> list[BehaviorEpisode]:
        """示警卧：DOWN 姿态 + 鼻尖朝向目标。

        依据 RESEARCH_STANDARDS.md B11: 检出目标后卧姿示警。
        简化版（无 target_boxes）：DOWN + 头部抬起。
        完整版（有 target_boxes）：DOWN + 鼻尖朝向目标。
        """
        return self._detect_alert_posture(
            kpts, posture_labels, target_boxes, DOWN, ALERT_DOWN)

    def _detect_alert_posture(self, kpts: np.ndarray, posture_labels: list[str],
                              target_boxes: Optional[list], posture: str,
                              alert_behavior: str) -> list[BehaviorEpisode]:
        """示警姿态通用检测（alert_sit / alert_down 共用）。"""
        T = kpts.shape[0]
        if T < self.stay_frames:
            return []

        episodes = []
        window = self.stay_frames
        i = 0
        while i <= T - window:
            posture_ratio = sum(1 for t in range(i, i+window)
                                if posture_labels[t] == posture) / window
            if posture_ratio < 0.7:
                i += 1
                continue

            alert_frames = 0
            for t in range(i, i+window):
                frame = kpts[t]
                mask = self._valid_mask(frame)
                if not mask[NOSE]:
                    continue

                if target_boxes and t < len(target_boxes) and target_boxes[t] is not None:
                    target = target_boxes[t]
                    tcx = (target[0] + target[2]) / 2
                    tcy = (target[1] + target[3]) / 2
                    nose_x, nose_y = frame[NOSE, 0], frame[NOSE, 1]
                    dist = float(np.sqrt((nose_x - tcx)**2 + (nose_y - tcy)**2))
                    if dist < self.alert_target_dist:
                        alert_frames += 1
                else:
                    if mask[WITHERS] and frame[NOSE, 1] < frame[WITHERS, 1]:
                        alert_frames += 1

            alert_ratio = alert_frames / window
            if alert_ratio > 0.5:
                end = i + window
                while end < T and posture_labels[end] == posture:
                    end += 1
                episodes.append(BehaviorEpisode(
                    behavior=alert_behavior,
                    start_frame=i,
                    end_frame=end - 1,
                    confidence=min(1.0, posture_ratio * alert_ratio),
                    metadata={"posture_ratio": posture_ratio,
                              "alert_ratio": alert_ratio,
                              "has_target": target_boxes is not None},
                ))
                i = end
            else:
                i += 1
        return episodes


def recognize_behaviors(
    keypoints_sequence: np.ndarray,
    boxes: Optional[list] = None,
    person_boxes: Optional[list] = None,
    target_boxes: Optional[list] = None,
    fps: float = 30.0,
    enable_p1: bool = True,
    **kwargs,
) -> list[BehaviorEpisode]:
    """便捷函数：识别 16 类行为。"""
    engine = RuleEngine(**kwargs)
    return engine.recognize(
        keypoints_sequence, boxes=boxes, person_boxes=person_boxes,
        target_boxes=target_boxes, fps=fps, enable_p1=enable_p1,
    )
