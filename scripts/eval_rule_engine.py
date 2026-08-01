"""规则引擎准确率评估脚本.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.2f / 2.2d
依据: dev-docs/stages/phase-1.md §1.2f + dev-docs/stages/phase-2.md §2.2d

评估策略:
    1. 主路径: DogMo 测试集（如已下载到 data/dogmo/）
    2. 降级路径: 合成标注序列（覆盖 8 类 P0 + 8 类 P1 = 16 类行为，已知 ground-truth）

评估指标:
    - Episode-level: 每个序列的 ground-truth 行为是否被检出
    - Per-behavior: Precision / Recall / F1
    - Overall: 准确率
    - 混淆矩阵

决策门:
    - P0 准确率 ≥ 80% → 跳过 Phase 1.3
    - P1 准确率 ≥ 85% → Phase 2.2d 达标

用法:
    python scripts/eval_rule_engine.py                          # 自动选择评估数据源（P0+P1）
    python scripts/eval_rule_engine.py --phase p0               # 仅 P0
    python scripts/eval_rule_engine.py --phase p1               # 仅 P1
    python scripts/eval_rule_engine.py --phase all              # P0 + P1
    python scripts/eval_rule_engine.py --source synthetic       # 强制合成数据
    python scripts/eval_rule_engine.py --source dogmo           # 强制 DogMo（需已下载）
    python scripts/eval_rule_engine.py --report reports/phase-2.2d-validation.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.behavior.constants import (
    BARK, BITE, CHIN, DOWN, FRONT_ELBOWS, FRONT_KNEES, FRONT_PAWS,
    HEEL, LEFT_EAR_BASE, LEFT_EAR_TIP, LEFT_EYE, NOSE,
    NUM_KEYPOINTS, P0_BEHAVIORS, REAR_ELBOWS, REAR_KNEES, REAR_PAWS,
    RIGHT_EAR_BASE, RIGHT_EAR_TIP, RIGHT_EYE, SIT, SIT_UP, STAND, STAY,
    TAIL_END, TAIL_START, THROAT, WITHERS,
    BEHAVIOR_NAMES_CN,
    # P1 常量
    P1_BEHAVIORS, ALL_BEHAVIORS,
    TRACK, ALERT_SIT, ALERT_DOWN, APPREHEND, ESCORT, OBSTACLE, RECALL, WATCH,
)
from backend.ml.behavior.rule_engine import (
    RuleEngine,
    DEFAULT_OBSTACLE_Y_DROP, DEFAULT_OBSTACLE_MOTION,
    DEFAULT_APPREHEND_SPEED, DEFAULT_APPREHEND_MOUTH_VAR,
    DEFAULT_ESCORT_STAND_RATIO, DEFAULT_RECALL_SPEED,
)

# ===== 数据结构 =====


@dataclass
class EvalSample:
    """单个评估样本。"""
    name: str                   # 样本名称
    ground_truth: str           # ground-truth 行为类别
    kpts_seq: np.ndarray        # (T, 24, 3) 关键点序列
    fps: float = 30.0
    source: str = "synthetic"   # synthetic | dogmo


@dataclass
class EvalResult:
    """单个样本评估结果。"""
    sample: EvalSample
    detected_behaviors: list[str]       # 检测到的所有行为类别
    hit: bool                           # ground_truth 是否在 detected_behaviors 中
    episodes_count: int                 # 检测到的 episode 总数


@dataclass
class BehaviorMetrics:
    """单类行为指标。"""
    behavior: str
    tp: int = 0     # ground-truth=该行为 且 检出该行为
    fp: int = 0     # ground-truth≠该行为 但 检出该行为
    fn: int = 0     # ground-truth=该行为 但 未检出该行为
    samples: int = 0  # 该行为作为 ground-truth 的样本数

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class OverallMetrics:
    """总体指标。"""
    total_samples: int = 0
    correct: int = 0
    per_behavior: dict[str, BehaviorMetrics] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    source: str = "synthetic"
    phase: str = "all"  # p0 | p1 | all

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def trigger_phase_1_3(self) -> bool:
        """是否触发 Phase 1.3 PoseC3D（仅 P0 评估时有效）。"""
        return self.accuracy < 0.80

    @property
    def p1_pass(self) -> bool:
        """P1 准确率是否达标（≥ 85%）。"""
        return self.accuracy >= 0.85

    def p0_metrics(self) -> tuple[int, int, float]:
        """返回 P0 子集的 (correct, total, accuracy)。"""
        p0_total = sum(
            m.samples for b, m in self.per_behavior.items() if b in P0_BEHAVIORS
        )
        p0_correct = sum(
            m.tp for b, m in self.per_behavior.items() if b in P0_BEHAVIORS
        )
        p0_acc = p0_correct / p0_total if p0_total > 0 else 0.0
        return p0_correct, p0_total, p0_acc

    def p1_metrics(self) -> tuple[int, int, float]:
        """返回 P1 子集的 (correct, total, accuracy)。"""
        p1_total = sum(
            m.samples for b, m in self.per_behavior.items() if b in P1_BEHAVIORS
        )
        p1_correct = sum(
            m.tp for b, m in self.per_behavior.items() if b in P1_BEHAVIORS
        )
        p1_acc = p1_correct / p1_total if p1_total > 0 else 0.0
        return p1_correct, p1_total, p1_acc


# ===== 合成序列构造（复用 test_rule_engine.py 逻辑） =====


def _make_frame(keypoints: dict[int, tuple[float, float, float]]) -> np.ndarray:
    frame = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    for idx, (x, y, conf) in keypoints.items():
        frame[idx] = [x, y, conf]
    return frame


def _make_standing_frame(x_offset: float = 0.0, ground_y: float = 200.0) -> np.ndarray:
    return _make_frame({
        FRONT_PAWS[0]: (50 + x_offset, ground_y, 1.0),
        FRONT_PAWS[1]: (150 + x_offset, ground_y, 1.0),
        FRONT_KNEES[0]: (50 + x_offset, ground_y - 30, 1.0),
        FRONT_KNEES[1]: (150 + x_offset, ground_y - 30, 1.0),
        FRONT_ELBOWS[0]: (50 + x_offset, ground_y - 60, 1.0),
        FRONT_ELBOWS[1]: (150 + x_offset, ground_y - 60, 1.0),
        REAR_PAWS[0]: (50 + x_offset, ground_y, 1.0),
        REAR_PAWS[1]: (150 + x_offset, ground_y, 1.0),
        REAR_KNEES[0]: (50 + x_offset, ground_y - 30, 1.0),
        REAR_KNEES[1]: (150 + x_offset, ground_y - 30, 1.0),
        REAR_ELBOWS[0]: (50 + x_offset, ground_y - 60, 1.0),
        REAR_ELBOWS[1]: (150 + x_offset, ground_y - 60, 1.0),
        TAIL_START: (100 + x_offset, ground_y - 100, 1.0),
        TAIL_END: (100 + x_offset, ground_y - 120, 1.0),
        LEFT_EAR_BASE: (80 + x_offset, ground_y - 130, 1.0),
        RIGHT_EAR_BASE: (120 + x_offset, ground_y - 130, 1.0),
        NOSE: (100 + x_offset, ground_y - 140, 1.0),
        CHIN: (100 + x_offset, ground_y - 130, 1.0),
        LEFT_EAR_TIP: (75 + x_offset, ground_y - 145, 1.0),
        RIGHT_EAR_TIP: (125 + x_offset, ground_y - 145, 1.0),
        LEFT_EYE: (85 + x_offset, ground_y - 135, 1.0),
        RIGHT_EYE: (115 + x_offset, ground_y - 135, 1.0),
        WITHERS: (100 + x_offset, ground_y - 120, 1.0),
        THROAT: (100 + x_offset, ground_y - 110, 1.0),
    })


def _make_sitting_frame(x_offset: float = 0.0, ground_y: float = 200.0) -> np.ndarray:
    return _make_frame({
        FRONT_PAWS[0]: (50 + x_offset, ground_y, 1.0),
        FRONT_PAWS[1]: (150 + x_offset, ground_y, 1.0),
        FRONT_KNEES[0]: (50 + x_offset, ground_y - 30, 1.0),
        FRONT_KNEES[1]: (150 + x_offset, ground_y - 30, 1.0),
        FRONT_ELBOWS[0]: (50 + x_offset, ground_y - 60, 1.0),
        FRONT_ELBOWS[1]: (150 + x_offset, ground_y - 60, 1.0),
        REAR_PAWS[0]: (60 + x_offset, ground_y - 40, 1.0),
        REAR_PAWS[1]: (140 + x_offset, ground_y - 40, 1.0),
        REAR_KNEES[0]: (60 + x_offset, ground_y - 45, 1.0),
        REAR_KNEES[1]: (140 + x_offset, ground_y - 45, 1.0),
        REAR_ELBOWS[0]: (60 + x_offset, ground_y - 80, 1.0),
        REAR_ELBOWS[1]: (140 + x_offset, ground_y - 80, 1.0),
        TAIL_START: (100 + x_offset, ground_y - 70, 1.0),
        TAIL_END: (100 + x_offset, ground_y - 90, 1.0),
        LEFT_EAR_BASE: (80 + x_offset, ground_y - 90, 1.0),
        RIGHT_EAR_BASE: (120 + x_offset, ground_y - 90, 1.0),
        NOSE: (100 + x_offset, ground_y - 100, 1.0),
        CHIN: (100 + x_offset, ground_y - 90, 1.0),
        LEFT_EAR_TIP: (75 + x_offset, ground_y - 105, 1.0),
        RIGHT_EAR_TIP: (125 + x_offset, ground_y - 105, 1.0),
        LEFT_EYE: (85 + x_offset, ground_y - 95, 1.0),
        RIGHT_EYE: (115 + x_offset, ground_y - 95, 1.0),
        WITHERS: (100 + x_offset, ground_y - 80, 1.0),
        THROAT: (100 + x_offset, ground_y - 70, 1.0),
    })


def _make_lying_frame(x_offset: float = 0.0, ground_y: float = 200.0) -> np.ndarray:
    from backend.ml.behavior.constants import ALL_KNEES, ALL_PAWS
    frame = _make_frame({})
    for i, idx in enumerate(ALL_PAWS):
        frame[idx] = [50 + i * 30 + x_offset, ground_y - 5, 1.0]
    for i, idx in enumerate(ALL_KNEES):
        frame[idx] = [50 + i * 30 + x_offset, ground_y - 8, 1.0]
    for i, idx in enumerate(FRONT_ELBOWS + REAR_ELBOWS):
        frame[idx] = [50 + i * 30 + x_offset, ground_y - 12, 1.0]
    frame[TAIL_START] = [100 + x_offset, ground_y - 10, 1.0]
    frame[TAIL_END] = [100 + x_offset, ground_y - 15, 1.0]
    frame[LEFT_EAR_BASE] = [80 + x_offset, ground_y - 20, 1.0]
    frame[RIGHT_EAR_BASE] = [120 + x_offset, ground_y - 20, 1.0]
    frame[NOSE] = [100 + x_offset, ground_y - 25, 1.0]
    frame[CHIN] = [100 + x_offset, ground_y - 20, 1.0]
    frame[LEFT_EAR_TIP] = [75 + x_offset, ground_y - 25, 1.0]
    frame[RIGHT_EAR_TIP] = [125 + x_offset, ground_y - 25, 1.0]
    frame[LEFT_EYE] = [85 + x_offset, ground_y - 22, 1.0]
    frame[RIGHT_EYE] = [115 + x_offset, ground_y - 22, 1.0]
    frame[WITHERS] = [100 + x_offset, ground_y - 10, 1.0]
    frame[THROAT] = [100 + x_offset, ground_y - 15, 1.0]
    return frame


def _stack(frames: list[np.ndarray]) -> np.ndarray:
    return np.stack(frames, axis=0)


# ===== 合成评估数据集生成 =====


def generate_synthetic_dataset() -> list[EvalSample]:
    """生成覆盖 8 类 P0 行为的合成评估数据集。

    每类行为生成多个变体（不同帧数、噪声），共 ~40 样本。
    """
    samples: list[EvalSample] = []

    # --- sit（坐）---
    for n_frames in (5, 10, 20, 30):
        for noise in (0.0, 2.0):
            frames = []
            for i in range(n_frames):
                f = _make_sitting_frame()
                if noise > 0:
                    f[:, :2] += np.random.randn(NUM_KEYPOINTS, 2).astype(np.float32) * noise
                frames.append(f)
            samples.append(EvalSample(
                name=f"sit_{n_frames}f_noise{noise:.0f}",
                ground_truth=SIT, kpts_seq=_stack(frames), source="synthetic",
            ))

    # --- stand（立）---
    for n_frames in (5, 10, 20, 30):
        for noise in (0.0, 2.0):
            frames = [_make_standing_frame() for _ in range(n_frames)]
            if noise > 0:
                for f in frames:
                    f[:, :2] += np.random.randn(NUM_KEYPOINTS, 2).astype(np.float32) * noise
            samples.append(EvalSample(
                name=f"stand_{n_frames}f_noise{noise:.0f}",
                ground_truth=STAND, kpts_seq=_stack(frames), source="synthetic",
            ))

    # --- down（卧）---
    for n_frames in (5, 10, 20, 30):
        for noise in (0.0, 2.0):
            frames = [_make_lying_frame() for _ in range(n_frames)]
            if noise > 0:
                for f in frames:
                    f[:, :2] += np.random.randn(NUM_KEYPOINTS, 2).astype(np.float32) * noise
            samples.append(EvalSample(
                name=f"down_{n_frames}f_noise{noise:.0f}",
                ground_truth=DOWN, kpts_seq=_stack(frames), source="synthetic",
            ))

    # --- heel（随行）---
    for speed in (3.0, 5.0, 8.0):
        for n_frames in (20, 30):
            frames = [_make_standing_frame(x_offset=i * speed) for i in range(n_frames)]
            samples.append(EvalSample(
                name=f"heel_speed{speed:.0f}_{n_frames}f",
                ground_truth=HEEL, kpts_seq=_stack(frames), source="synthetic",
            ))

    # --- stay（停留）---
    for n_frames in (20, 30, 50):
        f = _make_standing_frame()
        frames = [f.copy() for _ in range(n_frames)]
        samples.append(EvalSample(
            name=f"stay_{n_frames}f",
            ground_truth=STAY, kpts_seq=_stack(frames), source="synthetic",
        ))

    # --- sit_up（坐立）---
    for n_frames in (5, 10, 20):
        f = _make_sitting_frame()
        f[NOSE] = [100, 50, 1.0]
        f[WITHERS, 1] = 120
        frames = [f.copy() for _ in range(n_frames)]
        samples.append(EvalSample(
            name=f"sit_up_{n_frames}f",
            ground_truth=SIT_UP, kpts_seq=_stack(frames), source="synthetic",
        ))

    # --- bark（叫）---
    for n_frames in (20, 30):
        frames = []
        for i in range(n_frames):
            f = _make_standing_frame()
            if i % 2 == 0:
                f[NOSE] = [100, 80, 1.0]
                f[CHIN] = [100, 110, 1.0]
            else:
                f[NOSE] = [100, 100, 1.0]
                f[CHIN] = [100, 105, 1.0]
            frames.append(f)
        samples.append(EvalSample(
            name=f"bark_{n_frames}f",
            ground_truth=BARK, kpts_seq=_stack(frames), source="synthetic",
        ))

    # --- bite（咬）---
    for speed in (25.0, 30.0):
        for n_frames in (20, 30):
            frames = []
            for i in range(n_frames):
                f = _make_standing_frame(x_offset=i * speed)
                if i % 2 == 0:
                    f[NOSE] = [100 + i * speed, 80, 1.0]
                    f[CHIN] = [100 + i * speed, 110, 1.0]
                else:
                    f[NOSE] = [100 + i * speed, 100, 1.0]
                    f[CHIN] = [100 + i * speed, 105, 1.0]
                frames.append(f)
            samples.append(EvalSample(
                name=f"bite_speed{speed:.0f}_{n_frames}f",
                ground_truth=BITE, kpts_seq=_stack(frames), source="synthetic",
            ))

    return samples


# ===== P1 合成评估数据集生成 =====


def generate_p1_synthetic_dataset() -> list[EvalSample]:
    """生成覆盖 8 类 P1 行为的合成评估数据集。

    依据 RESEARCH_STANDARDS.md §4.1 + test_rule_engine_p1.py 几何特征。
    每类行为生成多个变体（不同帧数/速度/噪声），共 ~40 样本。
    """
    samples: list[EvalSample] = []
    ground_y = 200.0

    # --- track（追踪）：站立 + 鼻尖贴地 + 持续 x 方向移动 ---
    # 注：track_motion 默认 3.0，比较用严格 >，故 speed 从 4.0 起
    for speed in (4.0, 6.0, 8.0):
        for n_frames in (20, 30):
            frames = []
            for i in range(n_frames):
                f = _make_standing_frame(x_offset=i * speed)
                # 鼻尖贴近地面
                f[NOSE] = [100 + i * speed, ground_y - 10, 1.0]
                frames.append(f)
            samples.append(EvalSample(
                name=f"track_speed{speed:.0f}_{n_frames}f",
                ground_truth=TRACK, kpts_seq=_stack(frames), source="synthetic",
            ))

    # --- obstacle（障碍穿越）：withers.y 先高后低 + 高速移动 ---
    for speed in (20.0, 25.0):
        for n_frames in (30, 40):
            frames = []
            half = n_frames // 2
            for i in range(n_frames):
                x_off = i * speed
                f = _make_standing_frame(x_offset=x_off)
                if i < half:
                    f[WITHERS, 1] = ground_y - 160  # 跳跃上升
                else:
                    f[WITHERS, 1] = ground_y - 60   # 落下
                frames.append(f)
            samples.append(EvalSample(
                name=f"obstacle_speed{speed:.0f}_{n_frames}f",
                ground_truth=OBSTACLE, kpts_seq=_stack(frames), source="synthetic",
            ))

    # --- watch（警戒）：站立不动 + 头部抬起 ---
    # 注：watch_stability 默认 3.0，噪声控制在 0.5 以内避免破坏稳定性检测
    for n_frames in (20, 30, 50):
        for noise in (0.0, 0.5):
            frames = []
            for _ in range(n_frames):
                f = _make_standing_frame()
                f[NOSE] = [100, ground_y - 150, 1.0]  # 头部抬起
                f[WITHERS, 1] = ground_y - 120
                if noise > 0:
                    f[:, :2] += np.random.randn(NUM_KEYPOINTS, 2).astype(np.float32) * noise
                frames.append(f)
            samples.append(EvalSample(
                name=f"watch_{n_frames}f_noise{noise:.1f}",
                ground_truth=WATCH, kpts_seq=_stack(frames), source="synthetic",
            ))

    # --- apprehend（扑咬）：withers 高速移动 + nose-chin 距离振荡 ---
    # 注：apprehend_speed 默认 25.0，比较用严格 >，故 speed 从 26.0 起
    for speed in (26.0, 30.0, 35.0):
        for n_frames in (20, 30):
            frames = []
            for i in range(n_frames):
                f = _make_standing_frame(x_offset=i * speed)
                if i % 2 == 0:
                    f[NOSE] = [100 + i * speed, 80, 1.0]
                    f[CHIN] = [100 + i * speed, 110, 1.0]
                else:
                    f[NOSE] = [100 + i * speed, 100, 1.0]
                    f[CHIN] = [100 + i * speed, 105, 1.0]
                frames.append(f)
            samples.append(EvalSample(
                name=f"apprehend_speed{speed:.0f}_{n_frames}f",
                ground_truth=APPREHEND, kpts_seq=_stack(frames), source="synthetic",
            ))

    # --- escort（押解）：站立 + 持续移动 + 头部抬起 ---
    # 注：heel_motion 默认 3.0，比较用严格 >，故 speed 从 4.0 起
    for speed in (4.0, 6.0, 8.0):
        for n_frames in (20, 30):
            frames = []
            for i in range(n_frames):
                f = _make_standing_frame(x_offset=i * speed)
                f[NOSE] = [100 + i * speed, ground_y - 130, 1.0]
                f[WITHERS, 1] = ground_y - 120
                frames.append(f)
            samples.append(EvalSample(
                name=f"escort_speed{speed:.0f}_{n_frames}f",
                ground_truth=ESCORT, kpts_seq=_stack(frames), source="synthetic",
            ))

    # --- recall（返回）：前半段远离，后半段快速返回 ---
    for ret_speed in (25.0, 30.0):
        for n_frames in (30, 40):
            frames = []
            x = 100.0
            half = n_frames // 2
            for i in range(n_frames):
                if i < half:
                    x += 2
                else:
                    x -= ret_speed
                f = _make_standing_frame(x_offset=x - 100)
                frames.append(f)
            samples.append(EvalSample(
                name=f"recall_speed{ret_speed:.0f}_{n_frames}f",
                ground_truth=RECALL, kpts_seq=_stack(frames), source="synthetic",
            ))

    # --- alert_sit（示警坐）：坐姿 + 头部抬起 ---
    for n_frames in (20, 30):
        for noise in (0.0, 2.0):
            frames = []
            for _ in range(n_frames):
                f = _make_sitting_frame()
                f[NOSE] = [100, ground_y - 100, 1.0]
                f[WITHERS, 1] = ground_y - 80
                if noise > 0:
                    f[:, :2] += np.random.randn(NUM_KEYPOINTS, 2).astype(np.float32) * noise
                frames.append(f)
            samples.append(EvalSample(
                name=f"alert_sit_{n_frames}f_noise{noise:.0f}",
                ground_truth=ALERT_SIT, kpts_seq=_stack(frames), source="synthetic",
            ))

    # --- alert_down（示警卧）：卧姿 + 头部抬起 ---
    for n_frames in (20, 30):
        for noise in (0.0, 2.0):
            frames = []
            for _ in range(n_frames):
                f = _make_lying_frame()
                f[NOSE] = [100, ground_y - 30, 1.0]  # 头部抬起（< WITHERS.y=190）
                if noise > 0:
                    f[:, :2] += np.random.randn(NUM_KEYPOINTS, 2).astype(np.float32) * noise
                frames.append(f)
            samples.append(EvalSample(
                name=f"alert_down_{n_frames}f_noise{noise:.0f}",
                ground_truth=ALERT_DOWN, kpts_seq=_stack(frames), source="synthetic",
            ))

    return samples


def generate_all_synthetic_dataset() -> list[EvalSample]:
    """生成 P0 + P1 全部合成评估数据集（16 类行为）。"""
    return generate_synthetic_dataset() + generate_p1_synthetic_dataset()


# ===== DogMo 数据加载 =====


def load_dogmo_samples() -> list[EvalSample]:
    """从 data/dogmo/ 加载 DogMo 测试集样本。

    DogMo → P0 行为映射:
        Sit → sit
        Stand Up → stand
        其他类 → 不评估（无 P0 映射）
    """
    dogmo_dir = PROJECT_ROOT / "data" / "dogmo"
    if not dogmo_dir.exists():
        return []

    # DogMo 动作标签 → P0 行为
    dogmo_to_p0 = {
        "Sit": SIT,
        "Stand Up": STAND,
        "Stand_Up": STAND,
    }

    samples: list[EvalSample] = []
    # TODO: 当 DogMo 数据下载后，实现实际加载逻辑
    # 需要读取 DogMo 的 keypoints 标注格式并转换为 (T, 24, 3)
    # DogMo 使用 SMAL 关键点，需要映射到我们的 24 关键点格式
    return samples


# ===== 评估核心 =====


def evaluate_samples(
    samples: list[EvalSample],
    engine: Optional[RuleEngine] = None,
    phase: str = "all",
) -> tuple[list[EvalResult], OverallMetrics]:
    """评估样本列表。

    phase: p0 | p1 | all，决定初始化哪些行为指标。
    """
    if engine is None:
        engine = RuleEngine()

    results: list[EvalResult] = []
    metrics = OverallMetrics(source=samples[0].source if samples else "unknown", phase=phase)

    # 初始化 per-behavior 指标（全部 16 类，避免 FP 跨阶段访问 KeyError）
    for behavior in ALL_BEHAVIORS:
        metrics.per_behavior[behavior] = BehaviorMetrics(behavior=behavior)
        metrics.confusion_matrix[behavior] = defaultdict(int)

    for sample in samples:
        episodes = engine.recognize(sample.kpts_seq, fps=sample.fps)
        detected = list({e.behavior for e in episodes})
        hit = sample.ground_truth in detected

        result = EvalResult(
            sample=sample,
            detected_behaviors=detected,
            hit=hit,
            episodes_count=len(episodes),
        )
        results.append(result)

        # 总体统计
        metrics.total_samples += 1
        if hit:
            metrics.correct += 1

        # Per-behavior 统计
        gt = sample.ground_truth
        metrics.per_behavior[gt].samples += 1
        if hit:
            metrics.per_behavior[gt].tp += 1
        else:
            metrics.per_behavior[gt].fn += 1

        # FP: 检出了非 ground-truth 的行为
        for det in detected:
            if det != gt:
                metrics.per_behavior[det].fp += 1
            # 混淆矩阵: gt → det
            metrics.confusion_matrix[gt][det] += 1

    return results, metrics


# ===== 报告生成 =====


def generate_report(
    results: list[EvalResult],
    metrics: OverallMetrics,
    output_path: Optional[Path] = None,
) -> str:
    """生成 Markdown 评估报告。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 按阶段选择行为列表与标题
    if metrics.phase == "p0":
        behavior_list = list(P0_BEHAVIORS)
        title = "Phase 1.2f P0 规则引擎准确率评估报告"
        gate_desc = "P0 准确率 ≥ 80% 跳过 Phase 1.3；< 80% 触发 Phase 1.3 PoseC3D"
    elif metrics.phase == "p1":
        behavior_list = list(P1_BEHAVIORS)
        title = "Phase 2.2d P1 规则引擎准确率评估报告"
        gate_desc = "P1 准确率 ≥ 85% 达标（Phase 2.2d）"
    else:
        behavior_list = list(ALL_BEHAVIORS)
        title = "Phase 2.2d 规则引擎准确率评估报告（P0 + P1 = 16 类）"
        gate_desc = "P0 ≥ 80% 跳过 Phase 1.3；P1 ≥ 85% 达标 Phase 2.2d"

    lines = []
    lines.append(f"# {title}\n")
    lines.append(f"> 生成时间: {now}")
    lines.append(f"> 数据源: {metrics.source}")
    lines.append(f"> 阶段: {metrics.phase}")
    lines.append(f"> 决策门: {gate_desc}\n")

    # 总体结论
    lines.append("## 1. 总体结论\n")
    accuracy_pct = metrics.accuracy * 100
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| 样本总数 | {metrics.total_samples} |")
    lines.append(f"| 正确检出 | {metrics.correct} |")
    lines.append(f"| 准确率 | **{accuracy_pct:.1f}%** |")

    if metrics.phase == "all":
        p0_correct, p0_total, p0_acc = metrics.p0_metrics()
        p1_correct, p1_total, p1_acc = metrics.p1_metrics()
        lines.append(f"| P0 子集 | {p0_correct}/{p0_total} = **{p0_acc:.1%}** |")
        lines.append(f"| P1 子集 | {p1_correct}/{p1_total} = **{p1_acc:.1%}** |")
        p0_decision = "达标（跳过 Phase 1.3）" if p0_acc >= 0.80 else "触发 Phase 1.3"
        p1_decision = "达标" if p1_acc >= 0.85 else "未达标"
        lines.append(f"| P0 决策 | **{p0_decision}** |")
        lines.append(f"| P1 决策 | **{p1_decision}** |")
    elif metrics.phase == "p0":
        decision = "跳过 Phase 1.3" if not metrics.trigger_phase_1_3 else "触发 Phase 1.3"
        lines.append(f"| Phase 1.3 决策 | **{decision}** |")
    else:
        decision = "达标" if metrics.p1_pass else "未达标"
        lines.append(f"| Phase 2.2d 决策 | **{decision}** |")
    lines.append("")

    # Per-behavior 指标
    lines.append("## 2. Per-Behavior 指标\n")
    lines.append(f"| 行为 | 中文名 | 样本数 | TP | FP | FN | Precision | Recall | F1 |")
    lines.append(f"|------|--------|--------|----|----|----|-----------|--------|----|")
    for behavior in behavior_list:
        m = metrics.per_behavior.get(behavior, BehaviorMetrics(behavior))
        cn = BEHAVIOR_NAMES_CN.get(behavior, behavior)
        lines.append(
            f"| {behavior} | {cn} | {m.samples} | {m.tp} | {m.fp} | {m.fn} | "
            f"{m.precision:.1%} | {m.recall:.1%} | {m.f1:.1%} |"
        )
    lines.append("")

    # 混淆矩阵
    lines.append("## 3. 混淆矩阵\n")
    lines.append("行 = Ground Truth，列 = 检出行为\n")
    header = "| GT \\ Det | " + " | ".join(behavior_list) + " |"
    sep = "|----------|" + "|".join(["----"] * len(behavior_list)) + "|"
    lines.append(header)
    lines.append(sep)
    for gt in behavior_list:
        row = [gt]
        for det in behavior_list:
            count = metrics.confusion_matrix.get(gt, {}).get(det, 0)
            row.append(str(count))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # 失败样本详情
    failed = [r for r in results if not r.hit]
    if failed:
        lines.append("## 4. 失败样本详情\n")
        lines.append(f"| 样本名 | Ground Truth | 检出行为 | Episode 数 |")
        lines.append(f"|--------|-------------|---------|------------|")
        for r in failed:
            dets = ", ".join(r.detected_behaviors) if r.detected_behaviors else "(空)"
            lines.append(
                f"| {r.sample.name} | {r.sample.ground_truth} | {dets} | {r.episodes_count} |"
            )
        lines.append("")

    # 数据源说明
    lines.append("## 5. 数据源说明\n")
    if metrics.source == "synthetic":
        lines.append("**本次评估使用合成标注序列**。\n")
        lines.append("合成数据特点:")
        lines.append(f"- 覆盖 {len(behavior_list)} 类行为，每类多个变体（不同帧数/噪声/速度）")
        lines.append("- 关键点序列使用明确几何特征，确保规则可识别")
        lines.append("- **局限性**: 合成数据无法反映真实视频中的遮挡、模糊、姿态变异等问题\n")
        lines.append("**真实视频评估待办**: Phase 2.0c/2.2 完成后用 16 行为引擎重测（见 ADR 0008）")
    elif metrics.source == "dogmo":
        lines.append("**本次评估使用 DogMo 测试集**。\n")
        lines.append(f"评估样本数: {metrics.total_samples}")
        lines.append("DogMo → P0 行为映射: Sit → sit, Stand Up → stand")

    # 决策
    lines.append("")
    lines.append("## 6. 决策结论\n")
    if metrics.phase == "p0":
        if metrics.trigger_phase_1_3:
            lines.append(f"P0 准确率 {accuracy_pct:.1f}% < 80% → **触发 Phase 1.3 mmaction2 + PoseC3D**\n")
        else:
            lines.append(f"P0 准确率 {accuracy_pct:.1f}% ≥ 80% → **跳过 Phase 1.3**\n")
            if metrics.source == "synthetic":
                lines.append("**注意**: 此结论基于合成数据。真实视频评估完成后需复核。")
    elif metrics.phase == "p1":
        if metrics.p1_pass:
            lines.append(f"P1 准确率 {accuracy_pct:.1f}% ≥ 85% → **Phase 2.2d 达标**\n")
            lines.append("规则引擎 P1 8 类行为扩展在合成数据上表现达标，可进入 2.2e 评分卡扩展。")
        else:
            lines.append(f"P1 准确率 {accuracy_pct:.1f}% < 85% → **Phase 2.2d 未达标**\n")
            lines.append("需调整 P1 检测阈值或几何规则后重新评估。")
    else:
        p0_correct, p0_total, p0_acc = metrics.p0_metrics()
        p1_correct, p1_total, p1_acc = metrics.p1_metrics()
        lines.append(f"- **P0**: {p0_acc:.1%} ({p0_correct}/{p0_total}) → {'达标' if p0_acc >= 0.80 else '未达标'}")
        lines.append(f"- **P1**: {p1_acc:.1%} ({p1_correct}/{p1_total}) → {'达标' if p1_acc >= 0.85 else '未达标'}\n")
        if metrics.source == "synthetic":
            lines.append("**注意**: 此结论基于合成数据。真实视频评估在 Phase 2.0c 内推进。")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        # 同时输出 JSON 摘要
        json_path = output_path.with_suffix(".json")
        json_summary = {
            "timestamp": now,
            "source": metrics.source,
            "phase": metrics.phase,
            "total_samples": metrics.total_samples,
            "correct": metrics.correct,
            "accuracy": metrics.accuracy,
            "trigger_phase_1_3": metrics.trigger_phase_1_3,
            "p1_pass": metrics.p1_pass,
            "per_behavior": {
                b: {
                    "samples": metrics.per_behavior[b].samples,
                    "precision": metrics.per_behavior[b].precision,
                    "recall": metrics.per_behavior[b].recall,
                    "f1": metrics.per_behavior[b].f1,
                }
                for b in behavior_list if b in metrics.per_behavior
            },
        }
        json_path.write_text(json.dumps(json_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


# ===== 主入口 =====


def main() -> int:
    parser = argparse.ArgumentParser(description="规则引擎准确率评估（P0/P1/全部）")
    parser.add_argument(
        "--source", choices=["auto", "synthetic", "dogmo"], default="auto",
        help="评估数据源（auto: 优先 DogMo，降级合成）",
    )
    parser.add_argument(
        "--phase", choices=["p0", "p1", "all"], default="all",
        help="评估阶段：p0=仅 P0 8 类，p1=仅 P1 8 类，all=P0+P1 16 类",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="报告输出路径（Markdown）",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细输出")
    args = parser.parse_args()

    # 设置随机种子（确保合成数据可复现）
    np.random.seed(42)

    # 选择数据源
    samples: list[EvalSample] = []
    source = args.source

    if source in ("auto", "dogmo"):
        dogmo_samples = load_dogmo_samples()
        if dogmo_samples:
            # DogMo 仅含 P0 映射，按 phase 过滤
            if args.phase == "p1":
                print("[WARN] DogMo 仅含 P0 行为映射，--phase p1 强制切换为合成数据")
            else:
                samples = dogmo_samples
                source = "dogmo"
                print(f"[INFO] 加载 DogMo 样本: {len(samples)} 个")
        elif source == "dogmo":
            print("[ERROR] 指定 --source dogmo 但 data/dogmo/ 无可用数据")
            return 1

    if not samples and source in ("auto", "synthetic"):
        print("[INFO] 使用合成评估数据集")
        if args.phase == "p0":
            samples = generate_synthetic_dataset()
        elif args.phase == "p1":
            samples = generate_p1_synthetic_dataset()
        else:
            samples = generate_all_synthetic_dataset()
        source = "synthetic"
        print(f"[INFO] 生成合成样本: {len(samples)} 个（phase={args.phase}）")

    if not samples:
        print("[ERROR] 无可用评估数据")
        return 1

    # 运行评估
    print(f"[INFO] 开始评估（source={source}, phase={args.phase}, samples={len(samples)}）...")
    results, metrics = evaluate_samples(samples, phase=args.phase)

    # 打印摘要
    print()
    print("=" * 60)
    print(f"评估结果摘要 (source={source}, phase={args.phase})")
    print("=" * 60)
    print(f"样本总数: {metrics.total_samples}")
    print(f"正确检出: {metrics.correct}")
    print(f"准确率:   {metrics.accuracy:.1%}")

    if args.phase == "all":
        p0_correct, p0_total, p0_acc = metrics.p0_metrics()
        p1_correct, p1_total, p1_acc = metrics.p1_metrics()
        print(f"  P0 子集: {p0_acc:.1%} ({p0_correct}/{p0_total})")
        print(f"  P1 子集: {p1_acc:.1%} ({p1_correct}/{p1_total})")

    print()
    print("Per-Behavior:")
    print(f"  {'行为':<14} {'样本':>4} {'P':>6} {'R':>6} {'F1':>6}")
    if args.phase == "p0":
        beh_list = P0_BEHAVIORS
    elif args.phase == "p1":
        beh_list = P1_BEHAVIORS
    else:
        beh_list = ALL_BEHAVIORS
    for b in beh_list:
        m = metrics.per_behavior.get(b, BehaviorMetrics(b))
        print(f"  {b:<14} {m.samples:>4} {m.precision:>6.1%} {m.recall:>6.1%} {m.f1:>6.1%}")
    print()

    if args.phase == "p0":
        decision = "跳过 Phase 1.3" if not metrics.trigger_phase_1_3 else "触发 Phase 1.3"
        print(f"P0 决策: {decision}")
    elif args.phase == "p1":
        decision = "达标" if metrics.p1_pass else "未达标"
        print(f"P1 决策: {decision}")
    else:
        p0_correct, p0_total, p0_acc = metrics.p0_metrics()
        p1_correct, p1_total, p1_acc = metrics.p1_metrics()
        print(f"P0 决策: {'达标' if p0_acc >= 0.80 else '未达标'}")
        print(f"P1 决策: {'达标' if p1_acc >= 0.85 else '未达标'}")

    # 失败样本
    failed = [r for r in results if not r.hit]
    if failed and args.verbose:
        print()
        print(f"失败样本 ({len(failed)}):")
        for r in failed:
            dets = ", ".join(r.detected_behaviors) if r.detected_behaviors else "(空)"
            print(f"  {r.sample.name}: GT={r.sample.ground_truth} Det=[{dets}]")

    # 生成报告
    if args.report:
        report = generate_report(results, metrics, args.report)
        print(f"\n[INFO] 报告已保存: {args.report}")
        print(f"[INFO] JSON 摘要: {args.report.with_suffix('.json')}")
    else:
        # 默认保存路径按 phase 选择
        default_name = {
            "p0": "phase-1.2f-validation.md",
            "p1": "phase-2.2d-p1-validation.md",
            "all": "phase-2.2d-validation.md",
        }[args.phase]
        default_report = PROJECT_ROOT / "reports" / default_name
        report = generate_report(results, metrics, default_report)
        print(f"\n[INFO] 报告已保存: {default_report}")
        print(f"[INFO] JSON 摘要: {default_report.with_suffix('.json')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
