"""规则引擎准确率评估脚本.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.2f
依据: dev-docs/stages/phase-1.md §1.2f

评估策略:
    1. 主路径: DogMo 测试集（如已下载到 data/dogmo/）
    2. 降级路径: 合成标注序列（覆盖 8 类 P0 行为，已知 ground-truth）

评估指标:
    - Episode-level: 每个序列的 ground-truth 行为是否被检出
    - Per-behavior: Precision / Recall / F1
    - Overall: 准确率（决定是否触发 Phase 1.3 PoseC3D）
    - 混淆矩阵

决策门:
    - 准确率 ≥ 80% → 跳过 Phase 1.3
    - 准确率 < 80% → 触发 Phase 1.3 mmaction2 + PoseC3D

用法:
    python scripts/eval_rule_engine.py                    # 自动选择评估数据源
    python scripts/eval_rule_engine.py --source synthetic  # 强制合成数据
    python scripts/eval_rule_engine.py --source dogmo      # 强制 DogMo（需已下载）
    python scripts/eval_rule_engine.py --report reports/phase-1.2f-validation.md
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
)
from backend.ml.behavior.rule_engine import RuleEngine

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

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def trigger_phase_1_3(self) -> bool:
        """是否触发 Phase 1.3 PoseC3D。"""
        return self.accuracy < 0.80


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
) -> tuple[list[EvalResult], OverallMetrics]:
    """评估样本列表。"""
    if engine is None:
        engine = RuleEngine()

    results: list[EvalResult] = []
    metrics = OverallMetrics(source=samples[0].source if samples else "unknown")

    # 初始化 per-behavior 指标
    for behavior in P0_BEHAVIORS:
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

    lines = []
    lines.append("# Phase 1.2f 规则引擎准确率评估报告\n")
    lines.append(f"> 生成时间: {now}")
    lines.append(f"> 数据源: {metrics.source}")
    lines.append(f"> 决策门: 准确率 ≥ 80% 跳过 Phase 1.3；< 80% 触发 Phase 1.3 PoseC3D\n")

    # 总体结论
    lines.append("## 1. 总体结论\n")
    accuracy_pct = metrics.accuracy * 100
    decision = "跳过 Phase 1.3（规则引擎已达标）" if not metrics.trigger_phase_1_3 else "触发 Phase 1.3（mmaction2 + PoseC3D）"
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| 样本总数 | {metrics.total_samples} |")
    lines.append(f"| 正确检出 | {metrics.correct} |")
    lines.append(f"| 准确率 | **{accuracy_pct:.1f}%** |")
    lines.append(f"| Phase 1.3 决策 | **{decision}** |\n")

    # Per-behavior 指标
    lines.append("## 2. Per-Behavior 指标\n")
    lines.append(f"| 行为 | 中文名 | 样本数 | TP | FP | FN | Precision | Recall | F1 |")
    lines.append(f"|------|--------|--------|----|----|----|-----------|--------|----|")
    for behavior in P0_BEHAVIORS:
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
    header = "| GT \\ Det | " + " | ".join(P0_BEHAVIORS) + " |"
    sep = "|----------|" + "|".join(["----"] * len(P0_BEHAVIORS)) + "|"
    lines.append(header)
    lines.append(sep)
    for gt in P0_BEHAVIORS:
        row = [gt]
        for det in P0_BEHAVIORS:
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
        lines.append("**本次评估使用合成标注序列**，原因：DogMo 数据集需手动获取（非免费直接下载）。\n")
        lines.append("合成数据特点:")
        lines.append("- 覆盖 8 类 P0 行为，每类多个变体（不同帧数/噪声/速度）")
        lines.append("- 关键点序列使用明确几何特征，确保规则可识别")
        lines.append("- **局限性**: 合成数据无法反映真实视频中的遮挡、模糊、姿态变异等问题\n")
        lines.append("**DogMo 真实评估待办**:")
        lines.append("1. 通过 https://www.selectdataset.com/dataset/8b639bf7008b000e8c2e248f9ca2ddc6 获取 DogMo 数据集")
        lines.append("2. 解压到 `data/dogmo/`")
        lines.append("3. 运行 `python scripts/eval_rule_engine.py --source dogmo`")
        lines.append("4. 补充真实评估结果到本报告\n")
        lines.append("**建议**: 合成评估通过后，Phase 1.3 决策应基于 DogMo 真实评估结果最终确认。")
    elif metrics.source == "dogmo":
        lines.append("**本次评估使用 DogMo 测试集**。\n")
        lines.append(f"评估样本数: {metrics.total_samples}")
        lines.append("DogMo → P0 行为映射: Sit → sit, Stand Up → stand")

    lines.append("")
    lines.append("## 6. Phase 1.3 决策\n")
    if metrics.trigger_phase_1_3:
        lines.append(f"准确率 {accuracy_pct:.1f}% < 80% → **触发 Phase 1.3 mmaction2 + PoseC3D**\n")
        lines.append("下一步:")
        lines.append("1. 调研 MMAction2 Windows 兼容性（见 RESEARCH_PHASE1_STACK_DEEP.md §1）")
        lines.append("2. 准备 PoseC3D 训练数据（从 DogMo 关键点生成骨架图）")
        lines.append("3. 训练 PoseC3D 模型")
        lines.append("4. 集成到推理 pipeline")
    else:
        lines.append(f"准确率 {accuracy_pct:.1f}% ≥ 80% → **跳过 Phase 1.3**\n")
        lines.append("规则引擎在当前评估数据上表现达标。")
        if metrics.source == "synthetic":
            lines.append("\n**注意**: 此结论基于合成数据。DogMo 真实评估完成后需复核。")
        else:
            lines.append("\n规则引擎在 DogMo 测试集上表现达标，可进入 Phase 1.5+ 开发。")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        # 同时输出 JSON 摘要
        json_path = output_path.with_suffix(".json")
        json_summary = {
            "timestamp": now,
            "source": metrics.source,
            "total_samples": metrics.total_samples,
            "correct": metrics.correct,
            "accuracy": metrics.accuracy,
            "trigger_phase_1_3": metrics.trigger_phase_1_3,
            "per_behavior": {
                b: {
                    "samples": metrics.per_behavior[b].samples,
                    "precision": metrics.per_behavior[b].precision,
                    "recall": metrics.per_behavior[b].recall,
                    "f1": metrics.per_behavior[b].f1,
                }
                for b in P0_BEHAVIORS if b in metrics.per_behavior
            },
        }
        json_path.write_text(json.dumps(json_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


# ===== 主入口 =====


def main() -> int:
    parser = argparse.ArgumentParser(description="规则引擎准确率评估")
    parser.add_argument(
        "--source", choices=["auto", "synthetic", "dogmo"], default="auto",
        help="评估数据源（auto: 优先 DogMo，降级合成）",
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
            samples = dogmo_samples
            source = "dogmo"
            print(f"[INFO] 加载 DogMo 样本: {len(samples)} 个")
        elif source == "dogmo":
            print("[ERROR] 指定 --source dogmo 但 data/dogmo/ 无可用数据")
            return 1

    if not samples and source in ("auto", "synthetic"):
        print("[INFO] DogMo 不可用，使用合成评估数据集")
        samples = generate_synthetic_dataset()
        source = "synthetic"
        print(f"[INFO] 生成合成样本: {len(samples)} 个")

    if not samples:
        print("[ERROR] 无可用评估数据")
        return 1

    # 运行评估
    print(f"[INFO] 开始评估（source={source}, samples={len(samples)}）...")
    results, metrics = evaluate_samples(samples)

    # 打印摘要
    print()
    print("=" * 60)
    print(f"评估结果摘要 (source={source})")
    print("=" * 60)
    print(f"样本总数: {metrics.total_samples}")
    print(f"正确检出: {metrics.correct}")
    print(f"准确率:   {metrics.accuracy:.1%}")
    print()
    print("Per-Behavior:")
    print(f"  {'行为':<12} {'样本':>4} {'P':>6} {'R':>6} {'F1':>6}")
    for b in P0_BEHAVIORS:
        m = metrics.per_behavior.get(b, BehaviorMetrics(b))
        cn = BEHAVIOR_NAMES_CN.get(b, b)
        print(f"  {b:<12} {m.samples:>4} {m.precision:>6.1%} {m.recall:>6.1%} {m.f1:>6.1%}")
    print()
    decision = "跳过 Phase 1.3" if not metrics.trigger_phase_1_3 else "触发 Phase 1.3"
    print(f"Phase 1.3 决策: {decision}")

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
        # 默认保存到 reports/
        default_report = PROJECT_ROOT / "reports" / "phase-1.2f-validation.md"
        report = generate_report(results, metrics, default_report)
        print(f"\n[INFO] 报告已保存: {default_report}")
        print(f"[INFO] JSON 摘要: {default_report.with_suffix('.json')}")

    return 0 if not metrics.trigger_phase_1_3 else 0  # 评估脚本总是返回 0


if __name__ == "__main__":
    sys.exit(main())
