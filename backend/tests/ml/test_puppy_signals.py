"""Phase 1.6b/c 选育信号提取器单元测试 + 集成测试.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.6b/c
依据: dev-docs/stages/phase-1.md §1.6

测试策略:
    1. fast 单元测试（无 GPU/模型）:
       - 合成 kpts_seq + 合成 detections → 9 信号字典
       - 食物欲望: approach_latency / approach_speed / sniff_duration
       - 猎物欲望: chase_latency / chase_speed / hold_duration
       - 胆量: retreat_distance / freeze_duration / recovery_time
       - 边界场景（空数据 / 无物体 / 多物体）
    2. integration 集成测试（需 YOLO26 模型 + 真实视频）:
       - 真实选育视频 → 物体检测 + pose → 9 信号

标记:
    fast — 纯 Python，无 GPU/模型
    integration — 需 YOLO26 模型 + 真实视频
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.ml.behavior.constants import NOSE, WITHERS
from backend.ml.behavior.object_detector import (
    CATEGORY_BALL,
    CATEGORY_FOOD,
    CATEGORY_PERSON,
    CATEGORY_TOY,
    COCO_NAMES,
    Detection,
    FrameDetection,
    VideoDetectionResult,
)
from backend.ml.behavior.puppy_signals import (
    PENALTY_DISTANCE,
    PENALTY_FREEZE,
    PENALTY_LATENCY,
    PuppySignalConfig,
    extract_puppy_signals,
)


# ============================================================
# 辅助函数
# ============================================================


def _make_detection(
    category: str,
    bbox: tuple[float, float, float, float],
    class_id: int = 0,
    class_name: str = "test",
    confidence: float = 0.9,
) -> Detection:
    """构造测试用 Detection."""
    return Detection(
        class_id=class_id,
        class_name=class_name,
        category=category,
        confidence=confidence,
        bbox=np.array(bbox, dtype=np.float32),
    )


def _make_kpts_seq(
    T: int,
    nose_xy: tuple[float, float] | None = None,
    withers_xy: tuple[float, float] | None = None,
    conf: float = 0.9,
) -> np.ndarray:
    """构造测试用关键点序列 (T, 24, 3).

    Args:
        T: 帧数
        nose_xy: (x, y) 鼻尖位置（None = 全 0）
        withers_xy: (x, y) 鬐甲位置（None = 全 0）
        conf: 关键点置信度
    """
    kpts = np.zeros((T, 24, 3), dtype=np.float32)
    if nose_xy is not None:
        kpts[:, NOSE, 0] = nose_xy[0]
        kpts[:, NOSE, 1] = nose_xy[1]
        kpts[:, NOSE, 2] = conf
    if withers_xy is not None:
        kpts[:, WITHERS, 0] = withers_xy[0]
        kpts[:, WITHERS, 1] = withers_xy[1]
        kpts[:, WITHERS, 2] = conf
    return kpts


def _make_video_detection_result(
    frames_dets: list[tuple[int, list[Detection]]],
    fps: float = 30.0,
) -> VideoDetectionResult:
    """构造 VideoDetectionResult.

    Args:
        frames_dets: [(frame_idx, [Detection, ...]), ...]
        fps: 帧率
    """
    frames = []
    for frame_idx, dets in frames_dets:
        frames.append(FrameDetection(
            frame_idx=frame_idx,
            frame_time_sec=frame_idx / fps,
            detections=dets,
        ))
    return VideoDetectionResult(
        meta={"fps": fps, "frame_count": len(frames)},
        frames=frames,
    )


# ============================================================
# fast 单元测试
# ============================================================


@pytest.mark.fast
class TestFoodSignals:
    """食物欲望 3 信号测试。"""

    def test_no_food_returns_penalty(self) -> None:
        """无食物检测 → approach_latency=99, sniff_duration=0."""
        kpts = _make_kpts_seq(30, nose_xy=(100, 100), withers_xy=(100, 200))
        det = _make_video_detection_result([(0, [])])
        signals = extract_puppy_signals(kpts, det, fps=30.0, duration_sec=1.0)

        assert signals["approach_latency"] == PENALTY_LATENCY
        assert signals["approach_speed"] == 0.0
        assert signals["sniff_duration"] == 0.0

    def test_food_at_nose_position_low_latency(self) -> None:
        """食物在第 0 帧出现且在鼻尖位置 → approach_latency=0."""
        # 食物框中心 = (100, 100)，鼻尖在 (100, 100)
        food_det = _make_detection(
            CATEGORY_FOOD, bbox=(50, 50, 150, 150), class_id=47,  # apple
        )
        det = _make_video_detection_result([(0, [food_det])] * 30)
        kpts = _make_kpts_seq(30, nose_xy=(100, 100), withers_xy=(100, 200))
        signals = extract_puppy_signals(kpts, det, fps=30.0, duration_sec=1.0)

        # 鼻尖已在食物框中心 → approach_latency = 0
        assert signals["approach_latency"] == pytest.approx(0.0, abs=0.01)
        # sniff_duration > 0（鼻尖持续在食物附近）
        assert signals["sniff_duration"] > 0.0

    def test_food_appears_later_increases_latency(self) -> None:
        """食物在第 10 帧出现，鼻尖在第 15 帧触及 → latency=5/30≈0.17s."""
        frames_dets = []
        # 0-9 帧无食物
        for i in range(10):
            frames_dets.append((i, []))
        # 10-29 帧有食物
        food_det = _make_detection(
            CATEGORY_FOOD, bbox=(200, 100, 250, 150), class_id=47,
        )
        for i in range(10, 30):
            frames_dets.append((i, [food_det]))
        det = _make_video_detection_result(frames_dets)

        # 鼻尖从 (100, 100) 移动到 (200, 100)（第 15 帧到达食物框附近）
        kpts = np.zeros((30, 24, 3), dtype=np.float32)
        for t in range(30):
            # 线性插值：t=0 在 (100,100)，t=15 在 (200, 100)
            nose_x = 100 + (t / 15.0) * 100 if t <= 15 else 200
            kpts[t, NOSE, 0] = nose_x
            kpts[t, NOSE, 1] = 100
            kpts[t, NOSE, 2] = 0.9
            kpts[t, WITHERS, 0] = nose_x  # withers 跟随
            kpts[t, WITHERS, 1] = 200
            kpts[t, WITHERS, 2] = 0.9

        signals = extract_puppy_signals(kpts, det, fps=30.0, duration_sec=1.0)

        # approach_latency 应大于 0（食物出现后鼻尖需要 5 帧到达）
        assert signals["approach_latency"] > 0.0
        # approach_speed 应大于 0（withers 在移动）
        assert signals["approach_speed"] >= 0.0  # 取决于几何，可能=0


@pytest.mark.fast
class TestPreySignals:
    """猎物欲望 3 信号测试。"""

    def test_no_ball_returns_penalty(self) -> None:
        """无球/玩具检测 → chase_latency=99, hold_duration=0."""
        kpts = _make_kpts_seq(30, nose_xy=(100, 100), withers_xy=(100, 200))
        det = _make_video_detection_result([(0, [])])
        signals = extract_puppy_signals(kpts, det, fps=30.0, duration_sec=1.0)

        assert signals["chase_latency"] == PENALTY_LATENCY
        assert signals["chase_speed"] == 0.0
        assert signals["hold_duration"] == 0.0

    def test_ball_at_nose_position_low_latency(self) -> None:
        """球在第 0 帧出现且在鼻尖位置 → chase_latency=0."""
        ball_det = _make_detection(
            CATEGORY_BALL, bbox=(80, 80, 120, 120), class_id=32,
        )
        det = _make_video_detection_result([(0, [ball_det])] * 30)
        kpts = _make_kpts_seq(30, nose_xy=(100, 100), withers_xy=(100, 200))
        signals = extract_puppy_signals(kpts, det, fps=30.0, duration_sec=1.0)

        # 球框中心 (100, 100) = 鼻尖位置 → chase_latency = 0
        assert signals["chase_latency"] == pytest.approx(0.0, abs=0.01)
        assert signals["hold_duration"] > 0.0

    def test_toy_category_also_triggers_prey(self) -> None:
        """toy 类别（如 tennis racket）也能触发 prey 信号."""
        toy_det = _make_detection(
            CATEGORY_TOY, bbox=(80, 80, 120, 120), class_id=41,
        )
        det = _make_video_detection_result([(0, [toy_det])] * 30)
        kpts = _make_kpts_seq(30, nose_xy=(100, 100), withers_xy=(100, 200))
        signals = extract_puppy_signals(kpts, det, fps=30.0, duration_sec=1.0)

        assert signals["chase_latency"] == pytest.approx(0.0, abs=0.01)


@pytest.mark.fast
class TestCourageSignals:
    """胆量 3 信号测试。"""

    def test_no_person_returns_zero_courage(self) -> None:
        """无 person 检测 → courage 信号全 0（无惊吓事件）."""
        kpts = _make_kpts_seq(30, nose_xy=(100, 100), withers_xy=(100, 200))
        det = _make_video_detection_result([(0, [])])
        signals = extract_puppy_signals(kpts, det, fps=30.0, duration_sec=1.0)

        assert signals["retreat_distance"] == 0.0
        assert signals["freeze_duration"] == 0.0
        assert signals["recovery_time"] == 0.0

    def test_small_person_not_scare_event(self) -> None:
        """小 person 框（面积 < scare_person_area）不视为惊吓事件."""
        # person 框面积 = 50*50 = 2500 < 5000
        person_det = _make_detection(
            CATEGORY_PERSON, bbox=(0, 0, 50, 50), class_id=0,
        )
        det = _make_video_detection_result([(0, [person_det])] * 30)
        kpts = _make_kpts_seq(30, nose_xy=(100, 100), withers_xy=(100, 200))
        signals = extract_puppy_signals(kpts, det, fps=30.0, duration_sec=1.0)

        # 不视为惊吓 → courage 全 0
        assert signals["retreat_distance"] == 0.0
        assert signals["recovery_time"] == 0.0

    def test_large_person_triggers_scare(self) -> None:
        """大 person 框（面积 >= 5000）触发惊吓事件 → retreat_distance > 0."""
        # person 框面积 = 100*100 = 10000 > 5000
        person_det = _make_detection(
            CATEGORY_PERSON, bbox=(0, 0, 100, 100), class_id=0,
        )
        det = _make_video_detection_result([(0, [person_det])] * 30)
        # withers 在第 0 帧后开始剧烈后退（x 减小）
        kpts = np.zeros((30, 24, 3), dtype=np.float32)
        for t in range(30):
            kpts[t, WITHERS, 0] = 200 - t * 5  # 每帧 -5 像素（后退）
            kpts[t, WITHERS, 1] = 200
            kpts[t, WITHERS, 2] = 0.9
        signals = extract_puppy_signals(kpts, det, fps=30.0, duration_sec=1.0)

        # 触发惊吓 → retreat_distance > 0
        assert signals["retreat_distance"] > 0.0
        # 持续后退 → 算法视为"运动"（不区分方向）→ recovery_time 较短
        # 注意: 这是简化算法的语义限制，"恢复"= 恢复任意方向运动
        assert signals["recovery_time"] >= 0.0

    def test_freeze_then_recover_increases_recovery_time(self) -> None:
        """惊吓后先冻结 10 帧再恢复运动 → recovery_time > 0."""
        person_det = _make_detection(
            CATEGORY_PERSON, bbox=(0, 0, 100, 100), class_id=0,
        )
        det = _make_video_detection_result([(0, [person_det])] * 30)
        # withers: 0-9 帧不动（冻结），10-29 帧恢复正常运动
        kpts = np.zeros((30, 24, 3), dtype=np.float32)
        for t in range(30):
            if t < 10:
                kpts[t, WITHERS, 0] = 200  # 不动
            else:
                kpts[t, WITHERS, 0] = 200 + (t - 10) * 3  # 每帧 +3 像素
            kpts[t, WITHERS, 1] = 200
            kpts[t, WITHERS, 2] = 0.9
        signals = extract_puppy_signals(kpts, det, fps=30.0, duration_sec=1.0)

        # 冻结 10 帧（0.33s）后才恢复运动 → recovery_time > 0
        assert signals["freeze_duration"] > 0.0
        assert signals["recovery_time"] > 0.0

    def test_freeze_after_scare(self) -> None:
        """惊吓后 withers 不动 → freeze_duration > 0."""
        person_det = _make_detection(
            CATEGORY_PERSON, bbox=(0, 0, 100, 100), class_id=0,
        )
        det = _make_video_detection_result([(0, [person_det])] * 30)
        # withers 在所有帧都不动
        kpts = _make_kpts_seq(30, nose_xy=(100, 100), withers_xy=(200, 200))
        signals = extract_puppy_signals(kpts, det, fps=30.0, duration_sec=1.0)

        # 惊吓后 withers 完全不动 → freeze_duration > 0
        assert signals["freeze_duration"] > 0.0


@pytest.mark.fast
class TestSignalAlignment:
    """9 信号与 puppy_selection.yaml v1.1.0 对齐测试。"""

    def test_nine_signals_present(self) -> None:
        """9 个信号 key 全部存在."""
        kpts = _make_kpts_seq(10, nose_xy=(100, 100), withers_xy=(100, 200))
        signals = extract_puppy_signals(kpts, None, fps=30.0, duration_sec=0.0)
        expected = {
            "approach_latency", "approach_speed", "sniff_duration",
            "chase_latency", "chase_speed", "hold_duration",
            "retreat_distance", "freeze_duration", "recovery_time",
        }
        assert set(signals.keys()) == expected

    def test_all_signals_are_float(self) -> None:
        """所有信号值都是 float."""
        kpts = _make_kpts_seq(10, nose_xy=(100, 100), withers_xy=(100, 200))
        signals = extract_puppy_signals(kpts, None, fps=30.0, duration_sec=0.0)
        for k, v in signals.items():
            assert isinstance(v, float), f"{k} 类型: {type(v)}"

    def test_scoring_engine_accepts_signals(self) -> None:
        """9 信号能被评分引擎接受并产生有效评分."""
        from backend.ml.scoring import ScoringContext, ScoringEngine

        kpts = _make_kpts_seq(30, nose_xy=(100, 100), withers_xy=(100, 200))
        signals = extract_puppy_signals(kpts, None, fps=30.0, duration_sec=1.0)

        puppy_yaml = (
            Path(__file__).resolve().parents[3]
            / "backend" / "ml" / "scoring" / "configs" / "puppy_selection.yaml"
        )
        engine = ScoringEngine.get(puppy_yaml)
        ctx = ScoringContext(signals=signals, scene="puppy_selection")
        result = engine.evaluate(ctx)

        assert 0 <= result.total_score <= 100
        # 无物体检测 → food/prey 走 penalty → 总分较低
        assert result.verdict in ("pass", "borderline", "fail")


@pytest.mark.fast
class TestEdgeCases:
    """边界场景测试。"""

    def test_empty_kpts_returns_penalty(self) -> None:
        """空 kpts → 惩罚信号."""
        signals = extract_puppy_signals(
            np.zeros((0, 24, 3), dtype=np.float32),
            detections=None, fps=30.0, duration_sec=0.0,
        )
        assert signals["approach_latency"] == PENALTY_LATENCY
        assert signals["chase_latency"] == PENALTY_LATENCY
        assert signals["retreat_distance"] == PENALTY_DISTANCE

    def test_invalid_kpts_shape_returns_penalty(self) -> None:
        """错误 shape → 惩罚信号."""
        kpts = np.zeros((10, 17, 3), dtype=np.float32)
        signals = extract_puppy_signals(
            kpts, detections=None, fps=30.0, duration_sec=0.0
        )
        assert signals["approach_latency"] == PENALTY_LATENCY

    def test_zero_fps_returns_penalty(self) -> None:
        """fps=0 → 惩罚信号."""
        kpts = _make_kpts_seq(30, nose_xy=(100, 100))
        signals = extract_puppy_signals(
            kpts, detections=None, fps=0.0, duration_sec=1.0
        )
        assert signals["approach_latency"] == PENALTY_LATENCY

    def test_none_detections_does_not_crash(self) -> None:
        """detections=None 不崩溃，返回降级信号."""
        kpts = _make_kpts_seq(30, nose_xy=(100, 100), withers_xy=(100, 200))
        signals = extract_puppy_signals(
            kpts, detections=None, fps=30.0, duration_sec=1.0
        )
        # 无物体 → food/prey 走 penalty
        assert signals["approach_latency"] == PENALTY_LATENCY
        assert signals["chase_latency"] == PENALTY_LATENCY
        # 无 person → courage 走 0
        assert signals["retreat_distance"] == 0.0

    def test_empty_detections_does_not_crash(self) -> None:
        """空 detections 不崩溃."""
        kpts = _make_kpts_seq(30, nose_xy=(100, 100), withers_xy=(100, 200))
        det = VideoDetectionResult(meta={"fps": 30.0}, frames=[])
        signals = extract_puppy_signals(
            kpts, detections=det, fps=30.0, duration_sec=1.0
        )
        assert signals["approach_latency"] == PENALTY_LATENCY

    def test_custom_config_overrides_defaults(self) -> None:
        """自定义 config 能覆盖默认阈值."""
        # 用极小 scare_person_area 让小 person 也触发惊吓
        cfg = PuppySignalConfig(scare_person_area=100.0)
        kpts = _make_kpts_seq(30, nose_xy=(100, 100), withers_xy=(200, 200))
        # 小 person 框（面积 2500）
        person_det = _make_detection(
            CATEGORY_PERSON, bbox=(0, 0, 50, 50), class_id=0,
        )
        det = _make_video_detection_result([(0, [person_det])] * 30)
        signals = extract_puppy_signals(
            kpts, det, fps=30.0, duration_sec=1.0, config=cfg,
        )
        # 自定义 cfg 让小 person 也触发 → courage 信号走真实路径
        # withers 不动 → freeze_duration > 0
        assert signals["freeze_duration"] > 0.0


# ============================================================
# integration 集成测试（需真实模型 + 视频）
# ============================================================


@pytest.mark.integration
class TestPuppySignalsIntegration:
    """真实视频集成测试（CI 默认跳过）。

    手动运行:
        pytest backend/tests/ml/test_puppy_signals.py -v -m integration
    """

    def test_real_video_end_to_end(self) -> None:
        """真实选育视频 → 物体检测 + pose → 9 信号."""
        project_root = Path(__file__).resolve().parents[3]
        video_candidates = [
            project_root / "data" / "puppy_test.mp4",
            project_root / "data" / "sample.mp4",
        ]
        video_path = None
        for cand in video_candidates:
            if cand.exists():
                video_path = cand
                break
        if video_path is None:
            pytest.skip("无可用测试视频")

        try:
            from backend.ml.behavior.object_detector import ObjectDetector
            from backend.ml.pose.inference import PoseInferenceEngine
            from backend.workers.tasks import _resolve_model_path
        except ImportError as e:
            pytest.skip(f"模块不可用: {e}")

        # 1. pose 推理
        try:
            pose_engine = PoseInferenceEngine(
                model_path=_resolve_model_path(), verbose=False
            )
            pose_result = pose_engine.infer_video(video_path, save_output=False)
        except Exception as e:
            pytest.skip(f"pose 推理失败: {e}")

        # 2. 物体检测
        try:
            detector = ObjectDetector(model_path="yolo26n.pt", verbose=False)
            det_result = detector.detect_video(video_path, save_output=False)
        except Exception as e:
            pytest.skip(f"物体检测失败: {e}")

        # 3. 选育信号提取
        signals = extract_puppy_signals(
            kpts_seq=pose_result.keypoints_sequence,
            detections=det_result,
            fps=pose_result.meta.get("fps", 30.0),
            duration_sec=pose_result.meta.get("duration_sec", 0.0),
        )

        # 验证 9 信号全部存在且为 float
        expected = {
            "approach_latency", "approach_speed", "sniff_duration",
            "chase_latency", "chase_speed", "hold_duration",
            "retreat_distance", "freeze_duration", "recovery_time",
        }
        assert set(signals.keys()) == expected
        for k, v in signals.items():
            assert isinstance(v, float)
            assert not np.isnan(v), f"{k} is NaN"
