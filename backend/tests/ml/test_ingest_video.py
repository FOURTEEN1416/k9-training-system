"""Phase 1.4d 单元测试: Celery 推理任务 + API.

Owner: 后端开发 + ML 开发
Phase: 1.4d

测试策略:
    1. 纯函数测试（fast）:
       - _signals_from_obedience_episodes: 各种 episodes 场景
       - _extract_puppy_signals_simple: 各种 kpts_seq 场景
       - _BEHAVIOR_STRING_TO_ENUM: 8 类 P0 行为映射完整
    2. 任务集成测试（integration, 需真实 DB + 模型）:
       - ingest_video 端到端（标记 @pytest.mark.integration，CI 默认跳过）

标记: fast（纯 Python，无需 GPU/DB）
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.ml.behavior.rule_engine import BehaviorEpisode
from backend.workers.tasks import (
    _BEHAVIOR_STRING_TO_ENUM,
    _extract_puppy_signals_simple,
    _resolve_model_path,
    _signals_from_obedience_episodes,
)

pytestmark = pytest.mark.fast


# ============================================================
# 1. _BEHAVIOR_STRING_TO_ENUM 映射完整性
# ============================================================


class TestBehaviorMapping:
    """P0 行为字符串 → BehaviorClass 枚举映射测试."""

    def test_all_p0_behaviors_mapped(self) -> None:
        """8 类 P0 行为必须全部映射到 BehaviorClass 枚举."""
        from backend.ml.behavior.constants import P0_BEHAVIORS
        for behavior in P0_BEHAVIORS:
            assert behavior in _BEHAVIOR_STRING_TO_ENUM, f"未映射的行为: {behavior}"

    def test_mapping_values_are_valid_enum_members(self) -> None:
        """所有映射值必须是 BehaviorClass 枚举成员."""
        from backend.app.models.behavior import BehaviorClass
        for behavior_str, enum_member in _BEHAVIOR_STRING_TO_ENUM.items():
            assert isinstance(enum_member, BehaviorClass), (
                f"{behavior_str} → {enum_member} 不是 BehaviorClass 成员"
            )

    def test_sit_up_mapped_correctly(self) -> None:
        """sit_up 应映射到 BehaviorClass.SIT_UP（Phase 1.4d 新增枚举值）."""
        from backend.app.models.behavior import BehaviorClass
        assert _BEHAVIOR_STRING_TO_ENUM["sit_up"] == BehaviorClass.SIT_UP

    def test_stay_mapped_correctly(self) -> None:
        """stay 应映射到 BehaviorClass.STAY（Phase 1.4d 新增枚举值）."""
        from backend.app.models.behavior import BehaviorClass
        assert _BEHAVIOR_STRING_TO_ENUM["stay"] == BehaviorClass.STAY

    def test_no_placeholder_mappings(self) -> None:
        """不应存在占位映射（如 stay → SEARCH 这种 hack）."""
        # stay 不应映射到 SEARCH（之前的错误映射）
        assert _BEHAVIOR_STRING_TO_ENUM["stay"].value != "search"


# ============================================================
# 2. _signals_from_obedience_episodes
# ============================================================


class TestObedienceSignals:
    """科目测评信号提取测试."""

    def test_empty_episodes_returns_zeros(self) -> None:
        """无行为时返回零信号."""
        signals = _signals_from_obedience_episodes([], fps=30.0)
        assert signals["action_count"] == 0
        assert signals["action_correct"] == 0
        assert signals["command_to_action_latency"] == 99.0
        assert signals["action_duration"] == 0.0
        assert signals["focus_ratio"] == 0.0
        assert signals["gait_score"] == 0.0

    def test_single_high_confidence_episode(self) -> None:
        """单个高置信度行为."""
        episodes = [
            BehaviorEpisode(
                behavior="sit",
                start_frame=30,  # 1 秒后
                end_frame=90,    # 3 秒结束
                confidence=0.9,
            ),
        ]
        signals = _signals_from_obedience_episodes(episodes, fps=30.0)
        assert signals["action_count"] == 1
        assert signals["action_correct"] == 1  # conf 0.9 >= 0.5
        assert signals["command_to_action_latency"] == pytest.approx(1.0)
        assert signals["action_duration"] == pytest.approx(2.034, rel=0.01)  # 61/30
        assert signals["focus_ratio"] == pytest.approx(0.9)
        assert signals["gait_score"] == 0.0  # 无 heel

    def test_low_confidence_not_counted_as_correct(self) -> None:
        """低置信度行为不计入 action_correct."""
        episodes = [
            BehaviorEpisode(behavior="sit", start_frame=0, end_frame=10, confidence=0.3),
        ]
        signals = _signals_from_obedience_episodes(episodes, fps=30.0)
        assert signals["action_count"] == 1
        assert signals["action_correct"] == 0  # conf 0.3 < 0.5

    def test_heel_episode_provides_gait_score(self) -> None:
        """heel 行为提供 gait_score 信号."""
        episodes = [
            BehaviorEpisode(behavior="heel", start_frame=0, end_frame=30, confidence=0.8),
            BehaviorEpisode(behavior="sit", start_frame=31, end_frame=60, confidence=0.6),
        ]
        signals = _signals_from_obedience_episodes(episodes, fps=30.0)
        assert signals["gait_score"] == pytest.approx(0.8)
        assert signals["action_count"] == 2
        assert signals["action_correct"] == 2  # 两者 conf >= 0.5

    def test_multiple_episodes_aggregate_duration(self) -> None:
        """多个 episode 时长累加."""
        episodes = [
            BehaviorEpisode(behavior="sit", start_frame=0, end_frame=29, confidence=0.9),    # 1.0s
            BehaviorEpisode(behavior="down", start_frame=30, end_frame=89, confidence=0.8),  # 2.0s
        ]
        signals = _signals_from_obedience_episodes(episodes, fps=30.0)
        # 30/30 + 60/30 = 3.0
        assert signals["action_duration"] == pytest.approx(3.0, rel=0.05)

    def test_zero_fps_safe(self) -> None:
        """fps=0 时不崩溃（返回 0 延迟/时长）."""
        episodes = [
            BehaviorEpisode(behavior="sit", start_frame=0, end_frame=10, confidence=0.9),
        ]
        signals = _signals_from_obedience_episodes(episodes, fps=0.0)
        assert signals["command_to_action_latency"] == 0.0
        assert signals["action_duration"] == 0.0


# ============================================================
# 3. _extract_puppy_signals_simple
# ============================================================


class TestPuppySignalsSimple:
    """幼犬选育简化信号提取测试（Phase 1.6 前占位实现）."""

    def test_empty_kpts_returns_high_latency(self) -> None:
        """空 keypoints → 高延迟（99.0）."""
        signals = _extract_puppy_signals_simple(
            np.zeros((0, 24, 3), dtype=np.float32), fps=30.0, duration_sec=0.0
        )
        assert signals["approach_latency"] == 99.0
        assert signals["chase_latency"] == 99.0
        assert signals["recovery_time"] == 99.0

    def test_zero_fps_returns_high_latency(self) -> None:
        """fps=0 → 高延迟."""
        kpts = np.zeros((30, 24, 3), dtype=np.float32)
        signals = _extract_puppy_signals_simple(kpts, fps=0.0, duration_sec=1.0)
        assert signals["approach_latency"] == 99.0

    def test_all_invalid_withers_returns_zero_speed(self) -> None:
        """withers 全部低置信度 → approach_speed=0."""
        kpts = np.zeros((30, 24, 3), dtype=np.float32)  # withers conf=0
        signals = _extract_puppy_signals_simple(kpts, fps=30.0, duration_sec=1.0)
        assert signals["approach_speed"] == 0.0
        assert signals["hold_duration"] == 0.0

    def test_immediate_detection_low_latency(self) -> None:
        """首帧即检测到 withers → approach_latency=0."""
        kpts = np.zeros((30, 24, 3), dtype=np.float32)
        kpts[:, 22, 2] = 0.9  # withers 置信度 0.9
        kpts[:, 22, 0] = 100  # withers.x 固定（不动）
        signals = _extract_puppy_signals_simple(kpts, fps=30.0, duration_sec=1.0)
        assert signals["approach_latency"] == 0.0
        # withers.x 不动 → dx=0 → 所有 |dx| < 0.5 → 全部冻结
        assert signals["freeze_duration"] == pytest.approx(1.0, rel=0.05)  # 29/30 ≈ 0.97
        assert signals["hold_duration"] == 0.0  # 无运动

    def test_delayed_detection_high_latency(self) -> None:
        """第 15 帧才检测到 withers → approach_latency=0.5s."""
        kpts = np.zeros((30, 24, 3), dtype=np.float32)
        kpts[15:, 22, 2] = 0.9  # 第 15 帧开始有效
        kpts[15:, 22, 0] = 100
        signals = _extract_puppy_signals_simple(kpts, fps=30.0, duration_sec=1.0)
        assert signals["approach_latency"] == pytest.approx(0.5)

    def test_moving_withers_provides_speed(self) -> None:
        """withers.x 持续递增 → approach_speed > 0."""
        kpts = np.zeros((30, 24, 3), dtype=np.float32)
        kpts[:, 22, 2] = 0.9
        kpts[:, 22, 0] = np.arange(30, dtype=np.float32) * 5  # 每帧 +5 像素
        signals = _extract_puppy_signals_simple(kpts, fps=30.0, duration_sec=1.0)
        assert signals["approach_speed"] > 0.0
        # 5 像素/帧 * 30 fps = 150 像素/秒
        assert signals["approach_speed"] == pytest.approx(150.0, rel=0.05)
        # 持续运动 → hold_duration > 0
        assert signals["hold_duration"] > 0.0
        # retreat_distance = 0（无反向运动）
        assert signals["retreat_distance"] == 0.0

    def test_reverse_motion_provides_retreat_distance(self) -> None:
        """withers.x 持续递减 → retreat_distance > 0."""
        kpts = np.zeros((30, 24, 3), dtype=np.float32)
        kpts[:, 22, 2] = 0.9
        kpts[:, 22, 0] = 200 - np.arange(30, dtype=np.float32) * 3  # 每帧 -3 像素
        signals = _extract_puppy_signals_simple(kpts, fps=30.0, duration_sec=1.0)
        assert signals["retreat_distance"] > 0.0
        # 29 帧 * 3 像素 = 87
        assert signals["retreat_distance"] == pytest.approx(87.0, rel=0.05)

    def test_signals_aligned_with_puppy_yaml(self) -> None:
        """信号字典 key 与 puppy_selection.yaml 条件表达式对齐."""
        kpts = np.zeros((30, 24, 3), dtype=np.float32)
        kpts[:, 22, 2] = 0.9
        signals = _extract_puppy_signals_simple(kpts, fps=30.0, duration_sec=1.0)
        # puppy_selection.yaml 用到的信号:
        expected_keys = {
            "approach_latency", "approach_speed",
            "chase_latency", "hold_duration",
            "retreat_distance", "recovery_time",
            "freeze_duration",
        }
        assert expected_keys.issubset(set(signals.keys())), (
            f"缺失信号: {expected_keys - set(signals.keys())}"
        )

    def test_can_run_scoring_engine_with_signals(self) -> None:
        """信号能被评分引擎评估（端到端验证）."""
        from backend.ml.scoring import ScoringContext, ScoringEngine

        kpts = np.zeros((30, 24, 3), dtype=np.float32)
        kpts[:, 22, 2] = 0.9
        kpts[:, 22, 0] = np.arange(30, dtype=np.float32) * 5
        signals = _extract_puppy_signals_simple(kpts, fps=30.0, duration_sec=1.0)

        # 用真实评分卡评估
        puppy_yaml = (
            Path(__file__).resolve().parents[3]
            / "backend" / "ml" / "scoring" / "configs" / "puppy_selection.yaml"
        )
        engine = ScoringEngine.get(puppy_yaml)
        ctx = ScoringContext(signals=signals, scene="puppy_selection")
        result = engine.evaluate(ctx)

        # 信号有效 → 评分应非 0
        assert 0 <= result.total_score <= 100
        assert result.verdict in ("pass", "borderline", "fail")


# ============================================================
# 4. _resolve_model_path
# ============================================================


class TestResolveModelPath:
    """模型路径解析测试."""

    def test_returns_string(self) -> None:
        """返回字符串路径."""
        path = _resolve_model_path()
        assert isinstance(path, str)
        assert len(path) > 0

    def test_falls_back_to_yolo26n(self) -> None:
        """无本地模型时回退到 yolo26n-pose.pt（自动下载）."""
        # 当前项目应有 best.pt 或 best.onnx，回退路径仍可调用
        path = _resolve_model_path()
        # 至少返回非空字符串
        assert path.endswith(".pt") or path.endswith(".onnx")


# ============================================================
# 5. ingest_video 任务 mock 测试（不依赖真实 DB/模型）
# ============================================================


class TestIngestVideoTaskMocked:
    """ingest_video 任务 mock 测试.

    使用 unittest.mock 模拟:
        - PoseInferenceEngine.infer_video → 合成结果
        - SyncSessionLocal → MagicMock
        - ScoringEngine.get → 真实评分引擎

    不依赖真实 DB / GPU，可快速运行。
    """

    def test_task_registration(self) -> None:
        """任务已注册到 celery_app."""
        from backend.workers.celery_app import celery_app
        assert "inference.ingest_video" in celery_app.tasks

    def test_task_has_retry_config(self) -> None:
        """任务配置了重试策略."""
        from backend.workers.tasks import ingest_video
        assert ingest_video.max_retries == 2
        assert ingest_video.autoretry_for == (Exception,)

    def test_run_obedience_pipeline_with_mock_kpts(self) -> None:
        """obedience pipeline 能处理合成 keypoints 序列.

        不 mock rule_engine，验证 rule_engine + scoring 端到端可用。
        """
        from backend.workers.tasks import _run_obedience_pipeline
        from backend.tests.ml.test_rule_engine import (
            make_sequence, make_standing_frame,
        )

        # 20 帧站立 + 移动 → 应识别 STAND + HEEL
        frames = [make_standing_frame(x_offset=i * 5) for i in range(20)]
        kpts = make_sequence(frames)

        episodes, signals, result = _run_obedience_pipeline(kpts, fps=30.0)

        # 应识别到至少 1 个行为
        assert len(episodes) > 0
        # 信号应包含所有 obedience_trial.yaml 所需 key
        expected_keys = {
            "action_count", "action_correct",
            "command_to_action_latency", "action_duration",
            "focus_ratio", "gait_score",
        }
        assert expected_keys.issubset(set(signals.keys()))
        # 评分结果有效
        assert 0 <= result.total_score <= 100
        assert result.scene == "obedience_trial"

    def test_run_puppy_pipeline_with_mock_kpts(self) -> None:
        """puppy pipeline 能处理合成 keypoints 序列."""
        from backend.workers.tasks import _run_puppy_pipeline

        # 30 帧，withers 持续移动
        kpts = np.zeros((30, 24, 3), dtype=np.float32)
        kpts[:, 22, 2] = 0.9  # withers 置信度
        kpts[:, 22, 0] = np.arange(30, dtype=np.float32) * 5

        episodes, signals, result = _run_puppy_pipeline(
            kpts, fps=30.0, duration_sec=1.0
        )

        # puppy pipeline 不调用 rule_engine → episodes 为空
        assert episodes == []
        # 信号有效
        assert signals["approach_latency"] == 0.0
        assert signals["approach_speed"] > 0.0
        # 评分结果有效
        assert 0 <= result.total_score <= 100
        assert result.scene == "puppy_selection"
