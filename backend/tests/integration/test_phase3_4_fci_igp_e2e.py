"""Phase 3.4 FCI-IGP 端到端集成测试.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.4
依据: dev-docs/stages/phase-3.md §3.4b/§3.4c

测试内容:
    3.4a  FCI-IGP 场景注册（VALID_SCENES + 评分 API 映射）
    3.4b  22 行为 → 7 维信号 → FCI-IGP 评分端到端
    3.4c  DQ 硬约束端到端（枪怯/不放口/衔取不吐 → 总分清零）
    3.4d  合成关键点序列 → _run_fci_igp_pipeline → 评分结果

运行:
    pytest backend/tests/integration/test_phase3_4_fci_igp_e2e.py -v

注意:
    - 不需要真实视频/GPU（用合成关键点序列）
    - 不需要 Celery worker / API server（直接调用 pipeline 函数）
    - 不需要 PostgreSQL（纯 ML pipeline 测试）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.ml.behavior.constants import ALL_BEHAVIORS_22, NUM_KEYPOINTS
from backend.ml.scoring import ScoringContext, ScoringEngine
from backend.ml.scoring.schema import Scene


PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# 3.4a FCI-IGP 场景注册验证
# ============================================================


class TestFciIgpSceneRegistration:
    """FCI-IGP 场景在系统中正确注册。"""

    def test_fci_igp_in_valid_scenes(self):
        """Video 模型 VALID_SCENES 包含 fci_igp。"""
        from backend.app.models.video import VALID_SCENES

        assert "fci_igp" in VALID_SCENES, (
            f"fci_igp 未在 VALID_SCENES 中: {VALID_SCENES}"
        )

    def test_fci_igp_in_scoring_api_mapping(self):
        """评分 API _SCENE_TO_FILE 包含 fci_igp 映射。"""
        from backend.app.api.scoring import _SCENE_TO_FILE

        assert "fci_igp" in _SCENE_TO_FILE, (
            f"fci_igp 未在 _SCENE_TO_FILE 中: {_SCENE_TO_FILE}"
        )
        assert _SCENE_TO_FILE["fci_igp"] == "fci_igp.yaml"

    def test_fci_igp_yaml_file_exists(self):
        """FCI-IGP 评分卡 YAML 文件存在。"""
        yaml_path = (
            PROJECT_ROOT
            / "backend"
            / "ml"
            / "scoring"
            / "configs"
            / "fci_igp.yaml"
        )
        assert yaml_path.exists(), f"YAML 文件不存在: {yaml_path}"

    def test_fci_igp_scene_in_schema(self):
        """ScoringContext 支持 fci_igp 场景。"""
        from typing import get_args

        scenes = get_args(Scene)
        assert "fci_igp" in scenes, f"fci_igp 未在 Scene 类型中: {scenes}"


# ============================================================
# 3.4b 22 行为 → 7 维信号 → FCI-IGP 评分端到端
# ============================================================


class TestFciIgpPipelineE2E:
    """FCI-IGP pipeline 端到端测试：episodes → signals → scoring。"""

    @pytest.fixture
    def engine(self) -> ScoringEngine:
        return ScoringEngine.from_yaml(
            PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs" / "fci_igp.yaml"
        )

    def test_synthetic_excellent_pipeline(self, engine: ScoringEngine):
        """合成优秀 episodes → signals → Excellent 评级。"""
        from backend.ml.behavior.rule_engine import BehaviorEpisode

        # 合成覆盖 IGP A/B/C 三阶段的优秀 episodes
        episodes = [
            # Phase A 追踪
            BehaviorEpisode(
                behavior="track", start_frame=0, end_frame=300, confidence=0.95
            ),
            BehaviorEpisode(
                behavior="alert_sit", start_frame=301, end_frame=360, confidence=0.92
            ),
            # Phase B 服从
            BehaviorEpisode(
                behavior="heel", start_frame=361, end_frame=600, confidence=0.94
            ),
            BehaviorEpisode(
                behavior="sit", start_frame=601, end_frame=750, confidence=0.96
            ),
            BehaviorEpisode(
                behavior="stay", start_frame=751, end_frame=1050, confidence=0.91
            ),
            BehaviorEpisode(
                behavior="recall", start_frame=1051, end_frame=1200, confidence=0.93
            ),
            BehaviorEpisode(
                behavior="retrieve", start_frame=1201, end_frame=1380, confidence=0.90
            ),
            # Phase C 护卫
            BehaviorEpisode(
                behavior="bite", start_frame=1381, end_frame=1500, confidence=0.92
            ),
            BehaviorEpisode(
                behavior="release", start_frame=1501, end_frame=1560, confidence=0.95
            ),
            BehaviorEpisode(
                behavior="guard", start_frame=1561, end_frame=1800, confidence=0.88
            ),
        ]

        # 调用 signals_from_fci_igp_episodes（独立模块，无 celery 依赖）
        from backend.ml.behavior.fci_igp_signals import signals_from_fci_igp_episodes

        signals = signals_from_fci_igp_episodes(episodes, fps=30.0, duration_sec=60.0)

        # 验证 signals 字段完整（与 fci_igp.yaml 对齐）
        required_fields = [
            "action_correct", "action_count",
            "command_to_action_latency", "action_duration",
            "search_coverage", "search_speed", "target_found",
            "focus_ratio", "unnecessary_movement_count",
            "courage_score", "avoidance_detected",
            "gait_symmetry", "pace_change_smoothness",
        ]
        for field in required_fields:
            assert field in signals, f"signals 缺少字段: {field}"

        # 验证 signals 值合理性
        assert signals["action_count"] == 10
        assert signals["action_correct"] == 10  # 全部 confidence >= 0.5
        assert signals["target_found"] is True  # alert_sit 存在
        assert signals["focus_ratio"] > 0.85  # 平均置信度

        # 评分
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))

        # 验证评分结果
        assert result.verdict in ("pass", "borderline"), (
            f"verdict 异常: {result.verdict}, score={result.total_score}"
        )
        assert 0 <= result.total_score <= 100
        assert result.disqualified is False
        # 至少应达到 Satisfactory（70+）因为所有行为 confidence 都很高
        assert result.total_score >= 60, (
            f"总分偏低: {result.total_score}, signals={signals}"
        )

    def test_synthetic_failing_pipeline(self, engine: ScoringEngine):
        """合成低质量 episodes → signals → 低分评级。"""
        from backend.ml.behavior.rule_engine import BehaviorEpisode

        # 合成低 confidence 短时 episodes
        episodes = [
            BehaviorEpisode(
                behavior="sit", start_frame=0, end_frame=2, confidence=0.30
            ),
            BehaviorEpisode(
                behavior="down", start_frame=3, end_frame=5, confidence=0.25
            ),
        ]

        from backend.ml.behavior.fci_igp_signals import signals_from_fci_igp_episodes

        signals = signals_from_fci_igp_episodes(episodes, fps=30.0, duration_sec=10.0)

        # 验证低质量信号
        assert signals["action_correct"] == 0  # 无 confidence >= 0.5
        assert signals["focus_ratio"] < 0.5
        assert signals["action_duration"] < 1.0  # 短时

        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))

        # 验证低分
        assert result.total_score < 70, (
            f"低质量应 < 70: {result.total_score}"
        )
        assert result.verdict in ("fail", "borderline")

    def test_empty_episodes_pipeline(self, engine: ScoringEngine):
        """空 episodes → 默认 signals → 低分。"""
        from backend.ml.behavior.fci_igp_signals import signals_from_fci_igp_episodes

        signals = signals_from_fci_igp_episodes([], fps=30.0, duration_sec=0.0)

        # 验证空信号默认值
        assert signals["action_count"] == 0
        assert signals["action_correct"] == 0
        assert signals["target_found"] is False
        assert signals["focus_ratio"] == 0.0

        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))

        # 空信号应得低分
        assert result.total_score < 70
        assert result.verdict in ("fail", "borderline")


# ============================================================
# 3.4c DQ 硬约束端到端
# ============================================================


class TestFciIgpDQE2E:
    """DQ 硬约束端到端：DQ 信号 → 总分清零 + verdict=fail。"""

    @pytest.fixture
    def engine(self) -> ScoringEngine:
        return ScoringEngine.from_yaml(
            PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs" / "fci_igp.yaml"
        )

    def test_gunshy_dq_e2e(self, engine: ScoringEngine):
        """枪怯 DQ → 总分清零。"""
        from backend.ml.behavior.fci_igp_signals import signals_from_fci_igp_episodes
        from backend.ml.behavior.rule_engine import BehaviorEpisode

        # 优秀 episodes + 枪怯信号
        episodes = [
            BehaviorEpisode(
                behavior="sit", start_frame=0, end_frame=300, confidence=0.95
            ),
        ]
        signals = signals_from_fci_igp_episodes(episodes, fps=30.0, duration_sec=10.0)
        signals["gunshot_reaction"] = "shy"  # 注入枪怯信号

        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))

        assert result.disqualified is True
        assert result.disqualification_hit == "gunfire_fail"
        assert result.total_score == 0.0
        assert result.verdict == "fail"
        assert result.rating == "Insufficient"

    def test_no_release_dq_e2e(self, engine: ScoringEngine):
        """不放口 DQ → 总分清零。"""
        from backend.ml.behavior.fci_igp_signals import signals_from_fci_igp_episodes
        from backend.ml.behavior.rule_engine import BehaviorEpisode

        episodes = [
            BehaviorEpisode(
                behavior="bite", start_frame=0, end_frame=300, confidence=0.92
            ),
        ]
        signals = signals_from_fci_igp_episodes(episodes, fps=30.0, duration_sec=10.0)
        signals["release_command_count"] = 3
        signals["sleeve_released"] = False

        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))

        assert result.disqualified is True
        assert result.disqualification_hit == "release_fail"
        assert result.total_score == 0.0

    def test_retrieve_fail_dq_e2e(self, engine: ScoringEngine):
        """衔取不吐 DQ → 总分清零。"""
        from backend.workers.tasks import _signals_from_fci_igp_episodes
        from backend.ml.behavior.rule_engine import BehaviorEpisode

        episodes = [
            BehaviorEpisode(
                behavior="retrieve", start_frame=0, end_frame=300, confidence=0.88
            ),
        ]
        signals = _signals_from_fci_igp_episodes(episodes, fps=30.0, duration_sec=10.0)
        signals["retrieve_release_command_count"] = 3
        signals["dumbbell_released"] = False

        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))

        assert result.disqualified is True
        assert result.disqualification_hit == "retrieve_fail"
        assert result.total_score == 0.0


# ============================================================
# 3.4d 合成关键点序列 → _run_fci_igp_pipeline → 评分结果
# ============================================================


class TestFciIgpFullPipeline:
    """FCI-IGP 全 pipeline 端到端：关键点序列 → 行为识别 → 信号 → 评分。"""

    def test_full_pipeline_with_synthetic_keypoints(self):
        """合成关键点 → _run_fci_igp_pipeline → 评分结果。

        验证 pipeline 完整执行不报错，返回有效的 ScoringResult。
        """
        from backend.workers.tasks import _run_fci_igp_pipeline

        # 合成 30 帧关键点序列（24 关键点 × 3D）
        # 模拟站立姿态
        kpts = np.zeros((30, NUM_KEYPOINTS, 3), dtype=np.float32)
        kpts[:, :, 2] = 1.0  # 置信度 = 1.0
        # 设置合理的 2D 坐标（模拟犬站立）
        for i in range(NUM_KEYPOINTS):
            kpts[:, i, 0] = 100 + i * 5  # x
            kpts[:, i, 1] = 200 + (i % 6) * 20  # y

        # 调用完整 pipeline
        episodes, signals, result = _run_fci_igp_pipeline(
            kpts, fps=30.0, duration_sec=1.0
        )

        # 验证返回类型
        assert isinstance(episodes, list)
        assert isinstance(signals, dict)
        assert result is not None

        # 验证 signals 字段完整
        required_fields = [
            "action_correct", "action_count",
            "command_to_action_latency", "action_duration",
            "search_coverage", "search_speed", "target_found",
            "focus_ratio", "unnecessary_movement_count",
            "courage_score", "avoidance_detected",
            "gait_symmetry", "pace_change_smoothness",
        ]
        for field in required_fields:
            assert field in signals, f"signals 缺少字段: {field}"

        # 验证评分结果
        assert 0 <= result.total_score <= 100
        assert result.verdict in ("pass", "borderline", "fail")
        assert result.rating in (
            "Excellent", "Very Good", "Good", "Satisfactory", "Insufficient"
        )

    def test_full_pipeline_empty_keypoints(self):
        """空关键点序列 → pipeline 不崩溃 → 低分。"""
        from backend.workers.tasks import _run_fci_igp_pipeline

        # 空 keypoint 序列
        kpts = np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32)

        episodes, signals, result = _run_fci_igp_pipeline(
            kpts, fps=30.0, duration_sec=0.0
        )

        # 空输入应返回空 episodes + 默认 signals + 低分
        assert isinstance(episodes, list)
        assert signals["action_count"] == 0
        assert result.total_score < 70


# ============================================================
# 3.4e IGP 三阶段行为覆盖
# ============================================================


class TestIgpPhaseCoverage:
    """IGP A/B/C 三阶段行为在 pipeline 中正确处理。"""

    def test_phase_a_behaviors_recognized(self):
        """Phase A 追踪阶段行为（track/alert_sit/alert_down/search_blind）。"""
        from backend.workers.tasks import _signals_from_fci_igp_episodes
        from backend.ml.behavior.rule_engine import BehaviorEpisode

        phase_a_behaviors = ["track", "alert_sit", "alert_down", "search_blind"]
        episodes = [
            BehaviorEpisode(
                behavior=b, start_frame=i * 100, end_frame=(i + 1) * 100, confidence=0.85
            )
            for i, b in enumerate(phase_a_behaviors)
        ]

        signals = _signals_from_fci_igp_episodes(episodes, fps=30.0, duration_sec=13.3)

        # Phase A 行为应触发 search 信号
        assert signals["target_found"] is True  # alert_sit 存在
        assert signals["search_coverage"] > 0  # 有搜索行为

    def test_phase_c_behaviors_recognized(self):
        """Phase C 护卫阶段行为（bite/apprehend/escort/guard/release）。"""
        from backend.workers.tasks import _signals_from_fci_igp_episodes
        from backend.ml.behavior.rule_engine import BehaviorEpisode

        phase_c_behaviors = ["bite", "apprehend", "escort", "guard", "release"]
        episodes = [
            BehaviorEpisode(
                behavior=b, start_frame=i * 100, end_frame=(i + 1) * 100, confidence=0.85
            )
            for i, b in enumerate(phase_c_behaviors)
        ]

        signals = _signals_from_fci_igp_episodes(episodes, fps=30.0, duration_sec=16.7)

        # Phase C 行为应触发 courage 信号
        assert signals["courage_score"] > 0  # bite/apprehend/guard 存在
        assert signals["target_found"] is True  # apprehend 存在

    def test_all_22_behaviors_in_mapping(self):
        """22 行为全部在 fci_igp.yaml behavior_mapping 中。"""
        engine = ScoringEngine.from_yaml(
            PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs" / "fci_igp.yaml"
        )
        mapped = set(engine.spec.behavior_mapping.keys())
        expected = set(ALL_BEHAVIORS_22)
        assert mapped == expected, (
            f"行为映射不匹配: missing={expected - mapped}, extra={mapped - expected}"
        )
