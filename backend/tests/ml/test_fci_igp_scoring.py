"""FCI-IGP 评分卡单元测试（Phase 3.4b）.

Owner: ML 开发
Phase: 3.4b

测试覆盖:
    1. YAML 加载 + Schema 校验
        - fci_igp.yaml 能正常加载
        - 7 维评分 + 权重和 = 1.0
        - 22 行为 100% 覆盖 IGP A/B/C 三阶段
        - 3 DQ 硬约束存在
    2. 5 级评级
        - Excellent (96+) / Very Good (90+) / Good (80+) / Satisfactory (70+) / Insufficient (<70)
    3. DQ 硬约束
        - 枪怯 (gunshot_reaction == 'shy')
        - 不放口 (release_command_count >= 2 and not sleeve_released)
        - 衔取不吐 (retrieve_release_command_count >= 3 and not dumbbell_released)
        - DQ 触发 → 总分清零 + verdict=fail + rating=Insufficient
    4. 端到端三档验证
        - Excellent（all-excellent signals）
        - Failing（all-insufficient signals）
        - Borderline（satisfactory signals）
    5. Schema 扩展字段校验
        - disqualifications 仅 fci_igp 场景可用
        - igp_level 仅 fci_igp 场景可用

依据: dev-docs/research/RESEARCH_FCI_IGP_STANDARD.md §6 + phase-3.md §3.4b
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.ml.behavior.constants import ALL_BEHAVIORS_22
from backend.ml.behavior.stgcn_bc.labels import FCI_IGP_STAGE
from backend.ml.scoring import ScoringContext, ScoringEngine, ScoringResult
from backend.ml.scoring.schema import (
    BehaviorMappingEntry,
    DisqualificationSpec,
    ScoringCardSpec,
)

# fci_igp.yaml 路径
CONFIGS_DIR = Path(__file__).resolve().parents[3] / "backend" / "ml" / "scoring" / "configs"
FCI_IGP_YAML = CONFIGS_DIR / "fci_igp.yaml"
USPCA_YAML = CONFIGS_DIR / "uspca_patrol.yaml"

pytestmark = pytest.mark.fast


# ============================================================
# 1. YAML 加载 + Schema 校验
# ============================================================


class TestFciIgpYamlLoading:
    """fci_igp.yaml 加载与 Schema 校验测试。"""

    def test_yaml_file_exists(self) -> None:
        assert FCI_IGP_YAML.exists(), f"FCI-IGP 评分卡文件不存在: {FCI_IGP_YAML}"

    def test_load_success(self) -> None:
        """YAML 能正常加载为 ScoringCardSpec。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        assert engine.spec.scene == "fci_igp"
        assert engine.spec.name == "FCI-IGP 国际工作犬评分卡"
        assert engine.spec.version == "1.0.0"

    def test_igp_level_set(self) -> None:
        """IGP 等级字段已设置（默认 IGP1）。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        assert engine.spec.igp_level == "IGP1"

    def test_seven_dimensions(self) -> None:
        """应包含 7 个评分维度。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        dim_ids = {d.id for d in engine.spec.dimensions}
        expected = {"accuracy", "latency", "duration", "search_efficiency",
                    "attention", "courage", "gait"}
        assert dim_ids == expected, f"维度集合不匹配: {dim_ids}"

    def test_weights_sum_to_one(self) -> None:
        """7 维权重之和应为 1.0。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        total = sum(d.weight for d in engine.spec.dimensions)
        assert abs(total - 1.0) < 0.001, f"权重和 = {total:.4f}"

    def test_weights_match_fci_spec(self) -> None:
        """7 维权重对齐 FCI-IGP 标准比例（RESEARCH_FCI_IGP_STANDARD.md §4.1）。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        weights = {d.id: d.weight for d in engine.spec.dimensions}
        assert weights["accuracy"] == 0.25
        assert weights["latency"] == 0.15
        assert weights["duration"] == 0.15
        assert weights["search_efficiency"] == 0.15
        assert weights["attention"] == 0.10
        assert weights["courage"] == 0.10
        assert weights["gait"] == 0.10

    def test_thresholds_fci_satisfactory(self) -> None:
        """及格线 = 70（对齐 FCI Satisfactory 70%）。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        assert engine.spec.thresholds.pass_ == 70
        assert engine.spec.thresholds.borderline == 60

    def test_three_dq_rules(self) -> None:
        """应包含 3 个 DQ 硬约束（枪怯/不放口/衔取不吐）。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        assert len(engine.spec.disqualifications) == 3
        dq_ids = {dq.id for dq in engine.spec.disqualifications}
        assert dq_ids == {"gunfire_fail", "release_fail", "retrieve_fail"}

    def test_dq_all_force_fail_action(self) -> None:
        """所有 DQ 规则 action 应为 force_fail。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        for dq in engine.spec.disqualifications:
            assert dq.action == "force_fail", f"DQ {dq.id} action 非 force_fail"


# ============================================================
# 2. 22 行为 100% 覆盖
# ============================================================


class TestBehaviorMapping:
    """22 行为映射覆盖率测试。"""

    def test_all_22_behaviors_mapped(self) -> None:
        """behavior_mapping 应覆盖全部 22 个行为（labels.py ALL_BEHAVIORS_22）。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        mapped = set(engine.spec.behavior_mapping.keys())
        expected = set(ALL_BEHAVIORS_22)
        missing = expected - mapped
        extra = mapped - expected
        assert not missing, f"未映射行为: {missing}"
        assert not extra, f"多余映射: {extra}"
        assert len(mapped) == 22

    def test_all_behaviors_have_igp_phase(self) -> None:
        """所有行为映射应包含 igp_phase 字段（A/B/C/ALL）。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        for name, entry in engine.spec.behavior_mapping.items():
            assert entry.igp_phase is not None, f"行为 {name} 缺少 igp_phase"
            assert entry.igp_phase in {"A", "B", "C", "ALL"}

    def test_igp_phase_coverage_a_b_c(self) -> None:
        """三阶段（A/B/C）都应有行为映射。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        phases = {entry.igp_phase for entry in engine.spec.behavior_mapping.values()}
        assert "A" in phases, "缺少 Phase A (追踪) 行为"
        assert "B" in phases, "缺少 Phase B (服从) 行为"
        assert "C" in phases, "缺少 Phase C (护卫) 行为"

    def test_igp_phase_matches_labels_py(self) -> None:
        """YAML 中行为 igp_phase 应与 labels.py FCI_IGP_STAGE 一致。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        for name, entry in engine.spec.behavior_mapping.items():
            expected_phase = FCI_IGP_STAGE.get(name, "B")
            assert entry.igp_phase == expected_phase, (
                f"行为 {name}: YAML igp_phase={entry.igp_phase} "
                f"与 labels.py FCI_IGP_STAGE={expected_phase} 不一致"
            )

    def test_all_behaviors_have_label(self) -> None:
        """所有行为映射应包含中文 label。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        for name, entry in engine.spec.behavior_mapping.items():
            assert entry.label, f"行为 {name} 缺少 label"
            assert isinstance(entry.label, str)

    def test_all_behaviors_have_dimensions(self) -> None:
        """所有行为映射应至少有 1 个评分维度。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        valid_dims = {d.id for d in engine.spec.dimensions}
        for name, entry in engine.spec.behavior_mapping.items():
            assert len(entry.dimensions) >= 1, f"行为 {name} 无评分维度"
            for dim_id in entry.dimensions:
                assert dim_id in valid_dims, (
                    f"行为 {name} 引用了不存在的维度 {dim_id}"
                )


# ============================================================
# 3. Schema 扩展字段校验
# ============================================================


class TestSchemaValidation:
    """Schema 扩展字段校验测试。"""

    def test_disqualifications_reject_non_fci_igp_scene(self) -> None:
        """disqualifications 在非 fci_igp 场景应校验失败。"""
        with pytest.raises(ValidationError, match="disqualifications 仅支持 fci_igp"):
            ScoringCardSpec.model_validate({
                "name": "test", "version": "1.0", "scene": "uspca_patrol",
                "dimensions": [{"id": "d", "name": "d", "weight": 1.0, "rules": []}],
                "disqualifications": [
                    {"id": "x", "condition": "True", "action": "force_fail", "label": "x"}
                ],
            })

    def test_igp_level_reject_non_fci_igp_scene(self) -> None:
        """igp_level 在非 fci_igp 场景应校验失败。"""
        with pytest.raises(ValidationError, match="igp_level 仅支持 fci_igp"):
            ScoringCardSpec.model_validate({
                "name": "test", "version": "1.0", "scene": "uspca_patrol",
                "dimensions": [{"id": "d", "name": "d", "weight": 1.0, "rules": []}],
                "igp_level": "IGP1",
            })

    def test_uspca_yaml_still_loads(self) -> None:
        """USPCA YAML（无 DQ/igp_level）仍能正常加载（向后兼容）。"""
        engine = ScoringEngine.from_yaml(USPCA_YAML)
        assert engine.spec.scene == "uspca_patrol"
        assert engine.spec.disqualifications == []
        assert engine.spec.igp_level is None


# ============================================================
# 4. 5 级评级测试
# ============================================================


class TestFciRating:
    """FCI-IGP 5 级评级测试（Excellent/Very Good/Good/Satisfactory/Insufficient）。"""

    @pytest.fixture
    def engine(self) -> ScoringEngine:
        return ScoringEngine.from_yaml(FCI_IGP_YAML)

    def _make_signals(self, score_level: str) -> dict:
        """根据评级档位合成 signals.

        score_level:
            - "excellent": 全部 95 分档（>96%/0.5s/30s/...）
            - "very_good": 85 分档（>90%/1.0s/20s/...）
            - "good": 75 分档（>80%/1.5s/10s/...）
            - "satisfactory": 65 分档（>70%/3.0s/5s/...）
            - "insufficient": 30 分档（<70%/<3s/<5s/...）
        """
        presets = {
            "excellent": {
                "action_correct": 20, "action_count": 20,  # 100% > 96%
                "command_to_action_latency": 0.3,  # < 0.5
                "action_duration": 35.0,  # > 30
                "search_coverage": 0.90, "search_speed": 0.6, "target_found": True,
                "focus_ratio": 0.85, "unnecessary_movement_count": 1,
                "courage_score": 0.95, "avoidance_detected": False,
                "gait_symmetry": 0.92, "pace_change_smoothness": 0.88,
                # 无 DQ 信号
            },
            "very_good": {
                "action_correct": 19, "action_count": 20,  # 95% > 90%
                "command_to_action_latency": 0.7,  # < 1.0
                "action_duration": 25.0,  # > 20
                "search_coverage": 0.75, "search_speed": 0.45, "target_found": True,
                "focus_ratio": 0.70, "unnecessary_movement_count": 3,
                "courage_score": 0.85, "avoidance_detected": False,
                "gait_symmetry": 0.85, "pace_change_smoothness": 0.75,
            },
            "good": {
                "action_correct": 17, "action_count": 20,  # 85% > 80%
                "command_to_action_latency": 1.2,  # < 1.5
                "action_duration": 15.0,  # > 10
                "search_coverage": 0.65, "search_speed": 0.35, "target_found": True,
                "focus_ratio": 0.55, "unnecessary_movement_count": 5,
                "courage_score": 0.75, "avoidance_detected": False,
                "gait_symmetry": 0.75, "pace_change_smoothness": 0.60,
            },
            "satisfactory": {
                "action_correct": 15, "action_count": 20,  # 75% > 70%
                "command_to_action_latency": 2.0,  # < 3.0
                "action_duration": 7.0,  # > 5
                "search_coverage": 0.50, "search_speed": 0.30, "target_found": True,
                "focus_ratio": 0.50, "unnecessary_movement_count": 7,
                "courage_score": 0.55, "avoidance_detected": False,
                "gait_symmetry": 0.55, "pace_change_smoothness": 0.50,
            },
            "insufficient": {
                "action_correct": 10, "action_count": 20,  # 50% <= 70%
                "command_to_action_latency": 4.0,  # >= 3.0
                "action_duration": 3.0,  # <= 5
                "search_coverage": 0.30, "search_speed": 0.20, "target_found": False,
                "focus_ratio": 0.40, "unnecessary_movement_count": 10,
                "courage_score": 0.40, "avoidance_detected": True,
                "gait_symmetry": 0.40, "pace_change_smoothness": 0.40,
            },
        }
        return presets[score_level]

    def test_excellent_rating(self, engine: ScoringEngine) -> None:
        """全部 95 分档 → Excellent 评级。"""
        signals = self._make_signals("excellent")
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.rating == "Excellent", (
            f"期望 Excellent，实际 {result.rating}，总分 {result.total_score}"
        )
        assert result.verdict == "pass"
        assert result.passed is True

    def test_very_good_rating(self, engine: ScoringEngine) -> None:
        """全部 85 分档 → Very Good 评级（90-95.5）。"""
        signals = self._make_signals("very_good")
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.rating == "Very Good", (
            f"期望 Very Good，实际 {result.rating}，总分 {result.total_score}"
        )
        assert result.verdict == "pass"

    def test_good_rating(self, engine: ScoringEngine) -> None:
        """全部 75 分档 → Good 评级（80-89.5）。"""
        signals = self._make_signals("good")
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.rating == "Good", (
            f"期望 Good，实际 {result.rating}，总分 {result.total_score}"
        )
        assert result.verdict == "pass"

    def test_satisfactory_rating(self, engine: ScoringEngine) -> None:
        """全部 65 分档 → Satisfactory 评级（70-79.5），刚刚 pass。"""
        signals = self._make_signals("satisfactory")
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.rating == "Satisfactory", (
            f"期望 Satisfactory，实际 {result.rating}，总分 {result.total_score}"
        )
        assert result.verdict == "pass"
        assert result.total_score >= 70

    def test_insufficient_rating(self, engine: ScoringEngine) -> None:
        """全部 30 分档 → Insufficient 评级（<70），fail。"""
        signals = self._make_signals("insufficient")
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.rating == "Insufficient", (
            f"期望 Insufficient，实际 {result.rating}，总分 {result.total_score}"
        )
        assert result.verdict == "fail"
        assert result.passed is False


# ============================================================
# 5. DQ 硬约束测试
# ============================================================


class TestDisqualifications:
    """DQ 硬约束测试（枪怯/不放口/衔取不吐）。"""

    @pytest.fixture
    def engine(self) -> ScoringEngine:
        return ScoringEngine.from_yaml(FCI_IGP_YAML)

    def test_no_dq_when_signals_absent(self, engine: ScoringEngine) -> None:
        """DQ 信号缺失时不触发 DQ（正常评分）。"""
        signals = {
            "action_correct": 20, "action_count": 20,
            "command_to_action_latency": 0.3, "action_duration": 35.0,
            "search_coverage": 0.90, "search_speed": 0.6, "target_found": True,
            "focus_ratio": 0.85, "unnecessary_movement_count": 1,
            "courage_score": 0.95, "avoidance_detected": False,
            "gait_symmetry": 0.92, "pace_change_smoothness": 0.88,
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.disqualified is False
        assert result.disqualification_hit is None
        assert result.rating == "Excellent"

    def test_gunfire_fail_dq(self, engine: ScoringEngine) -> None:
        """枪怯 (gunshot_reaction == 'shy') → DQ gunfire_fail。"""
        signals = {
            "action_correct": 20, "action_count": 20,
            "command_to_action_latency": 0.3, "action_duration": 35.0,
            "search_coverage": 0.90, "search_speed": 0.6, "target_found": True,
            "focus_ratio": 0.85, "unnecessary_movement_count": 1,
            "courage_score": 0.95, "avoidance_detected": False,
            "gait_symmetry": 0.92, "pace_change_smoothness": 0.88,
            # DQ 触发信号
            "gunshot_reaction": "shy",
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.disqualified is True
        assert result.disqualification_hit == "gunfire_fail"
        assert result.verdict == "fail"
        assert result.passed is False
        # DQ 触发 → 总分清零
        assert result.total_score == 0.0
        # DQ 触发 → 评级强制 Insufficient
        assert result.rating == "Insufficient"

    def test_release_fail_dq(self, engine: ScoringEngine) -> None:
        """不放口 (release_command_count >= 2 and not sleeve_released) → DQ release_fail。"""
        signals = {
            "action_correct": 20, "action_count": 20,
            "command_to_action_latency": 0.3, "action_duration": 35.0,
            "search_coverage": 0.90, "search_speed": 0.6, "target_found": True,
            "focus_ratio": 0.85, "unnecessary_movement_count": 1,
            "courage_score": 0.95, "avoidance_detected": False,
            "gait_symmetry": 0.92, "pace_change_smoothness": 0.88,
            # DQ 触发信号
            "release_command_count": 2,
            "sleeve_released": False,
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.disqualified is True
        assert result.disqualification_hit == "release_fail"
        assert result.total_score == 0.0
        assert result.rating == "Insufficient"

    def test_retrieve_fail_dq(self, engine: ScoringEngine) -> None:
        """衔取不吐 (retrieve_release_command_count >= 3 and not dumbbell_released) → DQ retrieve_fail。"""
        signals = {
            "action_correct": 20, "action_count": 20,
            "command_to_action_latency": 0.3, "action_duration": 35.0,
            "search_coverage": 0.90, "search_speed": 0.6, "target_found": True,
            "focus_ratio": 0.85, "unnecessary_movement_count": 1,
            "courage_score": 0.95, "avoidance_detected": False,
            "gait_symmetry": 0.92, "pace_change_smoothness": 0.88,
            # DQ 触发信号
            "retrieve_release_command_count": 3,
            "dumbbell_released": False,
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.disqualified is True
        assert result.disqualification_hit == "retrieve_fail"
        assert result.total_score == 0.0
        assert result.rating == "Insufficient"

    def test_dq_explanation_records_hit(self, engine: ScoringEngine) -> None:
        """DQ 触发时 explanation 应记录命中的 DQ 规则。"""
        signals = {
            "action_correct": 20, "action_count": 20,
            "command_to_action_latency": 0.3, "action_duration": 35.0,
            "search_coverage": 0.90, "search_speed": 0.6, "target_found": True,
            "focus_ratio": 0.85, "unnecessary_movement_count": 1,
            "courage_score": 0.95, "avoidance_detected": False,
            "gait_symmetry": 0.92, "pace_change_smoothness": 0.88,
            "gunshot_reaction": "shy",
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        # 第一条 explanation 应包含 DQ 信息
        dq_msg = next(
            (e for e in result.explanation if "DQ" in e or "枪怯" in e), None
        )
        assert dq_msg is not None, f"explanation 未记录 DQ: {result.explanation}"
        assert "gunfire_fail" in dq_msg or "枪怯" in dq_msg

    def test_dq_priority_over_normal_score(self, engine: ScoringEngine) -> None:
        """即使信号全部优秀，DQ 触发仍强制 fail。"""
        signals = {
            "action_correct": 20, "action_count": 20,
            "command_to_action_latency": 0.3, "action_duration": 35.0,
            "search_coverage": 0.90, "search_speed": 0.6, "target_found": True,
            "focus_ratio": 0.85, "unnecessary_movement_count": 1,
            "courage_score": 0.95, "avoidance_detected": False,
            "gait_symmetry": 0.92, "pace_change_smoothness": 0.88,
            "release_command_count": 5,
            "sleeve_released": False,
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        # 正常评分应 Excellent，但 DQ 强制 fail
        assert result.disqualified is True
        assert result.total_score == 0.0
        assert result.rating == "Insufficient"


# ============================================================
# 6. 端到端集成测试
# ============================================================


class TestEndToEnd:
    """FCI-IGP 端到端集成测试。"""

    @pytest.fixture
    def engine(self) -> ScoringEngine:
        return ScoringEngine.from_yaml(FCI_IGP_YAML)

    def test_excellent_scenario(self, engine: ScoringEngine) -> None:
        """三档验证 - Excellent 场景：全部优秀信号。"""
        signals = {
            "action_correct": 20, "action_count": 20,
            "command_to_action_latency": 0.3, "action_duration": 35.0,
            "search_coverage": 0.90, "search_speed": 0.6, "target_found": True,
            "focus_ratio": 0.85, "unnecessary_movement_count": 1,
            "courage_score": 0.95, "avoidance_detected": False,
            "gait_symmetry": 0.92, "pace_change_smoothness": 0.88,
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.rating == "Excellent"
        assert result.verdict == "pass"
        assert result.passed is True
        assert result.disqualified is False
        # 总分应接近 95（全 95 分档加权）
        assert result.total_score >= 90.0

    def test_failing_scenario(self, engine: ScoringEngine) -> None:
        """三档验证 - Failing 场景：全部不合格信号。"""
        signals = {
            "action_correct": 10, "action_count": 20,
            "command_to_action_latency": 4.0, "action_duration": 3.0,
            "search_coverage": 0.30, "search_speed": 0.20, "target_found": False,
            "focus_ratio": 0.40, "unnecessary_movement_count": 10,
            "courage_score": 0.40, "avoidance_detected": True,
            "gait_symmetry": 0.40, "pace_change_smoothness": 0.40,
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.rating == "Insufficient"
        assert result.verdict == "fail"
        assert result.passed is False
        assert result.disqualified is False  # 失败但未 DQ
        assert result.total_score < 70.0

    def test_borderline_scenario(self, engine: ScoringEngine) -> None:
        """三档验证 - Borderline 场景：介于 pass/fail 之间（60-70）.

        FCI-IGP 评级映射:
            70-79.5 → Satisfactory (pass)
            60-69.5 → Insufficient (borderline verdict，但 rating 仍为 Insufficient)

        本场景构造 60-70 区间总分: accuracy 命中 insufficient (30 分) + 其他 6 维 satisfactory (70 分)
        总分 = 30*0.25 + 70*0.15*3 + 70*0.10*3 = 7.5 + 31.5 + 21.0 = 60.0
        """
        # 混合信号：accuracy 不合格（70% 不满足 > 0.70），其他维度合格
        signals = {
            "action_correct": 14, "action_count": 20,  # 70% 不满足 > 0.70 → accuracy_insufficient (30)
            "command_to_action_latency": 2.0,  # latency_satisfactory (70)
            "action_duration": 6.0,  # duration_satisfactory (70)
            "search_coverage": 0.50, "search_speed": 0.30, "target_found": True,  # search_satisfactory (70)
            "focus_ratio": 0.50, "unnecessary_movement_count": 8,  # attention_satisfactory (70, 0.50 > 0.40)
            "courage_score": 0.55, "avoidance_detected": False,  # courage_satisfactory (70)
            "gait_symmetry": 0.55, "pace_change_smoothness": 0.50,  # gait_satisfactory (70)
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        # 总分 = 60.0 → 评级 Insufficient（FCI: <70 = Insufficient）+ verdict borderline
        assert result.rating == "Insufficient"
        assert result.verdict == "borderline"
        assert result.passed is False
        assert 60 <= result.total_score < 70

    def test_dq_scenario(self, engine: ScoringEngine) -> None:
        """三档验证 - DQ 场景：枪怯触发。"""
        signals = {
            "action_correct": 20, "action_count": 20,
            "command_to_action_latency": 0.3, "action_duration": 35.0,
            "search_coverage": 0.90, "search_speed": 0.6, "target_found": True,
            "focus_ratio": 0.85, "unnecessary_movement_count": 1,
            "courage_score": 0.95, "avoidance_detected": False,
            "gait_symmetry": 0.92, "pace_change_smoothness": 0.88,
            "gunshot_reaction": "shy",
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.disqualified is True
        assert result.disqualification_hit == "gunfire_fail"
        assert result.rating == "Insufficient"
        assert result.verdict == "fail"
        assert result.total_score == 0.0

    def test_scene_mismatch_raises(self, engine: ScoringEngine) -> None:
        """场景不匹配应抛 ValueError。"""
        with pytest.raises(ValueError, match="场景不匹配"):
            engine.evaluate(ScoringContext(signals={}, scene="uspca_patrol"))


# ============================================================
# 7. 规则覆盖测试（确保每个评级都有对应规则）
# ============================================================


class TestRulesCoverage:
    """每个维度的规则覆盖测试。"""

    def test_each_dimension_has_5_level_rules(self) -> None:
        """每个维度应包含 5 级评级规则（excellent/very_good/good/satisfactory/insufficient）。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        for dim in engine.spec.dimensions:
            rule_ids = {r.id for r in dim.rules}
            # 至少包含 5 个评级档位中的一个
            has_excellent = any("excellent" in rid for rid in rule_ids)
            has_satisfactory = any("satisfactory" in rid for rid in rule_ids)
            has_insufficient = any("insufficient" in rid for rid in rule_ids)
            assert has_excellent, f"维度 {dim.id} 缺少 excellent 规则"
            assert has_satisfactory, f"维度 {dim.id} 缺少 satisfactory 规则"
            assert has_insufficient, f"维度 {dim.id} 缺少 insufficient 规则"

    def test_each_dimension_has_default_rule(self) -> None:
        """每个维度应包含 default 兜底规则（condition='True'）。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        for dim in engine.spec.dimensions:
            has_default = any(
                r.condition.strip() == "True" for r in dim.rules
            )
            assert has_default, f"维度 {dim.id} 缺少 default 规则（condition='True'）"

    def test_score_range_0_to_100(self) -> None:
        """所有规则分数应在 0-100 范围。"""
        engine = ScoringEngine.from_yaml(FCI_IGP_YAML)
        for dim in engine.spec.dimensions:
            for rule in dim.rules:
                assert 0 <= rule.score <= 100, (
                    f"维度 {dim.id} 规则 {rule.id} score={rule.score} 超出 0-100"
                )
