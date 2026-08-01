"""评分引擎单元测试.

Owner: ML 开发
Phase: 1.4b

测试策略:
    1. 条件表达式解析: 合法/非法语法/信号缺失
    2. 单维度评分: 规则命中/默认/顺序匹配
    3. 多维度加权聚合: weighted_sum / max / min
    4. 热加载: YAML 修改后自动重载
    5. 双场景端到端: 选育 3 维 + 科目 5 维

标记: fast（纯 Python，无需 GPU）
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from backend.ml.scoring import ScoringContext, ScoringEngine, ScoringResult
from backend.ml.scoring.conditions import (
    ConditionError,
    evaluate_condition,
    validate_expression,
)

# 评分卡 YAML 路径
# __file__ = backend/tests/ml/test_scoring_engine.py
# parents[3] = 项目根目录 d:\Desktop\k9-training-system
CONFIGS_DIR = Path(__file__).resolve().parents[3] / "backend" / "ml" / "scoring" / "configs"
PUPPY_YAML = CONFIGS_DIR / "puppy_selection.yaml"
OBEDIENCE_YAML = CONFIGS_DIR / "obedience_trial.yaml"
USPCA_YAML = CONFIGS_DIR / "uspca_patrol.yaml"

pytestmark = pytest.mark.fast


# ============================================================
# 1. 条件表达式解析
# ============================================================


class TestConditionParser:
    """条件表达式安全解析测试。"""

    def test_simple_comparison(self) -> None:
        assert evaluate_condition("x < 1.0", {"x": 0.5}) is True
        assert evaluate_condition("x < 1.0", {"x": 2.0}) is False

    def test_and_logic(self) -> None:
        expr = "x < 1.0 and y > 2.0"
        assert evaluate_condition(expr, {"x": 0.5, "y": 3.0}) is True
        assert evaluate_condition(expr, {"x": 0.5, "y": 1.0}) is False

    def test_or_logic(self) -> None:
        expr = "x < 1.0 or y > 2.0"
        assert evaluate_condition(expr, {"x": 2.0, "y": 3.0}) is True
        assert evaluate_condition(expr, {"x": 2.0, "y": 1.0}) is False

    def test_arithmetic(self) -> None:
        # accuracy 维度: action_correct / action_count > 0.95
        # 20/20 = 1.0 > 0.95 = True
        assert evaluate_condition("a / b > 0.95", {"a": 20, "b": 20}) is True
        # 18/20 = 0.9 > 0.95 = False
        assert evaluate_condition("a / b > 0.95", {"a": 18, "b": 20}) is False
        # 19/20 = 0.95 > 0.95 = False（边界值）
        assert evaluate_condition("a / b > 0.95", {"a": 19, "b": 20}) is False

    def test_parentheses(self) -> None:
        expr = "(x < 1.0 or y < 1.0) and z > 5.0"
        assert evaluate_condition(expr, {"x": 0.5, "y": 2.0, "z": 6.0}) is True
        assert evaluate_condition(expr, {"x": 2.0, "y": 2.0, "z": 6.0}) is False

    def test_true_literal(self) -> None:
        assert evaluate_condition("True", {}) is True

    def test_signal_missing_returns_false(self) -> None:
        """信号缺失时条件视为 False（不命中）。"""
        assert evaluate_condition("missing_signal < 1.0", {"x": 0.5}) is False

    def test_or_short_circuit_with_missing_signal(self) -> None:
        """or 短路: 左操作数为 True 时，右操作数信号缺失不应使整体为 False。"""
        # approach_latency >= 5.0 为 True，sniff_duration 缺失 → 整体应为 True
        assert evaluate_condition(
            "approach_latency >= 5.0 or sniff_duration <= 0.0",
            {"approach_latency": 6.0},
        ) is True

    def test_and_short_circuit_with_missing_signal(self) -> None:
        """and 短路: 左操作数为 False 时，右操作数信号缺失不应影响整体。"""
        assert evaluate_condition(
            "x > 10.0 and missing_signal < 1.0",
            {"x": 0.5},
        ) is False

    def test_or_both_missing_returns_false(self) -> None:
        """or 两边信号都缺失 → False。"""
        assert evaluate_condition(
            "missing1 > 1.0 or missing2 < 2.0", {}
        ) is False

    def test_forbidden_import(self) -> None:
        with pytest.raises(ConditionError):
            evaluate_condition("__import__('os')", {})

    def test_forbidden_exec(self) -> None:
        with pytest.raises(ConditionError):
            evaluate_condition("exec('print(1)')", {})

    def test_syntax_error(self) -> None:
        with pytest.raises(ConditionError):
            evaluate_condition("x < ", {"x": 1.0})

    def test_validate_expression_extracts_names(self) -> None:
        names = validate_expression("x < 1.0 and y > 2.0")
        assert set(names) == {"x", "y"}

    def test_validate_expression_excludes_true_false(self) -> None:
        names = validate_expression("True or x > 1.0")
        assert names == ["x"]


# ============================================================
# 2. 评分卡加载 + 单维度评分
# ============================================================


class TestScoringCardLoading:
    """评分卡 YAML 加载 + Schema 校验测试。"""

    def test_load_puppy_card(self) -> None:
        engine = ScoringEngine.from_yaml(PUPPY_YAML)
        assert engine.spec.scene == "puppy_selection"
        assert engine.spec.name == "幼犬选育评分卡"
        assert len(engine.spec.dimensions) == 3
        # 权重之和 = 1.0
        total_weight = sum(d.weight for d in engine.spec.dimensions)
        assert abs(total_weight - 1.0) < 0.001

    def test_load_obedience_card(self) -> None:
        engine = ScoringEngine.from_yaml(OBEDIENCE_YAML)
        assert engine.spec.scene == "obedience_trial"
        assert len(engine.spec.dimensions) == 5
        total_weight = sum(d.weight for d in engine.spec.dimensions)
        assert abs(total_weight - 1.0) < 0.001

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ScoringEngine.from_yaml("nonexistent.yaml")

    def test_weights_validation(self, tmp_path: Path) -> None:
        """权重之和不为 1.0 时应校验失败。"""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(
            """
scoring_engine:
  name: "测试"
  version: "1.0.0"
  scene: "puppy_selection"
  dimensions:
    - id: a
      name: "A"
      weight: 0.5
      rules:
        - id: r1
          condition: "True"
          score: 50
          label: "中"
    - id: b
      name: "B"
      weight: 0.6
      rules:
        - id: r2
          condition: "True"
          score: 50
          label: "中"
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="权重之和"):
            ScoringEngine.from_yaml(bad_yaml)


# ============================================================
# 3. 幼犬选育场景端到端
# ============================================================


class TestPuppySelectionScoring:
    """幼犬选育 3 维评分端到端测试。"""

    @pytest.fixture
    def engine(self) -> ScoringEngine:
        ScoringEngine.invalidate(PUPPY_YAML)
        return ScoringEngine.from_yaml(PUPPY_YAML)

    def test_high_score_puppy(self, engine: ScoringEngine) -> None:
        """3 维全部高分的优秀幼犬 → 总分 ≥ 85。"""
        ctx = ScoringContext(
            signals={
                "approach_latency": 0.5,    # < 1.0
                "approach_speed": 3.0,      # > 2.0
                "sniff_duration": 2.0,      # > 1.0
                "chase_latency": 0.3,       # < 0.5
                "chase_speed": 4.0,         # > 3.0
                "hold_duration": 5.0,       # > 3.0
                "retreat_distance": 0.2,    # < 0.5
                "recovery_time": 1.0,       # < 2.0
                "freeze_duration": 0.5,     # < 1.0
            },
            scene="puppy_selection",
        )
        result = engine.evaluate(ctx)
        assert result.total_score >= 85
        assert result.verdict == "pass"
        assert result.dimension_labels["food_drive"] == "高"
        assert result.dimension_labels["prey_drive"] == "高"
        assert result.dimension_labels["courage"] == "高"

    def test_low_score_puppy(self, engine: ScoringEngine) -> None:
        """3 维全部低分的幼犬 → 总分 ≤ 25。"""
        ctx = ScoringContext(
            signals={
                "approach_latency": 6.0,    # >= 5.0
                "chase_latency": 4.0,       # >= 3.0
                "retreat_distance": 2.5,    # >= 2.0
                "freeze_duration": 15.0,    # > 10.0
            },
            scene="puppy_selection",
        )
        result = engine.evaluate(ctx)
        assert result.total_score <= 25
        assert result.verdict == "fail"
        assert not result.passed

    def test_medium_score_puppy(self, engine: ScoringEngine) -> None:
        """中等水平幼犬 → borderline。"""
        ctx = ScoringContext(
            signals={
                "approach_latency": 2.0,    # 中等（food_medium 命中）
                "approach_speed": 1.5,
                "sniff_duration": 0.8,      # > 0.5
                "chase_latency": 1.0,       # 中等
                "chase_speed": 1.5,
                "hold_duration": 1.5,
                "retreat_distance": 1.0,    # 中等
                "recovery_time": 3.0,
                "freeze_duration": 1.5,     # < 3.0
            },
            scene="puppy_selection",
        )
        result = engine.evaluate(ctx)
        assert 50 <= result.total_score <= 70
        assert result.dimension_labels["food_drive"] == "中"

    def test_signal_missing_falls_to_default(self, engine: ScoringEngine) -> None:
        """信号缺失时命中兜底规则（True）→ default 分。"""
        ctx = ScoringContext(
            signals={},
            scene="puppy_selection",
        )
        result = engine.evaluate(ctx)
        # 所有维度都命中 "True" 兜底规则 → default 分数
        assert result.dimension_labels["food_drive"] == "中低"
        assert result.dimension_labels["prey_drive"] == "中低"
        assert result.dimension_labels["courage"] == "中低"

    def test_scene_mismatch_raises(self, engine: ScoringEngine) -> None:
        ctx = ScoringContext(signals={}, scene="obedience_trial")
        with pytest.raises(ValueError, match="场景不匹配"):
            engine.evaluate(ctx)

    def test_explanation_contains_all_dimensions(self, engine: ScoringEngine) -> None:
        ctx = ScoringContext(
            signals={"approach_latency": 0.5, "approach_speed": 3.0},
            scene="puppy_selection",
        )
        result = engine.evaluate(ctx)
        # 解释应包含 3 个维度 + 1 行总分
        assert len(result.explanation) == 4
        assert "食物欲望" in result.explanation[1]
        assert "猎物欲望" in result.explanation[2]
        assert "胆量" in result.explanation[3]


# ============================================================
# 4. 科目测评场景端到端
# ============================================================


class TestObedienceTrialScoring:
    """科目测评 5 维评分端到端测试。"""

    @pytest.fixture
    def engine(self) -> ScoringEngine:
        ScoringEngine.invalidate(OBEDIENCE_YAML)
        return ScoringEngine.from_yaml(OBEDIENCE_YAML)

    def test_excellent_dog(self, engine: ScoringEngine) -> None:
        """5 维全部优秀的犬 → 总分 ≥ 85。"""
        ctx = ScoringContext(
            signals={
                "action_correct": 20,
                "action_count": 20,         # 1.0 > 0.95
                "command_to_action_latency": 0.3,  # < 0.5
                "action_duration": 35.0,    # > 30.0
                "focus_ratio": 0.90,        # > 0.80
                "gait_score": 0.90,         # > 0.85
            },
            scene="obedience_trial",
        )
        result = engine.evaluate(ctx)
        assert result.total_score >= 85
        assert result.verdict == "pass"
        assert result.dimension_labels["accuracy"] == "优秀"

    def test_failing_dog(self, engine: ScoringEngine) -> None:
        """5 维全部不合格的犬 → 总分 ≤ 35。"""
        ctx = ScoringContext(
            signals={
                "action_correct": 10,
                "action_count": 20,         # 0.5 <= 0.60
                "command_to_action_latency": 4.0,  # >= 3.0
                "action_duration": 1.0,     # <= 3.0
                "focus_ratio": 0.30,        # <= 0.50
                "gait_score": 0.40,         # <= 0.50
            },
            scene="obedience_trial",
        )
        result = engine.evaluate(ctx)
        assert result.total_score <= 35
        assert result.verdict == "fail"

    def test_accuracy_division(self, engine: ScoringEngine) -> None:
        """测试除法运算: action_correct / action_count > 0.95。"""
        # 19/20 = 0.95，不 > 0.95 → accuracy_good 命中
        ctx = ScoringContext(
            signals={
                "action_correct": 19,
                "action_count": 20,
            },
            scene="obedience_trial",
        )
        result = engine.evaluate(ctx)
        assert result.dimension_labels["accuracy"] == "良好"
        assert result.dimension_scores["accuracy"] == 75

    def test_all_dimensions_present(self, engine: ScoringEngine) -> None:
        ctx = ScoringContext(
            signals={},
            scene="obedience_trial",
        )
        result = engine.evaluate(ctx)
        # 5 个维度 + 1 行总分
        assert len(result.explanation) == 6
        expected_dims = {"accuracy", "latency", "duration", "attention", "gait"}
        assert set(result.dimension_scores.keys()) == expected_dims


# ============================================================
# 5. 热加载机制
# ============================================================


class TestHotReload:
    """YAML 修改后自动重载测试。"""

    def test_get_returns_cached_instance(self) -> None:
        """同一文件未修改时返回缓存实例。"""
        ScoringEngine.invalidate(PUPPY_YAML)
        e1 = ScoringEngine.get(PUPPY_YAML)
        e2 = ScoringEngine.get(PUPPY_YAML)
        assert e1 is e2

    def test_reload_after_modification(self, tmp_path: Path) -> None:
        """YAML 修改后下次 get 返回新实例。"""
        # 复制一份评分卡到临时目录
        test_yaml = tmp_path / "test_card.yaml"
        test_yaml.write_text(
            """
scoring_engine:
  name: "v1"
  version: "1.0.0"
  scene: "puppy_selection"
  dimensions:
    - id: food_drive
      name: "食物欲望"
      weight: 1.0
      rules:
        - id: r1
          condition: "True"
          score: 50
          label: "中"
""",
            encoding="utf-8",
        )

        ScoringEngine.invalidate(test_yaml)
        e1 = ScoringEngine.get(test_yaml)
        assert e1.spec.name == "v1"

        # 修改 YAML
        time.sleep(0.1)  # 确保 mtime 变化
        test_yaml.write_text(
            """
scoring_engine:
  name: "v2"
  version: "1.0.0"
  scene: "puppy_selection"
  dimensions:
    - id: food_drive
      name: "食物欲望"
      weight: 1.0
      rules:
        - id: r1
          condition: "True"
          score: 50
          label: "中"
""",
            encoding="utf-8",
        )

        e2 = ScoringEngine.get(test_yaml)
        assert e2.spec.name == "v2"
        assert e1 is not e2

    def test_invalidate_clears_cache(self) -> None:
        ScoringEngine.invalidate(PUPPY_YAML)
        e1 = ScoringEngine.get(PUPPY_YAML)
        ScoringEngine.invalidate(PUPPY_YAML)
        e2 = ScoringEngine.get(PUPPY_YAML)
        assert e1 is not e2


# ============================================================
# 6. 聚合方式
# ============================================================


class TestAggregation:
    """聚合方式测试: weighted_sum / max / min。"""

    @pytest.fixture
    def tmp_card(self, tmp_path: Path) -> Path:
        yaml_path = tmp_path / "agg.yaml"
        yaml_path.write_text(
            """
scoring_engine:
  name: "聚合测试"
  version: "1.0.0"
  scene: "puppy_selection"
  dimensions:
    - id: a
      name: "A"
      weight: 0.5
      rules:
        - id: r1
          condition: "True"
          score: 80
          label: "高"
    - id: b
      name: "B"
      weight: 0.5
      rules:
        - id: r2
          condition: "True"
          score: 60
          label: "中"
  thresholds:
    pass: 70
    borderline: 50
    fail: 0
  aggregation: "weighted_sum"
""",
            encoding="utf-8",
        )
        return yaml_path

    def test_weighted_sum(self, tmp_card: Path) -> None:
        engine = ScoringEngine.from_yaml(tmp_card)
        result = engine.evaluate(ScoringContext(signals={}, scene="puppy_selection"))
        # 80*0.5 + 60*0.5 = 70
        assert result.total_score == 70.0

    def test_max_aggregation(self, tmp_card: Path) -> None:
        # 修改 aggregation 为 max
        content = tmp_card.read_text(encoding="utf-8")
        content = content.replace("weighted_sum", "max")
        tmp_card.write_text(content, encoding="utf-8")
        engine = ScoringEngine.from_yaml(tmp_card)
        result = engine.evaluate(ScoringContext(signals={}, scene="puppy_selection"))
        assert result.total_score == 80.0

    def test_min_aggregation(self, tmp_card: Path) -> None:
        content = tmp_card.read_text(encoding="utf-8")
        content = content.replace("weighted_sum", "min")
        tmp_card.write_text(content, encoding="utf-8")
        engine = ScoringEngine.from_yaml(tmp_card)
        result = engine.evaluate(ScoringContext(signals={}, scene="puppy_selection"))
        assert result.total_score == 60.0


# ============================================================
# 7. 阈值判定
# ============================================================


class TestVerdictThresholds:
    """pass / borderline / fail 判定测试。"""

    @pytest.fixture
    def engine(self) -> ScoringEngine:
        ScoringEngine.invalidate(PUPPY_YAML)
        return ScoringEngine.from_yaml(PUPPY_YAML)

    def test_pass_verdict(self, engine: ScoringEngine) -> None:
        ctx = ScoringContext(
            signals={
                "approach_latency": 0.5,
                "approach_speed": 3.0,
                "sniff_duration": 2.0,
                "chase_latency": 0.3,
                "chase_speed": 4.0,
                "hold_duration": 5.0,
                "retreat_distance": 0.2,
                "recovery_time": 1.0,
                "freeze_duration": 0.5,
            },
            scene="puppy_selection",
        )
        result = engine.evaluate(ctx)
        assert result.total_score >= 70  # pass 阈值
        assert result.verdict == "pass"
        assert result.passed is True

    def test_fail_verdict(self, engine: ScoringEngine) -> None:
        ctx = ScoringContext(
            signals={
                "approach_latency": 6.0,
                "chase_latency": 4.0,
                "retreat_distance": 2.5,
            },
            scene="puppy_selection",
        )
        result = engine.evaluate(ctx)
        assert result.total_score < 60  # borderline 阈值
        assert result.verdict == "fail"
        assert result.passed is False


# ============================================================
# 6. USPCA PDI 5 维评分（Phase 2.3b）
# ============================================================


class TestUSPCAPatrolScoring:
    """USPCA PDI 巡逻犬认证 5 维评分端到端测试。

    依据: RESEARCH_STANDARDS.md §2.2 USPCA PDI 详解
    5 维: 准确度(0.30) + 延迟(0.20) + 保持(0.20) + 搜索效率(0.15) + 注意力(0.15)
    """

    @pytest.fixture
    def engine(self) -> ScoringEngine:
        ScoringEngine.invalidate(USPCA_YAML)
        return ScoringEngine.from_yaml(USPCA_YAML)

    def test_scene_is_uspca_patrol(self, engine: ScoringEngine) -> None:
        """评分卡场景标识应为 uspca_patrol。"""
        assert engine.spec.scene == "uspca_patrol"

    def test_five_dimensions_only(self, engine: ScoringEngine) -> None:
        """USPCA 5 维: 不含 courage/gait（与 working_dog_trial 7 维区分）。"""
        dim_ids = {d.id for d in engine.spec.dimensions}
        assert dim_ids == {
            "accuracy",
            "latency",
            "duration",
            "search_efficiency",
            "attention",
        }
        # 不应包含 working_dog_trial 的 courage / gait
        assert "courage" not in dim_ids
        assert "gait" not in dim_ids

    def test_weights_sum_to_one(self, engine: ScoringEngine) -> None:
        """5 维权重之和 = 1.0。"""
        total = sum(d.weight for d in engine.spec.dimensions)
        assert abs(total - 1.0) < 0.001

    def test_thresholds_uspca_aligned(self, engine: ScoringEngine) -> None:
        """USPCA PDI 及格线 70%+ 缓冲 → pass=75。"""
        assert engine.spec.thresholds.pass_ == 75
        assert engine.spec.thresholds.borderline == 60

    def test_excellent_patrol_dog(self, engine: ScoringEngine) -> None:
        """5 维全部优秀 → 总分 ≥ 85，verdict=pass。"""
        ctx = ScoringContext(
            signals={
                "action_correct": 20,
                "action_count": 20,             # 1.0 > 0.95 → 优秀 90
                "command_to_action_latency": 0.3,  # < 0.5 → 优秀 90
                "action_duration": 35.0,        # > 30.0 → 优秀 90
                "search_coverage": 0.90,        # > 0.85
                "search_speed": 0.6,            # > 0.5
                "target_found": True,           # → 优秀 90
                "focus_ratio": 0.90,            # > 0.80
                "unnecessary_movement_count": 0,  # < 2 → 高 90
            },
            scene="uspca_patrol",
        )
        result = engine.evaluate(ctx)
        assert result.total_score >= 85
        assert result.verdict == "pass"
        assert result.passed is True
        assert result.dimension_labels["accuracy"] == "优秀"
        assert result.dimension_labels["search_efficiency"] == "优秀"
        assert result.dimension_labels["attention"] == "高"

    def test_failing_patrol_dog(self, engine: ScoringEngine) -> None:
        """5 维全部不合格 → 总分 ≤ 30，verdict=fail。"""
        ctx = ScoringContext(
            signals={
                "action_correct": 8,
                "action_count": 20,             # 0.4 <= 0.60 → 不合格 30
                "command_to_action_latency": 4.0,  # >= 3.0 → 不合格 30
                "action_duration": 2.0,         # <= 5.0 → 不合格 30
                "search_coverage": 0.20,        # <= 0.40
                "search_speed": 0.1,
                "target_found": False,          # → 不合格 30
                "focus_ratio": 0.30,            # <= 0.50 → 低 30
                "unnecessary_movement_count": 10,
            },
            scene="uspca_patrol",
        )
        result = engine.evaluate(ctx)
        assert result.total_score <= 30
        assert result.verdict == "fail"
        assert result.passed is False

    def test_uspca_latency_3s_threshold(self, engine: ScoringEngine) -> None:
        """USPCA 扣分规则: 指令后 > 3 秒未响应扣 2-5 分 → latency_fail 阈值 3.0s。"""
        # 2.9s → 合格（< 3.0）
        ctx_pass = ScoringContext(
            signals={"command_to_action_latency": 2.9},
            scene="uspca_patrol",
        )
        result_pass = engine.evaluate(ctx_pass)
        assert result_pass.dimension_labels["latency"] == "合格"
        assert result_pass.dimension_scores["latency"] == 60

        # 3.0s → 不合格（>= 3.0）
        ctx_fail = ScoringContext(
            signals={"command_to_action_latency": 3.0},
            scene="uspca_patrol",
        )
        result_fail = engine.evaluate(ctx_fail)
        assert result_fail.dimension_labels["latency"] == "不合格"
        assert result_fail.dimension_scores["latency"] == 30

    def test_uspca_duration_5s_threshold(self, engine: ScoringEngine) -> None:
        """USPCA 扣分规则: 动作保持 < 5 秒扣 2-4 分 → duration_fail 阈值 5.0s。"""
        # 5.1s → 合格（> 5.0）
        ctx_pass = ScoringContext(
            signals={"action_duration": 5.1},
            scene="uspca_patrol",
        )
        result_pass = engine.evaluate(ctx_pass)
        assert result_pass.dimension_labels["duration"] == "合格"

        # 5.0s → 不合格（<= 5.0）
        ctx_fail = ScoringContext(
            signals={"action_duration": 5.0},
            scene="uspca_patrol",
        )
        result_fail = engine.evaluate(ctx_fail)
        assert result_fail.dimension_labels["duration"] == "不合格"

    def test_search_efficiency_with_target_found(self, engine: ScoringEngine) -> None:
        """target_found=True 即使覆盖率低也能达到合格。"""
        ctx = ScoringContext(
            signals={
                "search_coverage": 0.30,  # <= 0.40
                "search_speed": 0.1,
                "target_found": True,     # search_pass 命中（or target_found）
            },
            scene="uspca_patrol",
        )
        result = engine.evaluate(ctx)
        assert result.dimension_labels["search_efficiency"] == "合格"
        assert result.dimension_scores["search_efficiency"] == 60

    def test_attention_with_unnecessary_movement(self, engine: ScoringEngine) -> None:
        """USPCA 扣分: 不必要移动/吠叫扣 1-2 分 → attention_high 要求 < 2。"""
        # 高聚焦 + 1 次不必要移动 → 高
        ctx_high = ScoringContext(
            signals={
                "focus_ratio": 0.85,
                "unnecessary_movement_count": 1,  # < 2
            },
            scene="uspca_patrol",
        )
        result_high = engine.evaluate(ctx_high)
        assert result_high.dimension_labels["attention"] == "高"

        # 高聚焦 + 2 次不必要移动 → 中（不满足 < 2，回退到 focus_ratio > 0.50）
        ctx_medium = ScoringContext(
            signals={
                "focus_ratio": 0.85,
                "unnecessary_movement_count": 2,  # 不 < 2
            },
            scene="uspca_patrol",
        )
        result_medium = engine.evaluate(ctx_medium)
        assert result_medium.dimension_labels["attention"] == "中"

    def test_scene_mismatch_raises(self, engine: ScoringEngine) -> None:
        """场景不匹配应抛出 ValueError。"""
        ctx = ScoringContext(signals={}, scene="obedience_trial")
        with pytest.raises(ValueError, match="场景不匹配"):
            engine.evaluate(ctx)

    def test_missing_signals_use_default(self, engine: ScoringEngine) -> None:
        """所有信号缺失 → 命中各维度的 default True 规则（待评估 40）。"""
        ctx = ScoringContext(signals={}, scene="uspca_patrol")
        result = engine.evaluate(ctx)
        # 5 维都命中 default True → 40 分
        assert result.total_score == 40.0
        assert result.verdict == "fail"  # 40 < 60 borderline
        for label in result.dimension_labels.values():
            assert label == "待评估"

    def test_borderline_dog(self, engine: ScoringEngine) -> None:
        """中等水平犬 → verdict=borderline（60 <= 总分 < 75）。"""
        ctx = ScoringContext(
            signals={
                "action_correct": 14,
                "action_count": 20,             # 0.70 > 0.60 → 合格 60
                "command_to_action_latency": 2.0,  # < 3.0 → 合格 60
                "action_duration": 8.0,         # > 5.0 → 合格 60
                "search_coverage": 0.50,        # > 0.40 → 合格 60
                "search_speed": 0.4,
                "target_found": False,
                "focus_ratio": 0.60,            # > 0.50 → 中 70
                "unnecessary_movement_count": 3,
            },
            scene="uspca_patrol",
        )
        result = engine.evaluate(ctx)
        # 60*0.30 + 60*0.20 + 60*0.20 + 60*0.15 + 70*0.15 = 18+12+12+9+10.5 = 61.5
        assert 60 <= result.total_score < 75
        assert result.verdict == "borderline"
        assert result.passed is False
