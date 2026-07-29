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
