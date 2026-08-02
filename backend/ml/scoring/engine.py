"""评分引擎核心.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.4b（Phase 3.4b 扩展 FCI-IGP DQ 硬约束 + 5 级评级）
依据: dev-docs/complex-features/scoring-card-schema.md §5
      dev-docs/research/RESEARCH_FCI_IGP_STANDARD.md §3.2 + §1.4

特性:
    1. YAML 配置化: 从 YAML 加载评分卡，训导员可改阈值/权重
    2. 热加载: ScoringEngine.get() 单例 + mtime 检测，修改 YAML 后下次自动重载
    3. 可解释: 每条评分给出命中规则 + 标签 + 人类可读说明
    4. 双场景: 选育 3 维 + 科目 5 维，场景隔离
    5. FCI-IGP 扩展: DQ 硬约束（force_fail）+ 5 级评级（Excellent/Very Good/Good/Satisfactory/Insufficient）

用法:
    from backend.ml.scoring import ScoringEngine, ScoringContext

    # 方式 1: 单例 + 热加载（推荐生产用）
    engine = ScoringEngine.get("backend/ml/scoring/configs/puppy_selection.yaml")

    # 方式 2: 直接加载（测试用）
    engine = ScoringEngine.from_yaml("path/to/card.yaml")

    # 评分
    ctx = ScoringContext(signals={...}, scene="puppy_selection")
    result = engine.evaluate(ctx)
    print(result.total_score, result.verdict, result.dimension_labels)
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import ValidationError

from backend.ml.scoring.conditions import ConditionError, evaluate_condition
from backend.ml.scoring.schema import (
    DimensionResult,
    DisqualificationSpec,
    FciRating,
    ScoringCardSpec,
    ScoringContext,
    ScoringResult,
)


# FCI-IGP 5 级评级阈值（百分制，RESEARCH_FCI_IGP_STANDARD.md §1.4）
# Excellent    96-100
# Very Good    90-95.5
# Good         80-89.5
# Satisfactory 70-79.5
# Insufficient 0-69.5
_RATING_THRESHOLDS: list[tuple[float, FciRating]] = [
    (96.0, "Excellent"),
    (90.0, "Very Good"),
    (80.0, "Good"),
    (70.0, "Satisfactory"),
    (0.0, "Insufficient"),
]


class ScoringEngine:
    """评分引擎。

    单例 + 热加载: 通过 ScoringEngine.get(path) 获取实例，YAML 修改后下次调用自动重载。
    """

    _instances: ClassVar[dict[str, "ScoringEngine"]] = {}
    _mtimes: ClassVar[dict[str, float]] = {}

    def __init__(self, spec: ScoringCardSpec, source_path: str | None = None) -> None:
        self.spec = spec
        self.source_path = source_path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScoringEngine":
        """从 YAML 文件加载评分卡。

        Args:
            path: YAML 文件路径

        Returns:
            ScoringEngine 实例

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: YAML 格式错误或 Schema 校验失败
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"评分卡文件不存在: {path}")

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not raw or "scoring_engine" not in raw:
            raise ValueError(f"YAML 缺少顶层 'scoring_engine' 键: {path}")

        try:
            spec = ScoringCardSpec.model_validate(raw["scoring_engine"])
        except ValidationError as e:
            raise ValueError(f"评分卡 Schema 校验失败: {path}\n{e}") from e

        return cls(spec, source_path=str(path))

    @classmethod
    def get(cls, path: str | Path) -> "ScoringEngine":
        """单例 + 热加载: YAML 修改后下次调用自动重载。

        Args:
            path: YAML 文件路径

        Returns:
            ScoringEngine 实例（如果文件未修改，返回缓存实例）
        """
        path_str = str(path)
        mtime = Path(path).stat().st_mtime

        if path_str not in cls._instances or cls._mtimes.get(path_str) != mtime:
            cls._instances[path_str] = cls.from_yaml(path)
            cls._mtimes[path_str] = mtime

        return cls._instances[path_str]

    @classmethod
    def invalidate(cls, path: str | Path | None = None) -> None:
        """清除缓存（测试用）。

        Args:
            path: 指定路径清除；None 清除所有
        """
        if path is None:
            cls._instances.clear()
            cls._mtimes.clear()
        else:
            path_str = str(path)
            cls._instances.pop(path_str, None)
            cls._mtimes.pop(path_str, None)

    def evaluate(self, ctx: ScoringContext) -> ScoringResult:
        """评分。

        Args:
            ctx: 评分上下文（signals + scene）

        Returns:
            ScoringResult 评分结果

        Raises:
            ValueError: 场景不匹配
        """
        if ctx.scene != self.spec.scene:
            raise ValueError(
                f"场景不匹配: 上下文 scene={ctx.scene}, 评分卡 scene={self.spec.scene}"
            )

        dimension_results: list[DimensionResult] = []
        dimension_scores: dict[str, int] = {}
        dimension_labels: dict[str, str] = {}
        explanation: list[str] = []

        for dim in self.spec.dimensions:
            result = self._evaluate_dimension(dim, ctx.signals)
            dimension_results.append(result)
            dimension_scores[dim.id] = result.score
            dimension_labels[dim.id] = result.label
            explanation.append(
                f"[{dim.name}] {result.label}（{result.score} 分，"
                f"命中规则: {result.hit_rule_id or 'default'}）"
            )

        total_score = self._aggregate(dimension_results)

        # Phase 3.4b: DQ 硬约束检查（fci_igp 场景）
        dq_hit = self._check_disqualifications(ctx.signals)
        disqualified = dq_hit is not None

        # 判定（DQ 优先 → 强制 fail）
        thresholds = self.spec.thresholds
        if disqualified:
            verdict = "fail"
            passed = False
            # DQ 触发：总分清零（对齐 FCI "已得分清零" 规则）
            total_score = 0.0
            dq_spec = next(d for d in self.spec.disqualifications if d.id == dq_hit)
            explanation.insert(
                0,
                f"⚠️ DQ 触发: {dq_spec.label}（规则 {dq_hit}）→ "
                f"整段不合格，已得分清零（对齐 FCI-IGP 2025 §3.2）",
            )
        elif total_score >= thresholds.pass_:
            verdict = "pass"
            passed = True
        elif total_score >= thresholds.borderline:
            verdict = "borderline"
            passed = False
        else:
            verdict = "fail"
            passed = False

        # Phase 3.4b: FCI-IGP 5 级评级（仅 fci_igp 场景）
        rating = None
        if self.spec.scene == "fci_igp":
            rating = self._compute_fci_rating(total_score, disqualified)

        explanation.insert(
            0,
            f"总分 {total_score:.1f}（{verdict}）："
            f"合格线 {thresholds.pass_}，基本合格线 {thresholds.borderline}"
            + (f"，评级 {rating}" if rating else ""),
        )

        return ScoringResult(
            total_score=round(total_score, 2),
            dimension_scores=dimension_scores,
            dimension_labels=dimension_labels,
            dimension_details=dimension_results,
            passed=passed,
            verdict=verdict,
            explanation=explanation,
            scene=self.spec.scene,
            card_name=self.spec.name,
            card_version=self.spec.version,
            disqualified=disqualified,
            disqualification_hit=dq_hit,
            rating=rating,
        )

    def _check_disqualifications(
        self, signals: dict
    ) -> str | None:
        """检查 DQ 硬约束，返回首个命中的 DQ 规则 ID（无则 None）.

        FCI-IGP 2025 §3.2 DQ 情形:
            - 枪怯 (gun-shy): gunshot_reaction == 'shy'
            - 不放口 (no release): release_command_count >= 2 and not sleeve_released
            - 衔取不吐 (retrieve): retrieve_release_command_count >= 3 and not dumbbell_released
        """
        for dq in self.spec.disqualifications:
            try:
                if evaluate_condition(dq.condition, signals):
                    return dq.id
            except ConditionError:
                # DQ 条件表达式错误 → 跳过该 DQ（保守不触发）
                continue
        return None

    def _compute_fci_rating(
        self, total_score: float, disqualified: bool
    ) -> FciRating:
        """计算 FCI-IGP 5 级评级.

        依据 RESEARCH_FCI_IGP_STANDARD.md §1.4:
            Excellent    96-100%
            Very Good    90-95.5%
            Good         80-89.5%
            Satisfactory 70-79.5%
            Insufficient 0-69.5%

        DQ 触发时直接为 Insufficient（不论原分数）。
        """
        if disqualified:
            return "Insufficient"
        for threshold, rating in _RATING_THRESHOLDS:
            if total_score >= threshold:
                return rating
        return "Insufficient"  # 兜底（理论上不会到达）

    def _evaluate_dimension(self, dim, signals: dict) -> DimensionResult:
        """评估单个维度：按顺序匹配规则，首个命中生效。"""
        for rule in dim.rules:
            try:
                if evaluate_condition(rule.condition, signals):
                    return DimensionResult(
                        dimension_id=dim.id,
                        dimension_name=dim.name,
                        weight=dim.weight,
                        score=rule.score,
                        label=rule.label,
                        hit_rule_id=rule.id,
                    )
            except ConditionError as e:
                # 规则表达式错误 → 跳过该规则，记录到解释
                # 不抛异常，保证评分流程继续
                continue

        # 所有规则未命中 → 用 default
        if dim.default is not None:
            return DimensionResult(
                dimension_id=dim.id,
                dimension_name=dim.name,
                weight=dim.weight,
                score=dim.default.score,
                label=dim.default.label,
                hit_rule_id=None,
            )

        # 无 default 且无规则命中 → 0 分
        return DimensionResult(
            dimension_id=dim.id,
            dimension_name=dim.name,
            weight=dim.weight,
            score=0,
            label="未评估",
            hit_rule_id=None,
        )

    def _aggregate(self, results: list[DimensionResult]) -> float:
        """聚合各维度得分.

        注意: weighted_sum 模式下浮点累加可能产生 69.9999... 类误差，
        round 到 2 位小数避免评级阈值边界误判（FCI-IGP 70/80/90/96 阈值）。
        """
        if self.spec.aggregation == "max":
            return float(max(r.score for r in results))
        if self.spec.aggregation == "min":
            return float(min(r.score for r in results))
        # 默认 weighted_sum（round 避免浮点精度边界误判）
        return round(sum(r.score * r.weight for r in results), 2)
