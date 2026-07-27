"""评分引擎模块.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.4b
依据: dev-docs/complex-features/scoring-card-schema.md

公开 API:
    ScoringEngine    — 评分引擎（from_yaml / get / evaluate）
    ScoringContext   — 评分输入（signals + scene）
    ScoringResult    — 评分输出（total_score + dimension_scores + verdict）
    evaluate_condition — 条件表达式求值（条件解析器）
"""
from backend.ml.scoring.engine import ScoringEngine
from backend.ml.scoring.schema import (
    DimensionResult,
    ScoringCardSpec,
    ScoringContext,
    ScoringResult,
)

__all__ = [
    "ScoringEngine",
    "ScoringContext",
    "ScoringResult",
    "ScoringCardSpec",
    "DimensionResult",
]
