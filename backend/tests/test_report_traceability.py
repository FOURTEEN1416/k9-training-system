"""P0-5 报告溯源三字段 + 人工复核决策测试。

Owner: 后端开发
Phase: P0-5（production-readiness-gap-plan.md P0-5 验收线：
      任一 PDF 结果可回答「这个分从哪来、用了哪个模型、是否需复核」）
"""

from __future__ import annotations

from backend.app.services.report import (
    REVIEW_MEAN_CONF_THRESHOLD,
    REVIEW_MIN_CONF_THRESHOLD,
    ReportInput,
    compute_confidence_summary,
    decide_review,
    generate_report,
)
from backend.ml.behavior.rule_engine import BehaviorEpisode
from backend.ml.scoring.schema import DimensionResult, ScoringResult


def _ep(conf: float, behavior: str = "SIT", start: int = 0, end: int = 30) -> BehaviorEpisode:
    return BehaviorEpisode(behavior=behavior, start_frame=start, end_frame=end, confidence=conf)


def _scoring_result() -> ScoringResult:
    return ScoringResult(
        total_score=80.0,
        dimension_scores={"accuracy": 80},
        dimension_labels={"accuracy": "良好"},
        dimension_details=[
            DimensionResult(
                dimension_id="accuracy",
                dimension_name="准确度",
                weight=1.0,
                score=80,
                label="良好",
                hit_rule_id="acc_hit",
            )
        ],
        passed=True,
        verdict="pass",
        explanation=["测试说明"],
        scene="fci_igp",
        card_name="fci_igp",
        card_version="1.0",
    )


class TestConfidenceSummary:
    def test_empty_behaviors_returns_zeros(self):
        s = compute_confidence_summary([])
        assert s == {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "low_ratio": 0.0}

    def test_stats_correct(self):
        s = compute_confidence_summary([_ep(0.9), _ep(0.7), _ep(0.5)])
        assert s["count"] == 3
        assert s["mean"] == 0.7
        assert s["min"] == 0.5
        assert s["max"] == 0.9
        assert s["low_ratio"] == round(1 / 3, 3)


class TestDecideReview:
    def test_rule_only_always_requires_review(self):
        reasons = decide_review("rule_only", [_ep(0.95)])
        assert any("规则引擎" in r for r in reasons)

    def test_empty_episodes_requires_review(self):
        reasons = decide_review("shadow", [])
        assert any("未识别到" in r for r in reasons)

    def test_low_mean_confidence_flagged(self):
        confs = [0.3, 0.5, 0.55]  # mean 0.45 < 0.60
        reasons = decide_review("shadow", [_ep(c) for c in confs])
        assert any(REVIEW_MEAN_CONF_THRESHOLD.__str__()[:4] in r or "平均置信度" in r for r in reasons)

    def test_low_min_confidence_flagged(self):
        reasons = decide_review("shadow", [_ep(0.95), _ep(0.2)])
        assert any("最低 episode 置信度" in r for r in reasons)
        assert any(REVIEW_MIN_CONF_THRESHOLD.__str__() in r for r in reasons)

    def test_healthy_shadow_no_review(self):
        reasons = decide_review("shadow", [_ep(0.9), _ep(0.8), _ep(0.7)])
        assert reasons == []


class TestGenerateReportTraceability:
    def _input(self, **overrides) -> ReportInput:
        kwargs = dict(
            dog_name="测试犬",
            scoring_result=_scoring_result(),
            behaviors=[_ep(0.9), _ep(0.85)],
            recognition_mode="shadow",
            model_version="pose=best.onnx; stgcn_bc=stgcn_bc_dog24.onnx",
            review_reasons=["规则引擎模式（未加载深度识别模型，真实精度未验收）"],
        )
        kwargs.update(overrides)
        return ReportInput(**kwargs)

    def test_pdf_generated_with_trace_fields(self, tmp_path):
        out = tmp_path / "report.pdf"
        result = generate_report(self._input(), output_path=out)
        assert result.exists()
        assert result.stat().st_size > 500
        assert result.read_bytes()[:4] == b"%PDF"

    def test_pdf_generated_without_review(self, tmp_path):
        out = tmp_path / "report.pdf"
        result = generate_report(self._input(review_reasons=[]), output_path=out)
        assert result.exists()
        assert result.read_bytes()[:4] == b"%PDF"

    def test_pdf_generated_empty_behaviors(self, tmp_path):
        out = tmp_path / "report.pdf"
        result = generate_report(self._input(behaviors=[], review_reasons=["未识别到任何行为 episode"]), output_path=out)
        assert result.exists()
