"""FCI-IGP 评分卡 YAML 验证脚本（Phase 3.4c）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.4c
依据: dev-docs/stages/phase-3.md §3.4c
      dev-docs/research/RESEARCH_FCI_IGP_STANDARD.md §6

验证策略（三档 + DQ 硬约束）:
    1. excellent 档: 所有信号达到 Excellent 级别（96+），期望 verdict=pass + rating=Excellent
    2. failing 档: 所有信号低于 Satisfactory（<70），期望 verdict=fail + rating=Insufficient
    3. borderline 档: 信号在 Satisfactory 边界（70-80），期望 verdict=pass + rating=Satisfactory
    4. DQ 硬约束: 枪怯/不放口/衔取不吐，期望 verdict=fail + disqualified=True + 总分=0

评估指标:
    - YAML Schema 校验通过
    - 7 维权重和 = 1.0
    - 22 行为 100% 覆盖 IGP 三阶段（A=4 / B=12 / C=6）
    - 三档评分合理性（excellent ≥ 96 / borderline 70-80 / failing < 70）
    - DQ 触发正确性（force_fail + 总分清零）

决策门:
    - 4 档全部通过 → Phase 3.4c 达标
    - 任一档失败 → 评分卡 YAML 需修正

用法:
    python scripts/eval_fci_igp.py
    python scripts/eval_fci_igp.py --report reports/phase-3.4c-fci-igp-eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.scoring import ScoringContext, ScoringEngine, ScoringResult
from backend.ml.scoring.schema import ScoringCardSpec

FCI_IGP_YAML = PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs" / "fci_igp.yaml"


# ============================================================
# 三档测试信号（对齐 fci_igp.yaml 规则条件）
# ============================================================

# Excellent 档：所有维度达到 96 分阈值
EXCELLENT_SIGNALS = {
    # accuracy: action_correct / action_count > 0.96
    "action_correct": 97,
    "action_count": 100,
    # latency: command_to_action_latency < 0.5
    "command_to_action_latency": 0.3,
    # duration: action_duration > 30.0
    "action_duration": 35.0,
    # search: search_coverage > 0.85 and search_speed > 0.5 and target_found
    "search_coverage": 0.90,
    "search_speed": 0.60,
    "target_found": True,
    # attention: focus_ratio > 0.80 and unnecessary_movement_count < 2
    "focus_ratio": 0.85,
    "unnecessary_movement_count": 1,
    # courage: courage_score > 0.90 and not avoidance_detected
    "courage_score": 0.92,
    "avoidance_detected": False,
    # gait: gait_symmetry > 0.90 and pace_change_smoothness > 0.85
    "gait_symmetry": 0.92,
    "pace_change_smoothness": 0.88,
    # DQ 信号（不触发）
    "gunshot_reaction": "steady",
    "release_command_count": 1,
    "sleeve_released": True,
    "retrieve_release_command_count": 1,
    "dumbbell_released": True,
    "excretion_detected": False,
    "down_break_distance": 1.0,
}

# Borderline 档：信号在 Satisfactory 边界（70-80）
BORDERLINE_SIGNALS = {
    # accuracy: action_correct / action_count > 0.70 (Satisfactory)
    "action_correct": 75,
    "action_count": 100,
    # latency: command_to_action_latency < 3.0 (Satisfactory)
    "command_to_action_latency": 2.5,
    # duration: action_duration > 5.0 (Satisfactory)
    "action_duration": 7.0,
    # search: search_coverage > 0.40 or target_found (Satisfactory)
    "search_coverage": 0.50,
    "search_speed": 0.35,
    "target_found": True,
    # attention: focus_ratio > 0.40 (Satisfactory)
    "focus_ratio": 0.50,
    "unnecessary_movement_count": 3,
    # courage: courage_score > 0.50 (Satisfactory)
    "courage_score": 0.55,
    "avoidance_detected": False,
    # gait: gait_symmetry > 0.50 (Satisfactory)
    "gait_symmetry": 0.60,
    "pace_change_smoothness": 0.55,
    # DQ 信号（不触发）
    "gunshot_reaction": "steady",
    "release_command_count": 1,
    "sleeve_released": True,
    "retrieve_release_command_count": 1,
    "dumbbell_released": True,
    "excretion_detected": False,
    "down_break_distance": 1.0,
}

# Failing 档：所有维度低于 Satisfactory
FAILING_SIGNALS = {
    # accuracy: action_correct / action_count <= 0.70
    "action_correct": 50,
    "action_count": 100,
    # latency: command_to_action_latency >= 3.0
    "command_to_action_latency": 5.0,
    # duration: action_duration <= 5.0
    "action_duration": 3.0,
    # search: search_coverage <= 0.40 and not target_found
    "search_coverage": 0.20,
    "search_speed": 0.10,
    "target_found": False,
    # attention: focus_ratio <= 0.40
    "focus_ratio": 0.30,
    "unnecessary_movement_count": 5,
    # courage: courage_score <= 0.50 or avoidance_detected
    "courage_score": 0.30,
    "avoidance_detected": True,
    # gait: gait_symmetry <= 0.50
    "gait_symmetry": 0.40,
    "pace_change_smoothness": 0.35,
    # DQ 信号（不触发，仅评分低）
    "gunshot_reaction": "steady",
    "release_command_count": 1,
    "sleeve_released": True,
    "retrieve_release_command_count": 1,
    "dumbbell_released": True,
    "excretion_detected": False,
    "down_break_distance": 1.0,
}

# DQ 档：枪怯（gunshot_reaction == 'shy'）
DQ_GUNSHY_SIGNALS = {**EXCELLENT_SIGNALS, "gunshot_reaction": "shy"}

# DQ 档：不放口（release_command_count >= 2 and not sleeve_released）
DQ_NO_RELEASE_SIGNALS = {
    **EXCELLENT_SIGNALS,
    "release_command_count": 2,
    "sleeve_released": False,
}

# DQ 档：衔取不吐（retrieve_release_command_count >= 3 and not dumbbell_released）
DQ_RETRIEVE_FAIL_SIGNALS = {
    **EXCELLENT_SIGNALS,
    "retrieve_release_command_count": 3,
    "dumbbell_released": False,
}


# ============================================================
# 评估器
# ============================================================


class FciIgpEvaluator:
    """FCI-IGP 评分卡验证器。"""

    def __init__(self, yaml_path: Path = FCI_IGP_YAML) -> None:
        self.yaml_path = yaml_path
        self.engine = ScoringEngine.from_yaml(yaml_path)
        self.spec: ScoringCardSpec = self.engine.spec

    def validate_schema(self) -> dict:
        """1. YAML Schema 校验 + 7 维权重和 + 22 行为覆盖。"""
        result = {
            "yaml_loaded": True,
            "scene": self.spec.scene,
            "igp_level": self.spec.igp_level,
            "dimensions_count": len(self.spec.dimensions),
            "weights_sum": round(sum(d.weight for d in self.spec.dimensions), 4),
            "weights_sum_valid": abs(sum(d.weight for d in self.spec.dimensions) - 1.0) < 0.001,
            "dq_count": len(self.spec.disqualifications),
            "behavior_mapping_count": len(self.spec.behavior_mapping),
            "thresholds": {
                "pass": self.spec.thresholds.pass_,
                "borderline": self.spec.thresholds.borderline,
                "fail": self.spec.thresholds.fail,
            },
        }

        # 22 行为覆盖检查
        expected_behaviors = {
            # P0 8
            "sit", "down", "stand", "heel", "sit_up", "stay", "bark", "bite",
            # P1 8
            "track", "alert_sit", "alert_down", "apprehend", "escort", "obstacle", "recall", "watch",
            # P2 6
            "retrieve", "jump", "scale", "search_blind", "guard", "release",
        }
        actual_behaviors = set(self.spec.behavior_mapping.keys())
        result["behaviors_expected"] = len(expected_behaviors)
        result["behaviors_actual"] = len(actual_behaviors)
        result["behaviors_missing"] = sorted(expected_behaviors - actual_behaviors)
        result["behaviors_extra"] = sorted(actual_behaviors - expected_behaviors)
        result["behaviors全覆盖"] = len(expected_behaviors - actual_behaviors) == 0

        # IGP 三阶段覆盖
        phase_a = [b for b, m in self.spec.behavior_mapping.items() if m.igp_phase == "A"]
        phase_b = [b for b, m in self.spec.behavior_mapping.items() if m.igp_phase == "B"]
        phase_c = [b for b, m in self.spec.behavior_mapping.items() if m.igp_phase == "C"]
        result["igp_phase_a_count"] = len(phase_a)
        result["igp_phase_b_count"] = len(phase_b)
        result["igp_phase_c_count"] = len(phase_c)
        result["igp_phase_distribution"] = {"A": phase_a, "B": phase_b, "C": phase_c}

        result["passed"] = (
            result["weights_sum_valid"]
            and result["behaviors全覆盖"]
            and result["dq_count"] == 3
        )
        return result

    def evaluate_tier(self, tier_name: str, signals: dict) -> dict:
        """评估单档信号。"""
        ctx = ScoringContext(signals=signals, scene="fci_igp")
        result: ScoringResult = self.engine.evaluate(ctx)
        return {
            "tier": tier_name,
            "total_score": result.total_score,
            "verdict": result.verdict,
            "passed": result.passed,
            "rating": result.rating,
            "disqualified": result.disqualified,
            "dimension_scores": result.dimension_scores,
            "dimension_labels": result.dimension_labels,
        }

    def evaluate_dq(self, dq_name: str, signals: dict, expected_dq_id: str) -> dict:
        """评估 DQ 触发。"""
        ctx = ScoringContext(signals=signals, scene="fci_igp")
        result: ScoringResult = self.engine.evaluate(ctx)
        return {
            "dq_case": dq_name,
            "expected_dq_id": expected_dq_id,
            "actual_dq_id": result.disqualification_hit,
            "disqualified": result.disqualified,
            "total_score": result.total_score,
            "verdict": result.verdict,
            "rating": result.rating,
            "passed": (
                result.disqualified
                and result.disqualification_hit == expected_dq_id
                and result.total_score == 0.0
                and result.verdict == "fail"
                and result.rating == "Insufficient"
            ),
        }

    def run_all(self) -> dict:
        """运行完整评估。"""
        print("=" * 70)
        print("FCI-IGP 评分卡验证（Phase 3.4c）")
        print(f"YAML: {self.yaml_path}")
        print("=" * 70)

        # 1. Schema 校验
        print("\n[1/5] YAML Schema 校验...")
        schema_result = self.validate_schema()
        self._print_schema(schema_result)

        # 2. Excellent 档
        print("\n[2/5] Excellent 档验证...")
        excellent = self.evaluate_tier("excellent", EXCELLENT_SIGNALS)
        self._print_tier(excellent, expected_score_ge=96, expected_verdict="pass", expected_rating="Excellent")

        # 3. Borderline 档
        print("\n[3/5] Borderline 档验证...")
        borderline = self.evaluate_tier("borderline", BORDERLINE_SIGNALS)
        self._print_tier(borderline, expected_score_ge=70, expected_score_lt=80, expected_verdict="pass", expected_rating="Satisfactory")

        # 4. Failing 档
        print("\n[4/5] Failing 档验证...")
        failing = self.evaluate_tier("failing", FAILING_SIGNALS)
        self._print_tier(failing, expected_score_lt=70, expected_verdict="fail", expected_rating="Insufficient")

        # 5. DQ 硬约束
        print("\n[5/5] DQ 硬约束验证...")
        dq_gunshy = self.evaluate_dq("gun_shy", DQ_GUNSHY_SIGNALS, "gunfire_fail")
        dq_no_release = self.evaluate_dq("no_release", DQ_NO_RELEASE_SIGNALS, "release_fail")
        dq_retrieve = self.evaluate_dq("retrieve_fail", DQ_RETRIEVE_FAIL_SIGNALS, "retrieve_fail")
        self._print_dq(dq_gunshy)
        self._print_dq(dq_no_release)
        self._print_dq(dq_retrieve)

        # 汇总
        all_passed = (
            schema_result["passed"]
            and excellent["total_score"] >= 96.0
            and excellent["verdict"] == "pass"
            and excellent["rating"] == "Excellent"
            and 70.0 <= borderline["total_score"] < 80.0
            and borderline["verdict"] == "pass"
            and borderline["rating"] == "Satisfactory"
            and failing["total_score"] < 70.0
            and failing["verdict"] == "fail"
            and failing["rating"] == "Insufficient"
            and dq_gunshy["passed"]
            and dq_no_release["passed"]
            and dq_retrieve["passed"]
        )

        summary = {
            "schema": schema_result,
            "excellent": excellent,
            "borderline": borderline,
            "failing": failing,
            "dq_gunshy": dq_gunshy,
            "dq_no_release": dq_no_release,
            "dq_retrieve": dq_retrieve,
            "all_passed": all_passed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "yaml_path": str(self.yaml_path),
        }

        print("\n" + "=" * 70)
        if all_passed:
            print("✅ Phase 3.4c 验证通过：4 档全部符合预期")
        else:
            print("❌ Phase 3.4c 验证失败：部分档位不符合预期")
        print("=" * 70)

        return summary

    def _print_schema(self, r: dict) -> None:
        print(f"  scene: {r['scene']}")
        print(f"  igp_level: {r['igp_level']}")
        print(f"  dimensions: {r['dimensions_count']} 维")
        print(f"  weights_sum: {r['weights_sum']} ({'✅' if r['weights_sum_valid'] else '❌'})")
        print(f"  DQ 规则: {r['dq_count']} 条 ({'✅' if r['dq_count'] == 3 else '❌'} 期望 3)")
        print(f"  行为映射: {r['behaviors_actual']}/{r['behaviors_expected']} ({'✅' if r['behaviors全覆盖'] else '❌'})")
        if r['behaviors_missing']:
            print(f"  缺失行为: {r['behaviors_missing']}")
        print(f"  IGP 阶段分布: A={r['igp_phase_a_count']} / B={r['igp_phase_b_count']} / C={r['igp_phase_c_count']}")
        print(f"  thresholds: pass={r['thresholds']['pass']} / borderline={r['thresholds']['borderline']} / fail={r['thresholds']['fail']}")
        print(f"  结果: {'✅ 通过' if r['passed'] else '❌ 失败'}")

    def _print_tier(
        self,
        r: dict,
        expected_score_ge: float | None = None,
        expected_score_lt: float | None = None,
        expected_verdict: str | None = None,
        expected_rating: str | None = None,
    ) -> None:
        print(f"  总分: {r['total_score']}")
        print(f"  verdict: {r['verdict']} ({'✅' if expected_verdict is None or r['verdict'] == expected_verdict else '❌ 期望 ' + expected_verdict})")
        print(f"  rating: {r['rating']} ({'✅' if expected_rating is None or r['rating'] == expected_rating else '❌ 期望 ' + expected_rating})")
        if expected_score_ge is not None:
            print(f"  分数 >= {expected_score_ge}: {'✅' if r['total_score'] >= expected_score_ge else '❌'}")
        if expected_score_lt is not None:
            print(f"  分数 < {expected_score_lt}: {'✅' if r['total_score'] < expected_score_lt else '❌'}")
        print(f"  维度分数: {r['dimension_scores']}")
        print(f"  维度标签: {r['dimension_labels']}")

    def _print_dq(self, r: dict) -> None:
        print(f"  {r['dq_case']}: DQ={r['disqualified']} ({'✅' if r['passed'] else '❌'})")
        print(f"    expected_dq={r['expected_dq_id']} / actual_dq={r['actual_dq_id']}")
        print(f"    total_score={r['total_score']} (期望 0.0)")
        print(f"    verdict={r['verdict']} / rating={r['rating']}")


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="FCI-IGP 评分卡验证（Phase 3.4c）")
    parser.add_argument(
        "--yaml",
        type=str,
        default=str(FCI_IGP_YAML),
        help="FCI-IGP 评分卡 YAML 路径",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="reports/phase-3.4c-fci-igp-eval.json",
        help="JSON 报告输出路径",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        print(f"❌ YAML 文件不存在: {yaml_path}")
        sys.exit(1)

    evaluator = FciIgpEvaluator(yaml_path=yaml_path)
    summary = evaluator.run_all()

    report_path = PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n📄 报告已保存: {report_path}")

    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
