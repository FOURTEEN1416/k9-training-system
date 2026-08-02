"""标注质量校验脚本（Phase 3 真实数据标注辅线）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1d 辅线
依据: dev-docs/stages/phase-3.md §3.1d 数据源优先级

功能:
    校验 pyskl pickle 格式标注的质量，确保训练数据可靠性。

校验项:
    1. 关键点完整性: 每个样本是否有 24 个关键点 + (T, 24, 3) shape
    2. 置信度分布: 低置信度关键点占比（conf < 0.3 视为不可靠）
    3. 行为类别合法性: label 是否在 22 类中
    4. 边界标签合理性: 边界帧占比（正常 1%-10%，过高/过低异常）
    5. 时序一致性: 帧间关键点位移（异常大位移 = 标注错误）
    6. 标签分布平衡: 22 类样本数分布（避免严重不平衡）

用法:
    python scripts/validate_annotations.py \\
        --input data/stgcn_bc/train_real.pkl \\
        --report reports/annotation_quality.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.behavior.constants import NUM_KEYPOINTS
from backend.ml.behavior.stgcn_bc.labels import (
    BEHAVIOR_TO_IDX,
    IDX_TO_BEHAVIOR,
    NUM_BEHAVIORS,
)


def validate_pyskl_pickle(pkl_path: str) -> Dict:
    """校验 pyskl pickle 格式标注质量.

    Args:
        pkl_path: pyskl pickle 路径

    Returns:
        校验结果字典
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        return {"valid": False, "error": f"文件不存在: {pkl_path}"}

    with open(pkl_path, "rb") as f:
        samples: List[Dict] = pickle.load(f)

    result = {
        "file": str(pkl_path),
        "total_samples": len(samples),
        "checks": {},
        "issues": [],
    }

    if not samples:
        result["valid"] = False
        result["error"] = "空数据集（0 样本）"
        return result

    # ===== 1. 关键点完整性 =====
    shape_ok = 0
    shape_issues = []
    for i, s in enumerate(samples):
        kpt = s.get("keypoint")
        if kpt is None:
            shape_issues.append(f"样本 {i}: 无 keypoint 字段")
            continue
        if not isinstance(kpt, np.ndarray):
            shape_issues.append(f"样本 {i}: keypoint 非 ndarray ({type(kpt)})")
            continue
        if kpt.ndim != 4:
            shape_issues.append(f"样本 {i}: keypoint ndim={kpt.ndim} (期望 4)")
            continue
        M, T, V, C = kpt.shape
        if M != 1:
            shape_issues.append(f"样本 {i}: M={M} (期望 1)")
            continue
        if V != NUM_KEYPOINTS:
            shape_issues.append(f"样本 {i}: V={V} (期望 {NUM_KEYPOINTS})")
            continue
        if C < 2:
            shape_issues.append(f"样本 {i}: C={C} (期望 >=2)")
            continue
        shape_ok += 1

    result["checks"]["keypoint_shape"] = {
        "passed": len(shape_issues) == 0,
        "ok_count": shape_ok,
        "issue_count": len(shape_issues),
        "issues_preview": shape_issues[:5],
    }
    if shape_issues:
        result["issues"].extend(shape_issues[:10])

    # ===== 2. 置信度分布 =====
    low_conf_ratios = []
    for s in samples:
        kpt = s.get("keypoint")
        if kpt is None or not isinstance(kpt, np.ndarray) or kpt.ndim != 4:
            continue
        if kpt.shape[-1] >= 3:
            conf = kpt[0, :, :, 2]  # (T, V)
            low_conf_ratio = float(np.mean(conf < 0.3))
            low_conf_ratios.append(low_conf_ratio)

    if low_conf_ratios:
        result["checks"]["confidence"] = {
            "mean_low_conf_ratio": round(float(np.mean(low_conf_ratios)), 4),
            "max_low_conf_ratio": round(float(np.max(low_conf_ratios)), 4),
            "samples_above_50pct_low_conf": int(np.sum(np.array(low_conf_ratios) > 0.5)),
            "passed": float(np.mean(low_conf_ratios)) < 0.5,
        }
    else:
        result["checks"]["confidence"] = {"passed": True, "note": "无 3 通道关键点，跳过"}

    # ===== 3. 行为类别合法性 =====
    label_counter = Counter()
    unknown_labels = []
    for i, s in enumerate(samples):
        label = s.get("label", -1)
        if label not in IDX_TO_BEHAVIOR:
            unknown_labels.append((i, label))
        else:
            label_counter[label] += 1

    result["checks"]["label_validity"] = {
        "passed": len(unknown_labels) == 0,
        "valid_count": sum(label_counter.values()),
        "unknown_count": len(unknown_labels),
        "unknown_preview": unknown_labels[:5],
        "unique_labels": len(label_counter),
        "expected_labels": NUM_BEHAVIORS,
    }

    # ===== 4. 边界标签合理性 =====
    boundary_ratios = []
    for s in samples:
        boundary = s.get("boundary")
        if boundary is None:
            continue
        if isinstance(boundary, np.ndarray) and len(boundary) > 0:
            ratio = float(np.mean(boundary > 0.5))
            boundary_ratios.append(ratio)

    if boundary_ratios:
        result["checks"]["boundary"] = {
            "mean_ratio": round(float(np.mean(boundary_ratios)), 4),
            "min_ratio": round(float(np.min(boundary_ratios)), 4),
            "max_ratio": round(float(np.max(boundary_ratios)), 4),
            "samples_high_boundary": int(np.sum(np.array(boundary_ratios) > 0.20)),
            "samples_low_boundary": int(np.sum(np.array(boundary_ratios) < 0.01)),
            "passed": 0.01 <= float(np.mean(boundary_ratios)) <= 0.20,
        }
    else:
        result["checks"]["boundary"] = {"passed": True, "note": "无边界标签，跳过"}

    # ===== 5. 时序一致性（帧间位移） =====
    abnormal_displacements = []
    for i, s in enumerate(samples):
        kpt = s.get("keypoint")
        if kpt is None or not isinstance(kpt, np.ndarray) or kpt.ndim != 4:
            continue
        if kpt.shape[1] < 2:
            continue
        # 计算帧间关键点位移
        diff = np.diff(kpt[0, :, :, :2], axis=0)  # (T-1, V, 2)
        displacement = np.sqrt(np.sum(diff**2, axis=-1))  # (T-1, V)
        # 归一化到 [0, 1] 坐标系下，位移 > 0.3 视为异常
        max_disp = float(np.max(displacement))
        if max_disp > 0.3:
            abnormal_displacements.append((i, round(max_disp, 4)))

    result["checks"]["temporal_consistency"] = {
        "abnormal_count": len(abnormal_displacements),
        "abnormal_preview": abnormal_displacements[:5],
        "passed": len(abnormal_displacements) < len(samples) * 0.1,  # < 10% 异常
    }

    # ===== 6. 标签分布平衡 =====
    label_distribution = {
        IDX_TO_BEHAVIOR.get(idx, f"unknown_{idx}"): count
        for idx, count in sorted(label_counter.items())
    }
    counts = list(label_counter.values())
    if counts:
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    else:
        min_count = max_count = 0
        imbalance_ratio = 0

    result["checks"]["label_balance"] = {
        "distribution": label_distribution,
        "min_count": min_count,
        "max_count": max_count,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "passed": imbalance_ratio < 10.0,  # 最大/最小 < 10 视为可接受
    }

    # ===== 总体判定 =====
    all_passed = all(
        check.get("passed", False) for check in result["checks"].values()
    )
    result["valid"] = all_passed
    result["summary"] = f"{'✅ 通过' if all_passed else '❌ 存在问题'}: " + \
        f"{sum(1 for c in result['checks'].values() if c.get('passed', False))}/" + \
        f"{len(result['checks'])} 项检查通过"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="标注质量校验")
    parser.add_argument(
        "--input", type=str, required=True,
        help="pyskl pickle 路径",
    )
    parser.add_argument(
        "--report", type=str, default="reports/annotation_quality.json",
        help="JSON 报告输出路径",
    )
    args = parser.parse_args()

    result = validate_pyskl_pickle(args.input)

    print("\n" + "=" * 70)
    print("标注质量校验报告")
    print("=" * 70)
    print(f"文件: {result.get('file', args.input)}")
    print(f"总样本: {result.get('total_samples', 0)}")
    print(f"总体: {result.get('summary', '未知')}")

    for check_name, check_result in result.get("checks", {}).items():
        passed = check_result.get("passed", False)
        print(f"\n[{check_name}] {'✅' if passed else '❌'}")
        for k, v in check_result.items():
            if k != "passed":
                print(f"  {k}: {v}")

    if result.get("issues"):
        print(f"\n问题列表（前 10 条）:")
        for issue in result["issues"][:10]:
            print(f"  - {issue}")

    # 保存报告
    report_path = PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n📄 报告已保存: {report_path}")

    sys.exit(0 if result.get("valid", False) else 1)


if __name__ == "__main__":
    main()
