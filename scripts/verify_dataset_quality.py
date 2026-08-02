"""训练数据集质量验证脚本（Phase 3 真实数据标注辅线）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1d 辅线
依据: dev-docs/stages/phase-3.md §3.1d 数据源优先级

功能:
    验证 pyskl pickle 格式训练数据集的完整性和可用性，
    在执行 build_training_dataset.py 之前/之后运行，确保数据质量。

验证项:
    1. 格式合规性: keypoint shape / dtype / label 范围
    2. 数值健康性: NaN / Inf / 异常值检测
    3. 标签分布: 22 类覆盖率 + 长尾类别识别
    4. 边界标签: 非空 + 时间对齐
    5. 训练/验证泄漏: clip_id 跨集检测
    6. STGCNBCDataset 兼容性: 实例化 + 取样测试

决策门:
    - 全部通过 → 数据集可用于训练
    - 任一 critical 失败 → 数据集需修复
    - warning 不阻塞但需评估影响

用法:
    python scripts/verify_dataset_quality.py \\
        --pkl data/stgcn_bc/train_real.pkl \\
        --pkl data/stgcn_bc/val_real.pkl

    python scripts/verify_dataset_quality.py \\
        --pkl data/stgcn_bc/train_real.pkl \\
        --check-leakage --val-pkl data/stgcn_bc/val_real.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.behavior.constants import NUM_KEYPOINTS
from backend.ml.behavior.stgcn_bc.labels import (
    BEHAVIOR_TO_IDX,
    IDX_TO_BEHAVIOR,
    NUM_BEHAVIORS,
    P0_IDX,
    P1_IDX,
    P2_IDX,
)


# ============================================================
# 验证器
# ============================================================


class DatasetQualityVerifier:
    """训练数据集质量验证器."""

    def __init__(self) -> None:
        self.critical_failures: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []

    def _record_pass(self, name: str, detail: str = "") -> None:
        msg = f"{name}" + (f": {detail}" if detail else "")
        self.passed.append(msg)

    def _record_critical(self, name: str, detail: str) -> None:
        self.critical_failures.append(f"{name}: {detail}")

    def _record_warning(self, name: str, detail: str) -> None:
        self.warnings.append(f"{name}: {detail}")

    # --------------------------------------------------------
    # 1. 格式合规性
    # --------------------------------------------------------

    def verify_format(self, samples: List[Dict], pkl_name: str) -> None:
        """验证 pyskl pickle 格式合规性."""
        name = f"[{pkl_name}] 格式合规性"

        if not samples:
            self._record_critical(name, "空数据集（0 样本）")
            return

        invalid_count = 0
        for i, s in enumerate(samples):
            # 必须字段：keypoint（pyskl 标准）或 keypoints（项目内部）
            if "keypoint" not in s and "keypoints" not in s:
                self._record_critical(name, f"样本 {i} 缺少 'keypoint'/'keypoints' 字段")
                invalid_count += 1
                continue
            if "label" not in s:
                self._record_critical(name, f"样本 {i} 缺少 'label' 字段")
                invalid_count += 1
                continue

            kpt = s.get("keypoint", s.get("keypoints"))
            # 形状检查: (M, T, V, C) 或 (T, V, C)
            if not isinstance(kpt, np.ndarray):
                self._record_critical(name, f"样本 {i} keypoint 非 ndarray: {type(kpt)}")
                invalid_count += 1
                continue

            if kpt.ndim == 4:
                M, T, V, C = kpt.shape
            elif kpt.ndim == 3:
                T, V, C = kpt.shape
                M = 1
            else:
                self._record_critical(name, f"样本 {i} keypoint 维度错误: {kpt.ndim}D")
                invalid_count += 1
                continue

            if V != NUM_KEYPOINTS:
                self._record_critical(
                    name, f"样本 {i} 关键点数 {V} ≠ {NUM_KEYPOINTS}"
                )
                invalid_count += 1
                continue

            if C not in (2, 3):
                self._record_critical(name, f"样本 {i} 通道数 {C} 非 2/3")
                invalid_count += 1
                continue

            if T < 1:
                self._record_critical(name, f"样本 {i} 帧数 {T} < 1")
                invalid_count += 1
                continue

            # dtype 检查
            if kpt.dtype not in (np.float32, np.float64):
                self._record_warning(name, f"样本 {i} dtype {kpt.dtype} 非 float32/64")

            # label 范围
            label = s["label"]
            if not isinstance(label, int) and not np.issubdtype(type(label), np.integer):
                self._record_critical(name, f"样本 {i} label 非整数: {type(label)}")
                invalid_count += 1
                continue
            label_int = int(label)
            if label_int < 0 or label_int >= NUM_BEHAVIORS:
                self._record_critical(
                    name, f"样本 {i} label {label_int} 越界 [0, {NUM_BEHAVIORS})"
                )
                invalid_count += 1
                continue

        if invalid_count == 0:
            self._record_pass(name, f"{len(samples)} 样本全部合规")
        else:
            self._record_warning(name, f"{invalid_count}/{len(samples)} 样本不合规")

    # --------------------------------------------------------
    # 2. 数值健康性
    # --------------------------------------------------------

    def verify_numerical_health(self, samples: List[Dict], pkl_name: str) -> None:
        """检测 NaN / Inf / 异常值."""
        name = f"[{pkl_name}] 数值健康性"

        total_nan = 0
        total_inf = 0
        total_outliers = 0
        total_points = 0

        for i, s in enumerate(samples):
            kpt = s.get("keypoint", s.get("keypoints"))
            if not isinstance(kpt, np.ndarray):
                continue

            # 归一化到 (T*V, C)
            if kpt.ndim == 4:
                flat = kpt.reshape(-1, kpt.shape[-1])
            elif kpt.ndim == 3:
                flat = kpt.reshape(-1, kpt.shape[-1])
            else:
                continue

            nan_count = int(np.isnan(flat).sum())
            inf_count = int(np.isinf(flat).sum())

            # 异常值: |x| > 1e4 或 |y| > 1e4（像素坐标或归一化坐标都不应超此范围）
            xy = flat[..., :2] if flat.shape[-1] >= 2 else flat
            outlier_mask = np.abs(xy) > 1e4
            outlier_count = int(outlier_mask.sum())

            total_nan += nan_count
            total_inf += inf_count
            total_outliers += outlier_count
            total_points += flat.shape[0] * flat.shape[-1]

        if total_nan > 0:
            self._record_critical(name, f"NaN 值 {total_nan} 个")
        if total_inf > 0:
            self._record_critical(name, f"Inf 值 {total_inf} 个")
        if total_outliers > 0:
            self._record_warning(name, f"异常值 |xy|>1e4 共 {total_outliers} 个")

        if total_nan == 0 and total_inf == 0 and total_outliers == 0:
            self._record_pass(name, f"{total_points} 数值点全部健康")

    # --------------------------------------------------------
    # 3. 标签分布
    # --------------------------------------------------------

    def verify_label_distribution(
        self, samples: List[Dict], pkl_name: str
    ) -> Dict[int, int]:
        """验证 22 类行为分布."""
        name = f"[{pkl_name}] 标签分布"

        label_counter: Counter = Counter()
        for s in samples:
            label = int(s.get("label", -1))
            if 0 <= label < NUM_BEHAVIORS:
                label_counter[label] += 1

        # 覆盖率
        covered = len(label_counter)
        if covered == 0:
            self._record_critical(name, "无有效标签")
            return {}

        coverage_pct = covered / NUM_BEHAVIORS * 100

        # 长尾类别（< 总样本数的 1%）
        total = sum(label_counter.values())
        long_tail = {
            IDX_TO_BEHAVIOR[l]: c
            for l, c in label_counter.items()
            if c < max(1, total * 0.01)
        }

        # 缺失类别
        missing = [IDX_TO_BEHAVIOR[i] for i in range(NUM_BEHAVIORS) if i not in label_counter]

        # P0/P1/P2 分层
        p0_count = sum(label_counter.get(i, 0) for i in P0_IDX)
        p1_count = sum(label_counter.get(i, 0) for i in P1_IDX)
        p2_count = sum(label_counter.get(i, 0) for i in P2_IDX)

        if coverage_pct < 100:
            self._record_warning(
                name, f"覆盖率 {covered}/{NUM_BEHAVIORS} ({coverage_pct:.1f}%), 缺失: {missing}"
            )
        else:
            self._record_pass(name, f"22 类全覆盖")

        if long_tail:
            self._record_warning(
                name, f"长尾类别 ({len(long_tail)}): {dict(list(long_tail.items())[:5])}"
            )

        # 严重不平衡: P0/P1/P2 任一为 0
        if p0_count == 0 or p1_count == 0 or p2_count == 0:
            self._record_critical(
                name, f"P0={p0_count} P1={p1_count} P2={p2_count} 存在空层级"
            )

        return dict(label_counter)

    # # --------------------------------------------------------
    # # 4. 边界标签
    # # --------------------------------------------------------

    def verify_boundary_labels(self, samples: List[Dict], pkl_name: str) -> None:
        """验证边界标签非空 + 时间对齐."""
        name = f"[{pkl_name}] 边界标签"

        no_boundary_count = 0
        misaligned_count = 0
        total_boundary_frames = 0

        for s in samples:
            boundary = s.get("boundary")
            kpt = s.get("keypoint", s.get("keypoints"))

            if boundary is None:
                no_boundary_count += 1
                continue

            if not isinstance(boundary, np.ndarray):
                no_boundary_count += 1
                continue

            # 时间对齐: boundary 长度应等于 keypoint 时间维度
            if isinstance(kpt, np.ndarray) and kpt.ndim >= 3:
                expected_T = kpt.shape[-3] if kpt.ndim == 4 else kpt.shape[0]
                if len(boundary) != expected_T:
                    misaligned_count += 1
                    continue

            total_boundary_frames += int((boundary > 0.5).sum())

        if no_boundary_count == len(samples):
            self._record_warning(name, "全部样本无 boundary 字段（BC 头将仅用分类损失）")
        elif no_boundary_count > 0:
            self._record_warning(
                name, f"{no_boundary_count}/{len(samples)} 样本无 boundary"
            )

        if misaligned_count > 0:
            self._record_critical(
                name, f"{misaligned_count} 样本 boundary 长度与 keypoint 时间维不匹配"
            )

        if no_boundary_count == 0 and misaligned_count == 0:
            self._record_pass(
                name, f"{len(samples)} 样本全有 boundary, 共 {total_boundary_frames} 边界帧"
            )

    # --------------------------------------------------------
    # 5. 训练/验证泄漏
    # --------------------------------------------------------

    def verify_leakage(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
    ) -> None:
        """检测 train/val 之间的 clip_id 泄漏."""
        name = "[泄漏检测]"

        train_clips = {s.get("frame_dir", "") for s in train_samples}
        val_clips = {s.get("frame_dir", "") for s in val_samples}

        # 移除空字符串（无 frame_dir 的样本）
        train_clips.discard("")
        val_clips.discard("")

        if not train_clips or not val_clips:
            self._record_warning(name, "无法检测泄漏（部分样本无 frame_dir）")
            return

        leaked = train_clips & val_clips
        if leaked:
            self._record_critical(
                name, f"发现 {len(leaked)} 个 clip_id 同时在 train 和 val: {list(leaked)[:5]}"
            )
        else:
            self._record_pass(
                name, f"无泄漏 (train={len(train_clips)} clips, val={len(val_clips)} clips)"
            )

    # --------------------------------------------------------
    # 6. STGCNBCDataset 兼容性
    # --------------------------------------------------------

    def verify_dataset_compatibility(
        self, samples: List[Dict], pkl_name: str
    ) -> None:
        """验证能否被 STGCNBCDataset 加载."""
        name = f"[{pkl_name}] STGCNBCDataset 兼容性"

        try:
            from backend.ml.behavior.stgcn_bc.dataset import STGCNBCDataset

            ds = STGCNBCDataset(samples=samples, T=30, augment=False, normalize=True)
            if len(ds) == 0:
                self._record_critical(name, "STGCNBCDataset 实例化后长度为 0")
                return

            # 取样测试
            sample = ds[0]
            kpt = sample["keypoints"]
            label = sample["label"]

            if not isinstance(kpt, torch.Tensor if (torch := _try_import_torch()) else type(None)):
                self._record_warning(name, f"sample keypoints 类型 {type(kpt)} 非 Tensor")
            if kpt.shape[-2] != NUM_KEYPOINTS:
                self._record_critical(name, f"取样关键点 V={kpt.shape[-2]} ≠ {NUM_KEYPOINTS}")

            self._record_pass(name, f"加载成功 ({len(ds)} 样本, 取样 shape={kpt.shape})")
        except Exception as e:
            self._record_critical(name, f"STGCNBCDataset 加载失败: {type(e).__name__}: {e}")

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------

    def summarize(self) -> Dict:
        return {
            "critical_count": len(self.critical_failures),
            "warning_count": len(self.warnings),
            "passed_count": len(self.passed),
            "critical": self.critical_failures,
            "warnings": self.warnings,
            "passed": self.passed,
            "verdict": "PASS" if not self.critical_failures else "FAIL",
        }


def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


# ============================================================
# 主流程
# ============================================================


def load_pkl(pkl_path: Path) -> List[Dict]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{pkl_path} 内容非 List[Dict]: {type(data)}")
    return data


def verify_single_pkl(
    pkl_path: Path,
    verifier: DatasetQualityVerifier,
) -> Dict[int, int]:
    """验证单个 pickle 文件."""
    pkl_name = pkl_path.name
    print(f"\n{'=' * 70}")
    print(f"验证: {pkl_path}")
    print(f"{'=' * 70}")

    try:
        samples = load_pkl(pkl_path)
    except Exception as e:
        verifier._record_critical(f"[{pkl_name}] 加载", f"{type(e).__name__}: {e}")
        return {}

    print(f"  样本数: {len(samples)}")

    verifier.verify_format(samples, pkl_name)
    verifier.verify_numerical_health(samples, pkl_name)
    label_dist = verifier.verify_label_distribution(samples, pkl_name)
    verifier.verify_boundary_labels(samples, pkl_name)
    verifier.verify_dataset_compatibility(samples, pkl_name)

    return label_dist


def main() -> None:
    parser = argparse.ArgumentParser(description="训练数据集质量验证")
    parser.add_argument(
        "--pkl", nargs="+", required=True,
        help="待验证的 pyskl pickle 路径（可多个）",
    )
    parser.add_argument(
        "--val-pkl", type=str, default=None,
        help="验证集 pickle 路径（用于泄漏检测）",
    )
    parser.add_argument(
        "--check-leakage", action="store_true",
        help="执行 train/val clip_id 泄漏检测",
    )
    parser.add_argument(
        "--report", type=str, default="reports/dataset_quality.json",
        help="JSON 报告输出路径",
    )
    args = parser.parse_args()

    verifier = DatasetQualityVerifier()

    train_label_dist: Dict[int, int] = {}
    val_label_dist: Dict[int, int] = {}

    for pkl_str in args.pkl:
        pkl_path = PROJECT_ROOT / pkl_str
        if not pkl_path.exists():
            verifier._record_critical(f"[{pkl_str}]", "文件不存在")
            continue
        dist = verify_single_pkl(pkl_path, verifier)
        if "train" in pkl_str.lower():
            train_label_dist = dist
        elif "val" in pkl_str.lower():
            val_label_dist = dist

    # 泄漏检测
    if args.check_leakage and args.val_pkl:
        val_path = PROJECT_ROOT / args.val_pkl
        train_path = next(
            (Path(p) for p in args.pkl if "train" in Path(p).name.lower()),
            None,
        )
        if train_path and val_path.exists():
            try:
                train_samples = load_pkl(PROJECT_ROOT / train_path)
                val_samples = load_pkl(val_path)
                verifier.verify_leakage(train_samples, val_samples)
            except Exception as e:
                verifier._record_critical(
                    "[泄漏检测]", f"加载失败: {type(e).__name__}: {e}"
                )

    # 汇总
    summary = verifier.summarize()

    print(f"\n{'=' * 70}")
    print(f"验证结果: {summary['verdict']}")
    print(f"  通过: {summary['passed_count']}")
    print(f"  警告: {summary['warning_count']}")
    print(f"  严重: {summary['critical_count']}")
    print(f"{'=' * 70}")

    if summary["critical"]:
        print("\n严重问题:")
        for c in summary["critical"]:
            print(f"  ❌ {c}")

    if summary["warnings"]:
        print("\n警告:")
        for w in summary["warnings"]:
            print(f"  ⚠️  {w}")

    if summary["passed"]:
        print("\n通过项:")
        for p in summary["passed"]:
            print(f"  ✅ {p}")

    # 保存报告
    report_path = PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n📄 报告已保存: {report_path}")

    sys.exit(0 if summary["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
