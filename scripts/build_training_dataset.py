"""训练数据集构建管线（Phase 3 真实数据标注辅线）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1d 辅线
依据: dev-docs/stages/phase-3.md §3.1d 数据源优先级

功能:
    合并多个数据源 → 统一 pyskl pickle 格式 → train/val 无泄漏划分。

数据源优先级:
    1. Label Studio 人工标注（主路径，最准确）
    2. YouTube 自标（辅路径，需 YOLO26-pose 预标注 + 人工修正）
    3. dog-pose val（同域数据，仅用于评估，不进训练集）
    4. InterPet4D kp_world（3D 关键点，无行为标签，仅用于无监督预训练）
    5. 合成数据（baseline，已验证 46.97% 准确率）

划分策略:
    - 按 clip_id（视频 ID）划分，避免同一视频的帧同时出现在 train 和 val
    - 默认 80/20 划分
    - 支持分层抽样（保持 22 类比例）

用法:
    python scripts/build_training_dataset.py \\
        --sources data/stgcn_bc/labelstudio.pkl data/stgcn_bc/youtube.pkl \\
        --output-dir data/stgcn_bc/ \\
        --split 0.8 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.behavior.stgcn_bc.labels import (
    BEHAVIOR_TO_IDX,
    IDX_TO_BEHAVIOR,
    NUM_BEHAVIORS,
)


def load_pyskl_sources(source_paths: List[str]) -> List[Dict]:
    """加载多个 pyskl pickle 数据源并合并."""
    all_samples: List[Dict] = []
    source_stats = {}

    for path in source_paths:
        p = Path(path)
        if not p.exists():
            print(f"⚠️  跳过不存在的数据源: {path}")
            continue
        with open(p, "rb") as f:
            samples = pickle.load(f)
        for s in samples:
            s["_source"] = p.name  # 标记来源
        all_samples.extend(samples)
        source_stats[p.name] = len(samples)
        print(f"  加载 {p.name}: {len(samples)} 样本")

    return all_samples, source_stats


def stratified_split_by_clip(
    samples: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """按 clip_id 分层划分 train/val（无泄漏）.

    策略:
        1. 按 clip_id 分组样本
        2. 按 label 分层
        3. 每个 label 内按 clip_id 划分（保证同一 clip 不跨集）

    Args:
        samples: 样本列表
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        (train_samples, val_samples)
    """
    rng = random.Random(seed)

    # 按 label 分组 clip
    label_to_clips: Dict[int, set] = defaultdict(set)
    clip_to_samples: Dict[str, List[Dict]] = defaultdict(list)

    for s in samples:
        clip_id = s.get("frame_dir", str(s.get("label", 0)))
        label = s.get("label", 0)
        label_to_clips[label].add(clip_id)
        clip_to_samples[clip_id].append(s)

    train_samples: List[Dict] = []
    val_samples: List[Dict] = []

    for label in sorted(label_to_clips.keys()):
        clips = sorted(label_to_clips[label])
        rng.shuffle(clips)

        n_train = max(1, int(len(clips) * train_ratio))
        train_clips = set(clips[:n_train])
        val_clips = set(clips[n_train:])

        for clip_id in train_clips:
            train_samples.extend(clip_to_samples[clip_id])
        for clip_id in val_clips:
            val_samples.extend(clip_to_samples[clip_id])

    return train_samples, val_samples


def compute_statistics(samples: List[Dict]) -> Dict:
    """计算数据集统计信息."""
    label_counts = defaultdict(int)
    source_counts = defaultdict(int)
    total_frames = 0
    frame_lengths = []

    for s in samples:
        label = s.get("label", -1)
        label_name = IDX_TO_BEHAVIOR.get(label, f"unknown_{label}")
        label_counts[label_name] += 1

        source = s.get("_source", "unknown")
        source_counts[source] += 1

        kpt = s.get("keypoint")
        if isinstance(kpt, np.ndarray) and kpt.ndim == 4:
            T = kpt.shape[1]
            total_frames += T
            frame_lengths.append(T)

    return {
        "total_samples": len(samples),
        "total_frames": total_frames,
        "avg_frames_per_sample": round(np.mean(frame_lengths), 2) if frame_lengths else 0,
        "min_frames": min(frame_lengths) if frame_lengths else 0,
        "max_frames": max(frame_lengths) if frame_lengths else 0,
        "label_distribution": dict(sorted(label_counts.items())),
        "source_distribution": dict(sorted(source_counts.items())),
        "unique_labels": len(label_counts),
        "expected_labels": NUM_BEHAVIORS,
    }


def build_dataset(
    source_paths: List[str],
    output_dir: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict:
    """构建训练数据集.

    Args:
        source_paths: 数据源 pyskl pickle 路径列表
        output_dir: 输出目录
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        构建统计
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("训练数据集构建管线")
    print("=" * 70)

    # 1. 加载所有数据源
    print("\n[1/4] 加载数据源...")
    all_samples, source_stats = load_pyskl_sources(source_paths)
    print(f"  合计: {len(all_samples)} 样本")

    if not all_samples:
        return {"success": False, "error": "无有效样本"}

    # 2. 统计原始数据
    print("\n[2/4] 原始数据统计...")
    orig_stats = compute_statistics(all_samples)
    print(f"  总帧数: {orig_stats['total_frames']}")
    print(f"  平均帧/样本: {orig_stats['avg_frames_per_sample']}")
    print(f"  行为类别数: {orig_stats['unique_labels']}/{orig_stats['expected_labels']}")
    print(f"  来源分布: {orig_stats['source_distribution']}")

    # 3. 划分 train/val
    print(f"\n[3/4] 划分 train/val (ratio={train_ratio}, seed={seed})...")
    train_samples, val_samples = stratified_split_by_clip(
        all_samples, train_ratio=train_ratio, seed=seed
    )
    print(f"  train: {len(train_samples)} 样本")
    print(f"  val: {len(val_samples)} 样本")

    # 清理 _source 标记（不写入 pickle）
    for s in train_samples + val_samples:
        s.pop("_source", None)

    # 4. 保存
    print("\n[4/4] 保存数据集...")
    train_path = output_dir / "train_real.pkl"
    val_path = output_dir / "val_real.pkl"

    with open(train_path, "wb") as f:
        pickle.dump(train_samples, f)
    with open(val_path, "wb") as f:
        pickle.dump(val_samples, f)

    print(f"  train: {train_path}")
    print(f"  val: {val_path}")

    # 统计
    train_stats = compute_statistics(train_samples)
    val_stats = compute_statistics(val_samples)

    summary = {
        "success": True,
        "source_stats": source_stats,
        "original": orig_stats,
        "train": train_stats,
        "val": val_stats,
        "split_config": {
            "train_ratio": train_ratio,
            "seed": seed,
            "strategy": "stratified_by_clip_no_leak",
        },
        "output_paths": {
            "train": str(train_path),
            "val": str(val_path),
        },
    }

    print("\n" + "=" * 70)
    print("✅ 数据集构建完成")
    print(f"  train: {len(train_samples)} 样本 / {train_stats['total_frames']} 帧")
    print(f"  val: {len(val_samples)} 样本 / {val_stats['total_frames']} 帧")
    print("=" * 70)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="训练数据集构建管线")
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="数据源 pyskl pickle 路径列表",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--split", type=float, default=0.8,
        help="训练集比例（默认 0.8）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）",
    )
    parser.add_argument(
        "--report", type=str, default="reports/dataset_build.json",
        help="JSON 报告输出路径",
    )
    args = parser.parse_args()

    summary = build_dataset(
        source_paths=args.sources,
        output_dir=args.output_dir,
        train_ratio=args.split,
        seed=args.seed,
    )

    report_path = PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n📄 报告已保存: {report_path}")

    sys.exit(0 if summary.get("success") else 1)


if __name__ == "__main__":
    main()
