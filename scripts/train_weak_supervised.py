"""Path 4: 弱监督自训练 — 规则引擎伪标签 + 轻量分类器.

Phase 2 数据策略 Path 4: 用 3D 姿态分类 + 运动学特征生成伪标签，训练 Random Forest 分类器。

策略:
    1. 加载 InterPet4D kp_world 序列（226 序列, (T, 24, 3)）
    2. 逐帧 3D 姿态分类（sit/down/stand/unknown）→ P0 伪标签
    3. 运动学特征启发式 → P1 伪标签（track/obstacle/watch/apprehend/recall）
    4. 滑动窗口特征提取（30 帧 = 1s @ 30fps）
    5. Random Forest 训练 + 5 折交叉验证
    6. 输出模型 + 评估报告

优势:
    - 不需要人工标注
    - 不需要 keypoint-MoSeq 训练完成
    - 直接利用现有数据生成分类器
    - 可作为 Path 1 的补充验证

依据:
    - dev-docs/decisions/0009-data-strategy-four-paths.md Path 4
    - scripts/map_syllables_to_behaviors.py 3D 姿态分类逻辑

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 2.0d（数据策略 Path 4）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.behavior.constants import (
    ALL_KNEES, ALL_PAWS, CHIN, DOWN,
    FRONT_ELBOWS, FRONT_KNEES, FRONT_PAWS,
    NOSE, REAR_ELBOWS, REAR_KNEES, REAR_PAWS,
    SIT, STAND, WITHERS,
    TRACK, APPREHEND, OBSTACLE, RECALL, WATCH,
    ALL_BEHAVIORS,
)

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "interpet4d" / "smal_npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "weak_supervised"

# 3D 世界坐标阈值（米，y-up）— 与 map_syllables_to_behaviors.py 一致
CONF_THRESHOLD = 0.3
SIT_REAR_FOLD_MAX = 0.05
SIT_FRONT_EXTEND_MIN = 0.10
DOWN_WITHERS_HEIGHT_MAX = 0.40
DOWN_ALL_FOLD_MAX = 0.08
STAND_WITHERS_HEIGHT_MIN = 0.50
STAND_LEG_EXTEND_MIN = 0.15

# P1 运动学阈值（米/帧）
TRACK_NOSE_GROUND_RATIO = 0.85  # 鼻尖高度 < 地面 * 0.85
TRACK_MOTION_MIN = 0.02         # 持续移动
OBSTACLE_Y_RISE_MIN = 0.10      # 跳跃高度
OBSTACLE_MOTION_MIN = 0.10      # 高速
WATCH_STABILITY_MAX = 0.02      # 稳定
APPREHEND_SPEED_MIN = 0.15      # 高速接近
APPREHEND_MOUTH_VAR_MIN = 0.001 # 嘴部活跃
RECALL_SPEED_MIN = 0.12         # 返回速度

WINDOW_SIZE = 30  # 30 帧 ≈ 1s @ 30fps
STRIDE = 15       # 50% 重叠


def classify_posture_3d(frame: np.ndarray) -> str:
    """3D 世界坐标单帧姿态分类（y-up, 米）。返回 sit/down/stand/unknown。"""
    mask = frame[:, 2] >= CONF_THRESHOLD
    if not mask.any():
        return "unknown"

    valid = frame[mask]
    # y 轴向上，地面 ≈ min y
    ground_y = float(valid[:, 1].min())

    def mean_y(indices: tuple) -> float | None:
        v = [i for i in indices if mask[i]]
        if not v:
            return None
        return float(np.mean(frame[v, 1]))

    def mean_abs_diff_y(a: tuple, b: tuple) -> float | None:
        pairs = [(i, j) for i, j in zip(a, b) if mask[i] and mask[j]]
        if not pairs:
            return None
        return float(np.mean([abs(frame[i, 1] - frame[j, 1]) for i, j in pairs]))

    withers_y = mean_y((WITHERS,))
    rear_fold = mean_abs_diff_y(REAR_PAWS, REAR_KNEES)
    front_extend = mean_abs_diff_y(FRONT_PAWS, FRONT_ELBOWS)
    all_fold = mean_abs_diff_y(ALL_PAWS, ALL_KNEES)

    if None in (withers_y, rear_fold, front_extend, all_fold):
        return "unknown"

    # 卧：肩甲低 + 四肢折叠
    if withers_y < ground_y + DOWN_WITHERS_HEIGHT_MAX and all_fold < DOWN_ALL_FOLD_MAX:
        return DOWN

    # 坐：后腿折叠 + 前腿伸直
    if rear_fold < SIT_REAR_FOLD_MAX and front_extend > SIT_FRONT_EXTEND_MIN:
        return SIT

    # 立：肩甲高 + 腿伸直
    if withers_y > ground_y + STAND_WITHERS_HEIGHT_MIN and front_extend > STAND_LEG_EXTEND_MIN:
        return STAND

    return "unknown"


def classify_p1_motion(kpts: np.ndarray, start: int, end: int) -> str | None:
    """运动学特征启发式 P1 行为分类。

    Args:
        kpts: (T, 24, 3) 全序列
        start, end: 窗口起止帧

    Returns:
        track/obstacle/watch/apprehend/recall 或 None
    """
    window = kpts[start:end]
    if len(window) < 5:
        return None

    # withers 帧间位移
    withers = window[:, WITHERS, :2]  # (T, 2) xz 平面
    motion = np.diff(withers, axis=0)
    speed = np.linalg.norm(motion, axis=1)
    avg_speed = float(np.mean(speed))

    # withers 高度变化（y 轴）
    withers_y = window[:, WITHERS, 1]
    y_rise = float(np.max(withers_y) - np.min(withers_y))

    # 鼻尖高度
    nose_y = window[:, NOSE, 1]
    ground_y = float(np.min(window[:, :, 1]))
    nose_ground_ratio = float(np.mean(nose_y) / (ground_y + 1e-6))

    # 嘴部活跃度（nose-chin 距离方差）
    nose_chin = np.linalg.norm(
        window[:, NOSE, :2] - window[:, CHIN, :2], axis=1
    )
    mouth_var = float(np.var(nose_chin))

    # 稳定性
    stability = float(np.mean(speed))

    # 方向反转检测
    if len(motion) > 4:
        directions = np.sign(motion[:, 0])  # x 方向
        reversals = np.sum(np.abs(np.diff(directions)) > 0)
    else:
        reversals = 0

    # P1 启发式（优先级从高到低）
    if avg_speed > APPREHEND_SPEED_MIN and mouth_var > APPREHEND_MOUTH_VAR_MIN:
        return APPREHEND
    if y_rise > OBSTACLE_Y_RISE_MIN and avg_speed > OBSTACLE_MOTION_MIN:
        return OBSTACLE
    if nose_ground_ratio < TRACK_NOSE_GROUND_RATIO and avg_speed > TRACK_MOTION_MIN:
        return TRACK
    if reversals > 2 and avg_speed > RECALL_SPEED_MIN:
        return RECALL
    if stability < WATCH_STABILITY_MAX:
        return WATCH

    return None


def extract_window_features(kpts: np.ndarray, start: int, end: int) -> np.ndarray:
    """提取窗口特征向量。

    特征（共 ~80 维）:
    - 姿态分布: sit/down/stand/unknown 比例 (4)
    - 关键点位置统计: 24 点 x/y/z 的 mean/std (72)
    - 运动特征: withers/nose 速度 mean/std (4)
    """
    window = kpts[start:end]
    T = len(window)

    # 1. 姿态分布
    posture_counts = Counter(classify_posture_3d(window[t]) for t in range(T))
    posture_feat = np.array([
        posture_counts.get("sit", 0) / T,
        posture_counts.get("down", 0) / T,
        posture_counts.get("stand", 0) / T,
        posture_counts.get("unknown", 0) / T,
    ])

    # 2. 关键点位置统计（归一化到窗口第一帧的质心）
    valid_mask = window[:, :, 2] >= CONF_THRESHOLD
    centered = window[:, :, :2].copy()
    for t in range(T):
        m = valid_mask[t]
        if m.any():
            centroid = window[t, m, :2].mean(axis=0)
            centered[t, m] = window[t, m, :2] - centroid

    pos_mean = centered.mean(axis=0).flatten()  # (48,)
    pos_std = centered.std(axis=0).flatten()    # (48,)

    # 3. 运动特征
    withers = window[:, WITHERS, :2]
    nose = window[:, NOSE, :2]
    withers_speed = np.linalg.norm(np.diff(withers, axis=0), axis=1)
    nose_speed = np.linalg.norm(np.diff(nose, axis=0), axis=1)
    motion_feat = np.array([
        float(withers_speed.mean()) if len(withers_speed) > 0 else 0.0,
        float(withers_speed.std()) if len(withers_speed) > 0 else 0.0,
        float(nose_speed.mean()) if len(nose_speed) > 0 else 0.0,
        float(nose_speed.std()) if len(nose_speed) > 0 else 0.0,
    ])

    return np.concatenate([posture_feat, pos_mean, pos_std, motion_feat])


def generate_pseudo_labels(kpts: np.ndarray) -> list[tuple[int, int, str]]:
    """为序列生成伪标签窗口。

    Returns:
        [(start, end, label), ...]
    """
    T = kpts.shape[0]
    windows = []

    for start in range(0, T - WINDOW_SIZE, STRIDE):
        end = start + WINDOW_SIZE
        window = kpts[start:end]

        # P1 检测优先（运动学特征更显著）
        p1_label = classify_p1_motion(kpts, start, end)
        if p1_label is not None:
            windows.append((start, end, p1_label))
            continue

        # P0 姿态分类（多数投票）
        postures = [classify_posture_3d(window[t]) for t in range(len(window))]
        counts = Counter(postures)
        top_posture, top_count = counts.most_common(1)[0]
        if top_posture != "unknown" and top_count / len(window) > 0.6:
            windows.append((start, end, top_posture))
        else:
            windows.append((start, end, "unknown"))

    return windows


def load_sequences(data_dir: Path) -> list[tuple[str, np.ndarray]]:
    """加载所有 npz 序列。"""
    files = sorted(data_dir.glob("*.npz"))
    sequences = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        kp = data.get("kp_world", data.get("keypoints"))
        if kp is None:
            continue
        kp = np.asarray(kp, dtype=np.float32)
        if kp.ndim == 3 and kp.shape[1] == 24 and kp.shape[2] >= 3:
            kp = kp[:, :, :3]  # 取 x, y, conf
            sequences.append((f.stem, kp))
    return sequences


def main() -> int:
    parser = argparse.ArgumentParser(description="Path 4: 弱监督自训练")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    args = parser.parse_args()

    print("=" * 60)
    print("Path 4: 弱监督自训练 — 规则引擎伪标签 + Random Forest")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载 InterPet4D 序列...")
    sequences = load_sequences(args.data_dir)
    print(f"  加载 {len(sequences)} 个序列")
    if not sequences:
        print("[ERROR] 无序列可加载")
        return 1

    # 2. 生成伪标签 + 特征提取
    print("\n[2/5] 生成伪标签 + 特征提取...")
    t0 = time.time()
    all_features = []
    all_labels = []
    label_counts = Counter()

    for name, kpts in sequences:
        windows = generate_pseudo_labels(kpts)
        for start, end, label in windows:
            feat = extract_window_features(kpts, start, end)
            all_features.append(feat)
            all_labels.append(label)
            label_counts[label] += 1

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)
    print(f"  窗口数: {len(y)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  标签分布: {dict(label_counts.most_common())}")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # 过滤 unknown 标签（只训练有标签的窗口）
    labeled_mask = y != "unknown"
    X_labeled = X[labeled_mask]
    y_labeled = y[labeled_mask]
    print(f"  有标签窗口: {len(y_labeled)}/{len(y)} ({len(y_labeled)/len(y)*100:.1f}%)")

    if len(y_labeled) < 50:
        print("[WARN] 有标签窗口太少，无法训练分类器")
        return 1

    # 3. 训练 Random Forest
    print("\n[3/5] 训练 Random Forest 分类器...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    # 5 折交叉验证
    min_class_size = min(Counter(y_labeled).values())
    n_splits = min(5, min_class_size)
    if n_splits < 2:
        print(f"[WARN] 最小类别样本数 {min_class_size}，无法交叉验证")
        n_splits = 2

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_labeled, y_labeled, cv=cv, scoring="accuracy")
    print(f"  {n_splits} 折交叉验证准确率: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  各折: {[f'{s:.3f}' for s in scores]}")

    # 全量训练
    rf.fit(X_labeled, y_labeled)
    train_acc = rf.score(X_labeled, y_labeled)
    print(f"  训练集准确率: {train_acc:.3f}")

    # 4. 特征重要性
    print("\n[4/5] 特征重要性分析...")
    importances = rf.feature_importances_
    top_k = 10
    top_idx = np.argsort(importances)[::-1][:top_k]
    print(f"  Top {top_k} 重要特征索引: {top_idx.tolist()}")
    print(f"  Top {top_k} 重要性: {[f'{importances[i]:.4f}' for i in top_idx]}")

    # 5. 保存模型 + 报告
    print("\n[5/5] 保存模型 + 报告...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import pickle
    model_path = args.output_dir / "weak_supervised_rf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": rf,
            "feature_dim": X.shape[1],
            "labels": list(set(y_labeled)),
            "label_counts": dict(label_counts),
            "cv_scores": scores.tolist(),
            "train_acc": train_acc,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
        }, f)
    print(f"  模型: {model_path}")

    report = {
        "path": "Path 4: 弱监督自训练",
        "data_source": "InterPet4D smal_npy",
        "num_sequences": len(sequences),
        "num_windows": len(y),
        "num_labeled_windows": len(y_labeled),
        "feature_dim": X.shape[1],
        "label_distribution": dict(label_counts.most_common()),
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "train_accuracy": float(train_acc),
        "top_features": top_idx.tolist(),
        "model_path": str(model_path),
    }
    report_path = args.output_dir / "weak_supervised_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  报告: {report_path}")

    print("\n" + "=" * 60)
    print(f"Path 4 完成！交叉验证准确率: {scores.mean():.1%} ± {scores.std():.1%}")
    print(f"模型已保存: {model_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
