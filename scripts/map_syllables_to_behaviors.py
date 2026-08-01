"""keypoint-MoSeq syllable → 16 行为映射脚本.

Phase 2 数据策略 Path 1: 将 keypoint-MoSeq 发现的 syllables 映射到 16 行为类。

策略（用户确认：规则引擎自动映射 P0 + 人工映射 P1）:
    - P0 8 类（sit/down/stand/heel/sit_up/stay/bark/bite）:
        几何规则自动映射，每个 syllable 采样帧 → 姿态分类 → 多数投票
    - P1 8 类（track/alert_sit/alert_down/apprehend/escort/obstacle/recall/watch）:
        计算运动学特征 + 生成可视化数据，供人工映射

输入:
    - keypoint-MoSeq 项目目录（含 results.h5）
    - InterPet4D smal_npy/*.npz（原始关键点，用于帧采样）

输出:
    - data/kpm_project/syllable_mapping.json: syllable_id → behavior
    - data/kpm_project/syllable_features.json: 每个 syllable 的运动学特征
    - data/kpm_project/syllable_report.md: 可读报告（P0 自动 + P1 待人工）

依据:
    - dev-docs/research/PHASE2_DATA_STRATEGY.md Path 1
    - 用户决策: 规则引擎自动映射 P0 + 人工映射 P1（推荐）

坐标系统注意:
    InterPet4D kp_world 是 3D 世界坐标（y-up, 米）。
    本脚本直接用 3D 几何规则分类姿态，不复用 2D 图像坐标的 RuleEngine。

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 2.0d（数据策略 Path 1）
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.behavior.constants import (
    ALL_KNEES, ALL_PAWS, BARK, BITE, CHIN, DOWN,
    FRONT_ELBOWS, FRONT_KNEES, FRONT_PAWS,
    HEEL, NOSE, P0_BEHAVIORS, REAR_ELBOWS, REAR_KNEES, REAR_PAWS,
    SIT, SIT_UP, STAND, STAY, WITHERS,
    TRACK, ALERT_SIT, ALERT_DOWN, APPREHEND, ESCORT, OBSTACLE, RECALL, WATCH,
    P1_BEHAVIORS, ALL_BEHAVIORS, BEHAVIOR_NAMES_CN,
)

DEFAULT_KPM_DIR = PROJECT_ROOT / "data" / "kpm_project"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "interpet4d" / "smal_npy"

# 3D 世界坐标阈值（米，y-up）
# 注意：与 rule_engine.py 的 2D 像素阈值不同
CONF_THRESHOLD = 0.3
SIT_REAR_FOLD_MAX = 0.05      # 坐：后腿折叠（paw.y - knee.y 绝对差 < 5cm）
SIT_FRONT_EXTEND_MIN = 0.10   # 坐：前腿伸直（paw 低于 elbow ≥ 10cm，即 paw.y - elbow.y < -0.10）
DOWN_WITHERS_HEIGHT_MAX = 0.40  # 卧：肩甲高度 < 40cm（地面参考 0）
DOWN_ALL_FOLD_MAX = 0.08      # 卧：四肢折叠 < 8cm
STAND_WITHERS_HEIGHT_MIN = 0.50  # 立：肩甲高度 > 50cm
STAND_LEG_EXTEND_MIN = 0.15   # 立：腿伸直 > 15cm

# P0 映射置信度阈值
P0_CONFIDENCE_THRESHOLD = 0.70  # 70% 帧分类一致才映射


def classify_posture_3d(frame: np.ndarray) -> str:
    """3D 世界坐标单帧姿态分类（y-up, 米）.

    Args:
        frame: (24, 3) [x, y, conf]，y 轴向上

    Returns:
        "sit" / "down" / "stand" / "unknown"
    """
    mask = frame[:, 2] >= CONF_THRESHOLD
    if not mask.any():
        return "unknown"

    # 取地面 y（最低有效点的 y，y-up 坐标系下最小 y 接近地面）
    valid_y = frame[mask, 1]
    ground_y = float(valid_y.min())  # y-up: 地面 y 最小

    # 关键点 y 值
    def _mean_y_valid(indices: tuple) -> float | None:
        valid = [i for i in indices if mask[i]]
        if not valid:
            return None
        return float(np.mean(frame[valid, 1]))

    withers_y = _mean_y_valid((WITHERS,))
    rear_paw_y = _mean_y_valid(REAR_PAWS)
    rear_knee_y = _mean_y_valid(REAR_KNEES)
    front_paw_y = _mean_y_valid(FRONT_PAWS)
    front_elbow_y = _mean_y_valid(FRONT_ELBOWS)
    all_paw_y = _mean_y_valid(ALL_PAWS)
    all_knee_y = _mean_y_valid(ALL_KNEES)

    if None in (withers_y, rear_paw_y, rear_knee_y, front_paw_y, front_elbow_y,
                all_paw_y, all_knee_y):
        return "unknown"

    # y-up 坐标系:
    # - "高" = y 大
    # - "低" = y 小
    # - "paw 在地面" = paw.y 接近 ground_y
    # - "腿伸直" = paw.y 显著低于 elbow.y（paw 在 elbow 下方）
    # - "腿折叠" = paw.y 接近 knee.y

    withers_height = withers_y - ground_y  # 肩甲离地高度
    rear_fold = abs(rear_paw_y - rear_knee_y)  # 后腿折叠度
    front_extend = front_elbow_y - front_paw_y  # 前腿伸直度（elbow 高于 paw）
    all_fold = abs(all_paw_y - all_knee_y)  # 四肢折叠度

    # 卧：肩甲低 + 四肢折叠
    if withers_height < DOWN_WITHERS_HEIGHT_MAX and all_fold < DOWN_ALL_FOLD_MAX:
        return DOWN

    # 坐：后腿折叠 + 前腿伸直
    if rear_fold < SIT_REAR_FOLD_MAX and front_extend > SIT_FRONT_EXTEND_MIN:
        return SIT

    # 立：肩甲高 + 四腿伸直
    if (withers_height > STAND_WITHERS_HEIGHT_MIN and
            front_extend > STAND_LEG_EXTEND_MIN):
        rear_extend = rear_knee_y - rear_paw_y  # 后腿伸直（knee 高于 paw）
        if rear_extend > STAND_LEG_EXTEND_MIN * 0.5:  # 后腿至少半伸
            return STAND

    return "unknown"


def compute_syllable_motion_features(
    kpts: np.ndarray, syllable_labels: np.ndarray, syllable_id: int
) -> dict[str, float]:
    """计算单个 syllable 的运动学特征（用于 P1 人工映射辅助）.

    Args:
        kpts: (T, 24, 3) 关键点序列
        syllable_labels: (T,) syllable ID 序列
        syllable_id: 目标 syllable ID

    Returns:
        dict: 运动学特征
    """
    mask = syllable_labels == syllable_id
    if mask.sum() < 2:
        return {"frame_count": int(mask.sum())}

    clip = kpts[mask]
    T = clip.shape[0]

    # withers 运动
    withers_valid = clip[:, WITHERS, 2] >= CONF_THRESHOLD
    if withers_valid.sum() < 2:
        return {"frame_count": T}

    withers_xy = clip[withers_valid, WITHERS, :2]
    withers_motion = np.diff(withers_xy, axis=0)
    withers_speed = np.linalg.norm(withers_motion, axis=1)

    # nose-withers 相对位置（头部朝向）
    nose_valid = clip[:, NOSE, 2] >= CONF_THRESHOLD
    nose_above_withers = 0
    nose_below_withers = 0
    if nose_valid.any():
        both_valid = nose_valid & withers_valid
        if both_valid.any():
            nose_y = clip[both_valid, NOSE, 1]
            withers_y = clip[both_valid, WITHERS, 1]
            nose_above_withers = int((nose_y > withers_y).sum())  # y-up: nose.y > withers.y = 头抬起
            nose_below_withers = int((nose_y < withers_y).sum())

    # nose 接近地面比例（追踪特征）
    valid_y_all = clip[clip[:, :, 2] >= CONF_THRESHOLD, 1]
    if len(valid_y_all) > 0:
        ground_y = float(valid_y_all.min())
        nose_near_ground = 0
        if nose_valid.any():
            nose_y = clip[nose_valid, NOSE, 1]
            nose_near_ground = int((nose_y - ground_y < 0.10).sum())  # 鼻尖距地面 < 10cm
    else:
        nose_near_ground = 0

    # 嘴部活跃度（nose-chin 距离方差）
    chin_valid = clip[:, CHIN, 2] >= CONF_THRESHOLD
    both_mouth = nose_valid & chin_valid
    if both_mouth.sum() >= 3:
        nose_pts = clip[both_mouth, NOSE, :2]
        chin_pts = clip[both_mouth, CHIN, :2]
        mouth_dist = np.linalg.norm(nose_pts - chin_pts, axis=1)
        mouth_var = float(mouth_dist.var())
        mouth_std = float(mouth_dist.std())
    else:
        mouth_var = 0.0
        mouth_std = 0.0

    # x 方向速度（方向反转检测）
    withers_x = clip[withers_valid, WITHERS, 0]
    if len(withers_x) >= 4:
        x_vel = np.diff(withers_x)
        first_half_mean = float(x_vel[:len(x_vel)//2].mean()) if len(x_vel) >= 4 else 0.0
        second_half_mean = float(x_vel[len(x_vel)//2:].mean()) if len(x_vel) >= 4 else 0.0
        direction_reversed = (np.sign(first_half_mean) != np.sign(second_half_mean)
                              and abs(first_half_mean) > 0.01
                              and abs(second_half_mean) > 0.01)
    else:
        first_half_mean = 0.0
        second_half_mean = 0.0
        direction_reversed = False

    # y 方向跳跃检测（withers.y 先升后降）
    if len(withers_xy) >= 4:
        wy = withers_xy[:, 1]
        mid = len(wy) // 2
        y_rise = float(wy[mid:].mean() - wy[:mid].mean())  # 正值 = 先低后高 = 跳跃
    else:
        y_rise = 0.0

    return {
        "frame_count": T,
        "mean_speed": float(withers_speed.mean()) if len(withers_speed) > 0 else 0.0,
        "max_speed": float(withers_speed.max()) if len(withers_speed) > 0 else 0.0,
        "speed_std": float(withers_speed.std()) if len(withers_speed) > 0 else 0.0,
        "head_up_ratio": nose_above_withers / max(nose_above_withers + nose_below_withers, 1),
        "nose_near_ground_ratio": nose_near_ground / max(int(nose_valid.sum()), 1),
        "mouth_var": mouth_var,
        "mouth_std": mouth_std,
        "x_velocity_first": first_half_mean,
        "x_velocity_second": second_half_mean,
        "direction_reversed": bool(direction_reversed),
        "y_rise_jump": y_rise,
    }


def suggest_p1_behavior(features: dict) -> str | None:
    """基于运动学特征启发式建议 P1 行为（仅辅助人工决策）.

    Returns:
        建议行为名或 None（无明显特征）
    """
    if features.get("frame_count", 0) < 3:
        return None

    mean_speed = features.get("mean_speed", 0)
    nose_ground = features.get("nose_near_ground_ratio", 0)
    head_up = features.get("head_up_ratio", 0)
    mouth_var = features.get("mouth_var", 0)
    y_rise = features.get("y_rise_jump", 0)
    direction_reversed = features.get("direction_reversed", False)

    # 追踪：鼻尖贴地 + 持续移动
    if nose_ground > 0.7 and mean_speed > 0.05:
        return TRACK

    # 障碍穿越：跳跃轨迹（y 先升后降）+ 高速
    if y_rise > 0.10 and mean_speed > 0.15:
        return OBSTACLE

    # 扑咬：高速 + 嘴部活跃
    if mean_speed > 0.20 and mouth_var > 0.001:
        return APPREHEND

    # 返回：方向反转 + 高速
    if direction_reversed and mean_speed > 0.15:
        return RECALL

    # 警戒：头部抬起 + 低速（稳定）
    if head_up > 0.6 and mean_speed < 0.05:
        return WATCH

    return None


def load_kpm_results(project_dir: Path) -> dict[str, Any]:
    """加载 keypoint-MoSeq 训练结果.

    Returns:
        dict: {clip_name: {"syllable": np.ndarray(T,), ...}}
    """
    results_path = project_dir / "results.h5"
    if not results_path.exists():
        print(f"[ERROR] 结果文件不存在: {results_path}", file=sys.stderr)
        print("请先运行: python scripts/train_keypoint_moseq.py", file=sys.stderr)
        sys.exit(1)

    try:
        import h5py
    except ImportError:
        print("[ERROR] h5py 未安装", file=sys.stderr)
        sys.exit(1)

    results: dict[str, Any] = {}
    with h5py.File(str(results_path), "r") as f:
        for clip_name in f.keys():
            grp = f[clip_name]
            syllable = grp["syllable"][:] if "syllable" in grp else None
            if syllable is not None:
                results[clip_name] = {"syllable": syllable}

    print(f"[load] 加载 {len(results)} 个 clip 的 syllable 标注")
    return results


def load_interpet4d_keypoints(data_dir: Path) -> dict[str, np.ndarray]:
    """加载 InterPet4D 关键点序列.

    Returns:
        dict: {clip_name: (T, 24, 3) keypoints}
    """
    npz_files = sorted(data_dir.glob("*.npz"))
    print(f"[load] 发现 {len(npz_files)} 个 .npz 文件")

    data: dict[str, np.ndarray] = {}
    for npz_path in npz_files:
        try:
            npz = np.load(str(npz_path), allow_pickle=True)
            if "kp_world" not in npz:
                continue
            kp = np.asarray(npz["kp_world"], dtype=np.float32)[:, :, :3]
            if kp.ndim != 3 or kp.shape[1] != 24:
                continue
            # 置信度
            if "kp_weight" in npz:
                conf = np.asarray(npz["kp_weight"], dtype=np.float32)
                # 构造 (T, 24, 3) [x, y, conf]
                kp_out = np.zeros_like(kp)
                kp_out[:, :, :2] = kp[:, :, :2]
                kp_out[:, :, 2] = np.clip(conf, 0.0, 1.0)
            else:
                kp_out = np.zeros_like(kp)
                kp_out[:, :, :2] = kp[:, :, :2]
                kp_out[:, :, 2] = 1.0
            data[npz_path.stem] = kp_out
        except Exception as e:
            print(f"  [WARN] 加载失败 {npz_path.name}: {e}")
            continue

    print(f"[load] 成功加载 {len(data)} 个序列")
    return data


def map_syllables_to_behaviors(
    kpm_results: dict[str, dict],
    keypoints: dict[str, np.ndarray],
) -> tuple[dict[int, str], dict[int, dict], dict[int, dict]]:
    """映射 syllables → 16 行为.

    Returns:
        syllable_to_behavior: {syllable_id: behavior_name}（P0 自动 + P1 建议）
        syllable_features: {syllable_id: motion_features}
        syllable_posture_stats: {syllable_id: {posture: ratio}}
    """
    # 1. 收集每个 syllable 的所有帧
    syllable_frames: dict[int, list[tuple[str, int]]] = {}  # sid → [(clip, frame_idx)]
    for clip_name, result in kpm_results.items():
        if clip_name not in keypoints:
            continue
        syllable = result["syllable"]
        for t, sid in enumerate(syllable):
            sid_int = int(sid)
            syllable_frames.setdefault(sid_int, []).append((clip_name, t))

    print(f"[map] 发现 {len(syllable_frames)} 个 syllables")
    print(f"[map] 总帧数: {sum(len(v) for v in syllable_frames.values())}")

    # 2. 对每个 syllable 分类姿态（P0 自动映射）
    syllable_to_behavior: dict[int, str] = {}
    syllable_features: dict[int, dict] = {}
    syllable_posture_stats: dict[int, dict] = {}

    for sid, frames in sorted(syllable_frames.items()):
        # 采样最多 200 帧以加速
        if len(frames) > 200:
            indices = np.linspace(0, len(frames) - 1, 200, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames

        # 姿态分类
        posture_counts: Counter = Counter()
        for clip_name, frame_idx in sampled_frames:
            frame = keypoints[clip_name][frame_idx]
            posture = classify_posture_3d(frame)
            posture_counts[posture] += 1

        total = sum(posture_counts.values())
        posture_stats = {k: v / total for k, v in posture_counts.items()}
        syllable_posture_stats[sid] = {
            "total_sampled": total,
            "posture_ratios": posture_stats,
            "posture_counts": dict(posture_counts),
        }

        # P0 自动映射：70%+ 帧分类一致
        mapped_behavior = None
        for posture in (SIT, DOWN, STAND):
            if posture_stats.get(posture, 0) >= P0_CONFIDENCE_THRESHOLD:
                mapped_behavior = posture
                break

        # 计算运动学特征（用于 P1 建议）
        # 取第一个包含此 syllable 的 clip 计算
        if frames:
            clip_name, _ = frames[0]
            if clip_name in keypoints:
                kp = keypoints[clip_name]
                syllable_labels = kpm_results[clip_name]["syllable"]
                features = compute_syllable_motion_features(kp, syllable_labels, sid)
            else:
                features = {"frame_count": len(frames)}
        else:
            features = {"frame_count": 0}

        # 全局帧数（不限于单 clip）
        features["total_frames"] = len(frames)
        syllable_features[sid] = features

        if mapped_behavior:
            syllable_to_behavior[sid] = mapped_behavior
        else:
            # P1 启发式建议
            suggestion = suggest_p1_behavior(features)
            if suggestion:
                syllable_to_behavior[sid] = suggestion
            # 否则保留未映射，待人工决策

    return syllable_to_behavior, syllable_features, syllable_posture_stats


def generate_report(
    syllable_to_behavior: dict[int, str],
    syllable_features: dict[int, dict],
    syllable_posture_stats: dict[int, dict],
    output_path: Path,
) -> None:
    """生成 Markdown 报告."""
    total = len(syllable_features)
    mapped = len(syllable_to_behavior)
    p0_mapped = sum(1 for b in syllable_to_behavior.values() if b in P0_BEHAVIORS)
    p1_suggested = sum(1 for b in syllable_to_behavior.values() if b in P1_BEHAVIORS)
    unmapped = total - mapped

    lines = [
        "# keypoint-MoSeq Syllable → 行为映射报告",
        "",
        f"- 总 syllables: {total}",
        f"- 已映射: {mapped}（P0 自动: {p0_mapped}, P1 建议: {p1_suggested}）",
        f"- 待人工映射: {unmapped}",
        f"- P0 置信度阈值: {P0_CONFIDENCE_THRESHOLD * 100:.0f}%",
        "",
        "## 映射结果（按 syllable ID 排序）",
        "",
        "| syllable | 总帧数 | P0 姿态分布 | 自动映射 | P1 建议 | 运动学特征 |",
        "|----------|--------|-------------|----------|---------|-----------|",
    ]

    for sid in sorted(syllable_features.keys()):
        features = syllable_features[sid]
        stats = syllable_posture_stats.get(sid, {})
        posture_ratios = stats.get("posture_ratios", {})
        total_frames = features.get("total_frames", 0)

        # 姿态分布
        posture_str = " / ".join(
            f"{p}: {r * 100:.0f}%"
            for p, r in sorted(posture_ratios.items(), key=lambda x: -x[1])
            if r > 0.05
        ) or "无有效姿态"

        behavior = syllable_to_behavior.get(sid, "—")
        behavior_cn = BEHAVIOR_NAMES_CN.get(behavior, behavior) if behavior != "—" else "待人工"

        # P1 建议标注
        is_p1_suggestion = behavior in P1_BEHAVIORS
        behavior_display = f"{behavior_cn}{'（建议）' if is_p1_suggestion else ''}"

        # 关键运动学特征
        mean_speed = features.get("mean_speed", 0)
        head_up = features.get("head_up_ratio", 0)
        nose_ground = features.get("nose_near_ground_ratio", 0)
        y_rise = features.get("y_rise_jump", 0)
        motion_str = (
            f"speed={mean_speed:.3f}m/f, head_up={head_up:.2f}, "
            f"nose_ground={nose_ground:.2f}, y_rise={y_rise:.3f}"
        )

        lines.append(
            f"| {sid} | {total_frames} | {posture_str} | "
            f"{behavior if behavior != '—' else '—'} | {behavior_display} | {motion_str} |"
        )

    lines.extend([
        "",
        "## P0 自动映射依据",
        "",
        f"- 3D 坐标系（y-up, 米）",
        f"- 坐 sit: 后腿折叠 < {SIT_REAR_FOLD_MAX}m + 前腿伸直 > {SIT_FRONT_EXTEND_MIN}m",
        f"- 卧 down: 肩甲高度 < {DOWN_WITHERS_HEIGHT_MAX}m + 四肢折叠 < {DOWN_ALL_FOLD_MAX}m",
        f"- 立 stand: 肩甲高度 > {STAND_WITHERS_HEIGHT_MIN}m + 腿伸直 > {STAND_LEG_EXTEND_MIN}m",
        f"- 映射阈值: 单 syllable 内 ≥ {P0_CONFIDENCE_THRESHOLD * 100:.0f}% 帧分类一致",
        "",
        "## P1 待人工映射说明",
        "",
        "P1 行为（追踪/示警坐/示警卧/扑咬/押解/障碍穿越/返回/警戒）需要结合上下文判断，"
        "建议人工查看代表片段后确认。脚本已基于运动学特征给出启发式建议（标注「建议」）。",
        "",
        "### P1 启发式规则",
        "",
        f"- 追踪 track: 鼻尖贴地比例 > 0.7 + 平均速度 > 0.05m/帧",
        f"- 障碍 obstacle: y_rise > 0.10m + 高速 > 0.15m/帧",
        f"- 扑咬 apprehend: 高速 > 0.20m/帧 + 嘴部方差 > 0.001",
        f"- 返回 recall: 方向反转 + 高速 > 0.15m/帧",
        f"- 警戒 watch: 头部抬起 > 0.6 + 低速 < 0.05m/帧",
        "",
        "### 人工映射流程",
        "",
        "1. 查看 `data/kpm_project/syllable_features.json` 中未映射 syllables 的特征",
        "2. 对每个未映射 syllable，在 InterPet4D 视频中定位代表帧（可用 kpm GUI）",
        "3. 根据上下文判定 P1 行为",
        "4. 更新 `data/kpm_project/syllable_mapping.json`",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] 报告已保存: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="syllable → 16 行为映射")
    parser.add_argument(
        "--kpm-dir", type=Path, default=DEFAULT_KPM_DIR,
        help=f"keypoint-MoSeq 项目目录（默认: {DEFAULT_KPM_DIR}）",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"InterPet4D smal_npy 目录（默认: {DEFAULT_DATA_DIR}）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("keypoint-MoSeq syllable → 16 行为映射")
    print(f"kpm 目录: {args.kpm_dir}")
    print(f"数据目录: {args.data_dir}")
    print("=" * 60)

    # 1. 加载 kpm 结果
    kpm_results = load_kpm_results(args.kpm_dir)
    if not kpm_results:
        print("[ERROR] 无 kpm 结果", file=sys.stderr)
        sys.exit(1)

    # 2. 加载 InterPet4D 关键点
    keypoints = load_interpet4d_keypoints(args.data_dir)
    if not keypoints:
        print("[ERROR] 无关键点数据", file=sys.stderr)
        sys.exit(1)

    # 3. 映射
    syllable_to_behavior, syllable_features, syllable_posture_stats = \
        map_syllables_to_behaviors(kpm_results, keypoints)

    # 4. 统计
    total = len(syllable_features)
    mapped = len(syllable_to_behavior)
    p0_mapped = sum(1 for b in syllable_to_behavior.values() if b in P0_BEHAVIORS)
    p1_suggested = sum(1 for b in syllable_to_behavior.values() if b in P1_BEHAVIORS)
    unmapped = total - mapped

    print(f"\n[stats] 总 syllables: {total}")
    print(f"[stats] P0 自动映射: {p0_mapped}")
    print(f"[stats] P1 启发式建议: {p1_suggested}")
    print(f"[stats] 待人工映射: {unmapped}")

    # 5. 保存映射
    mapping_path = args.kpm_dir / "syllable_mapping.json"
    mapping_path.write_text(
        json.dumps(syllable_to_behavior, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[save] 映射: {mapping_path}")

    # 6. 保存特征
    features_path = args.kpm_dir / "syllable_features.json"
    # 移除不可序列化字段
    features_clean = {
        sid: {k: v for k, v in f.items() if isinstance(v, (int, float, str, bool, type(None)))}
        for sid, f in syllable_features.items()
    }
    features_path.write_text(
        json.dumps(features_clean, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[save] 特征: {features_path}")

    # 7. 生成报告
    report_path = args.kpm_dir / "syllable_report.md"
    generate_report(syllable_to_behavior, syllable_features, syllable_posture_stats, report_path)

    print("\n[done] 映射完成！")
    print(f"下一步: 人工审核 {report_path}，更新 {mapping_path}")


if __name__ == "__main__":
    main()
