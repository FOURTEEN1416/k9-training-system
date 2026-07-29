"""幼犬选育 3 维 9 信号提取器.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.6a
依据: dev-docs/stages/phase-1.md §1.6 + RESEARCH_PUPPY_SELECTION_BEHAVIOR.md

3 维 9 信号（与 puppy_selection.yaml 评分卡对齐）:
    食物欲望（food_drive）:
        - approach_latency:   接近食物的延迟（秒）— 从食物出现到鼻尖触及食物框
        - approach_speed:     接近食物的平均速度（像素/秒）— withers 位移向食物
        - sniff_duration:     嗅闻持续时长（秒）— 鼻尖在食物框附近停留帧数

    猎物欲望（prey_drive）:
        - chase_latency:      追逐玩具的延迟（秒）— 从球出现到鼻尖触及球框
        - chase_speed:        追逐速度（像素/秒）— withers 朝向球的位移
        - hold_duration:      咬住/保持玩具的时长（秒）— 鼻尖持续在球框附近

    胆量（courage）:
        - retreat_distance:   后退距离（像素）— 人出现后 withers 反向最大位移
        - freeze_duration:    僵直时长（秒）— 人出现后 |dx| < 阈值的持续帧数
        - recovery_time:      恢复时间（秒）— 从人出现到恢复正常运动

输入:
    kpts_seq: np.ndarray, shape=(T, 24, 3), pose 关键点序列
    detections: VideoDetectionResult, 物体检测结果（Phase 1.5）
    fps: float, 视频帧率
    duration_sec: float, 视频时长（秒）

输出:
    dict: 9 个信号（key 与 puppy_selection.yaml 对齐）

坐标系统:
    图像坐标，y 轴向下（y 大 = 位置低/接近地面）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.ml.behavior.constants import NOSE, WITHERS
from backend.ml.behavior.object_detector import (
    CATEGORY_BALL,
    CATEGORY_FOOD,
    CATEGORY_PERSON,
    CATEGORY_TOY,
    Detection,
    FrameDetection,
    VideoDetectionResult,
)

logger = logging.getLogger(__name__)

# === 默认阈值（可被调用方覆盖） ===
DEFAULT_CONF_THRESHOLD = 0.3        # 关键点置信度低于此值视为不可靠
DEFAULT_PROXIMITY_RATIO = 0.3       # 鼻尖距物体框中心的"触及"距离阈值（按框对角线比例）
DEFAULT_MOTION_THRESHOLD = 1.0      # 帧间位移 > 此值视为"在运动"（像素）
DEFAULT_FREEZE_THRESHOLD = 0.5      # 帧间位移 < 此值视为"冻结"（像素）
DEFAULT_SCARE_PERSON_AREA = 5000.0  # person 框面积 > 此值视为"惊吓源"出现

# 缺失信号时的占位值（与评分卡阈值对齐，使降级路径得分低）
PENALTY_LATENCY = 99.0   # 极长延迟
PENALTY_DISTANCE = 99.0  # 极大后退
PENALTY_FREEZE = 99.0    # 极长冻结


# ============================================================
# 信号提取主入口
# ============================================================


@dataclass
class PuppySignalConfig:
    """幼犬选育信号提取配置（可调阈值）。"""

    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    proximity_ratio: float = DEFAULT_PROXIMITY_RATIO
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD
    freeze_threshold: float = DEFAULT_FREEZE_THRESHOLD
    scare_person_area: float = DEFAULT_SCARE_PERSON_AREA


def extract_puppy_signals(
    kpts_seq: np.ndarray,
    detections: Optional[VideoDetectionResult],
    fps: float,
    duration_sec: float = 0.0,
    config: Optional[PuppySignalConfig] = None,
) -> dict:
    """提取 9 个选育信号.

    Args:
        kpts_seq: pose 关键点序列 (T, 24, 3)
        detections: 物体检测结果（None 表示未运行物体检测，信号降级）
        fps: 视频帧率
        duration_sec: 视频时长（秒）
        config: 阈值配置（None 用默认）

    Returns:
        dict: 9 个信号
            approach_latency, approach_speed, sniff_duration,
            chase_latency, chase_speed, hold_duration,
            retreat_distance, freeze_duration, recovery_time
    """
    cfg = config or PuppySignalConfig()

    # 输入校验 + 空数据降级
    if kpts_seq is None or kpts_seq.size == 0 or fps <= 0:
        return _empty_signals()

    T = kpts_seq.shape[0]
    if kpts_seq.shape[1] != 24:
        logger.warning(
            f"puppy_signals: kpts_seq shape {kpts_seq.shape} 不符合 (T, 24, 3)"
        )
        return _empty_signals()

    # 提取狗的姿态序列
    nose_seq = kpts_seq[:, NOSE, :]      # (T, 3)
    withers_seq = kpts_seq[:, WITHERS, :]  # (T, 3)
    nose_valid = nose_seq[:, 2] >= cfg.conf_threshold
    withers_valid = withers_seq[:, 2] >= cfg.conf_threshold

    # 提取物体检测帧索引
    det_frames: list[FrameDetection] = (
        detections.frames if detections is not None else []
    )
    # 帧索引 → FrameDetection 映射
    det_by_idx: dict[int, FrameDetection] = {
        f.frame_idx: f for f in det_frames
    }

    # ===== 食物欲望信号 =====
    food_signals = _extract_food_signals(
        nose_seq, withers_seq, nose_valid, withers_valid,
        det_by_idx, T, fps, cfg,
    )

    # ===== 猎物欲望信号 =====
    prey_signals = _extract_prey_signals(
        nose_seq, withers_seq, nose_valid, withers_valid,
        det_by_idx, T, fps, cfg,
    )

    # ===== 胆量信号 =====
    courage_signals = _extract_courage_signals(
        withers_seq, withers_valid,
        det_by_idx, T, fps, cfg,
    )

    return {
        **food_signals,
        **prey_signals,
        **courage_signals,
    }


# ============================================================
# 食物欲望信号
# ============================================================


def _extract_food_signals(
    nose_seq: np.ndarray,
    withers_seq: np.ndarray,
    nose_valid: np.ndarray,
    withers_valid: np.ndarray,
    det_by_idx: dict[int, FrameDetection],
    T: int,
    fps: float,
    cfg: PuppySignalConfig,
) -> dict:
    """提取食物欲望 3 信号: approach_latency, approach_speed, sniff_duration."""
    # 找到食物首次出现的帧
    food_first_frame, food_frames = _find_object_frames(
        det_by_idx, T, {CATEGORY_FOOD}
    )

    if food_first_frame is None:
        # 全程无食物 → 极差评分（高延迟，无嗅闻）
        return {
            "approach_latency": PENALTY_LATENCY,
            "approach_speed": 0.0,
            "sniff_duration": 0.0,
        }

    # 计算鼻尖到食物框的距离序列（仅对食物出现的帧）
    nose_to_food_dist = np.full(T, np.inf, dtype=np.float64)
    food_box_diag = np.full(T, 0.0, dtype=np.float64)
    for fi in food_frames:
        if fi >= T or not nose_valid[fi]:
            continue
        frame_det = det_by_idx.get(fi)
        if frame_det is None:
            continue
        food_dets = frame_det.filter_by_category(CATEGORY_FOOD)
        if not food_dets:
            continue
        nose_xy = nose_seq[fi, :2]
        # 取距离最近的食物框
        min_dist = np.inf
        max_diag = 0.0
        for d in food_dets:
            cx, cy = d.center
            dist = float(np.hypot(nose_xy[0] - cx, nose_xy[1] - cy))
            if dist < min_dist:
                min_dist = dist
            diag = float(np.hypot(d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]))
            if diag > max_diag:
                max_diag = diag
        nose_to_food_dist[fi] = min_dist
        food_box_diag[fi] = max_diag

    # approach_latency: 食物首次出现 → 鼻尖首次触及食物框（距离 < 阈值）
    valid_diags = food_box_diag[food_box_diag > 0]
    if len(valid_diags) > 0:
        proximity_threshold = cfg.proximity_ratio * float(np.median(valid_diags))
    else:
        proximity_threshold = 30.0  # 兜底：30 像素

    approach_frame = None
    for fi in food_frames:
        if fi >= T:
            continue
        if nose_to_food_dist[fi] <= proximity_threshold:
            approach_frame = fi
            break

    if approach_frame is None:
        # 鼻尖从未触及食物
        approach_latency = PENALTY_LATENCY
        sniff_duration = 0.0
    else:
        approach_latency = (approach_frame - food_first_frame) / fps
        # sniff_duration: 鼻尖在食物框附近的持续帧数
        sniff_frames = int(np.sum(
            (nose_to_food_dist[food_frames] <= proximity_threshold)
        ))
        sniff_duration = sniff_frames / fps

    # approach_speed: 食物出现后到首次触及之间，withers 朝食物方向的平均速度
    if approach_frame is not None and approach_frame > food_first_frame:
        approach_window = slice(food_first_frame, approach_frame)
        # withers 朝食物中心位移
        withers_x = withers_seq[approach_window, 0]
        withers_y = withers_seq[approach_window, 1]
        w_valid = withers_valid[approach_window]
        if w_valid.sum() >= 2:
            dx = np.diff(withers_x[w_valid])
            dy = np.diff(withers_y[w_valid])
            # 总位移 / 时间
            total_disp = float(np.mean(np.hypot(dx, dy)))
            approach_speed = total_disp * fps
        else:
            approach_speed = 0.0
    else:
        approach_speed = 0.0

    return {
        "approach_latency": float(approach_latency),
        "approach_speed": float(approach_speed),
        "sniff_duration": float(sniff_duration),
    }


# ============================================================
# 猎物欲望信号
# ============================================================


def _extract_prey_signals(
    nose_seq: np.ndarray,
    withers_seq: np.ndarray,
    nose_valid: np.ndarray,
    withers_valid: np.ndarray,
    det_by_idx: dict[int, FrameDetection],
    T: int,
    fps: float,
    cfg: PuppySignalConfig,
) -> dict:
    """提取猎物欲望 3 信号: chase_latency, chase_speed, hold_duration."""
    # 找到球/玩具首次出现的帧
    ball_first_frame, ball_frames = _find_object_frames(
        det_by_idx, T, {CATEGORY_BALL, CATEGORY_TOY}
    )

    if ball_first_frame is None:
        # 全程无球/玩具
        return {
            "chase_latency": PENALTY_LATENCY,
            "chase_speed": 0.0,
            "hold_duration": 0.0,
        }

    # 鼻尖到球框的距离
    nose_to_ball_dist = np.full(T, np.inf, dtype=np.float64)
    ball_box_diag = np.full(T, 0.0, dtype=np.float64)
    for fi in ball_frames:
        if fi >= T or not nose_valid[fi]:
            continue
        frame_det = det_by_idx.get(fi)
        if frame_det is None:
            continue
        ball_dets = (
            frame_det.filter_by_category(CATEGORY_BALL)
            + frame_det.filter_by_category(CATEGORY_TOY)
        )
        if not ball_dets:
            continue
        nose_xy = nose_seq[fi, :2]
        min_dist = np.inf
        max_diag = 0.0
        for d in ball_dets:
            cx, cy = d.center
            dist = float(np.hypot(nose_xy[0] - cx, nose_xy[1] - cy))
            if dist < min_dist:
                min_dist = dist
            diag = float(np.hypot(d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]))
            if diag > max_diag:
                max_diag = diag
        nose_to_ball_dist[fi] = min_dist
        ball_box_diag[fi] = max_diag

    # chase_latency: 球首次出现 → 鼻尖触及球框
    valid_diags = ball_box_diag[ball_box_diag > 0]
    if len(valid_diags) > 0:
        proximity_threshold = cfg.proximity_ratio * float(np.median(valid_diags))
    else:
        proximity_threshold = 30.0  # 兜底：30 像素

    chase_frame = None
    for fi in ball_frames:
        if fi >= T:
            continue
        if nose_to_ball_dist[fi] <= proximity_threshold:
            chase_frame = fi
            break

    if chase_frame is None:
        chase_latency = PENALTY_LATENCY
        hold_duration = 0.0
    else:
        chase_latency = (chase_frame - ball_first_frame) / fps
        # hold_duration: 鼻尖持续在球框附近的时长
        hold_frames = int(np.sum(
            (nose_to_ball_dist[ball_frames] <= proximity_threshold)
        ))
        hold_duration = hold_frames / fps

    # chase_speed: 球出现后到首次触及之间，withers 朝球的平均速度
    if chase_frame is not None and chase_frame > ball_first_frame:
        chase_window = slice(ball_first_frame, chase_frame)
        withers_x = withers_seq[chase_window, 0]
        withers_y = withers_seq[chase_window, 1]
        w_valid = withers_valid[chase_window]
        if w_valid.sum() >= 2:
            dx = np.diff(withers_x[w_valid])
            dy = np.diff(withers_y[w_valid])
            total_disp = float(np.mean(np.hypot(dx, dy)))
            chase_speed = total_disp * fps
        else:
            chase_speed = 0.0
    else:
        chase_speed = 0.0

    return {
        "chase_latency": float(chase_latency),
        "chase_speed": float(chase_speed),
        "hold_duration": float(hold_duration),
    }


# ============================================================
# 胆量信号
# ============================================================


def _extract_courage_signals(
    withers_seq: np.ndarray,
    withers_valid: np.ndarray,
    det_by_idx: dict[int, FrameDetection],
    T: int,
    fps: float,
    cfg: PuppySignalConfig,
) -> dict:
    """提取胆量 3 信号: retreat_distance, freeze_duration, recovery_time.

    惊吓源定义: person 框首次出现（且面积 >= scare_person_area）
    """
    # 找到人首次出现的帧（惊吓事件）
    scare_frame = None
    for fi in sorted(det_by_idx.keys()):
        if fi >= T:
            continue
        frame_det = det_by_idx[fi]
        persons = frame_det.filter_by_category(CATEGORY_PERSON)
        for p in persons:
            if p.area >= cfg.scare_person_area:
                scare_frame = fi
                break
        if scare_frame is not None:
            break

    if scare_frame is None:
        # 无惊吓事件 → 胆量评分走高（无后退、无冻结、无需恢复）
        return {
            "retreat_distance": 0.0,
            "freeze_duration": 0.0,
            "recovery_time": 0.0,
        }

    # 惊吓后的 withers 运动序列
    post_scare = slice(scare_frame, T)
    withers_x_post = withers_seq[post_scare, 0]
    withers_y_post = withers_seq[post_scare, 1]
    w_valid_post = withers_valid[post_scare]

    # retreat_distance: 惊吓后 withers 反向最大位移（绝对值）
    if w_valid_post.sum() >= 2:
        # 计算累积位移（相对惊吓时刻位置）
        valid_x = withers_x_post[w_valid_post]
        valid_y = withers_y_post[w_valid_post]
        if len(valid_x) >= 2:
            # 帧间位移向量
            dx = np.diff(valid_x)
            dy = np.diff(valid_y)
            disp = np.hypot(dx, dy)
            # 后退 = 位移方向与初始运动方向相反
            # 简化: 取惊吓后 0.5s 内位移的逆向累积
            scare_window = max(1, int(0.5 * fps))
            scare_disp_x = dx[:scare_window] if len(dx) >= scare_window else dx
            scare_disp_y = dy[:scare_window] if len(dy) >= scare_window else dy
            # 后退距离 = 惊吓后 0.5s 内总位移（取绝对值，因狗可能向任意方向逃离）
            retreat_distance = float(np.sum(np.hypot(scare_disp_x, scare_disp_y)))
        else:
            retreat_distance = 0.0
    else:
        retreat_distance = 0.0

    # freeze_duration: 惊吓后 |位移| < freeze_threshold 的持续帧数
    if w_valid_post.sum() >= 2:
        valid_x = withers_x_post[w_valid_post]
        valid_y = withers_y_post[w_valid_post]
        dx = np.diff(valid_x)
        dy = np.diff(valid_y)
        disp = np.hypot(dx, dy)
        frozen_mask = disp < cfg.freeze_threshold
        # 找到最长连续冻结段
        freeze_duration = _longest_true_run(frozen_mask) / fps
    else:
        freeze_duration = 0.0

    # recovery_time: 从惊吓到恢复正常运动（|位移| > motion_threshold）的时间
    if w_valid_post.sum() >= 2:
        valid_x = withers_x_post[w_valid_post]
        valid_y = withers_y_post[w_valid_post]
        dx = np.diff(valid_x)
        dy = np.diff(valid_y)
        disp = np.hypot(dx, dy)
        moving_mask = disp > cfg.motion_threshold
        # 找到首次恢复运动的帧（连续 motion_threshold 帧）
        recovery_frames = _find_first_run(moving_mask, min_run=3)
        recovery_time = recovery_frames / fps if recovery_frames is not None else (
            (T - scare_frame) / fps  # 始终未恢复
        )
    else:
        recovery_time = (T - scare_frame) / fps

    return {
        "retreat_distance": float(retreat_distance),
        "freeze_duration": float(freeze_duration),
        "recovery_time": float(recovery_time),
    }


# ============================================================
# 辅助函数
# ============================================================


def _empty_signals() -> dict:
    """空信号占位（输入无效时返回）。"""
    return {
        "approach_latency": PENALTY_LATENCY,
        "approach_speed": 0.0,
        "sniff_duration": 0.0,
        "chase_latency": PENALTY_LATENCY,
        "chase_speed": 0.0,
        "hold_duration": 0.0,
        "retreat_distance": PENALTY_DISTANCE,
        "freeze_duration": PENALTY_FREEZE,
        "recovery_time": PENALTY_LATENCY,
    }


def _find_object_frames(
    det_by_idx: dict[int, FrameDetection],
    T: int,
    categories: set[str],
) -> tuple[Optional[int], list[int]]:
    """找到指定类别物体出现的所有帧.

    Returns:
        (first_frame, all_frames) — 无检测时 first_frame=None, all_frames=[]
    """
    frames_with_obj: list[int] = []
    for fi in sorted(det_by_idx.keys()):
        if fi >= T:
            continue
        frame_det = det_by_idx[fi]
        for cat in categories:
            if frame_det.filter_by_category(cat):
                frames_with_obj.append(fi)
                break
    if not frames_with_obj:
        return None, []
    return frames_with_obj[0], frames_with_obj


def _longest_true_run(mask: np.ndarray) -> int:
    """找到 bool 数组中最长连续 True 段的长度。"""
    if len(mask) == 0:
        return 0
    max_run = 0
    cur_run = 0
    for v in mask:
        if v:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return max_run


def _find_first_run(mask: np.ndarray, min_run: int = 3) -> Optional[int]:
    """找到 bool 数组中首次出现连续 min_run 个 True 的位置（返回帧索引）。

    Returns:
        首次满足条件的帧索引，或 None（全程未满足）
    """
    if len(mask) < min_run:
        return None
    cur_run = 0
    for i, v in enumerate(mask):
        if v:
            cur_run += 1
            if cur_run >= min_run:
                return i - min_run + 1
        else:
            cur_run = 0
    return None


# ============================================================
# CLI（手动测试用）
# ============================================================


def _cli() -> int:
    """命令行测试: 输入 pkl + 关键点 → 9 信号输出。"""
    import argparse
    import pickle
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="幼犬选育信号提取器（Phase 1.6）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kpts", required=True, help="关键点序列 .pkl 路径 (T, 24, 3)")
    parser.add_argument("--detections", required=True,
                        help="物体检测结果 .pkl 路径（ObjectDetector 输出）")
    parser.add_argument("--fps", type=float, default=30.0, help="视频帧率")
    parser.add_argument("--duration", type=float, default=0.0, help="视频时长（秒）")
    args = parser.parse_args()

    with open(args.kpts, "rb") as fp:
        kpts_seq = pickle.load(fp)
    if isinstance(kpts_seq, dict) and "keypoints" in kpts_seq:
        kpts_seq = kpts_seq["keypoints"]
    kpts_seq = np.asarray(kpts_seq, dtype=np.float32)
    print(f"kpts_seq shape: {kpts_seq.shape}", flush=True)

    # 加载物体检测 pkl（ObjectDetector._save_pkl 输出格式）
    with open(args.detections, "rb") as fp:
        det_payload = pickle.load(fp)

    # 重建 VideoDetectionResult
    frames: list[FrameDetection] = []
    for f_dict in det_payload["frames"]:
        dets = [
            Detection(
                class_id=d["class_id"],
                class_name=d["class_name"],
                category=d["category"],
                confidence=d["confidence"],
                bbox=np.array(d["bbox"], dtype=np.float32),
            )
            for d in f_dict["detections"]
        ]
        frames.append(FrameDetection(
            frame_idx=f_dict["frame_idx"],
            frame_time_sec=f_dict["frame_time_sec"],
            detections=dets,
        ))
    det_result = VideoDetectionResult(meta=det_payload["meta"], frames=frames)
    print(f"detections: {len(frames)} frames", flush=True)

    signals = extract_puppy_signals(
        kpts_seq=kpts_seq,
        detections=det_result,
        fps=args.fps,
        duration_sec=args.duration,
    )

    print("\n=== 选育信号 ===", flush=True)
    for k, v in signals.items():
        print(f"  {k}: {v:.3f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
