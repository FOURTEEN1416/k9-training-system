"""Celery 推理任务定义.

Owner: 后端开发 + ML 开发（见 AGENTS.md §2.2）
Phase: 1.4d
依据: dev-docs/stages/phase-1.md §1.4d

任务:
    ingest_video(video_id) — 视频上传后异步推理主入口

流程:
    1. 加载视频元数据
    2. 标记 PROCESSING
    3. YOLO26-pose 24 关键点检测 → keypoints 序列
    4. 写入 keypoints 表
    5. 场景分支:
       - obedience_trial: rule_engine → 行为 episodes → 计算 signals → 评分
       - puppy_selection: 简化 signals（基于 pose 运动学，Phase 1.6 替换）→ 评分
    6. 写入 behaviors 表
    7. 评分引擎评估 → ScoringResult
    8. 生成 PDF 报告 → reports/{video_id}.pdf
    9. 标记 COMPLETED + report_path
    异常 → 标记 FAILED + error_message
"""
from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from backend.app.core.config import settings
from backend.app.core.database_sync import SyncSessionLocal
from backend.app.models.behavior import Behavior, BehaviorClass, BehaviorDetector
from backend.app.models.dog import Dog
from backend.app.models.handler import Handler
from backend.app.models.keypoint import Keypoint
from backend.app.models.video import Video, VideoStatus
from backend.app.services.report import ReportInput, generate_report
from backend.ml.behavior.rule_engine import RuleEngine
from backend.ml.pose.inference import PoseInferenceEngine
from backend.ml.scoring import ScoringContext, ScoringEngine
from backend.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# 评分卡 YAML 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORING_CONFIGS_DIR = PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs"
PUPPY_YAML = SCORING_CONFIGS_DIR / "puppy_selection.yaml"
OBEDIENCE_YAML = SCORING_CONFIGS_DIR / "obedience_trial.yaml"

# 默认模型路径（ONNX Runtime GPU 推理，Phase 1.1 实测最快）
# 优先 onnx，回退 pt
DEFAULT_MODEL_CANDIDATES = [
    PROJECT_ROOT / "runs" / "train-2" / "weights" / "best.onnx",
    PROJECT_ROOT / "runs" / "train-2" / "weights" / "best.pt",
    PROJECT_ROOT / "yolo26n-pose.pt",  # 自动下载
]


def _resolve_model_path() -> str:
    """按优先级解析模型路径。"""
    for p in DEFAULT_MODEL_CANDIDATES:
        if p.exists():
            return str(p)
    # 兜底: 让 ultralytics 自动下载
    return "yolo26n-pose.pt"


def _pose_engine_singleton() -> PoseInferenceEngine:
    """单例 PoseInferenceEngine（避免每次任务重新加载模型）."""
    global _pose_engine
    if _pose_engine is None:
        model_path = _resolve_model_path()
        logger.info(f"[worker] 加载 pose 模型: {model_path}")
        _pose_engine = PoseInferenceEngine(model_path=model_path, verbose=False)
    return _pose_engine


_pose_engine: Optional[PoseInferenceEngine] = None


# ============================================================
# 主任务
# ============================================================


@celery_app.task(
    name="inference.ingest_video",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=2,
)
def ingest_video(self, video_id: int) -> dict:
    """视频推理主任务.

    Args:
        video_id: videos.id

    Returns:
        dict: {video_id, status, scene, report_path?, error?}
    """
    logger.info(f"[worker] ingest_video start video_id={video_id}")

    with SyncSessionLocal() as db:
        try:
            video = db.get(Video, video_id)
            if video is None:
                raise RuntimeError(f"Video {video_id} not found")

            # 标记 PROCESSING
            video.status = VideoStatus.PROCESSING
            video.error_message = None
            db.commit()

            # === 1. YOLO26-pose 推理 ===
            video_full_path = settings.data_dir / video.storage_path
            if not video_full_path.exists():
                raise FileNotFoundError(f"视频文件不存在: {video_full_path}")

            engine = _pose_engine_singleton()
            result = engine.infer_video(
                video_path=video_full_path,
                save_output=False,
            )

            # 更新视频技术元数据
            meta = result.meta
            video.duration_sec = meta.get("duration_sec")
            video.fps = meta.get("fps")
            video.width = meta.get("width")
            video.height = meta.get("height")
            db.commit()

            # === 2. 写入 keypoints 表 ===
            _save_keypoints(db, video_id, result.frames)
            db.commit()

            kpts_seq = result.keypoints_sequence  # (T, 24, 3)

            # === 3. 场景分支 ===
            scene = video.scene
            if scene == "obedience_trial":
                episodes, signals, scoring_result = _run_obedience_pipeline(
                    kpts_seq, meta.get("fps", 30.0)
                )
            elif scene == "puppy_selection":
                episodes, signals, scoring_result = _run_puppy_pipeline(
                    kpts_seq, meta.get("fps", 30.0), meta.get("duration_sec", 0.0)
                )
            else:
                raise ValueError(f"未知 scene: {scene}")

            # === 4. 写入 behaviors 表 ===
            if episodes:
                _save_behaviors(db, video_id, episodes, meta.get("fps", 30.0))
                db.commit()

            # === 5. 生成 PDF 报告 ===
            dog = db.get(Dog, video.dog_id) if video.dog_id else None
            handler = db.get(Handler, video.handler_id) if video.handler_id else None

            report_input = ReportInput(
                dog_name=dog.name if dog else "未关联",
                breed=dog.breed if dog else None,
                birth_date=dog.birth_date.isoformat() if dog and dog.birth_date else None,
                gender=dog.gender.value if dog and dog.gender else None,
                chip_id=dog.chip_id if dog else None,
                training_stage=dog.training_stage.value if dog else None,
                video_filename=video.original_filename,
                video_duration_sec=meta.get("duration_sec"),
                video_fps=meta.get("fps"),
                video_resolution=f"{meta.get('width')}x{meta.get('height')}" if meta.get("width") else None,
                uploaded_at=video.uploaded_at.isoformat() if video.uploaded_at else None,
                scoring_result=scoring_result,
                behaviors=episodes,
                video_id=video_id,
                handler_name=handler.name if handler else None,
            )

            reports_dir = PROJECT_ROOT / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = reports_dir / f"{video_id}.pdf"
            generate_report(report_input, output_path=pdf_path)

            # report_path 存相对 reports_dir 的路径
            video.report_path = f"{video_id}.pdf"
            video.status = VideoStatus.COMPLETED
            video.error_message = None
            video.processed_at = datetime.now(timezone.utc)
            db.commit()

            logger.info(
                f"[worker] ingest_video done video_id={video_id} "
                f"scene={scene} verdict={scoring_result.verdict} "
                f"score={scoring_result.total_score:.1f} pdf={pdf_path.name}"
            )

            return {
                "video_id": video_id,
                "status": "completed",
                "scene": scene,
                "verdict": scoring_result.verdict,
                "total_score": scoring_result.total_score,
                "report_path": video.report_path,
            }

        except Exception as e:
            db.rollback()
            # 重新开 session 标记 FAILED
            with SyncSessionLocal() as db2:
                v = db2.get(Video, video_id)
                if v is not None:
                    v.status = VideoStatus.FAILED
                    v.error_message = f"{type(e).__name__}: {e}"
                    v.processed_at = datetime.now(timezone.utc)
                    db2.commit()
            tb = traceback.format_exc()
            logger.error(f"[worker] ingest_video FAILED video_id={video_id}\n{tb}")
            # 不再重试，让前端看到 FAILED 状态
            raise self.retry(exc=e, countdown=10) if self.request.retries < self.max_retries else None


# ============================================================
# Keypoints 持久化
# ============================================================


def _save_keypoints(db, video_id: int, frames) -> None:
    """逐帧写入 keypoints 表.

    策略: 批量 add + commit，避免逐行往返。
    """
    # 先删除旧 keypoints（重试场景）
    db.query(Keypoint).filter(Keypoint.video_id == video_id).delete()
    db.commit()

    objs = []
    for f in frames:
        objs.append(Keypoint(
            video_id=video_id,
            frame_idx=f.frame_idx,
            frame_time_sec=f.frame_time_sec,
            keypoints_json=f.keypoints.tolist(),
            detection_confidence=f.box_conf,
        ))
    db.add_all(objs)
    # 不在此处 commit，由调用方控制


# ============================================================
# 科目测评 pipeline
# ============================================================


def _run_obedience_pipeline(kpts_seq: np.ndarray, fps: float):
    """科目测评: rule_engine → behaviors → signals → 评分.

    Returns:
        (episodes, signals, ScoringResult)
    """
    # 1. 规则引擎识别 8 类行为
    rule_engine = RuleEngine()
    episodes = rule_engine.recognize(kpts_seq, fps=fps)

    # 2. 从 episodes 计算 signals（与 obedience_trial.yaml 信号对齐）
    signals = _signals_from_obedience_episodes(episodes, fps)

    # 3. 评分
    scoring_engine = ScoringEngine.get(OBEDIENCE_YAML)
    ctx = ScoringContext(signals=signals, scene="obedience_trial")
    result = scoring_engine.evaluate(ctx)

    return episodes, signals, result


def _signals_from_obedience_episodes(episodes, fps: float) -> dict:
    """从行为 episodes 提取评分信号.

    与 obedience_trial.yaml 信号对齐:
        - action_count: 识别到的行为总数
        - action_correct: 视为"正确"的行为数（confidence >= 0.5）
        - command_to_action_latency: 第一个行为起始时间（秒）
        - action_duration: 行为总持续时长（秒）
        - focus_ratio: 有效关键点比例（用 episode 置信度均值近似）
        - gait_score: 步态评分（用 heel episode 置信度近似）
    """
    if not episodes:
        return {
            "action_count": 0,
            "action_correct": 0,
            "command_to_action_latency": 99.0,
            "action_duration": 0.0,
            "focus_ratio": 0.0,
            "gait_score": 0.0,
        }

    action_count = len(episodes)
    action_correct = sum(1 for e in episodes if e.confidence >= 0.5)
    first_start_sec = episodes[0].start_frame / fps if fps > 0 else 0.0
    total_duration = sum(
        (e.end_frame - e.start_frame + 1) / fps if fps > 0 else 0.0
        for e in episodes
    )
    focus_ratio = float(np.mean([e.confidence for e in episodes])) if episodes else 0.0
    # 步态: 取 heel episode 的置信度（无 heel 则 0）
    heel_episodes = [e for e in episodes if e.behavior == "heel"]
    gait_score = float(np.mean([e.confidence for e in heel_episodes])) if heel_episodes else 0.0

    return {
        "action_count": action_count,
        "action_correct": action_correct,
        "command_to_action_latency": float(first_start_sec),
        "action_duration": float(total_duration),
        "focus_ratio": focus_ratio,
        "gait_score": gait_score,
    }


# ============================================================
# 幼犬选育 pipeline（Phase 1.6 前的简化版）
# ============================================================


def _run_puppy_pipeline(kpts_seq: np.ndarray, fps: float, duration_sec: float):
    """幼犬选育: 简化 signals（基于 pose 运动学） → 评分.

    Phase 1.6 实现 puppy_signals.py 后，本函数将被替换为真实信号提取。

    Returns:
        (episodes, signals, ScoringResult)
    """
    signals = _extract_puppy_signals_simple(kpts_seq, fps, duration_sec)
    scoring_engine = ScoringEngine.get(PUPPY_YAML)
    ctx = ScoringContext(signals=signals, scene="puppy_selection")
    result = scoring_engine.evaluate(ctx)

    # 幼犬选育无 rule_engine episodes
    return [], signals, result


def _extract_puppy_signals_simple(
    kpts_seq: np.ndarray, fps: float, duration_sec: float
) -> dict:
    """简化幼犬信号提取（Phase 1.6 前的占位实现）.

    基于 pose 关键点运动学估算，未集成物体检测（球/食物）。
    Phase 1.6 将替换为 puppy_signals.py 真实信号提取。

    估算逻辑:
        - approach_latency: 第一帧到检测到稳定关键点的帧数 / fps
        - approach_speed: withers.x 平均位移速度（像素/秒）
        - chase_latency: 同 approach_latency（无玩具检测）
        - hold_duration: 持续运动时长（秒）
        - retreat_distance: 反向运动最大位移（像素）
        - freeze_duration: 帧间位移 < 阈值的持续秒数
        - recovery_time: 0.0（无惊吓源检测）

    Returns:
        dict: 7 个信号（与 puppy_selection.yaml 对齐）
    """
    if kpts_seq.size == 0 or fps <= 0:
        return {
            "approach_latency": 99.0,
            "approach_speed": 0.0,
            "chase_latency": 99.0,
            "hold_duration": 0.0,
            "retreat_distance": 99.0,
            "freeze_duration": 99.0,
            "recovery_time": 99.0,
        }

    T, K, _ = kpts_seq.shape
    WITHERS = 22

    # 有效帧 mask（withers conf > 0.3）
    valid_mask = kpts_seq[:, WITHERS, 2] >= 0.3
    first_valid = int(np.argmax(valid_mask)) if valid_mask.any() else T
    approach_latency = first_valid / fps if first_valid < T else 99.0

    # withers.x 位移序列
    withers_x = kpts_seq[:, WITHERS, 0]
    valid_withers_x = withers_x[valid_mask]

    if len(valid_withers_x) > 1:
        # 帧间位移
        dx = np.diff(valid_withers_x)
        # 平均速度（像素/秒）
        approach_speed = float(np.mean(np.abs(dx)) * fps)
        # 反向运动最大累积位移（近似 retreat）
        signs = np.sign(dx)
        # 简化: retreat_distance = max(|negative displacement|)
        neg_dx = dx[dx < 0]
        retreat_distance = float(np.sum(np.abs(neg_dx))) if len(neg_dx) > 0 else 0.0
        # 持续运动时长（|dx| > 1.0 像素的帧数 / fps）
        moving_frames = int(np.sum(np.abs(dx) > 1.0))
        hold_duration = moving_frames / fps
        # 冻结时长（|dx| < 0.5 像素的帧数 / fps）
        frozen_frames = int(np.sum(np.abs(dx) < 0.5))
        freeze_duration = frozen_frames / fps
    else:
        approach_speed = 0.0
        retreat_distance = 99.0
        hold_duration = 0.0
        freeze_duration = duration_sec if duration_sec > 0 else 99.0

    return {
        "approach_latency": float(approach_latency),
        "approach_speed": float(approach_speed),
        "chase_latency": float(approach_latency),  # 无玩具检测，用 approach 近似
        "hold_duration": float(hold_duration),
        "retreat_distance": float(retreat_distance),
        "freeze_duration": float(freeze_duration),
        "recovery_time": 0.0,  # 无惊吓源检测
    }


# ============================================================
# Behaviors 持久化
# ============================================================


# P0 行为 → BehaviorClass 枚举映射
# rule_engine 输出小写字符串（如 "sit"），BehaviorClass 枚举成员名大写（如 SIT）
_BEHAVIOR_STRING_TO_ENUM = {
    "sit": BehaviorClass.SIT,
    "down": BehaviorClass.DOWN,
    "stand": BehaviorClass.STAND,
    "heel": BehaviorClass.HEEL,
    "sit_up": BehaviorClass.SIT_UP,
    "stay": BehaviorClass.STAY,
    "bark": BehaviorClass.BARK,
    "bite": BehaviorClass.BITE,
}


def _save_behaviors(db, video_id: int, episodes, fps: float) -> None:
    """写入 behaviors 表."""
    # 先删除旧行（重试场景）
    db.query(Behavior).filter(Behavior.video_id == video_id).delete()
    db.commit()

    objs = []
    for ep in episodes:
        behavior_enum = _BEHAVIOR_STRING_TO_ENUM.get(ep.behavior)
        if behavior_enum is None:
            # 未在枚举中映射的行为跳过（避免写入失败）
            continue
        start_sec = ep.start_frame / fps if fps > 0 else 0.0
        end_sec = ep.end_frame / fps if fps > 0 else 0.0
        objs.append(Behavior(
            video_id=video_id,
            behavior_class=behavior_enum,
            detector=BehaviorDetector.RULE,
            start_frame=ep.start_frame,
            end_frame=ep.end_frame,
            start_sec=start_sec,
            end_sec=end_sec,
            confidence=ep.confidence,
        ))
    db.add_all(objs)


# 注意: health_check 任务定义在 backend.workers.celery_app 中，此处不再重复。
