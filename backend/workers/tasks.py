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
from backend.ml.behavior import BehaviorRecognizer, DeployMode
from backend.ml.behavior.mamba_inference import MambaInferer
from backend.ml.behavior.mamba_bc_inference import MambaBCInferer
from backend.ml.behavior.object_detector import (
    ObjectDetector,
    VideoDetectionResult,
    get_detector_singleton,
)
from backend.ml.behavior.puppy_signals import extract_puppy_signals
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
USPCA_YAML = SCORING_CONFIGS_DIR / "uspca_patrol.yaml"
FCI_IGP_YAML = SCORING_CONFIGS_DIR / "fci_igp.yaml"

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


# ST-GCN+BC 模型路径候选（Phase 3.1e 双轨部署）
STGCN_BC_ONNX_CANDIDATES = [
    PROJECT_ROOT / "data" / "models" / "stgcn_bc" / "stgcn_bc_dog24.onnx",
]
STGCN_BC_CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "runs" / "stgcn_bc_synthetic" / "best.pt",
]


def _resolve_stgcn_bc_path() -> tuple[Optional[str], Optional[str]]:
    """解析 ST-GCN+BC 模型路径.

    Returns:
        (onnx_path, checkpoint_path) — 优先 ONNX，回退 checkpoint，全无返回 (None, None)
    """
    for p in STGCN_BC_ONNX_CANDIDATES:
        if p.exists():
            return (str(p), None)
    for p in STGCN_BC_CHECKPOINT_CANDIDATES:
        if p.exists():
            return (None, str(p))
    return (None, None)


# 部署模式（可通过 settings.behavior_deploy_mode 配置切换）
# shadow: 影子模式（ST-GCN+BC 推理 + 规则引擎返回，仅记录对比）
# primary_stgcn: ST-GCN+BC 主 + 规则引擎备（失败降级）
# rule_only: 仅规则引擎（ST-GCN+BC 不可用）
_behavior_recognizer: Optional[BehaviorRecognizer] = None
_mamba_inferer: Optional[MambaInferer] = None
_mamba_bc_inferer: Optional[MambaBCInferer] = None


def _resolve_mamba_model_path() -> Optional[str]:
    """解析 Mamba 模型路径."""
    candidates = [
        PROJECT_ROOT / "data" / "models" / "mamba" / "mamba_dog24.onnx",
        PROJECT_ROOT / "runs" / "mamba_synthetic" / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _resolve_mamba_bc_path() -> Optional[str]:
    """解析 Mamba+BC 模型路径."""
    candidates = [
        PROJECT_ROOT / "data" / "models" / "mamba_bc" / "mamba_bc_dog24.onnx",
        PROJECT_ROOT / "runs" / "mamba_bc_synthetic" / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _mamba_inferer_singleton() -> Optional[MambaInferer]:
    """单例 MambaInferer（按需懒加载）."""
    global _mamba_inferer
    if _mamba_inferer is None:
        model_path = _resolve_mamba_model_path()
        if model_path is None:
            return None
        if model_path.endswith(".onnx"):
            _mamba_inferer = MambaInferer(onnx_path=model_path)
        else:
            _mamba_inferer = MambaInferer(checkpoint_path=model_path)
    return _mamba_inferer


def _mamba_bc_inferer_singleton() -> Optional[MambaBCInferer]:
    """单例 MambaBCInferer（按需懒加载）."""
    global _mamba_bc_inferer
    if _mamba_bc_inferer is None:
        model_path = _resolve_mamba_bc_path()
        if model_path is None:
            return None
        if model_path.endswith(".onnx"):
            _mamba_bc_inferer = MambaBCInferer(onnx_path=model_path)
        else:
            _mamba_bc_inferer = MambaBCInferer(checkpoint_path=model_path)
    return _mamba_bc_inferer


def _behavior_recognizer_singleton() -> BehaviorRecognizer:
    """单例 BehaviorRecognizer（双轨部署入口）.

    自动检测 ST-GCN+BC 模型可用性：
        - 模型存在 → SHADOW 模式（默认，安全过渡）
        - 模型不存在 → RULE_ONLY 模式（降级到规则引擎）
    """
    global _behavior_recognizer
    if _behavior_recognizer is None:
        mode_str = getattr(settings, "behavior_deploy_mode", "shadow")
        if mode_str.startswith("mamba"):
            mamba_inferer = _mamba_inferer_singleton()
            mamba_bc_inferer = _mamba_bc_inferer_singleton()
            mode = DeployMode(mode_str)
            if mode in (DeployMode.MAMBA_ONLY, DeployMode.MAMBA_SHADOW, DeployMode.MAMBA_VOTE):
                if mamba_inferer is None:
                    logger.info("[worker] Mamba 模型不可用，降级 RULE_ONLY")
                    _behavior_recognizer = BehaviorRecognizer(mode=DeployMode.RULE_ONLY)
                else:
                    _behavior_recognizer = BehaviorRecognizer(mode=mode, mamba_inferer=mamba_inferer)
            else:
                if mamba_bc_inferer is None:
                    logger.info("[worker] Mamba+BC 模型不可用，降级 RULE_ONLY")
                    _behavior_recognizer = BehaviorRecognizer(mode=DeployMode.RULE_ONLY)
                else:
                    _behavior_recognizer = BehaviorRecognizer(mode=mode, mamba_bc_inferer=mamba_bc_inferer)
            logger.info(f"[worker] 行为识别部署模式: {mode_str}")
        else:
            onnx_path, ckpt_path = _resolve_stgcn_bc_path()
            if onnx_path is None and ckpt_path is None:
                logger.info("[worker] ST-GCN+BC 模型不可用，使用 RULE_ONLY 模式")
                _behavior_recognizer = BehaviorRecognizer(mode=DeployMode.RULE_ONLY)
            else:
                from backend.ml.behavior.stgcn_bc.inference import STGCNBCInferer

                if onnx_path:
                    inferer = STGCNBCInferer(onnx_path=onnx_path)
                    logger.info(f"[worker] ST-GCN+BC ONNX 后端: {onnx_path}")
                else:
                    inferer = STGCNBCInferer(checkpoint_path=ckpt_path)
                    logger.info(f"[worker] ST-GCN+BC PyTorch 后端: {ckpt_path}")

                _behavior_recognizer = BehaviorRecognizer(
                    mode=DeployMode(mode_str),
                    stgcn_inferer=inferer,
                )
                logger.info(f"[worker] 行为识别部署模式: {mode_str}")
    return _behavior_recognizer


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
                # Phase 1.5/1.6: 物体检测 → 选育信号提取
                detection_result = _run_object_detection(video_full_path)
                episodes, signals, scoring_result = _run_puppy_pipeline(
                    kpts_seq, detection_result,
                    meta.get("fps", 30.0), meta.get("duration_sec", 0.0),
                )
            elif scene == "uspca_patrol":
                # Phase 2.3/2.6: USPCA 巡逻犬 16 行为 + 5 维评分
                episodes, signals, scoring_result = _run_uspca_pipeline(
                    kpts_seq, meta.get("fps", 30.0), meta.get("duration_sec", 0.0)
                )
            elif scene == "fci_igp":
                # Phase 3.1e/3.4: FCI-IGP 国际工作犬 22 行为 + 7 维评分
                episodes, signals, scoring_result = _run_fci_igp_pipeline(
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
    """科目测评: behavior_recognizer → behaviors → signals → 评分.

    Phase 3.1e: 双轨部署（ST-GCN+BC 影子 + 规则引擎返回）

    Returns:
        (episodes, signals, ScoringResult)
    """
    # 1. 行为识别（双轨路由：shadow/vote/primary_stgcn/rule_only）
    recognizer = _behavior_recognizer_singleton()
    episodes = recognizer.recognize(kpts_seq, fps=fps)

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
# USPCA 巡逻犬 pipeline（Phase 2.3/2.6: 16 行为 + 5 维评分）
# ============================================================


def _run_uspca_pipeline(
    kpts_seq: np.ndarray,
    fps: float,
    duration_sec: float,
):
    """USPCA 巡逻犬认证: 16 行为识别 → signals → 5 维评分.

    Phase 3.1e: 双轨部署（ST-GCN+BC 影子 + 规则引擎返回）

    Returns:
        (episodes, signals, ScoringResult)
    """
    # 1. 行为识别（双轨路由）
    recognizer = _behavior_recognizer_singleton()
    episodes = recognizer.recognize(kpts_seq, fps=fps)

    # 2. 从 episodes 提取 USPCA 5 维信号
    signals = _signals_from_uspca_episodes(episodes, fps, duration_sec)

    # 3. USPCA 5 维评分
    scoring_engine = ScoringEngine.get(USPCA_YAML)
    ctx = ScoringContext(signals=signals, scene="uspca_patrol")
    result = scoring_engine.evaluate(ctx)

    return episodes, signals, result


def _signals_from_uspca_episodes(episodes, fps: float, duration_sec: float) -> dict:
    """从 16 行为 episodes 提取 USPCA 5 维评分信号.

    与 uspca_patrol.yaml 信号对齐:
        - action_correct / action_count: 准确度（confidence >= 0.80 视为正确）
        - command_to_action_latency: 延迟（第一个行为起始时间）
        - action_duration: 保持（行为总持续时长）
        - search_coverage / search_speed / target_found: 搜索效率
        - focus_ratio / unnecessary_movement_count: 注意力
    """
    if not episodes:
        return {
            "action_correct": 0,
            "action_count": 0,
            "command_to_action_latency": 99.0,
            "action_duration": 0.0,
            "search_coverage": 0.0,
            "search_speed": 0.0,
            "target_found": False,
            "focus_ratio": 0.0,
            "unnecessary_movement_count": 0,
        }

    action_count = len(episodes)
    action_correct = sum(1 for e in episodes if e.confidence >= 0.80)
    first_start_sec = episodes[0].start_frame / fps if fps > 0 else 0.0
    total_duration = sum(
        (e.end_frame - e.start_frame + 1) / fps if fps > 0 else 0.0
        for e in episodes
    )
    focus_ratio = float(np.mean([e.confidence for e in episodes]))

    # 搜索效率: 从 track/search 相关行为估算
    # P1 行为 track（追踪）反映搜索能力
    track_episodes = [e for e in episodes if e.behavior == "track"]
    if track_episodes and duration_sec > 0:
        track_duration = sum(
            (e.end_frame - e.start_frame + 1) / fps for e in track_episodes if fps > 0
        )
        search_coverage = min(1.0, track_duration / duration_sec)
        search_speed = len(track_episodes) / duration_sec if duration_sec > 0 else 0.0
        target_found = any(e.confidence >= 0.7 for e in track_episodes)
    else:
        search_coverage = 0.3  # 默认中性值（无 track 行为时）
        search_speed = 0.1
        target_found = False

    # 注意力: bark（吠叫）视为不必要移动/噪音
    bark_episodes = [e for e in episodes if e.behavior == "bark"]
    unnecessary_movement_count = len(bark_episodes)

    return {
        "action_correct": action_correct,
        "action_count": action_count,
        "command_to_action_latency": float(first_start_sec),
        "action_duration": float(total_duration),
        "search_coverage": float(search_coverage),
        "search_speed": float(search_speed),
        "target_found": bool(target_found),
        "focus_ratio": focus_ratio,
        "unnecessary_movement_count": unnecessary_movement_count,
    }


# ============================================================
# FCI-IGP 国际工作犬 pipeline（Phase 3.1e/3.4: 22 行为 + 7 维评分）
# ============================================================


def _run_fci_igp_pipeline(
    kpts_seq: np.ndarray,
    fps: float,
    duration_sec: float,
):
    """FCI-IGP 国际工作犬认证: 22 行为识别 → signals → 7 维评分.

    Phase 3.1e: 双轨部署（ST-GCN+BC 主 + 规则引擎备）
    Phase 3.4: FCI-IGP 7 维评分卡（accuracy/latency/duration/search/attention/courage/gait）

    Returns:
        (episodes, signals, ScoringResult)
    """
    # 1. 行为识别（双轨路由：FCI-IGP 场景优先 ST-GCN+BC 覆盖 22 类）
    recognizer = _behavior_recognizer_singleton()
    episodes = recognizer.recognize(kpts_seq, fps=fps)

    # 2. 从 episodes 提取 FCI-IGP 7 维信号
    signals = _signals_from_fci_igp_episodes(episodes, fps, duration_sec)

    # 3. FCI-IGP 7 维评分
    scoring_engine = ScoringEngine.get(FCI_IGP_YAML)
    ctx = ScoringContext(signals=signals, scene="fci_igp")
    result = scoring_engine.evaluate(ctx)

    return episodes, signals, result


def _signals_from_fci_igp_episodes(episodes, fps: float, duration_sec: float) -> dict:
    """从 22 行为 episodes 提取 FCI-IGP 7 维评分信号（委托到独立模块）.

    实际实现已提取到 backend/ml/behavior/fci_igp_signals.py，
    消除 celery 依赖以提升单元测试可测试性。
    """
    from backend.ml.behavior.fci_igp_signals import signals_from_fci_igp_episodes

    return signals_from_fci_igp_episodes(episodes, fps, duration_sec)


# ============================================================
# 幼犬选育 pipeline（Phase 1.6: 真实信号提取）
# ============================================================


def _run_object_detection(video_path: Path) -> Optional[VideoDetectionResult]:
    """运行 YOLO26 COCO 80 类物体检测（仅幼犬选育场景调用）.

    Phase 1.5/1.6 集成:
        - 单例缓存 ObjectDetector（避免每任务重新加载模型）
        - 失败时返回 None（信号提取器会降级为纯 pose 估算）

    Args:
        video_path: 视频文件绝对路径

    Returns:
        VideoDetectionResult 或 None（检测失败时）
    """
    try:
        detector = get_detector_singleton()
        return detector.detect_video(video_path, save_output=False)
    except Exception as e:
        logger.warning(
            f"[worker] 物体检测失败，信号提取将降级: {type(e).__name__}: {e}"
        )
        return None


def _run_puppy_pipeline(
    kpts_seq: np.ndarray,
    detection_result: Optional[VideoDetectionResult],
    fps: float,
    duration_sec: float,
):
    """幼犬选育: 物体检测 + pose → 9 信号 → 评分.

    Phase 1.6 实现:
        - 调用 puppy_signals.extract_puppy_signals 提取 9 信号
        - 信号字典与 puppy_selection.yaml v1.1.0 对齐
        - detection_result=None 时（检测失败），信号提取器降级为纯 pose 估算

    Returns:
        (episodes, signals, ScoringResult)
        - episodes: 空列表（选育场景无 rule_engine episodes）
        - signals: 9 信号字典
        - scoring_result: ScoringResult
    """
    signals = extract_puppy_signals(
        kpts_seq=kpts_seq,
        detections=detection_result,
        fps=fps,
        duration_sec=duration_sec,
    )

    scoring_engine = ScoringEngine.get(PUPPY_YAML)
    ctx = ScoringContext(signals=signals, scene="puppy_selection")
    result = scoring_engine.evaluate(ctx)

    # 幼犬选育无 rule_engine episodes
    return [], signals, result


# ============================================================
# Behaviors 持久化
# ============================================================


# P0 + P1 行为 → BehaviorClass 枚举映射（16 类）
# rule_engine / ST-GCN+BC 输出小写字符串（如 "sit"），BehaviorClass 枚举成员名大写（如 SIT）
# 注意: P2 高级 6 类（guard/release/retrieve/jump/scale/search_blind）在 constants.py
# 与 BehaviorClass 枚举（FOOD_DRIVE/COURAGE/...）命名不一致，需 ADR 决策对齐
# 暂未映射，_save_behaviors 会跳过未映射行为（line 524-527 已有逻辑）
_BEHAVIOR_STRING_TO_ENUM = {
    # P0 基础 8 类
    "sit": BehaviorClass.SIT,
    "down": BehaviorClass.DOWN,
    "stand": BehaviorClass.STAND,
    "heel": BehaviorClass.HEEL,
    "sit_up": BehaviorClass.SIT_UP,
    "stay": BehaviorClass.STAY,
    "bark": BehaviorClass.BARK,
    "bite": BehaviorClass.BITE,
    # P1 训练专项 8 类（Phase 2 扩展）
    "track": BehaviorClass.TRACK,
    "alert_sit": BehaviorClass.ALERT_SIT,
    "alert_down": BehaviorClass.ALERT_DOWN,
    "apprehend": BehaviorClass.APPREHEND,
    "escort": BehaviorClass.ESCORT,
    "obstacle": BehaviorClass.OBSTACLE,
    "recall": BehaviorClass.RECALL,
    "watch": BehaviorClass.WATCH,
}


def _save_behaviors(db, video_id: int, episodes, fps: float) -> None:
    """写入 behaviors 表.

    Phase 3.1e: 根据 episode.metadata["detector"] 判断检测器类型
    """
    # 先删除旧行（重试场景）
    db.query(Behavior).filter(Behavior.video_id == video_id).delete()
    db.commit()

    objs = []
    for ep in episodes:
        behavior_enum = _BEHAVIOR_STRING_TO_ENUM.get(ep.behavior)
        if behavior_enum is None:
            # 未在枚举中映射的行为跳过（避免写入失败）
            # P2 高级 6 类（guard/release/...）因命名不一致暂未映射
            logger.debug(f"[worker] 跳过未映射行为: {ep.behavior}")
            continue
        start_sec = ep.start_frame / fps if fps > 0 else 0.0
        end_sec = ep.end_frame / fps if fps > 0 else 0.0
        # 检测器类型：从 metadata 读取，默认 RULE
        detector_str = (ep.metadata or {}).get("detector", "rule")
        if detector_str == "stgcn_bc":
            detector = BehaviorDetector.STGCN_BC
        else:
            detector = BehaviorDetector.RULE
        objs.append(Behavior(
            video_id=video_id,
            behavior_class=behavior_enum,
            detector=detector,
            start_frame=ep.start_frame,
            end_frame=ep.end_frame,
            start_sec=start_sec,
            end_sec=end_sec,
            confidence=ep.confidence,
        ))
    db.add_all(objs)


# 注意: health_check 任务定义在 backend.workers.celery_app 中，此处不再重复。
