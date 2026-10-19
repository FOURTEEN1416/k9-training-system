"""数据飞轮导出服务（P1-2）.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: P1-2（数据飞轮工程链）

职责:
    1. 审核覆盖层（Annotation.data_json, review_schema_version=1）→ 16 类受限导出
    2. 按 video_id 分组划分 train/val，防止同视频泄漏
    3. 生成 pyskl pickle + episodes.jsonl + manifest.json（含 sha256）
    4. 提供 sync session 工厂依赖，供 FastAPI 与 Celery 共用

安全边界:
    - 导出目录必须位于 ``{data_root}/finetune_exports/`` 之下（trusted-local）。
    - pickle 反序列化可执行任意代码；本模块只加载本地可信缓存，
      生产环境建议迁移到 npz/parquet（见 risks）。
    - 本链路只做工程验收：合成数据仅验证链路可运行，
      **工程验收 ≠ 生产验收**，不代表识别精度或生产可用。

标签空间（MUSTFIX-4）:
    DB ``BehaviorClass`` 与 ML ``ALL_BEHAVIORS_22`` 的 P2 六类命名不一致
    （DB: food_drive/courage/attention/search_efficiency/impulse_control/gait；
    ML: guard/release/retrieve/jump/scale/search_blind）。
    本模块只允许两者交集的 16 类，显式映射见 ``MAPPED_BEHAVIORS``。
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import PROJECT_ROOT, settings
from backend.app.models.annotation import (
    Annotation,
    AnnotationTask,
    AnnotationType,
)
from backend.app.models.behavior import Behavior, BehaviorClass
from backend.app.models.dog import Dog
from backend.app.models.dog_associations import DogHandlerAssociation
from backend.app.models.handler import Handler, UserRole
from backend.app.models.keypoint import NUM_KEYPOINTS, Keypoint
from backend.app.models.training_session import TrainingSession  # noqa: F401  (relationship registry)
from backend.app.models.video import Video
from backend.ml.behavior.constants import ALL_BEHAVIORS_22
from backend.ml.behavior.stgcn_bc.labels import BEHAVIOR_TO_IDX, NUM_BEHAVIORS

logger = logging.getLogger(__name__)

REVIEW_SCHEMA_VERSION = 1
EXPORT_MAX_EPISODES = 500
EXPORT_SUBDIR = "finetune_exports"

DISCLAIMER = (
    "本导出/微调链路仅用于工程验收：合成数据仅验证审核→导出→训练→注册链路可运行，"
    "工程验收 ≠ 生产验收，不代表识别精度或生产可用。模型 is_active=False，"
    "激活须由人通过 POST /api/models/{id}/activate 决定。"
)

# ============================================================
# 16 类受限标签空间（DB 值 → ML 名称）
# ============================================================
# 显式列出，禁止隐式 set 交集推导；DB 与 ML 在 16 类上名称一致。
MAPPED_BEHAVIORS: dict[str, str] = {
    # P0 基础 8 类
    "sit": "sit",
    "down": "down",
    "stand": "stand",
    "heel": "heel",
    "sit_up": "sit_up",
    "stay": "stay",
    "bark": "bark",
    "bite": "bite",
    # P1 训练专项 8 类
    "track": "track",
    "alert_sit": "alert_sit",
    "alert_down": "alert_down",
    "apprehend": "apprehend",
    "escort": "escort",
    "obstacle": "obstacle",
    "recall": "recall",
    "watch": "watch",
}

DB_BEHAVIOR_VALUES: set[str] = {b.value for b in BehaviorClass}
ML_BEHAVIOR_VALUES: set[str] = set(ALL_BEHAVIORS_22)
DB_ONLY_BEHAVIORS: set[str] = DB_BEHAVIOR_VALUES - ML_BEHAVIOR_VALUES
ML_ONLY_BEHAVIORS: set[str] = ML_BEHAVIOR_VALUES - DB_BEHAVIOR_VALUES


class ExportError(ValueError):
    """导出业务错误（映射为 HTTP 400）。"""


class ExportLimitExceeded(ExportError):
    """单次导出超过 EXPORT_MAX_EPISODES（映射为 HTTP 413）。"""


# ============================================================
# 依赖注入：sync session 工厂
# ============================================================


def get_sync_session_factory() -> Callable[[], Session]:
    """运行时返回 ``SyncSessionLocal``（不在 import 时绑定实例）.

    这使 FastAPI ``dependency_overrides[get_sync_session_factory]`` 能生效，
    也让 Celery / 测试可以注入自己的 session factory。
    """
    from backend.app.core.database_sync import SyncSessionLocal

    return SyncSessionLocal


# ============================================================
# 审核权限（纯函数，可在无 DB 环境单测）
# ============================================================


def _role_value(handler: Any) -> str:
    role = getattr(handler, "role", None)
    return getattr(role, "value", role) or ""


def _safe_relationship(obj: Any, name: str) -> Any:
    """读取 relationship；分离态/未加载时返回 None（不抛 DetachedInstanceError）."""
    if obj is None:
        return None
    try:
        return getattr(obj, name, None)
    except Exception:  # DetachedInstanceError / lazy load 失败
        return None


def review_access_decision(
    me: Handler,
    video: Video,
    dog: Optional[Dog],
    session_handler_id: Optional[int],
    association_exists: bool,
    *,
    action: str = "read",
) -> tuple[bool, str]:
    """审核/读取权限判定（纯函数）.

    规则（MUSTFIX-6）:
        - ADMIN / is_superuser：全通
        - MANAGER / RESEARCHER / VIEWER：需
          ``dog.home_base_id`` 或 ``video.handler.base_id`` 或
          ``video.session.handler.base_id`` 等于 ``me.base_id``；
          ``me.base_id is None`` 拒绝
        - HANDLER：需 ``video.handler_id == me.id`` 或 ``dog.handler_id == me.id``
          或 DogHandlerAssociation 存在 或 ``session_handler_id == me.id``
        - VIEWER：``action="write"`` 一律拒绝（只读）

    注意:
        ``video.handler_id`` 来自客户端上传 Form，只是“显式指派信号”，
        不是身份证明；NULL 归属视频对 HANDLER 不可见。
    """
    if me is None or video is None:
        return False, "缺少用户或视频上下文"

    role = _role_value(me)
    if getattr(me, "is_superuser", False) or role == UserRole.ADMIN.value:
        return True, "admin"

    if role == UserRole.HANDLER.value:
        me_id = getattr(me, "id", None)
        if me_id is None:
            return False, "用户缺少 id"
        if getattr(video, "handler_id", None) == me_id:
            return True, "video_handler"
        if dog is not None and getattr(dog, "handler_id", None) == me_id:
            return True, "dog_handler"
        if association_exists:
            return True, "dog_handler_association"
        if session_handler_id is not None and session_handler_id == me_id:
            return True, "session_handler"
        return False, "无权访问该视频（非视频/犬只/会话归属人）"

    if role in (UserRole.MANAGER.value, UserRole.RESEARCHER.value, UserRole.VIEWER.value):
        if action == "write" and role == UserRole.VIEWER.value:
            return False, "VIEWER 只读，不能审核"

        base_id = getattr(me, "base_id", None)
        if base_id is None:
            return False, "账号未绑定基地，无权访问该视频"

        if dog is not None and getattr(dog, "home_base_id", None) == base_id:
            return True, "dog_home_base"
        video_handler = _safe_relationship(video, "handler")
        if video_handler is not None and getattr(video_handler, "base_id", None) == base_id:
            return True, "video_handler_base"
        video_session = _safe_relationship(video, "session")
        session_handler = _safe_relationship(video_session, "handler") if video_session else None
        if session_handler is not None and getattr(session_handler, "base_id", None) == base_id:
            return True, "session_handler_base"
        return False, "无权访问其他基地的视频"

    return False, f"未知角色: {role or 'None'}"


# ============================================================
# 路径守卫
# ============================================================


def exports_root(data_root: Optional[Path] = None) -> Path:
    """返回 ``{data_root}/finetune_exports`` 的绝对路径。"""
    root = Path(data_root) if data_root is not None else Path(settings.data_dir)
    return (root / EXPORT_SUBDIR).resolve()


def resolve_export_dir(export_id: str, data_root: Optional[Path] = None) -> Path:
    """解析 export_id → 导出目录，并校验目录守卫.

    Raises:
        ExportError: export_id 含路径分隔/越界。
    """
    if not export_id or Path(export_id).name != export_id:
        raise ExportError(f"非法 export_id: {export_id!r}")
    root = exports_root(data_root)
    export_dir = (root / export_id).resolve()
    if not export_dir.is_relative_to(root):
        raise ExportError("导出目录越界，拒绝访问")
    return export_dir


def validate_export_assets(
    export_dir: Path,
    manifest: dict,
    *,
    data_root: Optional[Path] = None,
) -> dict[str, Path]:
    """校验 manifest 声明的 pkl/jsonl 文件均在导出目录内（防路径穿越）.

    Returns:
        {文件名: 绝对路径}
    """
    root = exports_root(data_root)
    export_dir = Path(export_dir).resolve()
    if not export_dir.is_relative_to(root):
        raise ExportError("导出目录不在 data_root/finetune_exports 下")

    files = manifest.get("files") or {}
    if not isinstance(files, dict) or not files:
        raise ExportError("manifest.files 缺失")

    resolved: dict[str, Path] = {}
    for name in ("train.pkl", "val.pkl", "episodes.jsonl"):
        if name not in files:
            raise ExportError(f"manifest.files 缺少 {name}")
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ExportError(f"非法文件名: {name!r}")
        path = (export_dir / name).resolve()
        if path.parent != export_dir:
            raise ExportError(f"文件越界: {name!r}")
        if not path.exists():
            raise ExportError(f"manifest 声明的文件不存在: {name}")
        resolved[name] = path
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ============================================================
# 关键点提取 + 插值
# ============================================================


def _coerce_keypoints(raw: Any) -> Optional[np.ndarray]:
    """把 keypoints_json 转成 (24, 3) float32；非法返回 None。"""
    try:
        arr = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.shape != (NUM_KEYPOINTS, 3):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _interpolate_clip(arr: np.ndarray, valid: np.ndarray) -> Optional[np.ndarray]:
    """逐关键点/通道线性插值；首尾用最近有效帧，插值帧 conf=0.

    Returns:
        (T, 24, 3) float32，或 None（完全无有效帧）
    """
    total = arr.shape[0]
    if total == 0 or not valid.any():
        return None

    out = arr.copy()
    frame_idx = np.arange(total)
    valid_frames = frame_idx[valid]
    for joint in range(NUM_KEYPOINTS):
        for channel in range(3):
            values = arr[valid, joint, channel]
            out[:, joint, channel] = np.interp(frame_idx, valid_frames, values)
    # 插值帧置信度置 0（保留可解释性）
    out[~valid, :, 2] = 0.0
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _load_clip(
    session: Session,
    video_id: int,
    start_frame: int,
    end_frame: int,
) -> tuple[Optional[np.ndarray], int, float]:
    """读取并插值一个片段的 (T, 24, 3) 关键点.

    Returns:
        (clip 或 None, invalid_rows, coverage)
    """
    start = max(0, int(start_frame))
    end = max(start, int(end_frame))
    total = end - start + 1

    rows = session.execute(
        select(Keypoint)
        .where(
            Keypoint.video_id == video_id,
            Keypoint.frame_idx >= start,
            Keypoint.frame_idx <= end,
        )
        .order_by(Keypoint.frame_idx)
    ).scalars().all()

    arr = np.full((total, NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
    valid = np.zeros(total, dtype=bool)
    invalid_rows = 0
    for row in rows:
        kp = _coerce_keypoints(row.keypoints_json)
        if kp is None:
            invalid_rows += 1
            continue
        idx = int(row.frame_idx) - start
        if 0 <= idx < total:
            arr[idx] = kp
            valid[idx] = True

    coverage = float(valid.sum()) / float(total) if total else 0.0
    if coverage < 0.5:
        return None, invalid_rows, coverage
    return _interpolate_clip(arr, valid), invalid_rows, coverage


# ============================================================
# 审核覆盖层
# ============================================================


def _iter_reviews(session: Session, video_ids: list[int]) -> dict[int, Annotation]:
    """返回 ``{behavior_id: Annotation}``（每 behavior 取最新一条审核）."""
    if not video_ids:
        return {}
    rows = session.execute(
        select(Annotation)
        .where(
            Annotation.video_id.in_(video_ids),
            Annotation.annotation_type == AnnotationType.BEHAVIOR,
        )
        .order_by(Annotation.id)
    ).scalars().all()

    reviews: dict[int, Annotation] = {}
    for row in rows:
        data = row.data_json or {}
        if data.get("review_schema_version") != REVIEW_SCHEMA_VERSION:
            continue
        behavior_id = data.get("behavior_id")
        if isinstance(behavior_id, int):
            reviews[behavior_id] = row
    return reviews


def review_payload(
    *,
    behavior: Behavior,
    decision: str,
    effective_behavior: str,
    effective_start: int,
    effective_end: int,
    reviewed_by: int,
    note: Optional[str] = None,
) -> dict:
    """构造 ``data_json`` 审核覆盖层（服务端快照 original）."""
    return {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "behavior_id": int(behavior.id),
        "decision": decision,
        "behavior": effective_behavior,
        "start_frame": int(effective_start),
        "end_frame": int(effective_end),
        "original": {
            "behavior": behavior.behavior_class.value,
            "start_frame": int(behavior.start_frame),
            "end_frame": int(behavior.end_frame),
        },
        "note": note,
        "reviewed_by": int(reviewed_by),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# 导出主流程
# ============================================================


def _split_by_video(
    video_ids: list[int],
    seed: int,
) -> tuple[set[int], set[int]]:
    """按 video_id 分组随机划分 train/val（至少 2 个视频、两侧非空）."""
    unique = sorted(set(video_ids))
    if len(unique) < 2:
        raise ExportError(
            "至少需要 2 个有可导出 episode 的视频才能划分 train/val（当前 "
            f"{len(unique)} 个）"
        )
    rng = random.Random(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * 0.2)))
    n_val = min(n_val, len(shuffled) - 1)
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])
    return train_ids, val_ids


def _filter_accessible_videos(
    session: Session,
    requester: Optional[Handler],
    video_ids: list[int],
) -> list[int]:
    """按审核权限过滤可导出视频（防 MANAGER 跨基地读取）."""
    if requester is None:
        return video_ids

    allowed_ids: list[int] = []
    for video_id in video_ids:
        video = session.get(Video, video_id)
        if video is None:
            continue
        dog = session.get(Dog, video.dog_id) if video.dog_id else None
        session_obj = video.session
        session_handler_id = getattr(session_obj, "handler_id", None) if session_obj else None
        association = False
        if video.dog_id is not None:
            association = (
                session.execute(
                    select(DogHandlerAssociation).where(
                        DogHandlerAssociation.handler_id == requester.id,
                        DogHandlerAssociation.dog_id == video.dog_id,
                    )
                ).scalar_one_or_none()
                is not None
            )
        allowed, _reason = review_access_decision(
            requester,
            video,
            dog,
            session_handler_id,
            association,
            action="read",
        )
        if allowed:
            allowed_ids.append(video_id)
    return allowed_ids


def run_export(
    *,
    session_factory: Optional[Callable[[], Session]] = None,
    task_ids: Optional[list[int]] = None,
    include_unreviewed: bool = False,
    data_source: str,
    seed: int = 42,
    created_by_handler_id: Optional[int] = None,
    requester: Optional[Handler] = None,
    project_root: Path = PROJECT_ROOT,
    data_root: Optional[Path] = None,
) -> dict:
    """执行一次导出并返回 ``{export_id, export_dir, stats, manifest, disclaimer}``.

    Args:
        session_factory: 同步 session 工厂（None → 运行时 SyncSessionLocal）
        task_ids: 限定标注任务；None → 全体有 behavior 的视频
        include_unreviewed: 是否导出未审核 episode（默认 False）
        data_source: ``synthetic`` | ``real``（必填，不猜测）
        seed: train/val 视频划分随机种子
        created_by_handler_id: 导出发起人
        requester: 请求者（用于基地/归属过滤；None = 不做资源过滤，仅测试/内部）
        project_root: 项目根（路径相对化用）
        data_root: 数据根（默认 settings.data_dir；测试可注入 tmp_path/data）
    """
    from backend.ml.behavior.stgcn_bc.bc_head import generate_boundary_labels
    from backend.ml.behavior.stgcn_bc.data_adapter import keypoints_to_pyskl

    if data_source not in ("synthetic", "real"):
        raise ExportError("data_source 必须显式指定为 'synthetic' 或 'real'")

    root = Path(data_root) if data_root is not None else Path(settings.data_dir)
    factory = session_factory or get_sync_session_factory()

    with factory() as session:
        # 1) 任务/视频范围
        task_stmt = select(AnnotationTask).order_by(AnnotationTask.id)
        if task_ids:
            task_stmt = task_stmt.where(AnnotationTask.id.in_(task_ids))
        tasks = list(session.execute(task_stmt).scalars().all())

        task_statuses: dict[str, str] = {}
        if task_ids:
            video_ids = sorted({t.video_id for t in tasks})
        else:
            video_ids = sorted(
                set(session.execute(select(Behavior.video_id).distinct()).scalars().all())
            )

        if not video_ids:
            raise ExportError("没有可导出的视频")

        video_ids = _filter_accessible_videos(session, requester, video_ids)
        if not video_ids:
            raise ExportError("当前账号无权导出任何视频（基地/归属限制）")

        allowed_tasks = session.execute(
            select(AnnotationTask).where(AnnotationTask.video_id.in_(video_ids))
        ).scalars().all()
        task_statuses = {str(t.id): t.status.value for t in allowed_tasks}

        # 2) 审核覆盖 + 行为
        reviews = _iter_reviews(session, video_ids)
        behaviors = list(
            session.execute(
                select(Behavior)
                .where(Behavior.video_id.in_(video_ids))
                .order_by(Behavior.video_id, Behavior.start_frame, Behavior.id)
            ).scalars().all()
        )
        behavior_ids = {b.id for b in behaviors}
        orphaned = sum(1 for bid in reviews if bid not in behavior_ids)

        stats: dict[str, Any] = {
            "episodes_exported": 0,
            "train_episodes": 0,
            "val_episodes": 0,
            "train_videos": 0,
            "val_videos": 0,
            "label_distribution": {},
            "review_counts": {"accepted": 0, "rejected": 0, "corrected": 0, "pending": 0},
            "skipped": {
                "unmapped_behaviors": 0,
                "orphaned_reviews": orphaned,
                "missing_keypoints": 0,
                "invalid_keypoints": 0,
                "rejected": 0,
                "unreviewed": 0,
                "invalid_boundary": 0,
            },
            "task_statuses": task_statuses,
        }

        # 3) 先统计候选（避免超限时做昂贵的 keypoint 读取）
        candidates: list[tuple[Behavior, dict, str]] = []
        for behavior in behaviors:
            review = reviews.get(behavior.id)
            if review is not None:
                data = review.data_json or {}
                decision = data.get("decision")
                if decision == "rejected":
                    stats["skipped"]["rejected"] += 1
                    stats["review_counts"]["rejected"] += 1
                    continue
                if decision in ("accepted", "corrected"):
                    stats["review_counts"][decision] += 1
                else:
                    stats["review_counts"]["pending"] += 1
                    if not include_unreviewed:
                        stats["skipped"]["unreviewed"] += 1
                        continue
                behavior_value = data.get("behavior") or behavior.behavior_class.value
                start = data.get("start_frame", behavior.start_frame)
                end = data.get("end_frame", behavior.end_frame)
            else:
                stats["review_counts"]["pending"] += 1
                if not include_unreviewed:
                    stats["skipped"]["unreviewed"] += 1
                    continue
                decision = "pending"
                behavior_value = behavior.behavior_class.value
                start = behavior.start_frame
                end = behavior.end_frame

            if behavior_value not in MAPPED_BEHAVIORS:
                stats["skipped"]["unmapped_behaviors"] += 1
                continue
            candidates.append((behavior, {"decision": decision, "start": start, "end": end}, behavior_value))

        if len(candidates) > EXPORT_MAX_EPISODES:
            raise ExportLimitExceeded(
                f"单次导出 episode 数 {len(candidates)} 超过上限 {EXPORT_MAX_EPISODES}，"
                "请缩小 task_ids 范围"
            )
        if not candidates:
            raise ExportError(
                "没有可导出的 episode（审核数据为空；include_unreviewed=false 时需先审核）"
            )

        train_videos, val_videos = _split_by_video(
            [behavior.video_id for behavior, _, _ in candidates], seed
        )

        train_records: list[dict] = []
        val_records: list[dict] = []
        label_counter: Counter[int] = Counter()

        for behavior, review_info, behavior_value in candidates:
            ml_name = MAPPED_BEHAVIORS[behavior_value]
            label = BEHAVIOR_TO_IDX[ml_name]  # 显式 KeyError，禁止 .get(-1)
            if not (0 <= label < NUM_BEHAVIORS):
                raise ExportError(f"非法标签索引: {label}")

            clip, invalid_rows, coverage = _load_clip(
                session, behavior.video_id, review_info["start"], review_info["end"]
            )
            if invalid_rows:
                stats["skipped"]["invalid_keypoints"] += 1
            if clip is None:
                stats["skipped"]["missing_keypoints"] += 1
                continue

            length = int(clip.shape[0])
            boundary_tensor = generate_boundary_labels(length, [0], [length - 1])
            boundary = boundary_tensor.detach().cpu().numpy().astype("float32")
            if boundary.shape != (length,):
                stats["skipped"]["invalid_boundary"] += 1
                continue

            split = "train" if behavior.video_id in train_videos else "val"
            ann = keypoints_to_pyskl(
                clip,
                frame_dir=f"video_{behavior.video_id}_behavior_{behavior.id}",
                label=label,
                label_name=ml_name,
                split=split,
            )
            ann["boundary"] = boundary
            ann["behavior_id"] = int(behavior.id)
            ann["video_id"] = int(behavior.video_id)
            ann["review_decision"] = review_info["decision"]
            ann["start_frame"] = int(review_info["start"])
            ann["end_frame"] = int(review_info["end"])
            ann["coverage"] = round(float(coverage), 4)

            record = {
                "behavior_id": int(behavior.id),
                "video_id": int(behavior.video_id),
                "task_id": _task_id_for_video(tasks, behavior.video_id),
                "label": int(label),
                "label_name": ml_name,
                "decision": review_info["decision"],
                "start_frame": int(review_info["start"]),
                "end_frame": int(review_info["end"]),
                "num_frames": length,
                "coverage": round(float(coverage), 4),
                "split": split,
            }
            (train_records if split == "train" else val_records).append({"ann": ann, "record": record})
            label_counter[label] += 1

        if not train_records or not val_records:
            raise ExportError(
                "train/val 划分后某一侧为空；至少需要 2 个视频且各自产出 episode"
            )

        # 4) 写文件
        export_id = (
            f"exp-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        export_dir = (root / EXPORT_SUBDIR / export_id).resolve()
        if not export_dir.is_relative_to(exports_root(root)):
            raise ExportError("导出目录越界")
        export_dir.mkdir(parents=True, exist_ok=True)

        from backend.ml.behavior.stgcn_bc.data_adapter import save_pyskl_pickle

        train_pkl = export_dir / "train.pkl"
        val_pkl = export_dir / "val.pkl"
        episodes_jsonl = export_dir / "episodes.jsonl"
        save_pyskl_pickle([r["ann"] for r in train_records], str(train_pkl))
        save_pyskl_pickle([r["ann"] for r in val_records], str(val_pkl))
        with open(episodes_jsonl, "w", encoding="utf-8") as f:
            for r in train_records + val_records:
                f.write(json.dumps(r["record"], ensure_ascii=False) + "\n")

        train_video_set = {r["record"]["video_id"] for r in train_records}
        val_video_set = {r["record"]["video_id"] for r in val_records}
        stats.update(
            {
                "episodes_exported": len(train_records) + len(val_records),
                "train_episodes": len(train_records),
                "val_episodes": len(val_records),
                "train_videos": len(train_video_set),
                "val_videos": len(val_video_set),
                "label_distribution": {
                    str(idx): {
                        "name": next(
                            (n for n, i in BEHAVIOR_TO_IDX.items() if i == idx), str(idx)
                        ),
                        "count": count,
                    }
                    for idx, count in sorted(label_counter.items())
                },
                "coverage": {
                    "min": round(min(r["record"]["coverage"] for r in train_records + val_records), 4),
                    "mean": round(
                        sum(r["record"]["coverage"] for r in train_records + val_records)
                        / (len(train_records) + len(val_records)),
                        4,
                    ),
                },
                "data_source": data_source,
            }
        )

        files = {
            "train.pkl": _sha256_file(train_pkl),
            "val.pkl": _sha256_file(val_pkl),
            "episodes.jsonl": _sha256_file(episodes_jsonl),
        }
        manifest = {
            "export_id": export_id,
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by_handler_id": created_by_handler_id,
            "data_source": data_source,
            "engineering_only": True,
            "production_accepted": False,
            "disclaimer": DISCLAIMER,
            "split": {"strategy": "by_video_id", "seed": int(seed)},
            "mapped_behaviors": sorted(MAPPED_BEHAVIORS),
            "stats": stats,
            "files": files,
            "hash_scope": (
                "files 仅包含 train.pkl / val.pkl / episodes.jsonl；"
                "manifest.json 自身不参与自哈希"
            ),
            "num_classes": NUM_BEHAVIORS,
            "data_root": str(root.resolve()),
        }
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info(
        "导出完成: export_id=%s episodes=%d train=%d val=%d",
        export_id,
        stats["episodes_exported"],
        stats["train_episodes"],
        stats["val_episodes"],
    )
    return {
        "export_id": export_id,
        "export_dir": str(export_dir),
        "stats": stats,
        "manifest": manifest,
        "disclaimer": DISCLAIMER,
        "engineering_only": True,
        "production_accepted": False,
    }


def _task_id_for_video(tasks: list[AnnotationTask], video_id: int) -> Optional[int]:
    for task in tasks:
        if task.video_id == video_id:
            return int(task.id)
    return None


__all__ = [
    "DISCLAIMER",
    "DB_ONLY_BEHAVIORS",
    "EXPORT_MAX_EPISODES",
    "ExportError",
    "ExportLimitExceeded",
    "MAPPED_BEHAVIORS",
    "ML_ONLY_BEHAVIORS",
    "REVIEW_SCHEMA_VERSION",
    "exports_root",
    "get_sync_session_factory",
    "resolve_export_dir",
    "review_access_decision",
    "review_payload",
    "run_export",
    "validate_export_assets",
]
