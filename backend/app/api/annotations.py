"""标注管理 API（数据飞轮基础设施 2.1b + P1-2 审核覆盖层）.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 2.1b / P1-2
依据: dev-docs/stages/phase-2.md §2.1b + ADR 0008

端点:
    POST   /api/annotations/tasks                    创建标注任务
    GET    /api/annotations/tasks                    列出标注任务
    GET    /api/annotations/tasks/{id}               获取标注任务详情
    GET    /api/annotations/tasks/{id}/annotations   获取任务的标注数据
    POST   /api/annotations/sync                     501 — LS 同步未实现（见下）

P1-2 审核端点（data_json 审核覆盖层）:
    GET    /api/annotations/review/queue                     审核队列（按角色过滤）
    GET    /api/annotations/videos/{video_id}/episodes       视频 episode + 审核覆盖
    POST   /api/annotations/episodes/{behavior_id}/review    提交审核（幂等 upsert）
    GET    /api/annotations/videos/{video_id}/review-summary 审核统计
    POST   /api/annotations/tasks/{task_id}/complete         完成任务

审核契约:
    data_json = {
        review_schema_version: 1,
        behavior_id, decision, behavior, start_frame, end_frame,
        original: {behavior, start_frame, end_frame},  # 服务端快照
        note, reviewed_by, reviewed_at,
    }
    标签空间只允许 16 类交集（见 services.training_export.MAPPED_BEHAVIORS）；
    DB-only 的 P2 六类管线未产出，不可审核。

状态机:
    PENDING/IN_PROGRESS → 可审，首次审核置 IN_PROGRESS
    COMPLETED → 再审需 reopen=true，否则 409
    FAILED → 409

说明:
    YouTube 自标 + Label Studio 管线已于 2026-08-19 废弃（ADR 0012），改为
    AimCLR+SMQ+TCL+主动学习。`POST /api/annotations/sync` 保留路由但返回
    **501 Not Implemented**，避免文档/前端继续撞 404。
"""
from __future__ import annotations

import logging
from typing import Annotated, Callable, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.config import settings
from backend.app.core.deps import get_current_handler
from backend.app.models.annotation import (
    Annotation, AnnotationSource, AnnotationTask, AnnotationTaskStatus, AnnotationType,
)
from backend.app.models.behavior import Behavior
from backend.app.models.dog import Dog
from backend.app.models.dog_associations import DogHandlerAssociation
from backend.app.models.handler import Handler
from backend.app.models.video import Video
from backend.app.schemas.common import (
    AnnotationRead, AnnotationTaskCreate, AnnotationTaskRead,
)
from backend.app.services.training_export import (
    MAPPED_BEHAVIORS,
    REVIEW_SCHEMA_VERSION,
    get_sync_session_factory,
    review_access_decision,
    review_payload,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/annotations",
    tags=["annotations"],
    dependencies=[Depends(get_current_handler)],
)

DbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentHandler = Annotated[Handler, Depends(get_current_handler)]
SyncFactory = Annotated[Callable[[], Session], Depends(get_sync_session_factory)]


class AnnotationSyncRequest(BaseModel):
    """Sync request body（保留 schema 便于 OpenAPI；同步本身未实现）。"""

    task_id: Optional[int] = Field(default=None, description="可选：仅同步指定标注任务")


class AnnotationSyncStubResponse(BaseModel):
    """501 stub response — honest contract for missing LS sync."""

    detail: str
    status: str = "not_implemented"
    available: bool = False
    alternatives: list[str] = Field(default_factory=list)


# ============================================================
# P1-2 审核 schemas
# ============================================================


class EpisodeReviewRequest(BaseModel):
    """单个行为 episode 的审核请求。

    - accepted: 接受原值（behavior/边界字段忽略）
    - rejected: 拒绝（其余字段忽略，仍记录原值）
    - corrected: 改类/改边界，至少提供一项
    - reopen: 任务已 COMPLETED 时需显式 true
    """

    decision: Literal["accepted", "rejected", "corrected"]
    behavior: Optional[str] = Field(default=None, description="生效行为（16 类之一）")
    start_frame: Optional[int] = Field(default=None, ge=0)
    end_frame: Optional[int] = Field(default=None, ge=0)
    note: Optional[str] = Field(default=None, max_length=512)
    reopen: bool = False

    @model_validator(mode="after")
    def _validate_corrected(self) -> "EpisodeReviewRequest":
        if self.decision == "corrected":
            if self.behavior is None and self.start_frame is None and self.end_frame is None:
                raise ValueError(
                    "decision=corrected 时 behavior/start_frame/end_frame 至少提供一项"
                )
        # rejected 时其余字段忽略（不校验标签空间，仍记录原值）
        if (
            self.decision != "rejected"
            and self.behavior is not None
            and self.behavior not in MAPPED_BEHAVIORS
        ):
            raise ValueError(
                f"behavior={self.behavior!r} 不在可审核的 16 类标签空间内"
            )
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.start_frame > self.end_frame
        ):
            raise ValueError("start_frame 不得大于 end_frame")
        return self


# ============================================================
# 同步辅助（run_in_threadpool 内执行）
# ============================================================


class ReviewConflict(Exception):
    """状态机冲突（409）。"""


class ReviewForbidden(Exception):
    """无权限（403）。"""


def _close_session(session: Session) -> None:
    session.close()


def _get_video_context(session: Session, video_id: int) -> tuple[Optional[Video], Optional[Dog]]:
    video = session.get(Video, video_id)
    dog = session.get(Dog, video.dog_id) if video is not None and video.dog_id else None
    return video, dog


def _session_handler_id(video: Video) -> Optional[int]:
    session_obj = video.session
    return getattr(session_obj, "handler_id", None) if session_obj is not None else None


def _association_exists(session: Session, handler_id: int, dog_id: Optional[int]) -> bool:
    if dog_id is None:
        return False
    return (
        session.execute(
            select(DogHandlerAssociation).where(
                DogHandlerAssociation.handler_id == handler_id,
                DogHandlerAssociation.dog_id == dog_id,
            )
        ).scalar_one_or_none()
        is not None
    )


def _assert_access(
    session: Session,
    me: Handler,
    video: Video,
    dog: Optional[Dog],
    *,
    action: str = "read",
) -> None:
    allowed, reason = review_access_decision(
        me,
        video,
        dog,
        _session_handler_id(video),
        _association_exists(session, me.id, video.dog_id),
        action=action,
    )
    if not allowed:
        raise ReviewForbidden(reason)


def _get_or_create_task(session: Session, video_id: int) -> AnnotationTask:
    """懒创建标注任务（MUSTFIX-5）.

    PG 下 ``get(Video, with_for_update=True)`` 行锁串行化并发创建。
    SQLite 测试不覆盖并发语义。
    """
    video = session.execute(
        select(Video).where(Video.id == video_id).with_for_update()
    ).scalar_one_or_none()
    if video is None:
        raise ReviewConflict(f"视频 {video_id} 不存在")

    task = session.execute(
        select(AnnotationTask)
        .where(AnnotationTask.video_id == video_id)
        .order_by(AnnotationTask.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    if task is not None:
        return task

    total_frames: Optional[int] = None
    if video.fps and video.duration_sec:
        total_frames = int(video.fps * video.duration_sec)
    if not total_frames:
        max_idx = session.execute(
            select(Behavior.start_frame).where(Behavior.video_id == video_id)
        ).scalars().all()
        if max_idx:
            total_frames = int(max(max_idx)) + 1

    task = AnnotationTask(
        video_id=video_id,
        handler_id=video.handler_id,
        status=AnnotationTaskStatus.IN_PROGRESS,
        total_frames=total_frames,
        annotation_types=["behavior"],
    )
    session.add(task)
    session.flush()
    logger.info("懒创建标注任务: task_id=%d video_id=%d", task.id, video_id)
    return task


def _behavior_annotations(session: Session, task_id: int) -> list[Annotation]:
    return list(
        session.execute(
            select(Annotation)
            .where(
                Annotation.task_id == task_id,
                Annotation.annotation_type == AnnotationType.BEHAVIOR,
            )
            .order_by(Annotation.id)
        ).scalars().all()
    )


def _find_review(session: Session, task_id: int, behavior_id: int) -> Optional[Annotation]:
    for row in _behavior_annotations(session, task_id):
        data = row.data_json or {}
        if (
            data.get("review_schema_version") == REVIEW_SCHEMA_VERSION
            and data.get("behavior_id") == behavior_id
        ):
            return row
    return None


def _parse_review(row: Annotation) -> dict:
    return dict(row.data_json or {})


def _effective_frames(reviews: list[Annotation]) -> int:
    """非 rejected 审核覆盖的 distinct 帧数（区间并集）."""
    intervals: list[tuple[int, int]] = []
    for row in reviews:
        data = row.data_json or {}
        if data.get("review_schema_version") != REVIEW_SCHEMA_VERSION:
            continue
        if data.get("decision") == "rejected":
            continue
        start = data.get("start_frame")
        end = data.get("end_frame")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if end < start:
            continue
        intervals.append((start, end))

    if not intervals:
        return 0
    intervals.sort()
    total = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end + 1:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start + 1
            cur_start, cur_end = start, end
    total += cur_end - cur_start + 1
    return total


def _episode_payload(behavior: Behavior, review: Optional[Annotation]) -> dict:
    original = {
        "behavior": behavior.behavior_class.value,
        "start_frame": int(behavior.start_frame),
        "end_frame": int(behavior.end_frame),
    }
    if review is not None:
        data = _parse_review(review)
        effective = {
            "behavior": data.get("behavior", original["behavior"]),
            "start_frame": data.get("start_frame", original["start_frame"]),
            "end_frame": data.get("end_frame", original["end_frame"]),
        }
        return {
            "behavior_id": int(behavior.id),
            "video_id": int(behavior.video_id),
            "reviewed": True,
            "decision": data.get("decision"),
            "note": data.get("note"),
            "reviewed_by": data.get("reviewed_by"),
            "reviewed_at": data.get("reviewed_at"),
            "review_schema_version": data.get("review_schema_version"),
            "original": original,
            "effective": effective,
            # 兼容前端平铺字段
            **effective,
            "confidence": float(behavior.confidence or 0.0),
            "detector": behavior.detector.value,
            "mappable": original["behavior"] in MAPPED_BEHAVIORS,
        }
    return {
        "behavior_id": int(behavior.id),
        "video_id": int(behavior.video_id),
        "reviewed": False,
        "decision": None,
        "note": None,
        "reviewed_by": None,
        "reviewed_at": None,
        "review_schema_version": None,
        "original": original,
        "effective": dict(original),
        **original,
        "confidence": float(behavior.confidence or 0.0),
        "detector": behavior.detector.value,
        "mappable": original["behavior"] in MAPPED_BEHAVIORS,
    }


# ============================================================
# 同步实现
# ============================================================


def _queue_sync(factory: Callable[[], Session], me: Handler, limit: int, offset: int) -> dict:
    session = factory()
    try:
        tasks = list(
            session.execute(
                select(AnnotationTask)
                .order_by(AnnotationTask.id.desc())
                .limit(limit)
                .offset(offset)
            ).scalars().all()
        )
        items: list[dict] = []
        task_video_ids: set[int] = set()
        for task in tasks:
            task_video_ids.add(task.video_id)
            video, dog = _get_video_context(session, task.video_id)
            if video is None:
                continue
            allowed, reason = review_access_decision(
                me,
                video,
                dog,
                _session_handler_id(video),
                _association_exists(session, me.id, video.dog_id),
                action="read",
            )
            if not allowed:
                continue

            behaviors = list(
                session.execute(
                    select(Behavior).where(Behavior.video_id == video.id)
                ).scalars().all()
            )
            reviews = [
                r for r in _behavior_annotations(session, task.id)
                if (r.data_json or {}).get("review_schema_version") == REVIEW_SCHEMA_VERSION
            ]
            decisions = [(r.data_json or {}).get("decision") for r in reviews]
            items.append(
                {
                    "task_id": int(task.id),
                    "video_id": int(video.id),
                    "video_filename": video.original_filename,
                    "scene": video.scene,
                    "status": task.status.value,
                    "progress": float(task.progress),
                    "total_frames": task.total_frames,
                    "annotated_frames": task.annotated_frames,
                    "episodes_total": len(behaviors),
                    "episodes_reviewed": len(reviews),
                    "episodes_pending": max(0, len(behaviors) - len(reviews)),
                    "accepted": decisions.count("accepted"),
                    "rejected": decisions.count("rejected"),
                    "corrected": decisions.count("corrected"),
                    "access_reason": reason,
                }
            )

        # 正常 pipeline 视频可能从未创建 AnnotationTask（懒创建）。
        # 这类视频也必须有入口可审核，因此把它们作为 task_id=None 的队列项列出。
        video_stmt = (
            select(Video)
            .where(Video.id.in_(select(Behavior.video_id).distinct()))
            .order_by(Video.id.desc())
        )
        untasked = [
            v for v in session.execute(video_stmt).scalars().all()
            if v.id not in task_video_ids
        ]
        for video in untasked:
            dog = session.get(Dog, video.dog_id) if video.dog_id else None
            allowed, reason = review_access_decision(
                me,
                video,
                dog,
                _session_handler_id(video),
                _association_exists(session, me.id, video.dog_id),
                action="read",
            )
            if not allowed:
                continue
            count = session.execute(
                select(Behavior).where(Behavior.video_id == video.id)
            ).scalars().all()
            items.append(
                {
                    "task_id": None,
                    "video_id": int(video.id),
                    "video_filename": video.original_filename,
                    "scene": video.scene,
                    "status": "no_task",
                    "progress": 0.0,
                    "total_frames": None,
                    "annotated_frames": 0,
                    "episodes_total": len(count),
                    "episodes_reviewed": 0,
                    "episodes_pending": len(count),
                    "accepted": 0,
                    "rejected": 0,
                    "corrected": 0,
                    "access_reason": reason,
                }
            )

        return {"items": items, "total": len(items)}
    finally:
        _close_session(session)


def _episodes_sync(factory: Callable[[], Session], me: Handler, video_id: int) -> dict:
    session = factory()
    try:
        video, dog = _get_video_context(session, video_id)
        if video is None:
            raise LookupError(f"视频 {video_id} 不存在")
        _assert_access(session, me, video, dog, action="read")

        task = _get_existing_task(session, video_id)
        reviews = _behavior_annotations(session, task.id) if task is not None else []
        review_map = {
            (r.data_json or {}).get("behavior_id"): r for r in reviews
            if (r.data_json or {}).get("review_schema_version") == REVIEW_SCHEMA_VERSION
        }

        behaviors = list(
            session.execute(
                select(Behavior)
                .where(Behavior.video_id == video_id)
                .order_by(Behavior.start_frame, Behavior.id)
            ).scalars().all()
        )
        return {
            "video_id": int(video_id),
            "task_id": int(task.id) if task is not None else None,
            "task_status": task.status.value if task is not None else None,
            "episodes": [
                _episode_payload(b, review_map.get(b.id)) for b in behaviors
            ],
            "mapped_behaviors": sorted(MAPPED_BEHAVIORS),
            "unmappable_db_only": sorted(
                {b.behavior_class.value for b in behaviors if b.behavior_class.value not in MAPPED_BEHAVIORS}
            ),
        }
    finally:
        _close_session(session)


def _get_existing_task(session: Session, video_id: int) -> Optional[AnnotationTask]:
    return session.execute(
        select(AnnotationTask)
        .where(AnnotationTask.video_id == video_id)
        .order_by(AnnotationTask.id.desc())
        .limit(1)
    ).scalar_one_or_none()


def _review_sync(
    factory: Callable[[], Session],
    me: Handler,
    behavior_id: int,
    payload: EpisodeReviewRequest,
) -> dict:
    session = factory()
    try:
        behavior = session.get(Behavior, behavior_id)
        if behavior is None:
            raise LookupError(f"行为片段 {behavior_id} 不存在")

        video, dog = _get_video_context(session, behavior.video_id)
        if video is None:
            raise LookupError(f"视频 {behavior.video_id} 不存在")
        _assert_access(session, me, video, dog, action="write")

        original_behavior = behavior.behavior_class.value
        if original_behavior not in MAPPED_BEHAVIORS:
            # DB-only P2 类（food_drive 等）管线未产出，UI 标注“不可审核”
            raise ValueError(
                f"行为类 {original_behavior!r} 不在 16 类可审核标签空间内（管线未产出）"
            )

        task = _get_or_create_task(session, video.id)
        if task.status == AnnotationTaskStatus.FAILED:
            raise ReviewConflict(f"任务 {task.id} 状态为 FAILED，不可审核")
        if task.status == AnnotationTaskStatus.COMPLETED:
            if not payload.reopen:
                raise ReviewConflict(
                    f"任务 {task.id} 已完成；如需再审请传 reopen=true"
                )
            task.status = AnnotationTaskStatus.IN_PROGRESS
        elif task.status == AnnotationTaskStatus.PENDING:
            task.status = AnnotationTaskStatus.IN_PROGRESS

        eff_behavior = original_behavior
        eff_start = int(behavior.start_frame)
        eff_end = int(behavior.end_frame)

        if payload.decision == "corrected":
            if payload.behavior is not None:
                eff_behavior = payload.behavior
            if payload.start_frame is not None:
                eff_start = int(payload.start_frame)
            if payload.end_frame is not None:
                eff_end = int(payload.end_frame)
            if (
                eff_behavior == original_behavior
                and eff_start == int(behavior.start_frame)
                and eff_end == int(behavior.end_frame)
            ):
                raise ValueError("corrected 必须与原值不同（behavior/start_frame/end_frame 至少一项）")
        if eff_start > eff_end:
            raise ValueError("start_frame 不得大于 end_frame")

        data = review_payload(
            behavior=behavior,
            decision=payload.decision,
            effective_behavior=eff_behavior,
            effective_start=eff_start,
            effective_end=eff_end,
            reviewed_by=me.id,
            note=payload.note,
        )

        existing = _find_review(session, task.id, behavior.id)
        created = existing is None
        if existing is None:
            existing = Annotation(
                task_id=task.id,
                video_id=video.id,
                frame_idx=None,
                annotation_type=AnnotationType.BEHAVIOR,
                source=AnnotationSource.HUMAN,
                data_json=data,
                confidence=float(behavior.confidence or 0.0),
            )
            session.add(existing)
        else:
            existing.data_json = data
            existing.source = AnnotationSource.HUMAN
        session.flush()

        # 重算 annotated_frames（非 rejected 覆盖的 distinct 帧）
        session.flush()
        all_reviews = _behavior_annotations(session, task.id)
        task.annotated_frames = _effective_frames(all_reviews)
        session.commit()

        return {
            "behavior_id": int(behavior.id),
            "task_id": int(task.id),
            "status": task.status.value,
            "annotated_frames": int(task.annotated_frames),
            "progress": float(task.progress),
            "review": data,
            "created": created,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        _close_session(session)


def _summary_sync(factory: Callable[[], Session], me: Handler, video_id: int) -> dict:
    session = factory()
    try:
        video, dog = _get_video_context(session, video_id)
        if video is None:
            raise LookupError(f"视频 {video_id} 不存在")
        _assert_access(session, me, video, dog, action="read")

        task = _get_existing_task(session, video_id)
        behaviors = list(
            session.execute(
                select(Behavior).where(Behavior.video_id == video_id)
            ).scalars().all()
        )
        reviews = _behavior_annotations(session, task.id) if task is not None else []
        decisions = [
            (r.data_json or {}).get("decision") for r in reviews
            if (r.data_json or {}).get("review_schema_version") == REVIEW_SCHEMA_VERSION
        ]
        behavior_ids = {b.id for b in behaviors}
        reviewed_ids = {
            (r.data_json or {}).get("behavior_id") for r in reviews
            if (r.data_json or {}).get("review_schema_version") == REVIEW_SCHEMA_VERSION
        }
        pending = len([bid for bid in behavior_ids if bid not in reviewed_ids])
        return {
            "video_id": int(video_id),
            "task_id": int(task.id) if task is not None else None,
            "task_status": task.status.value if task is not None else None,
            "total": len(behaviors),
            "accepted": decisions.count("accepted"),
            "rejected": decisions.count("rejected"),
            "corrected": decisions.count("corrected"),
            "pending": pending,
            "annotated_frames": int(task.annotated_frames) if task is not None else 0,
            "progress": float(task.progress) if task is not None else 0.0,
            "mappable": len([b for b in behaviors if b.behavior_class.value in MAPPED_BEHAVIORS]),
            "unmappable": len([b for b in behaviors if b.behavior_class.value not in MAPPED_BEHAVIORS]),
        }
    finally:
        _close_session(session)


def _complete_sync(factory: Callable[[], Session], me: Handler, task_id: int) -> dict:
    session = factory()
    try:
        task = session.get(AnnotationTask, task_id)
        if task is None:
            raise LookupError(f"标注任务 {task_id} 不存在")
        video, dog = _get_video_context(session, task.video_id)
        if video is None:
            raise LookupError(f"视频 {task.video_id} 不存在")
        _assert_access(session, me, video, dog, action="write")

        if task.status == AnnotationTaskStatus.FAILED:
            raise ReviewConflict(f"任务 {task_id} 状态为 FAILED，不可完成")

        reviews = [
            r for r in _behavior_annotations(session, task.id)
            if (r.data_json or {}).get("review_schema_version") == REVIEW_SCHEMA_VERSION
        ]
        non_rejected = [
            r for r in reviews if (r.data_json or {}).get("decision") != "rejected"
        ]
        if not non_rejected:
            raise ValueError("至少需要 1 条非 rejected 的审核才能完成任务")

        task.status = AnnotationTaskStatus.COMPLETED
        task.annotated_frames = _effective_frames(reviews)
        session.commit()
        return {
            "task_id": int(task.id),
            "status": task.status.value,
            "annotated_frames": int(task.annotated_frames),
            "progress": float(task.progress),
            "reviewed_episodes": len(reviews),
            "accepted_or_corrected": len(non_rejected),
        }
    except Exception:
        session.rollback()
        raise
    finally:
        _close_session(session)


# ============================================================
# 标注任务管理（既有端点）
# ============================================================

@router.post("/tasks", response_model=AnnotationTaskRead, status_code=status.HTTP_201_CREATED)
async def create_annotation_task(
    payload: AnnotationTaskCreate,
    db: DbSession,
    me: CurrentHandler,
) -> AnnotationTask:
    """创建标注任务。

    1. 校验视频存在
    2. 创建 AnnotationTask 记录
    3. （已废弃）不再自动在 Label Studio 创建 task
    """
    video = await db.get(Video, payload.video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"视频 {payload.video_id} 不存在")

    task = AnnotationTask(
        video_id=payload.video_id,
        handler_id=payload.handler_id,
        status=AnnotationTaskStatus.PENDING,
        total_frames=int(video.fps * video.duration_sec) if video.fps and video.duration_sec else None,
        annotation_types=payload.annotation_types or ["keypoint", "behavior"],
        ls_project_id=settings.ls_project_id,
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)
    logger.info("创建标注任务: task_id=%d video_id=%d by=%s", task.id, task.video_id, me.id)
    return task


@router.post(
    "/sync",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    response_model=AnnotationSyncStubResponse,
    summary="标注同步（未实现）",
    description=(
        "历史文档宣称可从 Label Studio 同步标注。该能力**未实现**，"
        "且 LS 管线已废弃（ADR 0012）。本路由始终返回 501。"
    ),
)
async def sync_annotations_stub(
    payload: Optional[AnnotationSyncRequest] = None,
) -> AnnotationSyncStubResponse:
    """Honest stub for POST /api/annotations/sync (findings C2)."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "POST /api/annotations/sync 未实现。Label Studio 同步管线已废弃"
            "（ADR 0012，2026-08-19）。替代方案："
            "1) POST /api/annotations/episodes/{behavior_id}/review 审核 behavior 片段；"
            "2) POST /api/finetune/export 导出受限 16 类数据集；"
            "3) 勿再调用本端点。"
        ),
    )


@router.get("/tasks", response_model=list[AnnotationTaskRead])
async def list_annotation_tasks(
    db: DbSession,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    video_id: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[AnnotationTask]:
    """列出标注任务（可按状态/视频筛选）。"""
    stmt = select(AnnotationTask).order_by(AnnotationTask.id.desc()).limit(limit).offset(offset)
    if status_filter:
        try:
            stmt = stmt.where(AnnotationTask.status == AnnotationTaskStatus(status_filter))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"非法 status: {status_filter}")
    if video_id is not None:
        stmt = stmt.where(AnnotationTask.video_id == video_id)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/tasks/{task_id}", response_model=AnnotationTaskRead)
async def get_annotation_task(task_id: int, db: DbSession) -> AnnotationTask:
    """获取标注任务详情。"""
    task = await db.get(AnnotationTask, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"标注任务 {task_id} 不存在")
    return task


@router.get("/tasks/{task_id}/annotations", response_model=list[AnnotationRead])
async def list_task_annotations(
    task_id: int,
    db: DbSession,
    annotation_type: Optional[str] = None,
    source: Optional[str] = None,
) -> list[Annotation]:
    """获取任务下的标注数据。"""
    task = await db.get(AnnotationTask, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"标注任务 {task_id} 不存在")

    stmt = select(Annotation).where(Annotation.task_id == task_id)
    if annotation_type:
        try:
            stmt = stmt.where(Annotation.annotation_type == AnnotationType(annotation_type))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"非法 annotation_type: {annotation_type}")
    if source:
        try:
            stmt = stmt.where(Annotation.source == AnnotationSource(source))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"非法 source: {source}")
    stmt = stmt.order_by(Annotation.frame_idx)
    result = await db.execute(stmt)
    return list(result.scalars().all())


# ============================================================
# P1-2 审核端点
# ============================================================


@router.get("/review/queue")
async def get_review_queue(
    sync_factory: SyncFactory,
    me: CurrentHandler,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """审核队列（按角色/基地过滤；VIEWER 可读不可写）."""
    return await run_in_threadpool(_queue_sync, sync_factory, me, limit, offset)


@router.get("/videos/{video_id}/episodes")
async def list_video_episodes(
    video_id: int,
    sync_factory: SyncFactory,
    me: CurrentHandler,
) -> dict:
    """视频 episode 列表（Behavior 原值 + 审核覆盖后的生效值）."""
    try:
        return await run_in_threadpool(_episodes_sync, sync_factory, me, video_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ReviewForbidden as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ReviewConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/episodes/{behavior_id}/review")
async def review_episode(
    behavior_id: int,
    payload: EpisodeReviewRequest,
    sync_factory: SyncFactory,
    me: CurrentHandler,
) -> dict:
    """提交 episode 审核（幂等 upsert；状态机见模块 docstring）."""
    try:
        return await run_in_threadpool(_review_sync, sync_factory, me, behavior_id, payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ReviewForbidden as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ReviewConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/videos/{video_id}/review-summary")
async def get_review_summary(
    video_id: int,
    sync_factory: SyncFactory,
    me: CurrentHandler,
) -> dict:
    """审核统计（total/accepted/rejected/corrected/pending + 进度）."""
    try:
        return await run_in_threadpool(_summary_sync, sync_factory, me, video_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ReviewForbidden as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ReviewConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/tasks/{task_id}/complete")
async def complete_annotation_task(
    task_id: int,
    sync_factory: SyncFactory,
    me: CurrentHandler,
) -> dict:
    """完成任务（要求 ≥1 条非 rejected 审核；重算 annotated_frames）."""
    try:
        return await run_in_threadpool(_complete_sync, sync_factory, me, task_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ReviewForbidden as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ReviewConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
