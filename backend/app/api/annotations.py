"""标注管理 API（数据飞轮基础设施 2.1b）.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 2.1b
依据: dev-docs/stages/phase-2.md §2.1b + ADR 0008

端点:
    POST   /api/annotations/tasks              创建标注任务（关联 LS）
    GET    /api/annotations/tasks              列出标注任务
    GET    /api/annotations/tasks/{id}         获取标注任务详情
    GET    /api/annotations/tasks/{id}/annotations  获取任务的标注数据
    POST   /api/annotations/sync               从 LS 同步标注结果
    GET    /api/annotations/ls/health          LS 健康检查
    GET    /api/annotations/ls/projects        LS 项目列表
    GET    /api/annotations/ls/tasks           LS 任务列表
"""
from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.config import settings
from backend.app.models.annotation import (
    Annotation, AnnotationSource, AnnotationTask, AnnotationTaskStatus, AnnotationType,
)
from backend.app.models.video import Video
from backend.app.schemas.common import (
    AnnotationRead, AnnotationTaskCreate, AnnotationTaskRead,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/annotations", tags=["annotations"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


# ===== 标注任务管理 =====

@router.post("/tasks", response_model=AnnotationTaskRead, status_code=status.HTTP_201_CREATED)
async def create_annotation_task(
    payload: AnnotationTaskCreate,
    db: DbSession,
) -> AnnotationTask:
    """创建标注任务（关联 Label Studio）。

    1. 校验视频存在
    2. 创建 AnnotationTask 记录
    3. （可选）在 LS 中创建对应 task
    """
    # 校验视频
    video = await db.get(Video, payload.video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"视频 {payload.video_id} 不存在")

    # 创建标注任务
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
    logger.info("创建标注任务: task_id=%d video_id=%d", task.id, task.video_id)
    return task


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
