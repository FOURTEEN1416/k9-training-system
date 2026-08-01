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
    AnnotationRead, AnnotationTaskCreate, AnnotationTaskRead, LabelStudioSyncResult,
)
from backend.app.services.label_studio import LabelStudioClient, LabelStudioError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/annotations", tags=["annotations"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


def get_ls_client() -> LabelStudioClient:
    """Label Studio 客户端依赖。"""
    return LabelStudioClient()


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


# ===== Label Studio 同步 =====

@router.post("/sync", response_model=LabelStudioSyncResult)
async def sync_from_label_studio(
    db: DbSession,
    project_id: int = Query(default=None, description="LS project ID（默认用配置）"),
) -> LabelStudioSyncResult:
    """从 Label Studio 同步标注结果到本地 DB。

    1. 调用 LS API 获取已完成标注
    2. 写入/更新本地 AnnotationTask + Annotation 表
    3. 更新任务进度
    """
    ls_project_id = project_id or settings.ls_project_id
    client = get_ls_client()

    try:
        tasks = client.list_tasks(ls_project_id, page_size=500)
    except LabelStudioError as e:
        raise HTTPException(status_code=502, detail=f"LS API 错误: {e}")

    synced_tasks = 0
    synced_annotations = 0
    completed_tasks = 0
    errors: list[str] = []

    for ls_task in tasks:
        ls_task_id = ls_task.get("id")
        annotations = ls_task.get("annotations", [])
        if not annotations:
            continue

        completed_tasks += 1

        # 尝试匹配本地 AnnotationTask（通过 ls_task_id）
        stmt = select(AnnotationTask).where(AnnotationTask.ls_task_id == ls_task_id)
        result = await db.execute(stmt)
        local_task = result.scalar_one_or_none()

        if local_task is None:
            # 未关联本地任务，跳过（可后续扩展自动创建）
            continue

        # 同步标注数据
        for ann in annotations:
            for result_item in ann.get("result", []):
                try:
                    ann_type = _infer_annotation_type(result_item)
                    if ann_type is None:
                        continue

                    data_json = _extract_annotation_data(result_item)
                    if data_json is None:
                        continue

                    frame_idx = result_item.get("value", {}).get("frameIndex")

                    # 避免重复写入（同 task + frame + type）
                    existing = await db.execute(
                        select(Annotation).where(
                            Annotation.task_id == local_task.id,
                            Annotation.annotation_type == ann_type,
                            Annotation.frame_idx == frame_idx,
                        )
                    )
                    if existing.scalar_one_or_none() is not None:
                        continue

                    annotation = Annotation(
                        task_id=local_task.id,
                        video_id=local_task.video_id,
                        frame_idx=frame_idx,
                        annotation_type=ann_type,
                        source=AnnotationSource.HUMAN,
                        data_json=data_json,
                    )
                    db.add(annotation)
                    synced_annotations += 1
                except Exception as e:
                    errors.append(f"task {ls_task_id} result: {e}")

        # 更新本地任务状态
        local_task.status = AnnotationTaskStatus.COMPLETED
        local_task.annotated_frames = len(annotations)
        synced_tasks += 1

    await db.flush()
    return LabelStudioSyncResult(
        project_id=ls_project_id,
        synced_tasks=synced_tasks,
        synced_annotations=synced_annotations,
        completed_tasks=completed_tasks,
        errors=errors,
    )


def _infer_annotation_type(result_item: dict) -> Optional[AnnotationType]:
    """从 LS result 推断标注类型。"""
    from_name = result_item.get("from_name", "")
    type_name = result_item.get("type", "")
    value = result_item.get("value", {})

    if "keypoint" in from_name or "keypoints" in value:
        return AnnotationType.KEYPOINT
    if "behavior" in from_name or "behaviorlabels" in value or "choices" in value:
        return AnnotationType.BEHAVIOR
    if "bbox" in from_name or "rectanglelabels" in value:
        return AnnotationType.BBOX
    return None


def _extract_annotation_data(result_item: dict) -> Optional[dict]:
    """从 LS result 提取标注数据。"""
    value = result_item.get("value", {})
    from_name = result_item.get("from_name", "")

    if "keypoints" in value:
        return {"keypoints": value["keypoints"]}
    if "rectanglelabels" in value:
        return {
            "x": value.get("x"), "y": value.get("y"),
            "width": value.get("width"), "height": value.get("height"),
            "labels": value.get("rectanglelabels", []),
        }
    if "choices" in value:
        return {"behaviors": value["choices"]}
    if "labels" in value:
        return {"labels": value["labels"]}
    return None


# ===== Label Studio 代理端点 =====

@router.get("/ls/health")
async def ls_health() -> dict:
    """Label Studio 健康检查。"""
    client = get_ls_client()
    try:
        ok = client.health()
        return {"healthy": ok, "url": settings.ls_url}
    except LabelStudioError as e:
        return {"healthy": False, "url": settings.ls_url, "error": str(e)}


@router.get("/ls/projects")
async def ls_list_projects() -> list[dict]:
    """列出 LS 项目。"""
    client = get_ls_client()
    try:
        return client.list_projects()
    except LabelStudioError as e:
        raise HTTPException(status_code=502, detail=f"LS API 错误: {e}")


@router.get("/ls/tasks")
async def ls_list_tasks(
    project_id: int = Query(default=None),
    page_size: int = 100,
) -> list[dict]:
    """列出 LS 任务。"""
    ls_project_id = project_id or settings.ls_project_id
    client = get_ls_client()
    try:
        return client.list_tasks(ls_project_id, page_size=page_size)
    except LabelStudioError as e:
        raise HTTPException(status_code=502, detail=f"LS API 错误: {e}")
