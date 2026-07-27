"""视频上传、查询与状态轮询路由.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 1.4d
依据: dev-docs/stages/phase-1.md §1.4d

接口:
    POST /api/videos/upload          上传视频 + 触发 Celery 推理任务
    GET  /api/videos                 列出视频
    GET  /api/videos/{id}            获取视频元数据
    GET  /api/videos/{id}/status     轮询推理状态
    GET  /api/videos/{id}/report     下载 PDF 报告（completed 后可用）
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.models.video import Video, VideoStatus, VALID_SCENES
from backend.app.schemas.common import VideoRead, VideoStatusRead
from backend.workers.tasks import ingest_video

router = APIRouter(prefix="/videos", tags=["videos"])


DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.get("", response_model=list[VideoRead])
async def list_videos(
    db: DbSession,
    dog_id: Optional[int] = None,
    status_filter: Optional[str] = None,
    scene: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Video]:
    """列出视频。"""
    stmt = select(Video).order_by(Video.id.desc()).limit(limit).offset(offset)
    if dog_id is not None:
        stmt = stmt.where(Video.dog_id == dog_id)
    if status_filter is not None:
        stmt = stmt.where(Video.status == status_filter)
    if scene is not None:
        stmt = stmt.where(Video.scene == scene)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{video_id}", response_model=VideoRead)
async def get_video(video_id: int, db: DbSession) -> Video:
    """获取单个视频元数据。"""
    video = await db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return video


@router.get("/{video_id}/status", response_model=VideoStatusRead)
async def get_video_status(video_id: int, db: DbSession) -> Video:
    """轮询视频推理状态。

    返回 status / scene / error_message / report_path / processed_at。
    前端按 1-2 秒间隔轮询，status == "completed" 后请求 /report 下载 PDF。
    """
    video = await db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return video


@router.get("/{video_id}/report")
async def download_report(video_id: int, db: DbSession):
    """下载 PDF 评分报告。

    仅在 video.status == completed 且 report_path 存在时可用。
    """
    video = await db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    if video.status != VideoStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"视频状态为 {video.status.value}，尚未完成推理",
        )
    if not video.report_path:
        raise HTTPException(status_code=404, detail="报告尚未生成")

    # report_path 相对 PROJECT_ROOT/reports/
    reports_dir = settings.data_dir.parent / "reports"
    pdf_path = reports_dir / video.report_path
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF 文件不存在: {pdf_path}")

    download_name = f"report_{video_id}_{video.original_filename.rsplit('.', 1)[0]}.pdf"
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=download_name,
    )


@router.post("/upload", response_model=VideoRead, status_code=status.HTTP_201_CREATED)
async def upload_video(
    db: DbSession,
    file: UploadFile = File(...),
    dog_id: Optional[int] = Form(default=None),
    handler_id: Optional[int] = Form(default=None),
    scene: str = Form(default="obedience_trial"),
) -> Video:
    """上传训练视频并异步触发推理.

    Args:
        file: 视频文件（mp4/avi/mov/mkv/webm）
        dog_id: 关联犬只 ID（可选）
        handler_id: 关联训导员 ID（可选）
        scene: 测试场景，puppy_selection / obedience_trial（默认科目测评）

    Returns:
        Video 元数据（status=uploaded），前端用 id 轮询 /status
    """
    # 校验场景
    if scene not in VALID_SCENES:
        raise HTTPException(
            status_code=400,
            detail=f"非法 scene: {scene}，允许: {VALID_SCENES}",
        )

    # 校验扩展名
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.upload_allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Extension {suffix} not allowed. Allowed: {settings.upload_allowed_extensions}",
        )

    # 存储文件
    storage_filename = f"{uuid.uuid4().hex}{suffix}"
    storage_path = settings.upload_dir / storage_filename

    # 写入磁盘（流式，避免大文件占内存）
    written = 0
    max_bytes = settings.upload_max_size_mb * 1024 * 1024
    with open(storage_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                f.close()
                storage_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max {settings.upload_max_size_mb} MB",
                )
            f.write(chunk)

    # 写入数据库
    video = Video(
        dog_id=dog_id,
        handler_id=handler_id,
        original_filename=file.filename,
        storage_path=str(storage_path.relative_to(settings.data_dir)),
        storage_filename=storage_filename,
        size_bytes=written,
        status=VideoStatus.UPLOADED,
        scene=scene,
    )
    db.add(video)
    await db.flush()
    await db.refresh(video)

    # 触发 Celery 异步推理任务
    # 注意: .delay 在 Celery worker 未启动时会入队但不报错
    # 前端通过 /status 轮询；若 worker 长期未消费，status 会停在 uploaded
    video_id = video.id
    try:
        ingest_video.delay(video_id)
    except Exception as e:
        # Celery broker 不可用时，标记为 failed 而非让上传接口 500
        video.status = VideoStatus.FAILED
        video.error_message = f"Celery 任务派发失败: {e}"
        video.processed_at = datetime.now(timezone.utc)
        await db.flush()
        await db.refresh(video)

    return video
