"""视频上传与查询路由（Phase 0 占位，Phase 1 完整实现）。"""

import uuid
from pathlib import Path
from typing import Annotated, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.models.video import Video, VideoStatus
from backend.app.schemas.common import VideoRead

router = APIRouter(prefix="/videos", tags=["videos"])


DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.get("", response_model=list[VideoRead])
async def list_videos(
    db: DbSession,
    dog_id: Optional[int] = None,
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Video]:
    """列出视频。"""
    stmt = select(Video).order_by(Video.id.desc()).limit(limit).offset(offset)
    if dog_id is not None:
        stmt = stmt.where(Video.dog_id == dog_id)
    if status_filter is not None:
        stmt = stmt.where(Video.status == status_filter)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{video_id}", response_model=VideoRead)
async def get_video(video_id: int, db: DbSession) -> Video:
    """获取单个视频元数据。"""
    video = await db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return video


@router.post("/upload", response_model=VideoRead, status_code=status.HTTP_201_CREATED)
async def upload_video(
    db: DbSession,
    file: UploadFile = File(...),
    dog_id: Optional[int] = None,
    handler_id: Optional[int] = None,
) -> Video:
    """上传训练视频。

    Phase 0：仅存储文件 + 写入元数据，不触发推理。
    Phase 1：增加 Celery 任务触发。
    """
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
    )
    db.add(video)
    await db.flush()
    await db.refresh(video)
    return video
