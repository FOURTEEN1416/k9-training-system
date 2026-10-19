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

import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Optional, Sequence

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.core.deps import (
    check_video_access,
    get_current_handler,
    require_roles,
    visible_dog_id_select,
)
from backend.app.models.handler import Handler, UserRole
from backend.app.models.video import Video, VideoStatus, VALID_SCENES
from backend.app.schemas.common import (
    ERROR_CODE_HINTS,
    PROCESSING_STALE_AFTER_SEC,
    ProcessingStatsRead,
    VideoErrorCode,
    VideoRead,
    VideoStatusRead,
    _ensure_aware,
    _is_processing_stale,
)
from backend.workers.tasks import _redact, ingest_video

router = APIRouter(
    prefix="/videos",
    tags=["videos"],
    dependencies=[Depends(get_current_handler)],
)


DbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentHandler = Annotated[Handler, Depends(get_current_handler)]
Writer = Annotated[
    Handler,
    Depends(
        require_roles(
            UserRole.ADMIN,
            UserRole.MANAGER,
            UserRole.HANDLER,
            UserRole.RESEARCHER,
        )
    ),
]


@router.get("", response_model=list[VideoRead])
async def list_videos(
    db: DbSession,
    me: CurrentHandler,
    dog_id: Optional[int] = None,
    status_filter: Optional[str] = None,
    scene: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Video]:
    """列出视频（需 JWT；非 ADMIN 仅见本基地 / 本人关联犬只所属视频）。"""
    stmt = select(Video).order_by(Video.id.desc()).limit(limit).offset(offset)
    visibility = visible_dog_id_select(me)
    if visibility is not None:
        stmt = stmt.where(
            or_(Video.dog_id.in_(visibility), Video.handler_id == me.id)
        )
    if dog_id is not None:
        stmt = stmt.where(Video.dog_id == dog_id)
    if status_filter is not None:
        stmt = stmt.where(Video.status == status_filter)
    if scene is not None:
        stmt = stmt.where(Video.scene == scene)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/ops/stats", response_model=ProcessingStatsRead)
async def get_processing_stats(
    db: DbSession,
    _ops_admin: Annotated[Handler, Depends(require_roles(UserRole.ADMIN, UserRole.MANAGER))],
    days: int = Query(default=7, ge=1, le=90),
) -> ProcessingStatsRead:
    """处理时延与可靠性统计（P1-6，仅 ADMIN/MANAGER）.

    统计口径（响应 method 字段同步说明）:
        - processing: processing_started_at → processed_at（纯处理耗时）
        - e2e:        uploaded_at → processed_at（端到端，含排队）
    样本 < 20 时 reliable=false，前端展示“样本不足”。
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # 1. 窗口内完成的样本（按 processed_at >= since）
    result = await db.execute(
        select(Video).where(Video.processed_at.isnot(None), Video.processed_at >= since)
    )
    window_videos = list(result.scalars().all())

    # 2. stale processing 计数（全表 processing；不依赖 processed_at 索引）
    processing_result = await db.execute(
        select(Video).where(Video.status == VideoStatus.PROCESSING)
    )
    processing_videos = list(processing_result.scalars().all())

    return _summarize_processing_stats(
        window_videos=window_videos,
        processing_videos=processing_videos,
        since=since,
        days=days,
    )


def _percentile_nearest_rank(values: Sequence[float], p: float) -> Optional[int]:
    """nearest-rank 百分位（无插值）；空序列返回 None。"""
    cleaned = sorted(float(v) for v in values if v is not None)
    if not cleaned:
        return None
    rank = max(1, math.ceil(p * len(cleaned)))
    return int(round(cleaned[rank - 1]))


def _summarize_processing_stats(
    *,
    window_videos: Sequence[Video],
    processing_videos: Sequence[Video],
    since: datetime,
    days: int,
    now: Optional[datetime] = None,
) -> ProcessingStatsRead:
    """纯函数：把窗口样本汇总为处理时延统计（双口径并排，不混用）。"""
    completed_videos = [v for v in window_videos if v.status == VideoStatus.COMPLETED]
    failed_videos = [v for v in window_videos if v.status == VideoStatus.FAILED]

    processing_durations = [
        v.processing_duration_ms
        for v in completed_videos
        if v.processing_duration_ms is not None and v.processing_duration_ms >= 0
    ]

    e2e_durations: list[float] = []
    for v in completed_videos:
        if v.uploaded_at is None or v.processed_at is None:
            continue
        uploaded = _ensure_aware(v.uploaded_at)
        processed = _ensure_aware(v.processed_at)
        e2e_ms = (processed - uploaded).total_seconds() * 1000
        if e2e_ms >= 0:
            e2e_durations.append(e2e_ms)

    completed_count = len(completed_videos)
    failed_count = len(failed_videos)
    denominator = completed_count + failed_count
    failure_rate = (failed_count / denominator) if denominator else None

    stale_count = sum(
        1
        for v in processing_videos
        if _is_processing_stale("processing", v.processing_started_at, v.uploaded_at, now)
    )

    return ProcessingStatsRead(
        window_days=days,
        since=since,
        method=(
            "nearest-rank percentile; processing=processing_started_at→processed_at; "
            "e2e=uploaded_at→processed_at (含排队); "
            f"stale=processing 且已运行>{PROCESSING_STALE_AFTER_SEC}s 或无 started_at 且上传已久"
        ),
        sample_size=len(processing_durations),
        p50_duration_ms=_percentile_nearest_rank(processing_durations, 0.50),
        p95_duration_ms=_percentile_nearest_rank(processing_durations, 0.95),
        max_duration_ms=int(max(processing_durations)) if processing_durations else None,
        e2e_sample_size=len(e2e_durations),
        e2e_p50_ms=_percentile_nearest_rank(e2e_durations, 0.50),
        e2e_p95_ms=_percentile_nearest_rank(e2e_durations, 0.95),
        e2e_max_ms=int(max(e2e_durations)) if e2e_durations else None,
        completed=completed_count,
        failed=failed_count,
        failure_rate=failure_rate,
        stale_processing=stale_count,
        reliable=len(processing_durations) >= 20,
    )


@router.get("/{video_id}", response_model=VideoRead)
async def get_video(video_id: int, db: DbSession, me: CurrentHandler) -> Video:
    """获取单个视频元数据（需 JWT；越权返回 404）。"""
    return await check_video_access(db, me, video_id)


@router.get("/{video_id}/status", response_model=VideoStatusRead)
async def get_video_status(video_id: int, db: DbSession, me: CurrentHandler) -> Video:
    """轮询视频推理状态（需 JWT；越权返回 404）。

    返回 status / scene / error_message / report_path / processed_at。
    前端按 1-2 秒间隔轮询，status == "completed" 后请求 /report 下载 PDF。
    """
    return await check_video_access(db, me, video_id)


@router.get("/{video_id}/report")
async def download_report(video_id: int, db: DbSession, me: CurrentHandler):
    """下载 PDF 评分报告（需 JWT；越权返回 404）。

    仅在 video.status == completed 且 report_path 存在时可用。
    """
    video = await check_video_access(db, me, video_id)
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
    me: Writer,
    file: UploadFile = File(...),
    dog_id: Optional[int] = Form(default=None),
    handler_id: Optional[int] = Form(default=None),
    scene: str = Form(default="obedience_trial"),
) -> Video:
    """上传训练视频并异步触发推理（需 JWT；VIEWER 不可上传）.

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
        video.error_message = (
            f"[{VideoErrorCode.TASK_DISPATCH_FAILED.value}] "
            f"{ERROR_CODE_HINTS[VideoErrorCode.TASK_DISPATCH_FAILED]} "
            f"详情: {_redact(e)}"
        )
        video.processed_at = datetime.now(timezone.utc)
        await db.flush()
        await db.refresh(video)

    return video
