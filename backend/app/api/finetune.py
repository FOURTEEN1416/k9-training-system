"""数据飞轮 API（Phase 2.1e）.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 2.1e
依据: dev-docs/stages/phase-2.md §2.1e

数据飞轮闭环:
    上传视频 → 触发标注 → 同步标注 → 触发微调 → 激活模型 → 回到上传

已有端点（其他路由）:
    POST /api/videos/upload          上传视频（Phase 1.4d）
    POST /api/annotations/tasks      创建标注任务（Phase 2.1b）
    POST /api/annotations/sync       同步 LS 标注（Phase 2.1b）
    POST /api/models/{id}/activate   激活模型（Phase 1.4a）

本路由新增:
    POST /api/finetune/trigger       触发微调训练（异步）
    GET  /api/finetune/status        查询微调状态
    GET  /api/finetune/pipeline      查看完整飞轮状态
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.database import get_db, AsyncSessionLocal
from backend.app.models.ml_model import MLModel, ModelType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/finetune", tags=["finetune"])

DbSession = Annotated[AsyncSession, Depends(get_db)]

# 全局微调状态（单进程内存，重启丢失；生产环境用 Redis 或 DB）
_finetune_status: dict = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "process_pid": None,
    "last_result": None,
    "error": None,
}


class FinetuneTriggerRequest(BaseModel):
    """触发微调请求。"""

    base_model: Optional[str] = None  # 基础模型路径（None = 默认 best.pt）
    epochs: int = 50
    batch: int = 8
    imgsz: int = 640


class FinetuneStatusResponse(BaseModel):
    """微调状态响应。"""

    running: bool
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    last_result: Optional[dict] = None
    error: Optional[str] = None
    pipeline: dict = {}


async def _run_finetune_background(
    base_model: Optional[str],
    epochs: int,
    batch: int,
    imgsz: int,
) -> None:
    """后台执行微调脚本."""
    import asyncio
    from datetime import datetime, timezone

    _finetune_status["running"] = True
    _finetune_status["started_at"] = datetime.now(timezone.utc).isoformat()
    _finetune_status["error"] = None

    try:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[3] / "scripts" / "finetune_from_annotations.py"),
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--imgsz", str(imgsz),
        ]
        if base_model:
            cmd.extend(["--base-model", base_model])

        logger.info("触发微调: %s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _finetune_status["process_pid"] = proc.pid

        stdout, stderr = await proc.communicate()
        _finetune_status["process_pid"] = None

        if proc.returncode == 0:
            _finetune_status["last_result"] = {
                "status": "success",
                "stdout": stdout.decode("utf-8", errors="replace")[-500:],  # 最后 500 字符
            }
            logger.info("微调完成")
        else:
            err = stderr.decode("utf-8", errors="replace")[-500:]
            _finetune_status["error"] = err
            _finetune_status["last_result"] = {"status": "failed", "stderr": err}
            logger.error("微调失败: %s", err)

    except Exception as e:
        _finetune_status["error"] = str(e)
        _finetune_status["last_result"] = {"status": "exception", "error": str(e)}
        logger.exception("微调异常")

    finally:
        _finetune_status["running"] = False
        _finetune_status["finished_at"] = datetime.now(timezone.utc).isoformat()


@router.post("/trigger", status_code=status.HTTP_202_ACCEPTED)
async def trigger_finetune(
    request: FinetuneTriggerRequest,
    background_tasks: BackgroundTasks,
    db: DbSession,
) -> dict:
    """触发数据飞轮微调训练（异步）。

    流程:
        1. 检查是否有已完成的标注数据
        2. 后台运行 finetune_from_annotations.py
        3. 训练完成后自动注册新模型（is_active=False）
        4. 用户通过 /api/models/{id}/activate 激活

    返回 202 Accepted（异步任务已接受）。
    """
    if _finetune_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="微调任务正在运行，请等待完成或查询 /status",
        )

    # 检查是否有已完成标注
    stmt = text("SELECT COUNT(*) FROM annotation_tasks WHERE status = 'completed'")
    result = await db.execute(stmt)
    completed_count = result.scalar() or 0

    if completed_count == 0:
        raise HTTPException(
            status_code=412,
            detail="无已完成标注任务。请先在 Label Studio 标注并调用 POST /api/annotations/sync",
        )

    # 触发后台任务
    background_tasks.add_task(
        _run_finetune_background,
        base_model=request.base_model,
        epochs=request.epochs,
        batch=request.batch,
        imgsz=request.imgsz,
    )

    return {
        "status": "accepted",
        "message": f"微调已触发（{completed_count} 个已完成标注任务）",
        "check_status": "GET /api/finetune/status",
    }


@router.get("/status", response_model=FinetuneStatusResponse)
async def get_finetune_status(db: DbSession) -> FinetuneStatusResponse:
    """查询微调训练状态 + 完整飞轮状态。"""
    # 查询飞轮各环节状态
    pipeline = {}

    # 1. 视频总数
    r = await db.execute(text("SELECT COUNT(*) FROM videos"))
    pipeline["videos"] = r.scalar() or 0

    # 2. 标注任务
    r = await db.execute(text("SELECT status, COUNT(*) FROM annotation_tasks GROUP BY status"))
    pipeline["annotation_tasks"] = {row[0]: row[1] for row in r.all()}

    # 3. 标注数据
    r = await db.execute(text("SELECT COUNT(*) FROM annotations"))
    pipeline["annotations"] = r.scalar() or 0

    # 4. 模型版本
    r = await db.execute(
        select(MLModel).where(MLModel.type == ModelType.POSE).order_by(MLModel.id.desc()).limit(5)
    )
    models = r.scalars().all()
    pipeline["models"] = [
        {
            "id": m.id,
            "version": m.version,
            "is_active": m.is_active,
            "metrics": m.metrics_json,
        }
        for m in models
    ]

    return FinetuneStatusResponse(
        running=_finetune_status["running"],
        started_at=_finetune_status["started_at"],
        finished_at=_finetune_status["finished_at"],
        last_result=_finetune_status["last_result"],
        error=_finetune_status["error"],
        pipeline=pipeline,
    )


@router.get("/pipeline")
async def get_pipeline_overview(db: DbSession) -> dict:
    """数据飞轮完整概览（前端仪表盘用）。"""
    # 各阶段数量
    r = await db.execute(text("SELECT COUNT(*) FROM videos"))
    total_videos = r.scalar() or 0

    r = await db.execute(text("SELECT COUNT(*) FROM annotation_tasks WHERE status = 'completed'"))
    completed_annotations = r.scalar() or 0

    r = await db.execute(text("SELECT COUNT(*) FROM ml_models WHERE type = 'POSE'"))
    total_models = r.scalar() or 0

    r = await db.execute(text("SELECT COUNT(*) FROM ml_models WHERE type = 'POSE' AND is_active = true"))
    active_models = r.scalar() or 0

    return {
        "stages": {
            "upload": {
                "total_videos": total_videos,
                "description": "视频上传",
            },
            "annotate": {
                "completed_tasks": completed_annotations,
                "description": "Label Studio 标注",
            },
            "train": {
                "total_models": total_models,
                "description": "微调训练",
            },
            "deploy": {
                "active_models": active_models,
                "description": "模型部署",
            },
        },
        "finetune_running": _finetune_status["running"],
        "next_action": _next_action(completed_annotations, active_models),
    }


def _next_action(completed_annotations: int, active_models: int) -> str:
    """根据飞轮状态推荐下一步操作。"""
    if _finetune_status["running"]:
        return "微调训练中，请等待完成"
    if completed_annotations == 0:
        return "请在 Label Studio 完成标注后调用 POST /api/annotations/sync"
    if active_models == 0:
        return "请激活一个模型: POST /api/models/{id}/activate"
    return "可以上传新视频测试: POST /api/videos/upload"
