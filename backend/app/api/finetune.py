"""数据飞轮 API（Phase 2.1e + P1-2 审核导出链路）.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 2.1e / P1-2

数据飞轮闭环（P1-2）:
    审核 episode → POST /finetune/export（16 类受限导出）
    → POST /finetune/trigger（Celery CPU 微调）
    → MLModel(is_active=False) → 人工 POST /api/models/{id}/activate

端点:
    POST /api/finetune/export                  导出受限数据集（ADMIN/MANAGER）
    GET  /api/finetune/exports                 导出历史（ADMIN 全量 / MANAGER 仅自己）
    POST /api/finetune/trigger                 触发 Celery 微调（ADMIN/MANAGER）
    GET  /api/finetune/tasks/{task_id}         查询 Celery 任务（Redis registry）
    GET  /api/finetune/status                  飞轮状态（兼容旧客户端）
    GET  /api/finetune/pipeline                飞轮概览

BREAKING CHANGE（P1-2）:
    ``POST /api/finetune/trigger`` 的 body 现为 ``Optional``，且必须提供
    ``export_id``；旧客户端 ``POST {}`` 从 202/412 → **400**。
    subprocess 后台路径（``_run_finetune_background``）已删除。

工程验收声明:
    本链路只验证“审核→导出→训练→注册”可运行。合成数据仅用于工程验收，
    **工程验收 ≠ 生产验收**，不代表识别精度或生产可用；不设精度门禁。

REQUIRES scripts/finetune_from_annotations.py update（行 79/101）:
    该 legacy 脚本仍用 ``t.status='completed'`` 与
    ``annotation_type='keypoint'`` 过滤。P1-2 迁移把枚举规范化为成员名
    （COMPLETED / BEHAVIOR），脚本将查不到任何行（导出为空）。
    新链路不依赖该脚本；脚本不在本线白名单，列为发布阻断项，交 scripts owner。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Callable, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.core.deps import get_current_handler, require_roles
from backend.app.models.annotation import AnnotationTask, AnnotationTaskStatus
from backend.app.models.handler import Handler, UserRole
from backend.app.models.ml_model import MLModel, ModelType
from backend.app.services.training_export import (
    DISCLAIMER,
    EXPORT_MAX_EPISODES,
    ExportError,
    ExportLimitExceeded,
    exports_root,
    get_sync_session_factory,
    resolve_export_dir,
    run_export,
    validate_export_assets,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/finetune",
    tags=["finetune"],
    dependencies=[Depends(get_current_handler)],
)
AdminOrManager = Annotated[Handler, Depends(require_roles(UserRole.ADMIN, UserRole.MANAGER))]

DbSession = Annotated[AsyncSession, Depends(get_db)]
SyncFactory = Annotated[Callable[[], Session], Depends(get_sync_session_factory)]


class FinetuneTriggerRequest(BaseModel):
    """触发 Celery 微调请求（P1-2 breaking change：export_id 必填）."""

    export_id: Optional[str] = Field(default=None, description="POST /finetune/export 返回的 ID")
    epochs: int = Field(default=20, ge=1, le=200)
    batch_size: int = Field(default=8, ge=1, le=64)
    T: int = Field(default=30, ge=4, le=300)
    base_channels: int = Field(default=64, ge=4, le=256)
    num_stages: int = Field(default=10, ge=1, le=20)
    device: str = Field(default="cpu")


class FinetuneExportRequest(BaseModel):
    """导出请求（data_source 必填，不做默认猜测）."""

    task_ids: Optional[list[int]] = None
    include_unreviewed: bool = False
    data_source: str = Field(description="synthetic | real（必填）")
    seed: int = 42


class FinetuneStatusResponse(BaseModel):
    """微调状态响应（字段兼容旧客户端）."""

    running: bool = False
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    last_result: Optional[dict] = None
    error: Optional[str] = None
    pipeline: dict = {}


# ============================================================
# Celery / Redis 工具
# ============================================================

REDIS_REGISTRY_PREFIX = "k9:finetune:task:"


def _redis_client():
    """同步 Redis 客户端（trigger 内做 broker 探测 + registry 写入）."""
    import redis

    return redis.Redis.from_url(settings.celery_broker_url, decode_responses=True)


def _probe_broker() -> None:
    """探测 broker（max_retries=0, timeout=3）；失败抛异常."""
    from backend.workers.celery_app import celery_app

    with celery_app.connection_for_write() as conn:
        conn.ensure_connection(max_retries=0, timeout=3)


def _registry_set(task_id: str, payload: dict) -> None:
    try:
        client = _redis_client()
        client.setex(
            f"{REDIS_REGISTRY_PREFIX}{task_id}",
            7 * 24 * 3600,
            json.dumps(payload, ensure_ascii=False),
        )
    except Exception as exc:  # registry 写失败只能降级为 PENDING
        logger.warning("finetune task registry 写入失败（降级 PENDING）: %s", exc)


def _registry_get(task_id: str) -> Optional[dict]:
    try:
        client = _redis_client()
        raw = client.get(f"{REDIS_REGISTRY_PREFIX}{task_id}")
    except Exception as exc:
        logger.warning("finetune task registry 读取失败: %s", exc)
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ============================================================
# 导出
# ============================================================


@router.post("/export", status_code=status.HTTP_201_CREATED)
async def export_training_data(
    payload: FinetuneExportRequest,
    sync_factory: SyncFactory,
    me: AdminOrManager,
) -> dict:
    """导出 16 类受限数据集（ADMIN/MANAGER）.

    - 只导出 ``data_json`` 审核覆盖中 ``decision != rejected`` 的 episode；
      ``include_unreviewed=false``（默认）时未审核不导出
    - 按 video_id 分组划分 train/val（防泄漏）
    - 单次上限 ``EXPORT_MAX_EPISODES=500``（超出 413）
    - 产物: train.pkl / val.pkl / episodes.jsonl / manifest.json
    """
    try:
        result = await run_in_threadpool(
            run_export,
            session_factory=sync_factory,
            task_ids=payload.task_ids,
            include_unreviewed=payload.include_unreviewed,
            data_source=payload.data_source,
            seed=payload.seed,
            created_by_handler_id=me.id,
            requester=me,
        )
    except ExportLimitExceeded as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@router.get("/exports")
async def list_exports(
    sync_factory: SyncFactory,
    me: AdminOrManager,
) -> dict:
    """列出导出记录（ADMIN 全量；MANAGER 仅 created_by_handler_id=me）."""

    def _list(factory: Callable[[], Session]) -> dict:
        root = exports_root()
        items: list[dict] = []
        if root.is_dir():
            for manifest_path in sorted(root.glob("*/manifest.json"), reverse=True):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    me.role != UserRole.ADMIN
                    and manifest.get("created_by_handler_id") != me.id
                ):
                    continue
                items.append(
                    {
                        "export_id": manifest.get("export_id"),
                        "created_at": manifest.get("created_at"),
                        "created_by_handler_id": manifest.get("created_by_handler_id"),
                        "data_source": manifest.get("data_source"),
                        "stats": manifest.get("stats"),
                        "engineering_only": manifest.get("engineering_only", True),
                        "production_accepted": manifest.get("production_accepted", False),
                    }
                )
        return {"items": items, "total": len(items)}

    return await run_in_threadpool(_list, sync_factory)


# ============================================================
# 触发微调
# ============================================================


def _load_owned_manifest(export_id: str, me: Handler) -> dict:
    """加载 manifest 并校验归属（MANAGER 只能触发自己导出的数据）."""
    try:
        export_dir = resolve_export_dir(export_id)
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"导出 {export_id} 不存在")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"manifest 读取失败: {exc}")

    if me.role != UserRole.ADMIN and manifest.get("created_by_handler_id") != me.id:
        raise HTTPException(status_code=403, detail="无权触发他人导出的微调任务")

    try:
        validate_export_assets(export_dir, manifest)
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return manifest


@router.post("/trigger", status_code=status.HTTP_202_ACCEPTED)
async def trigger_finetune(
    sync_factory: SyncFactory,
    me: AdminOrManager,
    payload: Optional[FinetuneTriggerRequest] = Body(default=None),
) -> dict:
    """触发 Celery CPU 微调（ADMIN/MANAGER）.

    BREAKING CHANGE: body 必须含 ``export_id``；旧客户端 ``POST {}`` → 400。
    """
    if payload is None or not payload.export_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "缺少 export_id（P1-2 breaking change）：请先 POST /api/finetune/export，"
                "再把返回的 export_id 传给本端点"
            ),
        )

    manifest = _load_owned_manifest(payload.export_id, me)

    try:
        await run_in_threadpool(_probe_broker)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Celery broker 不可用，无法派发微调任务: {exc}",
        )

    from backend.workers.finetune_tasks import train_from_annotations

    def _dispatch():
        return train_from_annotations.delay(
            export_id=payload.export_id,
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            T=payload.T,
            base_channels=payload.base_channels,
            num_stages=payload.num_stages,
            device=payload.device,
        )

    async_result = await run_in_threadpool(_dispatch)
    task_id = str(async_result.id)
    await run_in_threadpool(
        _registry_set,
        task_id,
        {
            "task_id": task_id,
            "export_id": payload.export_id,
            "created_by_handler_id": me.id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state": "PENDING",
        },
    )
    logger.info(
        "微调已派发: task_id=%s export_id=%s by=%s", task_id, payload.export_id, me.id
    )
    return {
        "status": "accepted",
        "task_id": task_id,
        "export_id": payload.export_id,
        "query": f"GET /api/finetune/tasks/{task_id}",
        "data_source": manifest.get("data_source"),
        "engineering_only": True,
        "production_accepted": False,
        "disclaimer": DISCLAIMER,
        "note": (
            "worker 未运行时 .delay 可能入队但状态停留 PENDING；"
            "registry 缺失时 /tasks/{task_id} 返回 404 以区分“不存在”。"
        ),
    }


@router.get("/tasks/{task_id}")
async def get_finetune_task(task_id: str, me: AdminOrManager) -> dict:
    """查询 Celery 任务（ADMIN 全量；MANAGER 仅自己的）.

    区分“不存在”（registry 无记录 → 404）与“排队/执行中”（PENDING/STARTED）。
    """
    registry = await run_in_threadpool(_registry_get, task_id)
    if registry is None:
        raise HTTPException(status_code=404, detail=f"微调任务 {task_id} 不存在")
    if me.role != UserRole.ADMIN and registry.get("created_by_handler_id") != me.id:
        raise HTTPException(status_code=403, detail="无权查看他人触发的任务")

    from backend.workers.celery_app import celery_app

    def _probe_result() -> dict:
        result = celery_app.AsyncResult(task_id)
        try:
            state = result.state
            ready = result.ready()
            successful = result.successful() if ready else False
        except Exception as exc:  # result backend 不可用 → 降级
            return {
                "state": "UNKNOWN",
                "ready": False,
                "successful": False,
                "result": {"error": f"result backend 不可用: {exc}"},
            }
        payload: dict = {}
        if successful:
            try:
                payload = result.result or {}
            except Exception:
                payload = {}
        elif result.failed():
            payload = {"error": str(result.result)}
        return {
            "state": state,
            "ready": ready,
            "successful": successful,
            "result": payload,
        }

    probe = await run_in_threadpool(_probe_result)

    return {
        "task_id": task_id,
        "export_id": registry.get("export_id"),
        "state": probe["state"],
        "ready": probe["ready"],
        "successful": probe["successful"],
        "result": probe["result"],
        "created_at": registry.get("created_at"),
        "engineering_only": True,
        "production_accepted": False,
        "disclaimer": DISCLAIMER,
    }


# ============================================================
# 兼容端点：/status + /pipeline
# ============================================================

_finetune_status: dict = {
    # 兼容字段：P1-2 后 subprocess 路径已删除，running 永远 false
    "running": False,
    "started_at": None,
    "finished_at": None,
    "last_result": None,
    "error": None,
}


@router.get("/status", response_model=FinetuneStatusResponse)
async def get_finetune_status(db: DbSession) -> FinetuneStatusResponse:
    """飞轮状态（兼容旧客户端）.

    注意: ``running`` 永远 false（subprocess 路径已删除）；真实任务状态请查
    ``GET /api/finetune/tasks/{task_id}``。
    """
    pipeline: dict = {}

    r = await db.execute(text("SELECT COUNT(*) FROM videos"))
    pipeline["videos"] = r.scalar() or 0

    # 用 ORM 聚合；键显式取 .value 保持旧小写契约
    r = await db.execute(
        select(AnnotationTask.status, func.count()).group_by(AnnotationTask.status)
    )
    pipeline["annotation_tasks"] = {
        status_obj.value if hasattr(status_obj, "value") else str(status_obj): count
        for status_obj, count in r.all()
    }

    r = await db.execute(text("SELECT COUNT(*) FROM annotations"))
    pipeline["annotations"] = r.scalar() or 0

    r = await db.execute(
        select(MLModel)
        .where(MLModel.type.in_([ModelType.POSE, ModelType.BEHAVIOR]))
        .order_by(MLModel.id.desc())
        .limit(10)
    )
    models = r.scalars().all()
    pipeline["models"] = [
        {
            "id": m.id,
            "version": m.version,
            "type": m.type.value,
            "is_active": m.is_active,
            "metrics": m.metrics_json,
        }
        for m in models
    ]
    pipeline["models_by_type"] = {
        t.value: sum(1 for m in models if m.type == t)
        for t in (ModelType.POSE, ModelType.BEHAVIOR)
    }

    return FinetuneStatusResponse(
        running=False,
        started_at=_finetune_status["started_at"],
        finished_at=_finetune_status["finished_at"],
        last_result=_finetune_status["last_result"],
        error=_finetune_status["error"],
        pipeline=pipeline,
    )


@router.get("/pipeline")
async def get_pipeline_overview(db: DbSession) -> dict:
    """数据飞轮完整概览（前端仪表盘用）."""
    r = await db.execute(text("SELECT COUNT(*) FROM videos"))
    total_videos = r.scalar() or 0

    r = await db.execute(
        select(func.count())
        .select_from(AnnotationTask)
        .where(AnnotationTask.status == AnnotationTaskStatus.COMPLETED)
    )
    completed_tasks = r.scalar() or 0

    r = await db.execute(
        select(func.count())
        .select_from(MLModel)
        .where(MLModel.type.in_([ModelType.POSE, ModelType.BEHAVIOR]))
    )
    total_models = r.scalar() or 0

    r = await db.execute(
        select(func.count())
        .select_from(MLModel)
        .where(
            MLModel.type.in_([ModelType.POSE, ModelType.BEHAVIOR]),
            MLModel.is_active.is_(True),
        )
    )
    active_models = r.scalar() or 0

    return {
        "stages": {
            "upload": {
                "total_videos": total_videos,
                "description": "视频上传",
            },
            "annotate": {
                "completed_tasks": completed_tasks,
                "description": "人工审核（behavior episode 覆盖）",
            },
            "train": {
                "total_models": total_models,
                "description": "微调训练（POSE + BEHAVIOR）",
            },
            "deploy": {
                "active_models": active_models,
                "description": "模型部署",
            },
        },
        "finetune_running": False,
        "engineering_only": True,
        "production_accepted": False,
        "disclaimer": DISCLAIMER,
        "next_action": _next_action(completed_tasks, active_models),
    }


def _next_action(completed_annotations: int, active_models: int) -> str:
    """根据飞轮状态推荐下一步操作."""
    if completed_annotations == 0:
        return "请先审核 behavior episode 并 POST /api/annotations/tasks/{id}/complete"
    if active_models == 0:
        return "请激活一个模型: POST /api/models/{id}/activate"
    return "可以上传新视频测试: POST /api/videos/upload"


__all__ = [
    "EXPORT_MAX_EPISODES",
    "router",
]
