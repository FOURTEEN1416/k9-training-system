"""Celery 应用实例。

Phase 0：仅建立实例 + 健康检查任务。
Phase 1：添加推理任务（pose/behavior/scoring）。

Windows 启动命令：
    celery -A backend.workers.celery_app worker -l info --pool solo
"""

from celery import Celery

from backend.app.core.config import settings


celery_app = Celery(
    "k9_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["backend.workers.tasks"],
)

celery_app.conf.update(
    # Windows 兼容
    worker_pool="solo",
    # 序列化
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # 时区
    timezone="Asia/Shanghai",
    enable_utc=True,
    # 任务可见性超时（视频处理耗时长，需加大）
    broker_visibility_timeout=3600,
    # 结果过期
    result_expires=86400,
)


@celery_app.task(name="health.check")
def health_check() -> dict:
    """Celery worker 健康检查任务。"""
    return {"status": "ok", "worker": "celery"}
