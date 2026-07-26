"""Celery 任务定义。

Phase 0 仅占位。Phase 1 实现推理任务。
"""

from backend.workers.celery_app import celery_app


@celery_app.task(name="inference.ingest_video")
def ingest_video(video_id: int) -> dict:
    """视频推理任务（Phase 1 实现）。

    流程：
    1. 加载视频元数据
    2. YOLO26-pose 24 关键点检测
    3. 规则引擎 + PoseC3D 行为识别
    4. GA-T 3 维评分
    5. 更新视频状态为 completed
    """
    return {"status": "not_implemented", "video_id": video_id}
