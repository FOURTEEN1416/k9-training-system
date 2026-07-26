"""评分查询路由（Phase 0 占位）。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.models.score import Score
from backend.app.schemas.common import ScoreRead

router = APIRouter(prefix="/scores", tags=["scores"])


DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.get("", response_model=list[ScoreRead])
async def list_scores(
    db: DbSession,
    video_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Score]:
    """列出评分。"""
    stmt = select(Score).order_by(Score.id.desc()).limit(limit).offset(offset)
    if video_id is not None:
        stmt = stmt.where(Score.video_id == video_id)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{score_id}", response_model=ScoreRead)
async def get_score(score_id: int, db: DbSession) -> Score:
    """获取单条评分。"""
    score = await db.get(Score, score_id)
    if score is None:
        raise HTTPException(status_code=404, detail=f"Score {score_id} not found")
    return score
