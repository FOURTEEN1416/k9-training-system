"""评分查询路由.

Owner: 后端开发
Phase: 0 占位 → Phase 2.6b 扩展（按 dog_id 历史查询）

路由:
    GET    /api/scores                       列出评分（可按 video_id 过滤）
    GET    /api/scores/by-dog/{dog_id}       按 dog_id 查询历史评分（2.6b）
    GET    /api/scores/{score_id}            获取单条评分
"""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.models.score import Score
from backend.app.models.training_session import ScoringStandard
from backend.app.models.video import Video
from backend.app.schemas.common import ScoreRead

router = APIRouter(prefix="/scores", tags=["scores"])


DbSession = Annotated[AsyncSession, Depends(get_db)]


# standard 字符串 → ScoringStandard 枚举映射（大小写/分隔符兼容）
_STANDARD_ALIASES = {
    "ga_t": ScoringStandard.GA_T, "ga-t": ScoringStandard.GA_T, "gat": ScoringStandard.GA_T,
    "ga": ScoringStandard.GA_T,
    "uspca": ScoringStandard.USPCA, "us_pca": ScoringStandard.USPCA,
    "fci_igp": ScoringStandard.FCI_IGP, "fci-igp": ScoringStandard.FCI_IGP,
    "fciigp": ScoringStandard.FCI_IGP, "igp": ScoringStandard.FCI_IGP,
    "custom": ScoringStandard.CUSTOM,
}


def _parse_standard(value: str) -> ScoringStandard:
    """将用户传入的 standard 字符串解析为 ScoringStandard 枚举。

    支持: GA-T/ga_t/ga-t/GA_T, USPCA/uspca, FCI-IGP/fci_igp, CUSTOM/custom
    """
    key = value.strip().lower()
    if key in _STANDARD_ALIASES:
        return _STANDARD_ALIASES[key]
    # 直接匹配枚举名/值
    for std in ScoringStandard:
        if key == std.name.lower() or key == std.value.lower():
            return std
    raise HTTPException(
        status_code=400,
        detail=f"非法 standard: {value}，允许: GA-T / USPCA / FCI-IGP / CUSTOM",
    )


@router.get("", response_model=list[ScoreRead])
async def list_scores(
    db: DbSession,
    video_id: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Score]:
    """列出评分。"""
    stmt = select(Score).order_by(Score.id.desc()).limit(limit).offset(offset)
    if video_id is not None:
        stmt = stmt.where(Score.video_id == video_id)
    result = await db.execute(stmt)
    return list(result.scalars().all())


# ============================================================
# Phase 2.6b 历史评分查询 API
# （必须在 /{score_id} 之前，否则 "by-dog" 会被当作 score_id）
# ============================================================


@router.get("/by-dog/{dog_id}", response_model=list[ScoreRead])
async def list_scores_by_dog(
    dog_id: int,
    db: DbSession,
    standard: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Score]:
    """按 dog_id 查询历史评分（Phase 2.6b）。

    用途:
        - 训导员查看某只犬的所有历史评分
        - 对比不同时期的评分变化（对比可视化延后 Phase 3）
        - 按评分标准过滤（如只看 USPCA 评分）

    Args:
        dog_id: 犬只 ID
        standard: 可选，按评分标准过滤（如 "ga_t", "us_pca", "fci_igp"）
        limit: 返回条数上限（默认 100）
        offset: 分页偏移

    Returns:
        评分列表（按创建时间降序，最新在前）
    """
    # Score join Video on video_id, filter by Video.dog_id
    stmt = (
        select(Score)
        .join(Video, Score.video_id == Video.id)
        .where(Video.dog_id == dog_id)
        .order_by(Score.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if standard is not None:
        stmt = stmt.where(Score.standard == _parse_standard(standard))

    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{score_id}", response_model=ScoreRead)
async def get_score(score_id: int, db: DbSession) -> Score:
    """获取单条评分。"""
    score = await db.get(Score, score_id)
    if score is None:
        raise HTTPException(status_code=404, detail=f"Score {score_id} not found")
    return score
