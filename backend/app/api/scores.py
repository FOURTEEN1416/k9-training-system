"""评分查询路由.

Owner: 后端开发
Phase: 0 占位 → Phase 2.6b 扩展（按 dog_id 历史查询）→ Phase 3.7a 扩展（多犬对比）

路由:
    GET    /api/scores                       列出评分（可按 video_id 过滤）
    GET    /api/scores/compare               多犬训练历史对比（3.7a）
    GET    /api/scores/by-dog/{dog_id}       按 dog_id 查询历史评分（2.6b）
    GET    /api/scores/{score_id}            获取单条评分
"""

from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.models.dog import Dog
from backend.app.models.score import Score
from backend.app.models.training_session import ScoringStandard
from backend.app.models.video import Video
from backend.app.schemas.common import (
    DogBrief,
    DogScoreSeries,
    DogScoreStats,
    ScoreCompareResponse,
    ScorePoint,
    ScoreRead,
)

router = APIRouter(prefix="/scores", tags=["scores"])


DbSession = Annotated[AsyncSession, Depends(get_db)]


# 7 维评分维度定义：(模型属性名, 中文标签)
_DIMENSIONS: list[tuple[str, str]] = [
    ("accuracy", "准确度"),
    ("response_latency", "响应延迟"),
    ("duration", "保持时长"),
    ("search_efficiency", "搜索效率"),
    ("attention", "注意力"),
    ("courage", "胆量欲望"),
    ("gait_quality", "步态质量"),
]

# 对比查询允许的最大犬只数（防止查询爆炸）
_MAX_COMPARE_DOGS = 10
# 对比查询每犬最多返回评分条数
_MAX_COMPARE_POINTS_PER_DOG = 200


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


def _score_dimensions(score: Score) -> dict[str, float]:
    """提取评分的非空维度（key=模型属性名 → value=分数）。"""
    dims: dict[str, float] = {}
    for attr, _label in _DIMENSIONS:
        val = getattr(score, attr, None)
        if val is not None:
            dims[attr] = float(val)
    return dims


def _compute_trend_slope(points: list[ScorePoint]) -> Optional[float]:
    """计算 overall 随时间变化的线性回归斜率（最小二乘法）。

    Args:
        points: 按时间升序排列的评分点

    Returns:
        斜率（分/天），正=提升，负=下降；点数 < 2 返回 None
    """
    if len(points) < 2:
        return None
    # 用 created_at 相对秒数作为 x，overall 作为 y
    t0 = points[0].created_at
    xs: list[float] = []
    ys: list[float] = []
    for p in points:
        delta = (p.created_at - t0).total_seconds()
        xs.append(delta)
        ys.append(p.overall)
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return None
    # 斜率单位：分/秒，转换为 分/天
    slope_per_sec = (n * sum_xy - sum_x * sum_y) / denom
    return slope_per_sec * 86400.0


def _avg_dimensions(points: list[ScorePoint]) -> dict[str, float]:
    """计算各维度平均值（仅包含至少一个非 None 值的维度）。"""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for p in points:
        for k, v in p.dimensions.items():
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


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
# Phase 3.7a 多犬训练历史对比查询
# （必须在 /{score_id} 之前，否则 "compare" 会被当作 score_id）
# ============================================================


@router.get("/compare", response_model=ScoreCompareResponse)
async def compare_scores(
    db: DbSession,
    dog_ids: str = Query(
        ...,
        description="逗号分隔的犬只 ID 列表，如 `1,2,3`，最多 10 只",
        min_length=1,
    ),
    date_from: Optional[datetime] = Query(
        None, description="起始时间（ISO 8601），含",
    ),
    date_to: Optional[datetime] = Query(
        None, description="结束时间（ISO 8601），含",
    ),
    standard: Optional[str] = Query(
        None, description="评分标准过滤: GA-T / USPCA / FCI-IGP / CUSTOM",
    ),
    limit_per_dog: int = Query(
        _MAX_COMPARE_POINTS_PER_DOG,
        ge=1,
        le=_MAX_COMPARE_POINTS_PER_DOG,
        description="每犬最多返回评分条数",
    ),
) -> ScoreCompareResponse:
    """多犬训练历史对比查询（Phase 3.7a）。

    用途:
        - 对比多只犬的训练进展（折线趋势图）
        - 评估不同犬只的维度优劣势（雷达图）
        - 选拔参考：选拔/比武场景横向对比

    Args:
        dog_ids: 逗号分隔的犬只 ID（1-10 只）
        date_from: 起始时间过滤（按 Score.created_at）
        date_to: 结束时间过滤
        standard: 评分标准过滤
        limit_per_dog: 每犬最多返回评分条数（默认 200）

    Returns:
        ScoreCompareResponse: 犬只列表 + 每犬时序 + 每犬统计 + 维度标签
    """
    # 解析 dog_ids
    raw_ids = [s.strip() for s in dog_ids.split(",") if s.strip()]
    if not raw_ids:
        raise HTTPException(status_code=400, detail="dog_ids 不能为空")
    if len(raw_ids) > _MAX_COMPARE_DOGS:
        raise HTTPException(
            status_code=400,
            detail=f"对比犬只数不能超过 {_MAX_COMPARE_DOGS} 只，当前 {len(raw_ids)}",
        )
    try:
        dog_id_list = [int(s) for s in raw_ids]
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"dog_ids 包含非整数: {raw_ids}",
        ) from e
    if any(d <= 0 for d in dog_id_list):
        raise HTTPException(
            status_code=400,
            detail=f"dog_ids 必须为正整数: {dog_id_list}",
        )
    # 去重保序
    seen: set[int] = set()
    unique_dog_ids: list[int] = []
    for d in dog_id_list:
        if d not in seen:
            seen.add(d)
            unique_dog_ids.append(d)

    # 解析 standard
    standard_enum: Optional[ScoringStandard] = None
    if standard is not None:
        standard_enum = _parse_standard(standard)

    # 1. 查询犬只信息（缺失的犬只跳过并返回 404 提示）
    dog_result = await db.execute(
        select(Dog).where(Dog.id.in_(unique_dog_ids))
    )
    dogs_db = list(dog_result.scalars().all())
    found_ids = {d.id for d in dogs_db}
    missing = [d for d in unique_dog_ids if d not in found_ids]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"犬只不存在: {missing}",
        )
    # 按用户传入顺序排序
    dogs_sorted = sorted(dogs_db, key=lambda d: unique_dog_ids.index(d.id))
    dog_briefs = [
        DogBrief(id=d.id, name=d.name, breed=d.breed) for d in dogs_sorted
    ]

    # 2. 查询每犬的评分（join Video 按 dog_id 过滤）
    stmt = (
        select(Score)
        .join(Video, Score.video_id == Video.id)
        .where(Video.dog_id.in_(unique_dog_ids))
        .order_by(Score.created_at.asc())
        .limit(len(unique_dog_ids) * limit_per_dog)
    )
    if date_from is not None:
        stmt = stmt.where(Score.created_at >= date_from)
    if date_to is not None:
        stmt = stmt.where(Score.created_at <= date_to)
    if standard_enum is not None:
        stmt = stmt.where(Score.standard == standard_enum)

    score_result = await db.execute(stmt)
    all_scores = list(score_result.scalars().all())

    # 3. 按 dog_id 分组（需要再次查询 video.dog_id 映射）
    video_ids = {s.video_id for s in all_scores}
    dog_id_map: dict[int, int] = {}  # video_id → dog_id
    if video_ids:
        vid_result = await db.execute(
            select(Video.id, Video.dog_id).where(Video.id.in_(video_ids))
        )
        for vid, did in vid_result.all():
            if did is not None:
                dog_id_map[vid] = did

    scores_by_dog: dict[int, list[Score]] = {d: [] for d in unique_dog_ids}
    for s in all_scores:
        did = dog_id_map.get(s.video_id)
        if did is not None and did in scores_by_dog:
            scores_by_dog[did].append(s)

    # 4. 构建响应（series + stats）
    series_list: list[DogScoreSeries] = []
    stats_list: list[DogScoreStats] = []
    for dog_brief in dog_briefs:
        dog_scores = scores_by_dog.get(dog_brief.id, [])
        # Score.created_at asc 已排序，限制条数（取最近 limit_per_dog 条）
        if len(dog_scores) > limit_per_dog:
            dog_scores = dog_scores[-limit_per_dog:]

        points: list[ScorePoint] = [
            ScorePoint(
                score_id=s.id,
                video_id=s.video_id,
                created_at=s.created_at,
                overall=s.overall,
                standard=s.standard.value if isinstance(s.standard, ScoringStandard) else str(s.standard),
                dimensions=_score_dimensions(s),
            )
            for s in dog_scores
        ]

        series_list.append(DogScoreSeries(dog=dog_brief, points=points))

        if points:
            overalls = [p.overall for p in points]
            avg_dim = _avg_dimensions(points)
            stats = DogScoreStats(
                dog=dog_brief,
                count=len(points),
                avg_overall=sum(overalls) / len(overalls),
                max_overall=max(overalls),
                min_overall=min(overalls),
                latest_overall=overalls[-1],  # points 按时间升序，最后一条为最新
                trend_slope=_compute_trend_slope(points),
                avg_dimensions=avg_dim,
            )
        else:
            stats = DogScoreStats(
                dog=dog_brief,
                count=0,
                avg_overall=0.0,
                max_overall=0.0,
                min_overall=0.0,
                latest_overall=None,
                trend_slope=None,
                avg_dimensions={},
            )
        stats_list.append(stats)

    dimension_labels = {attr: label for attr, label in _DIMENSIONS}

    return ScoreCompareResponse(
        dogs=dog_briefs,
        series=series_list,
        stats=stats_list,
        dimension_labels=dimension_labels,
        date_from=date_from,
        date_to=date_to,
        standard=standard_enum.value if standard_enum is not None else None,
    )


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
