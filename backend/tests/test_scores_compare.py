"""Phase 3.7a 训练历史对比 API 单元测试.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 3.7a

测试矩阵:
    1. 路由注册验证（不需要 DB）
    2. 辅助函数验证（_compute_trend_slope / _avg_dimensions / _score_dimensions / _parse_standard）
    3. compare_scores 端点逻辑（mock AsyncSession，验证参数解析/错误处理/响应构建）

标记: fast（纯 Python + mock，无 DB/GPU 依赖）
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.api.scores import (
    _avg_dimensions,
    _compute_trend_slope,
    _MAX_COMPARE_DOGS,
    _parse_standard,
    _score_dimensions,
    compare_scores,
)
from backend.app.models.training_session import ScoringStandard
from backend.app.schemas.common import ScorePoint

pytestmark = pytest.mark.fast


# ============================================================
# 1. 路由注册验证
# ============================================================


class TestRouteRegistration:
    """scores 路由注册验证（不需要 DB）。"""

    def test_compare_route_registered(self) -> None:
        """/scores/compare 端点已注册。"""
        from backend.app.api.scores import router

        paths = {r.path for r in router.routes}
        assert "/scores/compare" in paths

    def test_compare_route_before_score_id(self) -> None:
        """/scores/compare 必须在 /{score_id} 之前（FastAPI 按定义顺序匹配）。

        若顺序错误，"compare" 会被当作 score_id 解析为 int 失败返回 422。
        """
        from backend.app.api.scores import router

        paths_in_order = [r.path for r in router.routes if hasattr(r, "path")]
        compare_idx = next(i for i, p in enumerate(paths_in_order) if "compare" in p)
        score_id_idx = next(
            i for i, p in enumerate(paths_in_order) if p == "/scores/{score_id}"
        )
        assert compare_idx < score_id_idx, (
            f"/compare 必须在 /{{score_id}} 之前（当前 compare@{compare_idx}, "
            f"score_id@{score_id_idx}）"
        )

    def test_all_score_routes_registered(self) -> None:
        """所有 scores 路由都已注册。"""
        from backend.app.api.scores import router

        paths = {r.path for r in router.routes}
        assert "/scores" in paths
        assert "/scores/compare" in paths
        assert "/scores/by-dog/{dog_id}" in paths
        assert "/scores/{score_id}" in paths

    def test_main_app_includes_scores_router(self) -> None:
        """main app 已注册 scores 路由（含 compare 端点）。"""
        try:
            from backend.app.main import app
        except ModuleNotFoundError as e:
            if "celery" in str(e):
                pytest.skip(f"跳过: {e}")
            raise

        all_paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/api/scores" in all_paths
        assert "/api/scores/compare" in all_paths


# ============================================================
# 2. 辅助函数验证
# ============================================================


def _make_point(
    dt: datetime,
    overall: float,
    dimensions: dict[str, float] | None = None,
) -> ScorePoint:
    """构造测试用 ScorePoint。"""
    return ScorePoint(
        score_id=0,
        video_id=0,
        created_at=dt,
        overall=overall,
        standard="GA-T",
        dimensions=dimensions or {},
    )


class TestComputeTrendSlope:
    """_compute_trend_slope 线性回归斜率计算测试。"""

    def test_empty_points_returns_none(self) -> None:
        assert _compute_trend_slope([]) is None

    def test_single_point_returns_none(self) -> None:
        p = _make_point(datetime(2026, 1, 1, tzinfo=timezone.utc), 80.0)
        assert _compute_trend_slope([p]) is None

    def test_positive_trend(self) -> None:
        """评分递增 → 正斜率。"""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        points = [
            _make_point(t0, 60.0),
            _make_point(t0 + timedelta(days=1), 70.0),
            _make_point(t0 + timedelta(days=2), 80.0),
        ]
        slope = _compute_trend_slope(points)
        assert slope is not None
        assert slope == pytest.approx(10.0, abs=0.01)  # +10 分/天

    def test_negative_trend(self) -> None:
        """评分递减 → 负斜率。"""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        points = [
            _make_point(t0, 80.0),
            _make_point(t0 + timedelta(days=1), 70.0),
            _make_point(t0 + timedelta(days=2), 60.0),
        ]
        slope = _compute_trend_slope(points)
        assert slope is not None
        assert slope == pytest.approx(-10.0, abs=0.01)  # -10 分/天

    def test_stable_trend_near_zero(self) -> None:
        """评分稳定 → 斜率接近 0。"""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        points = [
            _make_point(t0, 75.0),
            _make_point(t0 + timedelta(days=1), 75.0),
            _make_point(t0 + timedelta(days=2), 75.0),
        ]
        slope = _compute_trend_slope(points)
        assert slope is not None
        assert abs(slope) < 0.01

    def test_same_timestamp_returns_none(self) -> None:
        """所有点时间戳相同 → denom=0，返回 None。"""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        points = [
            _make_point(t0, 60.0),
            _make_point(t0, 70.0),
            _make_point(t0, 80.0),
        ]
        assert _compute_trend_slope(points) is None

    def test_unit_is_per_day(self) -> None:
        """斜率单位是 分/天（不是 分/秒）。"""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        points = [
            _make_point(t0, 60.0),
            _make_point(t0 + timedelta(days=10), 80.0),  # 10 天涨 20 分
        ]
        slope = _compute_trend_slope(points)
        assert slope is not None
        assert slope == pytest.approx(2.0, abs=0.01)  # 2 分/天


class TestAvgDimensions:
    """_avg_dimensions 维度平均值计算测试。"""

    def test_empty_points_returns_empty(self) -> None:
        assert _avg_dimensions([]) == {}

    def test_single_point(self) -> None:
        p = _make_point(
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            80.0,
            {"accuracy": 90.0, "attention": 70.0},
        )
        avg = _avg_dimensions([p])
        assert avg == {"accuracy": 90.0, "attention": 70.0}

    def test_multi_points_same_dims(self) -> None:
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        points = [
            _make_point(t0, 80.0, {"accuracy": 80.0, "attention": 80.0}),
            _make_point(t0 + timedelta(days=1), 90.0, {"accuracy": 90.0, "attention": 90.0}),
            _make_point(t0 + timedelta(days=2), 70.0, {"accuracy": 70.0, "attention": 70.0}),
        ]
        avg = _avg_dimensions(points)
        assert avg["accuracy"] == pytest.approx(80.0)
        assert avg["attention"] == pytest.approx(80.0)

    def test_partial_dimensions_skipped(self) -> None:
        """不同点包含不同维度时，只对有值的点求平均。"""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        points = [
            _make_point(t0, 80.0, {"accuracy": 80.0, "attention": 80.0}),
            _make_point(t0 + timedelta(days=1), 90.0, {"accuracy": 90.0}),  # 无 attention
            _make_point(t0 + timedelta(days=2), 70.0, {"accuracy": 70.0, "attention": 70.0}),
        ]
        avg = _avg_dimensions(points)
        # accuracy 三个值平均
        assert avg["accuracy"] == pytest.approx(80.0)
        # attention 两个值平均
        assert avg["attention"] == pytest.approx(75.0)


class TestScoreDimensions:
    """_score_dimensions 评分维度提取测试。"""

    def test_full_dimensions(self) -> None:
        """7 维全有值时全部提取。"""
        from backend.app.models.score import Score
        from backend.app.models.training_session import ScoringStandard

        # 用 SimpleNamespace 模拟 Score 对象，避免 ORM 触发
        score = SimpleNamespace(
            accuracy=90.0,
            response_latency=85.0,
            duration=80.0,
            search_efficiency=75.0,
            attention=88.0,
            courage=70.0,
            gait_quality=82.0,
            standard=ScoringStandard.FCI_IGP,
        )
        dims = _score_dimensions(score)
        assert set(dims.keys()) == {
            "accuracy", "response_latency", "duration",
            "search_efficiency", "attention", "courage", "gait_quality",
        }
        assert dims["accuracy"] == 90.0
        assert dims["courage"] == 70.0

    def test_partial_dimensions(self) -> None:
        """Phase 1 评分仅 3 维（其余 None）→ 只返回非 None 维度。"""
        score = SimpleNamespace(
            accuracy=90.0,
            response_latency=None,
            duration=80.0,
            search_efficiency=None,
            attention=88.0,
            courage=None,
            gait_quality=None,
        )
        dims = _score_dimensions(score)
        assert set(dims.keys()) == {"accuracy", "duration", "attention"}

    def test_all_none_returns_empty(self) -> None:
        score = SimpleNamespace(
            accuracy=None,
            response_latency=None,
            duration=None,
            search_efficiency=None,
            attention=None,
            courage=None,
            gait_quality=None,
        )
        assert _score_dimensions(score) == {}


class TestParseStandard:
    """_parse_standard 评分标准解析测试。"""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("GA-T", ScoringStandard.GA_T),
            ("ga_t", ScoringStandard.GA_T),
            ("ga-t", ScoringStandard.GA_T),
            ("gat", ScoringStandard.GA_T),
            ("GA", ScoringStandard.GA_T),
            ("USPCA", ScoringStandard.USPCA),
            ("uspca", ScoringStandard.USPCA),
            ("us_pca", ScoringStandard.USPCA),
            ("FCI-IGP", ScoringStandard.FCI_IGP),
            ("fci_igp", ScoringStandard.FCI_IGP),
            ("fciigp", ScoringStandard.FCI_IGP),
            ("igp", ScoringStandard.FCI_IGP),
            ("CUSTOM", ScoringStandard.CUSTOM),
            ("custom", ScoringStandard.CUSTOM),
        ],
    )
    def test_valid_standards(self, value: str, expected: ScoringStandard) -> None:
        assert _parse_standard(value) == expected

    def test_invalid_standard_raises_400(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_standard("UNKNOWN")
        assert exc_info.value.status_code == 400
        assert "非法 standard" in exc_info.value.detail


# ============================================================
# 3. compare_scores 端点逻辑（mock AsyncSession）
# ============================================================


def _make_dog(dog_id: int, name: str, breed: str | None = None) -> SimpleNamespace:
    """构造测试用 Dog 对象（仅需 id/name/breed）。"""
    return SimpleNamespace(id=dog_id, name=name, breed=breed)


def _make_score(
    score_id: int,
    video_id: int,
    overall: float,
    created_at: datetime,
    standard: ScoringStandard = ScoringStandard.GA_T,
    accuracy: float = 80.0,
    response_latency: float | None = None,
    duration: float = 80.0,
    search_efficiency: float | None = None,
    attention: float = 80.0,
    courage: float | None = None,
    gait_quality: float | None = None,
) -> SimpleNamespace:
    """构造测试用 Score 对象（避免 ORM）。"""
    return SimpleNamespace(
        id=score_id,
        video_id=video_id,
        overall=overall,
        created_at=created_at,
        standard=standard,
        accuracy=accuracy,
        response_latency=response_latency,
        duration=duration,
        search_efficiency=search_efficiency,
        attention=attention,
        courage=courage,
        gait_quality=gait_quality,
        scoring_engine_version="1.0.0",
    )


def _make_mock_db(
    dogs: list[SimpleNamespace],
    scores: list[SimpleNamespace],
    video_dog_map: dict[int, int],
) -> AsyncMock:
    """构造 mock AsyncSession，模拟 compare_scores 的 3 次 db.execute 调用。

    调用顺序:
        1. select(Dog).where(Dog.id.in_(...)) → 返回 dogs
        2. select(Score).join(Video)... → 返回 scores
        3. select(Video.id, Video.dog_id) → 返回 video_dog_map 的 items
    """
    db = AsyncMock()

    # 第 1 次：Dog 查询
    dog_result = MagicMock()
    dog_result.scalars.return_value.all.return_value = dogs

    # 第 2 次：Score 查询
    score_result = MagicMock()
    score_result.scalars.return_value.all.return_value = scores

    # 第 3 次：Video.id, Video.dog_id 查询（返回 tuples）
    vid_result = MagicMock()
    vid_result.all.return_value = list(video_dog_map.items())

    db.execute.side_effect = [dog_result, score_result, vid_result]
    return db


async def _call_compare(
    db: AsyncMock,
    dog_ids: str,
    *,
    standard: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    limit_per_dog: int = 200,
):
    """显式传入所有参数调用 compare_scores，避免 FastAPI Query 默认值未解析。

    直接调用端点函数时，未传参的可选参数会保留 Query(None) 对象而非 None，
    需要显式传入 None 来覆盖默认值。
    """
    return await compare_scores(
        db=db,
        dog_ids=dog_ids,
        date_from=date_from,
        date_to=date_to,
        standard=standard,
        limit_per_dog=limit_per_dog,
    )


class TestCompareScoresValidation:
    """compare_scores 参数校验测试（不依赖 DB 查询结果）。"""

    @pytest.mark.asyncio
    async def test_empty_dog_ids_raises_400(self) -> None:
        from fastapi import HTTPException

        db = AsyncMock()
        with pytest.raises(HTTPException) as exc:
            await _call_compare(db=db, dog_ids="")
        assert exc.value.status_code == 400
        assert "dog_ids 不能为空" in exc.value.detail

    @pytest.mark.asyncio
    async def test_only_commas_raises_400(self) -> None:
        from fastapi import HTTPException

        db = AsyncMock()
        with pytest.raises(HTTPException) as exc:
            await _call_compare(db=db, dog_ids=",,,")
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_non_integer_dog_ids_raises_400(self) -> None:
        from fastapi import HTTPException

        db = AsyncMock()
        with pytest.raises(HTTPException) as exc:
            await _call_compare(db=db, dog_ids="1,abc,3")
        assert exc.value.status_code == 400
        assert "非整数" in exc.value.detail

    @pytest.mark.asyncio
    async def test_non_positive_dog_ids_raises_400(self) -> None:
        from fastapi import HTTPException

        db = AsyncMock()
        with pytest.raises(HTTPException) as exc:
            await _call_compare(db=db, dog_ids="0,-1")
        assert exc.value.status_code == 400
        assert "正整数" in exc.value.detail

    @pytest.mark.asyncio
    async def test_too_many_dogs_raises_400(self) -> None:
        from fastapi import HTTPException

        db = AsyncMock()
        # _MAX_COMPARE_DOGS + 1 只犬
        ids = ",".join(str(i) for i in range(1, _MAX_COMPARE_DOGS + 2))
        with pytest.raises(HTTPException) as exc:
            await _call_compare(db=db, dog_ids=ids)
        assert exc.value.status_code == 400
        assert "不能超过" in exc.value.detail

    @pytest.mark.asyncio
    async def test_invalid_standard_raises_400(self) -> None:
        from fastapi import HTTPException

        db = AsyncMock()
        with pytest.raises(HTTPException) as exc:
            await _call_compare(db=db, dog_ids="1", standard="UNKNOWN")
        assert exc.value.status_code == 400
        assert "非法 standard" in exc.value.detail


class TestCompareScoresLogic:
    """compare_scores 业务逻辑测试（mock DB）。"""

    @pytest.mark.asyncio
    async def test_dog_not_found_raises_404(self) -> None:
        from fastapi import HTTPException

        # 犬只不存在
        db = _make_mock_db(dogs=[], scores=[], video_dog_map={})
        with pytest.raises(HTTPException) as exc:
            await _call_compare(db=db, dog_ids="999")
        assert exc.value.status_code == 404
        assert "犬只不存在" in exc.value.detail

    @pytest.mark.asyncio
    async def test_single_dog_no_scores(self) -> None:
        """单犬无评分 → 返回空 series + count=0 stats。"""
        dog = _make_dog(1, "雷克斯", "马犬")
        db = _make_mock_db(dogs=[dog], scores=[], video_dog_map={})

        resp = await _call_compare(db=db, dog_ids="1")

        assert len(resp.dogs) == 1
        assert resp.dogs[0].id == 1
        assert resp.dogs[0].name == "雷克斯"
        assert resp.dogs[0].breed == "马犬"
        assert len(resp.series) == 1
        assert resp.series[0].points == []
        assert resp.stats[0].count == 0
        assert resp.stats[0].avg_overall == 0.0
        assert resp.stats[0].latest_overall is None
        assert resp.stats[0].trend_slope is None
        assert resp.stats[0].avg_dimensions == {}

    @pytest.mark.asyncio
    async def test_single_dog_with_scores(self) -> None:
        """单犬多条评分 → 时序按时间升序 + 统计正确。"""
        dog = _make_dog(1, "雷克斯", "马犬")
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        scores = [
            _make_score(1, 100, 60.0, t0, accuracy=60.0, attention=60.0),
            _make_score(2, 101, 70.0, t0 + timedelta(days=1), accuracy=70.0, attention=70.0),
            _make_score(3, 102, 80.0, t0 + timedelta(days=2), accuracy=80.0, attention=80.0),
        ]
        db = _make_mock_db(dogs=[dog], scores=scores, video_dog_map={100: 1, 101: 1, 102: 1})

        resp = await _call_compare(db=db, dog_ids="1")

        assert len(resp.series[0].points) == 3
        # 时序升序
        assert resp.series[0].points[0].overall == 60.0
        assert resp.series[0].points[2].overall == 80.0
        # 统计
        stats = resp.stats[0]
        assert stats.count == 3
        assert stats.avg_overall == pytest.approx(70.0)
        assert stats.max_overall == 80.0
        assert stats.min_overall == 60.0
        assert stats.latest_overall == 80.0  # 最新一条
        assert stats.trend_slope == pytest.approx(10.0, abs=0.01)  # +10 分/天
        # 维度平均
        assert stats.avg_dimensions["accuracy"] == pytest.approx(70.0)
        assert stats.avg_dimensions["attention"] == pytest.approx(70.0)

    @pytest.mark.asyncio
    async def test_multi_dogs_comparison(self) -> None:
        """多犬对比 → 每犬独立 series + stats。"""
        dog1 = _make_dog(1, "雷克斯", "马犬")
        dog2 = _make_dog(2, "黑虎", "德牧")
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        scores = [
            _make_score(1, 100, 60.0, t0, accuracy=60.0),
            _make_score(2, 101, 80.0, t0 + timedelta(days=1), accuracy=80.0),
            _make_score(3, 200, 70.0, t0, accuracy=70.0),
            _make_score(4, 201, 90.0, t0 + timedelta(days=1), accuracy=90.0),
        ]
        video_dog_map = {100: 1, 101: 1, 200: 2, 201: 2}
        db = _make_mock_db(dogs=[dog1, dog2], scores=scores, video_dog_map=video_dog_map)

        resp = await _call_compare(db=db, dog_ids="1,2")

        assert len(resp.dogs) == 2
        assert resp.dogs[0].id == 1
        assert resp.dogs[1].id == 2
        # 每犬 2 条评分
        assert len(resp.series[0].points) == 2
        assert len(resp.series[1].points) == 2
        # 各自统计
        assert resp.stats[0].avg_overall == pytest.approx(70.0)  # (60+80)/2
        assert resp.stats[1].avg_overall == pytest.approx(80.0)  # (70+90)/2

    @pytest.mark.asyncio
    async def test_dog_ids_dedup_preserves_order(self) -> None:
        """dog_ids 去重保序（2,1,2,3 → 2,1,3）。"""
        dog2 = _make_dog(2, "黑虎")
        dog1 = _make_dog(1, "雷克斯")
        dog3 = _make_dog(3, "大黄")
        db = _make_mock_db(
            dogs=[dog2, dog1, dog3], scores=[], video_dog_map={}
        )

        resp = await _call_compare(db=db, dog_ids="2,1,2,3")

        # 去重后 3 只犬
        assert len(resp.dogs) == 3
        # 保序：2,1,3
        assert [d.id for d in resp.dogs] == [2, 1, 3]

    @pytest.mark.asyncio
    async def test_standard_filter_passed_through(self) -> None:
        """standard 参数过滤透传。"""
        dog = _make_dog(1, "雷克斯")
        db = _make_mock_db(dogs=[dog], scores=[], video_dog_map={})

        resp = await _call_compare(db=db, dog_ids="1", standard="USPCA")

        assert resp.standard == "USPCA"

    @pytest.mark.asyncio
    async def test_date_range_passed_through(self) -> None:
        """date_from / date_to 透传到响应。"""
        dog = _make_dog(1, "雷克斯")
        db = _make_mock_db(dogs=[dog], scores=[], video_dog_map={})

        date_from = datetime(2026, 1, 1, tzinfo=timezone.utc)
        date_to = datetime(2026, 12, 31, tzinfo=timezone.utc)
        resp = await _call_compare(
            db=db, dog_ids="1", date_from=date_from, date_to=date_to
        )

        assert resp.date_from == date_from
        assert resp.date_to == date_to

    @pytest.mark.asyncio
    async def test_dimension_labels_returned(self) -> None:
        """响应包含 7 维中文标签。"""
        dog = _make_dog(1, "雷克斯")
        db = _make_mock_db(dogs=[dog], scores=[], video_dog_map={})

        resp = await _call_compare(db=db, dog_ids="1")

        assert resp.dimension_labels["accuracy"] == "准确度"
        assert resp.dimension_labels["response_latency"] == "响应延迟"
        assert resp.dimension_labels["duration"] == "保持时长"
        assert resp.dimension_labels["search_efficiency"] == "搜索效率"
        assert resp.dimension_labels["attention"] == "注意力"
        assert resp.dimension_labels["courage"] == "胆量欲望"
        assert resp.dimension_labels["gait_quality"] == "步态质量"

    @pytest.mark.asyncio
    async def test_standard_value_in_score_point(self) -> None:
        """ScorePoint.standard 是字符串（枚举 .value），不是枚举对象。"""
        dog = _make_dog(1, "雷克斯")
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        scores = [
            _make_score(
                1, 100, 80.0, t0, standard=ScoringStandard.FCI_IGP
            ),
        ]
        db = _make_mock_db(dogs=[dog], scores=scores, video_dog_map={100: 1})

        resp = await _call_compare(db=db, dog_ids="1")

        assert resp.series[0].points[0].standard == "FCI-IGP"
