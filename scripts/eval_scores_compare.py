"""Phase 3.7c 训练历史对比 API 端到端评估脚本。

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 3.7c

测试矩阵:
    1. 参数校验: 空 dog_ids / 非整数 / 非正整数 / 超过 10 只 / 非法 standard
    2. 资源存在性: 犬只不存在 → 404
    3. 单犬时序: 多条评分 → series 时序升序 + stats（avg/max/min/latest/trend_slope）
    4. 多犬对比: 3 只犬 → 各自独立 series + stats
    5. 标准过滤: GA-T / USPCA / FCI-IGP 别名解析 + 过滤生效
    6. 日期范围: date_from / date_to 过滤
    7. 空结果: 犬只存在但无评分 → count=0 + latest_overall=None
    8. 去重保序: dog_ids="2,1,2,3" → [2,1,3]
    9. 维度标签: dimension_labels 7 维中文标签

种子数据（带 eval_compare_ 前缀，便于清理）:
    犬只: alpha（马犬）/ beta（德牧）/ gamma（昆明犬）
    视频: 每犬 2 条（共 6 条）
    评分: 每视频 2 条（共 12 条），跨 4 个时间点（2026-01-01/02/03/04）
    评分标准: GA-T + USPCA + FCI-IGP 混合

运行:
    python scripts/eval_scores_compare.py

输出:
    reports/phase-3.7c-scores-compare-eval.json
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2  # noqa: E402
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from backend.app.main import app  # noqa: E402
from backend.app.core.config import settings  # noqa: E402

# ============================================================
# 配置
# ============================================================

TEST_PREFIX = "eval_compare_"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase-3.7c-scores-compare-eval.json"

# 种子时间基准（UTC，避免时区漂移）
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ============================================================
# DB 操作（psycopg2 同步）
# ============================================================


def get_db_conn():
    """从 settings 解析同步 DSN 并连接。"""
    return psycopg2.connect(settings.pg_dsn)


def cleanup_seed_data(conn) -> None:
    """清理旧的测试种子数据（带 eval_compare_ 前缀）。"""
    with conn.cursor() as cur:
        # 删除评分（依赖 video_id）
        cur.execute(
            "DELETE FROM scores WHERE video_id IN "
            "(SELECT id FROM videos WHERE original_filename LIKE %s)",
            (f"{TEST_PREFIX}%",),
        )
        # 删除视频
        cur.execute(
            "DELETE FROM videos WHERE original_filename LIKE %s",
            (f"{TEST_PREFIX}%",),
        )
        # 删除犬只
        cur.execute(
            "DELETE FROM dogs WHERE name LIKE %s",
            (f"{TEST_PREFIX}%",),
        )
    conn.commit()


def seed_data(conn) -> dict[str, list[int]]:
    """创建种子数据，返回 ID 映射。

    种子结构:
        3 只犬: alpha / beta / gamma
        每犬 2 条视频: video_{n}_1 / video_{n}_2
        每视频 2 条评分: 跨 T0 / T0+1d / T0+2d / T0+3d 共 4 个时间点
        评分标准: GA-T / USPCA 混合（让 standard 过滤能区分）
    """
    ids: dict[str, list[int]] = {"dogs": [], "videos": [], "scores": []}

    with conn.cursor() as cur:
        # === 1. 创建 3 只犬 ===
        # 注：PostgreSQL 枚举存储 enum.name（大写）
        dog_seeds = [
            (f"{TEST_PREFIX}alpha", "Belgian Malinois", "MALE", "P1"),
            (f"{TEST_PREFIX}beta", "German Shepherd", "MALE", "P2"),
            (f"{TEST_PREFIX}gamma", "Kunming Dog", "FEMALE", "P0"),
        ]
        for name, breed, gender, stage in dog_seeds:
            cur.execute(
                "INSERT INTO dogs (name, breed, gender, training_stage, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, now(), now()) RETURNING id",
                (name, breed, gender, stage),
            )
            ids["dogs"].append(cur.fetchone()[0])

        # === 2. 每犬 2 条视频（共 6 条） ===
        # 注：VideoStatus 枚举存储 enum.name = "COMPLETED"
        for dog_idx, dog_id in enumerate(ids["dogs"], start=1):
            for v_idx in (1, 2):
                cur.execute(
                    "INSERT INTO videos ("
                    "  dog_id, original_filename, storage_path, storage_filename, "
                    "  status, scene, duration_sec, fps, width, height, "
                    "  uploaded_at, processed_at, created_at, updated_at"
                    ") VALUES ("
                    "  %s, %s, %s, %s, "
                    "  %s, %s, %s, %s, %s, %s, "
                    "  now(), now(), now(), now()"
                    ") RETURNING id",
                    (
                        dog_id,
                        f"{TEST_PREFIX}video_{dog_idx}_{v_idx}.mp4",
                        f"uploads/{TEST_PREFIX}video_{dog_idx}_{v_idx}.mp4",
                        f"{TEST_PREFIX}video_{dog_idx}_{v_idx}.mp4",
                        "COMPLETED",
                        "obedience_trial",
                        30.0,
                        30.0,
                        1920,
                        1080,
                    ),
                )
                ids["videos"].append(cur.fetchone()[0])

        # === 3. 每视频 2 条评分（共 12 条）===
        # 评分跨 4 个时间点（T0 / T0+1d / T0+2d / T0+3d），形成时序
        # 评分标准：GA-T 和 USPCA 交替（便于 standard 过滤测试）
        # overall 趋势：递增（让 trend_slope 为正）
        score_seeds: list[tuple[int, str, datetime, float, float, float, float, float, float, float]] = []

        # alpha 犬（dog_id=ids["dogs"][0]）— video[0] 和 video[1]
        #   video[0] 的 2 条评分: T0 GA-T overall=60, T0+1d USPCA overall=65
        #   video[1] 的 2 条评分: T0+2d GA-T overall=70, T0+3d USPCA overall=75
        alpha_dog = ids["dogs"][0]
        alpha_v1 = ids["videos"][0]  # dog_idx=1, v_idx=1
        alpha_v2 = ids["videos"][1]  # dog_idx=1, v_idx=2
        score_seeds.extend([
            # (video_id, standard, created_at, overall, accuracy, response_latency, duration, search_efficiency, attention, courage, gait_quality)
            (alpha_v1, "GA_T", T0, 60.0, 60.0, 55.0, 60.0, 50.0, 65.0, 58.0, 62.0),
            (alpha_v1, "USPCA", T0 + timedelta(days=1), 65.0, 65.0, 60.0, 65.0, 55.0, 70.0, 62.0, 65.0),
            (alpha_v2, "GA_T", T0 + timedelta(days=2), 70.0, 70.0, 65.0, 70.0, 60.0, 75.0, 68.0, 70.0),
            (alpha_v2, "USPCA", T0 + timedelta(days=3), 75.0, 75.0, 70.0, 75.0, 65.0, 80.0, 72.0, 73.0),
        ])

        # beta 犬（dog_id=ids["dogs"][1]）— video[2] 和 video[3]
        #   评分更高（更优秀的犬），趋势更陡
        beta_dog = ids["dogs"][1]
        beta_v1 = ids["videos"][2]
        beta_v2 = ids["videos"][3]
        score_seeds.extend([
            (beta_v1, "GA_T", T0, 70.0, 70.0, 75.0, 72.0, 68.0, 73.0, 70.0, 71.0),
            (beta_v1, "USPCA", T0 + timedelta(days=1), 78.0, 78.0, 80.0, 76.0, 75.0, 80.0, 76.0, 77.0),
            (beta_v2, "FCI_IGP", T0 + timedelta(days=2), 85.0, 85.0, 88.0, 84.0, 82.0, 86.0, 83.0, 84.0),
            (beta_v2, "USPCA", T0 + timedelta(days=3), 90.0, 90.0, 92.0, 88.0, 87.0, 91.0, 88.0, 89.0),
        ])

        # gamma 犬（dog_id=ids["dogs"][2]）— video[4] 和 video[5]
        #   评分较低（训练中的幼犬），趋势平缓
        gamma_dog = ids["dogs"][2]
        gamma_v1 = ids["videos"][4]
        gamma_v2 = ids["videos"][5]
        score_seeds.extend([
            (gamma_v1, "GA_T", T0, 45.0, 45.0, None, 50.0, None, 48.0, None, None),
            (gamma_v1, "GA_T", T0 + timedelta(days=1), 48.0, 48.0, None, 52.0, None, 50.0, None, None),
            (gamma_v2, "GA_T", T0 + timedelta(days=2), 50.0, 50.0, None, 53.0, None, 52.0, None, None),
            (gamma_v2, "GA_T", T0 + timedelta(days=3), 52.0, 52.0, None, 55.0, None, 54.0, None, None),
        ])

        for (
            video_id, standard, created_at, overall,
            accuracy, response_latency, duration,
            search_efficiency, attention, courage, gait_quality
        ) in score_seeds:
            cur.execute(
                "INSERT INTO scores ("
                "  video_id, standard, accuracy, response_latency, duration, "
                "  search_efficiency, attention, courage, gait_quality, "
                "  overall, scoring_engine_version, created_at, updated_at"
                ") VALUES ("
                "  %s, %s, %s, %s, %s, "
                "  %s, %s, %s, %s, "
                "  %s, %s, %s, %s"
                ") RETURNING id",
                (
                    video_id, standard, accuracy, response_latency, duration,
                    search_efficiency, attention, courage, gait_quality,
                    overall, "1.0.0-eval", created_at, created_at,
                ),
            )
            ids["scores"].append(cur.fetchone()[0])

    conn.commit()
    return ids


# ============================================================
# 测试辅助
# ============================================================


def assert_true(condition: bool, detail: str) -> dict[str, Any]:
    return {"status": "pass" if condition else "fail", "detail": detail}


# ============================================================
# 测试套件（全部 async，共用单一 event loop）
# ============================================================


async def test_parameter_validation(client: httpx.AsyncClient, ids: dict) -> dict[str, Any]:
    """测试 1: 参数校验。"""
    results = {"name": "parameter_validation", "cases": []}
    valid_dog_id = ids["dogs"][0]

    # 1.1 空 dog_ids → 400
    resp = await client.get("/api/scores/compare", params={"dog_ids": ""})
    results["cases"].append({
        "case": "empty_dog_ids_returns_400",
        **assert_true(
            resp.status_code == 422,  # FastAPI Query min_length=1 触发 422
            f"期望 422（min_length=1），实际 {resp.status_code}",
        ),
    })

    # 1.2 只有逗号 → 400
    resp = await client.get("/api/scores/compare", params={"dog_ids": ",,,"})
    results["cases"].append({
        "case": "only_commas_returns_400",
        **assert_true(
            resp.status_code == 400,
            f"期望 400，实际 {resp.status_code} {resp.text[:200]}",
        ),
    })

    # 1.3 非整数 → 400
    resp = await client.get("/api/scores/compare", params={"dog_ids": "1,abc,3"})
    results["cases"].append({
        "case": "non_integer_returns_400",
        **assert_true(
            resp.status_code == 400 and "非整数" in resp.text,
            f"期望 400 + '非整数'，实际 {resp.status_code} {resp.text[:200]}",
        ),
    })

    # 1.4 非正整数 → 400
    resp = await client.get("/api/scores/compare", params={"dog_ids": "0,-1"})
    results["cases"].append({
        "case": "non_positive_returns_400",
        **assert_true(
            resp.status_code == 400 and "正整数" in resp.text,
            f"期望 400 + '正整数'，实际 {resp.status_code} {resp.text[:200]}",
        ),
    })

    # 1.5 超过 10 只 → 400
    too_many = ",".join(str(i) for i in range(1, 12))
    resp = await client.get("/api/scores/compare", params={"dog_ids": too_many})
    results["cases"].append({
        "case": "too_many_dogs_returns_400",
        **assert_true(
            resp.status_code == 400 and "不能超过" in resp.text,
            f"期望 400 + '不能超过'，实际 {resp.status_code} {resp.text[:200]}",
        ),
    })

    # 1.6 非法 standard → 400
    resp = await client.get(
        "/api/scores/compare",
        params={"dog_ids": str(valid_dog_id), "standard": "UNKNOWN"},
    )
    results["cases"].append({
        "case": "invalid_standard_returns_400",
        **assert_true(
            resp.status_code == 400 and "非法 standard" in resp.text,
            f"期望 400 + '非法 standard'，实际 {resp.status_code} {resp.text[:200]}",
        ),
    })

    return results


async def test_dog_not_found(client: httpx.AsyncClient) -> dict[str, Any]:
    """测试 2: 犬只不存在 → 404。"""
    results = {"name": "dog_not_found", "cases": []}

    resp = await client.get("/api/scores/compare", params={"dog_ids": "999999"})
    results["cases"].append({
        "case": "nonexistent_dog_returns_404",
        **assert_true(
            resp.status_code == 404 and "犬只不存在" in resp.text,
            f"期望 404 + '犬只不存在'，实际 {resp.status_code} {resp.text[:200]}",
        ),
    })

    return results


async def test_single_dog_timeseries(client: httpx.AsyncClient, ids: dict) -> dict[str, Any]:
    """测试 3: 单犬时序 + 统计。"""
    results = {"name": "single_dog_timeseries", "cases": []}
    alpha_id = ids["dogs"][0]  # alpha 犬有 4 条评分，overall 递增 60→65→70→75

    resp = await client.get("/api/scores/compare", params={"dog_ids": str(alpha_id)})
    data = resp.json()

    # 3.1 HTTP 200
    results["cases"].append({
        "case": "returns_200",
        **assert_true(resp.status_code == 200, f"期望 200，实际 {resp.status_code}"),
    })

    # 3.2 返回 1 只犬
    results["cases"].append({
        "case": "single_dog_returned",
        **assert_true(
            len(data["dogs"]) == 1 and data["dogs"][0]["id"] == alpha_id,
            f"期望 1 只犬 id={alpha_id}，实际 {data.get('dogs')}",
        ),
    })

    # 3.3 series 4 个点
    series = data["series"][0]
    points = series["points"]
    results["cases"].append({
        "case": "four_score_points",
        **assert_true(
            len(points) == 4,
            f"期望 4 个评分点，实际 {len(points)}",
        ),
    })

    # 3.4 时序升序（60→65→70→75）
    overalls = [p["overall"] for p in points]
    results["cases"].append({
        "case": "ascending_temporal_order",
        **assert_true(
            overalls == sorted(overalls) and overalls == [60.0, 65.0, 70.0, 75.0],
            f"期望 [60,65,70,75]，实际 {overalls}",
        ),
    })

    # 3.5 统计：count=4
    stats = data["stats"][0]
    results["cases"].append({
        "case": "stats_count",
        **assert_true(
            stats["count"] == 4,
            f"期望 count=4，实际 {stats['count']}",
        ),
    })

    # 3.6 统计：avg_overall = (60+65+70+75)/4 = 67.5
    results["cases"].append({
        "case": "stats_avg_overall",
        **assert_true(
            abs(stats["avg_overall"] - 67.5) < 0.01,
            f"期望 avg=67.5，实际 {stats['avg_overall']}",
        ),
    })

    # 3.7 统计：max=75 / min=60
    results["cases"].append({
        "case": "stats_max_min",
        **assert_true(
            stats["max_overall"] == 75.0 and stats["min_overall"] == 60.0,
            f"期望 max=75 min=60，实际 max={stats['max_overall']} min={stats['min_overall']}",
        ),
    })

    # 3.8 统计：latest_overall=75（最后一条）
    results["cases"].append({
        "case": "stats_latest_overall",
        **assert_true(
            stats["latest_overall"] == 75.0,
            f"期望 latest=75，实际 {stats['latest_overall']}",
        ),
    })

    # 3.9 统计：trend_slope 正值（递增趋势）
    results["cases"].append({
        "case": "stats_positive_trend_slope",
        **assert_true(
            stats["trend_slope"] is not None and stats["trend_slope"] > 0,
            f"期望 trend_slope > 0，实际 {stats['trend_slope']}",
        ),
    })

    # 3.10 维度平均值：accuracy avg = (60+65+70+75)/4 = 67.5
    avg_dims = stats["avg_dimensions"]
    results["cases"].append({
        "case": "avg_dimensions_accuracy",
        **assert_true(
            "accuracy" in avg_dims and abs(avg_dims["accuracy"] - 67.5) < 0.01,
            f"期望 accuracy avg=67.5，实际 {avg_dims.get('accuracy')}",
        ),
    })

    # 3.11 维度平均值：attention avg = (65+70+75+80)/4 = 72.5
    results["cases"].append({
        "case": "avg_dimensions_attention",
        **assert_true(
            "attention" in avg_dims and abs(avg_dims["attention"] - 72.5) < 0.01,
            f"期望 attention avg=72.5，实际 {avg_dims.get('attention')}",
        ),
    })

    return results


async def test_multi_dog_comparison(client: httpx.AsyncClient, ids: dict) -> dict[str, Any]:
    """测试 4: 多犬对比。"""
    results = {"name": "multi_dog_comparison", "cases": []}
    alpha_id, beta_id, gamma_id = ids["dogs"]

    resp = await client.get(
        "/api/scores/compare",
        params={"dog_ids": f"{alpha_id},{beta_id},{gamma_id}"},
    )
    data = resp.json()

    # 4.1 HTTP 200
    results["cases"].append({
        "case": "returns_200",
        **assert_true(resp.status_code == 200, f"期望 200，实际 {resp.status_code}"),
    })

    # 4.2 返回 3 只犬，顺序与请求一致
    results["cases"].append({
        "case": "three_dogs_in_order",
        **assert_true(
            len(data["dogs"]) == 3
            and [d["id"] for d in data["dogs"]] == [alpha_id, beta_id, gamma_id],
            f"期望 3 只犬顺序 [{alpha_id},{beta_id},{gamma_id}]，实际 {[d['id'] for d in data['dogs']]}",
        ),
    })

    # 4.3 每犬 4 个评分点
    series_counts = [len(s["points"]) for s in data["series"]]
    results["cases"].append({
        "case": "four_points_per_dog",
        **assert_true(
            series_counts == [4, 4, 4],
            f"期望每犬 4 个点 [4,4,4]，实际 {series_counts}",
        ),
    })

    # 4.4 alpha avg=67.5 / beta avg=80.75 / gamma avg=48.75
    avgs = [s["avg_overall"] for s in data["stats"]]
    results["cases"].append({
        "case": "avg_overall_per_dog",
        **assert_true(
            abs(avgs[0] - 67.5) < 0.01 and abs(avgs[1] - 80.75) < 0.01 and abs(avgs[2] - 48.75) < 0.01,
            f"期望 [67.5, 80.75, 48.75]，实际 {avgs}",
        ),
    })

    # 4.5 beta 趋势最陡（20 分/3 天 vs alpha 15 分/3 天 vs gamma 7 分/3 天）
    slopes = [s["trend_slope"] for s in data["stats"]]
    results["cases"].append({
        "case": "beta_steepest_trend",
        **assert_true(
            all(s is not None and s > 0 for s in slopes) and slopes[1] > slopes[0] > slopes[2],
            f"期望 beta > alpha > gamma 且都为正，实际 {slopes}",
        ),
    })

    # 4.6 每犬的 latest_overall（alpha=75 / beta=90 / gamma=52）
    latests = [s["latest_overall"] for s in data["stats"]]
    results["cases"].append({
        "case": "latest_overall_per_dog",
        **assert_true(
            latests == [75.0, 90.0, 52.0],
            f"期望 [75, 90, 52]，实际 {latests}",
        ),
    })

    return results


async def test_standard_filter(client: httpx.AsyncClient, ids: dict) -> dict[str, Any]:
    """测试 5: 评分标准过滤 + 别名解析。"""
    results = {"name": "standard_filter", "cases": []}
    alpha_id = ids["dogs"][0]
    # alpha 犬有 2 条 GA-T + 2 条 USPCA

    # 5.1 过滤 GA-T → 2 条评分（overall=60, 70）
    resp = await client.get(
        "/api/scores/compare",
        params={"dog_ids": str(alpha_id), "standard": "GA-T"},
    )
    data = resp.json()
    points = data["series"][0]["points"]
    results["cases"].append({
        "case": "gat_filter_two_points",
        **assert_true(
            resp.status_code == 200
            and len(points) == 2
            and [p["overall"] for p in points] == [60.0, 70.0]
            and data["standard"] == "GA-T",
            f"期望 2 条 [60,70] standard=GA-T，实际 {len(points)} 条 {[p['overall'] for p in points]} standard={data.get('standard')}",
        ),
    })

    # 5.2 别名 ga_t 等价于 GA-T
    resp = await client.get(
        "/api/scores/compare",
        params={"dog_ids": str(alpha_id), "standard": "ga_t"},
    )
    data = resp.json()
    results["cases"].append({
        "case": "ga_t_alias_resolved",
        **assert_true(
            resp.status_code == 200
            and data["standard"] == "GA-T"
            and len(data["series"][0]["points"]) == 2,
            f"期望 ga_t 别名解析为 GA-T + 2 条，实际 standard={data.get('standard')} count={len(data['series'][0]['points'])}",
        ),
    })

    # 5.3 过滤 USPCA → 2 条评分（overall=65, 75）
    resp = await client.get(
        "/api/scores/compare",
        params={"dog_ids": str(alpha_id), "standard": "USPCA"},
    )
    data = resp.json()
    points = data["series"][0]["points"]
    results["cases"].append({
        "case": "uspca_filter_two_points",
        **assert_true(
            resp.status_code == 200
            and len(points) == 2
            and [p["overall"] for p in points] == [65.0, 75.0],
            f"期望 2 条 [65,75]，实际 {len(points)} 条 {[p['overall'] for p in points]}",
        ),
    })

    # 5.4 别名 igp 等价于 FCI-IGP
    beta_id = ids["dogs"][1]  # beta 有 1 条 FCI-IGP
    resp = await client.get(
        "/api/scores/compare",
        params={"dog_ids": str(beta_id), "standard": "igp"},
    )
    data = resp.json()
    results["cases"].append({
        "case": "igp_alias_resolved",
        **assert_true(
            resp.status_code == 200
            and data["standard"] == "FCI-IGP"
            and len(data["series"][0]["points"]) == 1,
            f"期望 igp 别名解析为 FCI-IGP + 1 条，实际 standard={data.get('standard')} count={len(data['series'][0]['points'])}",
        ),
    })

    return results


async def test_date_range_filter(client: httpx.AsyncClient, ids: dict) -> dict[str, Any]:
    """测试 6: 日期范围过滤。"""
    results = {"name": "date_range_filter", "cases": []}
    alpha_id = ids["dogs"][0]
    # alpha 评分时间: T0 / T0+1d / T0+2d / T0+3d

    # 6.1 date_from = T0+2d → 仅后 2 条（70, 75）
    resp = await client.get(
        "/api/scores/compare",
        params={
            "dog_ids": str(alpha_id),
            "date_from": (T0 + timedelta(days=2)).isoformat(),
        },
    )
    data = resp.json()
    points = data["series"][0]["points"]
    results["cases"].append({
        "case": "date_from_filters_first_two",
        **assert_true(
            resp.status_code == 200
            and len(points) == 2
            and [p["overall"] for p in points] == [70.0, 75.0],
            f"期望 2 条 [70,75]，实际 {len(points)} 条 {[p['overall'] for p in points]}",
        ),
    })

    # 6.2 date_to = T0+1d → 仅前 2 条（60, 65）
    resp = await client.get(
        "/api/scores/compare",
        params={
            "dog_ids": str(alpha_id),
            "date_to": (T0 + timedelta(days=1)).isoformat(),
        },
    )
    data = resp.json()
    points = data["series"][0]["points"]
    results["cases"].append({
        "case": "date_to_filters_last_two",
        **assert_true(
            resp.status_code == 200
            and len(points) == 2
            and [p["overall"] for p in points] == [60.0, 65.0],
            f"期望 2 条 [60,65]，实际 {len(points)} 条 {[p['overall'] for p in points]}",
        ),
    })

    # 6.3 date_from + date_to → 中间 2 条（65, 70）
    resp = await client.get(
        "/api/scores/compare",
        params={
            "dog_ids": str(alpha_id),
            "date_from": (T0 + timedelta(days=1)).isoformat(),
            "date_to": (T0 + timedelta(days=2)).isoformat(),
        },
    )
    data = resp.json()
    points = data["series"][0]["points"]
    results["cases"].append({
        "case": "date_range_middle_two",
        **assert_true(
            resp.status_code == 200
            and len(points) == 2
            and [p["overall"] for p in points] == [65.0, 70.0],
            f"期望 2 条 [65,70]，实际 {len(points)} 条 {[p['overall'] for p in points]}",
        ),
    })

    # 6.4 响应包含 date_from / date_to 字段
    results["cases"].append({
        "case": "response_includes_date_fields",
        **assert_true(
            data["date_from"] is not None and data["date_to"] is not None,
            f"期望 date_from/date_to 非 None，实际 from={data.get('date_from')} to={data.get('date_to')}",
        ),
    })

    return results


async def test_empty_result(client: httpx.AsyncClient, ids: dict) -> dict[str, Any]:
    """测试 7: 犬只存在但无评分（用未来日期过滤）。"""
    results = {"name": "empty_result", "cases": []}
    alpha_id = ids["dogs"][0]

    # 用 2030 年的日期范围过滤 → 无评分
    future = datetime(2030, 1, 1, tzinfo=timezone.utc)
    resp = await client.get(
        "/api/scores/compare",
        params={
            "dog_ids": str(alpha_id),
            "date_from": future.isoformat(),
        },
    )
    data = resp.json()

    # 7.1 HTTP 200（不是 404，因为犬只存在）
    results["cases"].append({
        "case": "returns_200_with_empty_series",
        **assert_true(resp.status_code == 200, f"期望 200，实际 {resp.status_code}"),
    })

    # 7.2 series 0 个点
    results["cases"].append({
        "case": "empty_series_points",
        **assert_true(
            len(data["series"][0]["points"]) == 0,
            f"期望 0 个点，实际 {len(data['series'][0]['points'])}",
        ),
    })

    # 7.3 stats count=0 + latest_overall=None + trend_slope=None
    stats = data["stats"][0]
    results["cases"].append({
        "case": "empty_stats",
        **assert_true(
            stats["count"] == 0
            and stats["latest_overall"] is None
            and stats["trend_slope"] is None
            and stats["avg_overall"] == 0.0,
            f"期望 count=0 latest=None slope=None avg=0.0，实际 count={stats['count']} latest={stats['latest_overall']} slope={stats['trend_slope']} avg={stats['avg_overall']}",
        ),
    })

    return results


async def test_dedup_preserve_order(client: httpx.AsyncClient, ids: dict) -> dict[str, Any]:
    """测试 8: dog_ids 去重保序。"""
    results = {"name": "dedup_preserve_order", "cases": []}
    a, b, g = ids["dogs"]

    # 传入 "alpha,beta,alpha,gamma"（用 ID）→ 去重后 [alpha, beta, gamma]
    resp = await client.get(
        "/api/scores/compare",
        params={"dog_ids": f"{a},{b},{a},{g}"},
    )
    data = resp.json()

    results["cases"].append({
        "case": "dedup_to_three_dogs",
        **assert_true(
            resp.status_code == 200 and len(data["dogs"]) == 3,
            f"期望 3 只犬，实际 {len(data['dogs'])}",
        ),
    })

    results["cases"].append({
        "case": "order_preserved",
        **assert_true(
            [d["id"] for d in data["dogs"]] == [a, b, g],
            f"期望顺序 [{a},{b},{g}]，实际 {[d['id'] for d in data['dogs']]}",
        ),
    })

    return results


async def test_dimension_labels(client: httpx.AsyncClient, ids: dict) -> dict[str, Any]:
    """测试 9: 维度标签。"""
    results = {"name": "dimension_labels", "cases": []}
    alpha_id = ids["dogs"][0]

    resp = await client.get("/api/scores/compare", params={"dog_ids": str(alpha_id)})
    data = resp.json()
    labels = data["dimension_labels"]

    expected_labels = {
        "accuracy": "准确度",
        "response_latency": "响应延迟",
        "duration": "保持时长",
        "search_efficiency": "搜索效率",
        "attention": "注意力",
        "courage": "胆量欲望",
        "gait_quality": "步态质量",
    }

    results["cases"].append({
        "case": "seven_dimension_labels",
        **assert_true(
            resp.status_code == 200 and labels == expected_labels,
            f"期望 7 维中文标签，实际 {labels}",
        ),
    })

    return results


async def test_partial_dimensions(client: httpx.AsyncClient, ids: dict) -> dict[str, Any]:
    """测试 10: 部分维度为 None（gamma 犬仅 3 维有值）。"""
    results = {"name": "partial_dimensions", "cases": []}
    gamma_id = ids["dogs"][2]
    # gamma 评分: accuracy + duration + attention 有值，其余 None

    resp = await client.get("/api/scores/compare", params={"dog_ids": str(gamma_id)})
    data = resp.json()
    avg_dims = data["stats"][0]["avg_dimensions"]

    # 10.1 只包含 3 个维度
    results["cases"].append({
        "case": "only_three_dimensions",
        **assert_true(
            resp.status_code == 200 and set(avg_dims.keys()) == {"accuracy", "duration", "attention"},
            f"期望 3 维 {{accuracy, duration, attention}}，实际 {set(avg_dims.keys())}",
        ),
    })

    # 10.2 不包含 courage / gait_quality 等 None 维度
    results["cases"].append({
        "case": "no_none_dimensions",
        **assert_true(
            "courage" not in avg_dims and "gait_quality" not in avg_dims
            and "response_latency" not in avg_dims and "search_efficiency" not in avg_dims,
            f"期望不包含 None 维度，实际 {avg_dims.keys()}",
        ),
    })

    # 10.3 各 ScorePoint.dimensions 也只含 3 维
    points = data["series"][0]["points"]
    all_have_three = all(set(p["dimensions"].keys()) == {"accuracy", "duration", "attention"} for p in points)
    results["cases"].append({
        "case": "points_only_three_dimensions",
        **assert_true(
            all_have_three,
            f"期望每个 point 只含 3 维，实际 {[set(p['dimensions'].keys()) for p in points]}",
        ),
    })

    return results


# ============================================================
# 主流程
# ============================================================


async def run_tests(ids: dict) -> list[dict[str, Any]]:
    """在单一 event loop 中运行所有测试套件。"""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        test_suites = [
            ("参数校验", test_parameter_validation, (client, ids)),
            ("犬只不存在", test_dog_not_found, (client,)),
            ("单犬时序", test_single_dog_timeseries, (client, ids)),
            ("多犬对比", test_multi_dog_comparison, (client, ids)),
            ("标准过滤", test_standard_filter, (client, ids)),
            ("日期范围", test_date_range_filter, (client, ids)),
            ("空结果", test_empty_result, (client, ids)),
            ("去重保序", test_dedup_preserve_order, (client, ids)),
            ("维度标签", test_dimension_labels, (client, ids)),
            ("部分维度", test_partial_dimensions, (client, ids)),
        ]
        all_results = []
        for name, suite_fn, args in test_suites:
            print(f"\n  → {suite_fn.__name__}（{name}）...")
            result = await suite_fn(*args)
            all_results.append(result)
            passed = sum(1 for c in result["cases"] if c["status"] == "pass")
            total = len(result["cases"])
            print(f"    {passed}/{total} 通过")
        return all_results


def main() -> int:
    print("=" * 70)
    print("Phase 3.7c 训练历史对比 API 端到端评估")
    print("=" * 70)

    # 1. 连接 DB + 清理旧数据 + 种子
    print("\n[1/4] 准备种子数据...")
    conn = get_db_conn()
    try:
        cleanup_seed_data(conn)
        ids = seed_data(conn)
        print(f"  ✓ 犬只: {ids['dogs']}")
        print(f"  ✓ 视频: {ids['videos']}")
        print(f"  ✓ 评分: {ids['scores']}（共 {len(ids['scores'])} 条）")
    finally:
        conn.close()

    # 2. 运行测试（单一 async event loop）
    print("\n[2/4] 运行测试套件...")
    all_results = asyncio.run(run_tests(ids))

    # 3. 汇总
    print("\n[3/4] 汇总结果...")
    total_cases = sum(len(r["cases"]) for r in all_results)
    passed_cases = sum(1 for r in all_results for c in r["cases"] if c["status"] == "pass")
    failed_cases = total_cases - passed_cases
    overall = "pass" if failed_cases == 0 else "fail"

    report = {
        "phase": "3.7c",
        "test_name": "训练历史对比 API 端到端评估",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "overall": overall,
        "summary": {
            "total_suites": len(all_results),
            "total_cases": total_cases,
            "passed": passed_cases,
            "failed": failed_cases,
            "pass_rate": f"{passed_cases / total_cases * 100:.1f}%" if total_cases else "N/A",
        },
        "suites": all_results,
    }

    print(f"\n  总计: {passed_cases}/{total_cases} 通过 ({report['summary']['pass_rate']})")
    print(f"  总体: {overall.upper()}")

    # 打印失败用例
    if failed_cases > 0:
        print("\n  失败用例:")
        for r in all_results:
            for c in r["cases"]:
                if c["status"] == "fail":
                    print(f"    ✗ [{r['name']}] {c['case']}: {c['detail']}")

    # 4. 清理种子数据 + 写报告
    print("\n[4/4] 清理种子数据 + 写报告...")
    conn = get_db_conn()
    try:
        cleanup_seed_data(conn)
        print("  ✓ 种子数据已清理")
    finally:
        conn.close()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ 报告已写入 {REPORT_PATH}")

    print("\n" + "=" * 70)
    if overall == "pass":
        print("✅ Phase 3.7c 训练历史对比 API 端到端评估通过")
    else:
        print(f"❌ Phase 3.7c 训练历史对比 API 端到端评估失败（{failed_cases} 个用例未通过）")
    print("=" * 70)

    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
