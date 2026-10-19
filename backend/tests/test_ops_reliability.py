"""P1-6 运维与延迟可观测性单元测试（无 DB / 无 GPU）.

覆盖:
    - 可判定错误码解析与异常分类（同阶段冲突样例）
    - 已运行时长 / stale / retry_scheduled 派生
    - processing 与 e2e 双口径统计
    - 结构化日志（真实 emit，防 INFO 被 WARNING 级别吞掉）
    - 失败落库 None-safe（写库失败不掩盖原异常）
    - /api/videos/ops/stats 路由与 ADMIN/MANAGER 权限依赖
    - 迁移文件纯 additive（AST）与离线 SQL 真编译（subprocess）
"""
from __future__ import annotations

import ast
import inspect
import io
import json
import logging
import subprocess
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.app.models.video import VideoStatus
from backend.app.schemas.common import (
    PROCESSING_STALE_AFTER_SEC,
    VideoErrorCode,
    VideoRead,
    VideoStatusRead,
    _is_processing_stale,
    error_code_hint,
    parse_error_message,
    processing_elapsed_ms,
)
from backend.workers import tasks

pytestmark = pytest.mark.fast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_PATH = (
    PROJECT_ROOT
    / "backend"
    / "alembic"
    / "versions"
    / "d4e5f6a7b8c9_ops_p1_6_processing_observability.py"
)
EXPECTED_COLUMNS = {
    "processing_stage",
    "processing_percent",
    "processing_started_at",
    "processing_duration_ms",
}


# ============================================================
# 1. 错误码解析与异常分类
# ============================================================


class TestErrorCodes:
    def test_parse_known_code(self) -> None:
        code, text = parse_error_message("[FILE_MISSING] 视频文件在服务器上缺失")
        assert code == "FILE_MISSING"
        assert text == "视频文件在服务器上缺失"

    def test_parse_unknown_code_keeps_original(self) -> None:
        raw = "[UNKNOWN] 谁知道呢"
        code, text = parse_error_message(raw)
        assert code is None
        assert text == raw

    def test_parse_legacy_message(self) -> None:
        raw = "FileNotFoundError: 视频文件不存在"
        code, text = parse_error_message(raw)
        assert code is None
        assert text == raw

    def test_parse_empty(self) -> None:
        assert parse_error_message(None) == (None, "")
        assert parse_error_message("") == (None, "")

    def test_error_hint_covers_all_codes(self) -> None:
        for member in VideoErrorCode:
            assert error_code_hint(member.value)

    def test_video_read_unknown_code_no_serializer_warning(self) -> None:
        model = VideoRead(
            id=1,
            original_filename="x.mp4",
            status="failed",
            uploaded_at=datetime.now(timezone.utc),
            error_message="[UNKNOWN] 未知",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dumped = model.model_dump()
        assert dumped["error_code"] is None
        assert not [
            w for w in caught if "Pydantic serializer" in str(w.message)
        ], [str(w.message) for w in caught]

    def test_known_code_derives_hint(self) -> None:
        model = VideoRead(
            id=1,
            original_filename="x.mp4",
            status="failed",
            uploaded_at=datetime.now(timezone.utc),
            error_message="[MODEL_UNAVAILABLE] 模型挂了",
        )
        assert model.error_code == "MODEL_UNAVAILABLE"
        assert model.error_hint and "模型" in model.error_hint


class TestClassifyError:
    def test_video_missing_file(self) -> None:
        code = tasks._classify_error(
            FileNotFoundError("视频文件不存在: /data/x.mp4"), tasks.STAGE_POSE_INFERENCE
        )
        assert code is VideoErrorCode.FILE_MISSING

    def test_model_missing_file_same_stage(self) -> None:
        code = tasks._classify_error(
            FileNotFoundError("模型文件不存在: /models/best.onnx"),
            tasks.STAGE_POSE_INFERENCE,
        )
        assert code is VideoErrorCode.MODEL_UNAVAILABLE

    def test_video_unreadable_same_stage(self) -> None:
        code = tasks._classify_error(
            RuntimeError("无法打开视频: /data/x.mp4"), tasks.STAGE_POSE_INFERENCE
        )
        assert code is VideoErrorCode.VIDEO_UNREADABLE

    def test_oom_same_stage(self) -> None:
        code = tasks._classify_error(
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
            tasks.STAGE_POSE_INFERENCE,
        )
        assert code is VideoErrorCode.MODEL_UNAVAILABLE

    def test_generic_inference_pose(self) -> None:
        code = tasks._classify_error(
            RuntimeError("unexpected"), tasks.STAGE_POSE_INFERENCE
        )
        assert code is VideoErrorCode.INFERENCE_FAILED

    def test_video_not_found(self) -> None:
        code = tasks._classify_error(
            RuntimeError("Video 42 not found"), tasks.STAGE_PREPARING
        )
        assert code is VideoErrorCode.VIDEO_NOT_FOUND

    def test_unsupported_scene(self) -> None:
        code = tasks._classify_error(ValueError("未知 scene: x"), tasks.STAGE_PREPARING)
        assert code is VideoErrorCode.UNSUPPORTED_SCENE

    def test_scoring_stage(self) -> None:
        assert (
            tasks._classify_error(RuntimeError("boom"), tasks.STAGE_SCORING)
            is VideoErrorCode.SCORING_FAILED
        )

    def test_report_stage(self) -> None:
        assert (
            tasks._classify_error(RuntimeError("boom"), tasks.STAGE_REPORT)
            is VideoErrorCode.REPORT_FAILED
        )

    def test_persist_stage(self) -> None:
        assert (
            tasks._classify_error(RuntimeError("boom"), tasks.STAGE_PERSIST)
            is VideoErrorCode.DB_WRITE_FAILED
        )

    def test_fallback_internal(self) -> None:
        assert (
            tasks._classify_error(RuntimeError("boom"), tasks.STAGE_PREPARING)
            is VideoErrorCode.INTERNAL_ERROR
        )

    def test_sqlalchemy_error(self) -> None:
        from sqlalchemy.exc import SQLAlchemyError

        assert (
            tasks._classify_error(SQLAlchemyError("db down"), tasks.STAGE_PERSIST)
            is VideoErrorCode.DB_WRITE_FAILED
        )

    def test_error_message_has_code_prefix_and_truncation(self) -> None:
        msg = tasks._format_error_message(
            VideoErrorCode.FILE_MISSING, RuntimeError("x" * 5000)
        )
        assert msg.startswith("[FILE_MISSING]")
        assert "详情:" in msg
        assert "x" * 1001 not in msg


# ============================================================
# 2. elapsed / stale / retry
# ============================================================


class TestElapsed:
    def test_aware(self) -> None:
        start = datetime(2026, 10, 19, 0, 0, 0, tzinfo=timezone.utc)
        now = start + timedelta(seconds=12)
        assert processing_elapsed_ms(start, now) == 12000

    def test_naive_treated_as_utc(self) -> None:
        start = datetime(2026, 10, 19, 0, 0, 0)
        now = datetime(2026, 10, 19, 0, 0, 5, tzinfo=timezone.utc)
        assert processing_elapsed_ms(start, now) == 5000

    def test_clock_rollback_clamped(self) -> None:
        start = datetime(2026, 10, 19, 0, 0, 10, tzinfo=timezone.utc)
        now = datetime(2026, 10, 19, 0, 0, 0, tzinfo=timezone.utc)
        assert processing_elapsed_ms(start, now) == 0

    def test_none_start(self) -> None:
        assert processing_elapsed_ms(None) is None

    def test_worker_elapsed_none_safe(self) -> None:
        assert tasks._elapsed_ms(None) is None

    def test_worker_elapsed_naive(self) -> None:
        start = datetime(2026, 10, 19, 0, 0, 0)
        end = datetime(2026, 10, 19, 0, 0, 3, tzinfo=timezone.utc)
        assert tasks._elapsed_ms(start, end) == 3000


class TestStale:
    def test_stale_without_started_at_old_upload(self) -> None:
        now = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)
        uploaded = now - timedelta(seconds=PROCESSING_STALE_AFTER_SEC + 1)
        assert _is_processing_stale("processing", None, uploaded, now) is True

    def test_not_stale_without_started_at_recent_upload(self) -> None:
        now = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)
        uploaded = now - timedelta(minutes=5)
        assert _is_processing_stale("processing", None, uploaded, now) is False

    def test_stale_over_threshold(self) -> None:
        now = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)
        started = now - timedelta(seconds=PROCESSING_STALE_AFTER_SEC + 1)
        assert _is_processing_stale("processing", started, None, now) is True

    def test_long_task_not_stale(self) -> None:
        now = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)
        started = now - timedelta(hours=1)
        assert _is_processing_stale("processing", started, None, now) is False

    def test_non_processing_never_stale(self) -> None:
        now = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)
        started = now - timedelta(days=3)
        assert _is_processing_stale("completed", started, None, now) is False

    def test_status_read_retry_scheduled(self) -> None:
        now = datetime.now(timezone.utc)
        model = VideoStatusRead(
            id=1,
            status="processing",
            scene="obedience_trial",
            uploaded_at=now,
            processing_started_at=now,
            processing_stage="retrying",
        )
        assert model.retry_scheduled is True
        assert model.processing_stale is False
        assert model.processing_elapsed_ms is not None

    def test_status_read_not_retrying(self) -> None:
        now = datetime.now(timezone.utc)
        model = VideoStatusRead(
            id=1,
            status="processing",
            scene="obedience_trial",
            uploaded_at=now,
            processing_started_at=now,
            processing_stage="scoring",
        )
        assert model.retry_scheduled is False

    def test_status_read_stale_flag(self) -> None:
        now = datetime.now(timezone.utc)
        model = VideoStatusRead(
            id=1,
            status="processing",
            scene="obedience_trial",
            uploaded_at=now - timedelta(days=1),
            processing_started_at=None,
        )
        assert model.processing_stale is True
        assert model.processing_elapsed_ms is None


# ============================================================
# 3. 统计口径
# ============================================================


def _video(**kwargs: Any) -> SimpleNamespace:
    defaults = dict(
        status=VideoStatus.COMPLETED,
        processing_duration_ms=None,
        uploaded_at=None,
        processed_at=None,
        processing_started_at=None,
        processing_stage=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestPercentile:
    def test_empty(self) -> None:
        from backend.app.api.videos import _percentile_nearest_rank

        assert _percentile_nearest_rank([], 0.5) is None

    def test_single(self) -> None:
        from backend.app.api.videos import _percentile_nearest_rank

        assert _percentile_nearest_rank([42], 0.95) == 42

    def test_1_to_100(self) -> None:
        from backend.app.api.videos import _percentile_nearest_rank

        values = list(range(1, 101))
        assert _percentile_nearest_rank(values, 0.50) == 50
        assert _percentile_nearest_rank(values, 0.95) == 95


class TestStatsSummary:
    def test_failure_rate_and_samples(self) -> None:
        from backend.app.api.videos import _summarize_processing_stats

        now = datetime.now(timezone.utc)
        since = now - timedelta(days=7)
        window = [
            _video(processing_duration_ms=1000, uploaded_at=now - timedelta(hours=2), processed_at=now),
            _video(processing_duration_ms=3000, uploaded_at=now - timedelta(hours=2), processed_at=now),
            _video(status=VideoStatus.FAILED, processed_at=now),
        ]
        result = _summarize_processing_stats(
            window_videos=window,
            processing_videos=[],
            since=since,
            days=7,
            now=now,
        )
        assert result.sample_size == 2
        assert result.completed == 2
        assert result.failed == 1
        assert result.failure_rate == pytest.approx(1 / 3)
        assert result.p50_duration_ms == 1000
        assert result.reliable is False  # 样本 < 20

    def test_empty_failure_rate_none(self) -> None:
        from backend.app.api.videos import _summarize_processing_stats

        now = datetime.now(timezone.utc)
        result = _summarize_processing_stats(
            window_videos=[],
            processing_videos=[],
            since=now - timedelta(days=7),
            days=7,
            now=now,
        )
        assert result.failure_rate is None
        assert result.sample_size == 0
        assert result.p95_duration_ms is None

    def test_reliable_at_20_samples(self) -> None:
        from backend.app.api.videos import _summarize_processing_stats

        now = datetime.now(timezone.utc)
        window = [_video(processing_duration_ms=100) for _ in range(20)]
        result = _summarize_processing_stats(
            window_videos=window,
            processing_videos=[],
            since=now - timedelta(days=7),
            days=7,
            now=now,
        )
        assert result.sample_size == 20
        assert result.reliable is True

    def test_e2e_filtering(self) -> None:
        from backend.app.api.videos import _summarize_processing_stats

        now = datetime.now(timezone.utc)
        window = [
            # 合法 e2e = 10s
            _video(
                processing_duration_ms=1000,
                uploaded_at=now - timedelta(seconds=10),
                processed_at=now,
            ),
            # 缺 uploaded_at → 排除
            _video(processing_duration_ms=1000, uploaded_at=None, processed_at=now),
            # e2e 为负（时钟异常）→ 排除
            _video(
                processing_duration_ms=1000,
                uploaded_at=now + timedelta(seconds=5),
                processed_at=now,
            ),
            # failed 不计入 e2e
            _video(
                status=VideoStatus.FAILED,
                uploaded_at=now - timedelta(seconds=99),
                processed_at=now,
            ),
        ]
        result = _summarize_processing_stats(
            window_videos=window,
            processing_videos=[],
            since=now - timedelta(days=7),
            days=7,
            now=now,
        )
        assert result.e2e_sample_size == 1
        assert result.e2e_p50_ms == 10000
        assert result.sample_size == 3

    def test_stale_processing_count(self) -> None:
        from backend.app.api.videos import _summarize_processing_stats

        now = datetime.now(timezone.utc)
        processing = [
            _video(
                status=VideoStatus.PROCESSING,
                processing_started_at=now - timedelta(hours=3),
            ),
            _video(
                status=VideoStatus.PROCESSING,
                processing_started_at=now - timedelta(minutes=5),
            ),
            # 无 started_at 且上传已久 → stale（legacy 行）
            _video(
                status=VideoStatus.PROCESSING,
                processing_started_at=None,
                uploaded_at=now - timedelta(days=30),
            ),
        ]
        result = _summarize_processing_stats(
            window_videos=[],
            processing_videos=processing,
            since=now - timedelta(days=7),
            days=7,
            now=now,
        )
        assert result.stale_processing == 2


# ============================================================
# 4. 结构化日志
# ============================================================


class TestOpsLogging:
    def test_event_payload_json(self) -> None:
        payload = tasks._event_payload(
            "stage_enter", video_id=7, stage="scoring", elapsed_ms=12
        )
        data = json.loads(payload)
        assert data == {
            "event": "stage_enter",
            "video_id": 7,
            "stage": "scoring",
            "elapsed_ms": 12,
        }

    def test_event_payload_truncates_detail(self) -> None:
        payload = tasks._event_payload("ingest_failed", detail="y" * 2000)
        detail = json.loads(payload)["detail"]
        assert len(detail) <= 500

    def test_event_payload_serializes_enum_and_datetime(self) -> None:
        payload = tasks._event_payload(
            "ingest_failed",
            error_code=VideoErrorCode.FILE_MISSING,
            at=datetime(2026, 10, 19, tzinfo=timezone.utc),
        )
        data = json.loads(payload)
        assert data["error_code"] == "FILE_MISSING"
        assert data["at"].startswith("2026-10-19")

    def test_redact_password_and_path(self) -> None:
        text = tasks._redact(
            "postgresql://k9system:S3cret@127.0.0.1:5433/k9system C:\\Users\\FOUR\\x.mp4"
        )
        assert "S3cret" not in text
        assert "://***@" in text
        assert "C:\\Users" not in text
        assert "<path>" in text

    def test_redact_none(self) -> None:
        assert tasks._redact(None) == ""

    def test_real_info_emit_reaches_handler(self) -> None:
        """logger 与 handler 双设 INFO，真实 emit 一条，防 effective=WARNING 回归。"""
        assert tasks.ops_logger.level == logging.INFO
        assert tasks.ops_logger.propagate is False

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        tasks.ops_logger.addHandler(handler)
        try:
            tasks.ops_logger.info(
                tasks._event_payload("stage_enter", video_id=1, stage="preparing")
            )
        finally:
            tasks.ops_logger.removeHandler(handler)

        out = stream.getvalue()
        assert "stage_enter" in out
        assert '"video_id": 1' in out


# ============================================================
# 5. 进度与失败落库（None-safe）
# ============================================================


class _FakeDb:
    def __init__(self, fail_commit: bool = False) -> None:
        self.fail_commit = fail_commit
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1
        if self.fail_commit:
            raise RuntimeError("commit failed")

    def rollback(self) -> None:
        self.rollbacks += 1


class TestProgress:
    def test_monotonic(self) -> None:
        video = SimpleNamespace(id=1, processing_stage=None, processing_percent=50.0)
        db = _FakeDb()
        tasks._set_progress(db, video, tasks.STAGE_POSE_INFERENCE, None)
        assert video.processing_percent == 50.0  # 不回退
        tasks._set_progress(db, video, tasks.STAGE_SCORING, None)
        assert video.processing_percent == 96.0
        assert video.processing_stage == "scoring"

    def test_force_reset(self) -> None:
        video = SimpleNamespace(id=1, processing_stage="scoring", processing_percent=96.0)
        tasks._set_progress(_FakeDb(), video, tasks.STAGE_PREPARING, None, force=True)
        assert video.processing_percent == 0.0

    def test_clamped_to_100(self) -> None:
        video = SimpleNamespace(id=1, processing_stage=None, processing_percent=None)
        tasks._set_progress(_FakeDb(), video, tasks.STAGE_COMPLETED, 150)
        assert video.processing_percent == 100.0

    def test_commit_failure_swallowed_and_rollback(self) -> None:
        video = SimpleNamespace(id=1, processing_stage=None, processing_percent=None)
        db = _FakeDb(fail_commit=True)
        tasks._set_progress(db, video, tasks.STAGE_POSE_INFERENCE, None)
        assert db.rollbacks >= 1


class _ExplodingOnEnter:
    def __enter__(self):
        raise RuntimeError("db2 unavailable")

    def __exit__(self, *exc: Any) -> bool:
        return False


class TestFailureRecording:
    def test_record_failure_swallows_db_error(self, monkeypatch) -> None:
        monkeypatch.setattr(tasks, "SyncSessionLocal", lambda: _ExplodingOnEnter())
        # 不得抛出：写库失败不能掩盖原异常
        tasks._record_failure(
            1,
            None,
            VideoErrorCode.FILE_MISSING,
            FileNotFoundError("视频文件不存在"),
            retrying=False,
        )

    def test_record_failure_none_started_at(self, monkeypatch) -> None:
        saved: dict[str, Any] = {}

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *exc: Any) -> bool:
                return False

            def get(self, model, pk):
                return SimpleNamespace(
                    status=None,
                    error_message=None,
                    processed_at=None,
                    processing_stage=None,
                    processing_duration_ms=None,
                )

            def commit(self):
                saved["committed"] = True

        monkeypatch.setattr(tasks, "SyncSessionLocal", _Session)
        tasks._record_failure(
            1,
            None,
            VideoErrorCode.VIDEO_UNREADABLE,
            RuntimeError("无法打开视频"),
            retrying=False,
        )
        assert saved.get("committed") is True

    def test_record_failure_retrying_keeps_processing(self, monkeypatch) -> None:
        captured: dict[str, Any] = {}

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *exc: Any) -> bool:
                return False

            def get(self, model, pk):
                obj = SimpleNamespace(
                    status=None,
                    error_message=None,
                    processed_at=None,
                    processing_stage=None,
                    processing_duration_ms=None,
                )
                captured["obj"] = obj
                return obj

            def commit(self):
                pass

        monkeypatch.setattr(tasks, "SyncSessionLocal", _Session)
        tasks._record_failure(
            1,
            datetime.now(timezone.utc),
            VideoErrorCode.MODEL_UNAVAILABLE,
            RuntimeError("CUDA out of memory"),
            retrying=True,
        )
        obj = captured["obj"]
        assert obj.status is VideoStatus.PROCESSING
        assert obj.processing_stage == "retrying"
        assert obj.processed_at is None
        assert obj.processing_duration_ms is None
        assert obj.error_message.startswith("[MODEL_UNAVAILABLE]")

    def test_record_failure_final_marks_failed_with_duration(self, monkeypatch) -> None:
        captured: dict[str, Any] = {}

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *exc: Any) -> bool:
                return False

            def get(self, model, pk):
                obj = SimpleNamespace(
                    status=None,
                    error_message=None,
                    processed_at=None,
                    processing_stage=None,
                    processing_duration_ms=None,
                )
                captured["obj"] = obj
                return obj

            def commit(self):
                captured["committed"] = True

        monkeypatch.setattr(tasks, "SyncSessionLocal", _Session)
        started = datetime.now(timezone.utc) - timedelta(seconds=3)
        tasks._record_failure(
            1,
            started,
            VideoErrorCode.FILE_MISSING,
            FileNotFoundError("视频文件不存在"),
            retrying=False,
        )
        obj = captured["obj"]
        assert obj.status is VideoStatus.FAILED
        assert obj.processed_at is not None
        assert obj.processing_duration_ms is not None
        assert obj.processing_duration_ms >= 3000
        assert obj.error_message.startswith("[FILE_MISSING]")
        assert captured.get("committed") is True


# ============================================================
# 6. 路由与权限
# ============================================================


def _iter_dependency_calls(dependant) -> list[Any]:
    calls = []
    for dep in dependant.dependencies:
        calls.append(dep.call)
        calls.extend(_iter_dependency_calls(dep))
    return calls


class TestOpsStatsRoute:
    def test_route_registered_with_admin_manager(self) -> None:
        from backend.app.main import app

        route = next(
            r for r in app.routes if getattr(r, "path", None) == "/api/videos/ops/stats"
        )
        assert "GET" in route.methods

        allowed: Any = None
        for call in _iter_dependency_calls(route.dependant):
            if call is None:
                continue
            try:
                closure = inspect.getclosurevars(call).nonlocals
            except TypeError:
                continue
            if "allowed_values" in closure:
                allowed = closure["allowed_values"]

        assert allowed == {"admin", "manager"}

    def test_stats_route_before_video_id_route(self) -> None:
        from backend.app.main import app

        paths = [r.path for r in app.routes if getattr(r, "path", "").startswith("/api/videos")]
        assert paths.index("/api/videos/ops/stats") < paths.index("/api/videos/{video_id}")


# ============================================================
# 7. 迁移：AST 纯 additive + 离线真编译
# ============================================================


class TestMigration:
    def test_upgrade_exactly_four_additive_columns(self) -> None:
        tree = ast.parse(MIGRATION_PATH.read_text(encoding="utf-8"))
        upgrade = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "upgrade"
        )

        added: list[str] = []
        for node in ast.walk(upgrade):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr == "add_column":
                # op.add_column('videos', sa.Column('name', ...))
                column = node.args[1]
                assert isinstance(column, ast.Call)
                added.append(column.args[0].value)
            assert func.attr not in {
                "drop_column",
                "alter_column",
                "drop_table",
                "execute",
            }, f"upgrade 不得包含 {func.attr}"

        assert set(added) == EXPECTED_COLUMNS
        assert len(added) == 4

    def test_downgrade_drops_index_then_columns(self) -> None:
        tree = ast.parse(MIGRATION_PATH.read_text(encoding="utf-8"))
        downgrade = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "downgrade"
        )
        operations: list[str] = []
        for node in ast.walk(downgrade):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                operations.append(node.func.attr)
        assert operations[0] == "drop_index"
        assert operations.count("drop_column") == 4

    def test_offline_sql_compilation(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alembic",
                "-c",
                "alembic.ini",
                "upgrade",
                "c3d4e5f6a7b8:d4e5f6a7b8c9",
                "--sql",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        sql = result.stdout
        for column in EXPECTED_COLUMNS:
            assert f"ALTER TABLE videos ADD COLUMN {column}" in sql
        assert sql.count("ALTER TABLE videos ADD COLUMN") == 4
        assert "CREATE INDEX ix_videos_processed_at ON videos (processed_at)" in sql
        assert "ADD COLUMN" in sql and "DROP" not in sql.upper().replace("DROPPED", "")
