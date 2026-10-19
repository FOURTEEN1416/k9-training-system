"""P1-2 数据飞轮工程链定向测试.

覆盖:
    1. boundary tensor → numpy → pickle → Dataset 取样回归（MUSTFIX-1）
    2. tiny CPU 训练 + best/last checkpoint + MLModel 注册（MUSTFIX-2/10）
    3. 导出排除 rejected/unreviewed、corrected 生效值、16 类映射、label != -1
    4. 按 video 划分无泄漏
    5. 缺帧插值/坏形状跳过计数
    6. 无任务视频懒创建 AnnotationTask（MUSTFIX-5）
    7. 权限矩阵（MUSTFIX-6）
    8. complete 状态机 / 409 / reopen / annotated_frames
    9. 空 split / train < batch_size 报错（MUSTFIX-13）
    10. pickle 路径守卫（MUSTFIX-12）
    11. Celery 注册
    12. broker 探测失败 → 503
    13. DB∩ML 标签漂移（MUSTFIX-4）
    14. TestClient 端点（403/400/202/404）
    15. trigger 缺 export_id → 400

测试用文件型 SQLite + NullPool（aiosqlite 未安装）；所有 Video 显式 uploaded_at，
避免 server_default='now()' 在 SQLite 下产生 Invalid isoformat string（MUSTFIX-3）。

本文件只验证工程链路：工程验收 ≠ 生产验收。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault(
    "AUTH_SECRET_KEY",
    "TEST-ONLY-k9-training-system-not-for-prod-32chars!!",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

import backend.app.models  # noqa: F401  注册全部表
from backend.app.models.annotation import (
    Annotation,
    AnnotationSource,
    AnnotationTask,
    AnnotationTaskStatus,
    AnnotationType,
)
from backend.app.models.base import Base
from backend.app.models.base_entity import Base_
from backend.app.models.behavior import Behavior, BehaviorClass, BehaviorDetector
from backend.app.models.dog import Dog
from backend.app.models.handler import Handler, UserRole
from backend.app.models.keypoint import Keypoint
from backend.app.models.ml_model import MLModel, ModelType
from backend.app.models.training_session import TrainingSession
from backend.app.models.video import Video

pytestmark = pytest.mark.fast


# ============================================================
# DB 基础设施（MUSTFIX-3）
# ============================================================


def make_db(tmp_path: Path) -> tuple[object, sessionmaker]:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{data_dir / 'test.db'}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False, autoflush=False)
    return engine, factory


def make_handler(session: Session, name: str, role: UserRole, base_id: Optional[int] = None) -> Handler:
    handler = Handler(name=name, email=f"{name}@test.local", role=role, base_id=base_id, is_active=True)
    session.add(handler)
    session.flush()
    return handler


def make_video(
    session: Session,
    name: str,
    *,
    handler_id: Optional[int] = None,
    dog_id: Optional[int] = None,
    session_id: Optional[int] = None,
    fps: float = 30.0,
    duration: float = 2.0,
) -> Video:
    video = Video(
        handler_id=handler_id,
        dog_id=dog_id,
        session_id=session_id,
        original_filename=f"{name}.mp4",
        storage_path=f"uploads/{name}.mp4",
        storage_filename=f"{name}.mp4",
        scene="obedience_trial",
        fps=fps,
        duration_sec=duration,
        uploaded_at=datetime.now(timezone.utc),
    )
    session.add(video)
    session.flush()
    return video


def make_behavior(
    session: Session,
    video: Video,
    behavior_class: BehaviorClass,
    start: int,
    end: int,
) -> Behavior:
    behavior = Behavior(
        video_id=video.id,
        behavior_class=behavior_class,
        detector=BehaviorDetector.RULE,
        start_frame=start,
        end_frame=end,
        start_sec=start / 30.0,
        end_sec=end / 30.0,
        confidence=0.9,
    )
    session.add(behavior)
    session.flush()
    return behavior


def add_keypoints(
    session: Session,
    video: Video,
    frames: range | list[int],
    *,
    bad_frames: Optional[set[int]] = None,
) -> None:
    bad_frames = bad_frames or set()
    for idx in frames:
        row = [[float(idx), float(j), 0.9] for j in range(24)]
        if idx in bad_frames:
            row = row[:23]  # 非法形状 (23, 3)
        session.add(
            Keypoint(
                video_id=video.id,
                frame_idx=idx,
                frame_time_sec=idx / 30.0,
                keypoints_json=row,
                detection_confidence=0.9,
            )
        )
    session.flush()


def make_task(session: Session, video: Video, status: AnnotationTaskStatus = AnnotationTaskStatus.PENDING) -> AnnotationTask:
    task = AnnotationTask(
        video_id=video.id,
        handler_id=video.handler_id,
        status=status,
        total_frames=60,
        annotation_types=["behavior"],
    )
    session.add(task)
    session.flush()
    return task


def add_review(
    session: Session,
    task: AnnotationTask,
    behavior: Behavior,
    decision: str,
    *,
    behavior_name: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    reviewed_by: int = 1,
) -> Annotation:
    from backend.app.services.training_export import review_payload

    payload = review_payload(
        behavior=behavior,
        decision=decision,
        effective_behavior=behavior_name or behavior.behavior_class.value,
        effective_start=start if start is not None else behavior.start_frame,
        effective_end=end if end is not None else behavior.end_frame,
        reviewed_by=reviewed_by,
    )
    annotation = Annotation(
        task_id=task.id,
        video_id=behavior.video_id,
        frame_idx=None,
        annotation_type=AnnotationType.BEHAVIOR,
        source=AnnotationSource.HUMAN,
        data_json=payload,
        confidence=0.9,
    )
    session.add(annotation)
    session.flush()
    return annotation


# ============================================================
# 4.1 MUSTFIX-1：boundary tensor → numpy → pickle → Dataset
# ============================================================


def test_boundary_tensor_roundtrip_is_numpy(tmp_path: Path) -> None:
    import torch

    from backend.ml.behavior.stgcn_bc.bc_head import generate_boundary_labels
    from backend.ml.behavior.stgcn_bc.data_adapter import keypoints_to_pyskl, save_pyskl_pickle
    from backend.ml.behavior.stgcn_bc.dataset import STGCNBCDataset, load_pyskl_pickle

    T = 12
    tensor = generate_boundary_labels(T, [0], [T - 1])
    assert isinstance(tensor, torch.Tensor)
    # 原始 Tensor 没有 .copy()（评审实测 AttributeError），导出必须先转 numpy
    assert not hasattr(tensor, "copy")

    boundary = tensor.detach().cpu().numpy().astype("float32")
    assert isinstance(boundary, np.ndarray)
    assert boundary.shape == (T,)
    assert boundary.dtype == np.float32

    clip = np.zeros((T, 24, 3), dtype=np.float32)
    clip[:, :, 2] = 1.0
    ann = keypoints_to_pyskl(clip, frame_dir="roundtrip", label=0, label_name="sit")
    ann["boundary"] = boundary
    pkl_path = tmp_path / "roundtrip.pkl"
    save_pyskl_pickle([ann], str(pkl_path))

    loaded = load_pyskl_pickle(str(pkl_path))
    assert isinstance(loaded[0]["boundary"], np.ndarray)
    assert loaded[0]["boundary"].shape == (T,)

    ds = STGCNBCDataset(pkl_path=str(pkl_path), T=T)
    sample = ds[0]
    assert sample["keypoints"].shape == (T, 24, 3)
    assert sample["boundary"].shape == (T,)
    assert sample["label"].item() == 0


# ============================================================
# 4.13 标签空间漂移（MUSTFIX-4）
# ============================================================


def test_behavior_label_space_drift() -> None:
    from backend.app.services.training_export import (
        DB_ONLY_BEHAVIORS,
        MAPPED_BEHAVIORS,
        ML_ONLY_BEHAVIORS,
    )
    from backend.ml.behavior.constants import ALL_BEHAVIORS_22

    db_values = {b.value for b in BehaviorClass}
    ml_values = set(ALL_BEHAVIORS_22)

    assert len(MAPPED_BEHAVIORS) == 16
    assert set(MAPPED_BEHAVIORS) == db_values & ml_values
    assert DB_ONLY_BEHAVIORS == {
        "food_drive", "courage", "attention", "search_efficiency", "impulse_control", "gait",
    }
    assert ML_ONLY_BEHAVIORS == {
        "guard", "release", "retrieve", "jump", "scale", "search_blind",
    }
    assert len(DB_ONLY_BEHAVIORS) == 6 and len(ML_ONLY_BEHAVIORS) == 6


# ============================================================
# 4.3/4.4/4.5 导出语义
# ============================================================


@pytest.fixture
def export_env(tmp_path: Path, monkeypatch):
    """带 3 视频/5 行为的导出数据库 + 隔离 data 根."""
    from backend.app.services import training_export as te

    engine, factory = make_db(tmp_path)
    data_root = (tmp_path / "data").resolve()
    monkeypatch.setattr(te, "settings", SimpleNamespace(data_dir=data_root))

    with factory() as session:
        reviewer = make_handler(session, "reviewer", UserRole.ADMIN)
        video_a = make_video(session, "video_a", handler_id=reviewer.id)
        video_b = make_video(session, "video_b")
        video_c = make_video(session, "video_c")
        add_keypoints(session, video_a, range(0, 30))
        add_keypoints(session, video_b, range(0, 10))
        add_keypoints(session, video_c, range(0, 10))

        b_a1 = make_behavior(session, video_a, BehaviorClass.SIT, 0, 9)
        b_a2 = make_behavior(session, video_a, BehaviorClass.SIT, 10, 19)
        b_b1 = make_behavior(session, video_b, BehaviorClass.DOWN, 0, 9)
        b_c1 = make_behavior(session, video_c, BehaviorClass.BARK, 0, 9)
        b_c2 = make_behavior(session, video_c, BehaviorClass.HEEL, 0, 9)
        # DB-only P2 类：导出端必须拒绝（映射外）
        b_a3 = make_behavior(session, video_a, BehaviorClass.FOOD_DRIVE, 20, 29)

        task_a = make_task(session, video_a, AnnotationTaskStatus.IN_PROGRESS)
        task_b = make_task(session, video_b, AnnotationTaskStatus.IN_PROGRESS)
        task_c = make_task(session, video_c, AnnotationTaskStatus.PENDING)

        add_review(session, task_a, b_a1, "accepted", reviewed_by=reviewer.id)
        add_review(
            session, task_a, b_a2, "corrected",
            behavior_name="stand", start=12, end=19, reviewed_by=reviewer.id,
        )
        add_review(session, task_a, b_a3, "accepted", behavior_name="food_drive", reviewed_by=reviewer.id)
        add_review(session, task_b, b_b1, "accepted", reviewed_by=reviewer.id)
        add_review(session, task_c, b_c1, "rejected", reviewed_by=reviewer.id)
        # b_c2 未审核
        session.commit()

        ids = SimpleNamespace(
            reviewer_id=reviewer.id,
            a=video_a.id, b=video_b.id, c=video_c.id,
            b_a1=b_a1.id, b_a2=b_a2.id, b_a3=b_a3.id,
            b_b1=b_b1.id, b_c1=b_c1.id, b_c2=b_c2.id,
        )

    return SimpleNamespace(engine=engine, factory=factory, data_root=data_root, ids=ids)


def test_export_semantics(export_env) -> None:
    from backend.app.services.training_export import run_export
    from backend.ml.behavior.stgcn_bc.labels import BEHAVIOR_TO_IDX

    result = run_export(
        session_factory=export_env.factory,
        data_source="synthetic",
        seed=42,
        created_by_handler_id=export_env.ids.reviewer_id,
    )
    stats = result["stats"]
    manifest = result["manifest"]
    export_dir = Path(result["export_dir"])

    # rejected/unreviewed/映射外 均被排除
    assert stats["episodes_exported"] == 3
    assert stats["skipped"]["rejected"] == 1
    assert stats["skipped"]["unreviewed"] == 1
    assert stats["skipped"]["unmapped_behaviors"] == 1
    assert stats["review_counts"] == {
        "accepted": 3, "rejected": 1, "corrected": 1, "pending": 1,
    }

    records = [
        json.loads(line)
        for line in (export_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 3
    by_id = {r["behavior_id"]: r for r in records}
    assert export_env.ids.b_a1 in by_id
    assert export_env.ids.b_b1 in by_id
    assert export_env.ids.b_a2 in by_id
    assert export_env.ids.b_c1 not in by_id
    assert export_env.ids.b_c2 not in by_id
    assert export_env.ids.b_a3 not in by_id

    # corrected 生效值
    corrected = by_id[export_env.ids.b_a2]
    assert corrected["decision"] == "corrected"
    assert corrected["label_name"] == "stand"
    assert corrected["start_frame"] == 12 and corrected["end_frame"] == 19
    assert corrected["num_frames"] == 8

    # 16 类映射 + label 范围
    for record in records:
        assert record["label_name"] in BEHAVIOR_TO_IDX
        assert record["label"] == BEHAVIOR_TO_IDX[record["label_name"]]
        assert 0 <= record["label"] < 22

    # 按 video 分组划分、无泄漏
    train_videos = {r["video_id"] for r in records if r["split"] == "train"}
    val_videos = {r["video_id"] for r in records if r["split"] == "val"}
    assert train_videos and val_videos
    assert train_videos.isdisjoint(val_videos)
    assert stats["train_episodes"] + stats["val_episodes"] == 3

    # manifest 工程验收字段 + sha256 范围
    assert manifest["review_schema_version"] == 1
    assert manifest["engineering_only"] is True
    assert manifest["production_accepted"] is False
    assert manifest["data_source"] == "synthetic"
    assert "工程验收" in manifest["disclaimer"]
    for name in ("train.pkl", "val.pkl", "episodes.jsonl"):
        digest = hashlib.sha256((export_dir / name).read_bytes()).hexdigest()
        assert manifest["files"][name] == digest
    assert "manifest.json" not in manifest["files"]

    # 导出目录位于 data_root/finetune_exports 下
    assert Path(result["export_dir"]).resolve().is_relative_to(
        (export_env.data_root / "finetune_exports").resolve()
    )


def test_export_include_unreviewed(export_env) -> None:
    from backend.app.services.training_export import run_export

    result = run_export(
        session_factory=export_env.factory,
        include_unreviewed=True,
        data_source="synthetic",
        seed=42,
    )
    stats = result["stats"]
    assert stats["skipped"]["unreviewed"] == 0
    assert stats["episodes_exported"] == 4  # + b_c2（未审核 heel）
    assert stats["skipped"]["rejected"] == 1
    assert stats["skipped"]["unmapped_behaviors"] == 1


def test_export_requires_two_videos(tmp_path: Path, monkeypatch) -> None:
    from backend.app.services import training_export as te

    engine, factory = make_db(tmp_path)
    monkeypatch.setattr(te, "settings", SimpleNamespace(data_dir=(tmp_path / "data").resolve()))
    with factory() as session:
        handler = make_handler(session, "solo", UserRole.ADMIN)
        video = make_video(session, "solo", handler_id=handler.id)
        add_keypoints(session, video, range(0, 10))
        behavior = make_behavior(session, video, BehaviorClass.SIT, 0, 9)
        task = make_task(session, video, AnnotationTaskStatus.IN_PROGRESS)
        add_review(session, task, behavior, "accepted", reviewed_by=handler.id)
        session.commit()

    with pytest.raises(te.ExportError, match="至少需要 2 个"):
        te.run_export(session_factory=factory, data_source="synthetic")


def test_export_data_source_required(export_env) -> None:
    from backend.app.services.training_export import ExportError, run_export

    with pytest.raises(ExportError, match="data_source"):
        run_export(session_factory=export_env.factory, data_source="guessed")


# ============================================================
# 4.5 缺帧插值 / 坏形状跳过
# ============================================================


def test_missing_and_invalid_keypoints_counted(tmp_path: Path, monkeypatch) -> None:
    from backend.app.services import training_export as te

    engine, factory = make_db(tmp_path)
    monkeypatch.setattr(te, "settings", SimpleNamespace(data_dir=(tmp_path / "data").resolve()))

    with factory() as session:
        handler = make_handler(session, "reviewer2", UserRole.ADMIN)
        v_e = make_video(session, "good_e", handler_id=handler.id)
        v_f = make_video(session, "good_f", handler_id=handler.id)
        v_g = make_video(session, "missing_g", handler_id=handler.id)
        v_h = make_video(session, "partial_h", handler_id=handler.id)

        add_keypoints(session, v_e, range(0, 10))
        add_keypoints(session, v_f, range(0, 10))
        # v_g: 无 keypoints
        # v_h: 0-7 有行（其中 frame 3 形状非法），8-11 缺帧 → 7/12 覆盖
        add_keypoints(session, v_h, range(0, 8), bad_frames={3})

        behaviors = {}
        for key, video, cls in (
            ("e", v_e, BehaviorClass.SIT),
            ("f", v_f, BehaviorClass.DOWN),
            ("g", v_g, BehaviorClass.STAND),
            ("h", v_h, BehaviorClass.HEEL),
        ):
            behaviors[key] = make_behavior(session, video, cls, 0, 9 if key != "h" else 11)
            task = make_task(session, video, AnnotationTaskStatus.IN_PROGRESS)
            add_review(session, task, behaviors[key], "accepted", reviewed_by=handler.id)
        session.commit()

    # 固定划分：缺帧视频与部分缺帧视频放 train，正常视频 f 放 val
    def fake_split(video_ids, seed):
        return {v_e.id, v_g.id, v_h.id}, {v_f.id}

    monkeypatch.setattr(te, "_split_by_video", fake_split)

    result = te.run_export(session_factory=factory, data_source="synthetic", seed=42)
    stats = result["stats"]
    assert stats["skipped"]["missing_keypoints"] == 1
    assert stats["skipped"]["invalid_keypoints"] == 1
    assert stats["episodes_exported"] == 3  # e + h + f

    records = [
        json.loads(line)
        for line in (Path(result["export_dir"]) / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    h_record = next(r for r in records if r["behavior_id"] == behaviors["h"].id)
    assert h_record["num_frames"] == 12
    assert h_record["coverage"] == pytest.approx(7 / 12, abs=1e-3)

    # 插值后无 NaN（Dataset 取样可读）
    from backend.ml.behavior.stgcn_bc.dataset import STGCNBCDataset

    ds = STGCNBCDataset(pkl_path=str(Path(result["export_dir"]) / "train.pkl"), T=12)
    for i in range(len(ds)):
        sample = ds[i]
        assert np.isfinite(sample["keypoints"].numpy()).all()
        assert sample["boundary"].shape == (12,)


# ============================================================
# 4.2 tiny CPU 训练 + MLModel 注册 + 4.9/4.10 守卫
# ============================================================


@pytest.fixture
def training_env(export_env):
    return export_env


def test_tiny_cpu_finetune_registers_model(training_env, tmp_path: Path) -> None:
    from backend.app.services.training_export import run_export
    from backend.workers.finetune_tasks import run_finetune

    export = run_export(
        session_factory=training_env.factory,
        data_source="synthetic",
        seed=42,
        created_by_handler_id=training_env.ids.reviewer_id,
    )
    export_dir = Path(export["export_dir"])

    result = run_finetune(
        export_dir,
        epochs=1,
        batch_size=1,
        T=16,
        base_channels=8,
        num_stages=1,
        device="cpu",
        save_interval=1,
        session_factory=training_env.factory,
        project_root=tmp_path,
        data_root=training_env.data_root,
        runs_dir=tmp_path / "runs",
    )

    weights_dir = Path(result["run_dir"]) / "weights"
    assert (weights_dir / "best.pt").exists()
    assert (weights_dir / "last.pt").exists()
    assert Path(result["best_checkpoint"]).exists()

    with training_env.factory() as session:
        model = session.get(MLModel, result["model_id"])
        assert model is not None
        assert model.type == ModelType.BEHAVIOR
        assert model.is_active is False
        assert len(model.version) <= 32
        assert model.version.startswith("flywheel-")
        assert model.metrics_json["engineering_only"] is True
        assert model.metrics_json["production_accepted"] is False
        assert model.metrics_json["export_id"] == export["export_id"]
        assert "工程验收" in model.description

    assert result["engineering_only"] is True
    assert result["production_accepted"] is False


def test_finetune_path_guard_rejects_outside(tmp_path: Path) -> None:
    from backend.workers.finetune_tasks import FinetuneError, run_finetune

    outside = tmp_path / "outside_export"
    outside.mkdir()
    with pytest.raises(FinetuneError, match="finetune_exports"):
        run_finetune(outside, epochs=1, batch_size=1, data_root=tmp_path / "data")


def test_finetune_rejects_batch_larger_than_train(training_env, tmp_path: Path) -> None:
    from backend.app.services.training_export import run_export
    from backend.workers.finetune_tasks import run_finetune

    export = run_export(session_factory=training_env.factory, data_source="synthetic", seed=42)
    with pytest.raises(ValueError, match="batch_size"):
        run_finetune(
            Path(export["export_dir"]),
            epochs=1,
            batch_size=99,  # drop_last=True → 0 batch
            T=16,
            base_channels=8,
            num_stages=1,
            session_factory=training_env.factory,
            project_root=tmp_path,
            data_root=training_env.data_root,
            runs_dir=tmp_path / "runs",
        )


def test_model_version_length() -> None:
    from datetime import datetime, timezone

    from backend.workers.finetune_tasks import _model_version

    version = _model_version("exp-20261019235959-deadbeef")
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    assert version == f"flywheel-{today}-deadbeef"
    assert len(version) == 26 <= 32


# ============================================================
# 4.11 Celery 注册
# ============================================================


def test_celery_task_registered() -> None:
    from backend.workers.celery_app import celery_app

    import backend.workers.finetune_tasks  # noqa: F401

    assert "finetune.train_from_annotations" in celery_app.tasks
    assert "health.check" in celery_app.tasks


# ============================================================
# API 测试基础设施
# ============================================================


@pytest.fixture
def api_env(tmp_path: Path, monkeypatch):
    """权限/状态机/端点测试数据库."""
    from backend.app.services import training_export as te

    engine, factory = make_db(tmp_path)
    data_root = (tmp_path / "data").resolve()
    monkeypatch.setattr(te, "settings", SimpleNamespace(data_dir=data_root))

    with factory() as session:
        base = Base_(name="基地A", code="BASE-A")
        session.add(base)
        session.flush()

        admin = make_handler(session, "admin", UserRole.ADMIN)
        manager = make_handler(session, "manager", UserRole.MANAGER, base_id=base.id)
        outsider_manager = make_handler(session, "manager2", UserRole.MANAGER)
        researcher = make_handler(session, "researcher", UserRole.RESEARCHER, base_id=base.id)
        viewer = make_handler(session, "viewer", UserRole.VIEWER, base_id=base.id)
        owner = make_handler(session, "owner", UserRole.HANDLER, base_id=base.id)
        other = make_handler(session, "other", UserRole.HANDLER, base_id=base.id)

        dog = Dog(name="犬A", handler_id=owner.id, home_base_id=base.id)
        session.add(dog)
        session.flush()

        session_obj = TrainingSession(
            dog_id=dog.id,
            handler_id=owner.id,
            session_date=datetime.now(timezone.utc),
        )
        session.add(session_obj)
        session.flush()

        video = make_video(
            session, "api_video", handler_id=owner.id, dog_id=dog.id, session_id=session_obj.id
        )
        null_video = make_video(session, "null_video")
        add_keypoints(session, video, range(0, 10))
        behavior = make_behavior(session, video, BehaviorClass.SIT, 0, 9)
        null_behavior = make_behavior(session, null_video, BehaviorClass.DOWN, 0, 9)
        session.commit()

        ids = SimpleNamespace(
            base_id=base.id,
            admin=admin, manager=manager, outsider_manager=outsider_manager,
            researcher=researcher, viewer=viewer, owner=owner, other=other,
            dog=dog, video=video, null_video=null_video,
            behavior=behavior, null_behavior=null_behavior,
        )

    return SimpleNamespace(engine=engine, factory=factory, data_root=data_root, ids=ids)


def _client(factory, me) -> TestClient:
    from backend.app.core.deps import get_current_handler
    from backend.app.main import app
    from backend.app.services.training_export import get_sync_session_factory

    app.dependency_overrides[get_current_handler] = lambda: me
    app.dependency_overrides[get_sync_session_factory] = lambda: factory
    return TestClient(app)


def _clear_overrides() -> None:
    from backend.app.core.deps import get_current_handler
    from backend.app.main import app
    from backend.app.services.training_export import get_sync_session_factory

    app.dependency_overrides.pop(get_current_handler, None)
    app.dependency_overrides.pop(get_sync_session_factory, None)


# ============================================================
# 4.6 懒创建任务 + 4.8 状态机
# ============================================================


def test_review_lazily_creates_task(api_env) -> None:
    client = _client(api_env.factory, api_env.ids.manager)
    try:
        resp = client.post(
            f"/api/annotations/episodes/{api_env.ids.behavior.id}/review",
            json={"decision": "accepted"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["task_id"] is not None
        assert body["status"] == "in_progress"
        assert body["annotated_frames"] == 10

        with api_env.factory() as session:
            task = session.get(AnnotationTask, body["task_id"])
            assert task is not None
            assert task.video_id == api_env.ids.video.id
            assert task.status == AnnotationTaskStatus.IN_PROGRESS
            annotations = session.execute(
                select(Annotation).where(Annotation.task_id == task.id)
            ).scalars().all()
            assert len(annotations) == 1
            assert annotations[0].data_json["review_schema_version"] == 1
            assert annotations[0].data_json["original"] == {
                "behavior": "sit", "start_frame": 0, "end_frame": 9,
            }

        summary = client.get(f"/api/annotations/videos/{api_env.ids.video.id}/review-summary")
        assert summary.status_code == 200
        assert summary.json()["accepted"] == 1
        assert summary.json()["pending"] == 0
    finally:
        _clear_overrides()


def test_complete_state_machine_and_reopen(api_env) -> None:
    client = _client(api_env.factory, api_env.ids.manager)
    try:
        review = client.post(
            f"/api/annotations/episodes/{api_env.ids.behavior.id}/review",
            json={"decision": "accepted"},
        )
        assert review.status_code == 200, review.text
        task_id = review.json()["task_id"]

        completed = client.post(f"/api/annotations/tasks/{task_id}/complete")
        assert completed.status_code == 200, completed.text
        assert completed.json()["status"] == "completed"

        # 未 reopen → 409
        conflict = client.post(
            f"/api/annotations/episodes/{api_env.ids.behavior.id}/review",
            json={"decision": "corrected", "behavior": "stand"},
        )
        assert conflict.status_code == 409, conflict.text

        # reopen → 200 且回 IN_PROGRESS
        reopened = client.post(
            f"/api/annotations/episodes/{api_env.ids.behavior.id}/review",
            json={"decision": "corrected", "behavior": "stand", "reopen": True},
        )
        assert reopened.status_code == 200, reopened.text
        assert reopened.json()["status"] == "in_progress"

        # FAILED → 409
        with api_env.factory() as session:
            task = session.get(AnnotationTask, task_id)
            task.status = AnnotationTaskStatus.FAILED
            session.commit()
        failed = client.post(
            f"/api/annotations/episodes/{api_env.ids.behavior.id}/review",
            json={"decision": "accepted"},
        )
        assert failed.status_code == 409, failed.text
    finally:
        _clear_overrides()


def test_complete_requires_non_rejected(api_env) -> None:
    client = _client(api_env.factory, api_env.ids.manager)
    try:
        review = client.post(
            f"/api/annotations/episodes/{api_env.ids.behavior.id}/review",
            json={"decision": "rejected"},
        )
        assert review.status_code == 200, review.text
        task_id = review.json()["task_id"]

        resp = client.post(f"/api/annotations/tasks/{task_id}/complete")
        assert resp.status_code == 400, resp.text
        assert "非 rejected" in resp.json()["detail"]
    finally:
        _clear_overrides()


def test_corrected_requires_change(api_env) -> None:
    client = _client(api_env.factory, api_env.ids.manager)
    try:
        resp = client.post(
            f"/api/annotations/episodes/{api_env.ids.behavior.id}/review",
            json={"decision": "corrected", "behavior": "sit", "start_frame": 0, "end_frame": 9},
        )
        assert resp.status_code == 400, resp.text
    finally:
        _clear_overrides()


# ============================================================
# 4.7 权限矩阵
# ============================================================


def test_permission_matrix(api_env) -> None:
    from backend.app.services.training_export import review_access_decision

    ids = api_env.ids
    base_video = ids.video
    null_video = ids.null_video
    dog = ids.dog

    # ADMIN 全通
    assert review_access_decision(ids.admin, null_video, None, None, False, action="write")[0]
    # MANAGER 本基地（dog.home_base_id）
    assert review_access_decision(ids.manager, base_video, dog, None, False, action="write")[0]
    # MANAGER 无基地 / 跨基地
    assert not review_access_decision(ids.outsider_manager, base_video, dog, None, False)[0]
    ids.manager.base_id = None
    assert not review_access_decision(ids.manager, base_video, dog, None, False)[0]
    ids.manager.base_id = ids.base_id
    # RESEARCHER 本基地可审
    assert review_access_decision(ids.researcher, base_video, dog, None, False, action="write")[0]
    # VIEWER 只读
    assert review_access_decision(ids.viewer, base_video, dog, None, False, action="read")[0]
    assert not review_access_decision(ids.viewer, base_video, dog, None, False, action="write")[0]
    # HANDLER：video.handler / dog.handler / association / session
    assert review_access_decision(ids.owner, base_video, dog, None, False, action="write")[0]
    # other 既非 video.handler / dog.handler，也无 association/session → 拒绝
    assert not review_access_decision(ids.other, base_video, dog, None, False, action="write")[0]
    assert not review_access_decision(ids.other, null_video, None, None, False)[0]
    assert review_access_decision(ids.other, base_video, None, None, True)[0]  # association
    assert review_access_decision(ids.other, base_video, None, ids.other.id, False)[0]  # session
    # NULL 归属视频对 HANDLER 不可见
    assert not review_access_decision(ids.owner, null_video, None, None, False)[0]


def test_permission_matrix_handler_dog_owner(api_env) -> None:
    from backend.app.services.training_export import review_access_decision

    # 构造 dog.handler_id == other 的场景
    dog = api_env.ids.dog
    original = dog.handler_id
    dog.handler_id = api_env.ids.other.id
    try:
        allowed, reason = review_access_decision(
            api_env.ids.other, api_env.ids.video, dog, None, False, action="write"
        )
        assert allowed and reason == "dog_handler"
    finally:
        dog.handler_id = original


def test_api_permission_403_and_null_video(api_env) -> None:
    ids = api_env.ids

    # other handler 对 base_video：不是 video.handler、不是 dog.handler、无 association、session=owner → 403
    client = _client(api_env.factory, ids.other)
    try:
        resp = client.get(f"/api/annotations/videos/{ids.video.id}/episodes")
        assert resp.status_code == 403, resp.text

        queue = client.get("/api/annotations/review/queue")
        assert queue.status_code == 200
        # NULL 归属视频 + 非本人视频都不可见
        assert all(item["video_id"] != ids.null_video.id for item in queue.json()["items"])
    finally:
        _clear_overrides()

    # VIEWER 审核 → 403
    client = _client(api_env.factory, ids.viewer)
    try:
        resp = client.post(
            f"/api/annotations/episodes/{ids.behavior.id}/review",
            json={"decision": "accepted"},
        )
        assert resp.status_code == 403, resp.text
    finally:
        _clear_overrides()

    # HANDLER 对 NULL 归属视频 → 403
    client = _client(api_env.factory, ids.owner)
    try:
        resp = client.get(f"/api/annotations/videos/{ids.null_video.id}/episodes")
        assert resp.status_code == 403, resp.text
        resp = client.post(
            f"/api/annotations/episodes/{ids.null_behavior.id}/review",
            json={"decision": "accepted"},
        )
        assert resp.status_code == 403, resp.text
    finally:
        _clear_overrides()


def test_admin_can_review_null_video(api_env) -> None:
    client = _client(api_env.factory, api_env.ids.admin)
    try:
        resp = client.post(
            f"/api/annotations/episodes/{api_env.ids.null_behavior.id}/review",
            json={"decision": "accepted"},
        )
        assert resp.status_code == 200, resp.text
    finally:
        _clear_overrides()


# ============================================================
# 4.12 / 4.14 / 4.15 触发端点
# ============================================================


def _make_fake_export(data_root: Path) -> str:
    export_id = "exp-test-0001"
    export_dir = data_root / "finetune_exports" / export_id
    export_dir.mkdir(parents=True)
    (export_dir / "train.pkl").write_bytes(b"train")
    (export_dir / "val.pkl").write_bytes(b"val")
    (export_dir / "episodes.jsonl").write_text("", encoding="utf-8")
    manifest = {
        "export_id": export_id,
        "created_by_handler_id": 1,
        "data_source": "synthetic",
        "engineering_only": True,
        "production_accepted": False,
        "files": {
            "train.pkl": hashlib.sha256(b"train").hexdigest(),
            "val.pkl": hashlib.sha256(b"val").hexdigest(),
            "episodes.jsonl": hashlib.sha256(b"").hexdigest(),
        },
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return export_id


def test_trigger_missing_export_id_400(api_env) -> None:
    client = _client(api_env.factory, api_env.ids.admin)
    try:
        resp = client.post("/api/finetune/trigger", json={})
        assert resp.status_code == 400, resp.text
        assert "export_id" in resp.json()["detail"]

        resp = client.post("/api/finetune/trigger")
        assert resp.status_code == 400, resp.text
    finally:
        _clear_overrides()


def test_trigger_unknown_export_404(api_env) -> None:
    client = _client(api_env.factory, api_env.ids.admin)
    try:
        resp = client.post("/api/finetune/trigger", json={"export_id": "exp-does-not-exist"})
        assert resp.status_code == 404, resp.text
    finally:
        _clear_overrides()


def test_trigger_broker_failure_503(api_env, monkeypatch) -> None:
    from backend.app.api import finetune as finetune_api

    export_id = _make_fake_export(api_env.data_root)

    def boom() -> None:
        raise RuntimeError("broker down")

    monkeypatch.setattr(finetune_api, "_probe_broker", boom)

    client = _client(api_env.factory, api_env.ids.admin)
    try:
        resp = client.post("/api/finetune/trigger", json={"export_id": export_id})
        assert resp.status_code == 503, resp.text
        assert "broker" in resp.json()["detail"].lower()
    finally:
        _clear_overrides()


def test_trigger_success_202_and_task_404(api_env, monkeypatch) -> None:
    from backend.app.api import finetune as finetune_api
    from backend.workers import finetune_tasks

    export_id = _make_fake_export(api_env.data_root)

    monkeypatch.setattr(finetune_api, "_probe_broker", lambda: None)
    monkeypatch.setattr(finetune_api, "_registry_set", lambda task_id, payload: None)
    monkeypatch.setattr(finetune_api, "_registry_get", lambda task_id: None)
    monkeypatch.setattr(
        finetune_tasks.train_from_annotations,
        "delay",
        lambda **kwargs: SimpleNamespace(id="task-fake-123"),
    )

    client = _client(api_env.factory, api_env.ids.admin)
    try:
        resp = client.post("/api/finetune/trigger", json={"export_id": export_id, "epochs": 1})
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["task_id"] == "task-fake-123"
        assert body["export_id"] == export_id
        assert body["production_accepted"] is False

        # registry 无记录 → 404（区分“不存在”与 PENDING）
        missing = client.get("/api/finetune/tasks/task-not-registered")
        assert missing.status_code == 404, missing.text
    finally:
        _clear_overrides()


def test_export_endpoint_and_exports_list(api_env, monkeypatch) -> None:
    # 准备带审核的数据：manager 本基地视频 + 另一个视频
    with api_env.factory() as session:
        video2 = make_video(session, "api_video2", handler_id=api_env.ids.manager.id)
        add_keypoints(session, video2, range(0, 10))
        behavior2 = make_behavior(session, video2, BehaviorClass.STAND, 0, 9)
        task2 = make_task(session, video2, AnnotationTaskStatus.IN_PROGRESS)
        add_review(session, task2, behavior2, "accepted", reviewed_by=api_env.ids.manager.id)

        behavior1 = session.get(Behavior, api_env.ids.behavior.id)
        task1 = make_task(
            session, api_env.ids.video, AnnotationTaskStatus.IN_PROGRESS
        )
        add_review(session, task1, behavior1, "accepted", reviewed_by=api_env.ids.manager.id)
        session.commit()

    client = _client(api_env.factory, api_env.ids.manager)
    try:
        resp = client.post(
            "/api/finetune/export",
            json={"data_source": "synthetic", "seed": 42},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["export_id"]
        assert body["manifest"]["production_accepted"] is False
        assert body["stats"]["episodes_exported"] == 2

        listing = client.get("/api/finetune/exports")
        assert listing.status_code == 200, listing.text
        assert listing.json()["total"] >= 1
        assert all(
            item["created_by_handler_id"] == api_env.ids.manager.id
            for item in listing.json()["items"]
        )
    finally:
        _clear_overrides()


def test_export_limit_413(export_env, monkeypatch) -> None:
    from backend.app.services import training_export as te

    monkeypatch.setattr(te, "EXPORT_MAX_EPISODES", 1)
    with pytest.raises(te.ExportLimitExceeded, match="上限"):
        te.run_export(session_factory=export_env.factory, data_source="synthetic")


class _AsyncLikeSession:
    """把同步 SQLite session 包成 async 接口（仅用于 /status 聚合测试；无 aiosqlite）."""

    def __init__(self, session: Session) -> None:
        self._session = session

    async def execute(self, stmt):
        return self._session.execute(stmt)


def test_status_and_pipeline_include_behavior_models(api_env) -> None:
    """MUSTFIX-9：/status 键保持小写契约；/pipeline 统计 POSE+BEHAVIOR."""
    from backend.app.core.database import get_db
    from backend.app.main import app
    from backend.app.models.ml_model import ModelFramework

    with api_env.factory() as session:
        session.add(
            AnnotationTask(
                video_id=api_env.ids.video.id,
                handler_id=api_env.ids.owner.id,
                status=AnnotationTaskStatus.COMPLETED,
                total_frames=10,
                annotation_types=["behavior"],
            )
        )
        session.add(
            MLModel(
                name="pose", version="v1", type=ModelType.POSE,
                framework=ModelFramework.PYTORCH, storage_path="models/pose.pt",
                is_active=False,
            )
        )
        session.add(
            MLModel(
                name="behavior", version="v2", type=ModelType.BEHAVIOR,
                framework=ModelFramework.PYTORCH, storage_path="models/behavior.pt",
                is_active=True,
            )
        )
        session.commit()

    client = _client(api_env.factory, api_env.ids.admin)

    async def _get_db():
        session = api_env.factory()
        try:
            yield _AsyncLikeSession(session)
        finally:
            session.close()

    app.dependency_overrides[get_db] = _get_db
    try:
        status = client.get("/api/finetune/status")
        assert status.status_code == 200, status.text
        body = status.json()
        assert body["running"] is False
        # ORM 聚合键显式 .value → 旧小写契约
        assert body["pipeline"]["annotation_tasks"].get("completed") == 1
        assert body["pipeline"]["models_by_type"] == {"pose": 1, "behavior": 1}
        types = {m["type"] for m in body["pipeline"]["models"]}
        assert types == {"pose", "behavior"}

        pipeline = client.get("/api/finetune/pipeline")
        assert pipeline.status_code == 200, pipeline.text
        data = pipeline.json()
        assert data["stages"]["train"]["total_models"] == 2
        assert data["stages"]["deploy"]["active_models"] == 1
        assert data["production_accepted"] is False
    finally:
        app.dependency_overrides.pop(get_db, None)
        _clear_overrides()


def test_export_endpoint_limit_maps_413(api_env, monkeypatch) -> None:
    from backend.app.services import training_export as te

    def always_one(*args, **kwargs):
        raise te.ExportLimitExceeded("超过上限 500")

    monkeypatch.setattr(te, "EXPORT_MAX_EPISODES", 1)
    monkeypatch.setattr("backend.app.api.finetune.run_export", always_one)
    client = _client(api_env.factory, api_env.ids.admin)
    try:
        resp = client.post("/api/finetune/export", json={"data_source": "synthetic"})
        assert resp.status_code == 413, resp.text
    finally:
        _clear_overrides()


def test_export_requester_base_filter(export_env) -> None:
    """MANAGER 只能导出本基地/本人归属的视频（不跨基地读取）."""
    from backend.app.services.training_export import ExportError, run_export

    with export_env.factory() as session:
        base = Base_(name="基地B", code="BASE-B")
        session.add(base)
        session.flush()
        manager = make_handler(session, "manager_b", UserRole.MANAGER, base_id=base.id)
        session.commit()
        manager_id = manager.id

    # export_env 的视频都不属于基地B → 导出必须拒绝
    with pytest.raises(ExportError, match="无权导出"):
        run_export(
            session_factory=export_env.factory,
            data_source="synthetic",
            created_by_handler_id=manager_id,
            requester=_reload_handler(export_env.factory, manager_id),
        )


def _reload_handler(factory, handler_id: int):
    session = factory()
    try:
        handler = session.get(Handler, handler_id)
        session.expunge(handler)
        return handler
    finally:
        session.close()


def test_export_endpoint_rejects_video_as_403(api_env) -> None:
    client = _client(api_env.factory, api_env.ids.viewer)
    try:
        resp = client.post("/api/finetune/export", json={"data_source": "synthetic"})
        assert resp.status_code == 403, resp.text
    finally:
        _clear_overrides()


def test_export_endpoint_requires_data_source(api_env) -> None:
    client = _client(api_env.factory, api_env.ids.admin)
    try:
        resp = client.post("/api/finetune/export", json={})
        assert resp.status_code == 422, resp.text
    finally:
        _clear_overrides()
