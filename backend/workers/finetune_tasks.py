"""CPU 微调 Celery 任务（P1-2 数据飞轮工程链）.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: P1-2

职责:
    ``finetune.train_from_annotations(export_id, ...)``:
        manifest 校验 → STGCNBCDataset(train/val) → CPU 训练 →
        best.pt/last.pt 定位 → 同步 ORM 注册 MLModel(is_active=False)

工程验收声明:
    本任务只证明“审核→导出→训练→注册”链路可运行。
    合成数据仅用于工程验收，**工程验收 ≠ 生产验收**，不代表识别精度或生产可用。
    不设精度门禁；模型 is_active=False，激活仍由人走 POST /api/models/{id}/activate。

安全边界:
    - 顶层只 import celery/settings/ORM；torch 相关全部延迟到任务函数内。
    - pickle 仅允许 trusted-local：export_dir 必须位于
      ``{data_root}/finetune_exports/`` 下，且 manifest 声明的文件不得越界。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy.orm import Session

from backend.app.core.config import PROJECT_ROOT, settings
from backend.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

ENGINEERING_ONLY = True
PRODUCTION_ACCEPTED = False

FINETUNE_DISCLAIMER = (
    "本次微调使用合成/审核导出数据，仅验证链路可运行；工程验收 ≠ 生产验收，"
    "不代表识别精度或生产可用。模型 is_active=False。"
)


class FinetuneError(RuntimeError):
    """微调业务错误。"""


def _resolve_export_dir(export_dir: Path, data_root: Path) -> Path:
    """校验并返回导出目录（必须在 data_root/finetune_exports 下）。"""
    root = (Path(data_root) / "finetune_exports").resolve()
    resolved = Path(export_dir).resolve()
    if not resolved.is_relative_to(root):
        raise FinetuneError(
            f"导出目录必须是 {root} 的子目录（trusted-local 限制），实际 {resolved}"
        )
    if not resolved.is_dir():
        raise FinetuneError(f"导出目录不存在: {resolved}")
    return resolved


def _load_manifest(export_dir: Path) -> dict:
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        raise FinetuneError(f"manifest.json 不存在: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise FinetuneError("manifest.json 格式非法（应为 object）")
    return manifest


def _validated_assets(export_dir: Path, manifest: dict) -> dict[str, Path]:
    """校验 manifest.files 声明的文件都在 export_dir 内且存在."""
    files = manifest.get("files") or {}
    if not isinstance(files, dict) or not files:
        raise FinetuneError("manifest.files 缺失")
    resolved: dict[str, Path] = {}
    for name in ("train.pkl", "val.pkl"):
        if name not in files:
            raise FinetuneError(f"manifest.files 缺少 {name}")
        if Path(name).name != name or "/" in name or "\\" in name:
            raise FinetuneError(f"非法文件名: {name!r}")
        path = (export_dir / name).resolve()
        if path.parent != export_dir:
            raise FinetuneError(f"文件越界: {name!r}")
        if not path.exists():
            raise FinetuneError(f"manifest 声明的文件不存在: {name}")
        resolved[name] = path
    return resolved


def _default_session_factory() -> Callable[[], Session]:
    """运行时解析同步 session 工厂（不在 import 时绑定）。"""
    from backend.app.core.database_sync import SyncSessionLocal

    return SyncSessionLocal


def _model_version(export_id: str) -> str:
    """``flywheel-{YYYYMMDD}-{export_id 末 8 位 hex}`` → 26 字符（≤32）."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = (export_id or "")[-8:]
    version = f"flywheel-{today}-{suffix}"
    assert len(version) <= 32, f"MLModel.version 超长 ({len(version)}): {version}"
    return version


def _register_model(
    *,
    version: str,
    storage_path: str,
    metrics: dict,
    session_factory: Optional[Callable[[], Session]] = None,
) -> int:
    """同步 ORM 注册 MLModel（is_active=False）."""
    from backend.app.models.ml_model import MLModel, ModelFramework, ModelType

    factory = session_factory or _default_session_factory()
    session: Session = factory()
    try:
        model = MLModel(
            name="stgcn-bc-finetune",
            version=version,
            type=ModelType.BEHAVIOR,
            framework=ModelFramework.PYTORCH,
            storage_path=storage_path,
            is_active=False,
            metrics_json=metrics,
            description=(
                "ST-GCN+BC 审核数据微调（CPU 工程验收）。"
                "工程验收≠生产验收；不代表识别精度或生产可用，激活需人工决定。"
            ),
        )
        session.add(model)
        session.commit()
        model_id = int(model.id)
        logger.info("MLModel 已注册: id=%d version=%s path=%s", model_id, version, storage_path)
        return model_id
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def run_finetune(
    export_dir: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 8,
    T: int = 30,
    base_channels: int = 8,
    num_stages: int = 1,
    device: str = "cpu",
    save_interval: int = 1,
    session_factory: Optional[Callable[[], Session]] = None,
    project_root: Path = PROJECT_ROOT,
    data_root: Optional[Path] = None,
    runs_dir: Optional[Path] = None,
) -> dict:
    """执行一次 CPU 微调并注册模型.

    Args:
        export_dir: 导出目录（必须位于 data_root/finetune_exports 下）
        epochs / batch_size / T / base_channels / num_stages: 训练超参
        device: 默认 cpu
        save_interval: 固定 1（1-epoch 也会写 last.pt）
        session_factory: 同步 session 工厂（测试注入）
        project_root: 项目根（storage_path 相对化）
        data_root: 数据根（默认 settings.data_dir）
        runs_dir: 训练输出根（默认 project_root/runs）

    Returns:
        dict: model_id / version / run_dir / best_checkpoint / metrics
    """
    # 延迟导入 torch / ML（顶层保持轻量）
    import torch

    from backend.app.models.ml_model import ModelType  # noqa: F401  (registries)
    from backend.ml.behavior.stgcn_bc import STGCNBCDataset
    from backend.ml.behavior.stgcn_bc.labels import NUM_BEHAVIORS
    from backend.ml.behavior.stgcn_bc.model import build_stgcn_bc
    from backend.ml.behavior.stgcn_bc.trainer import STGCNBCTrainer, TrainConfig

    root = Path(data_root) if data_root is not None else Path(settings.data_dir)
    export_path = _resolve_export_dir(Path(export_dir), root)
    manifest = _load_manifest(export_path)
    assets = _validated_assets(export_path, manifest)

    if manifest.get("production_accepted") is not False:
        raise FinetuneError("manifest.production_accepted 必须显式为 false（工程验收）")

    train_ds = STGCNBCDataset(pkl_path=str(assets["train.pkl"]), T=T, augment=True)
    val_ds = STGCNBCDataset(pkl_path=str(assets["val.pkl"]), T=T, augment=False)

    # MUSTFIX-13: drop_last=True 下必须保证至少 1 个 batch；val 至少 1 样本
    if len(train_ds) < max(1, batch_size):
        raise ValueError(
            f"train 样本数 {len(train_ds)} < batch_size {batch_size}，"
            "训练将产生 0 个 batch。请降低 batch_size 或补充审核数据。"
        )
    if len(val_ds) < 1:
        raise ValueError("val 样本为空：至少需要 1 个验证视频")

    run_dir = Path(runs_dir) if runs_dir is not None else Path(project_root) / "runs"
    run_dir = run_dir / f"finetune_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model = build_stgcn_bc(
        in_channels=3,
        num_classes=NUM_BEHAVIORS,
        base_channels=base_channels,
        num_stages=num_stages,
    )
    config = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        output_dir=str(weights_dir),
        save_interval=save_interval,
        use_amp=False,
        early_stopping=False,
        warmup_epochs=min(1, max(0, epochs - 1)),
        val_interval=1,
    )
    trainer = STGCNBCTrainer(model, train_ds, val_ds, config)
    summary = trainer.fit(epochs=epochs)

    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    checkpoint_fallback = False
    if not best_path.exists() and last_path.exists():
        # trainer 在 val_acc 恒为 0 时不会写 best；回退 last
        best_path = last_path
        checkpoint_fallback = True
    if not best_path.exists():
        raise RuntimeError(
            f"训练结束但找不到 checkpoint：{weights_dir}/best.pt 与 last.pt 均不存在"
        )

    try:
        storage_path = str(best_path.resolve().relative_to(Path(project_root).resolve()))
        storage_path_base = "project_root"
    except ValueError:
        storage_path = str(best_path.resolve())
        storage_path_base = "absolute"
    if len(storage_path) > 512:
        raise FinetuneError("MLModel.storage_path 超过 512 字符")

    version = _model_version(str(manifest.get("export_id") or ""))
    metrics = {
        "export_id": manifest.get("export_id"),
        "data_source": manifest.get("data_source"),
        "engineering_only": ENGINEERING_ONLY,
        "production_accepted": PRODUCTION_ACCEPTED,
        "disclaimer": FINETUNE_DISCLAIMER,
        "storage_path_base": storage_path_base,
        "best_val_acc": summary.get("best_val_acc"),
        "best_epoch": summary.get("best_epoch"),
        "checkpoint_fallback_to_last": checkpoint_fallback,
        "total_epochs_trained": summary.get("total_epochs_trained"),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "batch_size": batch_size,
        "T": T,
        "base_channels": base_channels,
        "num_stages": num_stages,
        "device": device,
        "run_dir": str(run_dir),
    }

    model_id = _register_model(
        version=version,
        storage_path=storage_path,
        metrics=metrics,
        session_factory=session_factory,
    )

    result = {
        "model_id": model_id,
        "version": version,
        "export_id": manifest.get("export_id"),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "metrics": metrics,
        "engineering_only": ENGINEERING_ONLY,
        "production_accepted": PRODUCTION_ACCEPTED,
        "disclaimer": FINETUNE_DISCLAIMER,
    }
    logger.info("微调完成: %s", {k: result[k] for k in ("model_id", "version", "run_dir")})
    return result


@celery_app.task(name="finetune.train_from_annotations", bind=True)
def train_from_annotations(
    self,
    export_id: str,
    epochs: int = 20,
    batch_size: int = 8,
    T: int = 30,
    base_channels: int = 64,
    num_stages: int = 10,
    device: str = "cpu",
) -> dict:
    """Celery 入口：从导出目录执行 CPU 微调（不 autoretry，失败直接抛）."""
    root = Path(settings.data_dir)
    export_dir = (root / "finetune_exports" / export_id).resolve()
    if not export_dir.is_relative_to((root / "finetune_exports").resolve()):
        raise FinetuneError(f"非法 export_id: {export_id!r}")

    logger.info(
        "finetune.train_from_annotations 开始: export_id=%s epochs=%d batch_size=%d",
        export_id,
        epochs,
        batch_size,
    )
    return run_finetune(
        export_dir,
        epochs=epochs,
        batch_size=batch_size,
        T=T,
        base_channels=base_channels,
        num_stages=num_stages,
        device=device,
        data_root=root,
    )


__all__ = [
    "FINETUNE_DISCLAIMER",
    "FinetuneError",
    "run_finetune",
    "train_from_annotations",
]
