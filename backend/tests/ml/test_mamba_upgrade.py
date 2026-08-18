from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from backend.ml.behavior.mamba_inference import MambaInferer
from backend.ml.behavior.mamba_sequence import get_model, load_mamba_class, resolve_mamba_ssm_source_root
from backend.app.core.config import Settings
from backend.workers import tasks


def test_resolve_mamba_ssm_source_root_points_to_videomamba() -> None:
    root = resolve_mamba_ssm_source_root()
    assert root == Path(r"D:\Desktop\k9-training-system\external\VideoMamba\mamba")
    assert root.name == "mamba"


def test_load_mamba_class_returns_real_mamba() -> None:
    mamba_cls = load_mamba_class()
    assert mamba_cls.__name__ in {"Mamba", "MambaSequenceBaseline"}
    assert mamba_cls.__module__.startswith("mamba_ssm") or mamba_cls.__module__.endswith("mamba_sequence")


def test_get_model_uses_requested_dimension_and_runs_forward() -> None:
    model = get_model(num_classes=22, num_joints=24, d_model=96, n_layers=2, d_state=8, dropout=0.0)

    assert model.d_model == 96
    assert getattr(model, "backend_name", "") in {"mamba_ssm", "video_mamba"}

    x = torch.randn(2, 30, 24, 3)
    logits = model(x)

    assert logits.shape == (2, 22)


def test_mamba_inferer_loads_checkpoint_with_requested_shape(tmp_path: Path) -> None:
    model = get_model(num_classes=22, num_joints=24, d_model=96, n_layers=2, d_state=8, dropout=0.0)
    checkpoint_path = tmp_path / "mamba_ssm.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    inferer = MambaInferer(
        checkpoint_path=checkpoint_path,
        num_joints=24,
        num_classes=22,
        d_model=96,
        n_layers=2,
        d_state=8,
        device="cpu",
    )

    episodes = inferer.predict(np.zeros((12, 24, 3), dtype=np.float32), fps=30.0)

    assert isinstance(episodes, list)


def test_settings_support_mamba_deploy_mode() -> None:
    settings = Settings(behavior_deploy_mode="mamba_shadow")
    assert settings.behavior_deploy_mode == "mamba_shadow"


def test_mamba_inferer_singleton_returns_none_when_models_missing(monkeypatch) -> None:
    monkeypatch.setattr(tasks, "_mamba_inferer", None)
    monkeypatch.setattr(tasks, "_resolve_mamba_model_path", lambda: None)
    assert tasks._mamba_inferer_singleton() is None


def test_mamba_bc_inferer_singleton_returns_none_when_models_missing(monkeypatch) -> None:
    monkeypatch.setattr(tasks, "_mamba_bc_inferer", None)
    monkeypatch.setattr(tasks, "_resolve_mamba_bc_path", lambda: None)
    assert tasks._mamba_bc_inferer_singleton() is None


def test_behavior_recognizer_uses_mamba_inferer_when_mode_requests_mamba(monkeypatch) -> None:
    created = {}

    class DummyInferer:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(tasks, "_behavior_recognizer", None)
    monkeypatch.setattr(tasks.settings, "behavior_deploy_mode", "mamba_shadow", raising=False)
    monkeypatch.setattr(tasks, "_resolve_mamba_model_path", lambda: "dummy.pt")
    monkeypatch.setattr(tasks, "_resolve_mamba_bc_path", lambda: None)
    monkeypatch.setattr(tasks, "MambaInferer", DummyInferer)
    monkeypatch.setattr(tasks, "MambaBCInferer", DummyInferer)

    recognizer = tasks._behavior_recognizer_singleton()

    assert recognizer.mode.value == "mamba_shadow"
    assert created == {"checkpoint_path": "dummy.pt"}


def test_behavior_recognizer_uses_mamba_bc_inferer_when_mode_requests_mamba_bc(monkeypatch) -> None:
    created = {}

    class DummyInferer:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(tasks, "_behavior_recognizer", None)
    monkeypatch.setattr(tasks.settings, "behavior_deploy_mode", "mamba_bc_only", raising=False)
    monkeypatch.setattr(tasks, "_resolve_mamba_model_path", lambda: None)
    monkeypatch.setattr(tasks, "_resolve_mamba_bc_path", lambda: "dummy_bc.pt")
    monkeypatch.setattr(tasks, "MambaInferer", DummyInferer)
    monkeypatch.setattr(tasks, "MambaBCInferer", DummyInferer)

    recognizer = tasks._behavior_recognizer_singleton()

    assert recognizer.mode.value == "mamba_bc_only"
    assert created == {"checkpoint_path": "dummy_bc.pt"}
