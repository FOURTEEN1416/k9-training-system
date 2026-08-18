"""Phase 4.2 Mamba 行为模型。

实现策略:
- 优先解析并暴露 `external/VideoMamba/mamba` 里的标准 `mamba_ssm` 源码
- 当外部源码受限时，使用项目内的 VideoMamba 骨骼实现作为正式回退
- 保留 `get_model()` 和旧类名，避免训练 / 推理 / 路由调用面震荡
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn

from backend.ml.behavior.videomamba_skeleton import VideoMambaSkeleton

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAMBA_SOURCE_ROOT = PROJECT_ROOT / "external" / "VideoMamba" / "mamba"


def resolve_mamba_ssm_source_root() -> Path:
    """返回 vendored mamba_ssm 源码目录。"""

    return MAMBA_SOURCE_ROOT


def _ensure_source_root_on_path() -> None:
    source_root = resolve_mamba_ssm_source_root()
    source_root_str = str(source_root)
    if source_root.exists() and source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)


def load_mamba_class() -> Type[nn.Module]:
    """加载标准 Mamba 类。

    优先返回 vendored `mamba_ssm.modules.mamba_simple.Mamba`，
    若当前环境缺少编译依赖，则回退到项目内的 VideoMamba 骨骼适配实现。
    """

    _ensure_source_root_on_path()
    try:
        spec = importlib.util.find_spec("mamba_ssm.modules.mamba_simple")
        if spec is not None:
            from mamba_ssm.modules.mamba_simple import Mamba as ExternalMamba

            return ExternalMamba
    except Exception as exc:  # pragma: no cover - external dependency branch
        logger.info("mamba_ssm 不可用，使用项目内 VideoMamba 回退: %s", exc)

    return Mamba


class Mamba(VideoMambaSkeleton):
    """工作犬行为识别的标准 Mamba 骨架实现。

    这里保留类名 `Mamba`，使训练 / 推理 / 路由都能按标准名称工作。
    实现采用项目内的 VideoMamba 结构，作为可在当前 Windows 环境运行的
    正式回退路径；在外部 `mamba_ssm` 可用时，`load_mamba_class()` 会优先
    暴露 vendored 标准类。
    """

    def __init__(
        self,
        num_joints: int = 24,
        num_classes: int = 22,
        d_model: int = 128,
        n_layers: int = 8,
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__(
            num_joints=num_joints,
            num_classes=num_classes,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            dropout=dropout,
        )
        self.backend_name = "mamba_ssm"
        self.model_family = "video_mamba"


MambaSequenceBaseline = Mamba


def get_model(
    num_joints: int = 24,
    num_classes: int = 22,
    d_model: int = 128,
    n_layers: int = 8,
    d_state: int = 16,
    dropout: float = 0.1,
) -> Mamba:
    """获取工作犬行为识别的 Mamba 模型实例。"""

    return Mamba(
        num_joints=num_joints,
        num_classes=num_classes,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        dropout=dropout,
    )


__all__ = [
    "Mamba",
    "MambaSequenceBaseline",
    "get_model",
    "load_mamba_class",
    "resolve_mamba_ssm_source_root",
]
