"""backend.ml.pose.train 训练脚本单元测试.

Owner: ML 开发
Phase: 1.0
依据: dev-docs/stages/phase-1.md §1.0c

不实际训练（太慢），只验证参数、yaml、入口逻辑。
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml


@pytest.mark.fast
class TestTrainScript:
    """训练脚本结构与参数验证。"""

    def test_train_module_importable(self) -> None:
        """train 模块可导入。"""
        mod = importlib.import_module("backend.ml.pose.train")
        assert hasattr(mod, "train_model")
        assert hasattr(mod, "evaluate_model")
        assert hasattr(mod, "verify_dataset")
        assert hasattr(mod, "parse_args")

    def test_default_params_match_adr0003(self) -> None:
        """默认参数与 ADR 0003 + phase-1.md §1.0c 一致。"""
        from backend.ml.pose import train as train_mod
        assert train_mod.DEFAULT_EPOCHS == 100
        assert train_mod.DEFAULT_IMGSZ == 640
        assert train_mod.DEFAULT_BATCH == 16
        assert train_mod.DEFAULT_LR0 == 0.001
        assert train_mod.DEFAULT_BOX_LOSS == 7.5
        assert train_mod.DEFAULT_CLS_LOSS == 0.5
        assert train_mod.DEFAULT_POSE_LOSS == 12.0
        assert train_mod.DEFAULT_KOBJ_LOSS == 1.0
        # 数据增强关闭项
        assert train_mod.DEFAULT_SHEAR == 0.0
        assert train_mod.DEFAULT_PERSPECTIVE == 0.0
        assert train_mod.DEFAULT_FLIPUD == 0.0
        assert train_mod.DEFAULT_MIXUP == 0.0
        assert train_mod.DEFAULT_COPY_PASTE == 0.0

    def test_dataset_yaml_exists(self) -> None:
        """data/dog-pose.yaml 存在且配置正确。"""
        yaml_path = Path(__file__).resolve().parents[3] / "data" / "dog-pose.yaml"
        assert yaml_path.exists(), f"yaml 不存在: {yaml_path}"
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["kpt_shape"] == [24, 3]
        assert cfg["train"] == "images/train"
        assert cfg["val"] == "images/val"
        assert cfg["names"][0] == "dog"
        assert len(cfg["kpt_names"][0]) == 24

    def test_default_model_is_yolo26n_pose(self) -> None:
        """默认模型为 yolo26n-pose.pt。"""
        from backend.ml.pose import train as train_mod
        assert train_mod.DEFAULT_MODEL == "yolo26n-pose.pt"

    def test_verify_dataset_returns_none_when_exists(self) -> None:
        """verify_dataset 在真实数据集存在时不报错（数据集已下载）。"""
        from backend.ml.pose import train as train_mod
        yaml_path = Path(__file__).resolve().parents[3] / "data" / "dog-pose.yaml"
        if not yaml_path.exists():
            pytest.skip(f"yaml 不存在（数据集未下载）: {yaml_path}")
        # 真实数据集已下载（train 6773 / val 1703），verify_dataset 应不抛异常
        train_mod.verify_dataset(yaml_path)


@pytest.mark.fast
class TestDownloadDataset:
    """下载脚本逻辑验证（不实际下载）。"""

    def test_download_module_importable(self) -> None:
        """download_dataset 模块可导入。"""
        mod = importlib.import_module("backend.ml.pose.download_dataset")
        assert hasattr(mod, "download_and_extract")
        assert hasattr(mod, "verify_dataset")
        assert hasattr(mod, "dataset_exists")
        assert hasattr(mod, "DATASET_URL")
        assert hasattr(mod, "EXPECTED_TRAIN_IMAGES")
        assert hasattr(mod, "EXPECTED_VAL_IMAGES")

    def test_dataset_url_correct(self) -> None:
        """数据集 URL 正确。"""
        from backend.ml.pose.download_dataset import DATASET_URL
        assert "dog-pose.zip" in DATASET_URL
        assert "ultralytics" in DATASET_URL.lower()

    def test_expected_counts(self) -> None:
        """期望图像数量正确。"""
        from backend.ml.pose.download_dataset import (
            EXPECTED_TRAIN_IMAGES,
            EXPECTED_VAL_IMAGES,
        )
        assert EXPECTED_TRAIN_IMAGES == 6773
        assert EXPECTED_VAL_IMAGES == 1703
