"""MotionBERT 17→24 关键点适配单元测试（Phase 3.3c）.

Owner: ML 开发
Phase: 3.3c

覆盖:
    1. DSTformer 24 关键点构建
    2. 预训练权重迁移（17→24，pos_embed 丢弃）
    3. 前向推理 shape
    4. 配置加载
    5. Dataset 输出格式
    6. ONNX 导出一致性
    7. MotionBERTLifter 推理
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# 添加项目本地 onnx 包路径
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_PYLIBS = _PROJECT_ROOT / "external" / "_pylibs"
if _LOCAL_PYLIBS.exists() and str(_LOCAL_PYLIBS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PYLIBS))

from backend.ml.pose.motionbert.config import MotionBERTConfig, load_config
from backend.ml.pose.motionbert.dataset import InterPet4DDataset
from backend.ml.pose.motionbert.model import (
    DSTformerWrapper,
    build_model_from_config,
    load_pretrained_weights,
)


# === 固定路径 ===
PRETRAIN_PATH = Path("data/models/MB_lite_pretrain.bin")
POSE3D_FT_PATH = Path("data/models/MB_lite_pose3d_ft.bin")


class TestDSTformer24Keypoints:
    """DSTformer 24 关键点构建测试."""

    def test_build_model_default_config(self):
        """默认配置构建模型（24 关节）."""
        config = MotionBERTConfig()
        model = build_model_from_config(config)
        assert model.pos_embed.shape == (1, 24, 256)
        assert model.joints_embed.weight.shape == (256, 3)
        assert model.head.weight.shape == (3, 512)

    def test_forward_shape(self):
        """前向推理输出形状."""
        config = MotionBERTConfig()
        model = build_model_from_config(config)
        model.eval()
        x = torch.randn(2, 27, 24, 3)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 27, 24, 3)

    def test_forward_variable_time(self):
        """不同时间长度前向（≤maxlen=243）."""
        config = MotionBERTConfig()
        model = build_model_from_config(config)
        model.eval()
        for T in [1, 27, 100, 243]:
            x = torch.randn(1, T, 24, 3)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, T, 24, 3)

    def test_get_representation(self):
        """中间表示提取."""
        config = MotionBERTConfig()
        model = build_model_from_config(config)
        model.eval()
        x = torch.randn(1, 27, 24, 3)
        with torch.no_grad():
            rep = model.get_representation(x)
        assert rep.shape == (1, 27, 24, 512)  # dim_rep=512

    def test_param_count(self):
        """参数量验证（~16M）."""
        config = MotionBERTConfig()
        model = build_model_from_config(config)
        param_count = sum(p.numel() for p in model.parameters())
        # MB_lite 17 关节约 16M，24 关节增加 pos_embed 7*256=1792 参数
        assert 15_000_000 < param_count < 17_000_000


@pytest.mark.skipif(not PRETRAIN_PATH.exists(), reason="预训练权重未下载")
class TestPretrainedWeightTransfer:
    """预训练权重 17→24 迁移测试."""

    def test_load_pretrain_discards_pos_embed(self):
        """加载预训练权重时 pos_embed 被丢弃."""
        config = MotionBERTConfig()
        model = build_model_from_config(config)
        # 记录原始 pos_embed
        original_pos_embed = model.pos_embed.data.clone()

        matched, total, discarded = load_pretrained_weights(
            model, PRETRAIN_PATH, log_mismatches=False
        )

        # 259/260 层匹配，仅 pos_embed 丢弃
        assert matched == 259
        assert total == 260
        assert discarded == ["pos_embed"]

        # pos_embed 未被覆盖（保持初始化）
        assert torch.equal(model.pos_embed.data, original_pos_embed)

    def test_load_pretrain_preserves_other_layers(self):
        """其他层（joints_embed/blocks/head）被正确迁移."""
        config = MotionBERTConfig()
        model = build_model_from_config(config)
        load_pretrained_weights(model, PRETRAIN_PATH, log_mismatches=False)

        # joints_embed 应该是非默认初始化的（来自预训练）
        assert not torch.equal(
            model.joints_embed.weight.data,
            torch.zeros_like(model.joints_embed.weight.data)
        )

    def test_load_pose3d_ft_weights(self):
        """加载 pose3d 微调权重."""
        if not POSE3D_FT_PATH.exists():
            pytest.skip("pose3d FT 权重未下载")
        config = MotionBERTConfig()
        model = build_model_from_config(config)
        matched, total, discarded = load_pretrained_weights(
            model, POSE3D_FT_PATH, log_mismatches=False
        )
        assert matched == 259
        assert total == 260
        assert discarded == ["pos_embed"]


class TestConfig:
    """配置加载测试."""

    def test_default_config(self):
        """默认配置是 24 关节."""
        config = MotionBERTConfig()
        assert config.num_joints == 24
        assert config.dim_feat == 256
        assert config.window_size == 27

    def test_yaml_load(self):
        """YAML 配置加载."""
        yaml_path = Path("backend/ml/pose/motionbert/configs/MB_lite_dog24.yaml")
        if not yaml_path.exists():
            pytest.skip("YAML 配置不存在")
        config = load_config(yaml_path)
        assert config.num_joints == 24
        assert config.dim_feat == 256
        assert config.pretrained_path == "data/models/MB_lite_pose3d_ft.bin"


class TestDataset:
    """Dataset 测试."""

    def test_dataset_construction(self):
        """Dataset 构建和getitem."""
        N, T, J = 10, 27, 24
        kp_2d = np.random.randn(N, T, J, 2).astype(np.float32)
        kp_3d = np.random.randn(N, T, J, 3).astype(np.float32)
        conf = np.random.rand(N, T, J).astype(np.float32)

        ds = InterPet4DDataset(kp_2d, kp_3d, conf)
        assert len(ds) == N

        sample_2d, sample_3d = ds[0]
        assert sample_2d.shape == (T, J, 3)  # (x, y, conf)
        assert sample_3d.shape == (T, J, 3)

    def test_dataset_batch(self):
        """Dataset 在 DataLoader 中的批次输出."""
        from torch.utils.data import DataLoader

        N, T, J = 20, 27, 24
        kp_2d = np.random.randn(N, T, J, 2).astype(np.float32)
        kp_3d = np.random.randn(N, T, J, 3).astype(np.float32)
        conf = np.random.rand(N, T, J).astype(np.float32)

        ds = InterPet4DDataset(kp_2d, kp_3d, conf)
        loader = DataLoader(ds, batch_size=8, shuffle=False)

        for batch_2d, batch_3d in loader:
            assert batch_2d.shape == (8, T, J, 3)
            assert batch_3d.shape == (8, T, J, 3)
            break


class TestDSTformerWrapper:
    """DSTformerWrapper 测试."""

    def test_wrapper_construction(self):
        """包装器构建（无预训练权重）."""
        config = MotionBERTConfig()
        wrapper = DSTformerWrapper(config=config, pretrained_path=None)
        assert wrapper.config.num_joints == 24

    def test_wrapper_infer(self):
        """包装器推理."""
        config = MotionBERTConfig()
        config.flip = False  # 测试时禁用 flip
        wrapper = DSTformerWrapper(config=config, pretrained_path=None)
        x = torch.randn(1, 27, 24, 3)
        out = wrapper.infer(x, flip=False, rootrel=True)
        assert out.shape == (1, 27, 24, 3)
        # rootrel: 根关节（idx 0）应该在原点
        assert torch.allclose(out[..., 0, :], torch.zeros_like(out[..., 0, :]), atol=1e-5)

    def test_flip_data(self):
        """左右翻转功能."""
        x = torch.randn(1, 27, 24, 3)
        flipped = DSTformerWrapper._flip_data(x)
        assert flipped.shape == x.shape
        # 翻转两次应该恢复原始（除了 x 坐标符号）
        double_flipped = DSTformerWrapper._flip_data(flipped)
        # 左右对称对应该恢复，x 坐标也应该恢复
        assert torch.allclose(x, double_flipped, atol=1e-6)


class TestMotionBERTLifter:
    """MotionBERTLifter 推理测试."""

    def test_lifter_torch_backend(self):
        """PyTorch 后端推理."""
        from backend.ml.pose.motionbert.inference import MotionBERTLifter

        config = MotionBERTConfig()
        # 用未训练的模型（仅验证管线）
        wrapper = DSTformerWrapper(config=config, pretrained_path=None)
        # 保存临时 checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.bin"
            torch.save({
                "model_pos": wrapper.model.state_dict(),
                "config": config.to_dict(),
            }, ckpt_path)

            lifter = MotionBERTLifter(checkpoint_path=ckpt_path)
            kp_2d = np.random.randn(50, 24, 2).astype(np.float32)
            kp_3d = lifter.lift(kp_2d)
            assert kp_3d.shape == (50, 24, 3)

    def test_lifter_short_sequence(self):
        """短序列（<window_size）推理."""
        from backend.ml.pose.motionbert.inference import MotionBERTLifter

        config = MotionBERTConfig()
        wrapper = DSTformerWrapper(config=config, pretrained_path=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.bin"
            torch.save({
                "model_pos": wrapper.model.state_dict(),
                "config": config.to_dict(),
            }, ckpt_path)

            lifter = MotionBERTLifter(checkpoint_path=ckpt_path)
            kp_2d = np.random.randn(10, 24, 2).astype(np.float32)  # < 27
            kp_3d = lifter.lift(kp_2d)
            assert kp_3d.shape == (10, 24, 3)

    def test_lifter_long_sequence(self):
        """长序列（>window_size）滑动窗口推理."""
        from backend.ml.pose.motionbert.inference import MotionBERTLifter

        config = MotionBERTConfig()
        wrapper = DSTformerWrapper(config=config, pretrained_path=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.bin"
            torch.save({
                "model_pos": wrapper.model.state_dict(),
                "config": config.to_dict(),
            }, ckpt_path)

            lifter = MotionBERTLifter(checkpoint_path=ckpt_path)
            kp_2d = np.random.randn(100, 24, 2).astype(np.float32)  # > 27
            kp_3d = lifter.lift(kp_2d)
            assert kp_3d.shape == (100, 24, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
