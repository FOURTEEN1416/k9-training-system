"""ST-GCN+BC 部署集成单元测试（Phase 3.1e）.

Owner: ML 开发
Phase: 3.1e

覆盖:
    - ONNX 导出 + 一致性验证
    - STGCNBCInferer 双后端推理（PyTorch + ONNX）
    - BehaviorRecognizer 四种部署模式（shadow/vote/primary_stgcn/rule_only）
    - 滑动窗口推理 + episode 切分
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from backend.ml.behavior.constants import ALL_BEHAVIORS_22
from backend.ml.behavior.rule_engine import BehaviorEpisode
from backend.ml.behavior.router import BehaviorRecognizer, DeployMode, ShadowComparison
from backend.ml.behavior.stgcn_bc.export_onnx import export_onnx
from backend.ml.behavior.stgcn_bc.inference import STGCNBCInferer
from backend.ml.behavior.stgcn_bc.model import build_stgcn_bc
from backend.ml.behavior.stgcn_bc.trainer import STGCNBCTrainer, TrainConfig


# ============================================================
# 测试夹具
# ============================================================

@pytest.fixture
def synthetic_checkpoint(tmp_path) -> Path:
    """创建合成数据训练的 checkpoint（1 epoch 快速训练）."""
    from backend.ml.behavior.stgcn_bc.dataset import make_synthetic_dataset, STGCNBCDataset

    samples = make_synthetic_dataset(samples_per_class=2, T=30, seed=42)
    train_ds = STGCNBCDataset(samples=samples, T=30, augment=False)
    val_ds = STGCNBCDataset(samples=samples, T=30, augment=False)

    model = build_stgcn_bc(in_channels=3, base_channels=32, num_stages=4)
    config = TrainConfig(
        lr=1e-3,
        epochs=1,
        batch_size=8,
        num_workers=0,
        warmup_epochs=0,
        patience=1,
        use_amp=False,
        device="cpu",
        output_dir=str(tmp_path / "ckpt"),
    )
    trainer = STGCNBCTrainer(model, train_ds, val_ds, config=config)
    trainer.fit()

    return tmp_path / "ckpt" / "best.pt"


@pytest.fixture
def synthetic_onnx(synthetic_checkpoint, tmp_path) -> Path:
    """导出 ONNX 模型."""
    output_path = tmp_path / "stgcn_bc_test.onnx"
    result = export_onnx(
        synthetic_checkpoint,
        output_path,
        in_channels=3,
        base_channels=32,
        num_stages=4,
        window_size=30,
        verify=True,
    )
    assert result["verification"]["passed"], (
        f"ONNX 一致性验证失败: {result['verification']}"
    )
    return output_path


@pytest.fixture
def synthetic_keypoints() -> np.ndarray:
    """合成关键点序列 (T=60, 24, 3)."""
    rng = np.random.default_rng(42)
    # 模拟犬类关键点：基础站立姿态 + 随机扰动
    base = np.zeros((1, 24, 3), dtype=np.float32)
    base[0, 22] = [0.5, 0.3, 0.0]  # withers
    base[0, 12] = [0.4, 0.3, 0.0]  # tail_start
    base[0, 0] = [0.6, 0.5, 0.0]   # front_left_paw
    base[0, 6] = [0.4, 0.5, 0.0]   # front_right_paw
    base[0, 3] = [0.6, 0.1, 0.0]   # rear_left_paw
    base[0, 9] = [0.4, 0.1, 0.0]   # rear_right_paw

    T = 60
    kpts = np.tile(base, (T, 1, 1))
    # 添加随机扰动
    kpts += rng.normal(0, 0.02, kpts.shape).astype(np.float32)
    return kpts


# ============================================================
# ONNX 导出测试
# ============================================================

class TestExportOnnx:
    """ONNX 导出测试."""

    def test_export_creates_file(self, synthetic_checkpoint, tmp_path):
        """导出 ONNX 文件存在且非空."""
        output_path = tmp_path / "test.onnx"
        result = export_onnx(
            synthetic_checkpoint, output_path,
            in_channels=3, base_channels=32, num_stages=4,
            verify=False,
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert result["opset_version"] == 17
        assert result["output_path"] == str(output_path)

    def test_verification_passed(self, synthetic_onnx):
        """ONNX 一致性验证通过（夹具已验证）."""
        assert synthetic_onnx.exists()

    def test_dynamic_batch(self, synthetic_onnx):
        """动态 batch 维度：支持 B=4 输入."""
        import onnxruntime as ort
        sess = ort.InferenceSession(str(synthetic_onnx), providers=["CPUExecutionProvider"])
        batch_input = np.random.randn(4, 30, 24, 3).astype(np.float32)
        cls_logits, boundary_logits = sess.run(None, {"keypoints": batch_input})
        assert cls_logits.shape == (4, 22)
        assert boundary_logits.shape[0] == 4

    def test_dynamic_time(self, synthetic_onnx):
        """动态 time 维度：支持 T=45 输入."""
        import onnxruntime as ort
        sess = ort.InferenceSession(str(synthetic_onnx), providers=["CPUExecutionProvider"])
        time_input = np.random.randn(1, 45, 24, 3).astype(np.float32)
        cls_logits, boundary_logits = sess.run(None, {"keypoints": time_input})
        assert cls_logits.shape == (1, 22)
        # boundary 时间维度因 backbone 下采样与 T 相关
        assert boundary_logits.shape[0] == 1


# ============================================================
# STGCNBCInferer 测试
# ============================================================

class TestSTGCNBCInferer:
    """ST-GCN+BC 推理器测试."""

    def test_torch_backend(self, synthetic_checkpoint, synthetic_keypoints):
        """PyTorch 后端推理."""
        inferer = STGCNBCInferer(
            checkpoint_path=synthetic_checkpoint,
            in_channels=3, base_channels=32, num_stages=4,
            device="cpu",
        )
        assert inferer.backend == "torch"
        episodes = inferer.predict(synthetic_keypoints, fps=30.0)
        assert isinstance(episodes, list)
        # 合成数据可能识别到任意行为，但不应崩溃
        for ep in episodes:
            assert isinstance(ep, BehaviorEpisode)
            assert ep.behavior in ALL_BEHAVIORS_22
            assert 0 <= ep.confidence <= 1.0
            assert ep.start_frame <= ep.end_frame

    def test_onnx_backend(self, synthetic_onnx, synthetic_keypoints):
        """ONNX 后端推理."""
        inferer = STGCNBCInferer(onnx_path=synthetic_onnx, device="cpu")
        assert inferer.backend == "onnx"
        episodes = inferer.predict(synthetic_keypoints, fps=30.0)
        assert isinstance(episodes, list)
        for ep in episodes:
            assert isinstance(ep, BehaviorEpisode)
            assert ep.behavior in ALL_BEHAVIORS_22

    def test_torch_onnx_consistency(self, synthetic_checkpoint, synthetic_onnx, synthetic_keypoints):
        """PyTorch 与 ONNX 后端推理结果一致."""
        torch_inferer = STGCNBCInferer(
            checkpoint_path=synthetic_checkpoint,
            in_channels=3, base_channels=32, num_stages=4,
            device="cpu",
        )
        onnx_inferer = STGCNBCInferer(onnx_path=synthetic_onnx, device="cpu")

        torch_episodes = torch_inferer.predict(synthetic_keypoints, fps=30.0)
        onnx_episodes = onnx_inferer.predict(synthetic_keypoints, fps=30.0)

        # 行为类别序列应一致（数量可能因边界检测微小差异略不同）
        torch_behaviors = {ep.behavior for ep in torch_episodes}
        onnx_behaviors = {ep.behavior for ep in onnx_episodes}
        assert torch_behaviors == onnx_behaviors or len(torch_behaviors.symmetric_difference(onnx_behaviors)) <= 1

    def test_short_sequence(self, synthetic_onnx):
        """短序列（T < window_size）推理."""
        inferer = STGCNBCInferer(onnx_path=synthetic_onnx, device="cpu")
        short_kpts = np.random.randn(10, 24, 3).astype(np.float32)
        episodes = inferer.predict(short_kpts, fps=30.0)
        assert isinstance(episodes, list)

    def test_invalid_input(self, synthetic_onnx):
        """无效输入形状报错."""
        inferer = STGCNBCInferer(onnx_path=synthetic_onnx, device="cpu")
        with pytest.raises(ValueError, match="必须是"):
            inferer.predict(np.random.randn(24, 3), fps=30.0)
        with pytest.raises(ValueError, match="关键点数"):
            inferer.predict(np.random.randn(30, 17, 3), fps=30.0)

    def test_requires_model(self):
        """无模型路径报错."""
        with pytest.raises(ValueError, match="必须提供"):
            STGCNBCInferer()


# ============================================================
# BehaviorRecognizer 测试
# ============================================================

class TestBehaviorRecognizer:
    """双轨路由层测试."""

    def test_rule_only_mode(self, synthetic_keypoints):
        """RULE_ONLY 模式仅调用规则引擎."""
        recognizer = BehaviorRecognizer(mode=DeployMode.RULE_ONLY)
        assert recognizer.mode == DeployMode.RULE_ONLY
        episodes = recognizer.recognize(synthetic_keypoints, fps=30.0)
        assert isinstance(episodes, list)

    def test_shadow_mode_with_stgcn(self, synthetic_onnx, synthetic_keypoints):
        """SHADOW 模式: ST-GCN+BC 推理 + 规则引擎返回 + 对比记录."""
        inferer = STGCNBCInferer(onnx_path=synthetic_onnx, device="cpu")
        recognizer = BehaviorRecognizer(
            mode=DeployMode.SHADOW,
            stgcn_inferer=inferer,
        )
        episodes = recognizer.recognize(synthetic_keypoints, fps=30.0)
        assert isinstance(episodes, list)

        # 影子模式应记录对比
        comparison = recognizer.last_comparison
        assert comparison is not None
        assert isinstance(comparison, ShadowComparison)
        assert comparison.stgcn_count >= 0
        assert comparison.rule_count >= 0

    def test_shadow_mode_without_stgcn(self, synthetic_keypoints):
        """SHADOW 模式无 ST-GCN+BC 时降级 RULE_ONLY."""
        recognizer = BehaviorRecognizer(mode=DeployMode.SHADOW, stgcn_inferer=None)
        episodes = recognizer.recognize(synthetic_keypoints, fps=30.0)
        assert isinstance(episodes, list)

    def test_primary_stgcn_mode(self, synthetic_onnx, synthetic_keypoints):
        """PRIMARY_STGCN 模式: ST-GCN+BC 主."""
        inferer = STGCNBCInferer(onnx_path=synthetic_onnx, device="cpu")
        recognizer = BehaviorRecognizer(
            mode=DeployMode.PRIMARY_STGCN,
            stgcn_inferer=inferer,
        )
        episodes = recognizer.recognize(synthetic_keypoints, fps=30.0)
        assert isinstance(episodes, list)
        # ST-GCN+BC 输出的 episode 应带 metadata.detector
        for ep in episodes:
            assert ep.metadata.get("detector") == "stgcn_bc"

    def test_primary_stgcn_fallback(self, synthetic_keypoints):
        """PRIMARY_STGCN 模式 ST-GCN+BC 失败时降级规则引擎."""
        # 创建一个会失败的 mock inferer
        class FailingInferer:
            def predict(self, *args, **kwargs):
                raise RuntimeError("模拟推理失败")

        recognizer = BehaviorRecognizer(
            mode=DeployMode.PRIMARY_STGCN,
            stgcn_inferer=FailingInferer(),
        )
        episodes = recognizer.recognize(synthetic_keypoints, fps=30.0)
        # 应降级到规则引擎，不抛异常
        assert isinstance(episodes, list)

    def test_vote_mode(self, synthetic_onnx, synthetic_keypoints):
        """VOTE 模式: 双轨投票融合."""
        inferer = STGCNBCInferer(onnx_path=synthetic_onnx, device="cpu")
        recognizer = BehaviorRecognizer(
            mode=DeployMode.VOTE,
            stgcn_inferer=inferer,
            vote_conf_threshold=0.3,
        )
        episodes = recognizer.recognize(synthetic_keypoints, fps=30.0)
        assert isinstance(episodes, list)
        # 投票结果应按时间排序
        for i in range(len(episodes) - 1):
            assert episodes[i].start_frame <= episodes[i + 1].start_frame

    def test_mode_from_string(self):
        """字符串模式转换."""
        recognizer = BehaviorRecognizer(mode="shadow")
        assert recognizer.mode == DeployMode.SHADOW

        recognizer = BehaviorRecognizer(mode="primary_stgcn")
        assert recognizer.mode == DeployMode.PRIMARY_STGCN

    def test_lazy_rule_engine(self):
        """规则引擎懒加载."""
        recognizer = BehaviorRecognizer(mode=DeployMode.RULE_ONLY)
        assert recognizer._rule_engine is None
        # 访问 property 触发懒加载
        _ = recognizer.rule_engine
        assert recognizer._rule_engine is not None


# ============================================================
# Episode 切分测试
# ============================================================

class TestEpisodeSplit:
    """episode 切分逻辑测试."""

    def test_single_window_no_boundary(self, synthetic_onnx):
        """单窗口无边界: 整个窗口一个 episode."""
        inferer = STGCNBCInferer(onnx_path=synthetic_onnx, device="cpu")
        # 构造稳定序列（无边界变化）
        kpts = np.zeros((20, 24, 3), dtype=np.float32)
        kpts[:, 22] = [0.5, 0.3, 0.0]  # withers
        episodes = inferer.predict(kpts, fps=30.0)
        assert isinstance(episodes, list)

    def test_min_episode_len_filter(self, synthetic_onnx):
        """短 episode 被过滤."""
        inferer = STGCNBCInferer(
            onnx_path=synthetic_onnx,
            min_episode_len=100,  # 极大值，过滤所有 episode
            device="cpu",
        )
        kpts = np.random.randn(60, 24, 3).astype(np.float32)
        episodes = inferer.predict(kpts, fps=30.0)
        # 所有 episode 长度 < 100 应被过滤
        for ep in episodes:
            assert ep.duration_frames >= 100
