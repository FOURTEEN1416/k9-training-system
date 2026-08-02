"""ST-GCN+BC 模块单元测试（Phase 3.1b + 3.1c）.

Owner: ML 开发
Phase: 3.1b + 3.1c

测试覆盖:
    1. K9Graph 24 节点拓扑正确性
        - 节点数/根节点/边数
        - parent 数组一致性
        - 邻接矩阵对称性
        - 骨骼流定义
    2. 22 类标签映射
        - P0/P1/P2 索引连续性
        - 双向映射一致性
        - FCI-IGP 阶段覆盖
    3. 数据格式适配器
        - YOLO26-pose → pyskl 转换（2D/3D）
        - pyskl → YOLO26-pose 逆转换
        - 骨骼流计算
        - 运动流计算
        - 归一化
        - 批量标注构建
    4. ST-GCN 主干 (3.1c)
        - 邻接矩阵构建
        - UnitGCN / UnitTCN / MSTCN forward
        - STGCNBlock forward + residual
        - STGCN 主干 forward shape
    5. BC 头 + 联合损失 (3.1c)
        - BCHead forward
        - generate_boundary_labels 软标签
        - STGCNBCLoss 联合损失
    6. STGCNBC 整体模型 (3.1c)
        - forward + predict
        - 反向传播
        - 参数量合理
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.ml.behavior.constants import (
    NUM_KEYPOINTS,
    WITHERS, THROAT, NOSE, CHIN,
    TAIL_START, TAIL_END,
    FRONT_LEFT_PAW, FRONT_LEFT_KNEE, FRONT_LEFT_ELBOW,
    REAR_RIGHT_PAW, REAR_RIGHT_KNEE, REAR_RIGHT_ELBOW,
    P0_BEHAVIORS, P1_BEHAVIORS, P2_BEHAVIORS,
)
from backend.ml.behavior.stgcn_bc import (
    K9Graph,
    BEHAVIOR_TO_IDX,
    IDX_TO_BEHAVIOR,
    NUM_BEHAVIORS,
    P2_BEHAVIORS as P2_LABELS,
    keypoints_to_pyskl,
    pyskl_to_keypoints,
    compute_bone_flow,
    compute_motion_flow,
    normalize_keypoints,
    build_pyskl_annotation,
)
from backend.ml.behavior.stgcn_bc.labels import (
    P0_IDX, P1_IDX, P2_IDX, LAYER_LABELS, FCI_IGP_STAGE,
    get_behavior_idx, get_behavior_name, get_layer, get_fci_igp_stage,
)


# ===== K9Graph 拓扑测试 =====

class TestK9Graph:
    """K9Graph 24 节点犬类骨架拓扑测试。"""

    def test_num_nodes(self):
        g = K9Graph()
        assert g.num_nodes == 24
        assert g.num_nodes == NUM_KEYPOINTS

    def test_root_is_withers(self):
        g = K9Graph()
        assert g.root == WITHERS  # 22

    def test_edge_count(self):
        """24 节点应有 23 条边（树结构，N-1 边）。"""
        g = K9Graph()
        assert len(g.outward) == 23
        assert len(g.inward) == 23

    def test_parent_array_consistency(self):
        """parent 数组与 outward 边一致，根节点 parent=-1。"""
        g = K9Graph()
        assert g.parent[WITHERS] == -1  # 根
        # 所有非根节点 parent >= 0
        for v in range(g.num_nodes):
            if v != WITHERS:
                assert g.parent[v] >= 0, f"节点 {v} ({g.NODE_NAMES[v]}) parent 未设置"

    def test_adjacency_symmetric(self):
        """邻接矩阵应对称（无向图）。"""
        g = K9Graph()
        adj = g.adjacency
        assert adj.shape == (24, 24)
        assert np.array_equal(adj, adj.T)

    def test_adjacency_self_loop(self):
        """邻接矩阵对角线为 1（自环，pyskl 标准）。"""
        g = K9Graph()
        for i in range(24):
            assert g.adjacency[i, i] == 1

    def test_bones_definition(self):
        """骨骼流定义 (child, parent) 与 outward 一致。"""
        g = K9Graph()
        bones = g.get_bones()
        assert len(bones) == 23
        for (child, parent), (p, c) in zip(bones, g.outward):
            assert child == c
            assert parent == p

    def test_head_chain_topology(self):
        """头部链: withers → throat → nose → {chin, ears, eyes}."""
        g = K9Graph()
        assert g.parent[THROAT] == WITHERS
        assert g.parent[NOSE] == THROAT
        assert g.parent[CHIN] == NOSE

    def test_limb_chains(self):
        """四肢链: withers → elbow → knee → paw."""
        g = K9Graph()
        # 前左肢
        assert g.parent[FRONT_LEFT_ELBOW] == WITHERS
        assert g.parent[FRONT_LEFT_KNEE] == FRONT_LEFT_ELBOW
        assert g.parent[FRONT_LEFT_PAW] == FRONT_LEFT_KNEE
        # 后右肢
        assert g.parent[REAR_RIGHT_ELBOW] == WITHERS
        assert g.parent[REAR_RIGHT_KNEE] == REAR_RIGHT_ELBOW
        assert g.parent[REAR_RIGHT_PAW] == REAR_RIGHT_KNEE

    def test_tail_chain(self):
        """尾部: withers → tail_start → tail_end."""
        g = K9Graph()
        assert g.parent[TAIL_START] == WITHERS
        assert g.parent[TAIL_END] == TAIL_START

    def test_summary_and_repr(self):
        """摘要和 repr 不报错。"""
        g = K9Graph()
        s = g.summary()
        assert "K9Graph" in s
        assert "24" in s
        r = repr(g)
        assert "K9Graph" in r


# ===== 22 类标签映射测试 =====

class TestLabels:
    """22 类行为标签映射测试。"""

    def test_total_classes(self):
        assert NUM_BEHAVIORS == 22
        assert len(BEHAVIOR_TO_IDX) == 22
        assert len(IDX_TO_BEHAVIOR) == 22

    def test_p0_indices_continuous(self):
        """P0 索引 0-7 连续。"""
        assert P0_IDX == list(range(0, 8))

    def test_p1_indices_continuous(self):
        """P1 索引 8-15 连续。"""
        assert P1_IDX == list(range(8, 16))

    def test_p2_indices_continuous(self):
        """P2 索引 16-21 连续。"""
        assert P2_IDX == list(range(16, 22))

    def test_bidirectional_mapping(self):
        """行为名 ↔ 索引双向映射一致。"""
        for name, idx in BEHAVIOR_TO_IDX.items():
            assert IDX_TO_BEHAVIOR[idx] == name

    def test_layer_labels(self):
        """层级标签: 8 P0 + 8 P1 + 6 P2 = 22。"""
        assert len(LAYER_LABELS) == 22
        assert LAYER_LABELS.count("P0") == 8
        assert LAYER_LABELS.count("P1") == 8
        assert LAYER_LABELS.count("P2") == 6

    def test_get_behavior_idx(self):
        assert get_behavior_idx("sit") == 0
        assert get_behavior_idx("search_blind") == 21

    def test_get_behavior_name(self):
        assert get_behavior_name(0) == "sit"
        assert get_behavior_name(21) == "search_blind"

    def test_get_layer(self):
        assert get_layer(0) == "P0"
        assert get_layer(7) == "P0"
        assert get_layer(8) == "P1"
        assert get_layer(15) == "P1"
        assert get_layer(16) == "P2"
        assert get_layer(21) == "P2"

    def test_fci_igp_stage_coverage(self):
        """FCI-IGP 阶段覆盖全部 22 类。"""
        assert len(FCI_IGP_STAGE) == 22
        # A/B/C 三阶段都应有行为
        stages = set(FCI_IGP_STAGE.values())
        assert stages == {"A", "B", "C"}

    def test_get_fci_igp_stage(self):
        """已知行为的 IGP 阶段。"""
        assert get_fci_igp_stage("track") == "A"  # 追踪
        assert get_fci_igp_stage("sit") == "B"    # 服从
        assert get_fci_igp_stage("apprehend") == "C"  # 护卫
        assert get_fci_igp_stage("guard") == "C"  # 护卫


# ===== 数据格式适配器测试 =====

class TestDataAdapter:
    """YOLO26-pose ↔ pyskl 格式转换测试。"""

    @pytest.fixture
    def sample_keypoints(self) -> np.ndarray:
        """合成 (T=10, 24, 3) 关键点序列。"""
        np.random.seed(42)
        T = 10
        kpt = np.random.rand(T, NUM_KEYPOINTS, 3).astype(np.float32)
        # conf 通道设为 [0, 1]
        kpt[..., 2] = np.random.uniform(0.5, 0.99, size=(T, NUM_KEYPOINTS)).astype(np.float32)
        return kpt

    def test_to_pyskl_2d_shape(self, sample_keypoints):
        """2D 模式: keypoint (1, T, 24, 2) + keypoint_score (1, T, 24)。"""
        ann = keypoints_to_pyskl(sample_keypoints, frame_dir="test_clip", label=0, mode="2d")
        assert ann["keypoint"].shape == (1, 10, 24, 2)
        assert ann["keypoint_score"].shape == (1, 10, 24)
        assert ann["total_frames"] == 10
        assert ann["label"] == 0
        assert ann["frame_dir"] == "test_clip"
        assert ann["ann_info"]["num_persons"] == 1
        assert ann["ann_info"]["num_clases"] == 22
        assert ann["ann_info"]["num_joints"] == 24

    def test_to_pyskl_3d_shape(self, sample_keypoints):
        """3D 模式: keypoint (1, T, 24, 3)，无 keypoint_score。"""
        ann = keypoints_to_pyskl(sample_keypoints, frame_dir="test_3d", label=5, mode="3d")
        assert ann["keypoint"].shape == (1, 10, 24, 3)
        assert "keypoint_score" not in ann
        assert ann["label"] == 5

    def test_to_pyskl_no_label(self, sample_keypoints):
        """无标签模式（推理用）：label=-1。"""
        ann = keypoints_to_pyskl(sample_keypoints, frame_dir="infer_clip", mode="2d")
        assert ann["label"] == -1

    def test_to_pyskl_label_name(self, sample_keypoints):
        """通过 label_name 自动解析索引。"""
        ann = keypoints_to_pyskl(
            sample_keypoints, frame_dir="clip_named", label_name="sit", mode="2d"
        )
        assert ann["label"] == 0
        assert ann["label_name"] == "sit"

    def test_to_pyskl_invalid_shape(self):
        """错误 shape 触发 ValueError。"""
        bad = np.zeros((10, 17, 3), dtype=np.float32)  # 17 而非 24
        with pytest.raises(ValueError, match="shape"):
            keypoints_to_pyskl(bad, frame_dir="bad", mode="2d")

    def test_to_pyskl_invalid_mode(self, sample_keypoints):
        """非法 mode 触发 ValueError。"""
        with pytest.raises(ValueError, match="mode"):
            keypoints_to_pyskl(sample_keypoints, frame_dir="clip", mode="4d")

    def test_roundtrip_2d(self, sample_keypoints):
        """2D 往返: YOLO → pyskl → YOLO 应保持一致。"""
        ann = keypoints_to_pyskl(sample_keypoints, frame_dir="rt", label=0, mode="2d")
        recovered = pyskl_to_keypoints(ann, mode="2d")
        assert recovered.shape == sample_keypoints.shape
        np.testing.assert_allclose(recovered, sample_keypoints, atol=1e-6)

    def test_roundtrip_3d(self, sample_keypoints):
        """3D 往返: YOLO → pyskl → YOLO 应保持一致。"""
        ann = keypoints_to_pyskl(sample_keypoints, frame_dir="rt3d", label=0, mode="3d")
        recovered = pyskl_to_keypoints(ann, mode="3d")
        assert recovered.shape == sample_keypoints.shape
        np.testing.assert_allclose(recovered, sample_keypoints, atol=1e-6)

    def test_pyskl_to_keypoints_missing_score(self, sample_keypoints):
        """pyskl 无 keypoint_score 时填充 1.0。"""
        ann = keypoints_to_pyskl(sample_keypoints, frame_dir="no_score", mode="2d")
        del ann["keypoint_score"]
        recovered = pyskl_to_keypoints(ann, mode="2d")
        assert recovered.shape == (10, 24, 3)
        # conf 通道应为 1.0
        np.testing.assert_allclose(recovered[..., 2], 1.0, atol=1e-6)

    def test_compute_bone_flow_shape(self, sample_keypoints):
        """骨骼流 shape 与输入一致。"""
        bone = compute_bone_flow(sample_keypoints)
        assert bone.shape == sample_keypoints.shape

    def test_compute_bone_flow_root_zero(self, sample_keypoints):
        """根节点 (withers=22) 骨骼向量应为 0。"""
        bone = compute_bone_flow(sample_keypoints)
        np.testing.assert_allclose(bone[:, WITHERS, :], 0.0, atol=1e-6)

    def test_compute_bone_flow_definition(self):
        """骨骼流定义: bone[v] = joint[v] - joint[parent[v]]。"""
        T = 5
        kpt = np.random.rand(T, NUM_KEYPOINTS, 3).astype(np.float32)
        bone = compute_bone_flow(kpt)
        g = K9Graph()
        for v in range(NUM_KEYPOINTS):
            p = g.parent[v]
            if p >= 0:
                expected = kpt[:, v, :] - kpt[:, p, :]
                np.testing.assert_allclose(bone[:, v, :], expected, atol=1e-6)
            else:
                np.testing.assert_allclose(bone[:, v, :], 0.0, atol=1e-6)

    def test_compute_motion_flow_shape(self, sample_keypoints):
        """运动流 shape 与输入一致。"""
        motion = compute_motion_flow(sample_keypoints)
        assert motion.shape == sample_keypoints.shape

    def test_compute_motion_flow_first_frame_zero(self, sample_keypoints):
        """运动流第 0 帧应为 0。"""
        motion = compute_motion_flow(sample_keypoints)
        np.testing.assert_allclose(motion[0, :, :], 0.0, atol=1e-6)

    def test_compute_motion_flow_definition(self):
        """运动流定义: motion[t] = joint[t] - joint[t-1]。"""
        T = 5
        kpt = np.random.rand(T, NUM_KEYPOINTS, 3).astype(np.float32)
        motion = compute_motion_flow(kpt)
        np.testing.assert_allclose(motion[0, :, :], 0.0, atol=1e-6)
        for t in range(1, T):
            expected = kpt[t, :, :] - kpt[t - 1, :, :]
            np.testing.assert_allclose(motion[t, :, :], expected, atol=1e-6)

    def test_normalize_keypoints_center_zero(self, sample_keypoints):
        """归一化后中心点 (withers=22) 应在原点。"""
        normalized = normalize_keypoints(sample_keypoints, center_idx=WITHERS)
        np.testing.assert_allclose(normalized[:, WITHERS, :2], 0.0, atol=1e-5)

    def test_normalize_keypoints_conf_preserved(self, sample_keypoints):
        """归一化保留 conf 通道。"""
        normalized = normalize_keypoints(sample_keypoints, center_idx=WITHERS)
        np.testing.assert_allclose(normalized[..., 2], sample_keypoints[..., 2], atol=1e-6)

    def test_build_pyskl_annotation_batch(self, sample_keypoints):
        """批量构建标注列表。"""
        clips = [
            ("clip_0", sample_keypoints, 0),
            ("clip_1", sample_keypoints.copy(), 5),
            ("clip_2", sample_keypoints.copy(), 21),
        ]
        annotations = build_pyskl_annotation(clips, split="train", mode="2d")
        assert len(annotations) == 3
        assert annotations[0]["frame_dir"] == "clip_0"
        assert annotations[0]["label"] == 0
        assert annotations[1]["label"] == 5
        assert annotations[2]["label"] == 21
        assert all(a["split"] == "train" for a in annotations)
        assert all(a["ann_info"]["num_persons"] == 1 for a in annotations)


# ===== 集成测试: K9Graph + 数据适配 =====

class TestIntegration:
    """K9Graph + 数据适配器集成测试。"""

    def test_full_pipeline_synthetic(self):
        """完整管线: 合成关键点 → pyskl 格式 → 骨骼流 → 归一化 → 逆转换。"""
        np.random.seed(123)
        T = 15
        kpt = np.random.rand(T, NUM_KEYPOINTS, 3).astype(np.float32) * 100
        kpt[..., 2] = 0.85  # 统一置信度

        # 1. 转 pyskl 格式
        ann = keypoints_to_pyskl(kpt, frame_dir="integration_test", label=10, mode="2d")
        assert ann["keypoint"].shape == (1, T, 24, 2)

        # 2. 骨骼流
        bone = compute_bone_flow(kpt)
        assert bone.shape == (T, 24, 3)

        # 3. 运动流
        motion = compute_motion_flow(kpt)
        assert motion.shape == (T, 24, 3)

        # 4. 归一化
        normalized = normalize_keypoints(kpt, center_idx=WITHERS)
        assert normalized.shape == kpt.shape

        # 5. 逆转换回 YOLO 格式
        recovered = pyskl_to_keypoints(ann, mode="2d")
        np.testing.assert_allclose(recovered, kpt, atol=1e-6)

        # 6. K9Graph 拓扑可用于骨骼流解释
        g = K9Graph()
        assert g.num_nodes == 24
        # 前左肢骨骼链: paw ← knee ← elbow ← withers
        chain = []
        v = FRONT_LEFT_PAW
        while v != -1:
            chain.append(v)
            v = g.parent[v]
        # 应包含 paw → knee → elbow → withers
        assert FRONT_LEFT_PAW in chain
        assert FRONT_LEFT_KNEE in chain
        assert FRONT_LEFT_ELBOW in chain
        assert WITHERS in chain


# ===== 3.1c: ST-GCN 主干测试 =====

# torch 测试可选（CI 中若安装 torch 才跑）
torch = pytest.importorskip("torch", reason="PyTorch not installed, skipping 3.1c tests")

from backend.ml.behavior.stgcn_bc import (
    build_spatial_adjacency,
    UnitGCN,
    UnitTCN,
    MSTCN,
    STGCNBlock,
    STGCN,
    BCHead,
    generate_boundary_labels,
    STGCNBCLoss,
    STGCNBC,
    build_stgcn_bc,
)


class TestSpatialAdjacency:
    """ST-GCN 空间分区邻接矩阵测试。"""

    def test_shape(self):
        g = K9Graph()
        A = build_spatial_adjacency(g)
        assert A.shape == (3, 24, 24)  # 3 subsets, V=24

    def test_self_link_diagonal(self):
        """subset 0 应为单位矩阵（自环）。"""
        g = K9Graph()
        A = build_spatial_adjacency(g)
        for i in range(24):
            assert A[0, i, i] == 1.0

    def test_inward_outward_pattern_symmetric(self):
        """subset 1 (inward) 与 subset 2 (outward) 的非零位置应互为转置.

        注意: pyskl normalize_digraph 按列归一化（每列和为 1）。
        树结构中节点入度≠出度（叶节点入度=1/出度=0，根节点入度=0/出度>0），
        因此 inward 和 outward 的列和不同，归一化后数值不互为转置，
        但非零位置（即图拓扑）应保持转置关系。
        """
        g = K9Graph()
        A = build_spatial_adjacency(g)
        # 非零位置（拓扑）应互为转置
        pattern_in = (A[1] > 0).float()
        pattern_out = (A[2] > 0).float()
        assert torch.allclose(pattern_in, pattern_out.T, atol=1e-6)

    def test_normalization(self):
        """每列和应为 1（normalize_digraph）。"""
        g = K9Graph()
        A = build_spatial_adjacency(g)
        col_sum_in = A[1].sum(dim=0)
        col_sum_out = A[2].sum(dim=0)
        # 非零列和为 1，零列保持 0
        for i in range(24):
            if col_sum_in[i] > 0:
                assert abs(col_sum_in[i].item() - 1.0) < 1e-5


class TestUnitGCN:
    """空间图卷积单元测试。"""

    def test_forward_shape(self):
        g = K9Graph()
        A = build_spatial_adjacency(g)
        layer = UnitGCN(in_channels=3, out_channels=16, A=A)
        x = torch.randn(2, 3, 30, 24)  # (B, C, T, V)
        out = layer(x)
        assert out.shape == (2, 16, 30, 24)

    def test_adaptive_importance(self):
        """adaptive='importance' 时应有 PA 参数。"""
        g = K9Graph()
        A = build_spatial_adjacency(g)
        layer = UnitGCN(in_channels=8, out_channels=16, A=A, adaptive="importance")
        assert hasattr(layer, "PA")
        assert layer.PA.shape == A.shape

    def test_adaptive_none(self):
        """adaptive=None 时不应有 PA 参数。"""
        g = K9Graph()
        A = build_spatial_adjacency(g)
        layer = UnitGCN(in_channels=8, out_channels=16, A=A, adaptive=None)
        assert not hasattr(layer, "PA")

    def test_with_res(self):
        g = K9Graph()
        A = build_spatial_adjacency(g)
        layer = UnitGCN(in_channels=8, out_channels=16, A=A, with_res=True)
        x = torch.randn(2, 8, 30, 24)
        out = layer(x)
        assert out.shape == (2, 16, 30, 24)


class TestUnitTCN:
    """时间卷积单元测试。"""

    def test_forward_shape(self):
        layer = UnitTCN(in_channels=16, out_channels=32, kernel_size=9)
        x = torch.randn(2, 16, 30, 24)
        out = layer(x)
        assert out.shape == (2, 32, 30, 24)

    def test_stride_downsample(self):
        """stride=2 时 T 减半。"""
        layer = UnitTCN(in_channels=16, out_channels=32, kernel_size=9, stride=2)
        x = torch.randn(2, 16, 30, 24)
        out = layer(x)
        assert out.shape == (2, 32, 15, 24)


class TestMSTCN:
    """Multi-Scale Temporal Conv 测试。"""

    def test_forward_shape(self):
        layer = MSTCN(in_channels=32, out_channels=32)
        x = torch.randn(2, 32, 30, 24)
        out = layer(x)
        assert out.shape == (2, 32, 30, 24)

    def test_stride_downsample(self):
        layer = MSTCN(in_channels=32, out_channels=64, stride=2)
        x = torch.randn(2, 32, 30, 24)
        out = layer(x)
        assert out.shape == (2, 64, 15, 24)

    def test_branch_count(self):
        """默认 6 个分支。"""
        layer = MSTCN(in_channels=32, out_channels=64)
        assert layer.num_branches == 6
        assert len(layer.branches) == 6


class TestSTGCNBlock:
    """ST-GCN Block 测试。"""

    def test_forward_shape(self):
        g = K9Graph()
        A = build_spatial_adjacency(g)
        block = STGCNBlock(in_channels=16, out_channels=32, A=A)
        x = torch.randn(2, 16, 30, 24)
        out = block(x)
        assert out.shape == (2, 32, 30, 24)

    def test_residual_when_channels_match(self):
        """通道相同时 residual = identity。"""
        g = K9Graph()
        A = build_spatial_adjacency(g)
        block = STGCNBlock(in_channels=32, out_channels=32, A=A, residual=True)
        assert callable(block.residual)

    def test_no_residual(self):
        g = K9Graph()
        A = build_spatial_adjacency(g)
        block = STGCNBlock(in_channels=16, out_channels=32, A=A, residual=False)
        x = torch.randn(2, 16, 30, 24)
        out = block(x)
        assert out.shape == (2, 32, 30, 24)


class TestSTGCNBackbone:
    """ST-GCN 主干测试。"""

    def test_forward_4d_input(self):
        """输入 (B, T, V, C) 自动扩展 M=1。"""
        model = STGCN(in_channels=3, base_channels=32, num_stages=4)
        x = torch.randn(2, 30, 24, 3)
        out = model(x)
        assert out.ndim == 5
        assert out.shape[0] == 2
        assert out.shape[1] == 1  # M=1
        assert out.shape[2] == model.out_channels

    def test_forward_5d_input(self):
        """输入 (B, M, T, V, C) 直接处理。"""
        model = STGCN(in_channels=3, base_channels=32, num_stages=4)
        x = torch.randn(2, 1, 30, 24, 3)
        out = model(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 1

    def test_downsample_factor(self):
        """num_stages=10 + down_stages=[5,8] 应使 T 减少为 T/4。"""
        model = STGCN(in_channels=3, base_channels=32, num_stages=10,
                      down_stages=[5, 8])
        x = torch.randn(1, 60, 24, 3)
        out = model(x)
        # 60 → 30 (stage 5) → 15 (stage 8)
        assert out.shape[3] == 15

    def test_out_channels_attribute(self):
        """out_channels 应反映最终通道数。"""
        model = STGCN(in_channels=3, base_channels=64, num_stages=10)
        # base=64, inflate=[5,8]: stage 5 → 128, stage 8 → 256
        assert model.out_channels == 256


# ===== 3.1c: BC 头 + 联合损失测试 =====

class TestBCHead:
    """自研 BC 头测试。"""

    def test_forward_shapes(self):
        head = BCHead(in_channels=64, num_classes=22)
        x = torch.randn(2, 1, 64, 15, 24)  # (B, M, C, T', V)
        cls, bnd = head(x)
        assert cls.shape == (2, 22)
        assert bnd.shape == (2, 15)

    def test_multi_person_mean(self):
        """M>1 时对 M 维取均值。"""
        head = BCHead(in_channels=32, num_classes=22)
        x = torch.randn(2, 3, 32, 10, 24)  # M=3
        cls, bnd = head(x)
        assert cls.shape == (2, 22)
        assert bnd.shape == (2, 10)

    def test_invalid_input_ndim(self):
        head = BCHead(in_channels=32, num_classes=22)
        with pytest.raises(ValueError, match=r"\(B, M, C, T', V\)"):
            head(torch.randn(2, 32, 10, 24))


class TestGenerateBoundaryLabels:
    """边界软标签生成测试。"""

    def test_shape(self):
        labels = generate_boundary_labels(30, [0, 15], [14, 29])
        assert labels.shape == (30,)

    def test_range_0_to_1(self):
        """软标签值应在 [0, 1]。"""
        labels = generate_boundary_labels(30, [0, 15], [14, 29])
        assert labels.min() >= 0.0
        assert labels.max() <= 1.0

    def test_peak_at_boundary(self):
        """边界帧概率最高。"""
        labels = generate_boundary_labels(30, [0], [29])
        # 帧概率 0 和 29 应接近 1
        assert labels[0] > 0.99
        assert labels[29] > 0.99
        # 中间帧概率应较低
        assert labels[15] < 0.5

    def test_multiple_segments(self):
        """多 segment 取最大值。"""
        labels = generate_boundary_labels(30, [0, 20], [10, 29])
        # 帧 0/10/20/29 应有高概率
        for idx in [0, 10, 20, 29]:
            assert labels[idx] > 0.9

    def test_empty_segments(self):
        """空 segment 列表返回全 0。"""
        labels = generate_boundary_labels(30, [], [])
        assert labels.shape == (30,)
        assert labels.sum() == 0.0

    def test_mismatched_lengths(self):
        """starts 与 ends 长度不一致触发 ValueError。"""
        with pytest.raises(ValueError, match="长度不一致"):
            generate_boundary_labels(30, [0], [10, 20])


class TestSTGCNBCLoss:
    """联合损失 L = L_cls + 0.3 · L_boundary 测试。"""

    def test_loss_dict_keys(self):
        loss_fn = STGCNBCLoss()
        cls_logits = torch.randn(4, 22)
        bnd_logits = torch.randn(4, 15)
        cls_labels = torch.tensor([0, 5, 10, 21])
        bnd_labels = torch.rand(4, 15)
        result = loss_fn(cls_logits, bnd_logits, cls_labels, bnd_labels)
        assert set(result.keys()) == {"total", "cls", "boundary"}

    def test_boundary_weight_applied(self):
        """total = cls + 0.3 * boundary。"""
        loss_fn = STGCNBCLoss(boundary_weight=0.3)
        cls_logits = torch.randn(4, 22)
        bnd_logits = torch.randn(4, 15)
        cls_labels = torch.tensor([0, 5, 10, 21])
        bnd_labels = torch.rand(4, 15)
        result = loss_fn(cls_logits, bnd_logits, cls_labels, bnd_labels)
        expected = result["cls"] + 0.3 * result["boundary"]
        assert abs(result["total"].item() - expected.item()) < 1e-5

    def test_no_boundary_labels(self):
        """无 boundary_labels 时跳过边界损失。"""
        loss_fn = STGCNBCLoss()
        cls_logits = torch.randn(4, 22)
        bnd_logits = torch.randn(4, 15)
        cls_labels = torch.tensor([0, 5, 10, 21])
        result = loss_fn(cls_logits, bnd_logits, cls_labels, None)
        assert result["total"].item() == result["cls"].item()
        assert result["boundary"].item() == 0.0

    def test_temporal_interpolation(self):
        """boundary_logits 与 labels 时间维度不一致时自动插值。"""
        loss_fn = STGCNBCLoss()
        cls_logits = torch.randn(2, 22)
        bnd_logits = torch.randn(2, 15)  # T'=15（backbone 下采样后）
        cls_labels = torch.tensor([0, 5])
        bnd_labels = torch.rand(2, 30)  # 原始 T=30
        result = loss_fn(cls_logits, bnd_logits, cls_labels, bnd_labels)
        assert result["total"].ndim == 0  # scalar

    def test_backward(self):
        """损失可反向传播。"""
        loss_fn = STGCNBCLoss()
        cls_logits = torch.randn(4, 22, requires_grad=True)
        bnd_logits = torch.randn(4, 15, requires_grad=True)
        cls_labels = torch.tensor([0, 5, 10, 21])
        bnd_labels = torch.rand(4, 15)
        result = loss_fn(cls_logits, bnd_logits, cls_labels, bnd_labels)
        result["total"].backward()
        assert cls_logits.grad is not None
        assert bnd_logits.grad is not None


# ===== 3.1c: STGCNBC 整体模型测试 =====

class TestSTGCNBC:
    """ST-GCN+BC 整体模型测试。"""

    @pytest.fixture
    def small_model(self):
        """小模型用于快速测试。"""
        return STGCNBC(in_channels=3, num_classes=22,
                       base_channels=32, num_stages=4)

    def test_forward(self, small_model):
        x = torch.randn(2, 30, 24, 3)
        cls, bnd = small_model(x)
        assert cls.shape == (2, 22)
        assert bnd.shape == (2, 30)

    def test_predict(self, small_model):
        small_model.eval()
        x = torch.randn(2, 30, 24, 3)
        pred = small_model.predict(x)
        assert set(pred.keys()) == {"cls_probs", "cls_pred", "boundary_probs", "boundary_pred"}
        assert pred["cls_probs"].shape == (2, 22)
        assert pred["cls_pred"].shape == (2,)
        assert pred["boundary_probs"].shape == (2, 30)
        # cls_probs 应为概率分布（每行和为 1）
        assert torch.allclose(pred["cls_probs"].sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_compute_loss(self, small_model):
        small_model.train()
        x = torch.randn(2, 30, 24, 3)
        cls, bnd = small_model(x)
        labels = torch.tensor([0, 5])
        bl = generate_boundary_labels(30, [0, 15], [14, 29])
        bl_batch = bl.unsqueeze(0).repeat(2, 1)
        result = small_model.compute_loss(cls, bnd, labels, bl_batch)
        assert "total" in result
        assert result["total"].item() > 0

    def test_backward(self, small_model):
        """端到端反向传播。"""
        small_model.train()
        x = torch.randn(2, 30, 24, 3)
        cls, bnd = small_model(x)
        labels = torch.tensor([0, 5])
        bl = generate_boundary_labels(30, [0, 15], [14, 29])
        bl_batch = bl.unsqueeze(0).repeat(2, 1)
        result = small_model.compute_loss(cls, bnd, labels, bl_batch)
        result["total"].backward()
        # 检查所有参数都有梯度
        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 无梯度"

    def test_param_count_reasonable(self):
        """完整模型参数量应在 1-2M（接近 pyskl ST-GCN++ 1.45M）。"""
        model = STGCNBC(in_channels=3, num_classes=22,
                        base_channels=64, num_stages=10)
        total = sum(p.numel() for p in model.parameters())
        assert 1_000_000 < total < 2_000_000, f"参数量异常: {total}"

    def test_build_stgcn_bc_helper(self):
        """便捷构建函数测试。"""
        model = build_stgcn_bc(in_channels=3, num_classes=22,
                               base_channels=32, num_stages=4)
        assert isinstance(model, STGCNBC)
        x = torch.randn(1, 20, 24, 3)
        cls, bnd = model(x)
        assert cls.shape == (1, 22)


class TestSTGCNBCIntegration:
    """ST-GCN+BC 端到端集成测试。"""

    def test_full_pipeline_with_data_adapter(self):
        """完整管线: YOLO 格式 → 归一化 → STGCNBC → 预测。"""
        np.random.seed(42)
        # 1. 合成 YOLO26-pose 输出
        T = 30
        kpt = np.random.rand(T, NUM_KEYPOINTS, 3).astype(np.float32)
        kpt[..., 2] = 0.85

        # 2. 归一化（MotionBERT 3D 假设已归一化）
        normalized = normalize_keypoints(kpt, center_idx=WITHERS)

        # 3. 转 tensor (B=1, T, V, C)
        x = torch.from_numpy(normalized).unsqueeze(0)

        # 4. 模型推理
        model = STGCNBC(in_channels=3, num_classes=22,
                        base_channels=32, num_stages=4)
        model.eval()
        with torch.no_grad():
            pred = model.predict(x)

        # 5. 验证输出
        assert pred["cls_pred"].shape == (1,)
        assert 0 <= pred["cls_pred"].item() < 22
        assert pred["boundary_probs"].shape == (1, T)

    def test_training_step(self):
        """模拟一次训练 step（forward + loss + backward + 无 optimizer.step）。"""
        torch.manual_seed(0)
        model = STGCNBC(in_channels=3, num_classes=22,
                        base_channels=32, num_stages=4)
        model.train()

        # 合成 batch
        B = 4
        x = torch.randn(B, 30, 24, 3)
        labels = torch.randint(0, 22, (B,))
        # 边界标签（每样本 1 个 segment [0, 29]）
        bl = torch.stack([
            generate_boundary_labels(30, [0], [29]) for _ in range(B)
        ])

        cls, bnd = model(x)
        loss = model.compute_loss(cls, bnd, labels, bl)
        loss["total"].backward()

        # 验证损失有限
        assert torch.isfinite(loss["total"])
        assert loss["total"].item() > 0
