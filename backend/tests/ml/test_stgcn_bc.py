"""ST-GCN+BC 模块单元测试（Phase 3.1b）.

Owner: ML 开发
Phase: 3.1b

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
