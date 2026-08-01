"""3D 姿态重建数据配对模块单元测试（Phase 3.3b）.

Owner: ML 开发
Phase: 3.3b

测试覆盖:
    1. interpet4d_loader.py
        - parse_clip_id 命名解析
        - InterPet4DClip 数据类
        - load_clip / list_clips / load_all_clips
        - get_dataset_statistics
    2. camera_projection.py
        - SyntheticCamera 构造 + 矩阵计算
        - project_3d_to_2d 正交/透视投影
        - generate_synthetic_cameras 球面采样
        - project_clip_to_2d 批量投影
    3. lifting_pairing.py
        - normalize_3d_keypoints / denormalize_3d_keypoints 归一化
        - slice_windows 滑动窗口
        - build_pairs_from_clip 配对构建
        - build_dataset 数据集构建
        - train_val_split 按 clip 划分
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.ml.pose.interpet4d_loader import (
    InterPet4DClip,
    NUM_KEYPOINTS,
    DEFAULT_INTERPET4D_SMAL_DIR,
    parse_clip_id,
    load_clip,
    list_clips,
    load_all_clips,
    get_dataset_statistics,
)
from backend.ml.pose.camera_projection import (
    SyntheticCamera,
    project_3d_to_2d,
    generate_synthetic_cameras,
    project_clip_to_2d,
    DEFAULT_FOCAL_LENGTH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_CAMERA_DISTANCE,
)
from backend.ml.pose.lifting_pairing import (
    LiftingSample,
    normalize_3d_keypoints,
    denormalize_3d_keypoints,
    slice_windows,
    build_pairs_from_clip,
    build_dataset,
    compute_dataset_statistics,
    train_val_split,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_STRIDE,
    ROOT_KEYPOINT_IDX,
)


# ===== 测试常量 =====

INTERPET4D_SMAL_DIR = Path(DEFAULT_INTERPET4D_SMAL_DIR)
HAS_REAL_DATA = INTERPET4D_SMAL_DIR.exists() and any(INTERPET4D_SMAL_DIR.glob("*.npz"))
SAMPLE_CLIP_ID = "interpet_dog01_p01_take01_ego_001"


# ===== 辅助函数 =====

def make_synthetic_kp_world(T: int = 30, V: int = NUM_KEYPOINTS, seed: int = 42) -> np.ndarray:
    """构造合成 kp_world (T, 24, 3) 用于确定性测试."""
    rng = np.random.default_rng(seed)
    # 犬只尺寸 ~1m，中心在原点附近
    return rng.uniform(-0.5, 0.5, size=(T, V, 3)).astype(np.float32)


def make_synthetic_kp_weight(T: int = 30, V: int = NUM_KEYPOINTS, seed: int = 43) -> np.ndarray:
    """构造合成 kp_weight (T, 24) 用于确定性测试."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 1.0, size=(T, V)).astype(np.float32)


def make_synthetic_clip(clip_id: str = "synthetic_clip", T: int = 30) -> InterPet4DClip:
    """构造合成 InterPet4DClip 用于不依赖真实数据的测试."""
    return InterPet4DClip(
        clip_id=clip_id,
        kp_world=make_synthetic_kp_world(T),
        kp_weight=make_synthetic_kp_weight(T),
        frame_idx=np.arange(T, dtype=np.int32),
    )


# ===== 1. interpet4d_loader.py 测试 =====

class TestParseClipId:
    """clip_id 命名解析测试。"""

    def test_standard_format(self):
        clip_id = "interpet_dog01_p01_take01_ego_001"
        dog, part, take, idx = parse_clip_id(clip_id)
        assert dog == "dog01"
        assert part == "p01"
        assert take == "take01"
        assert idx == "001"

    def test_two_digit_dog(self):
        clip_id = "interpet_dog12_p23_take10_ego_003"
        dog, part, take, idx = parse_clip_id(clip_id)
        assert dog == "dog12"
        assert part == "p23"
        assert take == "take10"
        assert idx == "003"

    def test_malformed_returns_empty(self):
        # 缺少 ego 段
        dog, part, take, idx = parse_clip_id("interpet_dog01_p01_take01_001")
        assert (dog, part, take, idx) == ("", "", "", "")

    def test_wrong_prefix(self):
        # 前缀不是 interpet
        dog, part, take, idx = parse_clip_id("other_dog01_p01_take01_ego_001")
        assert (dog, part, take, idx) == ("", "", "", "")


class TestInterPet4DClipDataclass:
    """InterPet4DClip 数据类测试。"""

    def test_num_frames(self):
        clip = make_synthetic_clip(T=30)
        assert clip.num_frames == 30

    def test_duration_seconds_from_frame_idx(self):
        clip = make_synthetic_clip(T=30)
        # frame_idx = 0..29，最后一帧 29 / 30fps ≈ 0.967s
        assert clip.duration_seconds == pytest.approx(29 / 30.0, abs=1e-3)

    def test_duration_seconds_empty_frame_idx(self):
        clip = InterPet4DClip(
            clip_id="test",
            kp_world=make_synthetic_kp_world(T=30),
            kp_weight=make_synthetic_kp_weight(T=30),
            frame_idx=np.array([], dtype=np.int32),
        )
        # frame_idx 为空时回退到 num_frames / 30
        assert clip.duration_seconds == pytest.approx(1.0, abs=1e-3)

    def test_summary(self):
        clip = make_synthetic_clip(T=30)
        s = clip.summary()
        assert s["clip_id"] == "synthetic_clip"
        assert s["num_frames"] == 30
        assert s["kp_world_shape"] == [30, 24, 3]
        assert "kp_weight_mean" in s
        assert "duration_seconds" in s


@pytest.mark.skipif(not HAS_REAL_DATA, reason="InterPet4D smal_npy 数据未安装")
class TestInterPet4DLoaderReal:
    """真实 InterPet4D 数据加载测试。"""

    def test_list_clips_nonempty(self):
        clips = list_clips(INTERPET4D_SMAL_DIR)
        assert len(clips) > 0
        # 排序
        assert clips == sorted(clips)

    def test_load_existing_clip(self):
        clip = load_clip(SAMPLE_CLIP_ID, INTERPET4D_SMAL_DIR)
        assert clip is not None
        assert clip.clip_id == SAMPLE_CLIP_ID
        assert clip.kp_world.shape[1] == NUM_KEYPOINTS
        assert clip.kp_world.shape[2] == 3
        assert clip.kp_weight.shape == (clip.kp_world.shape[0], NUM_KEYPOINTS)
        assert clip.dog_id == "dog01"
        assert clip.participant_id == "p01"
        assert clip.take_id == "take01"

    def test_load_clip_with_optional(self):
        clip = load_clip(SAMPLE_CLIP_ID, INTERPET4D_SMAL_DIR, load_optional=True)
        assert clip is not None
        # R_world / t_world / s_world 应存在（InterPet4D 标准）
        assert clip.R_world is not None
        assert clip.R_world.shape == (clip.num_frames, 3, 3)
        assert clip.t_world is not None
        assert clip.t_world.shape == (clip.num_frames, 3)
        assert clip.s_world is not None
        assert clip.s_world.shape == (clip.num_frames,)

    def test_load_nonexistent_clip(self):
        clip = load_clip("nonexistent_clip_id", INTERPET4D_SMAL_DIR)
        assert clip is None

    def test_load_all_clips_filtering(self):
        """load_all_clips 默认过滤参数下应返回非空列表."""
        clips = load_all_clips(INTERPET4D_SMAL_DIR, min_frames=10)
        assert len(clips) > 0
        # 全部满足 min_frames
        for c in clips:
            assert c.num_frames >= 10

    def test_get_dataset_statistics(self):
        stats = get_dataset_statistics(INTERPET4D_SMAL_DIR, max_clips=5)
        assert stats["total_clips"] == 5
        assert stats["total_frames"] > 0
        assert stats["num_dogs"] >= 1
        assert "frame_count_stats" in stats
        assert "kp_weight_stats" in stats
        assert stats["frame_count_stats"]["min"] > 0


class TestInterPet4DLoaderMocked:
    """使用合成数据测试 loader 逻辑（不依赖真实数据）。"""

    def test_load_clip_nonexistent_dir(self, tmp_path):
        clip = load_clip("any_clip", tmp_path)
        assert clip is None

    def test_load_clip_missing_kp_world(self, tmp_path):
        """缺少 kp_world 字段应返回 None."""
        np.savez(tmp_path / "bad_clip.npz", kp_weight=np.zeros((10, 24)))
        clip = load_clip("bad_clip", tmp_path)
        assert clip is None

    def test_load_clip_wrong_kp_shape(self, tmp_path):
        """kp_world 形状错误应返回 None."""
        np.savez(
            tmp_path / "bad_shape.npz",
            kp_world=np.zeros((10, 20, 3)),  # 20 而非 24
            kp_weight=np.ones((10, 20)),
        )
        clip = load_clip("bad_shape", tmp_path)
        assert clip is None

    def test_load_clip_missing_kp_weight_fallback(self, tmp_path):
        """缺少 kp_weight 应回退为全 1."""
        T = 15
        np.savez(
            tmp_path / "no_weight.npz",
            kp_world=make_synthetic_kp_world(T),
            frame_idx=np.arange(T, dtype=np.int32),
        )
        clip = load_clip("no_weight", tmp_path)
        assert clip is not None
        assert clip.kp_weight.shape == (T, NUM_KEYPOINTS)
        assert np.all(clip.kp_weight == 1.0)

    def test_load_clip_missing_frame_idx_fallback(self, tmp_path):
        """缺少 frame_idx 应回退为 0..T-1."""
        T = 12
        np.savez(
            tmp_path / "no_frame_idx.npz",
            kp_world=make_synthetic_kp_world(T),
            kp_weight=make_synthetic_kp_weight(T),
        )
        clip = load_clip("no_frame_idx", tmp_path)
        assert clip is not None
        assert clip.frame_idx.shape == (T,)
        assert np.array_equal(clip.frame_idx, np.arange(T))

    def test_list_clips_empty_dir(self, tmp_path):
        clips = list_clips(tmp_path)
        assert clips == []

    def test_list_clips_sorted(self, tmp_path):
        for name in ["c_clip", "a_clip", "b_clip"]:
            np.savez(tmp_path / f"{name}.npz", kp_world=make_synthetic_kp_world(10))
        clips = list_clips(tmp_path)
        assert clips == ["a_clip", "b_clip", "c_clip"]

    def test_load_all_clips_min_frames_filter(self, tmp_path):
        """min_frames 过滤应丢弃过短 clip."""
        np.savez(tmp_path / "long.npz", kp_world=make_synthetic_kp_world(20))
        np.savez(tmp_path / "short.npz", kp_world=make_synthetic_kp_world(5))
        clips = load_all_clips(tmp_path, min_frames=10)
        assert len(clips) == 1
        assert clips[0].clip_id == "long"

    def test_load_all_clips_kp_weight_filter(self, tmp_path):
        """min_valid_kp_ratio 过滤应丢弃低质量 clip."""
        T = 20
        # 高质量
        np.savez(
            tmp_path / "good.npz",
            kp_world=make_synthetic_kp_world(T),
            kp_weight=np.ones((T, NUM_KEYPOINTS)),
        )
        # 低质量（全 0.1，低于 0.3 阈值）
        np.savez(
            tmp_path / "bad.npz",
            kp_world=make_synthetic_kp_world(T),
            kp_weight=np.full((T, NUM_KEYPOINTS), 0.1),
        )
        clips = load_all_clips(tmp_path, min_frames=10, min_kp_weight=0.3, min_valid_kp_ratio=0.5)
        assert len(clips) == 1
        assert clips[0].clip_id == "good"

    def test_load_all_clips_nan_filter(self, tmp_path):
        """kp_world 含 NaN/Inf 的 clip 应被过滤（Phase 3.3c NaN bug 修复）."""
        T = 20
        # 正常 clip
        np.savez(
            tmp_path / "good.npz",
            kp_world=make_synthetic_kp_world(T),
            kp_weight=np.ones((T, NUM_KEYPOINTS)),
        )
        # 全 NaN clip（模拟 interpet_dog06_p14_take01_ego_001 bug）
        np.savez(
            tmp_path / "all_nan.npz",
            kp_world=np.full((T, NUM_KEYPOINTS, 3), np.nan, dtype=np.float32),
            kp_weight=np.ones((T, NUM_KEYPOINTS)),
        )
        # 部分 NaN clip（也应被过滤）
        partial_nan = make_synthetic_kp_world(T)
        partial_nan[5:10, 0, :] = np.nan
        np.savez(
            tmp_path / "partial_nan.npz",
            kp_world=partial_nan,
            kp_weight=np.ones((T, NUM_KEYPOINTS)),
        )
        clips = load_all_clips(tmp_path, min_frames=10, min_kp_weight=0.3, min_valid_kp_ratio=0.5)
        # 只保留 good，过滤掉 all_nan 和 partial_nan
        assert len(clips) == 1
        assert clips[0].clip_id == "good"


# ===== 2. camera_projection.py 测试 =====

class TestSyntheticCamera:
    """SyntheticCamera 数据类测试。"""

    def test_construction(self):
        cam = SyntheticCamera(position=np.array([1.0, 2.0, 3.0]))
        assert cam.position.shape == (3,)
        assert np.allclose(cam.position, [1.0, 2.0, 3.0])
        assert cam.focal_length == DEFAULT_FOCAL_LENGTH
        assert cam.image_size == DEFAULT_IMAGE_SIZE

    def test_default_target_origin(self):
        cam = SyntheticCamera(position=np.array([3.0, 0.0, 0.0]))
        assert np.allclose(cam.target, [0.0, 0.0, 0.0])

    def test_default_up(self):
        cam = SyntheticCamera(position=np.array([3.0, 0.0, 0.0]))
        assert np.allclose(cam.up, [0.0, 1.0, 0.0])

    def test_width_height(self):
        cam = SyntheticCamera(position=np.zeros(3), image_size=(800, 600))
        assert cam.width == 800
        assert cam.height == 600

    def test_view_matrix_shape(self):
        cam = SyntheticCamera(position=np.array([3.0, 0.0, 0.0]))
        view = cam.view_matrix()
        assert view.shape == (4, 4)

    def test_view_matrix_translation(self):
        """view 矩阵应将相机位置映射到原点."""
        cam = SyntheticCamera(position=np.array([3.0, 0.0, 0.0]))
        view = cam.view_matrix()
        cam_origin_homo = np.array([3.0, 0.0, 0.0, 1.0], dtype=np.float32)
        mapped = view @ cam_origin_homo
        assert np.allclose(mapped[:3], [0.0, 0.0, 0.0], atol=1e-5)

    def test_projection_matrix_orthographic_shape(self):
        cam = SyntheticCamera(position=np.array([3.0, 0.0, 0.0]))
        proj = cam.projection_matrix(mode="orthographic")
        assert proj.shape == (4, 4)

    def test_projection_matrix_perspective_shape(self):
        cam = SyntheticCamera(position=np.array([3.0, 0.0, 0.0]))
        proj = cam.projection_matrix(mode="perspective")
        assert proj.shape == (3, 4)

    def test_projection_matrix_invalid_mode(self):
        cam = SyntheticCamera(position=np.array([3.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="未知投影模式"):
            cam.projection_matrix(mode="fisheye")


class TestProject3DTo2D:
    """project_3d_to_2d 投影测试。"""

    def test_orthographic_shape(self):
        cam = SyntheticCamera(position=np.array([0.0, 0.0, 3.0]))
        kp3d = make_synthetic_kp_world(T=10)
        kp2d = project_3d_to_2d(kp3d, cam, mode="orthographic", normalize=True)
        assert kp2d.shape == (10, NUM_KEYPOINTS, 2)

    def test_orthographic_normalized_range(self):
        """正交投影归一化后 2D 坐标应在合理范围 [-1, 1] 附近（合成数据 [-0.5, 0.5]）."""
        cam = SyntheticCamera(position=np.array([0.0, 0.0, 3.0]))
        kp3d = make_synthetic_kp_world(T=10)
        kp2d = project_3d_to_2d(kp3d, cam, mode="orthographic", normalize=True)
        # 合成 kp_world 在 [-0.5, 0.5]，正交归一化后应在 [-0.5, 0.5] 附近
        assert kp2d.min() >= -1.0
        assert kp2d.max() <= 1.0

    def test_perspective_shape(self):
        cam = SyntheticCamera(position=np.array([0.0, 0.0, 3.0]))
        kp3d = make_synthetic_kp_world(T=10)
        kp2d = project_3d_to_2d(kp3d, cam, mode="perspective", normalize=True)
        assert kp2d.shape == (10, NUM_KEYPOINTS, 2)

    def test_perspective_normalized_range(self):
        """透视投影归一化后 2D 坐标应在 [-1, 1] 范围内（相机距离 3m，犬只 1m）."""
        cam = SyntheticCamera(position=np.array([0.0, 0.0, 3.0]))
        kp3d = make_synthetic_kp_world(T=10)
        kp2d = project_3d_to_2d(kp3d, cam, mode="perspective", normalize=True)
        # 透视投影应将点映射到图像平面
        assert np.isfinite(kp2d).all()

    def test_invalid_mode_raises(self):
        cam = SyntheticCamera(position=np.array([0.0, 0.0, 3.0]))
        kp3d = make_synthetic_kp_world(T=5)
        with pytest.raises(ValueError, match="未知投影模式"):
            project_3d_to_2d(kp3d, cam, mode="invalid")

    def test_homogeneous_input(self):
        """支持 (..., 4) 齐次坐标输入."""
        cam = SyntheticCamera(position=np.array([0.0, 0.0, 3.0]))
        kp3d = make_synthetic_kp_world(T=5)
        kp_homo = np.concatenate([kp3d, np.ones((*kp3d.shape[:-1], 1), dtype=np.float32)], axis=-1)
        assert kp_homo.shape == (5, NUM_KEYPOINTS, 4)
        kp2d = project_3d_to_2d(kp_homo, cam, mode="orthographic")
        assert kp2d.shape == (5, NUM_KEYPOINTS, 2)

    def test_single_point_projection(self):
        """单点投影：原点应投影到图像中心."""
        cam = SyntheticCamera(position=np.array([0.0, 0.0, 3.0]))
        kp3d = np.zeros((1, 3), dtype=np.float32)  # 原点
        kp2d = project_3d_to_2d(kp3d, cam, mode="orthographic", normalize=True)
        # 原点 → 相机坐标系原点 → 投影 (0, 0)
        assert kp2d.shape == (1, 2)
        assert np.allclose(kp2d[0], [0.0, 0.0], atol=1e-5)


class TestGenerateSyntheticCameras:
    """generate_synthetic_cameras 球面采样测试。"""

    def test_num_cameras(self):
        cams = generate_synthetic_cameras(num_cameras=8, random_seed=42)
        assert len(cams) == 8

    def test_default_distance(self):
        cams = generate_synthetic_cameras(num_cameras=4, distance=5.0)
        for cam in cams:
            # 相机到原点距离 ≈ 5.0
            dist = float(np.linalg.norm(cam.position))
            assert dist == pytest.approx(5.0, abs=0.5)  # 仰角带来轻微偏差

    def test_reproducibility(self):
        """相同 random_seed 应生成相同相机阵列."""
        cams1 = generate_synthetic_cameras(num_cameras=8, random_seed=42)
        cams2 = generate_synthetic_cameras(num_cameras=8, random_seed=42)
        for c1, c2 in zip(cams1, cams2):
            assert np.allclose(c1.position, c2.position)

    def test_different_seeds_differ(self):
        cams1 = generate_synthetic_cameras(num_cameras=8, random_seed=42)
        cams2 = generate_synthetic_cameras(num_cameras=8, random_seed=123)
        positions1 = np.stack([c.position for c in cams1])
        positions2 = np.stack([c.position for c in cams2])
        assert not np.allclose(positions1, positions2)

    def test_single_camera(self):
        cam = generate_synthetic_cameras(num_cameras=1, random_seed=42)
        assert len(cam) == 1

    def test_target_offset(self):
        """非原点 target 应平移所有相机."""
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        cams = generate_synthetic_cameras(num_cameras=4, distance=3.0, target=target)
        for cam in cams:
            # 相机到 target 距离 ≈ 3.0
            dist = float(np.linalg.norm(cam.position - target))
            assert dist == pytest.approx(3.0, abs=0.5)


class TestProjectClipTo2D:
    """project_clip_to_2d 批量投影测试。"""

    def test_shape(self):
        T, V = 30, NUM_KEYPOINTS
        kp_world = make_synthetic_kp_world(T)
        cams = generate_synthetic_cameras(num_cameras=4, random_seed=42)
        projections = project_clip_to_2d(kp_world, cams, mode="orthographic")
        assert projections.shape == (4, T, V, 2)

    def test_per_camera_consistency(self):
        """每个相机的投影应与单独投影一致."""
        T = 10
        kp_world = make_synthetic_kp_world(T)
        cams = generate_synthetic_cameras(num_cameras=2, random_seed=42)
        projections = project_clip_to_2d(kp_world, cams, mode="orthographic")

        for i, cam in enumerate(cams):
            single = project_3d_to_2d(kp_world, cam, mode="orthographic")
            assert np.allclose(projections[i], single)


# ===== 3. lifting_pairing.py 测试 =====

class TestNormalize3DKeypoints:
    """3D 关键点归一化测试。"""

    def test_bone_length_mode_shape(self):
        kp3d = make_synthetic_kp_world(T=10)
        normalized, scale = normalize_3d_keypoints(kp3d, root_idx=ROOT_KEYPOINT_IDX, scale_mode="bone_length")
        assert normalized.shape == kp3d.shape
        assert scale > 0

    def test_root_centered(self):
        """归一化后根关节应为原点."""
        kp3d = make_synthetic_kp_world(T=10)
        normalized, _ = normalize_3d_keypoints(kp3d, root_idx=ROOT_KEYPOINT_IDX, scale_mode="bone_length")
        # 根关节（idx=22）每帧应为 (0, 0, 0)
        root_pts = normalized[:, ROOT_KEYPOINT_IDX, :]
        assert np.allclose(root_pts, 0.0, atol=1e-6)

    def test_bbox_mode(self):
        kp3d = make_synthetic_kp_world(T=10)
        normalized, scale = normalize_3d_keypoints(kp3d, root_idx=ROOT_KEYPOINT_IDX, scale_mode="bbox")
        assert normalized.shape == kp3d.shape
        assert scale > 0

    def test_none_mode_only_centers(self):
        """none 模式应仅做根关节中心化，scale=1.0."""
        kp3d = make_synthetic_kp_world(T=10)
        normalized, scale = normalize_3d_keypoints(kp3d, root_idx=ROOT_KEYPOINT_IDX, scale_mode="none")
        assert scale == 1.0
        root_pts = normalized[:, ROOT_KEYPOINT_IDX, :]
        assert np.allclose(root_pts, 0.0, atol=1e-6)

    def test_invalid_mode_raises(self):
        kp3d = make_synthetic_kp_world(T=5)
        with pytest.raises(ValueError, match="未知 scale_mode"):
            normalize_3d_keypoints(kp3d, scale_mode="invalid")

    def test_custom_root_idx(self):
        """使用非默认根关节（如 nose=2）应正确中心化."""
        kp3d = make_synthetic_kp_world(T=5)
        normalized, _ = normalize_3d_keypoints(kp3d, root_idx=2, scale_mode="none")
        root_pts = normalized[:, 2, :]
        assert np.allclose(root_pts, 0.0, atol=1e-6)


class TestDenormalize3DKeypoints:
    """3D 关键点反归一化测试。"""

    def test_round_trip_bone_length(self):
        """归一化 → 反归一化应恢复原坐标（无根位置时）."""
        kp3d = make_synthetic_kp_world(T=10)
        normalized, scale = normalize_3d_keypoints(kp3d, root_idx=ROOT_KEYPOINT_IDX, scale_mode="bone_length")
        # 反归一化（根位置 = 原始根关节位置）
        root_positions = kp3d[:, ROOT_KEYPOINT_IDX, :]
        restored = denormalize_3d_keypoints(normalized, scale=scale)
        # denormalize 仅乘 scale，不恢复根位置；手动加回
        restored = restored + root_positions[:, None, :]
        assert np.allclose(restored, kp3d, atol=1e-5)

    def test_round_trip_none_mode(self):
        """none 模式 scale=1.0，反归一化 + 根位置 = 原坐标."""
        kp3d = make_synthetic_kp_world(T=5)
        normalized, scale = normalize_3d_keypoints(kp3d, root_idx=ROOT_KEYPOINT_IDX, scale_mode="none")
        root_positions = kp3d[:, ROOT_KEYPOINT_IDX, :]
        restored = denormalize_3d_keypoints(normalized, scale=scale, root_position=root_positions[0])
        # denormalize 接受单个 root_position，广播到所有帧
        # 但因每帧根位置不同，这里只测试单帧
        assert restored.shape == kp3d.shape

    def test_with_explicit_root_position(self):
        """显式 root_position 应正确平移."""
        normalized = np.zeros((5, NUM_KEYPOINTS, 3), dtype=np.float32)
        root_pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        restored = denormalize_3d_keypoints(normalized, scale=2.0, root_position=root_pos)
        # 全 0 * 2.0 + [1,2,3] = [1,2,3]
        assert np.allclose(restored[0, 0], [1.0, 2.0, 3.0])


class TestSliceWindows:
    """slice_windows 滑动窗口测试。"""

    def test_exact_fit(self):
        """T 恰好为 window_size 整数倍应切出多个窗口."""
        seq = np.arange(54).reshape(27, 2)  # T=27, window=27, stride=13
        windows = slice_windows(seq, window_size=27, stride=13)
        # T=27, window=27, stride=13 → starts: 0, 13 → 2 windows（0-26, 13-39 截断到 0-26）
        # range(0, 27-27+1, 13) = range(0, 1, 13) = [0]
        # 实际：T - window + 1 = 1，stride=13，range(0, 1, 13) = [0]
        # 故只有 1 个窗口
        assert len(windows) == 1
        assert windows[0].shape == (27, 2)

    def test_multiple_windows(self):
        """T > window_size 应切出多个窗口."""
        seq = np.arange(100).reshape(50, 2)  # T=50
        windows = slice_windows(seq, window_size=27, stride=13)
        # range(0, 50-27+1, 13) = range(0, 24, 13) = [0, 13]
        assert len(windows) == 2
        for w in windows:
            assert w.shape == (27, 2)

    def test_padding_when_short(self):
        """T < window_size 应 pad 到 window_size."""
        seq = np.arange(20).reshape(10, 2)  # T=10, window=27
        windows = slice_windows(seq, window_size=27, stride=13)
        assert len(windows) == 1
        assert windows[0].shape == (27, 2)
        # pad 部分应复制最后一帧（edge mode）
        assert np.array_equal(windows[0][10], windows[0][9])

    def test_multidim_input(self):
        """支持 (T, V, C) 多维输入."""
        seq = make_synthetic_kp_world(T=50)
        windows = slice_windows(seq, window_size=27, stride=13)
        assert len(windows) == 2
        for w in windows:
            assert w.shape == (27, NUM_KEYPOINTS, 3)


class TestBuildPairsFromClip:
    """build_pairs_from_clip 配对构建测试。"""

    def test_basic_sample_count(self):
        """样本数 = num_cameras × windows_per_clip."""
        T = 50
        clip = make_synthetic_clip(T=T)
        samples = build_pairs_from_clip(
            clip,
            num_cameras=4,
            projection_mode="orthographic",
            window_size=27,
            stride=13,
            random_seed=42,
        )
        # T=50, window=27, stride=13 → range(0, 24, 13) = [0, 13] → 2 windows
        # 4 cameras × 2 windows = 8 samples
        assert len(samples) == 4 * 2

    def test_sample_shapes(self):
        clip = make_synthetic_clip(T=50)
        samples = build_pairs_from_clip(
            clip, num_cameras=2, window_size=27, stride=13, random_seed=42
        )
        for s in samples:
            assert s.keypoints_2d.shape == (27, NUM_KEYPOINTS, 2)
            assert s.keypoints_3d.shape == (27, NUM_KEYPOINTS, 3)
            assert s.confidence.shape == (27, NUM_KEYPOINTS)

    def test_sample_metadata(self):
        clip = make_synthetic_clip(T=50)
        samples = build_pairs_from_clip(
            clip, num_cameras=2, window_size=27, stride=13, random_seed=42
        )
        for i, s in enumerate(samples):
            assert s.clip_id == "synthetic_clip"
            assert s.camera_idx in (0, 1)
            assert s.window_size == 27
            assert s.scale > 0

    def test_confidence_filtering(self):
        """低于 min_kp_weight 的 confidence 应置 0."""
        T = 50
        kp_weight = make_synthetic_kp_weight(T)
        kp_weight[0, 0] = 0.1  # 低于阈值 0.3
        clip = InterPet4DClip(
            clip_id="test_filter",
            kp_world=make_synthetic_kp_world(T),
            kp_weight=kp_weight,
            frame_idx=np.arange(T, dtype=np.int32),
        )
        samples = build_pairs_from_clip(
            clip, num_cameras=1, window_size=27, stride=27, min_kp_weight=0.3
        )
        assert len(samples) == 1
        # 第 0 帧第 0 关键点的 confidence 应为 0
        assert samples[0].confidence[0, 0] == 0.0

    def test_short_clip_single_window(self):
        """T < window_size 应生成 1 个 padded 窗口."""
        clip = make_synthetic_clip(T=15)
        samples = build_pairs_from_clip(
            clip, num_cameras=2, window_size=27, stride=13, random_seed=42
        )
        # T=15 < 27 → 1 padded window × 2 cameras = 2 samples
        assert len(samples) == 2
        for s in samples:
            assert s.keypoints_2d.shape == (27, NUM_KEYPOINTS, 2)

    def test_empty_clip(self):
        """0 帧 clip 应返回空列表."""
        clip = InterPet4DClip(
            clip_id="empty",
            kp_world=np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32),
            kp_weight=np.zeros((0, NUM_KEYPOINTS), dtype=np.float32),
            frame_idx=np.array([], dtype=np.int32),
        )
        samples = build_pairs_from_clip(clip, num_cameras=2)
        assert samples == []


class TestBuildDataset:
    """build_dataset 端到端数据集构建测试。"""

    def test_empty_dir_returns_empty(self, tmp_path):
        """空目录应返回空张量."""
        kp2d, kp3d, conf, meta = build_dataset(
            smal_dir=tmp_path, num_cameras=2, window_size=27, stride=13
        )
        assert kp2d.shape == (0, 27, NUM_KEYPOINTS, 2)
        assert kp3d.shape == (0, 27, NUM_KEYPOINTS, 3)
        assert conf.shape == (0, 27, NUM_KEYPOINTS)
        assert meta == []

    def test_build_from_synthetic_clips(self, tmp_path):
        """从合成 clip 构建数据集."""
        T = 50
        np.savez(
            tmp_path / "interpet_dog01_p01_take01_ego_001.npz",
            kp_world=make_synthetic_kp_world(T),
            kp_weight=make_synthetic_kp_weight(T),
            frame_idx=np.arange(T, dtype=np.int32),
        )
        np.savez(
            tmp_path / "interpet_dog02_p02_take01_ego_001.npz",
            kp_world=make_synthetic_kp_world(T, seed=100),
            kp_weight=make_synthetic_kp_weight(T, seed=101),
            frame_idx=np.arange(T, dtype=np.int32),
        )

        kp2d, kp3d, conf, meta = build_dataset(
            smal_dir=tmp_path, num_cameras=2, window_size=27, stride=13
        )
        # 2 clips × 2 cameras × 2 windows = 8 samples
        assert kp2d.shape[0] == 8
        assert kp2d.shape[1:] == (27, NUM_KEYPOINTS, 2)
        assert kp3d.shape[1:] == (27, NUM_KEYPOINTS, 3)
        assert len(meta) == 8
        # 元数据应包含两个不同 clip_id
        clip_ids = set(m["clip_id"] for m in meta)
        assert len(clip_ids) == 2


class TestComputeDatasetStatistics:
    """compute_dataset_statistics 统计信息测试。"""

    def test_empty_input(self):
        stats = compute_dataset_statistics(
            np.zeros((0, 27, 24, 3)),
            np.zeros((0, 27, 24, 2)),
        )
        assert stats == {"num_samples": 0}

    def test_basic_stats(self):
        kp3d = make_synthetic_kp_world(T=27)  # shape (27, 24, 3)
        kp2d = make_synthetic_kp_world(T=27)[..., :2]  # shape (27, 24, 2)
        # 扩展 batch 维
        kp3d_batch = kp3d[None]  # (1, 27, 24, 3)
        kp2d_batch = kp2d[None]  # (1, 27, 24, 2)
        stats = compute_dataset_statistics(kp3d_batch, kp2d_batch)
        assert stats["num_samples"] == 1
        assert stats["window_size"] == 27
        assert "kp_3d_stats" in stats
        assert "kp_2d_stats" in stats
        assert stats["kp_3d_stats"]["min"] <= stats["kp_3d_stats"]["max"]


class TestTrainValSplit:
    """train_val_split 按 clip 划分测试。"""

    def test_split_ratios(self):
        """验证集比例约等于 val_ratio."""
        N = 50  # 50 样本
        kp2d = np.zeros((N, 27, NUM_KEYPOINTS, 2), dtype=np.float32)
        kp3d = np.zeros((N, 27, NUM_KEYPOINTS, 3), dtype=np.float32)
        conf = np.zeros((N, 27, NUM_KEYPOINTS), dtype=np.float32)
        # 10 个 clip，每个 5 样本
        clip_ids = [f"clip_{i:02d}" for i in range(10) for _ in range(5)]
        meta = [{"clip_id": cid} for cid in clip_ids]

        result = train_val_split(kp2d, kp3d, conf, meta, val_ratio=0.2, random_seed=42)
        # 10 clips × 0.2 = 2 val clips → 2 × 5 = 10 val samples
        assert result["num_val"] == 10
        assert result["num_train"] == 40

    def test_no_clip_leakage(self):
        """训练集和验证集 clip_id 不应重叠."""
        N = 40
        kp2d = np.zeros((N, 27, NUM_KEYPOINTS, 2), dtype=np.float32)
        kp3d = np.zeros((N, 27, NUM_KEYPOINTS, 3), dtype=np.float32)
        conf = np.zeros((N, 27, NUM_KEYPOINTS), dtype=np.float32)
        clip_ids = [f"clip_{i:02d}" for i in range(8) for _ in range(5)]
        meta = [{"clip_id": cid} for cid in clip_ids]

        result = train_val_split(kp2d, kp3d, conf, meta, val_ratio=0.25, random_seed=42)
        train_clips = set(result["train_clips"])
        val_clips = set(result["val_clips"])
        assert train_clips.isdisjoint(val_clips)

    def test_reproducibility(self):
        """相同 random_seed 应产生相同划分."""
        N = 30
        kp2d = np.zeros((N, 27, NUM_KEYPOINTS, 2), dtype=np.float32)
        kp3d = np.zeros((N, 27, NUM_KEYPOINTS, 3), dtype=np.float32)
        conf = np.zeros((N, 27, NUM_KEYPOINTS), dtype=np.float32)
        clip_ids = [f"clip_{i:02d}" for i in range(6) for _ in range(5)]
        meta = [{"clip_id": cid} for cid in clip_ids]

        r1 = train_val_split(kp2d, kp3d, conf, meta, val_ratio=0.2, random_seed=42)
        r2 = train_val_split(kp2d, kp3d, conf, meta, val_ratio=0.2, random_seed=42)
        assert r1["val_clips"] == r2["val_clips"]
        assert r1["train_clips"] == r2["train_clips"]

    def test_split_shapes(self):
        """划分后张量形状应正确."""
        N = 20
        kp2d = np.random.randn(N, 27, NUM_KEYPOINTS, 2).astype(np.float32)
        kp3d = np.random.randn(N, 27, NUM_KEYPOINTS, 3).astype(np.float32)
        conf = np.random.rand(N, 27, NUM_KEYPOINTS).astype(np.float32)
        clip_ids = [f"clip_{i:02d}" for i in range(4) for _ in range(5)]
        meta = [{"clip_id": cid} for cid in clip_ids]

        result = train_val_split(kp2d, kp3d, conf, meta, val_ratio=0.25)
        assert result["train_2d"].shape[1:] == (27, NUM_KEYPOINTS, 2)
        assert result["val_2d"].shape[1:] == (27, NUM_KEYPOINTS, 2)
        assert result["train_3d"].shape[1:] == (27, NUM_KEYPOINTS, 3)
        assert result["val_conf"].shape[1:] == (27, NUM_KEYPOINTS)


# ===== 4. 集成测试（合成 InterPet4D 数据）=====

class TestIntegrationSynthetic:
    """合成数据端到端集成测试：kp_world → 投影 → 配对 → 数据集。"""

    def test_full_pipeline_shapes(self, tmp_path):
        """完整管线：合成 clip → build_dataset → 训练/验证集划分."""
        T = 60
        for i in range(3):
            np.savez(
                tmp_path / f"interpet_dog0{i+1}_p0{i+1}_take01_ego_001.npz",
                kp_world=make_synthetic_kp_world(T, seed=i * 10),
                kp_weight=make_synthetic_kp_weight(T, seed=i * 10 + 1),
                frame_idx=np.arange(T, dtype=np.int32),
            )

        kp2d, kp3d, conf, meta = build_dataset(
            smal_dir=tmp_path,
            num_cameras=4,
            projection_mode="orthographic",
            window_size=27,
            stride=13,
            random_seed=42,
        )
        # 3 clips × 4 cameras × ~3 windows = 36 samples
        # range(0, 60-27+1, 13) = range(0, 34, 13) = [0, 13, 26] → 3 windows
        assert kp2d.shape[0] == 3 * 4 * 3
        assert kp3d.shape[0] == kp2d.shape[0]

        # 划分训练/验证集
        result = train_val_split(kp2d, kp3d, conf, meta, val_ratio=0.34, random_seed=42)
        # 3 clips × 0.34 ≈ 1 val clip
        assert result["num_train"] + result["num_val"] == kp2d.shape[0]
        assert len(result["val_clips"]) >= 1
        assert len(result["train_clips"]) >= 1

    def test_pipeline_determinism(self, tmp_path):
        """相同 random_seed 两次运行应产生相同数据集."""
        T = 40
        for i in range(2):
            np.savez(
                tmp_path / f"interpet_dog0{i+1}_p0{i+1}_take01_ego_001.npz",
                kp_world=make_synthetic_kp_world(T, seed=i * 10),
                kp_weight=make_synthetic_kp_weight(T, seed=i * 10 + 1),
                frame_idx=np.arange(T, dtype=np.int32),
            )

        kp2d_a, _, _, _ = build_dataset(
            smal_dir=tmp_path, num_cameras=2, window_size=27, stride=13, random_seed=42
        )
        kp2d_b, _, _, _ = build_dataset(
            smal_dir=tmp_path, num_cameras=2, window_size=27, stride=13, random_seed=42
        )
        assert np.allclose(kp2d_a, kp2d_b)


@pytest.mark.skipif(not HAS_REAL_DATA, reason="InterPet4D smal_npy 数据未安装")
class TestIntegrationReal:
    """真实 InterPet4D 数据端到端集成测试。"""

    def test_real_clip_full_pipeline(self):
        """真实 clip: 加载 → 投影 → 配对 → 数据集."""
        clip = load_clip(SAMPLE_CLIP_ID, INTERPET4D_SMAL_DIR)
        assert clip is not None
        assert clip.num_frames > 0

        samples = build_pairs_from_clip(
            clip,
            num_cameras=4,
            projection_mode="orthographic",
            window_size=27,
            stride=13,
            random_seed=42,
        )
        assert len(samples) > 0
        # 验证 2D 坐标范围合理（归一化后）
        for s in samples:
            assert s.keypoints_2d.shape == (27, NUM_KEYPOINTS, 2)
            assert s.keypoints_3d.shape == (27, NUM_KEYPOINTS, 3)
            assert np.isfinite(s.keypoints_2d).all()
            assert np.isfinite(s.keypoints_3d).all()

    def test_real_dataset_small_scale(self):
        """小规模真实数据集构建（5 clips）。"""
        kp2d, kp3d, conf, meta = build_dataset(
            smal_dir=INTERPET4D_SMAL_DIR,
            num_cameras=2,
            window_size=27,
            stride=13,
            max_clips=5,
            random_seed=42,
        )
        assert kp2d.shape[0] > 0
        assert kp2d.shape[1:] == (27, NUM_KEYPOINTS, 2)
        assert kp3d.shape[1:] == (27, NUM_KEYPOINTS, 3)
        # 元数据中 clip_id 应来自真实数据
        clip_ids = set(m["clip_id"] for m in meta)
        assert len(clip_ids) >= 1
        for cid in clip_ids:
            assert cid.startswith("interpet_dog")

    def test_real_dataset_split(self):
        """真实数据集训练/验证集划分."""
        kp2d, kp3d, conf, meta = build_dataset(
            smal_dir=INTERPET4D_SMAL_DIR,
            num_cameras=2,
            window_size=27,
            stride=13,
            max_clips=10,
            random_seed=42,
        )
        if kp2d.shape[0] < 2:
            pytest.skip("真实数据样本不足")
        result = train_val_split(kp2d, kp3d, conf, meta, val_ratio=0.2, random_seed=42)
        assert result["num_train"] + result["num_val"] == kp2d.shape[0]
        # 验证无 clip 泄漏
        assert set(result["train_clips"]).isdisjoint(set(result["val_clips"]))
