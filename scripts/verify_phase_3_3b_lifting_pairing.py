"""Phase 3.3b 端到端验证 — 2D-3D Lifting 配对管线.

Owner: ML 开发
Phase: 3.3b

目标:
    1. 验证 InterPet4D kp_world 加载（真实数据 226 clips）
    2. 验证合成相机阵列生成 + 3D→2D 投影
    3. 验证 2D-3D 配对构建（滑动窗口 + 归一化）
    4. 验证 train/val 划分（无 clip 泄漏）
    5. 输出验证报告到 reports/phase-3.3b-lifting-pairing-validation.json

验证项:
    - 数据加载: clip 数量、帧数、关键点形状
    - 投影正确性: 2D 坐标范围、正交/透视模式
    - 配对构建: 样本数 = clips × cameras × windows
    - 归一化正确性: 根关节中心化、尺度因子
    - 数据集划分: 训练/验证集无 clip 重叠
    - 统计信息: 2D/3D 坐标分布

注意:
    - 本脚本为验证脚本，不属于 production 代码
    - 验证完成后保留作为 3.3b 验证证据
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.pose.interpet4d_loader import (
    NUM_KEYPOINTS,
    DEFAULT_INTERPET4D_SMAL_DIR,
    list_clips,
    load_clip,
    load_all_clips,
    get_dataset_statistics,
)
from backend.ml.pose.camera_projection import (
    SyntheticCamera,
    project_3d_to_2d,
    generate_synthetic_cameras,
    project_clip_to_2d,
)
from backend.ml.pose.lifting_pairing import (
    LiftingSample,
    normalize_3d_keypoints,
    denormalize_3d_keypoints,
    build_pairs_from_clip,
    build_dataset,
    compute_dataset_statistics,
    train_val_split,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_STRIDE,
    ROOT_KEYPOINT_IDX,
)


REPORT_PATH = PROJECT_ROOT / "reports" / "phase-3.3b-lifting-pairing-validation.json"


def validate_data_loading() -> dict:
    """阶段 1: 验证 InterPet4D 数据加载."""
    print("\n[阶段 1] 验证 InterPet4D 数据加载...")

    smal_dir = Path(DEFAULT_INTERPET4D_SMAL_DIR)
    if not smal_dir.exists():
        return {"status": "SKIP", "reason": f"数据目录不存在: {smal_dir}"}

    clip_ids = list_clips(smal_dir)
    print(f"  发现 clips: {len(clip_ids)}")

    # 加载单个 clip 验证
    sample_clip = load_clip(clip_ids[0], smal_dir, load_optional=True)
    assert sample_clip is not None, f"无法加载 sample clip: {clip_ids[0]}"
    print(f"  Sample clip: {sample_clip.clip_id}")
    print(f"    kp_world shape: {sample_clip.kp_world.shape}")
    print(f"    kp_weight shape: {sample_clip.kp_weight.shape}")
    print(f"    num_frames: {sample_clip.num_frames}")
    print(f"    dog_id: {sample_clip.dog_id}")

    # 形状校验
    assert sample_clip.kp_world.shape[1] == NUM_KEYPOINTS, "kp_world 关键点数 != 24"
    assert sample_clip.kp_world.shape[2] == 3, "kp_world 坐标维度 != 3"
    assert sample_clip.kp_weight.shape == (sample_clip.num_frames, NUM_KEYPOINTS)

    # 数据集统计（采样前 10 个）
    stats = get_dataset_statistics(smal_dir, max_clips=10)
    print(f"  数据集统计（前 10 clips）:")
    print(f"    total_frames: {stats['total_frames']}")
    print(f"    num_dogs: {stats['num_dogs']}")
    print(f"    frame_count mean: {stats['frame_count_stats']['mean']}")

    return {
        "status": "PASS",
        "total_clips": len(clip_ids),
        "sample_clip_id": sample_clip.clip_id,
        "sample_kp_world_shape": list(sample_clip.kp_world.shape),
        "sample_num_frames": sample_clip.num_frames,
        "dataset_stats_sample": stats,
    }


def validate_camera_projection() -> dict:
    """阶段 2: 验证合成相机 + 3D→2D 投影."""
    print("\n[阶段 2] 验证合成相机 + 3D→2D 投影...")

    smal_dir = Path(DEFAULT_INTERPET4D_SMAL_DIR)
    clip_ids = list_clips(smal_dir)
    clip = load_clip(clip_ids[0], smal_dir)
    assert clip is not None

    # 生成 8 个合成相机
    cameras = generate_synthetic_cameras(num_cameras=8, distance=3.0, random_seed=42)
    print(f"  生成相机数: {len(cameras)}")
    for i, cam in enumerate(cameras[:3]):  # 打印前 3 个
        print(f"    Camera {i}: position={cam.position}, target={cam.target}")

    # 正交投影
    proj_ortho = project_clip_to_2d(clip.kp_world, cameras, mode="orthographic", normalize=True)
    print(f"  正交投影 shape: {proj_ortho.shape}")
    print(f"  正交投影范围: [{proj_ortho.min():.4f}, {proj_ortho.max():.4f}]")
    assert proj_ortho.shape == (8, clip.num_frames, NUM_KEYPOINTS, 2)
    assert np.isfinite(proj_ortho).all(), "正交投影包含非有限值"

    # 透视投影
    proj_persp = project_clip_to_2d(clip.kp_world, cameras, mode="perspective", normalize=True)
    print(f"  透视投影 shape: {proj_persp.shape}")
    print(f"  透视投影范围: [{proj_persp.min():.4f}, {proj_persp.max():.4f}]")
    assert proj_persp.shape == (8, clip.num_frames, NUM_KEYPOINTS, 2)
    assert np.isfinite(proj_persp).all(), "透视投影包含非有限值"

    # 相机一致性：每个相机单独投影应与批量投影一致
    for i, cam in enumerate(cameras):
        single = project_3d_to_2d(clip.kp_world, cam, mode="orthographic")
        assert np.allclose(proj_ortho[i], single), f"相机 {i} 投影不一致"
    print(f"  相机一致性验证: PASS（8 个相机批量 vs 单独投影一致）")

    return {
        "status": "PASS",
        "num_cameras": len(cameras),
        "orthographic_shape": list(proj_ortho.shape),
        "orthographic_range": [round(float(proj_ortho.min()), 4), round(float(proj_ortho.max()), 4)],
        "perspective_shape": list(proj_persp.shape),
        "perspective_range": [round(float(proj_persp.min()), 4), round(float(proj_persp.max()), 4)],
        "camera_consistency": True,
    }


def validate_normalization() -> dict:
    """阶段 3: 验证 3D 归一化 + 反归一化."""
    print("\n[阶段 3] 验证 3D 关键点归一化...")

    smal_dir = Path(DEFAULT_INTERPET4D_SMAL_DIR)
    clip_ids = list_clips(smal_dir)
    clip = load_clip(clip_ids[0], smal_dir)
    assert clip is not None

    # bone_length 模式
    normalized, scale = normalize_3d_keypoints(
        clip.kp_world, root_idx=ROOT_KEYPOINT_IDX, scale_mode="bone_length"
    )
    print(f"  bone_length 归一化:")
    print(f"    scale: {scale:.4f}")
    print(f"    normalized shape: {normalized.shape}")
    print(f"    normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")

    # 根关节中心化校验
    root_pts = normalized[:, ROOT_KEYPOINT_IDX, :]
    root_norm = float(np.linalg.norm(root_pts))
    print(f"    根关节中心化: root ||·||={root_norm:.6f} (应≈0)")
    assert root_norm < 1e-5, f"根关节未正确中心化: ||·||={root_norm}"

    # 反归一化往返测试
    root_positions = clip.kp_world[:, ROOT_KEYPOINT_IDX, :]
    restored = denormalize_3d_keypoints(normalized, scale=scale)
    restored = restored + root_positions[:, None, :]
    max_err = float(np.abs(restored - clip.kp_world).max())
    print(f"    反归一化往返误差: max_err={max_err:.6e}")
    assert max_err < 1e-4, f"反归一化往返误差过大: {max_err}"

    # bbox 模式
    normalized_bbox, scale_bbox = normalize_3d_keypoints(
        clip.kp_world, root_idx=ROOT_KEYPOINT_IDX, scale_mode="bbox"
    )
    print(f"  bbox 归一化: scale={scale_bbox:.4f}")

    # none 模式
    normalized_none, scale_none = normalize_3d_keypoints(
        clip.kp_world, root_idx=ROOT_KEYPOINT_IDX, scale_mode="none"
    )
    assert scale_none == 1.0, "none 模式 scale 应为 1.0"
    print(f"  none 模式: scale={scale_none}（仅中心化）")

    return {
        "status": "PASS",
        "bone_length_scale": round(float(scale), 4),
        "root_centering_error": round(root_norm, 6),
        "round_trip_max_error": round(max_err, 8),
        "bbox_scale": round(float(scale_bbox), 4),
        "none_scale": float(scale_none),
    }


def validate_pairing_construction() -> dict:
    """阶段 4: 验证 2D-3D 配对构建."""
    print("\n[阶段 4] 验证 2D-3D 配对构建...")

    smal_dir = Path(DEFAULT_INTERPET4D_SMAL_DIR)
    clip_ids = list_clips(smal_dir)
    clip = load_clip(clip_ids[0], smal_dir)
    assert clip is not None

    # 构建配对
    samples = build_pairs_from_clip(
        clip,
        num_cameras=4,
        projection_mode="orthographic",
        window_size=DEFAULT_WINDOW_SIZE,
        stride=DEFAULT_WINDOW_STRIDE,
        random_seed=42,
    )
    print(f"  单 clip 配对样本数: {len(samples)}")
    print(f"  窗口大小: {DEFAULT_WINDOW_SIZE}, 步长: {DEFAULT_WINDOW_STRIDE}")

    # 计算预期窗口数
    T = clip.num_frames
    if T < DEFAULT_WINDOW_SIZE:
        expected_windows = 1
    else:
        expected_windows = len(list(range(0, T - DEFAULT_WINDOW_SIZE + 1, DEFAULT_WINDOW_STRIDE)))
    expected_samples = 4 * expected_windows
    print(f"  预期窗口数: {expected_windows}, 预期样本数: {expected_samples}")
    assert len(samples) == expected_samples, f"样本数不匹配: {len(samples)} != {expected_samples}"

    # 样本形状校验
    for i, s in enumerate(samples):
        assert s.keypoints_2d.shape == (DEFAULT_WINDOW_SIZE, NUM_KEYPOINTS, 2), \
            f"样本 {i} 2D shape 错误: {s.keypoints_2d.shape}"
        assert s.keypoints_3d.shape == (DEFAULT_WINDOW_SIZE, NUM_KEYPOINTS, 3), \
            f"样本 {i} 3D shape 错误: {s.keypoints_3d.shape}"
        assert s.confidence.shape == (DEFAULT_WINDOW_SIZE, NUM_KEYPOINTS), \
            f"样本 {i} conf shape 错误: {s.confidence.shape}"
        assert s.clip_id == clip.clip_id
        assert s.scale > 0

    print(f"  样本形状校验: PASS（全部 {len(samples)} 样本）")
    print(f"  Sample 0: camera_idx={samples[0].camera_idx}, frame_start={samples[0].frame_start}, scale={samples[0].scale:.4f}")

    return {
        "status": "PASS",
        "clip_id": clip.clip_id,
        "num_frames": T,
        "num_cameras": 4,
        "expected_windows": expected_windows,
        "num_samples": len(samples),
        "sample_2d_shape": list(samples[0].keypoints_2d.shape),
        "sample_3d_shape": list(samples[0].keypoints_3d.shape),
    }


def validate_dataset_construction() -> dict:
    """阶段 5: 验证批量数据集构建 + train/val 划分."""
    print("\n[阶段 5] 验证批量数据集构建 + train/val 划分...")

    smal_dir = Path(DEFAULT_INTERPET4D_SMAL_DIR)
    # 使用前 10 个 clips 做小规模验证
    kp2d, kp3d, conf, meta = build_dataset(
        smal_dir=smal_dir,
        num_cameras=4,
        projection_mode="orthographic",
        window_size=DEFAULT_WINDOW_SIZE,
        stride=DEFAULT_WINDOW_STRIDE,
        max_clips=10,
        random_seed=42,
    )
    print(f"  数据集构建（10 clips × 4 cameras）:")
    print(f"    2D shape: {kp2d.shape}")
    print(f"    3D shape: {kp3d.shape}")
    print(f"    conf shape: {conf.shape}")
    print(f"    样本数: {len(meta)}")

    assert kp2d.shape[0] == len(meta), "样本数不一致"
    assert kp2d.shape[1:] == (DEFAULT_WINDOW_SIZE, NUM_KEYPOINTS, 2)
    assert kp3d.shape[1:] == (DEFAULT_WINDOW_SIZE, NUM_KEYPOINTS, 3)

    # 元数据校验
    clip_ids_in_meta = set(m["clip_id"] for m in meta)
    print(f"    涉及 clip 数: {len(clip_ids_in_meta)}")

    # 数据集统计
    stats = compute_dataset_statistics(kp3d, kp2d)
    print(f"    3D 坐标范围: [{stats['kp_3d_stats']['min']}, {stats['kp_3d_stats']['max']}]")
    print(f"    2D 坐标范围: [{stats['kp_2d_stats']['min']}, {stats['kp_2d_stats']['max']}]")

    # train/val 划分
    split = train_val_split(kp2d, kp3d, conf, meta, val_ratio=0.2, random_seed=42)
    print(f"  train/val 划分:")
    print(f"    train: {split['num_train']} 样本, {len(split['train_clips'])} clips")
    print(f"    val: {split['num_val']} 样本, {len(split['val_clips'])} clips")

    # 无 clip 泄漏校验
    train_clips = set(split["train_clips"])
    val_clips = set(split["val_clips"])
    overlap = train_clips & val_clips
    assert not overlap, f"clip 泄漏: {overlap}"
    print(f"    clip 泄漏校验: PASS（无重叠）")

    # 划分总数校验
    assert split["num_train"] + split["num_val"] == len(meta), "划分总数不一致"

    return {
        "status": "PASS",
        "dataset_2d_shape": list(kp2d.shape),
        "dataset_3d_shape": list(kp3d.shape),
        "num_samples": len(meta),
        "num_clips": len(clip_ids_in_meta),
        "kp_3d_stats": stats["kp_3d_stats"],
        "kp_2d_stats": stats["kp_2d_stats"],
        "split": {
            "num_train": split["num_train"],
            "num_val": split["num_val"],
            "num_train_clips": len(split["train_clips"]),
            "num_val_clips": len(split["val_clips"]),
            "clip_leakage": False,
        },
    }


def validate_determinism() -> dict:
    """阶段 6: 验证管线可复现性."""
    print("\n[阶段 6] 验证管线可复现性...")

    smal_dir = Path(DEFAULT_INTERPET4D_SMAL_DIR)
    # 两次相同 random_seed 构建，比较结果
    kp2d_a, _, _, _ = build_dataset(
        smal_dir=smal_dir, num_cameras=2, max_clips=3, random_seed=42,
        window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_WINDOW_STRIDE,
    )
    kp2d_b, _, _, _ = build_dataset(
        smal_dir=smal_dir, num_cameras=2, max_clips=3, random_seed=42,
        window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_WINDOW_STRIDE,
    )
    is_identical = np.array_equal(kp2d_a, kp2d_b)
    max_diff = float(np.abs(kp2d_a - kp2d_b).max())
    print(f"  两次构建结果: identical={is_identical}, max_diff={max_diff:.6e}")
    assert is_identical, f"管线不可复现: max_diff={max_diff}"

    return {
        "status": "PASS",
        "identical": is_identical,
        "max_diff": max_diff,
    }


def main():
    """主验证流程."""
    print("=" * 70)
    print("Phase 3.3b 端到端验证 — 2D-3D Lifting 配对管线")
    print("=" * 70)
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据目录: {DEFAULT_INTERPET4D_SMAL_DIR}")

    start_time = time.time()
    report = {
        "phase": "3.3b",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modules": {
            "interpet4d_loader": "backend/ml/pose/interpet4d_loader.py",
            "camera_projection": "backend/ml/pose/camera_projection.py",
            "lifting_pairing": "backend/ml/pose/lifting_pairing.py",
        },
    }

    try:
        report["stage_1_data_loading"] = validate_data_loading()
        report["stage_2_camera_projection"] = validate_camera_projection()
        report["stage_3_normalization"] = validate_normalization()
        report["stage_4_pairing_construction"] = validate_pairing_construction()
        report["stage_5_dataset_construction"] = validate_dataset_construction()
        report["stage_6_determinism"] = validate_determinism()
        report["overall_status"] = "PASS"
    except (AssertionError, Exception) as e:
        report["overall_status"] = "FAIL"
        report["error"] = str(e)
        import traceback
        report["traceback"] = traceback.format_exc()
        print(f"\n[FAIL] 验证失败: {e}")

    elapsed = time.time() - start_time
    report["elapsed_seconds"] = round(elapsed, 2)

    print("\n" + "=" * 70)
    print(f"验证完成: {report['overall_status']} (耗时 {elapsed:.2f}s)")
    print("=" * 70)

    # 写入报告
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告已保存: {REPORT_PATH}")

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
