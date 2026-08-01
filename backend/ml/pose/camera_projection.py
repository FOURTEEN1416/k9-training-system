"""合成相机参数 + 3D→2D 投影（Phase 3.3b）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.3b
依据: dev-docs/research/RESEARCH_3D_POSE_RECONSTRUCTION.md §6.2 + dev-docs/stages/phase-3.md §3.3b

设计目标:
    InterPet4D v1 无原始视频文件，无法直接做「视频→2D→3D」配对。
    本模块通过合成相机参数将 kp_world 3D 坐标投影回 2D，构造 (2D 投影, 3D 真值) 配对，
    用于 MotionBERT-Lite 2D-to-3D lifting 微调。

投影模型:
    1. 正交投影（orthographic）— 简单稳定，无深度歧义，适合 MotionBERT 输入归一化
    2. 透视投影（perspective）— 模拟真实相机，更接近 YOLO26-pose 实际输出
    两种模型都支持，默认正交（MotionBERT 训练用正交归一化坐标）

合成相机参数策略:
    - 多视角采样：围绕犬只球面采样 N 个视角（方位角 + 仰角）
    - 多距离采样：模拟不同相机距离（影响透视投影的缩放）
    - 随机扰动：轻微旋转/平移扰动增强泛化
    - 固定随机种子：可复现

与 YOLO26-pose 对齐:
    - YOLO26-pose 输出 (x, y, conf) 像素坐标
    - 投影后 2D 坐标归一化到 [-1, 1]（MotionBERT 标准）
    - 推理时 YOLO26-pose 输出也归一化到 [-1, 1]，保持训练/推理一致

不引入兜底层:
    - 不使用真实相机标定参数（InterPet4D 无标定数据）
    - 不做相机畸变矫正（合成数据无畸变）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# 默认相机参数
DEFAULT_FOCAL_LENGTH = 500.0  # 像素（模拟 640x480 分辨率下的中等焦距）
DEFAULT_IMAGE_SIZE = (640, 480)  # (width, height)
DEFAULT_CAMERA_DISTANCE = 3.0  # 米（相机到犬只中心的距离）
DEFAULT_AZIMUTH_RANGE = (-180.0, 180.0)  # 方位角范围（度）
DEFAULT_ELEVATION_RANGE = (-30.0, 60.0)  # 仰角范围（度，模拟人眼/三脚架高度）


@dataclass
class SyntheticCamera:
    """合成相机参数.

    相机模型: look-at 相机
    - position: 相机位置（世界坐标，米）
    - target: 目标点（世界坐标，米），默认原点
    - up: 上方向（世界坐标），默认 [0, 1, 0]
    - focal_length: 焦距（像素）
    - image_size: 图像尺寸 (width, height)

    Attributes:
        position: (3,) 相机位置
        target: (3,) 目标点
        up: (3,) 上方向
        focal_length: 焦距（像素）
        image_size: (width, height)
    """

    position: np.ndarray  # (3,)
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], dtype=np.float32))
    focal_length: float = DEFAULT_FOCAL_LENGTH
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32).reshape(3)
        self.target = np.asarray(self.target, dtype=np.float32).reshape(3)
        self.up = np.asarray(self.up, dtype=np.float32).reshape(3)

    @property
    def width(self) -> int:
        return int(self.image_size[0])

    @property
    def height(self) -> int:
        return int(self.image_size[1])

    def view_matrix(self) -> np.ndarray:
        """计算 view 矩阵（world → camera）.

        Returns:
            (4, 4) view 矩阵
        """
        # 相机前向 = target - position（归一化）
        forward = self.target - self.position
        forward = forward / (np.linalg.norm(forward) + 1e-12)
        # 右向 = forward × up
        right = np.cross(forward, self.up)
        right = right / (np.linalg.norm(right) + 1e-12)
        # 重新计算 up（保证正交）
        up_orth = np.cross(right, forward)

        # view 矩阵 [R | -R·t]
        R = np.stack([right, up_orth, -forward], axis=0)  # (3, 3)
        t = -R @ self.position  # (3,)

        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R
        view[:3, 3] = t
        return view

    def projection_matrix(self, mode: str = "orthographic") -> np.ndarray:
        """计算投影矩阵.

        Args:
            mode: "orthographic" 或 "perspective"

        Returns:
            (4, 4) 投影矩阵
        """
        w, h = self.image_size
        f = self.focal_length

        if mode == "orthographic":
            # 正交投影：归一化到 [-1, 1]
            # 假设犬只尺寸 ~1m，投影到 [-1, 1] 需要缩放 2/size
            proj = np.array([
                [2.0 / w * f / 100, 0, 0, 0],
                [0, 2.0 / h * f / 100, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
            # 简化：直接用归一化（x/w, y/h）
            proj = np.array([
                [2.0 / w, 0, 0, -1.0],
                [0, -2.0 / h, 0, 1.0],  # y 轴翻转（图像 y 向下）
                [0, 0, 0, 0],  # 正交丢弃 z
                [0, 0, 0, 1],
            ], dtype=np.float32)
            return proj
        elif mode == "perspective":
            # 透视投影: x' = f * x / z, y' = f * y / z
            # 齐次坐标矩阵
            proj = np.array([
                [f, 0, w / 2.0, 0],
                [0, f, h / 2.0, 0],
                [0, 0, 1, 0],
            ], dtype=np.float32)  # (3, 4)
            return proj
        else:
            raise ValueError(f"未知投影模式: {mode}")


def project_3d_to_2d(
    keypoints_3d: np.ndarray,
    camera: SyntheticCamera,
    mode: str = "orthographic",
    normalize: bool = True,
) -> np.ndarray:
    """将 3D 关键点投影到 2D.

    Args:
        keypoints_3d: (..., 3) 或 (..., 4) 齐次坐标
        camera: 合成相机
        mode: "orthographic" 或 "perspective"
        normalize: 是否归一化到 [-1, 1]（MotionBERT 标准）

    Returns:
        np.ndarray (..., 2) — 2D 坐标
            normalize=True: [-1, 1] 归一化
            normalize=False: 像素坐标
    """
    kp3d = np.asarray(keypoints_3d, dtype=np.float32)
    orig_shape = kp3d.shape
    if kp3d.shape[-1] == 4:
        # 齐次坐标，转回 3D
        kp3d = kp3d[..., :3] / (kp3d[..., 3:4] + 1e-12)
    # 统一 reshape 到 (N, 3) 以便矩阵运算
    kp3d = kp3d.reshape(-1, 3)

    n_points = kp3d.shape[0]

    if mode == "orthographic":
        # 正交投影：view 矩阵变换 → 取 xy → 归一化
        view = camera.view_matrix()  # (4, 4)
        kp_homo = np.concatenate(
            [kp3d, np.ones((n_points, 1), dtype=np.float32)], axis=1
        )  # (N, 4)
        kp_camera = (view @ kp_homo.T).T  # (N, 4)

        # 取 x, y（相机坐标系）
        kp_2d = kp_camera[:, :2]  # (N, 2)

        if normalize:
            # 归一化到 [-1, 1]：假设犬只尺寸 ~2m，缩放 1/1.0
            kp_2d = kp_2d / 1.0  # 米 → 单位坐标
        else:
            # 转像素坐标
            w, h = camera.image_size
            kp_2d[:, 0] = (kp_2d[:, 0] + 1.0) * 0.5 * w
            kp_2d[:, 1] = (-kp_2d[:, 1] + 1.0) * 0.5 * h  # y 翻转

    elif mode == "perspective":
        # 透视投影
        view = camera.view_matrix()  # (4, 4)
        proj = camera.projection_matrix(mode="perspective")  # (3, 4)

        kp_homo = np.concatenate(
            [kp3d, np.ones((n_points, 1), dtype=np.float32)], axis=1
        )  # (N, 4)
        kp_camera = (view @ kp_homo.T).T  # (N, 4)
        kp_image_homo = (proj @ kp_camera.T).T  # (N, 3)

        # 齐次 → 笛卡尔（除以 z）
        z = kp_image_homo[:, 2:3] + 1e-12
        kp_2d = kp_image_homo[:, :2] / z  # (N, 2) 像素坐标

        if normalize:
            w, h = camera.image_size
            kp_2d[:, 0] = 2.0 * kp_2d[:, 0] / w - 1.0
            kp_2d[:, 1] = 2.0 * kp_2d[:, 1] / h - 1.0
    else:
        raise ValueError(f"未知投影模式: {mode}")

    return kp_2d.reshape(orig_shape[:-1] + (2,))


def generate_synthetic_cameras(
    num_cameras: int = 8,
    distance: float = DEFAULT_CAMERA_DISTANCE,
    azimuth_range: Tuple[float, float] = DEFAULT_AZIMUTH_RANGE,
    elevation_range: Tuple[float, float] = DEFAULT_ELEVATION_RANGE,
    target: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    focal_length: float = DEFAULT_FOCAL_LENGTH,
    random_seed: int = 42,
) -> List[SyntheticCamera]:
    """生成合成相机阵列（球面采样 + 随机扰动）.

    Args:
        num_cameras: 相机数量
        distance: 相机距离（米）
        azimuth_range: 方位角范围（度）
        elevation_range: 仰角范围（度）
        target: 目标点（默认原点）
        image_size: 图像尺寸
        focal_length: 焦距
        random_seed: 随机种子

    Returns:
        List[SyntheticCamera] — 合成相机列表
    """
    rng = np.random.default_rng(random_seed)
    if target is None:
        target = np.zeros(3, dtype=np.float32)
    else:
        target = np.asarray(target, dtype=np.float32).reshape(3)

    # 均匀采样方位角 + 随机仰角
    if num_cameras == 1:
        azimuths = np.array([0.0])
    else:
        azimuths = np.linspace(
            azimuth_range[0], azimuth_range[1], num_cameras, endpoint=False
        )
    # 随机扰动方位角
    azimuth_jitter = rng.uniform(-5, 5, size=num_cameras)
    azimuths = azimuths + azimuth_jitter

    # 随机仰角
    elevations = rng.uniform(
        elevation_range[0], elevation_range[1], size=num_cameras
    )

    cameras = []
    for i in range(num_cameras):
        az = np.deg2rad(azimuths[i])
        el = np.deg2rad(elevations[i])

        # 球面坐标 → 笛卡尔
        x = distance * np.cos(el) * np.cos(az)
        y = distance * np.sin(el)
        z = distance * np.cos(el) * np.sin(az)

        position = target + np.array([x, y, z], dtype=np.float32)

        cameras.append(
            SyntheticCamera(
                position=position,
                target=target.copy(),
                up=np.array([0, 1, 0], dtype=np.float32),
                focal_length=focal_length,
                image_size=image_size,
            )
        )

    return cameras


def project_clip_to_2d(
    kp_world: np.ndarray,
    cameras: List[SyntheticCamera],
    mode: str = "orthographic",
    normalize: bool = True,
) -> np.ndarray:
    """将整段 clip 的 kp_world 投影到多相机 2D.

    Args:
        kp_world: (T, 24, 3) 世界坐标
        cameras: 相机列表
        mode: 投影模式
        normalize: 是否归一化

    Returns:
        np.ndarray (num_cameras, T, 24, 2) — 每个相机的 2D 投影
    """
    kp_world = np.asarray(kp_world, dtype=np.float32)
    num_cameras = len(cameras)
    T, V, _ = kp_world.shape

    projections = np.zeros((num_cameras, T, V, 2), dtype=np.float32)
    for i, cam in enumerate(cameras):
        # 投影 (T, 24, 3) → (T, 24, 2)
        projections[i] = project_3d_to_2d(kp_world, cam, mode=mode, normalize=normalize)

    return projections


__all__ = [
    "SyntheticCamera",
    "project_3d_to_2d",
    "generate_synthetic_cameras",
    "project_clip_to_2d",
    "DEFAULT_FOCAL_LENGTH",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_CAMERA_DISTANCE",
    "DEFAULT_AZIMUTH_RANGE",
    "DEFAULT_ELEVATION_RANGE",
]
