"""MotionBERT 运行时推理（2D → 3D lifting）.

Owner: ML 开发
Phase: 3.3c

功能:
    - 加载微调后的 MotionBERT 模型或 ONNX
    - 输入: 2D 关键点序列 (T, 24, 2|3) — 来自 YOLO26-pose
    - 输出: 3D 关键点序列 (T, 24, 3) — 根关节中心化
    - 支持滑动窗口推理（处理任意长度视频）
    - 支持反归一化（恢复真实尺度）

集成:
    - 上游: backend.ml.pose.inference.PoseInferenceEngine 输出 2D 关键点
    - 下游: backend.ml.behavior (ST-GCN+BC) 消费 3D 关键点
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from backend.ml.pose.lifting_pairing import (
    DEFAULT_WINDOW_SIZE,
    ROOT_KEYPOINT_IDX,
    denormalize_3d_keypoints,
    normalize_3d_keypoints,
)
from backend.ml.pose.motionbert.config import MotionBERTConfig, load_config
from backend.ml.pose.motionbert.model import (
    DSTformerWrapper,
    build_model_from_config,
    load_pretrained_weights,
)

logger = logging.getLogger(__name__)


class MotionBERTLifter:
    """MotionBERT 2D→3D lifting 推理器.

    支持两种后端:
        - PyTorch（用于训练后立即验证）
        - ONNX Runtime（用于生产部署）
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        onnx_path: Optional[Union[str, Path]] = None,
        config: Optional[MotionBERTConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """初始化 lifter.

        Args:
            checkpoint_path: PyTorch checkpoint 路径（与 onnx_path 二选一）
            onnx_path: ONNX 模型路径（优先使用，部署推荐）
            config: 配置（None=从 checkpoint 读取）
            device: PyTorch 推理设备
        """
        self.config = config or MotionBERTConfig()
        self.device = device
        self.backend = "onnx" if onnx_path else "torch"

        if onnx_path is not None:
            import onnxruntime as ort
            self._ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"],
            )
            self._torch_model = None
            logger.info(f"[MotionBERTLifter] ONNX 后端: {onnx_path}")
        elif checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "config" in checkpoint and config is None:
                self.config = MotionBERTConfig(**checkpoint["config"])

            model = build_model_from_config(self.config).to(device)
            state_dict = checkpoint.get("model_pos", checkpoint)
            cleaned = {}
            for k, v in state_dict.items():
                key = k[7:] if k.startswith("module.") else k
                cleaned[key] = v
            model.load_state_dict(cleaned, strict=True)
            model.eval()
            self._torch_model = model
            self._ort_session = None
            logger.info(f"[MotionBERTLifter] PyTorch 后端: {checkpoint_path}")
        else:
            raise ValueError("必须提供 checkpoint_path 或 onnx_path")

    def lift(
        self,
        keypoints_2d: np.ndarray,
        window_size: Optional[int] = None,
        scale: float = 1.0,
        root_position: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """2D → 3D lifting.

        Args:
            keypoints_2d: (T, 24, 2) 或 (T, 24, 3) 2D 关键点序列
                - 归一化到 [-1, 1]（与训练一致）
                - 若只有 2 通道，自动补 0 confidence
            window_size: 滑动窗口大小（None=配置默认 27）
            scale: 3D 反归一化缩放因子
            root_position: (3,) 根关节原始位置（None=原点）

        Returns:
            (T, 24, 3) 3D 关键点（已反归一化）
        """
        kp2d = np.asarray(keypoints_2d, dtype=np.float32)
        if kp2d.ndim != 3:
            raise ValueError(f"输入必须是 (T, 24, C), got {kp2d.shape}")

        T, J, C = kp2d.shape
        if J != self.config.num_joints:
            raise ValueError(f"关节数不匹配: 期望 {self.config.num_joints}, got {J}")

        # 补 confidence channel
        if C == 2:
            conf = np.zeros((T, J, 1), dtype=np.float32)
            kp2d = np.concatenate([kp2d, conf], axis=-1)
        elif C != 3:
            raise ValueError(f"通道数必须是 2 或 3, got {C}")

        window_size = window_size or self.config.window_size

        # 滑动窗口推理
        if T <= window_size:
            # 单窗口，pad 到 window_size
            padded = self._pad_to_window(kp2d, window_size)
            pred_3d = self._infer_window(padded[np.newaxis])  # (1, T_pad, 24, 3)
            pred_3d = pred_3d[0, :T]  # 裁剪回原长度
        else:
            # 多窗口滑动
            pred_3d = self._sliding_window_infer(kp2d, window_size)

        # 反归一化
        if scale != 1.0 or root_position is not None:
            pred_3d = denormalize_3d_keypoints(
                pred_3d, scale=scale, root_position=root_position,
                root_idx=ROOT_KEYPOINT_IDX,
            )

        return pred_3d

    def _infer_window(self, batch_input: np.ndarray) -> np.ndarray:
        """单批次推理.

        Args:
            batch_input: (B, T, 24, 3)

        Returns:
            (B, T, 24, 3) 3D 预测
        """
        if self.backend == "onnx":
            out = self._ort_session.run(
                None, {"keypoints_2d": batch_input.astype(np.float32)}
            )[0]
            return out
        else:
            with torch.no_grad():
                x = torch.from_numpy(batch_input).to(self.device)
                pred = self._torch_model(x)
                if self.config.rootrel:
                    pred = pred - pred[..., 0:1, :]
                return pred.cpu().numpy()

    def _sliding_window_infer(
        self, kp2d: np.ndarray, window_size: int
    ) -> np.ndarray:
        """滑动窗口推理，重叠区域取平均.

        Args:
            kp2d: (T, 24, 3)
            window_size: 窗口大小

        Returns:
            (T, 24, 3) 拼接的 3D 预测
        """
        T = kp2d.shape[0]
        stride = window_size // 2  # 50% 重叠
        if stride < 1:
            stride = 1

        # 构建窗口批次
        windows = []
        starts = []
        for start in range(0, max(1, T - window_size + 1), stride):
            end = start + window_size
            if end > T:
                # 末尾窗口
                end = T
                start = max(0, T - window_size)
            window = kp2d[start:end]
            if window.shape[0] < window_size:
                window = self._pad_to_window(window, window_size)
            windows.append(window)
            starts.append(start)
            if start + window_size >= T:
                break

        if not windows:
            # T < window_size 的 fallback
            padded = self._pad_to_window(kp2d, window_size)
            return self._infer_window(padded[np.newaxis])[0, :T]

        # 批量推理
        batch = np.stack(windows, axis=0)  # (N, T_win, 24, 3)
        preds = self._infer_window(batch)  # (N, T_win, 24, 3)

        # 重叠区域平均
        accum = np.zeros((T, 24, 3), dtype=np.float32)
        count = np.zeros((T, 1, 1), dtype=np.float32)
        for i, start in enumerate(starts):
            end = min(start + window_size, T)
            actual_len = end - start
            accum[start:end] += preds[i, :actual_len]
            count[start:end] += 1

        count = np.maximum(count, 1)
        return accum / count

    @staticmethod
    def _pad_to_window(
        sequence: np.ndarray, window_size: int
    ) -> np.ndarray:
        """边缘 padding 到 window_size."""
        T = sequence.shape[0]
        if T >= window_size:
            return sequence[:window_size]
        pad_len = window_size - T
        pad_shape = [(0, pad_len)] + [(0, 0)] * (sequence.ndim - 1)
        return np.pad(sequence, pad_shape, mode="edge")

    def lift_from_yolo(
        self,
        yolo_keypoints: np.ndarray,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """从 YOLO26-pose 输出直接推理（含归一化）.

        Args:
            yolo_keypoints: (T, 24, 3) — 像素坐标 (x, y, conf)
            image_size: (width, height) 图像尺寸

        Returns:
            (T, 24, 3) 3D 关键点（根关节中心化，归一化空间）
        """
        kp2d = np.asarray(yolo_keypoints, dtype=np.float32)
        T, J, C = kp2d.shape
        assert C == 3, f"YOLO 输出必须是 3 通道 (x,y,conf), got {C}"

        # 归一化到 [-1, 1]（图像中心原点，短边为单位长度）
        w, h = image_size
        short_side = min(w, h)
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        kp2d_norm = kp2d.copy()
        kp2d_norm[..., :2] = (kp2d[..., :2] - center) / (short_side / 2.0)
        # confidence 保持原值

        return self.lift(kp2d_norm)
