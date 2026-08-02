"""ST-GCN+BC 运行时推理器（PyTorch + ONNX 双后端）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1e

设计:
    - 参考 motionbert/inference.py::MotionBERTLifter 双后端范本
    - 输入: (T, 24, 3) 关键点序列（2D 像素或 3D 世界坐标）
    - 输出: list[BehaviorEpisode]（与 RuleEngine 输出格式一致，便于双轨替换）
    - 滑动窗口推理（处理任意长度视频，50% 重叠平均）
    - 边界检测 → episode 起止切分

集成:
    - 上游: backend.ml.pose.inference.PoseInferenceEngine 输出 2D 关键点
            或 backend.ml.pose.motionbert.inference.MotionBERTLifter 输出 3D 关键点
    - 下游: backend.ml.behavior.router.BehaviorRecognizer 双轨路由
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from backend.ml.behavior.constants import WITHERS, TAIL_START
from backend.ml.behavior.rule_engine import BehaviorEpisode
from backend.ml.behavior.stgcn_bc.labels import IDX_TO_BEHAVIOR, NUM_BEHAVIORS
from backend.ml.behavior.stgcn_bc.model import STGCNBC, build_stgcn_bc

logger = logging.getLogger(__name__)


# 推理默认参数
DEFAULT_WINDOW_SIZE = 30          # 训练时 T=30
DEFAULT_STRIDE = 15               # 50% 重叠
DEFAULT_BOUNDARY_THRESHOLD = 0.5  # 边界检测阈值
DEFAULT_CONF_THRESHOLD = 0.3      # 行为置信度过滤阈值
DEFAULT_MIN_EPISODE_LEN = 3       # 最小 episode 帧数


class STGCNBCInferer:
    """ST-GCN+BC 推理器（PyTorch + ONNX 双后端）.

    支持两种后端:
        - PyTorch（用于训练后立即验证）
        - ONNX Runtime（用于生产部署）

    用法:
        # ONNX 后端（部署推荐）
        inferer = STGCNBCInferer(onnx_path="data/models/stgcn_bc/stgcn_bc_dog24.onnx")
        episodes = inferer.predict(kpts_seq)

        # PyTorch 后端
        inferer = STGCNBCInferer(checkpoint_path="runs/stgcn_bc_synthetic/best.pt")
        episodes = inferer.predict(kpts_seq)
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        onnx_path: Optional[Union[str, Path]] = None,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 10,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
        boundary_threshold: float = DEFAULT_BOUNDARY_THRESHOLD,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        min_episode_len: int = DEFAULT_MIN_EPISODE_LEN,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """初始化推理器.

        Args:
            checkpoint_path: PyTorch checkpoint 路径（与 onnx_path 二选一）
            onnx_path: ONNX 模型路径（优先使用，部署推荐）
            in_channels: 输入通道数（需与训练一致）
            base_channels: STGCN 基础通道数（需与训练一致）
            num_stages: STGCN block 数量（需与训练一致）
            window_size: 滑动窗口大小
            stride: 滑动步长（默认 window_size // 2）
            boundary_threshold: 边界检测阈值
            conf_threshold: 行为置信度过滤阈值
            min_episode_len: 最小 episode 帧数
            device: PyTorch 推理设备
        """
        self.window_size = window_size
        self.stride = stride if stride > 0 else max(1, window_size // 2)
        self.boundary_threshold = boundary_threshold
        self.conf_threshold = conf_threshold
        self.min_episode_len = min_episode_len
        self.device = device
        self.backend = "onnx" if onnx_path else "torch"

        if onnx_path is not None:
            import onnxruntime as ort
            self._ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"],
            )
            self._torch_model = None
            logger.info(f"[STGCNBCInferer] ONNX 后端: {onnx_path}")
        elif checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_stgcn_bc(
                in_channels=in_channels,
                base_channels=base_channels,
                num_stages=num_stages,
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            self._torch_model = model.to(device)
            self._ort_session = None
            logger.info(f"[STGCNBCInferer] PyTorch 后端: {checkpoint_path}")
        else:
            raise ValueError("必须提供 checkpoint_path 或 onnx_path")

    def predict(
        self,
        keypoints_sequence: np.ndarray,
        fps: float = 30.0,
    ) -> List[BehaviorEpisode]:
        """行为识别.

        Args:
            keypoints_sequence: (T, 24, 3) 关键点序列
                - 2D 模式: (x, y, conf) 像素坐标
                - 3D 模式: (x, y, z) 世界坐标
            fps: 视频帧率（用于 episode 时间戳，不影响推理）

        Returns:
            list[BehaviorEpisode] — 识别到的行为片段
        """
        kpt = np.asarray(keypoints_sequence, dtype=np.float32)
        if kpt.ndim != 3:
            raise ValueError(f"输入必须是 (T, 24, C), got {kpt.shape}")
        T, V, C = kpt.shape
        if V != 24:
            raise ValueError(f"关键点数必须是 24, got {V}")

        # 空输入早返回（避免 _normalize 对空切片求均值产生 NaN）
        if T == 0:
            logger.debug("[STGCNBCInferer] 空关键点序列，返回空 episodes")
            return []

        # 归一化（withers 中心 + 体长尺度，与训练一致）
        kpt_norm = self._normalize(kpt)

        # 滑动窗口推理
        if T <= self.window_size:
            padded = self._pad_to_window(kpt_norm, self.window_size)
            cls_probs, boundary_probs = self._infer_window(padded[np.newaxis])
            cls_probs = cls_probs[0]                # (22,)
            boundary_probs = boundary_probs[0, :T]  # (T,)
            # 单窗口：整个序列一个分类结果
            episodes = self._episodes_from_single_window(
                cls_probs, boundary_probs, T, fps
            )
        else:
            cls_probs_seq, boundary_probs_seq = self._sliding_window_infer(kpt_norm)
            episodes = self._episodes_from_sequence(
                cls_probs_seq, boundary_probs_seq, T, fps
            )

        logger.info(
            f"[STGCNBCInferer] 识别到 {len(episodes)} 个行为片段 "
            f"(T={T}, fps={fps}, backend={self.backend})"
        )
        return episodes

    # ====================================================================
    # 推理后端
    # ====================================================================

    def _infer_window(
        self, batch_input: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """单批次推理.

        Args:
            batch_input: (B, T, 24, 3)

        Returns:
            cls_probs: (B, 22) softmax 概率
            boundary_probs: (B, T') sigmoid 概率
        """
        if self.backend == "onnx":
            cls_logits, boundary_logits = self._ort_session.run(
                None, {"keypoints": batch_input.astype(np.float32)}
            )
            # softmax（数值稳定：减去最大值）
            cls_logits = cls_logits - cls_logits.max(axis=-1, keepdims=True)
            exp = np.exp(cls_logits)
            cls_probs = exp / exp.sum(axis=-1, keepdims=True)
            # sigmoid（数值稳定：clip 避免 overflow）
            boundary_logits = np.clip(boundary_logits, -50.0, 50.0)
            boundary_probs = 1.0 / (1.0 + np.exp(-boundary_logits))
            return cls_probs, boundary_probs
        else:
            with torch.no_grad():
                x = torch.from_numpy(batch_input).to(self.device)
                cls_logits, boundary_logits = self._torch_model(x)
                cls_probs = torch.softmax(cls_logits, dim=-1).cpu().numpy()
                boundary_probs = torch.sigmoid(boundary_logits).cpu().numpy()
                return cls_probs, boundary_probs

    def _sliding_window_infer(
        self, kpt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """滑动窗口推理.

        Args:
            kpt: (T, 24, 3) 归一化关键点

        Returns:
            cls_probs_seq: (T,) 每帧的分类概率（窗口级别重复，取均值）
            boundary_probs_seq: (T,) 每帧边界概率（重叠区域平均）
        """
        T = kpt.shape[0]
        windows = []
        starts = []
        for start in range(0, max(1, T - self.window_size + 1), self.stride):
            end = start + self.window_size
            if end > T:
                end = T
                start = max(0, T - self.window_size)
            window = kpt[start:end]
            if window.shape[0] < self.window_size:
                window = self._pad_to_window(window, self.window_size)
            windows.append(window)
            starts.append(start)
            if start + self.window_size >= T:
                break

        if not windows:
            padded = self._pad_to_window(kpt, self.window_size)
            cls_probs, boundary_probs = self._infer_window(padded[np.newaxis])
            return cls_probs[0:1], boundary_probs[0, :T]

        batch = np.stack(windows, axis=0)  # (N, T_win, 24, 3)
        cls_probs, boundary_probs = self._infer_window(batch)  # (N, 22), (N, T')

        # 每帧分类概率：窗口级别平均
        cls_accum = np.zeros((T, NUM_BEHAVIORS), dtype=np.float32)
        cls_count = np.zeros((T, 1), dtype=np.float32)
        for i, start in enumerate(starts):
            end = min(start + self.window_size, T)
            # cls_probs 是窗口级，每帧共享同一概率
            cls_accum[start:end] += cls_probs[i]
            cls_count[start:end] += 1
        cls_count = np.maximum(cls_count, 1)
        cls_probs_seq = cls_accum / cls_count  # (T, 22)

        # 每帧边界概率：需要上采样 boundary_probs 到 window_size
        boundary_accum = np.zeros((T,), dtype=np.float32)
        boundary_count = np.zeros((T,), dtype=np.float32)
        for i, start in enumerate(starts):
            end = min(start + self.window_size, T)
            actual_len = end - start
            T_pred = boundary_probs.shape[1]
            # 上采样 boundary (T',) → (window_size,)
            if T_pred != self.window_size:
                idx = np.linspace(0, T_pred - 1, self.window_size).astype(np.int32)
                bp_upsampled = boundary_probs[i][idx]
            else:
                bp_upsampled = boundary_probs[i]
            boundary_accum[start:end] += bp_upsampled[:actual_len]
            boundary_count[start:end] += 1
        boundary_count = np.maximum(boundary_count, 1)
        boundary_probs_seq = boundary_accum / boundary_count  # (T,)

        return cls_probs_seq, boundary_probs_seq

    # ====================================================================
    # Episode 切分
    # ====================================================================

    def _episodes_from_single_window(
        self,
        cls_probs: np.ndarray,      # (22,)
        boundary_probs: np.ndarray, # (T,)
        T: int,
        fps: float,
    ) -> List[BehaviorEpisode]:
        """单窗口 episode 切分.

        策略: 整个窗口一个分类，边界检测切分 episode
        """
        cls_idx = int(cls_probs.argmax())
        cls_conf = float(cls_probs[cls_idx])
        behavior = IDX_TO_BEHAVIOR[cls_idx]

        if cls_conf < self.conf_threshold:
            return []

        # 边界检测切分
        boundary_pred = boundary_probs > self.boundary_threshold
        episodes = self._split_by_boundary(
            behavior, cls_conf, boundary_pred, T, fps
        )
        return episodes

    def _episodes_from_sequence(
        self,
        cls_probs_seq: np.ndarray,      # (T, 22)
        boundary_probs_seq: np.ndarray, # (T,)
        T: int,
        fps: float,
    ) -> List[BehaviorEpisode]:
        """序列 episode 切分.

        策略: 每帧分类，边界检测切分 episode 起止
        """
        cls_idx_seq = cls_probs_seq.argmax(axis=-1)  # (T,)
        cls_conf_seq = cls_probs_seq.max(axis=-1)    # (T,)
        boundary_pred = boundary_probs_seq > self.boundary_threshold

        episodes: List[BehaviorEpisode] = []
        i = 0
        while i < T:
            if cls_conf_seq[i] < self.conf_threshold:
                i += 1
                continue

            # 同类连续帧合并为一个 episode
            current_cls = int(cls_idx_seq[i])
            current_conf = float(cls_conf_seq[i])
            start = i
            j = i + 1
            while j < T:
                # 同类 + 置信度满足 + 非边界
                if (
                    int(cls_idx_seq[j]) == current_cls
                    and float(cls_conf_seq[j]) >= self.conf_threshold
                    and not bool(boundary_pred[j])
                ):
                    current_conf = max(current_conf, float(cls_conf_seq[j]))
                    j += 1
                else:
                    break

            end = j - 1
            if end - start + 1 >= self.min_episode_len:
                episodes.append(BehaviorEpisode(
                    behavior=IDX_TO_BEHAVIOR[current_cls],
                    start_frame=start,
                    end_frame=end,
                    confidence=current_conf,
                    metadata={
                        "detector": "stgcn_bc",
                        "avg_conf": float(cls_conf_seq[start:end + 1].mean()),
                    },
                ))
            i = j

        return episodes

    def _split_by_boundary(
        self,
        behavior: str,
        cls_conf: float,
        boundary_pred: np.ndarray,
        T: int,
        fps: float,
    ) -> List[BehaviorEpisode]:
        """边界检测切分 episode（单窗口场景）."""
        # 找边界上升沿
        boundaries = []
        for i in range(1, T):
            if bool(boundary_pred[i]) and not bool(boundary_pred[i - 1]):
                boundaries.append(i)

        # 无边界：整个窗口一个 episode
        if not boundaries:
            if T >= self.min_episode_len:
                return [BehaviorEpisode(
                    behavior=behavior,
                    start_frame=0,
                    end_frame=T - 1,
                    confidence=cls_conf,
                    metadata={"detector": "stgcn_bc", "split": "no_boundary"},
                )]
            return []

        # 多边界切分
        episodes: List[BehaviorEpisode] = []
        prev = 0
        for b in boundaries:
            if b - prev >= self.min_episode_len:
                episodes.append(BehaviorEpisode(
                    behavior=behavior,
                    start_frame=prev,
                    end_frame=b - 1,
                    confidence=cls_conf,
                    metadata={"detector": "stgcn_bc", "split": "boundary"},
                ))
            prev = b
        # 末尾段
        if T - prev >= self.min_episode_len:
            episodes.append(BehaviorEpisode(
                behavior=behavior,
                start_frame=prev,
                end_frame=T - 1,
                confidence=cls_conf,
                metadata={"detector": "stgcn_bc", "split": "boundary_tail"},
            ))
        return episodes

    # ====================================================================
    # 工具方法
    # ====================================================================

    @staticmethod
    def _normalize(kpt: np.ndarray) -> np.ndarray:
        """按 withers 中心 + 体长尺度归一化（与训练 dataset.py 一致）."""
        center = kpt[:, WITHERS:WITHERS + 1, :].mean(axis=0, keepdims=True)  # (1, 1, 3)
        kpt = kpt - center
        bone_ref = kpt[:, WITHERS, :2] - kpt[:, TAIL_START, :2]  # (T, 2)
        bone_len = float(np.linalg.norm(bone_ref, axis=-1).mean())
        if bone_len < 1e-6:
            bone_len = 1.0
        kpt[..., :2] = kpt[..., :2] / bone_len
        return kpt.astype(np.float32)

    @staticmethod
    def _pad_to_window(sequence: np.ndarray, window_size: int) -> np.ndarray:
        """边缘 padding 到 window_size（与 MotionBERTLifter 一致）."""
        T = sequence.shape[0]
        if T >= window_size:
            return sequence[:window_size]
        pad_len = window_size - T
        pad_shape = [(0, pad_len)] + [(0, 0)] * (sequence.ndim - 1)
        return np.pad(sequence, pad_shape, mode="edge")


__all__ = [
    "STGCNBCInferer",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_STRIDE",
    "DEFAULT_BOUNDARY_THRESHOLD",
    "DEFAULT_CONF_THRESHOLD",
]
