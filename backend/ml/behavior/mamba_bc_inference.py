"""Mamba+BC 运行时推理器（PyTorch + ONNX 双后端）.

Owner: ML 开发
Phase: 4.2

设计:
    - 参考 mamba_inference.py::MambaInferer 设计
    - 输入: (T, 24, 3) 关键点序列
    - 输出: list[BehaviorEpisode]（与 RuleEngine 格式一致）
    - 支持边界检测（BC Head 输出 boundary_probs）
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from backend.ml.behavior.mamba_bc import build_mamba_bc
from backend.ml.behavior.constants import WITHERS, TAIL_START
from backend.ml.behavior.rule_engine import BehaviorEpisode
from backend.ml.behavior.stgcn_bc.labels import IDX_TO_BEHAVIOR, NUM_BEHAVIORS

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZE = 30
DEFAULT_STRIDE = 15
DEFAULT_BOUNDARY_THRESHOLD = 0.5
DEFAULT_CONF_THRESHOLD = 0.3
DEFAULT_MIN_EPISODE_LEN = 3


class MambaBCInferer:
    """Mamba+BC 推理器（PyTorch + ONNX 双后端）.

    用法:
        # PyTorch 后端
        inferer = MambaBCInferer(checkpoint_path="runs/mamba_bc_synthetic/best.pt")
        episodes = inferer.predict(kpts_seq)

        # ONNX 后端
        inferer = MambaBCInferer(onnx_path="data/models/mamba_bc/mamba_bc_dog24.onnx")
        episodes = inferer.predict(kpts_seq)
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        onnx_path: Optional[Union[str, Path]] = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
        boundary_threshold: float = DEFAULT_BOUNDARY_THRESHOLD,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        min_episode_len: int = DEFAULT_MIN_EPISODE_LEN,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
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
            logger.info(f"[MambaBCInferer] ONNX 后端: {onnx_path}")
        elif checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_mamba_bc(num_classes=NUM_BEHAVIORS)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self._torch_model = model.to(device)
            self._ort_session = None
            logger.info(f"[MambaBCInferer] PyTorch 后端: {checkpoint_path}")
        else:
            raise ValueError("必须提供 checkpoint_path 或 onnx_path")

    def predict(self, keypoints_sequence: np.ndarray, fps: float = 30.0) -> List[BehaviorEpisode]:
        """行为识别（含边界检测）."""
        kpt = np.asarray(keypoints_sequence, dtype=np.float32)
        if kpt.ndim != 3:
            raise ValueError(f"输入必须是 (T, 24, C), got {kpt.shape}")
        T, V, C = kpt.shape
        if V != 24:
            raise ValueError(f"关键点数必须是 24, got {V}")
        if T == 0:
            logger.debug("[MambaBCInferer] 空关键点序列，返回空 episodes")
            return []

        kpt_norm = self._normalize(kpt)

        if T <= self.window_size:
            padded = self._pad_to_window(kpt_norm, self.window_size)
            cls_probs, boundary_probs = self._infer_window(padded[np.newaxis])
            cls_probs = cls_probs[0]
            boundary_probs = boundary_probs[0, :T]
            episodes = self._episodes_from_single_window(cls_probs, boundary_probs, T, fps)
        else:
            cls_probs_seq, boundary_probs_seq = self._sliding_window_infer(kpt_norm)
            episodes = self._episodes_from_sequence(cls_probs_seq, boundary_probs_seq, T, fps)

        logger.info(f"[MambaBCInferer] 识别到 {len(episodes)} 个行为片段 (T={T}, fps={fps})")
        return episodes

    def _infer_window(self, batch_input: np.ndarray):
        if self.backend == "onnx":
            cls_logits, bd_logits = self._ort_session.run(
                None, {"keypoints": batch_input.astype(np.float32)}
            )
            cls_logits = cls_logits - cls_logits.max(axis=-1, keepdims=True)
            exp = np.exp(cls_logits)
            cls_probs = exp / exp.sum(axis=-1, keepdims=True)
            bd_logits = np.clip(bd_logits, -50.0, 50.0)
            bd_probs = 1.0 / (1.0 + np.exp(-bd_logits))
            return cls_probs, bd_probs
        else:
            with torch.no_grad():
                x = torch.from_numpy(batch_input).to(self.device)
                cls_logits, bd_logits = self._torch_model(x)
                cls_probs = torch.softmax(cls_logits, dim=-1).cpu().numpy()
                bd_probs = torch.sigmoid(bd_logits).cpu().numpy()
                return cls_probs, bd_probs

    def _sliding_window_infer(self, kpt: np.ndarray):
        T = kpt.shape[0]
        windows, starts = [], []
        for start in range(0, max(1, T - self.window_size + 1), self.stride):
            end = min(start + self.window_size, T)
            if end - start < self.window_size:
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
            cls_probs, bd_probs = self._infer_window(padded[np.newaxis])
            return cls_probs[0:1], bd_probs[0, :T]

        batch = np.stack(windows, axis=0)
        cls_probs, bd_probs = self._infer_window(batch)

        # 每帧分类概率：窗口级别平均
        cls_accum = np.zeros((T, NUM_BEHAVIORS), dtype=np.float32)
        cls_count = np.zeros((T, 1), dtype=np.float32)
        for i, start in enumerate(starts):
            end = min(start + self.window_size, T)
            cls_accum[start:end] += cls_probs[i]
            cls_count[start:end] += 1
        cls_count = np.maximum(cls_count, 1)
        cls_probs_seq = cls_accum / cls_count

        # 每帧边界概率：重叠区域平均
        bd_accum = np.zeros((T,), dtype=np.float32)
        bd_count = np.zeros((T,), dtype=np.float32)
        for i, start in enumerate(starts):
            end = min(start + self.window_size, T)
            actual_len = end - start
            T_pred = bd_probs.shape[1]
            if T_pred != self.window_size:
                idx = np.linspace(0, T_pred - 1, self.window_size).astype(np.int32)
                bp = bd_probs[i][idx]
            else:
                bp = bd_probs[i]
            bd_accum[start:end] += bp[:actual_len]
            bd_count[start:end] += 1
        bd_count = np.maximum(bd_count, 1)
        bd_probs_seq = bd_accum / bd_count

        return cls_probs_seq, bd_probs_seq

    def _episodes_from_single_window(self, cls_probs, bd_probs, T, fps):
        cls_idx = int(cls_probs.argmax())
        cls_conf = float(cls_probs[cls_idx])
        behavior = IDX_TO_BEHAVIOR[cls_idx]
        if cls_conf < self.conf_threshold:
            return []
        boundary_pred = bd_probs > self.boundary_threshold
        return self._split_by_boundary(behavior, cls_conf, boundary_pred, T, fps)

    def _episodes_from_sequence(self, cls_probs_seq, bd_probs_seq, T, fps):
        cls_idx_seq = cls_probs_seq.argmax(axis=-1)
        cls_conf_seq = cls_probs_seq.max(axis=-1)
        boundary_pred = bd_probs_seq > self.boundary_threshold

        episodes = []
        i = 0
        while i < T:
            if cls_conf_seq[i] < self.conf_threshold:
                i += 1
                continue
            current_cls = int(cls_idx_seq[i])
            current_conf = float(cls_conf_seq[i])
            start = i
            j = i + 1
            while j < T:
                if (int(cls_idx_seq[j]) == current_cls
                        and float(cls_conf_seq[j]) >= self.conf_threshold
                        and not bool(boundary_pred[j])):
                    current_conf = max(current_conf, float(cls_conf_seq[j]))
                    j += 1
                else:
                    break
            end = j - 1
            if end - start + 1 >= self.min_episode_len:
                episodes.append(BehaviorEpisode(
                    behavior=IDX_TO_BEHAVIOR[current_cls],
                    start_frame=start, end_frame=end,
                    confidence=current_conf,
                    metadata={"detector": "mamba_bc", "avg_conf": float(cls_conf_seq[start:end+1].mean())},
                ))
            i = j
        return episodes

    def _split_by_boundary(self, behavior, cls_conf, boundary_pred, T, fps):
        boundaries = []
        for i in range(1, T):
            if bool(boundary_pred[i]) and not bool(boundary_pred[i - 1]):
                boundaries.append(i)
        if not boundaries:
            if T >= self.min_episode_len:
                return [BehaviorEpisode(
                    behavior=behavior, start_frame=0, end_frame=T-1,
                    confidence=cls_conf, metadata={"detector": "mamba_bc", "split": "no_boundary"},
                )]
            return []
        episodes = []
        prev = 0
        for b in boundaries:
            if b - prev >= self.min_episode_len:
                episodes.append(BehaviorEpisode(
                    behavior=behavior, start_frame=prev, end_frame=b-1,
                    confidence=cls_conf, metadata={"detector": "mamba_bc", "split": "boundary"},
                ))
            prev = b
        if T - prev >= self.min_episode_len:
            episodes.append(BehaviorEpisode(
                behavior=behavior, start_frame=prev, end_frame=T-1,
                confidence=cls_conf, metadata={"detector": "mamba_bc", "split": "boundary_tail"},
            ))
        return episodes

    @staticmethod
    def _normalize(kpt):
        center = kpt[:, WITHERS:WITHERS + 1, :].mean(axis=0, keepdims=True)
        kpt = kpt - center
        bone_ref = kpt[:, WITHERS, :2] - kpt[:, TAIL_START, :2]
        bone_len = float(np.linalg.norm(bone_ref, axis=-1).mean())
        if bone_len < 1e-6:
            bone_len = 1.0
        kpt[..., :2] = kpt[..., :2] / bone_len
        return kpt.astype(np.float32)

    @staticmethod
    def _pad_to_window(sequence, window_size):
        T = sequence.shape[0]
        if T >= window_size:
            return sequence[:window_size]
        pad_len = window_size - T
        pad_shape = [(0, pad_len)] + [(0, 0)] * (sequence.ndim - 1)
        return np.pad(sequence, pad_shape, mode="edge")


__all__ = ["MambaBCInferer"]