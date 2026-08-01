"""ReID 犬只身份关联模块（Phase 3.2c）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.2c
依据: dev-docs/research/RESEARCH_MULTI_DOG_TRACKING.md §4.3 + dev-docs/stages/phase-3.md §3.2c

设计目标:
    1. 封装 BoxMOT ReID 运行时（OSNet 默认），提供犬只身份特征提取接口
    2. 支持轨迹级身份聚合（temporal pooling）—— 单帧特征 → 轨迹 embedding
    3. 提供身份相似度计算，用于跨轨迹 ID 关联（轨迹断裂后重连）
    4. 收集犬只 ReID 标注样本（图像裁剪 + track_id），为 3.2c 自研触发后的
       OSNet 犬只微调提供数据接口（不触发实际微调，仅收集）

关键决策（AGENTS.md §5.2 用户逐案决策）:
    - 默认使用 OSNet（MIT, HuggingFace 权重），无需微调
    - 自研触发条件：同品种犬只 ID switch 过高时，用户决策启动 OSNet 微调
    - ID switch 阈值由 IDSwitchMonitor 评估，不在本模块强制触发

依赖:
    - boxmot.reid.core.reid.ReID（统一 ReID 运行时）
    - 不依赖犬只 ReID 微调权重（默认 MSMT17 预训练即可用）
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# 默认 ReID 模型（OSNet x1.0, MSMT17 预训练，512 维 embedding）
DEFAULT_REID_MODEL = "osnet_x1_0_msmt17.pt"
DEFAULT_REID_DEVICE = 0  # GPU 优先
DEFAULT_REID_HALF = False  # FP32 推理（ReID 特征精度敏感）
DEFAULT_TEMPORAL_POOL = "mean"  # 轨迹级聚合策略
DEFAULT_SIM_THRESHOLD = 0.7  # 轨迹身份匹配阈值（余弦相似度）


class ReIDExtractor:
    """犬只 ReID 特征提取器（封装 BoxMOT ReID 运行时）.

    用法:
        # 基本特征提取
        extractor = ReIDExtractor()
        feats = extractor.extract_from_boxes(img, boxes_xyxy)  # (N, 512)

        # 从轨迹裁剪提取特征
        feats = extractor.extract_from_crops([crop1, crop2, ...])  # (N, 512)

        # 轨迹级聚合（temporal pooling）
        track_feat = extractor.aggregate_track(frame_feats)  # (512,)

    Args:
        model_path: ReID 模型路径（默认 osnet_x1_0_msmt17.pt，BoxMOT 自动下载）
        device: 推理设备（0=GPU, 'cpu'=CPU）
        half: 是否使用 FP16（默认 False，ReID 精度敏感）
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_REID_MODEL,
        device: Union[int, str] = DEFAULT_REID_DEVICE,
        half: bool = DEFAULT_REID_HALF,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.half = half
        self._reid = None  # 延迟初始化

    def _init_reid(self):
        """延迟初始化 BoxMOT ReID 运行时."""
        if self._reid is not None:
            return
        from boxmot.reid.core.reid import ReID
        self._reid = ReID(
            self.model_path,
            device=self.device,
            half=self.half,
        )
        logger.info(
            f"[ReIDExtractor] 初始化完成: model={self.model_path}, "
            f"device={self.device}, half={self.half}"
        )

    def extract_from_boxes(
        self,
        img: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> np.ndarray:
        """从图像 + 检测框提取 ReID 特征.

        Args:
            img: np.ndarray (H, W, 3) BGR 图像
            boxes_xyxy: np.ndarray (N, 4) [x1, y1, x2, y2]

        Returns:
            np.ndarray (N, embedding_dim) — L2 归一化特征
            空输入返回 (0, 0)
        """
        self._init_reid()
        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        if boxes.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        # BoxMOT ReID.__call__(inputs, boxes) 返回 L2 归一化特征 (N, dim)
        feats = self._reid(img, boxes=boxes)
        return np.asarray(feats, dtype=np.float32)

    def extract_from_crops(
        self,
        crops: List[np.ndarray],
    ) -> np.ndarray:
        """从图像裁剪列表提取 ReID 特征.

        Args:
            crops: List[np.ndarray]，每个 (h, w, 3) BGR

        Returns:
            np.ndarray (N, embedding_dim) — L2 归一化特征
        """
        self._init_reid()
        if not crops:
            return np.zeros((0, 0), dtype=np.float32)
        # BoxMOT ReID.__call__(inputs) 当 boxes=None 时按裁剪列表处理
        feats = self._reid(crops)
        return np.asarray(feats, dtype=np.float32)

    @staticmethod
    def aggregate_track(
        frame_feats: np.ndarray,
        strategy: str = DEFAULT_TEMPORAL_POOL,
    ) -> np.ndarray:
        """将单帧特征聚合为轨迹级 embedding.

        Args:
            frame_feats: np.ndarray (T, embedding_dim) — 轨迹每帧的特征
            strategy: 聚合策略
                - "mean": 均值池化（默认）
                - "max": 最大池化
                - "median": 中位数

        Returns:
            np.ndarray (embedding_dim,) — L2 归一化的轨迹特征
            空输入返回 zeros(embedding_dim)
        """
        if frame_feats.size == 0 or frame_feats.shape[0] == 0:
            return np.zeros(
                frame_feats.shape[1] if frame_feats.ndim == 2 else 0,
                dtype=np.float32,
            )

        if strategy == "mean":
            feat = frame_feats.mean(axis=0)
        elif strategy == "max":
            feat = frame_feats.max(axis=0)
        elif strategy == "median":
            feat = np.median(frame_feats, axis=0)
        else:
            raise ValueError(f"未知聚合策略: {strategy}")

        # L2 归一化
        norm = np.linalg.norm(feat)
        if norm > 1e-12:
            feat = feat / norm
        return feat.astype(np.float32)


def cosine_similarity(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
) -> np.ndarray:
    """计算两组特征的余弦相似度矩阵.

    Args:
        feats_a: np.ndarray (M, dim) — L2 归一化
        feats_b: np.ndarray (N, dim) — L2 归一化

    Returns:
        np.ndarray (M, N) — 相似度矩阵 [-1, 1]
    """
    if feats_a.size == 0 or feats_b.size == 0:
        return np.zeros((feats_a.shape[0], feats_b.shape[0]), dtype=np.float32)
    a = np.asarray(feats_a, dtype=np.float32)
    b = np.asarray(feats_b, dtype=np.float32)
    # 归一化（防御性：即使输入未归一化也能正确计算）
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


class DogIdentityGallery:
    """犬只身份 gallery — 维护已知犬只的轨迹 embedding 并支持身份匹配.

    用途:
        1. 跨视频 ID 关联：同一犬只在不同视频中被分配不同 track_id 时，
           通过 ReID embedding 匹配到同一全局身份
        2. ReID 微调数据收集：收集 (crop, track_id) 对，为 OSNet 犬只微调
           提供数据（3.2c 自研触发后使用）

    用法:
        gallery = DogIdentityGallery()
        gallery.register(track_id=0, embedding=track_feat_0)
        gallery.register(track_id=1, embedding=track_feat_1)

        # 新轨迹身份匹配
        match_id, sim = gallery.match(new_feat, threshold=0.7)
        if match_id is not None:
            print(f"匹配到已知犬 #{match_id}, 相似度={sim:.3f}")
        else:
            new_id = gallery.next_id()
            gallery.register(new_id, new_feat)
    """

    def __init__(
        self,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    ) -> None:
        self.sim_threshold = float(sim_threshold)
        self._gallery: Dict[int, np.ndarray] = {}  # track_id -> embedding
        self._crops: Dict[int, List[np.ndarray]] = {}  # track_id -> 裁剪列表（微调用）
        self._next_id = 0

    def register(
        self,
        track_id: int,
        embedding: np.ndarray,
        crops: Optional[List[np.ndarray]] = None,
    ) -> None:
        """注册或更新犬只身份.

        Args:
            track_id: 轨迹 ID
            embedding: 轨迹级 ReID embedding（L2 归一化）
            crops: 可选裁剪列表，用于后续 ReID 微调数据收集
        """
        self._gallery[track_id] = np.asarray(embedding, dtype=np.float32)
        if crops is not None:
            if track_id not in self._crops:
                self._crops[track_id] = []
            self._crops[track_id].extend(crops)

    def match(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
        exclude_ids: Optional[List[int]] = None,
    ) -> Tuple[Optional[int], float]:
        """为新 embedding 匹配 gallery 中的已知身份.

        Args:
            embedding: 待匹配的轨迹 embedding (dim,)
            threshold: 匹配阈值（默认使用 self.sim_threshold）
            exclude_ids: 排除的 track_id 列表（避免自匹配）

        Returns:
            (match_id, similarity) — 无匹配时 match_id=None, similarity=最高相似度
        """
        if not self._gallery:
            return None, 0.0

        thresh = self.sim_threshold if threshold is None else float(threshold)
        exclude = set(exclude_ids or [])

        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        best_id: Optional[int] = None
        best_sim = -1.0

        for tid, ref_emb in self._gallery.items():
            if tid in exclude:
                continue
            ref = ref_emb.reshape(1, -1)
            sim = float(cosine_similarity(query, ref)[0, 0])
            if sim > best_sim:
                best_sim = sim
                best_id = tid

        if best_sim >= thresh:
            return best_id, best_sim
        return None, best_sim

    def next_id(self) -> int:
        """分配下一个全局 track_id."""
        new_id = self._next_id
        self._next_id += 1
        return new_id

    def export_crops(
        self,
        output_dir: Union[str, Path],
        track_id: Optional[int] = None,
    ) -> List[Path]:
        """导出犬只裁剪图像（为 ReID 微调数据收集）.

        目录结构:
            output_dir/
                track_<id>/
                    00000.jpg
                    00001.jpg
                    ...

        Args:
            output_dir: 输出目录
            track_id: 指定导出的 track_id（None=全部）

        Returns:
            List[Path] — 导出的文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported: List[Path] = []
        target_ids = [track_id] if track_id is not None else list(self._crops.keys())

        for tid in target_ids:
            if tid not in self._crops:
                continue
            track_dir = output_dir / f"track_{tid}"
            track_dir.mkdir(parents=True, exist_ok=True)
            for i, crop in enumerate(self._crops[tid]):
                fname = track_dir / f"{i:05d}.jpg"
                cv2.imwrite(str(fname), crop)
                exported.append(fname)

        logger.info(
            f"[DogIdentityGallery] 导出 {len(exported)} 张裁剪到 {output_dir} "
            f"(track_ids={target_ids})"
        )
        return exported

    @property
    def num_identities(self) -> int:
        """gallery 中已知身份数量."""
        return len(self._gallery)

    @property
    def track_ids(self) -> List[int]:
        """已知身份的 track_id 列表."""
        return list(self._gallery.keys())

    def summary(self) -> Dict:
        """返回 gallery 摘要."""
        return {
            "num_identities": self.num_identities,
            "track_ids": self.track_ids,
            "total_crops": sum(len(v) for v in self._crops.values()),
            "sim_threshold": self.sim_threshold,
        }


__all__ = [
    "ReIDExtractor",
    "DogIdentityGallery",
    "cosine_similarity",
    "DEFAULT_REID_MODEL",
    "DEFAULT_REID_DEVICE",
    "DEFAULT_REID_HALF",
    "DEFAULT_TEMPORAL_POOL",
    "DEFAULT_SIM_THRESHOLD",
]
