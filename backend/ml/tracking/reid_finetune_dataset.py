"""犬只 ReID 微调数据接口（Phase 3.2c）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.2c
依据: dev-docs/stages/phase-3.md §3.2c + dev-docs/research/RESEARCH_MULTI_DOG_TRACKING.md §4.5

设计目标:
    1. 定义犬只 ReID 微调数据格式（与 BoxMOT 训练流程兼容）
    2. 收集 (crop, track_id) 对，组织为 BoxMOT ReID 训练目录结构
    3. 不实际触发微调，仅提供数据准备接口
    4. 当 IDSwitchMonitor.should_trigger_reid_finetune=True 且用户决策通过后调用

BoxMOT ReID 训练数据格式（market1501 兼容）:
    dataset_root/
        train/
            <identity_id>/
                img_0001.jpg
                img_0002.jpg
                ...
        query/
            <identity_id>/
                img_0001.jpg
        gallery/
            <identity_id>/
                img_0001.jpg

约束:
    - 每只犬至少需要 20 张裁剪（推荐 50+）
    - 不同视频/角度/光照下的裁剪更佳
    - 同一身份的裁剪必须来自同一物理犬只

不引入兜底层:
    - 不自动调用 BoxMOT train 命令
    - 不预设微调触发条件
    - 仅提供数据准备和命令生成接口
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.ml.tracking.types import DogTrack, MultiDogTrackingResult

logger = logging.getLogger(__name__)


# 微调数据格式约束
MIN_CROPS_PER_IDENTITY = 20  # BoxMOT 训练每身份最少裁剪数
RECOMMENDED_CROPS_PER_IDENTITY = 50  # 推荐裁剪数
MIN_IDENTITIES = 2  # 至少需要 2 只犬（否则无 ReID 必要）
TRAIN_SPLIT_RATIO = 0.7  # train / (query + gallery)
QUERY_GALLERY_SPLIT_RATIO = 0.5  # query / gallery


@dataclass
class CanineReIDSample:
    """单条犬只 ReID 样本."""

    crop: np.ndarray  # (h, w, 3) BGR
    identity_id: int  # 全局身份 ID（同一物理犬只相同）
    source_video: str  # 源视频路径
    frame_idx: int  # 源帧索引
    track_id: int  # 原始 track_id（可能与 identity_id 不同）
    bbox: np.ndarray  # 原始 bbox [x1, y1, x2, y2]
    conf: float = 0.0  # 检测置信度

    def save(self, output_path: Union[str, Path]) -> Path:
        """保存裁剪到磁盘.

        Args:
            output_path: 输出路径

        Returns:
            Path — 实际保存路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), self.crop)
        return output_path


@dataclass
class CanineReIDDataset:
    """犬只 ReID 微调数据集.

    用法:
        # 从追踪结果收集裁剪
        dataset = CanineReIDDataset()
        dataset.collect_from_tracking_result(
            result=result,
            video_frames=frames,
            video_path="video.mp4",
            identity_mapping={0: 100, 1: 101},  # track_id -> 全局身份
        )

        # 检查数据完整性
        report = dataset.integrity_report()
        if not report["is_complete"]:
            print(f"数据不足: {report['issues']}")

        # 导出为 BoxMOT 训练格式
        dataset.export_boxmot_format(output_dir="data/canine_reid")
    """

    samples: List[CanineReIDSample] = field(default_factory=list)
    identity_to_track_ids: Dict[int, List[int]] = field(default_factory=dict)

    def collect_from_tracking_result(
        self,
        result: MultiDogTrackingResult,
        video_frames: List[np.ndarray],
        video_path: str,
        identity_mapping: Optional[Dict[int, int]] = None,
        frame_step: int = 5,
        min_conf: float = 0.3,
    ) -> int:
        """从追踪结果收集犬只裁剪.

        Args:
            result: 多犬追踪结果
            video_frames: 视频帧列表
            video_path: 源视频路径（用于元数据）
            identity_mapping: track_id -> 全局身份 ID 映射；
                              若 None，则 track_id 直接作为 identity_id
            frame_step: 采样步长（每 N 帧取 1 张裁剪，避免过密）
            min_conf: 最小检测置信度阈值

        Returns:
            int — 收集到的样本数
        """
        if not video_frames:
            logger.warning("[CanineReIDDataset] video_frames 为空")
            return 0

        identity_map = identity_mapping or {}
        collected = 0

        for track_id in result.track_ids:
            track = result.get_track(track_id)
            identity_id = identity_map.get(track_id, track_id)

            # 记录身份 -> track_id 映射
            if identity_id not in self.identity_to_track_ids:
                self.identity_to_track_ids[identity_id] = []
            if track_id not in self.identity_to_track_ids[identity_id]:
                self.identity_to_track_ids[identity_id].append(track_id)

            # 按 frame_step 采样
            for frame_idx in track.frame_indices[::frame_step]:
                if frame_idx < 0 or frame_idx >= len(video_frames):
                    continue
                # 找到该帧对应的 DogTrackFrame
                frame_entry = None
                for f in track.frames:
                    if f.frame_idx == frame_idx:
                        frame_entry = f
                        break
                if frame_entry is None:
                    continue
                if frame_entry.conf < min_conf:
                    continue

                # 裁剪
                img = video_frames[frame_idx]
                x1, y1, x2, y2 = frame_entry.bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue  # 太小，跳过

                crop = img[y1:y2, x1:x2].copy()
                self.samples.append(
                    CanineReIDSample(
                        crop=crop,
                        identity_id=identity_id,
                        source_video=video_path,
                        frame_idx=frame_idx,
                        track_id=track_id,
                        bbox=frame_entry.bbox.copy(),
                        conf=frame_entry.conf,
                    )
                )
                collected += 1

        logger.info(
            f"[CanineReIDDataset] 收集 {collected} 个样本 "
            f"({len(self.identity_to_track_ids)} 身份) from {video_path}"
        )
        return collected

    def integrity_report(self) -> Dict:
        """数据完整性检查.

        Returns:
            Dict — {
                "total_samples": 总样本数,
                "num_identities": 身份数,
                "per_identity": {id: count},
                "is_complete": 是否满足微调条件,
                "issues": [问题列表],
            }
        """
        per_identity: Dict[int, int] = {}
        for s in self.samples:
            per_identity[s.identity_id] = per_identity.get(s.identity_id, 0) + 1

        issues: List[str] = []
        if len(per_identity) < MIN_IDENTITIES:
            issues.append(
                f"身份数不足: {len(per_identity)} < {MIN_IDENTITIES}"
            )
        for iid, count in per_identity.items():
            if count < MIN_CROPS_PER_IDENTITY:
                issues.append(
                    f"身份 {iid} 样本不足: {count} < {MIN_CROPS_PER_IDENTITY}"
                )

        return {
            "total_samples": len(self.samples),
            "num_identities": len(per_identity),
            "per_identity": per_identity,
            "is_complete": len(issues) == 0,
            "issues": issues,
            "min_crops_per_identity": MIN_CROPS_PER_IDENTITY,
            "recommended_crops_per_identity": RECOMMENDED_CROPS_PER_IDENTITY,
            "min_identities": MIN_IDENTITIES,
        }

    def export_boxmot_format(
        self,
        output_dir: Union[str, Path],
        train_ratio: float = TRAIN_SPLIT_RATIO,
        random_seed: int = 42,
    ) -> Dict[str, List[Path]]:
        """导出为 BoxMOT market1501 兼容格式.

        目录结构:
            output_dir/
                train/
                    <identity_id>/
                        <video_basename>_frame<fidx>_track<tid>.jpg
                query/
                    <identity_id>/
                        *.jpg
                gallery/
                    <identity_id>/
                        *.jpg

        Args:
            output_dir: 输出目录
            train_ratio: train 集比例（剩余均分给 query/gallery）
            random_seed: 随机种子

        Returns:
            Dict[str, List[Path]] — {"train": [...], "query": [...], "gallery": [...]}
        """
        output_dir = Path(output_dir)
        # 清理已存在目录
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(random_seed)
        exported: Dict[str, List[Path]] = {"train": [], "query": [], "gallery": []}

        # 按身份分组
        per_identity: Dict[int, List[CanineReIDSample]] = {}
        for s in self.samples:
            per_identity.setdefault(s.identity_id, []).append(s)

        for identity_id, samples in per_identity.items():
            # 打乱顺序
            indices = list(range(len(samples)))
            rng.shuffle(indices)
            shuffled = [samples[i] for i in indices]

            n_train = max(1, int(len(shuffled) * train_ratio))
            n_rest = len(shuffled) - n_train
            n_query = n_rest // 2
            # n_gallery = n_rest - n_query

            splits = {
                "train": shuffled[:n_train],
                "query": shuffled[n_train:n_train + n_query],
                "gallery": shuffled[n_train + n_query:],
            }

            for split_name, split_samples in splits.items():
                if not split_samples:
                    continue
                identity_dir = output_dir / split_name / str(identity_id)
                identity_dir.mkdir(parents=True, exist_ok=True)
                for s in split_samples:
                    video_basename = Path(s.source_video).stem
                    fname = (
                        identity_dir
                        / f"{video_basename}_frame{s.frame_idx:06d}_track{s.track_id}.jpg"
                    )
                    saved_path = s.save(fname)
                    exported[split_name].append(saved_path)

        logger.info(
            f"[CanineReIDDataset] 导出完成: train={len(exported['train'])}, "
            f"query={len(exported['query'])}, gallery={len(exported['gallery'])} "
            f"to {output_dir}"
        )
        return exported

    def generate_boxmot_train_command(
        self,
        dataset_dir: Union[str, Path],
        model_arch: str = "osnet_x1_0",
        dataset_name: str = "canine_reid",
        epochs: int = 120,
        device: int = 0,
    ) -> str:
        """生成 BoxMOT 训练命令（用户决策后执行）.

        Args:
            dataset_dir: 数据集目录
            model_arch: 模型架构（默认 osnet_x1_0）
            dataset_name: 数据集名
            epochs: 训练轮数
            device: 训练设备

        Returns:
            str — BoxMOT train 命令字符串
        """
        return (
            f"boxmot train "
            f"--model {model_arch} "
            f"--dataset {dataset_name} "
            f"--data-dir {dataset_dir} "
            f"--epochs {epochs} "
            f"--device {device}"
        )

    def summary(self) -> Dict:
        """返回数据集摘要."""
        report = self.integrity_report()
        return {
            "total_samples": report["total_samples"],
            "num_identities": report["num_identities"],
            "per_identity": report["per_identity"],
            "identity_to_track_ids": self.identity_to_track_ids,
            "is_complete": report["is_complete"],
            "issues": report["issues"],
        }


__all__ = [
    "CanineReIDSample",
    "CanineReIDDataset",
    "MIN_CROPS_PER_IDENTITY",
    "RECOMMENDED_CROPS_PER_IDENTITY",
    "MIN_IDENTITIES",
    "TRAIN_SPLIT_RATIO",
    "QUERY_GALLERY_SPLIT_RATIO",
]
