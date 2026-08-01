"""多犬追踪器 — BoxMOT OccluBoost + YOLO26-pose 关键点关联.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.2b
依据: dev-docs/research/RESEARCH_MULTI_DOG_TRACKING.md §4.1 + §4.2

架构:
    1. YOLO26-pose 检测 → (N, [xyxy, conf, cls]) + (N, 24, 3) 关键点
    2. BoxMOT OccluBoost 追踪 → tracks 含 det_ind 字段
    3. 通过 det_ind 关联关键点到 track_id
    4. 输出每犬独立的 bbox + keypoints 时间序列

接口设计（单犬 → 多犬扩展）:
    - track_video(video_path): 整段视频多犬追踪
    - update_frame(dets, kps, img): 单帧增量更新（实时场景用）

注意:
    - 单犬场景自动退化为 1 个 track_id（向后兼容 Phase 1/2）
    - BoxMOT ReID 模型首次使用会自动下载（~50MB，HuggingFace）
    - 犬只 ReID 微调为 3.2c 自研触发项（默认 OSNet/lmbn）
"""
from __future__ import annotations

import enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.ml.behavior.constants import NUM_KEYPOINTS
from backend.ml.tracking.types import (
    DogTrack,
    DogTrackFrame,
    MultiDogTrackingResult,
)


class TrackerBackend(str, enum.Enum):
    """BoxMOT 追踪器后端枚举."""

    OCCLUBOOST = "occluboost"  # 默认，IDF1 最高（MOT17 84.14, SportsMOT 89.36）
    BOTSORT = "botsort"        # 备选，GMC + ReID
    BYTETRACK = "bytetrack"    # 无 ReID 轻量版
    STRONGSORT = "strongsort"  # DeepSORT 增强版


def _parse_yolo_detections(results) -> Tuple[np.ndarray, np.ndarray]:
    """解析 YOLO26-pose 推理结果为 BoxMOT 输入格式.

    Args:
        results: ultralytics YOLO predict 结果对象

    Returns:
        dets: np.ndarray (N, 6) [x1, y1, x2, y2, conf, cls]
        kps: np.ndarray (N, 24, 3) [x, y, conf]
    """
    if results.boxes is None or len(results.boxes.xyxy) == 0:
        # 空检测
        dets = np.zeros((0, 6), dtype=np.float32)
        kps = np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32)
        return dets, kps

    # 检测框 + 置信度 + 类别
    xyxy = results.boxes.xyxy.cpu().numpy()  # (N, 4)
    conf = results.boxes.conf.cpu().numpy()  # (N,)
    cls = results.boxes.cls.cpu().numpy()    # (N,)
    dets = np.concatenate(
        [xyxy, conf[:, None], cls[:, None]], axis=1
    ).astype(np.float32)  # (N, 6)

    # 关键点
    if results.keypoints is not None and len(results.keypoints.data) > 0:
        kps = results.keypoints.data.cpu().numpy()  # (N, 24, 3)
        # 兜底：关键点数不足 24 时 pad
        if kps.shape[1] < NUM_KEYPOINTS:
            pad = np.zeros(
                (kps.shape[0], NUM_KEYPOINTS - kps.shape[1], 3), dtype=np.float32
            )
            kps = np.concatenate([kps, pad], axis=1)
        elif kps.shape[1] > NUM_KEYPOINTS:
            kps = kps[:, :NUM_KEYPOINTS, :]
    else:
        kps = np.zeros((dets.shape[0], NUM_KEYPOINTS, 3), dtype=np.float32)

    return dets.astype(np.float32), kps.astype(np.float32)


def _tracks_to_frames(
    tracks,
    kps: np.ndarray,
    frame_idx: int,
) -> List[DogTrackFrame]:
    """将 BoxMOT TrackResults 转换为 DogTrackFrame 列表.

    通过 tracks.det_ind 关联 YOLO 检测索引，取回对应关键点。

    Args:
        tracks: BoxMOT TrackResults（ndarray 子类，含 id/det_ind/xyxy/conf/cls 列属性）
                或普通 2D ndarray [x1,y1,x2,y2,id,conf,cls,det_ind]
        kps: np.ndarray (N, 24, 3) — 本帧 YOLO 关键点
        frame_idx: 当前帧索引

    Returns:
        List[DogTrackFrame] — 本帧每犬的追踪结果
    """
    if tracks is None or len(tracks) == 0:
        return []

    # 列访问（兼容 TrackResults 属性访问 + 普通 2D ndarray 列切片）
    try:
        # TrackResults: 属性列访问（boxmot 19.0+）
        track_ids = np.asarray(tracks.id).ravel()
        det_inds = np.asarray(tracks.det_ind).ravel()
        confs = np.asarray(tracks.conf).ravel()
        xyxys = np.asarray(tracks.xyxy)
        if xyxys.ndim == 1:
            xyxys = xyxys.reshape(-1, 4)
    except (AttributeError, TypeError, ValueError):
        # 普通 2D ndarray: [x1,y1,x2,y2,id,conf,cls,det_ind]
        tracks_arr = np.asarray(tracks)
        if tracks_arr.ndim == 1:
            tracks_arr = tracks_arr.reshape(1, -1)
        xyxys = tracks_arr[:, :4]
        track_ids = tracks_arr[:, 4]
        confs = tracks_arr[:, 5]
        det_inds = tracks_arr[:, 7] if tracks_arr.shape[1] > 7 else np.full(len(tracks_arr), -1)

    frames: List[DogTrackFrame] = []
    for i in range(len(track_ids)):
        det_ind = int(det_inds[i])
        # 关键点关联
        if 0 <= det_ind < kps.shape[0]:
            keypoints = kps[det_ind].copy()
        else:
            keypoints = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

        frames.append(
            DogTrackFrame(
                frame_idx=frame_idx,
                track_id=int(track_ids[i]),
                bbox=np.asarray(xyxys[i], dtype=np.float32),
                keypoints=keypoints,
                conf=float(confs[i]),
                det_ind=det_ind,
            )
        )
    return frames


class MultiDogTracker:
    """多犬追踪器（BoxMOT OccluBoost + YOLO26-pose）.

    用法:
        # 整段视频追踪
        tracker = MultiDogTracker(pose_model_path="best.pt")
        result = tracker.track_video("video.mp4")
        print(result.summary())

        # 增量帧追踪（实时场景）
        tracker = MultiDogTracker(pose_model_path="best.pt")
        for frame in video_stream:
            dog_frames = tracker.update_frame(frame)
            for df in dog_frames:
                print(f"帧 {df.frame_idx} 犬 #{df.track_id}: bbox={df.bbox}")

    Args:
        pose_model_path: YOLO26-pose 模型路径（.pt/.onnx/.engine）
        tracker_backend: 追踪器后端（默认 OccluBoost）
        reid_model: ReID 模型名（默认 'osnet_x1_0'，犬只微调为 3.2c 触发项）
        device: 推理设备（0=GPU, 'cpu'=CPU）
        imgsz: YOLO 推理图像尺寸
        conf: YOLO 检测置信度阈值
        iou: YOLO NMS IoU 阈值
        new_track_thresh: 新轨迹创建置信度阈值（默认 0.3，低于 YOLO conf 以确保低置信度检测也能建轨）
        min_hits: 轨迹确认所需最小命中数（默认 1，短视频场景友好）
        tracker_kwargs: 额外 OccluBoost 参数（透传）
    """

    def __init__(
        self,
        pose_model_path: Union[str, Path],
        tracker_backend: TrackerBackend = TrackerBackend.OCCLUBOOST,
        reid_model: str = "osnet_x1_0",
        device: Union[int, str] = 0,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        new_track_thresh: float = 0.3,
        min_hits: int = 1,
        tracker_kwargs: Optional[dict] = None,
    ) -> None:
        self.pose_model_path = str(pose_model_path)
        self.tracker_backend = tracker_backend
        self.reid_model = reid_model
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.new_track_thresh = new_track_thresh
        self.min_hits = min_hits
        self.tracker_kwargs = tracker_kwargs or {}

        # 延迟初始化（避免构造时加载模型，便于测试）
        self._pose_model = None
        self._tracker = None

    def _init_pose_model(self):
        """延迟初始化 YOLO26-pose 模型."""
        if self._pose_model is not None:
            return
        from ultralytics import YOLO
        # 显式传 task='pose'（Phase 2.0c bug 修复）
        self._pose_model = YOLO(self.pose_model_path, task="pose")

    def _init_tracker(self):
        """延迟初始化 BoxMOT 追踪器."""
        if self._tracker is not None:
            return

        # 先校验后端（Phase 3.2b 仅支持 OccluBoost）
        if self.tracker_backend != TrackerBackend.OCCLUBOOST:
            raise NotImplementedError(
                f"追踪器后端 {self.tracker_backend} 暂未实现，Phase 3.2b 仅支持 OccluBoost"
            )

        from boxmot.trackers import OccluBoost
        # 合并追踪器参数：默认值 + 用户透传
        # 显式启用 with_reid=True（OccluBoost 继承 BoostTrack 默认 with_reid=False，
        # YAML 配置仅在 boxmot tracking CLI 路径生效，编程式实例化必须显式传入）
        # 用户透传优先：若 tracker_kwargs 显式覆盖 with_reid，则尊重用户设置
        with_reid = self.tracker_kwargs.get("with_reid", True)

        # ReID 模型构造（BoxMOT 19.0 期望 reid_model 是已初始化的 backend 对象，
        # 通过 ReID(...).model 获取；字符串权重路径不能直接用）
        reid_model_obj = None
        if with_reid:
            try:
                from boxmot.reid.core.reid import ReID
                reid_obj = ReID(
                    self.reid_model,
                    device=self.device,
                    half=False,
                )
                reid_model_obj = reid_obj.model
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"[MultiDogTracker] ReID 模型加载失败（{e}），降级为 with_reid=False"
                )
                with_reid = False

        tracker_params = {
            "reid_model": reid_model_obj,
            "with_reid": with_reid,
            "new_track_thresh": self.new_track_thresh,
            "min_hits": self.min_hits,
            **{k: v for k, v in self.tracker_kwargs.items() if k != "with_reid"},
        }
        self._tracker = OccluBoost(**tracker_params)

    def reset(self) -> None:
        """重置追踪器状态（新视频前调用）."""
        if self._tracker is not None:
            self._tracker.reset()

    def update_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> List[DogTrackFrame]:
        """单帧增量追踪.

        Args:
            frame: np.ndarray (H, W, 3) BGR 图像
            frame_idx: 帧索引

        Returns:
            List[DogTrackFrame] — 本帧每犬的追踪结果
        """
        self._init_pose_model()
        self._init_tracker()

        # 1. YOLO26-pose 检测
        results = self._pose_model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            save=False,
        )[0]

        dets, kps = _parse_yolo_detections(results)

        # 2. BoxMOT 追踪
        if dets.shape[0] == 0:
            # 空检测时仍需 update（让追踪器知道这帧无检测）
            tracks = self._tracker.update(dets, img=frame)
        else:
            tracks = self._tracker.update(dets, img=frame)

        # 3. 关键点关联
        return _tracks_to_frames(tracks, kps, frame_idx)

    def track_video(
        self,
        video_path: Union[str, Path],
        save_output: bool = False,
        output_path: Optional[Union[str, Path]] = None,
    ) -> MultiDogTrackingResult:
        """整段视频多犬追踪.

        Args:
            video_path: 视频文件路径
            save_output: 是否保存结果 pkl
            output_path: pkl 输出路径（默认与视频同目录 .tracking.pkl）

        Returns:
            MultiDogTrackingResult
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")

        self._init_pose_model()
        self._init_tracker()
        self.reset()  # 新视频前重置

        # 读取视频元数据
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result = MultiDogTrackingResult(
            total_frames=frame_count,
            fps=float(fps),
            meta={
                "video_path": str(video_path),
                "pose_model_path": self.pose_model_path,
                "tracker_backend": self.tracker_backend.value,
                "reid_model": self.reid_model,
                "width": width,
                "height": height,
                "imgsz": self.imgsz,
                "conf_threshold": self.conf,
                "iou_threshold": self.iou,
            },
        )

        frame_idx = 0
        print(f"[tracking] 开始追踪: {video_path.name} ({frame_count} frames @ {fps:.1f} fps)", flush=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dog_frames = self.update_frame(frame, frame_idx=frame_idx)
            for df in dog_frames:
                if df.track_id not in result.tracks:
                    result.tracks[df.track_id] = DogTrack(track_id=df.track_id)
                result.tracks[df.track_id].frames.append(df)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"[tracking] 进度: {frame_idx}/{frame_count} 帧, "
                      f"已追踪 {result.num_dogs} 犬", flush=True)

        cap.release()
        print(f"[tracking] 完成: {frame_idx} 帧, {result.num_dogs} 犬, "
              f"track_ids={result.track_ids}", flush=True)

        if save_output:
            import pickle
            if output_path is None:
                output_path = video_path.with_suffix(".tracking.pkl")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(result, f)
            print(f"[tracking] 保存: {output_path}", flush=True)

        return result


__all__ = [
    "MultiDogTracker",
    "TrackerBackend",
    "_parse_yolo_detections",
    "_tracks_to_frames",
]
