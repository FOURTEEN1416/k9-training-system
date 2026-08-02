"""ST-GCN+BC 行为识别模块（Phase 3.1）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1
依据: dev-docs/stages/phase-3.md §3.1 + dev-docs/research/RESEARCH_STGCN_BC.md

子模块:
    - k9_graph: K9Graph 24 节点犬类骨架拓扑
    - labels: 22 类行为标签映射（P0 8 + P1 8 + P2 6）
    - data_adapter: YOLO26-pose (T, 24, 3) → pyskl 骨架格式
    - stgcn: ST-GCN 主干（PyTorch 原生自研，3.1c）
    - bc_head: 自研 BC 头（边界分类联合优化，3.1c）
    - loss: 联合损失 L = L_cls + 0.3 · L_boundary（3.1c）
    - model: STGCNBC 整体模型（backbone + head + loss，3.1c）
    - dataset: 训练数据集 + 合成数据生成（3.1d）
    - trainer: 训练器 + 检查点管理（3.1d）
    - export_onnx: ONNX 导出（3.1e）
    - inference: 运行时推理器 PyTorch + ONNX 双后端（3.1e）

注意:
    本模块为项目自有实现，**完全不依赖 pyskl 安装**即可运行。
    pyskl 仓库已 clone 到 `external/pyskl/` 仅作参考蓝本（其 mmcv-full
    依赖与项目环境冲突，故采用 PyTorch 原生自研路线）。
"""
from backend.ml.behavior.stgcn_bc.k9_graph import K9Graph
from backend.ml.behavior.stgcn_bc.labels import (
    BEHAVIOR_TO_IDX,
    IDX_TO_BEHAVIOR,
    NUM_BEHAVIORS,
    P2_BEHAVIORS,
)
from backend.ml.behavior.stgcn_bc.data_adapter import (
    keypoints_to_pyskl,
    pyskl_to_keypoints,
    compute_bone_flow,
    compute_motion_flow,
    normalize_keypoints,
    build_pyskl_annotation,
    save_pyskl_pickle,
)
from backend.ml.behavior.stgcn_bc.stgcn import (
    build_spatial_adjacency,
    UnitGCN,
    UnitTCN,
    MSTCN,
    STGCNBlock,
    STGCN,
)
from backend.ml.behavior.stgcn_bc.bc_head import BCHead, generate_boundary_labels
from backend.ml.behavior.stgcn_bc.loss import STGCNBCLoss
from backend.ml.behavior.stgcn_bc.model import STGCNBC, build_stgcn_bc
from backend.ml.behavior.stgcn_bc.dataset import (
    STGCNBCDataset,
    make_synthetic_dataset,
    save_synthetic_dataset,
    load_pyskl_pickle,
    collate_fn,
)
from backend.ml.behavior.stgcn_bc.trainer import TrainConfig, STGCNBCTrainer
from backend.ml.behavior.stgcn_bc.inference import STGCNBCInferer
from backend.ml.behavior.stgcn_bc.export_onnx import export_onnx

__all__ = [
    # 3.1b
    "K9Graph",
    "BEHAVIOR_TO_IDX",
    "IDX_TO_BEHAVIOR",
    "NUM_BEHAVIORS",
    "P2_BEHAVIORS",
    "keypoints_to_pyskl",
    "pyskl_to_keypoints",
    "compute_bone_flow",
    "compute_motion_flow",
    "normalize_keypoints",
    "build_pyskl_annotation",
    "save_pyskl_pickle",
    # 3.1c
    "build_spatial_adjacency",
    "UnitGCN",
    "UnitTCN",
    "MSTCN",
    "STGCNBlock",
    "STGCN",
    "BCHead",
    "generate_boundary_labels",
    "STGCNBCLoss",
    "STGCNBC",
    "build_stgcn_bc",
    # 3.1d
    "STGCNBCDataset",
    "make_synthetic_dataset",
    "save_synthetic_dataset",
    "load_pyskl_pickle",
    "collate_fn",
    "TrainConfig",
    "STGCNBCTrainer",
    # 3.1e
    "STGCNBCInferer",
    "export_onnx",
]
