"""ST-GCN+BC 行为识别模块（Phase 3.1）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1
依据: dev-docs/stages/phase-3.md §3.1 + dev-docs/research/RESEARCH_STGCN_BC.md

子模块:
    - k9_graph: K9Graph 24 节点犬类骨架拓扑（pyskl 兼容）
    - labels: 22 类行为标签映射（P0 8 + P1 8 + P2 6）
    - data_adapter: YOLO26-pose (T, 24, 3) → pyskl 骨架格式

注意:
    本模块为项目自有实现，核心代码不依赖 pyskl 安装即可运行。
    pyskl 集成（ST-GCN++ 主干 + 训练管线）作为 3.1b 子任务，
    在 Clash 代理恢复后通过 `pip install git+https://github.com/kennymckormick/pyskl.git` 集成。
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

__all__ = [
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
]
