"""K9Graph — 24 节点犬类骨架拓扑（pyskl 兼容）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1b
依据: dev-docs/research/RESEARCH_STGCN_BC.md §4.3 + backend/ml/behavior/constants.py

设计:
    - 24 节点犬类专属拓扑（非人类 17/25 关键点）
    - pyskl Graph 接口兼容（num_nodes / inward / outward / partition）
    - 根节点: WITHERS (22) — 鬐甲/肩峰，犬体最高点，作为骨架中心枢纽
    - 解剖学简化: 项目无独立骨盆点，前后肢均从 withers 延伸

拓扑结构（parent 数组）:
    withers(22) ── 根
      ├─ throat(23) ── nose(16) ── chin(17)
      │                            ├─ left_ear_base(14) ── left_ear_tip(18)
      │                            ├─ right_ear_base(15) ── right_ear_tip(19)
      │                            ├─ left_eye(20)
      │                            └─ right_eye(21)
      ├─ tail_start(12) ── tail_end(13)
      ├─ front_left_elbow(2) ── front_left_knee(1) ── front_left_paw(0)
      ├─ front_right_elbow(8) ── front_right_knee(7) ── front_right_paw(6)
      ├─ rear_left_elbow(5) ── rear_left_knee(4) ── rear_left_paw(3)
      └─ rear_right_elbow(11) ── rear_right_knee(10) ── rear_right_paw(9)

注意:
    本类为项目自有实现，不依赖 pyskl 安装。pyskl 集成时可直接传入
    `pyskl.utils.graph.Graph` 的子类或作为独立图拓扑配置使用。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# 复用 constants.py 的 24 关键点索引（单一真相源）
from backend.ml.behavior.constants import (
    NUM_KEYPOINTS,
    WITHERS, THROAT, NOSE, CHIN,
    LEFT_EAR_BASE, RIGHT_EAR_BASE,
    LEFT_EAR_TIP, RIGHT_EAR_TIP,
    LEFT_EYE, RIGHT_EYE,
    TAIL_START, TAIL_END,
    FRONT_LEFT_PAW, FRONT_LEFT_KNEE, FRONT_LEFT_ELBOW,
    FRONT_RIGHT_PAW, FRONT_RIGHT_KNEE, FRONT_RIGHT_ELBOW,
    REAR_LEFT_PAW, REAR_LEFT_KNEE, REAR_LEFT_ELBOW,
    REAR_RIGHT_PAW, REAR_RIGHT_KNEE, REAR_RIGHT_ELBOW,
)


class K9Graph:
    """犬类 24 节点骨架拓扑（pyskl 兼容）.

    属性:
        num_nodes: 节点数 = 24
        root: 根节点索引 = WITHERS (22)
        inward: List[(neighbor, center)] — 指向根的边（child, parent）
        outward: List[(center, neighbor)] — 远离根的边（parent, child）
        parent: np.ndarray — parent[i] = 节点 i 的父节点索引，root 的 parent = -1
        adjacency: np.ndarray — (V, V) 邻接矩阵（对称，0/1）
    """

    # 24 节点关键点名称（与 data/dog-pose.yaml kpt_names 一致）
    NODE_NAMES: List[str] = [
        "front_left_paw", "front_left_knee", "front_left_elbow",
        "rear_left_paw", "rear_left_knee", "rear_left_elbow",
        "front_right_paw", "front_right_knee", "front_right_elbow",
        "rear_right_paw", "rear_right_knee", "rear_right_elbow",
        "tail_start", "tail_end",
        "left_ear_base", "right_ear_base",
        "nose", "chin",
        "left_ear_tip", "right_ear_tip",
        "left_eye", "right_eye",
        "withers", "throat",
    ]

    # 父子关系: (parent, child) — 远离根的方向
    # 根: withers(22)
    RAW_OUTWARD_EDGES: List[Tuple[int, int]] = [
        # 头部链: withers → throat → nose → {chin, ear_base, eye}
        (WITHERS, THROAT),
        (THROAT, NOSE),
        (NOSE, CHIN),
        (NOSE, LEFT_EAR_BASE),
        (NOSE, RIGHT_EAR_BASE),
        (LEFT_EAR_BASE, LEFT_EAR_TIP),
        (RIGHT_EAR_BASE, RIGHT_EAR_TIP),
        (NOSE, LEFT_EYE),
        (NOSE, RIGHT_EYE),
        # 尾部: withers → tail_start → tail_end
        (WITHERS, TAIL_START),
        (TAIL_START, TAIL_END),
        # 前左肢: withers → elbow → knee → paw
        (WITHERS, FRONT_LEFT_ELBOW),
        (FRONT_LEFT_ELBOW, FRONT_LEFT_KNEE),
        (FRONT_LEFT_KNEE, FRONT_LEFT_PAW),
        # 前右肢
        (WITHERS, FRONT_RIGHT_ELBOW),
        (FRONT_RIGHT_ELBOW, FRONT_RIGHT_KNEE),
        (FRONT_RIGHT_KNEE, FRONT_RIGHT_PAW),
        # 后左肢（简化: 从 withers 延伸，实际解剖从骨盆）
        (WITHERS, REAR_LEFT_ELBOW),
        (REAR_LEFT_ELBOW, REAR_LEFT_KNEE),
        (REAR_LEFT_KNEE, REAR_LEFT_PAW),
        # 后右肢
        (WITHERS, REAR_RIGHT_ELBOW),
        (REAR_RIGHT_ELBOW, REAR_RIGHT_KNEE),
        (REAR_RIGHT_KNEE, REAR_RIGHT_PAW),
    ]

    def __init__(self) -> None:
        self.num_nodes: int = NUM_KEYPOINTS  # 24
        self.root: int = WITHERS  # 22

        # outward: [(center, neighbor), ...] 即 (parent, child)
        self.outward: List[Tuple[int, int]] = list(self.RAW_OUTWARD_EDGES)
        # inward: [(neighbor, center), ...] 即 (child, parent)
        self.inward: List[Tuple[int, int]] = [(v, u) for (u, v) in self.outward]

        # parent 数组（用于骨骼流计算 bone[v] = joint[v] - joint[parent[v]]）
        self.parent: np.ndarray = np.full(self.num_nodes, -1, dtype=np.int64)
        for (p, c) in self.outward:
            self.parent[c] = p

        # 对称邻接矩阵（V, V）
        self.adjacency: np.ndarray = self._build_adjacency()

        # pyskl 分区策略占位（spatial config partitioning）
        # 0: 根节点自身, 1: 向心邻居, 2: 离心邻居
        self.partition_labeling: List[Tuple[int, int]] = self._build_partition()

    def _build_adjacency(self) -> np.ndarray:
        """构建对称邻接矩阵 (V, V)。"""
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int8)
        for (u, v) in self.outward:
            adj[u, v] = 1
            adj[v, u] = 1
        # 自环（pyskl ST-GCN 标准做法）
        np.fill_diagonal(adj, 1)
        return adj

    def _build_partition(self) -> List[Tuple[int, int]]:
        """构建 spatial partitioning 标签 (node, partition_label).

        partition_label:
            0: 根节点自身
            1: 向心邻居（指向根的方向）
            2: 离心邻居（远离根的方向）

        返回: [(node_idx, label), ...] 用于 ST-GCN 空间分区卷积
        """
        labeling: List[Tuple[int, int]] = []
        for v in range(self.num_nodes):
            if v == self.root:
                labeling.append((v, 0))
            else:
                # 检查是否在 inward（向心）或 outward（离心）边中
                is_inward = any(v == neighbor for (neighbor, _) in self.inward if neighbor == v)
                is_outward = any(v == neighbor for (_, neighbor) in self.outward if neighbor == v)
                if is_inward:
                    labeling.append((v, 1))
                elif is_outward:
                    labeling.append((v, 2))
                else:
                    labeling.append((v, 0))  # 孤立点（不应出现）
        return labeling

    def get_bones(self) -> List[Tuple[int, int]]:
        """返回骨骼流定义 (child, parent) 列表.

        用于 ST-GCN++ 多流融合中的骨骼流计算:
            bone[v] = joint[v] - joint[parent[v]]
        """
        return [(c, p) for (p, c) in self.outward]

    def get_neighbor_pairs(self) -> List[Tuple[int, int]]:
        """返回所有相邻节点对（对称，用于邻接矩阵构建）。"""
        return self.outward + self.inward

    def summary(self) -> str:
        """返回拓扑摘要字符串（调试用）。"""
        lines = [
            f"K9Graph (24-node canine skeleton topology)",
            f"  Root: {self.NODE_NAMES[self.root]} (idx={self.root})",
            f"  Nodes: {self.num_nodes}",
            f"  Edges: {len(self.outward)} (directed outward)",
            f"  Adjacency shape: {self.adjacency.shape}",
            f"  Non-zero adjacency: {int((self.adjacency > 0).sum())}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"K9Graph(num_nodes={self.num_nodes}, root={self.NODE_NAMES[self.root]}, "
            f"edges={len(self.outward)})"
        )
