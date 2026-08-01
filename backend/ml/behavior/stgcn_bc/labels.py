"""22 类行为标签映射（ST-GCN+BC 索引）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1b
依据: dev-docs/research/RESEARCH_STGCN_BC.md §5.1 + RESEARCH_FCI_IGP_STANDARD.md

22 类 = P0 基础 8 + P1 训练 8 + P2 高级 6
    P0: sit/down/stand/heel/sit_up/stay/bark/bite
    P1: track/alert_sit/alert_down/apprehend/escort/obstacle/recall/watch
    P2: guard/release/retrieve/jump/scale/search_blind

索引约定:
    - 0-7: P0 基础（与 Phase 1/2 规则引擎一致）
    - 8-15: P1 训练（与 Phase 2 规则引擎一致）
    - 16-21: P2 高级（Phase 3 新增，FCI-IGP 国际标准）

注意:
    索引顺序与 constants.py 中 P0_BEHAVIORS / P1_BEHAVIORS / P2_BEHAVIORS 元组顺序一致，
    保证规则引擎（Phase 2）与 ST-GCN+BC（Phase 3）标签空间兼容。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from backend.ml.behavior.constants import (
    P0_BEHAVIORS, P1_BEHAVIORS, P2_BEHAVIORS,
    ALL_BEHAVIORS_22, NUM_BEHAVIORS_22,
    BEHAVIOR_NAMES_CN, BEHAVIOR_SUBJECTS,
)


# ===== 22 类标签索引（ST-GCN+BC 输出空间） =====

# 行为名称 → 索引（训练/推理用）
BEHAVIOR_TO_IDX: Dict[str, int] = {
    name: idx for idx, name in enumerate(ALL_BEHAVIORS_22)
}

# 索引 → 行为名称（解码用）
IDX_TO_BEHAVIOR: Dict[int, str] = {
    idx: name for name, idx in BEHAVIOR_TO_IDX.items()
}

# 总类别数
NUM_BEHAVIORS: int = NUM_BEHAVIORS_22  # 22

# 按层级分组（用于分层评估）
P0_IDX: List[int] = [BEHAVIOR_TO_IDX[b] for b in P0_BEHAVIORS]  # [0..7]
P1_IDX: List[int] = [BEHAVIOR_TO_IDX[b] for b in P1_BEHAVIORS]  # [8..15]
P2_IDX: List[int] = [BEHAVIOR_TO_IDX[b] for b in P2_BEHAVIORS]  # [16..21]

# 层级标签（用于评估报告分层统计）
LAYER_LABELS: List[str] = (
    ["P0"] * len(P0_BEHAVIORS)
    + ["P1"] * len(P1_BEHAVIORS)
    + ["P2"] * len(P2_BEHAVIORS)
)

# FCI-IGP 阶段映射（用于评分卡对齐）
# A: 追踪 / B: 服从 / C: 护卫
FCI_IGP_STAGE: Dict[str, str] = {
    # P0 — 服从基础（IGP-B）
    "sit": "B", "down": "B", "stand": "B", "heel": "B",
    "sit_up": "B", "stay": "B", "bark": "B", "bite": "C",
    # P1 — 训练专项
    "track": "A", "alert_sit": "A", "alert_down": "A",
    "apprehend": "C", "escort": "C", "obstacle": "B",
    "recall": "B", "watch": "C",
    # P2 — 高级
    "guard": "C", "release": "C", "retrieve": "B",
    "jump": "B", "scale": "B", "search_blind": "A",
}


def get_behavior_idx(name: str) -> int:
    """行为名称 → 索引（不存在则 KeyError）."""
    return BEHAVIOR_TO_IDX[name]


def get_behavior_name(idx: int) -> str:
    """索引 → 行为名称（不存在则 KeyError）."""
    return IDX_TO_BEHAVIOR[idx]


def get_behavior_cn(name: str) -> str:
    """行为名称 → 中文名."""
    return BEHAVIOR_NAMES_CN.get(name, name)


def get_layer(idx: int) -> str:
    """索引 → 层级标签（P0/P1/P2）."""
    return LAYER_LABELS[idx]


def get_fci_igp_stage(name: str) -> str:
    """行为名称 → FCI-IGP 阶段（A/B/C）."""
    return FCI_IGP_STAGE.get(name, "B")


def get_all_labels() -> List[Tuple[int, str, str, str, str]]:
    """返回完整标签表 [(idx, name, cn, layer, fci_stage), ...]."""
    return [
        (idx, name, get_behavior_cn(name), get_layer(idx), get_fci_igp_stage(name))
        for idx, name in sorted(IDX_TO_BEHAVIOR.items())
    ]


def labels_summary() -> str:
    """标签摘要（调试用）."""
    lines = [
        f"ST-GCN+BC 22-class label mapping",
        f"  Total: {NUM_BEHAVIORS} classes",
        f"  P0 (basic):    {len(P0_IDX)} classes, idx {P0_IDX[0]}-{P0_IDX[-1]}",
        f"  P1 (training): {len(P1_IDX)} classes, idx {P1_IDX[0]}-{P1_IDX[-1]}",
        f"  P2 (advanced): {len(P2_IDX)} classes, idx {P2_IDX[0]}-{P2_IDX[-1]}",
        f"  FCI-IGP stages: A(追踪)={sum(1 for v in FCI_IGP_STAGE.values() if v=='A')}, "
        f"B(服从)={sum(1 for v in FCI_IGP_STAGE.values() if v=='B')}, "
        f"C(护卫)={sum(1 for v in FCI_IGP_STAGE.values() if v=='C')}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # CLI 调试: 打印完整标签表
    print(labels_summary())
    print()
    print(f"{'Idx':>4} | {'Name':<14} | {'中文':<8} | {'Layer':<4} | {'IGP':<3}")
    print("-" * 50)
    for idx, name, cn, layer, stage in get_all_labels():
        print(f"{idx:>4} | {name:<14} | {cn:<8} | {layer:<4} | {stage:<3}")
