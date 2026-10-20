# -*- coding: utf-8 -*-
"""P1 合成生产线路 — 类条件 / 骨长约束的犬运动合成 + 诚实评测（2026-10-20）.

Owner: ML 开发（合成生产线，见 AGENTS.md §2.2）

背景（侦察结论，见 reports/p1-synth-route-2026-10-20.md）
--------------------------------------------------------
- AniMo（CVPR2025）：代码可得，权重/AniMo4D 数据不可得（作者 issue 明确不提供
  checkpoint），且无 LICENSE。仅作设计参考，不进入训练。
- X-MoGen（AAAI2026）：仓库仅空壳（code/xmogen/README.md = 0 字节），零可用产物。
- OmniMotionGPT（CVPR2024, MIT）：唯一真正可下载项。1240 clip / 1771 caption
  随仓库发布，其中 dog 65 clip / 192 caption / 2624 帧；但无预训练权重、缺 ylib、
  motion_clip.py 硬编码 23*6，无法在本轮预算内形成可训练基线。本轮只做数据摄取
  与可读性核验。
- syn_v2（psd-framework）：实测"同 seed 下 track 与 jump 输出逐元素相同"，
  参数里没有任何类条件字段 → 无条件分布采样器 + 假标签，不能直接接文本条件。

本脚本的产出（不是对上述论文的复现）
------------------------------------
1. `datasets/synth_motion_samples/k9_synth_v3/`：
   类条件 PCA 隐轨迹 + AR(1) 时序 + 骨长投影 + 文本词法条件（7 类 PSD 行为）。
   每个类单独拟合 mu / PCA 基 / phi / sigma / 骨长 / 全局尺度，采样时按类取参数。
   —— 这是对 syn_v2 缺陷的最小修复版线路，不是扩散模型，也不是文本-运动对齐模型。
2. `datasets/synth_motion_samples/k9_synth_v2_baseline/`：
   psd-framework synth_stgcn_v2 的同域基线（只读调用，不修改对方仓库）。
3. `datasets/synth_motion_samples/omni_dog/`：
   OmniMotionGPT 仓库内 dog 子集（65 clip / 192 caption）的本地归档，
   供未来跨物种/文本词表研究；本轮不训练。
4. `reports/p1-synth-route-2026-10-20.{md,json}`：可获得性、指标、限制。

评测口径（三层分离铁律）
------------------------
本脚本一切数字属于**合成层自证**，不得与公开真实层 / 真实 K9 指标混排。
- 可控性：以真实训练集类质心做最近邻命中率（生成样本按构造拟合训练类统计，
  高命中率是预期结果，不能当生成质量证据）。
- 保真度：生成 vs 留出真实验证集的关节角/速度 KS、直方图 L1、姿态均值误差、
  骨长变异系数；并给"真实训练集折半"作为同量纲参照下限。

运行:
    "D:/Desktop/PR论文/k9-training-system/.venv/Scripts/python.exe" \
        scripts/eval_synth_motion_line.py

输出:
    reports/p1-synth-route-2026-10-20.json
    reports/p1-synth-route-2026-10-20.md
    D:/Desktop/PR论文/datasets/synth_motion_samples/...
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PSD_ROOT = Path("D:/Desktop/PR论文/psd-framework")
REFERENCE_PKL = PSD_ROOT / "runs" / "public_real_dataset" / "full12v2_T30.pkl"
OUT_ROOT = Path("D:/Desktop/PR论文/datasets/synth_motion_samples")
EXTERNAL_ANIMO = PROJECT_ROOT / "external" / "animo"
EXTERNAL_OMNI = PROJECT_ROOT / "external" / "omnimotiongpt" / "OmniMotionGPT-main"
REPORT_JSON = PROJECT_ROOT / "reports" / "p1-synth-route-2026-10-20.json"
REPORT_MD = PROJECT_ROOT / "reports" / "p1-synth-route-2026-10-20.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PSD_ROOT) not in sys.path:
    sys.path.insert(0, str(PSD_ROOT))

# 24 节点 K9 拓扑中，dog-pose GT 从未标注、YOLO 输出为高置信垃圾的 4 个死关节。
# 证据: psd-framework psd/data/ak_pose_extract.py:41 DEAD_JOINTS = (20,21,22,23)
LIVE_JOINTS: Tuple[int, ...] = tuple(range(20))
DEAD_JOINTS: Tuple[int, ...] = (20, 21, 22, 23)

# K9Graph 的 live 骨链（父 → 子）。16 是头部组件根，22(withers) 在 dead 集合中，
# 因此真实数据上的骨架是 6 个连通分量而非单树。骨长投影按分量从近端向末端传播。
BONES: Tuple[Tuple[int, int], ...] = (
    (2, 1), (1, 0),
    (5, 4), (4, 3),
    (8, 7), (7, 6),
    (11, 10), (10, 9),
    (12, 13),
    (16, 17), (16, 14), (14, 18), (16, 15), (15, 19),
)

CLASS_ORDER: Tuple[str, ...] = (
    "stay", "bite", "track", "apprehend", "watch", "retrieve", "jump",
)

# PSD 22 类标签索引（backend/ml/behavior/stgcn_bc/labels.py 的 ALL_BEHAVIORS_22 顺序）
CLASS_TO_PSD_IDX: Dict[str, int] = {
    "sit": 0, "down": 1, "stand": 2, "stay": 3, "heel": 4, "sit_up": 5,
    "bark": 6, "bite": 7, "track": 8, "alert_sit": 9, "alert_down": 10,
    "apprehend": 11, "escort": 12, "obstacle": 13, "recall": 14, "watch": 15,
    "guard": 16, "release": 17, "retrieve": 18, "jump": 19, "scale": 20,
    "search_blind": 21,
}
# 注意: full12v2 的 label 字段与上表并非同一索引空间，下面在加载时实测映射。
# 实测（2026-10-20）: {(0,sit),(3,stay),(5,bite),(6,track),(7,apprehend),
#                     (8,watch),(9,retrieve),(10,jump)} —— 与 psd_class 字段一一对应，
# 因此一律以 pickle 内的 label 为准，CLASS_TO_PSD_IDX 仅用于文档。


# ---------------------------------------------------------------------------
# 基础 IO
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(obj):
    """递归转 JSON 原生类型（numpy → Python）。"""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_reference(path: Path = REFERENCE_PKL) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """加载 full12v2 公开真实层参考集.

    Returns:
        kpts: (N, 30, 24, 3) float32
        labels: (N,) int64 — pickle 内 label（与 psd_class 对齐）
        classes: (N,) object str
        splits: list[str]
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)  # SECURITY: 本地可信训练缓存，非网络来源
    kpts = np.stack([np.asarray(s["keypoints"], dtype=np.float32) for s in raw])
    labels = np.asarray([int(s["label"]) for s in raw], dtype=np.int64)
    classes = np.asarray([str(s["psd_class"]) for s in raw])
    splits = [str(s["split"]) for s in raw]
    return kpts, labels, classes, splits


def normalize_k9(kpt: np.ndarray) -> np.ndarray:
    """与 backend STGCNBCDataset._normalize 完全一致的归一化（本地复刻）.

    注意实现事实：中心点取 joint 22（withers），而该关节在 dog-pose GT 中
    从未标注（全零），因此中心恒为 0；尺度取 ||joint22 - joint12|| 的均值，
    即"原点到尾根的距离"。这是训练域的真实口径，生成样本必须一致。
    """
    k = np.asarray(kpt, dtype=np.float32).copy()
    center = k[:, 22:23, :].mean(axis=0, keepdims=True)
    k = k - center
    bone_ref = k[:, 22, :2] - k[:, 12, :2]
    bone_len = np.linalg.norm(bone_ref, axis=-1).mean()
    if bone_len < 1e-6:
        bone_len = 1.0
    k[..., :2] = k[..., :2] / bone_len
    return k.astype(np.float32)


def assert_normalizer_matches_backend(sample_raw: np.ndarray) -> float:
    """本地归一化 vs backend 实现的逐元素最大差（应 < 1e-6）。"""
    from backend.ml.behavior.stgcn_bc.dataset import STGCNBCDataset

    a = normalize_k9(sample_raw)
    b = STGCNBCDataset._normalize(sample_raw.astype(np.float32))
    return float(np.abs(a - b).max())


# ---------------------------------------------------------------------------
# 文本词法条件（明确：词法映射，不是学习到的文本编码器）
# ---------------------------------------------------------------------------

# 词干用前缀匹配（允许 bites/biting/carries 等变形）；中文用子串。
TEXT_LEXICON: Dict[str, List[str]] = {
    "stay": ["stay", "stays", "keeping still", "keeps still", "remain", "hold position",
             "holds position", "freeze", "保持", "停留", "不动"],
    "bite": ["bit", "chew", "grip", "咬"],
    "track": ["track", "scent", "trail", "sniff", "follow", "nose down", "追踪", "嗅", "循迹"],
    "apprehend": ["apprehend", "attack", "lunge", "charge", "confront", "扑咬", "攻击", "冲"],
    "watch": ["watch", "alert", "stare", "guard", "attention", "注视", "警戒", "盯"],
    "retrieve": ["retriev", "fetch", "carri", "bring back", "brings back", "衔取", "取回", "叼回"],
    "jump": ["jump", "leap", "hurdle", "vault", "跳跃", "跳", "跃"],
}

TEXT_TEMPLATES: Dict[str, List[str]] = {
    "stay": [
        "the dog stays in place and remains still",
        "the dog keeps still while the handler walks away",
        "the dog holds position without moving its paws",
        "保持原地不动，等待指令",
    ],
    "bite": [
        "the dog bites the sleeve and grips it firmly",
        "the dog chews the tug toy with its back teeth",
        "the dog grips the decoy sleeve and does not release",
        "咬住护袖并牢牢咬紧",
    ],
    "track": [
        "the dog tracks the scent trail with its nose down",
        "the dog follows the trail and sniffs the ground",
        "the dog works the scent line across the field",
        "低头嗅闻并沿气味线追踪",
    ],
    "apprehend": [
        "the dog lunges and attacks the decoy",
        "the dog charges forward to apprehend the suspect",
        "the dog confronts the decoy with a forward lunge",
        "向前扑咬并攻击目标",
    ],
    "watch": [
        "the dog watches the decoy and stays alert",
        "the dog stares at the stranger with attention",
        "the dog guards the object and watches the approach",
        "警戒注视前方目标",
    ],
    "retrieve": [
        "the dog fetches the dumbbell and carries it back",
        "the dog retrieves the thrown object",
        "the dog brings back the dumbbell to the handler",
        "衔取哑铃并叼回",
    ],
    "jump": [
        "the dog jumps over the hurdle",
        "the dog leaps across the ditch",
        "the dog vaults the obstacle in one jump",
        "跃过障碍物",
    ],
}


def _keyword_hit(text: str, kw: str) -> bool:
    t = text.lower()
    if re.search(r"[\u4e00-\u9fff]", kw):
        return kw in t
    return re.search(r"(?<![a-z])" + re.escape(kw.lower()), t) is not None


def text_to_class(text: str) -> Optional[str]:
    """词法文本 → PSD 类名；无命中或跨类并列歧义返回 None."""
    scores: Dict[str, int] = {}
    lengths: Dict[str, int] = {}
    for cls, kws in TEXT_LEXICON.items():
        hits = [kw for kw in kws if _keyword_hit(text, kw)]
        if hits:
            scores[cls] = len(hits)
            lengths[cls] = sum(len(kw) for kw in hits)
    if not scores:
        return None
    best = max(scores.values())
    winners = [c for c, s in scores.items() if s == best]
    if len(winners) > 1:
        longest = max(lengths[c] for c in winners)
        winners = [c for c in winners if lengths[c] == longest]
    return winners[0] if len(winners) == 1 else None


# ---------------------------------------------------------------------------
# 特征 / 指标
# ---------------------------------------------------------------------------

def _bone_lengths(kpt: np.ndarray) -> np.ndarray:
    """(T, 14) 每帧骨长."""
    return np.stack(
        [np.linalg.norm(kpt[:, c, :2] - kpt[:, p, :2], axis=-1) for p, c in BONES],
        axis=-1,
    )


def clip_features(kpt: np.ndarray) -> np.ndarray:
    """单 clip → 定长特征向量（用于最近类质心命中率与保真度）."""
    k = np.asarray(kpt, dtype=np.float64)
    xy = k[:, LIVE_JOINTS, :2]
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=-1)  # (T-1, 20)
    bones = _bone_lengths(k)  # (T, 14)
    feats = [
        xy.mean(0).ravel(),          # 40 姿态均值（含全局位置）
        xy.std(0).ravel(),           # 40 姿态幅度
        speed.mean(0),               # 20 速度均值
        speed.std(0),                # 20 速度波动
        bones.mean(0),               # 14 骨长（未归一）
    ]
    return np.concatenate(feats).astype(np.float64)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(
        np.searchsorted(a, grid, side="right") / len(a)
        - np.searchsorted(b, grid, side="right") / len(b)
    )))


def hist_l1(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if not hi > lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    ha = np.histogram(a, bins=edges)[0] / len(a)
    hb = np.histogram(b, bins=edges)[0] / len(b)
    return float(np.abs(ha - hb).sum())


def class_speed(kpts: np.ndarray) -> np.ndarray:
    """(N,T,24,3) → (N, T-1, 20) 逐关节速度幅值."""
    xy = np.asarray(kpts, dtype=np.float64)[:, :, LIVE_JOINTS, :2]
    return np.linalg.norm(np.diff(xy, axis=1), axis=-1)


def fidelity_vs_real(gen: np.ndarray, real: np.ndarray) -> Dict[str, float]:
    """生成集 vs 真实集：速度 KS / 直方图 L1 / 姿态均值误差 / 骨长 CV."""
    gs, rs = class_speed(gen), class_speed(real)
    ks = [ks_distance(gs[..., j], rs[..., j]) for j in range(len(LIVE_JOINTS))]
    l1 = [hist_l1(gs[..., j], rs[..., j]) for j in range(len(LIVE_JOINTS))]

    # 姿态均值误差：先减每帧质心，只比较"狗自身姿态"，否则会被画面绝对位置主导
    # （实测真实 train 与 real val 的质心本身就差 1.1，远超任何模型误差）
    def _pose_only(X):
        xy = np.asarray(X, dtype=np.float64)[:, :, LIVE_JOINTS, :2]
        return (xy - xy.mean(2, keepdims=True)).reshape(-1, len(LIVE_JOINTS), 2).mean(0)

    gm, rm = _pose_only(gen), _pose_only(real)
    scale = float(_bone_lengths(np.asarray(real, dtype=np.float64)[0]).mean()) or 1.0
    pose_err = float(np.linalg.norm(gm - rm) / scale)

    gb = np.concatenate([_bone_lengths(np.asarray(x, dtype=np.float64)) for x in gen], axis=0)
    rb = np.concatenate([_bone_lengths(np.asarray(x, dtype=np.float64)) for x in real], axis=0)
    g_cv = float(np.mean(gb.std(0) / np.maximum(gb.mean(0), 1e-9)))
    r_cv = float(np.mean(rb.std(0) / np.maximum(rb.mean(0), 1e-9)))
    return {
        "speed_ks_mean": float(np.mean(ks)),
        "speed_l1_mean": float(np.mean(l1)),
        "speed_mean_gen": float(gs.mean()),
        "speed_mean_real": float(rs.mean()),
        "speed_ratio": float(gs.mean() / max(rs.mean(), 1e-9)),
        "pose_mean_err_rel": pose_err,
        "bone_cv_gen": g_cv,
        "bone_cv_real": r_cv,
        "n_gen": int(len(gen)),
        "n_real": int(len(real)),
    }


def nearest_centroid_eval(
    train_kpts: np.ndarray,
    train_labels: np.ndarray,
    probes: np.ndarray,
    probe_labels: np.ndarray,
) -> Dict[str, object]:
    """用真实训练集类质心给 probe 分类（标准化后欧氏最近）."""
    Xtr = np.stack([clip_features(x) for x in train_kpts])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr = (Xtr - mu) / sd
    classes = sorted(set(train_labels.tolist()))
    centroids = np.stack([Ztr[train_labels == c].mean(0) for c in classes])
    Zp = (np.stack([clip_features(x) for x in probes]) - mu) / sd
    d = ((Zp[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    pred_idx = d.argmin(1)
    pred = np.asarray([classes[i] for i in pred_idx])
    acc = float((pred == probe_labels).mean()) if len(probe_labels) else float("nan")
    confusion = Counter(zip(probe_labels.tolist(), pred.tolist()))
    return {
        "accuracy": acc,
        "n": int(len(probe_labels)),
        "chance": float(1.0 / len(classes)),
        "confusion": {f"{int(t)}->{int(p)}": int(n) for (t, p), n in sorted(confusion.items())},
        "classes": [int(c) for c in classes],
    }


# ---------------------------------------------------------------------------
# 类条件生成器：PCA 隐轨迹 + AR(1) + 骨长投影
# ---------------------------------------------------------------------------

def _split_shape_motion(clip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(T,24,3) → (相对姿态 (T,40), 质心轨迹 (T,2)).

    参考集归一化的中心点取 joint 22（dead，恒 0），因此归一化坐标仍携带
    大幅 clip 级平移（实测每 clip 质心空间 std ≈ 0.15）。把"狗在画面里的位置"
    与"狗自身的姿态"分开建模，避免类条件被画面位置主导。
    """
    xy = np.asarray(clip, dtype=np.float64)[:, LIVE_JOINTS, :2]
    centroid = xy.mean(1)                       # (T, 2)
    rel = (xy - centroid[:, None, :]).reshape(len(xy), -1)
    return rel, centroid


def _fit_class_generator(clips: np.ndarray, max_pca: int = 8) -> Dict[str, np.ndarray]:
    """单类拟合（相对姿态 PCA+AR(1)，质心轨迹随机游走，骨长统计）.

    Args:
        clips: (N, T, 24, 3) 已归一化的该类训练 clip.

    Returns:
        dict: mu(40), basis(k,40), z0_mean/z0_std(k), clip_mean_mean/std(k),
              phi_z(k), resid_z_pool(n,k),
              bone_len(14), scale_cv, bone_rel_mean(14), phi_b(14), resid_b_pool(n,14),
              centroid0_mean/std(2), centroid_clip_mean/std(2), phi_c(2), resid_c_pool(n,2)
    """
    N, T = clips.shape[:2]
    rel = np.stack([_split_shape_motion(c)[0] for c in clips])     # (N,T,40)
    cen = np.stack([_split_shape_motion(c)[1] for c in clips])     # (N,T,2)

    X = rel.reshape(N * T, len(LIVE_JOINTS) * 2)
    mu = X.mean(0)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(len(Xc) - 1, 1)
    ratio = var / max(var.sum(), 1e-12)
    cum = np.cumsum(ratio)
    k = int(np.searchsorted(cum, 0.90) + 1)
    k = max(2, min(max_pca, k, Vt.shape[0]))
    basis = Vt[:k]  # (k, 40)
    Z = (Xc @ basis.T).reshape(N, T, k)

    # 关键分解：clip 级静态姿态偏移（狗站姿/体型差异）与 clip 内时序变化必须分开。
    # 直接对合并的 Z 拟合 AR(1) 会把"类内不同 clip 的均值差"误当成时序自相关，
    # 导致 phi 被高估、sigma 失衡，生成序列速度偏高（本轮实测 0.268 vs 真实 0.145）。
    clip_mean = Z.mean(1)                      # (N, k)
    Zm = Z - clip_mean[:, None, :]             # 去 clip 均值后的帧内偏差
    zc_mean = clip_mean.mean(0)
    zc_std = clip_mean.std(0) + 1e-6
    # 帧内偏差的经验增量池。**为什么不用高斯 AR(1)**：本轮实测
    # ① clip 级均值必须先剥离，否则 phi 被"类间均值差"污染；
    # ② 剥离后残差仍强跨维负相关（质心残差 corr=-0.58）且重尾——
    #    高斯预测步长 0.207 vs 实测 0.143。独立高斯建出的序列速度是真实的 ~1.8x。
    # 因此隐轨迹与质心轨迹都直接用"真实相邻帧增量行"bootstrap（保留全部
    # 跨维相关、重尾与零膨胀），生成序列的帧间统计与训练集同分布。
    z0_mean = Zm[:, 0, :].mean(0)
    z0_std = Zm[:, 0, :].std(0) + 1e-6
    # 逐维 AR(1) 系数 + **经验多元残差行**。纯 cumsum（随机游走）会无界漂移：
    # 实测 30 帧内骨长漂到真实值的 2.6 倍、姿态均值误差爆到 34（真实折半下限 1.7）。
    # 用真实残差行做新息，既保留跨维相关/重尾/零膨胀，又由 AR(1) 保证平稳有界。
    phi_z = np.zeros(k)
    for j in range(k):
        a = Zm[:, :-1, j].ravel()
        b = Zm[:, 1:, j].ravel()
        den = float((a * a).sum())
        phi_z[j] = float(np.clip((a * b).sum() / den, 0.0, 0.98)) if den > 1e-12 else 0.0
    resid_z = (Zm[:, 1:, :] - phi_z * Zm[:, :-1, :]).reshape(-1, k)

    # 质心：同样分解为 clip 级位置偏移 + 帧内 AR(1)
    cen_clip_mean = cen.mean(1)                                  # (N, 2)
    cen_dev = cen - cen_clip_mean[:, None, :]
    phi_c = np.zeros(2)
    for j in range(2):
        a = cen_dev[:, :-1, j].ravel()
        b = cen_dev[:, 1:, j].ravel()
        den = float((a * a).sum())
        phi_c[j] = float(np.clip((a * b).sum() / den, 0.0, 0.98)) if den > 1e-12 else 0.0
    resid_c = (cen_dev[:, 1:, :] - phi_c * cen_dev[:, :-1, :]).reshape(-1, 2)
    # 首帧取"相对 clip 均值"的偏差，避免与 clip 级偏移重复计入
    cen0_mean = cen_dev[:, 0, :].mean(0)
    cen0_std = cen_dev[:, 0, :].std(0) + 1e-6

    bones = np.concatenate([
        _bone_lengths(c.astype(np.float64)) for c in clips
    ], axis=0)
    bone_len = np.median(bones, axis=0)
    scale = bones.mean(1)
    scale_median = float(np.median(scale))
    scale_cv = float(scale.std() / max(scale.mean(), 1e-9))
    # 逐帧骨长时变：同样用经验增量池（骨长 CV 很高，既有真实舒展/透视，也有检测抖动）
    rel_b = (bones / np.maximum(scale[:, None], 1e-9)).reshape(N, T, len(BONES))
    bm = rel_b.reshape(-1, len(BONES)).mean(0)
    phi_b = np.zeros(len(BONES))
    rb_dev = rel_b - bm
    for j in range(len(BONES)):
        a = rb_dev[:, :-1, j].ravel()
        b = rb_dev[:, 1:, j].ravel()
        den = float((a * a).sum())
        phi_b[j] = float(np.clip((a * b).sum() / den, 0.0, 0.98)) if den > 1e-12 else 0.0
    resid_b = (rb_dev[:, 1:, :] - phi_b * rb_dev[:, :-1, :]).reshape(-1, len(BONES))
    return {
        "mu": mu,
        "basis": basis,
        "z0_mean": z0_mean,
        "z0_std": z0_std,
        "clip_mean_mean": zc_mean,
        "clip_mean_std": zc_std,
        "phi_z": phi_z,
        "resid_z_pool": resid_z,
        "bone_len": bone_len,
        "scale_median": np.asarray([scale_median]),
        "scale_cv": np.asarray([scale_cv]),
        "bone_rel_mean": bm,
        "phi_b": phi_b,
        "resid_b_pool": resid_b,
        "centroid0_mean": cen0_mean,
        "centroid0_std": cen0_std,
        "centroid_clip_mean": cen_clip_mean.mean(0),
        "centroid_clip_std": cen_clip_mean.std(0) + 1e-6,
        "phi_c": phi_c,
        "resid_c_pool": resid_c,
        }


def _project_bones(xy: np.ndarray, target_len: np.ndarray) -> np.ndarray:
    """按 (T,20,2) 从近端向末端投影，使 14 根骨长精确等于 target_len.

    Args:
        xy: (T, 20, 2)
        target_len: (B,) 全帧共用，或 (T, B) 逐帧骨长
    """
    xy = xy.copy()
    tl = np.asarray(target_len, dtype=np.float64)
    per_frame = tl.ndim == 2
    for bi, (p, c) in enumerate(BONES):
        d = xy[:, c, :] - xy[:, p, :]
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        fallback = np.array([1.0, 0.0])
        unit = np.where(n > 1e-8, d / np.maximum(n, 1e-8), fallback)
        if per_frame:
            xy[:, c, :] = xy[:, p, :] + unit * tl[:, bi][:, None]
        else:
            xy[:, c, :] = xy[:, p, :] + unit * float(tl[bi])
    return xy


def _draw_residual_rows(pool: np.ndarray, need: int, block: int,
                        rng: np.random.Generator) -> np.ndarray:
    """从残差池按**连续块**抽样（block bootstrap）.

    为什么必须按块：残差行在构造上连续（来自同一段轨迹），逐行独立重抽会破坏
    残余的时间结构，生成序列的帧间速度被压低/抬高。实测 track 类：
    block=1 生成速度 0.230（真实 0.166），block=5 → 0.176、KS 0.35 → 0.18。
    """
    if block <= 1:
        return pool[rng.integers(0, len(pool), need)]
    max_start = max(len(pool) - block, 1)
    rows: List[np.ndarray] = []
    while sum(len(r) for r in rows) < need:
        st = int(rng.integers(0, max_start))
        rows.append(pool[st:st + block])
    return np.concatenate(rows, axis=0)[:need]


def sample_class(params: Dict[str, np.ndarray], n: int, rng: np.random.Generator,
                 block: int = 5) -> np.ndarray:
    """按类参数采样 (n, 30, 24, 3) float32（归一化域）.

    轨迹构造（全部为平稳 AR(1)，新息从该类**真实残差行**bootstrap）:
        z_t      = phi_z · z_{t-1} + resid_z_row
        pose_t   = mu + (z_t + clip_shift) @ basis
        rel_b_t  = bone_rel_mean + phi_b · dev_{t-1} + resid_b_row
        bone_t   = bone_len ⊙ rel_b_t        （注意 rel_b 已按 clip 平均骨长归一）
        centroid = clip 偏移 + clip 内 AR(1)
    """
    T = 30
    k = len(params["z0_mean"])
    rz = params["resid_z_pool"]
    rc = params["resid_c_pool"]
    rb_pool = params["resid_b_pool"]
    out = np.zeros((n, T, 24, 3), dtype=np.float32)
    for i in range(n):
        z = np.empty((T, k), dtype=np.float64)
        z[0] = params["z0_mean"] + params["z0_std"] * rng.standard_normal(k)
        rows_z = _draw_residual_rows(rz, T - 1, block, rng)
        for t in range(1, T):
            z[t] = params["phi_z"] * z[t - 1] + rows_z[t - 1]
        # 加回该 clip 的静态姿态偏移（类内 clip 间差异，不参与时序动态）
        clip_shift = params["clip_mean_mean"] + params["clip_mean_std"] * rng.standard_normal(k)
        pose = params["mu"] + (z + clip_shift) @ params["basis"]  # (T, 40)
        xy = pose.reshape(T, len(LIVE_JOINTS), 2)

        # 骨长：rel_b 已在拟合时按 clip 平均骨长归一（均值 ≈ 1）→ 直接乘 bone_len
        dev_b = np.empty((T, len(BONES)), dtype=np.float64)
        rows_b = _draw_residual_rows(rb_pool, T - 1, block, rng)
        dev_b[0] = rb_pool[rng.integers(0, len(rb_pool))]
        for t in range(1, T):
            dev_b[t] = params["phi_b"] * dev_b[t - 1] + rows_b[t - 1]
        rel_b = params["bone_rel_mean"] + dev_b
        s = float(np.clip(1.0 + params["scale_cv"][0] * rng.standard_normal(), 0.6, 1.4))
        xy = _project_bones(xy, np.clip(rel_b, 0.2, 3.0) * params["bone_len"] * s)

        cen = np.empty((T, 2), dtype=np.float64)
        cen[0] = params["centroid0_mean"] + params["centroid0_std"] * rng.standard_normal(2)
        rows_c = _draw_residual_rows(rc, T - 1, block, rng)
        for t in range(1, T):
            cen[t] = params["phi_c"] * cen[t - 1] + rows_c[t - 1]
        cen = cen + (params["centroid_clip_mean"]
                     + params["centroid_clip_std"] * rng.standard_normal(2))

        out[i][:, LIVE_JOINTS, :2] = (xy + cen[:, None, :]).astype(np.float32)
        # 置信度：从该类真实经验池逐关节采样（保持边缘分布）
        for j in LIVE_JOINTS:
            out[i, :, j, 2] = rng.choice(params["conf_pools"][j], size=T).astype(np.float32)
        # dead joints（含 withers 22）保持 0
    return out


def _fit_conf_pools(clips: np.ndarray) -> List[np.ndarray]:
    return [clips[:, :, j, 2].ravel() for j in LIVE_JOINTS]


# ---------------------------------------------------------------------------
# 数据集构建
# ---------------------------------------------------------------------------

def build_line(
    kpts: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    splits: Sequence[str],
    samples_per_class: int,
    seed: int,
) -> Dict[str, object]:
    """拟合并生成 k9_synth_v3；同时生成 syn_v2 基线与评测指标."""
    split_arr = np.asarray(splits)
    norm = np.stack([normalize_k9(x) for x in kpts])
    classes_all = sorted(set(classes.tolist()))
    fit_classes = [c for c in CLASS_ORDER if c in classes_all]
    excluded = [c for c in classes_all if c not in fit_classes]

    train_clips_by_class: Dict[str, np.ndarray] = {}
    val_clips_by_class: Dict[str, np.ndarray] = {}
    for c in fit_classes:
        tr = (classes == c) & (split_arr == "train")
        va = (classes == c) & (split_arr == "val")
        train_clips_by_class[c] = norm[tr]
        val_clips_by_class[c] = norm[va]

    # 最小拟合样本数：训练 clip ≥ 4（sit 仅 2 条且全在 val，被排除；apprehend 训练 6 条保留）
    excluded_reason: Dict[str, str] = {}
    for c in list(fit_classes):
        n_tr = len(train_clips_by_class[c])
        if n_tr < 4:
            excluded_reason[c] = f"训练 clip 仅 {n_tr} 条（<4），不足以拟合类条件参数"
    fit_classes = [c for c in fit_classes if c not in excluded_reason]
    excluded += [c for c in CLASS_ORDER if c in classes_all and c not in fit_classes and c not in excluded]

    rng = np.random.default_rng(seed)
    params: Dict[str, Dict[str, np.ndarray]] = {}
    gen_by_class: Dict[str, np.ndarray] = {}
    prompts: List[str] = []
    gen_labels: List[int] = []
    gen_classes: List[str] = []
    class_to_label = {}
    for c in fit_classes:
        cls_labels = labels[classes == c]
        class_to_label[c] = int(Counter(cls_labels.tolist()).most_common(1)[0][0])

        p = _fit_class_generator(train_clips_by_class[c])
        p["conf_pools"] = _fit_conf_pools(train_clips_by_class[c])
        params[c] = p
        samples = sample_class(p, samples_per_class, rng)
        gen_by_class[c] = samples
        templates = TEXT_TEMPLATES[c]
        for i in range(samples_per_class):
            prompt = templates[i % len(templates)]
            if text_to_class(prompt) != c:
                raise AssertionError(f"模板未通过自身词法映射: {prompt!r} -> {text_to_class(prompt)!r}")
            prompts.append(prompt)
            gen_labels.append(class_to_label[c])
            gen_classes.append(c)

    gen_all = np.concatenate([gen_by_class[c] for c in fit_classes], axis=0)

    # ---- syn_v2 基线（只读调用 psd-framework；同域、同训练集拟合）----
    from psd.data.synth_stgcn_v2 import fit_from_reference, make_synthetic_dataset_v2

    train_stack = np.concatenate([train_clips_by_class[c] for c in fit_classes], axis=0)
    v2_params = fit_from_reference(train_stack)
    v2_raw = make_synthetic_dataset_v2(
        v2_params, samples_per_class=samples_per_class, classes=fit_classes, seed=seed
    )
    v2_by_class = {
        c: np.stack([s["keypoints"] for s in v2_raw if s["label"] == i])
        for i, c in enumerate(fit_classes)
    }
    v2_all = np.concatenate([v2_by_class[c] for c in fit_classes], axis=0)

    # ---- 评测 ----
    real_train = np.concatenate([train_clips_by_class[c] for c in fit_classes], axis=0)
    real_train_labels = np.concatenate(
        [np.full(len(train_clips_by_class[c]), class_to_label[c]) for c in fit_classes]
    )
    real_val = np.concatenate([val_clips_by_class[c] for c in fit_classes], axis=0)
    real_val_labels = np.concatenate(
        [np.full(len(val_clips_by_class[c]), class_to_label[c]) for c in fit_classes]
    )
    # 验证集里若有 sit（本轮无模型）剔除；此处 fit_classes 已排除，故自然剔除
    sit_val = int(((classes == "sit") & (split_arr == "val")).sum())

    nc_real = nearest_centroid_eval(real_train, real_train_labels, real_val, real_val_labels)
    nc_ours = nearest_centroid_eval(real_train, real_train_labels, gen_all, np.asarray(gen_labels))
    nc_v2 = nearest_centroid_eval(real_train, real_train_labels, v2_all, np.asarray(gen_labels))

    fidelity = {}
    for c in fit_classes:
        fidelity[c] = {
            "ours": fidelity_vs_real(gen_by_class[c], val_clips_by_class[c]),
            "syn_v2": fidelity_vs_real(v2_by_class[c], val_clips_by_class[c]),
        }
    agg = {
        "ours": {k: float(np.mean([fidelity[c]["ours"][k] for c in fit_classes]))
                 for k in ("speed_ks_mean", "speed_l1_mean", "speed_mean_gen",
                           "speed_mean_real", "speed_ratio", "pose_mean_err_rel", "bone_cv_gen")},
        "syn_v2": {k: float(np.mean([fidelity[c]["syn_v2"][k] for c in fit_classes]))
                   for k in ("speed_ks_mean", "speed_l1_mean", "speed_mean_gen",
                             "speed_mean_real", "speed_ratio", "pose_mean_err_rel", "bone_cv_gen")},
    }
    # 同量纲参照：真实训练集按类随机折半后的一半 vs 另一半
    half = defaultdict(lambda: ([], []))
    for c in fit_classes:
        clips = train_clips_by_class[c]
        idx = np.arange(len(clips))
        rng.shuffle(idx)
        cut = max(1, len(idx) // 2)
        half[c] = (clips[idx[:cut]], clips[idx[cut:]] if len(idx) > cut else clips[idx[:cut]])
    real_floor = {
        "speed_ks_mean": float(np.mean([
            fidelity_vs_real(half[c][0], half[c][1])["speed_ks_mean"] for c in fit_classes
        ])),
        "speed_l1_mean": float(np.mean([
            fidelity_vs_real(half[c][0], half[c][1])["speed_l1_mean"] for c in fit_classes
        ])),
        "speed_mean_gen": float(np.mean([
            fidelity_vs_real(half[c][0], half[c][1])["speed_mean_gen"] for c in fit_classes
        ])),
        "speed_mean_real": float(np.mean([
            fidelity_vs_real(half[c][0], half[c][1])["speed_mean_real"] for c in fit_classes
        ])),
        "speed_ratio": float(np.mean([
            fidelity_vs_real(half[c][0], half[c][1])["speed_ratio"] for c in fit_classes
        ])),
        "pose_mean_err_rel": float(np.mean([
            fidelity_vs_real(half[c][0], half[c][1])["pose_mean_err_rel"] for c in fit_classes
        ])),
        "bone_cv_gen": float(np.mean([
            fidelity_vs_real(half[c][0], half[c][1])["bone_cv_gen"] for c in fit_classes
        ])),
        "bone_cv_real": float(np.mean([
            fidelity_vs_real(half[c][0], half[c][1])["bone_cv_real"] for c in fit_classes
        ])),
    }

    # ---- 2 折交叉拟合（**无泄漏**，本报告主口径）----
    # 每类训练 clip 随机两折：A 拟合 → B 折算保真度，交换再算一次。
    # 主口径的理由：留出验证集与训练集的速度水平本身就差很多
    # （实测 track 训练 0.166 vs 验证 0.083），这个混杂比任何模型误差都大；
    # 交叉拟合把"拟合集"和"被比较集"都限定在同一训练分布内，同量纲可比。
    split_rng = np.random.default_rng(seed + 1)
    crossfit: Dict[str, List[Dict[str, float]]] = {}
    for c in fit_classes:
        clips = train_clips_by_class[c]
        idx = np.arange(len(clips))
        split_rng.shuffle(idx)
        cut = max(1, len(idx) // 2)
        folds = [(idx[:cut], idx[cut:])]
        if len(idx[cut:]) >= 2:
            folds.append((idx[cut:], idx[:cut]))
        vals = []
        for a_idx, b_idx in folds:
            p_cf = _fit_class_generator(clips[a_idx])
            p_cf["conf_pools"] = _fit_conf_pools(clips[a_idx])
            # 生成量对齐到**留出折**规模（而非固定 40），让 KS 的分辨率与真实侧可比
            synth = sample_class(p_cf, max(len(b_idx), 4), np.random.default_rng(seed + 7))
            rec = fidelity_vs_real(synth, clips[b_idx])
            rec["n_fit"] = int(len(a_idx))
            rec["n_held"] = int(len(b_idx))
            vals.append(rec)
        crossfit[c] = vals
    crossfit_agg = {
        key: float(np.mean([np.mean([v[key] for v in crossfit[c]]) for c in fit_classes]))
        for key in ("speed_ks_mean", "speed_l1_mean", "speed_mean_gen", "speed_mean_real",
                    "speed_ratio", "pose_mean_err_rel", "bone_cv_gen", "bone_cv_real")
    }
    # 交叉拟合同折真实参照：A 折内部再对半（生成侧同折、同样本数级）
    floor_vals: Dict[str, List[float]] = defaultdict(list)
    for c in fit_classes:
        clips = train_clips_by_class[c]
        n_c = len(clips)
        idx = np.arange(n_c)
        np.random.default_rng(seed + 3).shuffle(idx)
        h = max(2, n_c // 2)
        a, b = clips[idx[:h]], clips[idx[h:]]
        if len(b) >= 2:
            for key, val in fidelity_vs_real(a, b).items():
                if isinstance(val, float):
                    floor_vals[key].append(val)
    crossfit_real_floor = {k: float(np.mean(v)) for k, v in floor_vals.items()}

    return {
        "fit_classes": fit_classes,
        "excluded_classes": excluded,
        "excluded_reason": excluded_reason,
        "sit_val_excluded": sit_val,
        "params": params,
        "gen_by_class": gen_by_class,
        "gen_all": gen_all,
        "prompts": prompts,
        "gen_labels": np.asarray(gen_labels, dtype=np.int64),
        "gen_classes": gen_classes,
        "class_to_label": class_to_label,
        "train_clips_by_class": train_clips_by_class,
        "val_clips_by_class": val_clips_by_class,
        "v2_by_class": v2_by_class,
        "v2_all": v2_all,
        "metrics": {
            "nearest_centroid": {"real_val": nc_real, "ours": nc_ours, "syn_v2": nc_v2},
            "fidelity_per_class": fidelity,
            "fidelity_aggregate": agg,
            "real_split_half_floor": real_floor,
            "crossfit": {
                "primary_metric": True,
                "note": "每类训练 clip 随机 2 折，A 拟合 → B 折算，交换再算；无验证集参与",
                "per_class": crossfit,
                "aggregate": crossfit_agg,
                "real_half_within_fold_floor": crossfit_real_floor,
            },
        },
    }


# ---------------------------------------------------------------------------
# OmniMotionGPT dog 子集摄取（只读仓库内 zip）
# ---------------------------------------------------------------------------

def ingest_omni_dog(out_dir: Path) -> Dict[str, object]:
    data_dir = EXTERNAL_OMNI / "data"
    zj = zipfile.ZipFile(data_dir / "animals_smal_joints.zip")
    zm = zipfile.ZipFile(data_dir / "animals_maa_motions.zip")
    zc = zipfile.ZipFile(data_dir / "motion_captions.zip")
    cap_names = set(zc.namelist())

    clip_ids = sorted({
        n.split("/")[1] for n in cap_names
        if n.count("/") >= 2 and n.split("/")[1].lower().startswith("doggie")
    })
    joints, lengths, maa, maa_lengths, kept, captions = [], [], [], [], [], {}
    for cid in clip_ids:
        jp = f"animals_smal_joints/{cid}.npy"
        mp = f"animals_maa_motions/{cid}.npy"
        cp = f"motion_captions/{cid}/screenshots.txt"
        if jp not in zj.namelist() or mp not in zm.namelist() or cp not in cap_names:
            continue
        j = np.load(io.BytesIO(zj.read(jp))).astype(np.float32)
        m = np.load(io.BytesIO(zm.read(mp))).astype(np.float32)
        if len(j) != len(m):
            continue
        joints.append(j); lengths.append(len(j))
        maa.append(m); maa_lengths.append(len(m))
        kept.append(cid)
        caps = [ln.strip() for ln in zc.read(cp).decode("utf-8", "ignore").splitlines() if ln.strip()]
        captions[cid] = caps

    np.savez_compressed(
        out_dir / "omni_dog_motions.npz",
        joints=np.concatenate(joints, axis=0),
        joints_lengths=np.asarray(lengths, dtype=np.int32),
        maa=np.concatenate(maa, axis=0),
        maa_lengths=np.asarray(maa_lengths, dtype=np.int32),
        clip_ids=np.asarray(kept),
    )
    with open(out_dir / "omni_dog_captions.json", "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=1)

    n_caps = sum(len(v) for v in captions.values())
    resolved = {c: 0 for c in TEXT_LEXICON}
    unresolved = 0
    for cid, cs in captions.items():
        for text in cs:
            cls = text_to_class(text)
            if cls is None:
                unresolved += 1
            else:
                resolved[cls] += 1
    manifest = {
        "source_repo": "USRC-SEA/OmniMotionGPT",
        "source_license": "MIT (LICENSE, Copyright (c) 2024 OPPO)",
        "source_archive_sha256": {
            "animals_smal_joints.zip": sha256_file(data_dir / "animals_smal_joints.zip"),
            "animals_maa_motions.zip": sha256_file(data_dir / "animals_maa_motions.zip"),
            "motion_captions.zip": sha256_file(data_dir / "motion_captions.zip"),
        },
        "clips": len(kept),
        "captions": n_caps,
        "frames": int(sum(lengths)),
        "joints_shape_note": "(sum_T, 35, 3) float32 — SMAL 35 关节",
        "maa_shape_note": "(sum_T, 36, 6) float32 — 35 关节 6D 旋转 + 根平移补 3 个 0",
        "caption_lexicon": {
            "resolved": resolved,
            "unresolved_out_of_taxonomy": unresolved,
            "note": "词法覆盖不是标注准确率；PSD 7 类词表无法覆盖 DeformingThings4D 的狗动作空间",
        },
        "file": {
            "name": "omni_dog_motions.npz",
            "bytes": int((out_dir / "omni_dog_motions.npz").stat().st_size),
            "sha256": sha256_file(out_dir / "omni_dog_motions.npz"),
        },
    }
    with open(out_dir / "omni_dog_manifest.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(manifest), f, ensure_ascii=False, indent=1)
    return manifest


# ---------------------------------------------------------------------------
# AniMo / X-MoGen 事实核验（本地已 clone 的代码，只读）
# ---------------------------------------------------------------------------

def verify_upstream_facts() -> Dict[str, object]:
    facts: Dict[str, object] = {}
    if EXTERNAL_ANIMO.exists():
        src = (EXTERNAL_ANIMO / "utils" / "paramUtil.py").read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"animo_raw_offsets\s*=\s*np\.array\(\s*\[(.*?)\]\s*\)", src, re.S)
        rows = re.findall(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", m.group(1)) if m else []
        tsrc = (EXTERNAL_ANIMO / "models" / "transformer" / "transformer.py").read_text(
            encoding="utf-8", errors="ignore")
        facts["animo"] = {
            "present": True,
            "offset_rows": len(rows),
            "layout_359_formula": 4 + (30 - 1) * 3 + (30 - 1) * 6 + 30 * 3 + 4,
            "license_file": (EXTERNAL_ANIMO / "LICENSE").exists(),
            "num_tokens_nameerror_in_base_transformer": (
                "self.token_emb = nn.Embedding(_num_tokens, self.code_dim)" in tsrc
                and "_num_tokens = opt.num_tokens + 1" in tsrc
            ),
            "has_checkpoint_files": any(
                p.suffix in {".pth", ".pt", ".bin", ".ckpt"}
                for p in EXTERNAL_ANIMO.rglob("*") if p.is_file()
            ),
        }
    else:
        facts["animo"] = {"present": False}
    facts["xmogen"] = {
        "note": "侦察结论：code/xmogen/README.md 为 0 字节空占位；本轮未重复克隆（零可用产物）",
    }
    facts["omnimotiongpt"] = {
        "present": EXTERNAL_OMNI.exists(),
        "license_head": (EXTERNAL_OMNI / "LICENSE").read_text(encoding="utf-8").splitlines()[0]
        if (EXTERNAL_OMNI / "LICENSE").exists() else None,
        "ylib_present": any(EXTERNAL_OMNI.rglob("ylib")),
        "motion_clip_input_feats_23x6": _grep_contains(
            EXTERNAL_OMNI / "OmGPT" / "motion_mapping" / "exp" / "motion_models" / "motion_clip.py",
            "self.input_feats = 23 * 6",
        ),
        "checkpoint_files": [
            str(p.relative_to(EXTERNAL_OMNI)) for p in EXTERNAL_OMNI.rglob("*")
            if p.is_file() and p.suffix in {".pth", ".pt", ".ckpt", ".bin"}
        ][:10],
    }
    return facts


def _grep_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def write_report_md(report: Dict[str, object], path: Path) -> None:
    m = report["metrics"]
    nc = m["nearest_centroid"]
    agg = m["fidelity_aggregate"]
    floor = m["real_split_half_floor"]
    acq = report["acquisition"]
    L = []
    L.append("# P1 合成生产线路 — 实施与评测报告（2026-10-20）\n")
    L.append("> 结论先行：**上游四个论文项目没有一个能在本机形成可训练/可复现基线**；"
             "本轮交付的是「类条件 + 骨长约束 + 词法文本条件」的合成生产线路、"
             "唯一可下载的真实犬文本-运动数据归档（OmniMotionGPT dog 子集），"
             "以及一套可复验的诚实评测。所有数字属**合成层自证**，不得与公开真实层/真实 K9 指标混排。\n")
    L.append("---\n")
    L.append("## 1. 可获得性实测\n")
    L.append("| 目标 | 可得 | 实测证据 | 处置 |")
    L.append("|------|------|----------|------|")
    for row in acq:
        L.append(f"| {row['target']} | {row['available']} | {row['evidence']} | {row['action']} |")
    L.append("")
    L.append("## 2. 本轮实现\n")
    L.append("- `scripts/eval_synth_motion_line.py`：拟合/生成/评测一体（本文件）。")
    L.append("- 生成器：**逐类** PCA（解释 ≥90% 方差，上限 8 维）建模「相对质心姿态」；"
             "clip 级静态偏移与 clip 内时序动态分开建模；时序动态 = 逐维 AR(1) + "
             "**真实残差行的块 bootstrap 新息**（block=5，保留跨维相关/重尾/短程自相关）；"
             "14 根骨长逐帧投影；逐类尺度抖动；逐关节经验置信度池。")
    L.append("- 文本条件：**词法映射**（模板关键词 → 类），不是 CLIP/学习到的文本编码器。"
             "全部 280 条 prompt 均通过自身词法映射回环校验。")
    L.append("- 基线：psd-framework `synth_stgcn_v2`（只读调用，同域、同训练集、同 seed）。")
    L.append("- 真实参考：`full12v2_T30.pkl`（352 clip / 30 帧 / 24 关节；视频级 train/val 无重叠，本轮实测）。")
    L.append("- 归一化口径：与 `backend.ml.behavior.stgcn_bc.dataset.STGCNBCDataset._normalize` "
             f"逐元素一致（最大差 {report['reference']['normalizer_max_abs_diff_vs_backend']:.2e}）。")
    L.append("")
    L.append("## 3. 结果\n")
    L.append("### 3.1 类区域命中率（最近类质心；非生成质量指标）\n")
    L.append("| 方法 | 命中率 | n | 随机 |")
    L.append("|------|--------|---|------|")
    L.append(f"| 真实留出验证集（描述子上限） | {nc['real_val']['accuracy']:.4f} | {nc['real_val']['n']} | {nc['real_val']['chance']:.4f} |")
    L.append(f"| 本线路 k9_synth_v3 | {nc['ours']['accuracy']:.4f} | {nc['ours']['n']} | {nc['ours']['chance']:.4f} |")
    L.append(f"| syn_v2 基线 | {nc['syn_v2']['accuracy']:.4f} | {nc['syn_v2']['n']} | {nc['syn_v2']['chance']:.4f} |")
    L.append("")
    L.append("> 生成样本按构造拟合训练类统计，命中率高是预期；它只证明\"类条件生效\"，不证明真实感。\n")
    cf = m["crossfit"]
    ca, cf_floor = cf["aggregate"], cf["real_half_within_fold_floor"]
    L.append("### 3.2 保真度 A（主口径：2 折交叉拟合，验证集不参与）\n")
    L.append("| 指标 | 本线路 | 同折真实对半（分辨率下限） |")
    L.append("|------|--------|---------------------------|")
    L.append("| 速度 KS 均值 | {:.4f} | {:.4f} |".format(ca['speed_ks_mean'], cf_floor['speed_ks_mean']))
    L.append("| 速度直方图 L1 | {:.4f} | {:.4f} |".format(ca['speed_l1_mean'], cf_floor['speed_l1_mean']))
    L.append("| 速度均值比（生成/真实） | {:.4f}x | {:.4f}x |".format(ca['speed_ratio'], cf_floor['speed_ratio']))
    L.append("| 姿态均值相对误差（质心归一） | {:.4f} | {:.4f} |".format(ca['pose_mean_err_rel'], cf_floor['pose_mean_err_rel']))
    L.append("| 骨长 CV（生成/真实） | {:.4f} / {:.4f} | {:.4f} / {:.4f} |".format(
        ca['bone_cv_gen'], ca['bone_cv_real'], cf_floor['bone_cv_gen'], cf_floor['bone_cv_real']))
    L.append("")
    L.append("> 拟合折 A → 合成 → 对留出折 B 算指标，交换再算一次，逐类二次平均。"
             "这是**主口径**：生成侧与被比较侧都限定在同一训练分布内。"
             "同折真实对半 = 同一切分下把 A 折再对半，代表真实数据在同样本数下的可分辨率。")
    L.append("")
    L.append("### 3.2b 保真度 B（次口径：生成 vs 留出真实验证集）\n")
    L.append("| 指标 | 本线路 | syn_v2 | 真实折半参照 |")
    L.append("|------|--------|--------|--------------|")
    L.append("| 速度 KS 均值 | {:.4f} | {:.4f} | {:.4f} |".format(
        agg['ours']['speed_ks_mean'], agg['syn_v2']['speed_ks_mean'], floor['speed_ks_mean']))
    L.append("| 速度直方图 L1 | {:.4f} | {:.4f} | {:.4f} |".format(
        agg['ours']['speed_l1_mean'], agg['syn_v2']['speed_l1_mean'], floor['speed_l1_mean']))
    L.append("| 速度均值比（生成/真实） | {:.4f}x | {:.4f}x | {:.4f}x |".format(
        agg['ours']['speed_ratio'], agg['syn_v2']['speed_ratio'], floor['speed_ratio']))
    L.append("| 姿态均值相对误差（质心归一） | {:.4f} | {:.4f} | {:.4f} |".format(
        agg['ours']['pose_mean_err_rel'], agg['syn_v2']['pose_mean_err_rel'], floor['pose_mean_err_rel']))
    L.append("| 骨长 CV（生成侧） | {:.4f} | {:.4f} | {:.4f} |".format(
        agg['ours']['bone_cv_gen'], agg['syn_v2']['bone_cv_gen'], floor['bone_cv_gen']))
    L.append("")
    L.append("> 次口径存在**已知混杂**：验证集与训练集的速度水平本身就不同"
             "（实测 track 训练 0.166 vs 验证 0.083；片段中位长度 52 帧被均匀抽成 30 帧，步长 ~1.7）。"
             "本线路在该口径 0.37 已明显低于 syn_v2 的 0.51，但仍远高于同量纲参照 0.13。"
             "此处只用于与 syn_v2 的相对比较，不单独作为质量证据。")
    L.append("")
    L.append("### 3.3 逐类保真度（本线路）\n")
    L.append("| 类 | n_gen | n_real | 速度 KS | 速度 L1 | 速度比 | 姿态误差 | 骨长 CV |")
    L.append("|----|-------|--------|---------|---------|--------|----------|---------|")
    for c, d in m["fidelity_per_class"].items():
        o = d["ours"]
        L.append(f"| {c} | {o['n_gen']} | {o['n_real']} | {o['speed_ks_mean']:.4f} | "
                 f"{o['speed_l1_mean']:.4f} | {o['speed_ratio']:.2f}× | "
                 f"{o['pose_mean_err_rel']:.4f} | {o['bone_cv_gen']:.4f} |")
    L.append("")
    L.append("### 3.4 文本词法\n")
    tl = report["text_lexicon"]
    L.append(f"- 生成 prompt：{tl['prompts_total']} 条，回环一致 {tl['prompt_roundtrip_ok']} 条。")
    L.append(f"- OmniMotionGPT dog 真实人工描述 {tl['omni_captions']} 条："
             f"词表命中 {tl['omni_resolved']} 条，未覆盖（超出 PSD 7 类动作空间）{tl['omni_unresolved']} 条。")
    L.append(f"- 按类命中：{tl['omni_resolved_by_class']}")
    L.append("")
    L.append("## 4. 产物\n")
    for a in report["artifacts"]:
        L.append(f"- `{a['path']}` — {a['bytes']} B, sha256 `{a['sha256'][:16]}…`")
    L.append("")
    L.append("## 5. 限制与未完成（如实）\n")
    for lim in report["limits"]:
        L.append(f"- {lim}")
    L.append("")
    L.append("## 5b. AniMo 359 维字段表（仅作设计参考，未使用其数据）\n")
    L.append("实测 `external/animo/utils/paramUtil.py` 的 `animo_raw_offsets` 为 30 行 → 30 关节；"
             "`train_vq.py:58-59` 设 `opt.joints_num=30, dim_pose=359`。该 359 维是 "
             "HumanML3D 风格向量向 30 关节的扩展：")
    L.append("")
    L.append("| 字段 | 宽度 | 计算 |")
    L.append("|------|------|------|")
    L.append("| 根旋转速度 | 1 | 固定 |")
    L.append("| 根线速度 | 2 | xy |")
    L.append("| 根高 | 1 | y |")
    L.append("| RIC（相对初始姿态） | 87 | (J-1)×3 |")
    L.append("| 6D 旋转 | 174 | (J-1)×6 |")
    L.append("| 局部速度 | 90 | J×3 |")
    L.append("| foot contact | 4 | 固定 |")
    L.append("| **合计** | **359** | 4+29×3+29×6+30×3+4 |")
    L.append("")
    L.append("这是**重定向后的骨架特征向量**，不是 SMPL/SMAL 参数直出；与我们 24 关节 COCO 域不可直接拼接。"
             "本线路未采用该编码（生成物保持 `(30,24,3)` 的 backend 原生域）。\n")
    L.append("## 6. 复现\n")
    L.append("```bash")
    L.append('"D:/Desktop/PR论文/k9-training-system/.venv/Scripts/python.exe" scripts/eval_synth_motion_line.py')
    L.append('"D:/Desktop/PR论文/k9-training-system/.venv/Scripts/python.exe" scripts/verify_synth_line.py')
    L.append("```")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1 合成生产线路：生成 + 评测 + 报告")
    ap.add_argument("--reference", type=Path, default=REFERENCE_PKL)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--samples-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args(argv)

    t0 = datetime.now(timezone.utc)
    kpts, labels, classes, splits = load_reference(args.reference)
    norm_check = assert_normalizer_matches_backend(kpts[0])

    line = build_line(kpts, labels, classes, splits, args.samples_per_class, args.seed)
    fit_classes = line["fit_classes"]

    # ---- 落盘：k9_synth_v3 ----
    v3_dir = args.out_root / "k9_synth_v3"
    v3_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        v3_dir / "samples.npz",
        keypoints=line["gen_all"],
        labels=line["gen_labels"],
        classes=np.asarray(line["gen_classes"]),
        prompts=np.asarray(line["prompts"]),
    )
    param_arrays = {}
    params_json = {}
    for c in fit_classes:
        p = line["params"][c]
        for k in ("mu", "basis", "z0_mean", "z0_std", "clip_mean_mean", "clip_mean_std",
                  "phi_z", "resid_z_pool", "bone_len", "scale_median", "scale_cv",
                  "bone_rel_mean", "phi_b", "resid_b_pool", "centroid0_mean",
                  "centroid0_std", "centroid_clip_mean", "centroid_clip_std",
                  "phi_c", "resid_c_pool"):
            param_arrays[f"{c}__{k}"] = p[k]
        params_json[c] = {
            "pca_dim": int(p["basis"].shape[0]),
            "n_train_clips": int(len(line["train_clips_by_class"][c])),
            "n_val_clips": int(len(line["val_clips_by_class"][c])),
            "label": int(line["class_to_label"][c]),
            "scale_median": float(p["scale_median"][0]),
        }
    np.savez_compressed(v3_dir / "generator_params.npz", **param_arrays)
    with open(v3_dir / "text_lexicon.json", "w", encoding="utf-8") as f:
        json.dump({"keywords": TEXT_LEXICON, "templates": TEXT_TEMPLATES,
                   "conditioning": "lexical (not a learned text encoder)"},
                  f, ensure_ascii=False, indent=1)
    v3_manifest = {
        "generator": "class-conditional PCA relative pose + empirical frame-delta bootstrap + bone-length projection + lexical text conditioning",
        "reference": {"path": str(args.reference), "sha256": sha256_file(args.reference),
                      "n_clips": int(len(kpts)), "n_classes_in_reference": len(set(classes.tolist()))},
        "fit_classes": fit_classes,
        "excluded_classes": line["excluded_classes"],
        "excluded_reason": line["excluded_reason"],
        "samples_per_class": args.samples_per_class,
        "n_samples": int(len(line["gen_all"])),
        "shape": list(line["gen_all"].shape),
        "dtype": str(line["gen_all"].dtype),
        "dead_joints": list(DEAD_JOINTS),
        "note": "dead joints 20-23（含 withers 22）恒为 0，与 dog-pose GT 的真实可观测性一致",
        "params": params_json,
        "files": {},
    }
    for name in ("samples.npz", "generator_params.npz", "text_lexicon.json"):
        v3_manifest["files"][name] = {
            "bytes": int((v3_dir / name).stat().st_size),
            "sha256": sha256_file(v3_dir / name),
        }
    with open(v3_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(v3_manifest), f, ensure_ascii=False, indent=1)

    # ---- 落盘：syn_v2 基线 ----
    v2_dir = args.out_root / "k9_synth_v2_baseline"
    v2_dir.mkdir(parents=True, exist_ok=True)
    v2_labels = np.asarray(line["gen_labels"], dtype=np.int64)
    np.savez_compressed(
        v2_dir / "samples.npz",
        keypoints=line["v2_all"],
        labels=v2_labels,
        classes=np.asarray([c for c in fit_classes for _ in range(args.samples_per_class)]),
    )
    with open(v2_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(jsonable({
            "source": "psd-framework psd/data/synth_stgcn_v2.py (read-only)",
            "known_defect": "params 无类条件字段；同 seed 下不同类输出逐元素相同（本轮实测）",
            "seed": args.seed,
            "samples_per_class": args.samples_per_class,
            "files": {
                "samples.npz": {
                    "bytes": int((v2_dir / "samples.npz").stat().st_size),
                    "sha256": sha256_file(v2_dir / "samples.npz"),
                }
            },
        }), f, ensure_ascii=False, indent=1)

    # ---- 落盘：Omni dog 子集 ----
    omni_dir = args.out_root / "omni_dog"
    omni_dir.mkdir(parents=True, exist_ok=True)
    omni_manifest = ingest_omni_dog(omni_dir)

    # ---- 上游事实 ----
    upstream = verify_upstream_facts()

    # ---- 指标 ----
    m = line["metrics"]
    prompts_total = len(line["prompts"])
    prompt_ok = sum(1 for p, c in zip(line["prompts"], line["gen_classes"]) if text_to_class(p) == c)
    omni_caps = sum(len(v) for v in json.load(
        open(omni_dir / "omni_dog_captions.json", encoding="utf-8")).values())
    omni_res_by_class = omni_manifest["caption_lexicon"]["resolved"]
    omni_resolved = int(sum(omni_res_by_class.values()))
    text_block = {
        "conditmoding_note": "lexical, not learned",
        "prompts_total": prompts_total,
        "prompt_roundtrip_ok": prompt_ok,
        "omni_captions": int(omni_caps),
        "omni_resolved": omni_resolved,
        "omni_unresolved": int(omni_manifest["caption_lexicon"]["unresolved_out_of_taxonomy"]),
        "omni_resolved_by_class": omni_res_by_class,
    }

    # ---- 报告 JSON ----
    pkl_rel = str(args.reference)
    v3_files = []
    for p in sorted(v3_dir.glob("*")):
        if p.is_file():
            v3_files.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    for p in sorted(v2_dir.glob("*")):
        if p.is_file():
            v3_files.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    for p in sorted(omni_dir.glob("*")):
        if p.is_file():
            v3_files.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)})

    total_bytes = sum(a["bytes"] for a in v3_files)
    report = {
        "protocol": "P1-SYNTH-ROUTE-2026-10-20",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_sec": (datetime.now(timezone.utc) - t0).total_seconds(),
        "command": "scripts/eval_synth_motion_line.py",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": _safe_version("torch"),
            "cuda_available": _safe_cuda(),
        },
        "acquisition": [
            {
                "target": "AniMo (CVPR2025)",
                "available": "code only",
                "evidence": f"git clone 成功（60 files, 736KB）；utils/paramUtil.py 实测 "
                            f"{upstream.get('animo', {}).get('offset_rows', '?')} 行 offsets；"
                            f"LICENSE 存在={upstream.get('animo', {}).get('license_file')}；"
                            f"仓库内 checkpoint 文件={upstream.get('animo', {}).get('has_checkpoint_files')}",
                "action": "仅作设计参考（359 维布局 / RVQ→base→residual 结构），不训练",
            },
            {
                "target": "X-MoGen (AAAI2026)",
                "available": "no",
                "evidence": "侦察实测：code/xmogen/README.md = 0 字节；仓库 13 条文件、无权重、无 LICENSE",
                "action": "零投入",
            },
            {
                "target": "OmniMotionGPT (CVPR2024)",
                "available": "code + data (MIT)",
                "evidence": f"codeload 81,414,958 B 解包成功；LICENSE 首行='{upstream.get('omnimotiongpt', {}).get('license_head')}'；"
                            f"dog 子集 {omni_manifest['clips']} clip / {omni_manifest['captions']} caption / "
                            f"{omni_manifest['frames']} 帧实测可读；ylib 存在={upstream.get('omnimotiongpt', {}).get('ylib_present')}；"
                            f"checkpoint={upstream.get('omnimotiongpt', {}).get('checkpoint_files')}",
                "action": "数据归档为 omni_dog/；不训练（无 ckpt + 缺 ylib + 硬编码 23*6）",
            },
            {
                "target": "Kirin / AiM3D",
                "available": "partial (action only)",
                "evidence": "侦察实测：代码未发布；dog.zip 24.5GB 中 npy 仅 ~123MB 压缩，但全部 caption 入口 404",
                "action": "本轮不下载（磁盘预算与文本缺口），仅记录为未来 Range 抽取候选",
            },
            {
                "target": "syn_v2 (psd-framework)",
                "available": "yes (in-repo)",
                "evidence": "本轮实测：fit_from_reference 返回 keys 无类条件字段；同 seed 下 track/jump 输出 max|Δ|=0.0",
                "action": "作为对照基线；本线路单独实现类条件生成器，不修改对方仓库",
            },
        ],
        "reference": {
            "path": pkl_rel,
            "sha256": sha256_file(args.reference),
            "n_clips": int(len(kpts)),
            "n_train": int(sum(s == "train" for s in splits)),
            "n_val": int(sum(s == "val" for s in splits)),
            "video_overlap_train_val": 0,
            "classes_in_reference": sorted(set(classes.tolist())),
            "normalizer_max_abs_diff_vs_backend": norm_check,
            "dead_joint_zero_rate": 1.0,
        },
        "generator": {
            "name": "k9_synth_v3",
            "type": "class-conditional PCA relative-pose + empirical frame-delta bootstrap + bone-length projection",
            "pca_dim_per_class": {c: int(line["params"][c]["basis"].shape[0]) for c in fit_classes},
            "text_conditioning": "lexical keyword mapping (not learned)",
            "samples_per_class": args.samples_per_class,
            "seed": args.seed,
        },
        "metrics": m,
        "text_lexicon": text_block,
        "upstream_facts": upstream,
        "omni_dog": {k: v for k, v in omni_manifest.items() if k != "file"},
        "artifacts": v3_files,
        "artifact_total_bytes": int(total_bytes),
        "disk_budget_bytes": 2 * 1024 ** 3,
        "limits": [
            "文本条件是词法映射，不是学习到的文本-运动对齐；不能声称具备文本生成能力。",
            "生成器是类条件 PCA + 经验增量 bootstrap 的统计模型（无神经网络、无训练损失），不是扩散/Transformer，也不是对 AniMo/X-MoGen/Omni 的复现。",
            "仅 7 个参考类可拟合；sit 只有 2 条且全在 val（训练 0 条），已排除。",
            "所有指标为合成层自证：真实留出集仅 96 条 clip、7 类，最近质心命中率的置信区间很宽。",
            "骨长逐帧投影保证骨骼刚性，但也会把真实检测抖动一并规整；骨长 CV 与真实仍有差距（见 3.2 表）。",
            "OmniMotionGPT 无法形成可训练基线：无预训练权重、无 ylib、motion_clip.py 硬编码 23*6；本轮只归档数据。",
            "AniMo 无 LICENSE、无权重、无 AniMo4D 数据，且 BaseTransformer 存在 _num_tokens NameError；未训练。",
            "Kirin 未下载：dog.zip 24.5GB（npy-only 需 Range 抽取）且 captions 全部 404；本轮未投入。",
            "X-MoGen 为空壳仓库，零可用产物。",
            "生成器新息直接取自真实训练残差行（block bootstrap），因此生成样本在动态统计上是"
            "训练片段的「重组」，无法外推训练集未出现的运动模式；这不是可泛化的生成模型。",
            "主口径速度均值仍为真实的 1.34 倍、姿态误差为真实折半下限的 1.9 倍——"
            "即本线路尚未达到\"与真实数据不可区分\"的保真度，只能作为受控合成层使用。",
            "脚本依赖本机路径 D:/Desktop/PR论文/psd-framework（参考集与 syn_v2 基线）与 "
            "external/omnimotiongpt（Omni dog 数据）；缺任一项时脚本会报错而非静默降级。",
            "白名单偏差：OmniMotionGPT 解包目录为 external/omnimotiongpt/（80MB，已被 .gitignore "
            "的 external/ 规则覆盖），非白名单列出的 external/animo/；因 Omni 数据是唯一可用真实犬文本-运动"
            "来源且 omni_dog 归档需要可复现地重建，故保留，特此声明。",
        ],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=1), encoding="utf-8")
    write_report_md(jsonable(report), REPORT_MD)

    print(f"[ok] report  {REPORT_JSON}")
    print(f"[ok] report  {REPORT_MD}")
    print(f"[ok] v3      {v3_dir} ({len(line['gen_all'])} samples, {len(fit_classes)} classes)")
    print(f"[ok] omni    {omni_dir} ({omni_manifest['clips']} clips / {omni_manifest['captions']} captions)")
    print(f"[metric] nearest-centroid real_val={m['nearest_centroid']['real_val']['accuracy']:.4f} "
          f"ours={m['nearest_centroid']['ours']['accuracy']:.4f} "
          f"syn_v2={m['nearest_centroid']['syn_v2']['accuracy']:.4f}")
    print(f"[metric] speed_ks ours={m['fidelity_aggregate']['ours']['speed_ks_mean']:.4f} "
          f"syn_v2={m['fidelity_aggregate']['syn_v2']['speed_ks_mean']:.4f} "
          f"floor={m['real_split_half_floor']['speed_ks_mean']:.4f}")
    print(f"[metric] pose_err ours={m['fidelity_aggregate']['ours']['pose_mean_err_rel']:.4f} "
          f"syn_v2={m['fidelity_aggregate']['syn_v2']['pose_mean_err_rel']:.4f} "
          f"floor={m['real_split_half_floor']['pose_mean_err_rel']:.4f}")
    print(f"[disk] artifacts={total_bytes/1e6:.2f} MB (budget 2048 MB)")
    return 0


def _safe_version(name: str) -> Optional[str]:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "?")
    except Exception:
        return None


def _safe_cuda() -> Optional[bool]:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
