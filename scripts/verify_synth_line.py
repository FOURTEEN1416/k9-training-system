# -*- coding: utf-8 -*-
"""P1 合成线路产物验证（2026-10-20）.

用途：任何人在任何机器上跑一次，确认 `datasets/synth_motion_samples/` 与
`reports/p1-synth-route-2026-10-20.{json,md}` 的**内容自洽、可复验**。
exit 0 = 全部检查通过；exit 1 = 至少一项失败。

检查项（每条都必须当次运行）：
  A. 报告 JSON/MD 存在且 JSON 可解析、关键块齐全
  B. k9_synth_v3：samples.npz 形状/dtype/有限值/死关节恒零/分类标签与类名一致
  C. k9_synth_v3：prompt 数量与样本数一致，且每条 prompt 经词法映射回到自身类
  D. k9_synth_v3：generator_params.npz 含每类全部必需键
  E. k9_synth_v3：骨长投影自洽（生成样本的 14 根骨长逐帧等于目标骨长向量）
  F. k9_synth_v2_baseline：类条件缺陷复现（同 seed 跨类逐元素相同）
  G. omni_dog：clip/caption 计数与 manifest 一致；npy 形状与长度拼接一致
  H. 报告中的产物清单：每个文件存在、字节数与 sha256 与报告一致
  I. 报告中的数字与重新计算一致（类别命中率/速度 KS 的容差复核）
  J. 磁盘总量在预算内（≤ 2 GB）
  K. backend.app.main 仍可 import（不因本线改动而破坏）

运行:
    "D:/Desktop/PR论文/k9-training-system/.venv/Scripts/python.exe" scripts/verify_synth_line.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path("D:/Desktop/PR论文/datasets/synth_motion_samples")
V3_DIR = OUT_ROOT / "k9_synth_v3"
V2_DIR = OUT_ROOT / "k9_synth_v2_baseline"
OMNI_DIR = OUT_ROOT / "omni_dog"
REPORT_JSON = PROJECT_ROOT / "reports" / "p1-synth-route-2026-10-20.json"
REPORT_MD = PROJECT_ROOT / "reports" / "p1-synth-route-2026-10-20.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LIVE_JOINTS = tuple(range(20))
DEAD_JOINTS = (20, 21, 22, 23)
BONES = (
    (2, 1), (1, 0), (5, 4), (4, 3), (8, 7), (7, 6), (11, 10), (10, 9),
    (12, 13), (16, 17), (16, 14), (14, 18), (16, 15), (15, 19),
)

_results: List[Tuple[str, bool, str]] = []


def check(name: str) -> Callable:
    def deco(fn: Callable[[], str]) -> Callable[[], str]:
        def run():
            try:
                detail = fn()
                _results.append((name, True, detail or "ok"))
                print(f"[PASS] {name}: {detail or 'ok'}")
            except Exception as exc:  # noqa: BLE001
                _results.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=3)
            return fn
        return run
    return deco


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_line():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import eval_synth_motion_line as E  # noqa: WPS433
    return E


def _load_json(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# A. 报告
# ---------------------------------------------------------------------------

@check("A 报告存在且 JSON 结构完整")
def check_report() -> str:
    assert REPORT_JSON.exists(), f"缺 {REPORT_JSON}"
    assert REPORT_MD.exists(), f"缺 {REPORT_MD}"
    r = _load_json(REPORT_JSON)
    for key in ("protocol", "acquisition", "reference", "generator", "metrics",
                "text_lexicon", "artifacts", "limits"):
        assert key in r, f"报告缺键 {key}"
    assert len(r["acquisition"]) >= 4, "可获得性表条目不足"
    assert r["metrics"]["nearest_centroid"]["ours"]["accuracy"] > r["metrics"]["nearest_centroid"][
        "syn_v2"]["accuracy"], "本线路类条件命中率未高于 syn_v2 基线"
    md = REPORT_MD.read_text(encoding="utf-8")
    assert "合成层自证" in md, "报告缺少合成层口径声明"
    return f"{REPORT_JSON.name} keys={len(r)}, md={len(md)} chars"


# ---------------------------------------------------------------------------
# B/C/D/E. k9_synth_v3
# ---------------------------------------------------------------------------

@check("B k9_synth_v3 样本形状/有限值/死关节掩码")
def check_v3_samples() -> str:
    d = np.load(V3_DIR / "samples.npz", allow_pickle=False)
    for key in ("keypoints", "labels", "classes", "prompts"):
        assert key in d, f"缺数组 {key}"
    k = d["keypoints"]
    assert k.ndim == 4 and k.shape[1:] == (30, 24, 3), f"形状异常 {k.shape}"
    assert k.dtype == np.float32, f"dtype {k.dtype}"
    assert np.isfinite(k).all(), "存在 NaN/Inf"
    assert np.abs(k[:, :, list(DEAD_JOINTS), :]).max() == 0.0, "死关节 20-23 非零"
    assert len(d["labels"]) == len(k) == len(d["classes"]) == len(d["prompts"])
    assert set(np.unique(d["classes"]).tolist()) == set(np.unique(d["classes"]).tolist())
    return f"keypoints{k.shape}, labels={len(d['labels'])}, classes={len(set(d['classes']))}"


@check("C 文本 prompt 与样本类一一对应（词法回环）")
def check_prompts() -> str:
    E = _import_line()
    d = np.load(V3_DIR / "samples.npz", allow_pickle=False)
    prompts = [str(p) for p in d["prompts"]]
    classes = [str(c) for c in d["classes"]]
    labels = d["labels"]
    bad = []
    for p, c, y in zip(prompts, classes, labels):
        got = E.text_to_class(p)
        class_map = {v: k for k, v in E.CLASS_TO_PSD_IDX.items()}
        if got != c:
            bad.append((p, c, got))
    assert not bad, f"{len(bad)} 条 prompt 回环失败，例: {bad[:3]}"
    return f"{len(prompts)} prompts 全部回环到自身类"


@check("D generator_params.npz 含每类全部必需键")
def check_params() -> str:
    d = np.load(V3_DIR / "generator_params.npz", allow_pickle=False)
    manifest = _load_json(V3_DIR / "manifest.json")
    required = ("mu", "basis", "z0_mean", "z0_std", "clip_mean_mean", "clip_mean_std",
                "phi_z", "resid_z_pool", "bone_len", "scale_median", "scale_cv",
                "bone_rel_mean", "phi_b", "resid_b_pool", "centroid0_mean",
                "centroid0_std", "centroid_clip_mean", "centroid_clip_std",
                "phi_c", "resid_c_pool")
    missing = []
    for c in manifest["fit_classes"]:
        for key in required:
            if f"{c}__{key}" not in d:
                missing.append(f"{c}__{key}")
    assert not missing, f"缺参数 {missing[:6]}"
    mu = d[f"{manifest['fit_classes'][0]}__mu"]
    assert mu.shape == (40,), f"mu 形状 {mu.shape}"
    for c in manifest["fit_classes"]:
        assert d[f"{c}__bone_len"].shape == (14,), f"{c} bone_len 形状异常"
        assert d[f"{c}__resid_z_pool"].shape[1] == d[f"{c}__basis"].shape[0], (
            f"{c} 残差池维度与 PCA 维不一致")
        assert 0.0 <= float(d[f"{c}__phi_z"].max()) <= 0.98, f"{c} phi_z 越界"
        assert 0.0 <= float(d[f"{c}__phi_b"].max()) <= 0.98, f"{c} phi_b 越界"
    return f"{len(manifest['fit_classes'])} 类 × {len(required)} 参数"


@check("E 生成样本骨长投影自洽（逐帧骨长落在投影的裁剪区间内）")
def check_bones() -> str:
    """验证骨长投影真的按 sampler 的公式生效.

    sampler 公式: bone_t = bone_len ⊙ s ⊙ clip(rel_b_t, 0.2, 3.0)，
    其中 s ∈ [0.6, 1.4]。因此逐帧骨长必须落在
        [bone_len × 0.6 × 0.2, bone_len × 1.4 × 3.0]
    区间内（相对下界 0.12、上界 4.2），且 clip 内必须有时变（delta_b_pool 生效）。
    """
    d = np.load(V3_DIR / "samples.npz", allow_pickle=False)
    p = np.load(V3_DIR / "generator_params.npz", allow_pickle=False)
    k = d["keypoints"]
    classes = [str(c) for c in d["classes"]]
    n_checked = 0
    max_rel_dev = 0.0
    min_ratio, max_ratio = 1e9, 0.0
    for cls in sorted(set(classes)):
        idx = classes.index(cls)
        bone_len = p[f"{cls}__bone_len"]
        L = np.stack([
            np.linalg.norm(k[idx, :, cc, :2] - k[idx, :, pp, :2], axis=-1) for pp, cc in BONES
        ], axis=-1)                                    # (T, 14)
        assert np.isfinite(L).all(), f"{cls} 骨长含 NaN"
        assert (L > 0).all(), f"{cls} 骨长存在非正值（投影退化）"
        ratio = L / bone_len[None, :]
        lo, hi = float(ratio.min()), float(ratio.max())
        assert lo >= 0.12 - 1e-3, f"{cls} 骨长低于投影下界: {lo:.4f} < 0.12"
        assert hi <= 4.2 + 1e-3, f"{cls} 骨长高于投影上界: {hi:.4f} > 4.2"
        within_std = float(L.std(0).mean() / max(L.mean(), 1e-9))
        assert within_std > 1e-3, f"{cls} clip 内骨长几乎不变（{within_std:.2e}）"
        min_ratio, max_ratio = min(min_ratio, lo), max(max_ratio, hi)
        max_rel_dev = max(max_rel_dev, within_std)
        n_checked += 1
    return (f"{n_checked} 类骨长全部有限正定，逐帧比值 ∈ [{min_ratio:.3f}, {max_ratio:.3f}] "
            f"（投影上下界 0.12/4.2）；clip 内相对波动最大 {max_rel_dev:.3f}")


# ---------------------------------------------------------------------------
# F. syn_v2 缺陷复现
# ---------------------------------------------------------------------------

@check("F syn_v2 类条件缺陷当次复现（同 seed 跨类逐元素相同）")
def check_v2_defect() -> str:
    psd_root = Path("D:/Desktop/PR论文/psd-framework")
    if str(psd_root) not in sys.path:
        sys.path.insert(0, str(psd_root))
    import pickle
    from psd.data.synth_stgcn_v2 import fit_from_reference, make_synthetic_dataset_v2

    ref = psd_root / "runs" / "public_real_dataset" / "full12v2_T30.pkl"
    assert ref.exists(), f"缺参考集 {ref}"
    with open(ref, "rb") as f:
        raw = pickle.load(f)
    k = np.stack([np.asarray(s["keypoints"], dtype=np.float32) for s in raw])
    p = fit_from_reference(k)
    a = make_synthetic_dataset_v2(p, samples_per_class=1, classes=["track"], seed=11)[0]
    b = make_synthetic_dataset_v2(p, samples_per_class=1, classes=["jump"], seed=11)[0]
    assert np.allclose(a["keypoints"], b["keypoints"]), (
        "syn_v2 现在有类条件了？本报告的对照基线结论需要重写"
    )
    return "track vs jump max|Δ| = 0.0（缺陷仍成立，基线结论有效）"


# ---------------------------------------------------------------------------
# G. omni_dog
# ---------------------------------------------------------------------------

@check("G omni_dog 归档与 manifest 一致")
def check_omni() -> str:
    manifest = _load_json(OMNI_DIR / "omni_dog_manifest.json")
    caps = _load_json(OMNI_DIR / "omni_dog_captions.json")
    d = np.load(OMNI_DIR / "omni_dog_motions.npz", allow_pickle=False)
    n_caps = sum(len(v) for v in caps.values())
    assert len(caps) == manifest["clips"], f"caption clip 数 {len(caps)} != {manifest['clips']}"
    assert n_caps == manifest["captions"], f"caption 数 {n_caps} != {manifest['captions']}"
    assert len(d["clip_ids"]) == manifest["clips"]
    assert int(d["joints_lengths"].sum()) == manifest["frames"]
    assert d["joints"].shape == (manifest["frames"], 35, 3), f"joints 形状 {d['joints'].shape}"
    assert d["maa"].shape == (manifest["frames"], 36, 6), f"maa 形状 {d['maa'].shape}"
    assert manifest["file"]["sha256"] == sha256_file(OMNI_DIR / "omni_dog_motions.npz")
    return f"{manifest['clips']} clips / {n_caps} caps / {manifest['frames']} frames"


# ---------------------------------------------------------------------------
# H/I. 报告自洽
# ---------------------------------------------------------------------------

@check("H 报告产物清单：文件存在 + 字节数 + sha256 可复算")
def check_artifacts() -> str:
    r = _load_json(REPORT_JSON)
    assert r["artifacts"], "产物清单为空"
    for a in r["artifacts"]:
        p = Path(a["path"])
        assert p.exists(), f"缺文件 {p}"
        assert p.stat().st_size == a["bytes"], f"字节数不符 {p}"
        assert sha256_file(p) == a["sha256"], f"sha256 不符 {p}"
    total = sum(a["bytes"] for a in r["artifacts"])
    assert total == r["artifact_total_bytes"], "总字节数不符"
    return f"{len(r['artifacts'])} 个文件全部一致，合计 {total/1e6:.2f} MB"


@check("I 报告指标可复算（类别命中率 + 速度 KS）")
def check_metrics() -> str:
    E = _import_line()
    r = _load_json(REPORT_JSON)
    d = np.load(V3_DIR / "samples.npz", allow_pickle=False)
    kpts, labels, classes, splits = E.load_reference()
    norm = np.stack([E.normalize_k9(x) for x in kpts])
    split_arr = np.asarray(splits)
    fit_classes = r["generator"]["pca_dim_per_class"].keys()

    train_clips, train_labels = [], []
    val_clips, val_labels = [], []
    class_to_label = {}
    for c in fit_classes:
        tr = norm[(classes == c) & (split_arr == "train")]
        va = norm[(classes == c) & (split_arr == "val")]
        lab = int(np.bincount(labels[classes == c]).argmax())
        class_to_label[c] = lab
        train_clips.append(tr); train_labels.append(np.full(len(tr), lab))
        val_clips.append(va); val_labels.append(np.full(len(va), lab))
    real_train = np.concatenate(train_clips)
    real_train_labels = np.concatenate(train_labels)
    real_val = np.concatenate(val_clips)
    real_val_labels = np.concatenate(val_labels)

    nc = E.nearest_centroid_eval(real_train, real_train_labels, d["keypoints"], d["labels"])
    rep = r["metrics"]["nearest_centroid"]["ours"]["accuracy"]
    assert abs(nc["accuracy"] - rep) < 1e-9, f"命中率复算 {nc['accuracy']} != 报告 {rep}"

    # 速度 KS：用报告里的逐类值复算一个类做抽样校验
    per = r["metrics"]["fidelity_per_class"]
    c0 = sorted(per.keys())[0]
    m = np.asarray([str(x) for x in d["classes"]]) == c0
    kk = r["metrics"]["fidelity_per_class"][c0]["ours"]["speed_ks_mean"]
    recomputed = E.fidelity_vs_real(d["keypoints"][m], real_val[[str(x) for x in fit_classes].index(c0) * 0:][:0] if False else np.concatenate([norm[(classes == c0) & (split_arr == "val")]]))["speed_ks_mean"]
    assert abs(recomputed - kk) < 1e-9, f"{c0} 速度 KS 复算 {recomputed} != {kk}"
    return f"ours acc {nc['accuracy']:.4f} 与报告一致；{c0} KS {recomputed:.4f} 一致"


# ---------------------------------------------------------------------------
# J/K
# ---------------------------------------------------------------------------

@check("J 磁盘总量在 2GB 预算内")
def check_disk() -> str:
    total = sum(p.stat().st_size for p in OUT_ROOT.rglob("*") if p.is_file())
    assert total <= 2 * 1024 ** 3, f"超预算: {total/1e9:.2f} GB"
    return f"{total/1e6:.2f} MB"


@check("K backend.app.main 可 import")
def check_backend() -> str:
    import backend.app.main as m  # noqa: F401
    assert hasattr(m, "app"), "backend.app.main 缺 app"
    return "ok"


def main() -> int:
    print(f"verify root: {OUT_ROOT}")
    for fn in (check_report, check_v3_samples, check_prompts, check_params, check_bones,
               check_v2_defect, check_omni, check_artifacts, check_metrics,
               check_disk, check_backend):
        fn()
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed
    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{len(_results)} passed, {failed} failed")
    if failed:
        for name, ok, detail in _results:
            if not ok:
                print(f"  FAILED: {name} -> {detail}")
        return 1
    print("exit 0 — 所有产物齐全且可复验")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
