"""P1 数据补充产出物校验脚本（MammalNet 家犬子集 + DogMo 预览素材）.

Owner: 数据实施员（P1 数据补充线）
日期: 2026-10-20

校验对象
--------
    D:\\Desktop\\PR论文\\datasets\\mammalnet_canine\\
    D:\\Desktop\\PR论文\\datasets\\dogmo_preview\\

判定
----
exit 0 = 产出物齐全、可复验（manifest 与实际字节一致、sha256 复算一致、
         全部视频 ffprobe 可解码、分层抽样覆盖 8 个家犬品种 × 3 个 split、
         每个数据集 ≤ 2 GB 预算）。
exit 1 = 任一 hard check 失败；报告写到 reports/p1-dataset-expansion-verify.json。

用法
----
    python scripts/verify_dataset_expansion.py
    python scripts/verify_dataset_expansion.py --datasets-root "D:/Desktop/PR论文/datasets" --json reports/p1-verify.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_ROOT = Path(r"D:\Desktop\PR论文\datasets")
BUDGET_BYTES = 2_000_000_000

EXPECTED_BREEDS = {
    "bulldog",
    "terrier",
    "malamute",
    "hound",
    "husky dog",
    "shepherd dog",
    "eskimo dog",
    "foxhound",
}
EXPECTED_SPLITS = {"train", "val", "test"}


class Checker:
    def __init__(self) -> None:
        self.checks: List[Dict] = []

    def add(self, name: str, ok: bool, detail: str, hard: bool = True) -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "hard": hard, "detail": detail})
        print(f"[{'PASS' if ok else ('FAIL' if hard else 'WARN')}] {name}: {detail}", flush=True)
        return ok

    @property
    def failed(self) -> List[Dict]:
        return [c for c in self.checks if not c["ok"] and c["hard"]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ffprobe(path: Path) -> Optional[Dict]:
    """返回 {codec, width, height, duration, nb_frames} 或 None。"""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    cmd = [
        exe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,duration,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return None
        s = streams[0]
        # 与 manifest 生成口径一致：优先 format.duration（容器总时长）
        fmt_dur = (data.get("format") or {}).get("duration")
        duration = float(fmt_dur if fmt_dur else (s.get("duration") or 0))
        return {
            "codec": s.get("codec_name"),
            "width": s.get("width"),
            "height": s.get("height"),
            "duration": duration,
            "nb_frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None,
        }
    except Exception:
        return None


def verify_manifest(
    name: str,
    root: Path,
    manifest_rel: str,
    expect_glb: bool,
    ck: Checker,
    probe_workers: int = 24,
) -> Optional[Dict]:
    ck.add(f"{name}/root", root.is_dir(), str(root))
    if not root.is_dir():
        return None
    manifest_path = root / manifest_rel
    ok = ck.add(f"{name}/manifest", manifest_path.is_file(), str(manifest_path))
    if not ok:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ck.add(f"{name}/manifest_count", len(manifest) > 0, f"{len(manifest)} entries")

    missing, size_bad, hash_bad = [], [], []
    total = 0
    for entry in manifest:
        path = root / entry["file"]
        if not path.is_file():
            missing.append(entry["file"])
            continue
        actual = path.stat().st_size
        total += actual
        if actual != entry["bytes"]:
            size_bad.append(f"{entry['file']} {actual}!={entry['bytes']}")
            continue
        if sha256_file(path) != entry["sha256"]:
            hash_bad.append(entry["file"])

    ck.add(f"{name}/files_present", not missing, f"missing={len(missing)} {missing[:5]}")
    ck.add(f"{name}/size_match", not size_bad, f"mismatch={len(size_bad)} {size_bad[:5]}")
    ck.add(f"{name}/sha256", not hash_bad, f"bad={len(hash_bad)} {hash_bad[:5]}")
    ck.add(f"{name}/budget", total <= BUDGET_BYTES, f"{total/1e9:.3f} GB <= 2.000 GB")

    if expect_glb:
        glbs = [e for e in manifest if e["file"].endswith(".glb")]
        bad = []
        for entry in glbs:
            path = root / entry["file"]
            if not path.is_file():
                bad.append(entry["file"])
                continue
            with open(path, "rb") as fh:
                magic = fh.read(4)
            if magic != b"glTF":
                bad.append(f"{entry['file']} magic={magic!r}")
        ck.add(f"{name}/glb_magic", bool(glbs) and not bad, f"glbs={len(glbs)} bad={bad}")

    videos = [e for e in manifest if e["file"].endswith(".mp4")]
    with ThreadPoolExecutor(max_workers=probe_workers) as pool:
        probes = list(pool.map(lambda e: ffprobe(root / e["file"]), videos))
    undecodable = [e["file"] for e, p in zip(videos, probes) if p is None]
    ck.add(f"{name}/decodable", not undecodable, f"videos={len(videos)} undecodable={len(undecodable)} {undecodable[:5]}")

    durations = [p["duration"] for p in probes if p and p["duration"]]
    resolutions = sorted({f"{p['width']}x{p['height']}" for p in probes if p})
    stats = {
        "files": len(manifest),
        "videos": len(videos),
        "bytes": total,
        "duration_s_total": round(sum(durations), 1),
        "duration_s_mean": round(sum(durations) / len(durations), 2) if durations else 0,
        "resolutions": resolutions,
        "glb_count": len([e for e in manifest if e["file"].endswith(".glb")]),
    }
    ck.add(f"{name}/duration", bool(durations) and stats["duration_s_total"] > 0, f"total={stats['duration_s_total']}s")
    return stats | {"manifest": manifest}


def check_mammalnet(root: Path, ck: Checker) -> Dict:
    ds = root / "mammalnet_canine"
    stats = verify_manifest("mammalnet", ds, "manifest.json", expect_glb=False, ck=ck) or {}
    manifest = stats.pop("manifest", [])

    ann_path = ds / "annotations" / "detection_annotations.json"
    ok = ck.add("mammalnet/annotations_json", ann_path.is_file(), str(ann_path))
    if ok:
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        ck.add("mammalnet/annotation_count", len(ann) == 18346, f"{len(ann)} videos (expect 18346)")
        domestic = [
            v
            for v in ann.values()
            if any(t.get("genus") == "canis" and t.get("keyword") not in ("wolf", "wolfdog") for t in v.get("taxnomy", []))
        ]
        ck.add("mammalnet/domestic_filter", len(domestic) == 1129, f"{len(domestic)} domestic clips (expect 1129)")

    tax = ds / "annotations" / "taxonomy.txt"
    if ck.add("mammalnet/taxonomy", tax.is_file(), str(tax)):
        text = tax.read_text(encoding="utf-8")
        canidae = [ln for ln in text.splitlines() if ln.lower().split()[-1] == "canidae" or "canidae" in ln.lower()]
        ck.add("mammalnet/canidae_rows", len(canidae) >= 8, f"{len(canidae)} canidae rows")

    if manifest:
        breeds = {m["keyword"] for m in manifest}
        splits = {m["subset"] for m in manifest}
        ck.add("mammalnet/breed_coverage", EXPECTED_BREEDS <= breeds, f"{sorted(breeds)}")
        ck.add("mammalnet/split_coverage", EXPECTED_SPLITS <= splits, f"{sorted(splits)}")
        bad_genus = [m["file"] for m in manifest if not any(t.get("genus") == "canis" for t in (m.get("taxonomy") or []))]
        ck.add("mammalnet/genus_canis", not bad_genus, f"non-canis={len(bad_genus)}")
        wild = [m["file"] for m in manifest if m["keyword"] in ("wolf", "wolfdog")]
        ck.add("mammalnet/no_wild", not wild, f"wild={len(wild)}")
        unprobed = [m["file"] for m in manifest if not m.get("clip")]
        ck.add("mammalnet/clip_probe", not unprobed, f"missing clip metadata={len(unprobed)}")
        sub = {}
        for m in manifest:
            sub.setdefault(m["subset"], {"files": 0, "bytes": 0, "clips": set(), "duration_s": 0.0})
            sub[m["subset"]]["files"] += 1
            sub[m["subset"]]["bytes"] += m["bytes"]
            sub[m["subset"]]["clips"].add(m["video_id"])
            sub[m["subset"]]["duration_s"] += (m.get("clip") or {}).get("duration_s", 0) or 0
        stats["by_split"] = {
            k: {"files": v["files"], "clips": len(v["clips"]), "bytes": v["bytes"], "duration_s": round(v["duration_s"], 1)}
            for k, v in sorted(sub.items())
        }
        stats["by_breed"] = {b: sum(1 for m in manifest if m["keyword"] == b) for b in sorted(breeds)}
        stats["clips"] = len({m["video_id"] for m in manifest})
        stats["clip_duration_s_total"] = round(sum((m.get("clip") or {}).get("duration_s", 0) or 0 for m in manifest), 1)
    return stats


def check_dogmo(root: Path, ck: Checker) -> Dict:
    ds = root / "dogmo_preview"
    stats = verify_manifest("dogmo", ds, "manifest.json", expect_glb=True, ck=ck) or {}
    manifest = stats.pop("manifest", [])
    if manifest:
        kinds = {}
        for m in manifest:
            kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
        ck.add("dogmo/preview_kinds", kinds.get("dataset", 0) >= 20 and kinds.get("results", 0) >= 7, f"{kinds}")
        # 预览素材不得超过 2 GB，且不应被当作训练集：数量远小于 1.2k clips
        ck.add("dogmo/preview_only", len(manifest) < 100, f"{len(manifest)} assets (official corpus is 1.2k clips)")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="校验 P1 数据补充产出物")
    ap.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    ap.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "p1-dataset-expansion-verify.json",
    )
    args = ap.parse_args(argv)

    ck = Checker()
    t0 = time.time()
    if not shutil.which("ffprobe"):
        ck.add("env/ffprobe", False, "ffprobe not on PATH", hard=False)

    summary = {
        "datasets_root": str(args.datasets_root),
        "mammalnet": check_mammalnet(args.datasets_root, ck),
        "dogmo": check_dogmo(args.datasets_root, ck),
        "checks": ck.checks,
        "elapsed_s": round(time.time() - t0, 1),
        "passed": not ck.failed,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'ALL CHECKS PASSED' if summary['passed'] else 'FAILED: ' + str(len(ck.failed))} -> {args.json}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
