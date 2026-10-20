"""Re-verify the Animal-CLIP zero-shot deliverable (P1).

exit 0 = every artefact listed below exists, parses, and is internally consistent.
Anything else -> non-zero with a precise reason on stderr.

Checks (each one is printed so the log itself is the evidence):
  1.  scripts/eval_animalclip_zeroshot.py exists and is syntactically valid (py_compile).
  2.  external/animal-clip/ is a real clone at the pinned commit (README.md / models/model_builder.py present).
  3.  external/animal-clip/_k9_stubs.py exists and, when installed, lets
      `import clip` + `models.model_builder` succeed in a fresh interpreter.
  4.  reports/p1-animalclip-zeroshot-2026-10-20.json parses and matches schema k9.animalclip.zeroshot.v1.
  5.  The JSON has one record per manifest row, and each record has a decode block.
  6.  No record claims an Animal-CLIP prediction unless a real weight file was present
      (guard against fabricated numbers).
  7.  reports/p1-animalclip-zeroshot-2026-10-20.md exists and its headline numbers agree
      with the JSON summary (counts + top-1 share for the CLIP baseline).
  8.  The CLIP backbone weight is present at the expected byte size.

Usage:
    python scripts/verify_animclip_eval.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "eval_animalclip_zeroshot.py"
VERIFY = ROOT / "scripts" / "verify_animclip_eval.py"
AC = ROOT / "external" / "animal-clip"
JSON_OUT = ROOT / "reports" / "p1-animalclip-zeroshot-2026-10-20.json"
MD_OUT = ROOT / "reports" / "p1-animalclip-zeroshot-2026-10-20.md"
MANIFEST = Path("D:/Desktop/PR论文/psd-framework/runs/data_campaign/video_gap_w42/manifest.jsonl")
BACKBONE = AC / "weights" / "ViT-B-16.pt"
BACKBONE_BYTES = 350_837_078
PINNED_COMMIT = "4cff5a8b36395b8a5473b4fe7763fdb6486c1d65"

FAILS: list[str] = []
PASSES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSES.append(name)
        print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAILS.append(f"{name}: {detail}")
        print(f"[FAIL] {name} — {detail}")


def main() -> int:
    print("=" * 78)
    print("verify_animclip_eval — P1 Animal-CLIP zero-shot deliverable")
    print("=" * 78)

    # 1 ---------------------------------------------------------------- script
    check("eval script exists", EVAL.is_file(), str(EVAL))
    if EVAL.is_file():
        r = subprocess.run([sys.executable, "-m", "py_compile", str(EVAL)],
                           capture_output=True, text=True)
        check("eval script compiles", r.returncode == 0, r.stderr.strip()[-400:])

    # 2 ------------------------------------------------------------- vendored
    check("vendored README.md", (AC / "README.md").is_file(), str(AC / "README.md"))
    check("vendored models/model_builder.py",
          (AC / "models" / "model_builder.py").is_file(),
          str(AC / "models" / "model_builder.py"))
    check("vendored data_files/MammalNet",
          (AC / "data_files" / "MammalNet" / "behavior_to_id.csv").is_file(), "")
    git_head = AC / ".git" / "HEAD"
    if git_head.exists():
        r = subprocess.run(["git", "-C", str(AC), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        head = r.stdout.strip()
        check("vendored repo at pinned commit", head == PINNED_COMMIT,
              f"HEAD={head or r.stderr.strip()}")
    else:
        check("vendored repo at pinned commit", False, "no .git/HEAD")

    # 3 ---------------------------------------------------------------- stubs
    check("_k9_stubs.py present", (AC / "_k9_stubs.py").is_file(), str(AC / "_k9_stubs.py"))
    probe = (
        "import sys;"
        f"sys.path.insert(0, r'{AC / '_vendor'}');"
        f"sys.path.insert(0, r'{AC}');"
        "import _k9_stubs; _k9_stubs.install(); _k9_stubs.prepare_sys_path();"
        "import clip; from models import model_builder;"
        "print('IMPORT_OK', hasattr(model_builder, 'build_model'))"
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    check("clip + model_builder import under stubs",
          r.returncode == 0 and "IMPORT_OK True" in r.stdout,
          (r.stdout + r.stderr).strip()[-500:])

    # 4 ---------------------------------------------------------------- json
    if not JSON_OUT.is_file():
        check("JSON report exists", False, str(JSON_OUT))
        return _finish()
    data = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    check("JSON parses + schema", data.get("schema") == "k9.animalclip.zeroshot.v1",
          f"schema={data.get('schema')!r}")
    check("JSON has status", data.get("status") in {"ok", "partial"},
          f"status={data.get('status')!r}")
    if data.get("status") == "partial":
        check("partial carries a non-empty reason",
              bool((data.get("status_reason") or "").strip()),
              (data.get("status_reason") or "")[:160])

    # 5 ------------------------------------------------------- manifest parity
    if MANIFEST.is_file():
        expected_ids = [
            json.loads(l)["fragment_id"]
            for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        got_ids = [r["fragment_id"] for r in data.get("results", [])]
        check("one record per manifest row",
              got_ids == expected_ids,
              f"json={len(got_ids)} manifest={len(expected_ids)}")
    else:
        check("manifest reachable", False, f"missing {MANIFEST}")

    results = data.get("results", [])
    check("every record has decode info",
          all(r.get("decode") for r in results),
          f"{sum(1 for r in results if not r.get('decode'))} without decode")
    check("decode_errors == 0",
          data["summary"]["decode_errors"] == 0,
          f"decode_errors={data['summary']['decode_errors']}")

    # 6 ------------------------------------------------- no fabricated claims
    weights_status = data["run"]["weights"]["status"]
    has_ac = [r for r in results if r.get("animalclip")]
    if weights_status != "present":
        check("no Animal-CLIP predictions without weights",
              len(has_ac) == 0,
              f"{len(has_ac)} records claim animalclip but weights.status={weights_status}")
    else:
        check("Animal-CLIP predictions present when weights exist", len(has_ac) == len(results),
              f"{len(has_ac)}/{len(results)}")

    # 7 --------------------------------------------------- markdown vs json
    if not MD_OUT.is_file():
        check("Markdown report exists", False, str(MD_OUT))
    else:
        md = MD_OUT.read_text(encoding="utf-8")
        check("Markdown exists and non-trivial", len(md) > 1200, f"{len(md)} chars")
        n = data["n_results"]
        check("Markdown states the run size", f"{n}" in md, f"expected '{n}' in md")
        for key in ("clip_baseline", "animalclip"):
            check(f"Markdown names the JSON key {key!r}", key in md, "")
        # Every top-1 share in the JSON must appear in the md as a percentage that
        # agrees to 0.01 pp. This catches a stale/hand-edited report.
        for space in ("native", "dogcentric"):
            blk = data["summary"].get("clip_baseline", {}).get(space, {})
            for label, share in (blk.get("top1_share") or {}).items():
                want = share * 100.0
                # the md writes "34.13 %" / "34.13%" — accept both spacings
                found = _pct_near(md, label, want, tol=0.011)
                check(f"md↔json share [{space}] {label!r}", found,
                      f"expected {want:.2f}% near label; json={share}")
        check("Markdown disclaims accuracy (no GT)",
              ("无 GT" in md) or ("无 ground truth" in md.lower()) or ("no ground truth" in md.lower()),
              "markdown must state there is no ground truth")

    # 8 ------------------------------------------------------------- backbone
    if BACKBONE.is_file():
        size = BACKBONE.stat().st_size
        check("CLIP backbone at expected size", size == BACKBONE_BYTES,
              f"{size} vs {BACKBONE_BYTES}")
    else:
        check("CLIP backbone present", False, str(BACKBONE))

    return _finish()


def _pct_near(md: str, label: str, want: float, tol: float = 0.011) -> bool:
    """True if some line containing `label` also carries a percentage within `tol` of `want`."""
    import re

    for line in md.splitlines():
        if label not in line:
            continue
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", line):
            if abs(float(m.group(1)) - want) <= tol:
                return True
    return False


def _finish() -> int:
    print("-" * 78)
    print(f"PASSED {len(PASSES)}  FAILED {len(FAILS)}")
    if FAILS:
        for f in FAILS:
            print(f"  [FAIL] {f}", file=sys.stderr)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
