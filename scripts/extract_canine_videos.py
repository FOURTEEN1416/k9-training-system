"""Extract canine videos from Animal Kingdom video.tar.gz.

Reads AR_metadata.xlsx to find canine-related videos, then extracts only
those videos from the tar.gz archive (instead of extracting all 30100 videos).

Canine species per ADR 0008 v1.4:
  Wolf, Dog, Wild Dog, African Painted Dog, Dingo Dog

Output: data/animal_kingdom/action_recognition/dataset/video/<VIDEO_ID>.mp4
"""
from __future__ import annotations

import ast
import json
import tarfile
from pathlib import Path

import pandas as pd

ARCHIVE = Path("data/animal_kingdom/action_recognition/dataset/video.tar.gz")
METADATA = Path("data/animal_kingdom/action_recognition/AR_metadata.xlsx")
OUTPUT_DIR = Path("data/animal_kingdom/action_recognition/dataset/video")
REPORT = Path("reports/phase-2-ak-canine-videos.json")

# Canine species (exact match, case-sensitive per metadata)
CANINE_SPECIES = {
    "Wolf", "Dog", "Wild Dog", "African Painted Dog", "Dingo Dog",
}


def parse_list(val):
    """Parse a string representation of a Python list."""
    if not isinstance(val, str):
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def find_canine_videos() -> list[dict]:
    """Find canine videos from AR metadata.

    Returns list of dicts with video_id, species, actions, labels.
    """
    df = pd.read_excel(METADATA, sheet_name="AR")
    print(f"Total videos in metadata: {len(df)}")

    canine_records = []
    for _, row in df.iterrows():
        animals = parse_list(row.get("list_animal", "[]"))
        if not animals:
            continue
        # Check if any animal in this video is a canine species
        video_species = set(animals) & CANINE_SPECIES
        if not video_species:
            continue

        # Parse actions
        actions_raw = parse_list(row.get("list_animal_action", "[]"))
        actions = []
        for item in actions_raw:
            if isinstance(item, tuple) and len(item) >= 2:
                animal, action = item[0], item[1]
                if animal in CANINE_SPECIES:
                    actions.append({"animal": animal, "action": action})

        canine_records.append({
            "video_id": row["video_id"],
            "type": row.get("type", ""),
            "labels": str(row.get("labels", "")),
            "species": sorted(video_species),
            "all_animals": list(animals),
            "canine_actions": actions,
        })

    print(f"Canine videos found: {len(canine_records)}")

    # Summary by species
    species_count: dict[str, int] = {}
    for rec in canine_records:
        for sp in rec["species"]:
            species_count[sp] = species_count.get(sp, 0) + 1
    print("By species:")
    for sp, cnt in sorted(species_count.items(), key=lambda x: -x[1]):
        print(f"  {sp}: {cnt}")

    # Summary by action
    action_count: dict[str, int] = {}
    for rec in canine_records:
        for act in rec["canine_actions"]:
            action_count[act["action"]] = action_count.get(act["action"], 0) + 1
    print(f"\nCanine actions: {len(action_count)} types")
    for act, cnt in sorted(action_count.items(), key=lambda x: -x[1])[:15]:
        print(f"  {act}: {cnt}")

    return canine_records


def extract_canine_videos(canine_records: list[dict]) -> int:
    """Extract only canine videos from the tar.gz archive.

    Uses single-pass iteration over tar members to avoid repeated
    decompression of the gzip stream (which is expensive for 15GB archives).
    """
    # Build set of video IDs to extract (filename: video/<VIDEO_ID>.mp4)
    video_ids = {rec["video_id"] for rec in canine_records}
    print(f"\nExtracting {len(video_ids)} canine videos from archive...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check which videos already exist
    existing = set()
    for vid in video_ids:
        if (OUTPUT_DIR / f"{vid}.mp4").exists():
            existing.add(vid)
    if existing:
        print(f"  {len(existing)} videos already extracted, skipping")

    to_extract = video_ids - existing
    if not to_extract:
        print("  All videos already extracted")
        return len(existing)

    # Build set of member names to extract
    target_names = {f"video/{vid}.mp4" for vid in to_extract}
    print(f"  Target members: {len(target_names)}")

    extracted = 0
    not_found_ids = set(to_extract)

    # Single-pass extraction: iterate tar stream once, extract matching members
    # This avoids re-reading the 15GB gzip stream for each file
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        for member in tar:
            if member.name in target_names:
                tar.extract(member, path=OUTPUT_DIR.parent)
                extracted += 1
                not_found_ids.discard(member.name.replace("video/", "").replace(".mp4", ""))
                if extracted % 20 == 0:
                    print(f"  Extracted {extracted}/{len(to_extract)}...", flush=True)
                if extracted >= len(to_extract):
                    break

    not_found = len(not_found_ids)
    if not_found > 0 and not_found <= 10:
        print(f"  [WARN] Not found: {sorted(not_found_ids)}")

    print(f"\nExtraction complete:")
    print(f"  Extracted: {extracted}")
    print(f"  Already existed: {len(existing)}")
    print(f"  Not found in archive: {not_found}")
    print(f"  Total canine videos: {extracted + len(existing)}")

    return extracted + len(existing)


def main() -> int:
    print("=== Animal Kingdom Canine Video Extractor ===\n")

    # Step 1: Find canine videos from metadata
    canine_records = find_canine_videos()

    if not canine_records:
        print("[ERROR] No canine videos found in metadata")
        return 1

    # Step 2: Extract canine videos from archive
    total = extract_canine_videos(canine_records)

    # Step 3: Save report
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total_canine_videos": len(canine_records),
        "extracted_videos": total,
        "species_count": {},
        "action_count": {},
        "videos": canine_records,
    }
    for rec in canine_records:
        for sp in rec["species"]:
            report["species_count"][sp] = report["species_count"].get(sp, 0) + 1
        for act in rec["canine_actions"]:
            report["action_count"][act["action"]] = \
                report["action_count"].get(act["action"], 0) + 1

    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"\nReport saved to: {REPORT}")

    # Step 4: Verify extraction
    print("\n=== Verification ===")
    video_files = list(OUTPUT_DIR.glob("*.mp4"))
    print(f"Total mp4 files in {OUTPUT_DIR}: {len(video_files)}")
    if video_files:
        sizes = [f.stat().st_size for f in video_files]
        print(f"  Size range: {min(sizes)/1024:.1f} KB - {max(sizes)/1024:.1f} KB")
        print(f"  Total size: {sum(sizes)/1024/1024:.1f} MB")
        print(f"  Sample files:")
        for f in video_files[:5]:
            print(f"    {f.name} ({f.stat().st_size/1024:.1f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
