"""Download YouTube dog training videos for 1.2f supplementation (Path B).

Targets:
  - Basic behaviors: sit / down / stand / come (for 1.2f accuracy validation)
  - Play ball: dog + ball interaction (for 1.6d object detection)

Usage:
    python scripts/download_youtube_videos.py [--max N] [--list]
    python scripts/download_youtube_videos.py --download
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("data/youtube_self_label")
RAW = ROOT / "raw"
META = ROOT / "metadata.json"

# Search queries targeting specific behaviors
# Each query maps to project behaviors we need to validate
SEARCH_QUERIES = [
    {
        "query": "dog training sit down stand come obedience",
        "behaviors": ["sit", "down", "stand", "come"],
        "purpose": "1.2f basic behavior accuracy validation",
        "max_results": 5,
    },
    {
        "query": "dog playing fetch ball training",
        "behaviors": ["fetch", "search"],
        "purpose": "1.6d object detection + 1.2f fetch behavior",
        "max_results": 3,
    },
    {
        "query": "german shepherd training commands sit down",
        "behaviors": ["sit", "down", "heel"],
        "purpose": "1.2f working dog breed behavior validation",
        "max_results": 3,
    },
]


def search_videos(query: str, max_results: int) -> list[dict]:
    """Use yt-dlp to search YouTube, return video metadata without downloading."""
    print(f"  [search] {query} (max {max_results})", flush=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        f"ytsearch{max_results}:{query}",
        "--dump-json",
        "--flat-playlist",
        "--no-warnings",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            v = json.loads(line)
            videos.append({
                "id": v.get("id", ""),
                "title": v.get("title", ""),
                "url": v.get("url", ""),
                "duration": v.get("duration"),
                "view_count": v.get("view_count"),
                "uploader": v.get("uploader", ""),
            })
        except json.JSONDecodeError:
            continue
    print(f"  [search] found {len(videos)} results", flush=True)
    return videos


def download_video(video_id: str, output_dir: Path, max_duration_sec: int = 300) -> dict:
    """Download a single video, capped at max_duration_sec.
    Uses best mp4 format, skips if too long.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "best[ext=mp4][height<=720]/best[height<=720]/best",
        "--max-filesize", "200M",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-warnings",
        "--no-playlist",
        "--write-info-json",
        url,
    ]
    print(f"  [dl] {video_id}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    ok = result.returncode == 0

    # Find downloaded file
    video_file = None
    info_file = None
    for f in output_dir.glob(f"{video_id}.*"):
        if f.suffix == ".mp4":
            video_file = f
        elif f.suffix == ".json":
            info_file = f

    return {
        "id": video_id,
        "url": url,
        "downloaded": ok and video_file is not None,
        "video_path": str(video_file) if video_file else None,
        "info_path": str(info_file) if info_file else None,
        "size_mb": round(video_file.stat().st_size / 1024 / 1024, 2) if video_file and video_file.exists() else 0,
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="search only, list results")
    parser.add_argument("--download", action="store_true", help="search + download")
    parser.add_argument("--max", type=int, default=8, help="max videos to download")
    args = parser.parse_args()

    if not args.list and not args.download:
        args.list = True  # default: list only

    RAW.mkdir(parents=True, exist_ok=True)

    all_results = []
    print("=== YouTube dog training video search ===", flush=True)

    for sq in SEARCH_QUERIES:
        print(f"\n[query] {sq['purpose']}", flush=True)
        videos = search_videos(sq["query"], sq["max_results"])
        for v in videos:
            v["query"] = sq["query"]
            v["target_behaviors"] = sq["behaviors"]
            v["purpose"] = sq["purpose"]
            all_results.append(v)

    # Save search results
    META.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[search] {len(all_results)} videos found, metadata: {META}", flush=True)

    # Print summary
    print("\n=== Search results ===", flush=True)
    for v in all_results:
        dur = f"{v['duration']}s" if v.get("duration") else "?"
        views = v.get("view_count") or 0
        print(f"  {v['id']:12s}  {dur:>6s}  views={views:>8d}  {v['title'][:60]}", flush=True)

    if args.download:
        # Filter: prefer videos with duration 20-300s, skip None duration
        candidates = [v for v in all_results if v.get("duration") and 20 <= v["duration"] <= 300]
        # Deduplicate by id
        seen = set()
        unique = []
        for v in candidates:
            if v["id"] not in seen:
                seen.add(v["id"])
                unique.append(v)
        to_download = unique[:args.max]
        print(f"\n=== Downloading {len(to_download)} videos ===", flush=True)

        dl_results = []
        for v in to_download:
            r = download_video(v["id"], RAW)
            r["title"] = v["title"]
            r["target_behaviors"] = v["target_behaviors"]
            r["purpose"] = v["purpose"]
            dl_results.append(r)
            status = "OK" if r["downloaded"] else "FAIL"
            print(f"  [{status}] {r['id']}  {r['size_mb']}MB  {v['title'][:50]}", flush=True)

        # Save download manifest
        manifest = ROOT / "download_manifest.json"
        manifest.write_text(json.dumps(dl_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== Manifest: {manifest} ===", flush=True)
        ok = sum(1 for r in dl_results if r["downloaded"])
        print(f"    {ok}/{len(dl_results)} downloaded successfully", flush=True)


if __name__ == "__main__":
    main()
