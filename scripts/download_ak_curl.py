"""Download Animal Kingdom video.tar.gz via curl with resume support.

gdown successfully resolves the Google Drive download URL but the proxy drops
the connection during transfer. curl has better proxy/retry/resume support.

Strategy:
  1. Use Google Drive's direct download URL with confirm token bypass
  2. curl with --retry, --continue-at, --proxy
  3. Loop until file size matches expected ~15.6 GB
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PROXY = "http://127.0.0.1:7897"

# File IDs discovered from Google Drive folder listing
FILES = {
    "video.tar.gz": {
        "id": "1X4rL5ey7M1_YM4GDa1DvvVdHoUfuHeJp",
        "path": "data/animal_kingdom/action_recognition/dataset/video.tar.gz",
        "expected_gb": 15.6,
    },
    "image.tar.gz": {
        "id": "1kcujjY81xwhhM9MVAamnrqo-ZxMK-W8n",
        "path": "data/animal_kingdom/action_recognition/dataset/image.tar.gz",
        "expected_gb": 42.1,
    },
}


def download_with_curl(file_id: str, output_path: Path, expected_gb: float,
                       max_retries: int = 24) -> bool:
    """Download a Google Drive file using curl with resume support.

    For large files, Google Drive requires a confirmation token.
    We use the new endpoint: drive.usercontent.google.com/download

    Includes quota detection: if Google Drive returns an HTML error page
    instead of the file, waits 1 hour before retrying (up to 24 times = 24h).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Google Drive direct download URL with confirm bypass for large files
    # The &confirm=t parameter bypasses the "virus scan" confirmation page
    download_url = (
        f"https://drive.usercontent.google.com/download?"
        f"id={file_id}&export=download&confirm=t"
    )

    QUOTA_MARKERS = [b"Quota exceeded", b"Too many users",
                     b"quota exceeded", b"too many users"]

    def is_quota_error(path: Path) -> bool:
        """Check if downloaded file is actually a quota error HTML page."""
        if not path.exists() or path.stat().st_size > 10240:
            return False
        try:
            with open(path, "rb") as f:
                content = f.read()
            return any(m in content for m in QUOTA_MARKERS)
        except Exception:
            return False

    for attempt in range(max_retries):
        # Check current file size for resume
        current_size = 0
        if output_path.exists():
            # If file is a quota error page, delete it
            if is_quota_error(output_path):
                print(f"[attempt {attempt+1}/{max_retries}] Previous file was "
                      f"quota error page, deleting", flush=True)
                output_path.unlink()
            else:
                current_size = output_path.stat().st_size
                current_gb = current_size / (1024**3)
                print(f"[attempt {attempt+1}/{max_retries}] Resume from "
                      f"{current_gb:.2f} GB", flush=True)
                # Check if download is complete
                if current_gb >= expected_gb * 0.95:
                    print(f"[OK] File size {current_gb:.2f} GB >= expected "
                          f"{expected_gb} GB", flush=True)
                    return True
        else:
            print(f"[attempt {attempt+1}/{max_retries}] Starting fresh download",
                  flush=True)

        # Build curl command
        cmd = [
            "curl",
            "-L",                           # follow redirects
            "--proxy", PROXY,               # use proxy
            "--retry", "5",                 # retry on transient errors
            "--retry-delay", "10",          # wait 10s between retries
            "--retry-all-errors",           # retry on all error types
            "--connect-timeout", "30",      # connection timeout
            "--max-time", "7200",           # max 2 hours per attempt
            "-C", "-",                      # resume download (continue at)
            "-o", str(output_path),         # output file
            "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            "-H", "Accept: text/html,application/xhtml+xml,application/xml,"
                  "*/*;q=0.8",
            "-H", "Accept-Language: en-US,en;q=0.5",
            "-H", "Connection: keep-alive",
            download_url,
        ]

        print(f"  URL: {download_url[:80]}...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode == 0:
            if output_path.exists():
                # Check if we got a quota error page
                if is_quota_error(output_path):
                    print(f"  [QUOTA] Got quota error page, will retry in 1h",
                          flush=True)
                    output_path.unlink()
                else:
                    size_gb = output_path.stat().st_size / (1024**3)
                    print(f"  [OK] Downloaded {size_gb:.2f} GB", flush=True)
                    if size_gb >= expected_gb * 0.95:
                        return True
                    else:
                        print(f"  [WARN] Size {size_gb:.2f} GB < expected "
                              f"{expected_gb} GB, retrying...", flush=True)
        else:
            print(f"  [curl exit {proc.returncode}]", flush=True)
            if proc.stderr:
                # Show last few lines of stderr
                err_lines = proc.stderr.strip().split("\n")
                for line in err_lines[-3:]:
                    print(f"    {line}", flush=True)

        # Wait before retry
        # If quota error: wait 1 hour (3600s)
        # Otherwise: exponential backoff, max 300s
        if output_path.exists() and is_quota_error(output_path):
            wait = 3600  # 1 hour for quota errors
            print(f"  [QUOTA] Waiting {wait}s (1h) before retry "
                  f"(attempt {attempt+1}/{max_retries})...", flush=True)
        elif not output_path.exists():
            # File was deleted (quota error), wait 1 hour
            wait = 3600
            print(f"  [QUOTA] Waiting {wait}s (1h) before retry "
                  f"(attempt {attempt+1}/{max_retries})...", flush=True)
        else:
            wait = min(10 * (attempt + 1), 300)
            print(f"  Waiting {wait}s before retry...", flush=True)
        time.sleep(wait)

    return False


def main() -> int:
    print("=== Animal Kingdom Dataset Downloader (curl) ===", flush=True)
    print(f"Proxy: {PROXY}", flush=True)

    # Allow selecting which file to download via command-line arg
    target = sys.argv[1] if len(sys.argv) > 1 else "video.tar.gz"

    if target not in FILES:
        print(f"[ERROR] Unknown file: {target}", flush=True)
        print(f"Available: {', '.join(FILES.keys())}", flush=True)
        return 1

    info = FILES[target]
    output_path = Path(info["path"])
    print(f"Target: {target}", flush=True)
    print(f"  File ID: {info['id']}", flush=True)
    print(f"  Output: {output_path.resolve()}", flush=True)
    print(f"  Expected size: ~{info['expected_gb']} GB", flush=True)

    success = download_with_curl(info["id"], output_path, info["expected_gb"])

    if success:
        size_gb = output_path.stat().st_size / (1024**3)
        print(f"\n=== SUCCESS ===", flush=True)
        print(f"File: {output_path}", flush=True)
        print(f"Size: {size_gb:.2f} GB", flush=True)
        return 0
    else:
        print(f"\n=== FAILED after retries ===", flush=True)
        if output_path.exists():
            size_gb = output_path.stat().st_size / (1024**3)
            print(f"Partial download: {size_gb:.2f} GB "
                  f"(expected ~{info['expected_gb']} GB)", flush=True)
            print(f"Re-run this script to resume download.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
