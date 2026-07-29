"""Deploy Label Studio for dog behavior annotation (Path B).

Starts Label Studio server locally, creates project with labeling config,
imports pre-annotated tasks from data/youtube_self_label/label_studio/.

Usage:
    python scripts/deploy_label_studio.py start    # start server + setup project
    python scripts/deploy_label_studio.py status    # check status
    python scripts/deploy_label_studio.py import    # import tasks only (server must be running)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path("data/youtube_self_label")
LS_DIR = ROOT / "label_studio"
DATA_DIR = ROOT / "ls_data"
CONFIG_PATH = LS_DIR / "label_config.xml"
TASKS_PATH = LS_DIR / "tasks_all.json"

LS_HOST = "http://localhost:8080"
LS_DATA_DIR = str(DATA_DIR)


def start_server():
    """Start Label Studio server in background."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[LS] Starting Label Studio on {LS_HOST} ...", flush=True)
    print(f"[LS] Data dir: {LS_DATA_DIR}", flush=True)

    # Start label-studio as background process
    cmd = [
        sys.executable, "-m", "label_studio",
        "start",
        "--port", "8080",
        "--data-dir", LS_DATA_DIR,
        "--host", "0.0.0.0",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )

    # Wait for server to be ready
    print("[LS] Waiting for server to start ...", flush=True)
    for i in range(60):  # 60s timeout
        try:
            r = requests.get(f"{LS_HOST}/health", timeout=2)
            if r.status_code == 200:
                print(f"[LS] Server ready (after {i+1}s)", flush=True)
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(1)

    print("[LS] Server failed to start within 60s", flush=True)
    # Print last output
    proc.terminate()
    return None


def get_token() -> str:
    """Get or create API token. First-run requires account setup via UI."""
    # Label Studio stores token in user data
    token_file = DATA_DIR / "api_token.txt"
    if token_file.exists():
        return token_file.read_text().strip()

    print("[LS] No API token found.", flush=True)
    print("[LS] Please:", flush=True)
    print("  1. Open http://localhost:8080 in browser", flush=True)
    print("  2. Create admin account (email + password)", flush=True)
    print("  3. Go to Account & Settings → Access Token", flush=True)
    print("  4. Copy token and paste here:", flush=True)
    token = input().strip()
    token_file.write_text(token, encoding="utf-8")
    return token


def create_project(token: str) -> str:
    """Create labeling project, return project ID."""
    config = CONFIG_PATH.read_text(encoding="utf-8")
    headers = {"Authorization": f"Token {token}"}
    payload = {
        "title": "K9 行为标注 — YouTube 自标（Phase 2.0a 路径 B）",
        "description": (
            "YOLO26-pose 预标注已生成。请：\n"
            "1. 修正视频中的犬只检测框\n"
            "2. 标注每个时间段的行为类别（sit/down/stand/come/heel/fetch 等）\n"
            "3. 备注遮挡/多犬/异常情况\n"
            "数据用途: 1.2f 真实视频准确率验证 + 1.6d 物体检测补充"
        ),
        "label_config": config,
    }
    r = requests.post(f"{LS_HOST}/api/projects/", json=payload, headers=headers, timeout=10)
    r.raise_for_status()
    project_id = str(r.json()["id"])
    print(f"[LS] Project created: id={project_id}", flush=True)
    return project_id


def import_tasks(token: str, project_id: str):
    """Import pre-annotated tasks into project."""
    if not TASKS_PATH.exists():
        print(f"[LS] No tasks file: {TASKS_PATH}", flush=True)
        return

    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    print(f"[LS] Importing {len(tasks)} tasks ...", flush=True)
    headers = {"Authorization": f"Token {token}"}

    # Use bulk import endpoint
    r = requests.post(
        f"{LS_HOST}/api/projects/{project_id}/import",
        json=tasks,
        headers=headers,
        timeout=30,
    )
    r.raise_for_status()
    result = r.json()
    task_ids = result.get("task_ids", [])
    print(f"[LS] Imported {len(task_ids)} tasks", flush=True)
    return task_ids


def upload_videos(token: str, project_id: str):
    """Upload video files to Label Studio storage."""
    raw_dir = ROOT / "raw"
    headers = {"Authorization": f"Token {token}"}
    for video in sorted(raw_dir.glob("*.mp4")):
        print(f"[LS] Uploading {video.name} ...", flush=True)
        with open(video, "rb") as f:
            r = requests.post(
                f"{LS_HOST}/api/projects/{project_id}/upload",
                files={"file": (video.name, f, "video/mp4")},
                headers=headers,
                timeout=120,
            )
        r.raise_for_status()
        print(f"    OK: {r.json()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "status", "import"])
    args = parser.parse_args()

    if args.action == "status":
        try:
            r = requests.get(f"{LS_HOST}/health", timeout=2)
            print(f"[LS] Server status: {r.status_code} {'UP' if r.status_code == 200 else 'DOWN'}")
        except requests.ConnectionError:
            print("[LS] Server not running")
        return

    if args.action == "start":
        proc = start_server()
        if proc is None:
            sys.exit(1)
        print("\n[LS] Server is running. Next steps:", flush=True)
        print("  1. Open http://localhost:8080", flush=True)
        print("  2. Create admin account", flush=True)
        print("  3. Run: python scripts/deploy_label_studio.py import", flush=True)
        print("\n[LS] Server PID:", proc.pid, flush=True)
        print("[LS] Press Ctrl+C to stop server", flush=True)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print("\n[LS] Server stopped", flush=True)

    elif args.action == "import":
        token = get_token()
        project_id = create_project(token)
        upload_videos(token, project_id)
        import_tasks(token, project_id)
        print(f"\n[LS] Project ready: {LS_HOST}/projects/{project_id}", flush=True)


if __name__ == "__main__":
    main()
