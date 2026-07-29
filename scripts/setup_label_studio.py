"""Setup Label Studio project via Django session login.

Label Studio 1.23 disables legacy DRF tokens by default (`TokenAuthenticationPhaseout`
rejects them with "legacy token authentication has been disabled"). The supported
authentications are JWT (via /api/token/) and Django session cookies. The session
cookie route is the simplest for a CLI setup script:

  1. GET /user/login/ -> obtain csrftoken cookie
  2. POST /user/login/ with email/password/csrfmiddlewaretoken -> establish session
  3. Use the session cookie for all subsequent API calls (SessionAuthentication)
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

LS_HOST = "http://127.0.0.1:8080"
EMAIL = "admin@k9.local"
PASSWORD = "k9admin2026"

ROOT = Path("data/youtube_self_label")
RAW = ROOT / "raw"
LS_DIR = ROOT / "label_studio"
CONFIG_PATH = LS_DIR / "label_config.xml"
TASKS_PATH = LS_DIR / "tasks_all.json"


def login_and_get_session() -> requests.Session:
    """Login via Django session form, return a session with auth cookies."""
    s = requests.Session()

    # 1. GET login page to obtain csrftoken cookie
    r = s.get(f"{LS_HOST}/user/login/", timeout=15)
    print(f"  [GET /user/login/] {r.status_code}", flush=True)
    if r.status_code != 200:
        raise RuntimeError(f"login page unreachable: {r.status_code} {r.text[:200]}")

    csrf = s.cookies.get("csrftoken")
    if not csrf:
        raise RuntimeError("no csrftoken cookie returned from login page")
    print(f"  [csrf] {csrf[:20]}...", flush=True)

    # 2. POST credentials
    r = s.post(
        f"{LS_HOST}/user/login/",
        data={
            "email": EMAIL,
            "password": PASSWORD,
            "persist_session": "on",
            "csrfmiddlewaretoken": csrf,
        },
        headers={"Referer": f"{LS_HOST}/user/login/"},
        timeout=15,
        allow_redirects=False,
    )
    print(f"  [POST /user/login/] {r.status_code} redirect={r.headers.get('Location', '')}",
          flush=True)
    if r.status_code not in (302, 303):
        raise RuntimeError(f"login failed: status={r.status_code} body={r.text[:300]}")

    # Sanity check: hit an authenticated endpoint
    r = s.get(f"{LS_HOST}/api/projects/?page=1&page_size=1", timeout=10)
    print(f"  [verify session] /api/projects/ -> {r.status_code}", flush=True)
    if r.status_code != 200:
        raise RuntimeError(f"session not authenticated: {r.status_code} {r.text[:300]}")
    return s


def main() -> int:
    print("=== Label Studio setup via session login ===", flush=True)

    # 0. Sanity-check inputs
    if not CONFIG_PATH.exists():
        print(f"[ERROR] label config missing: {CONFIG_PATH}", flush=True)
        return 1
    if not TASKS_PATH.exists():
        print(f"[ERROR] tasks file missing: {TASKS_PATH}", flush=True)
        return 1
    videos = sorted(RAW.glob("*.mp4"))
    print(f"  [inputs] config={CONFIG_PATH.name} tasks={TASKS_PATH.name} videos={len(videos)}",
          flush=True)

    # 1. Acquire session
    try:
        session = login_and_get_session()
    except RuntimeError as e:
        print(f"[ERROR] {e}", flush=True)
        return 1

    # For Django POST requests via session, must include csrftoken header
    csrf = session.cookies.get("csrftoken", "")
    headers = {"X-CSRFToken": csrf}

    # 2. Check existing projects (avoid duplicates)
    r = session.get(f"{LS_HOST}/api/projects/?page_size=100", timeout=15)
    print(f"  [list projects] {r.status_code}", flush=True)
    existing = r.json() if r.status_code == 200 else {}
    project_id = None
    title = "K9 行为标注 — YouTube 自标（Phase 2.0a 路径 B）"
    for p in existing.get("results", []):
        if p.get("title") == title:
            project_id = p["id"]
            print(f"  [reuse] existing project id={project_id}", flush=True)
            break

    # 3. Create project if not found
    if project_id is None:
        config = CONFIG_PATH.read_text(encoding="utf-8")
        payload = {
            "title": title,
            "description": "YOLO26-pose 预标注已生成。请修正检测框 + 标注行为类别。",
            "label_config": config,
        }
        r = session.post(
            f"{LS_HOST}/api/projects/", json=payload, headers=headers, timeout=15
        )
        print(f"  [create project] {r.status_code}", flush=True)
        if r.status_code not in (200, 201):
            print(f"  response: {r.text[:500]}", flush=True)
            return 1
        project_id = r.json()["id"]
        print(f"  [OK] project id={project_id}", flush=True)

    # 4. Check existing tasks count to decide on JSON import
    r = session.get(
        f"{LS_HOST}/api/projects/{project_id}/tasks/?page=1&page_size=1",
        timeout=15,
    )
    initial_task_count = 0
    if r.status_code == 200:
        data = r.json()
        # API may return a list or a dict with "tasks" key
        if isinstance(data, list):
            initial_task_count = len(data)
        elif isinstance(data, dict):
            initial_task_count = len(data.get("tasks", []))
            # Also check "total" field if present
            if "total" in data:
                initial_task_count = data["total"]
    print(f"  [existing tasks] {initial_task_count}", flush=True)

    # 5. Import pre-annotated tasks (JSON) — only if project is empty (idempotent)
    if initial_task_count == 0:
        tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
        print(f"  [import] {len(tasks)} pre-annotated tasks ...", flush=True)
        r = session.post(
            f"{LS_HOST}/api/projects/{project_id}/import",
            json=tasks,
            headers=headers,
            timeout=60,
        )
        print(f"    -> {r.status_code}", flush=True)
        if r.status_code not in (200, 201):
            print(f"  response: {r.text[:500]}", flush=True)
    else:
        print(f"  [skip import] project already has {initial_task_count} tasks", flush=True)

    # 6. Upload videos via /api/projects/{id}/import (multipart).
    #    Each POST creates a FileUpload record (served at /data/upload/<filename>)
    #    AND auto-creates an empty task. The pre-annotated tasks (imported in step 5)
    #    reference /data/upload/<filename>, so once the FileUpload exists, they resolve.
    #    We then delete the empty upload-created tasks (keeping the ones with predictions).
    uploaded = 0
    for video in videos:
        print(f"  [upload] {video.name} ...", flush=True)
        with open(video, "rb") as fh:
            files = {"file": (video.name, fh, "video/mp4")}
            r = session.post(
                f"{LS_HOST}/api/projects/{project_id}/import",
                files=files,
                headers=headers,
                timeout=300,
            )
        if r.status_code in (200, 201):
            uploaded += 1
            print(f"    -> {r.status_code} OK", flush=True)
        else:
            print(f"    -> {r.status_code} {r.text[:200]}", flush=True)

    # 7. Cleanup: delete empty tasks (those without predictions, created by file upload)
    r = session.get(
        f"{LS_HOST}/api/projects/{project_id}/tasks/?page=1&page_size=200",
        timeout=30,
    )
    print(f"  [list tasks] {r.status_code}", flush=True)
    if r.status_code == 200:
        data = r.json()
        all_tasks = data if isinstance(data, list) else data.get("tasks", [])
        print(f"    total tasks: {len(all_tasks)}", flush=True)
        deleted = 0
        for t in all_tasks:
            preds = t.get("predictions") or []
            preds_count = t.get("predictions_count", 0)
            if not preds and preds_count == 0:
                # Empty task created by file upload — delete it
                r = session.delete(
                    f"{LS_HOST}/api/tasks/{t['id']}",
                    headers=headers,
                    timeout=15,
                )
                if r.status_code in (200, 204):
                    deleted += 1
                else:
                    print(f"    [delete {t['id']}] {r.status_code} {r.text[:200]}",
                          flush=True)
        print(f"    deleted empty tasks: {deleted}", flush=True)

    # 8. Fix task video URLs to point to actual FileUpload records.
    #    Label Studio stores uploaded files with an 8-hex UUID prefix under
    #    /data/upload/<project_id>/<uuid8>-<original>. Our pre-annotated tasks
    #    reference the simpler /data/upload/<original> URL, which 404s. This
    #    step PATCHes each task to use the actual serving URL.
    r = session.get(
        f"{LS_HOST}/api/projects/{project_id}/file-uploads?all=true",
        timeout=15,
    )
    url_map: dict[str, str] = {}
    if r.status_code == 200:
        for fu in r.json():
            file_path = fu.get("file", "")
            if not file_path or not file_path.startswith("upload/"):
                continue
            path_after = file_path[len("upload/"):]
            serving_url = f"/data/upload/{path_after}"
            basename = path_after.split("/")[-1]
            # Strip "xxxxxxxx-" UUID prefix (8 hex chars + dash)
            if (len(basename) > 9 and basename[8] == "-"
                    and all(c in "0123456789abcdef" for c in basename[:8])):
                original = basename[9:]
            else:
                original = basename
            stem = original.rsplit(".", 1)[0] if "." in original else original
            url_map[stem] = serving_url
        print(f"  [file-uploads] mapped {len(url_map)} URLs", flush=True)

    if url_map:
        r = session.get(
            f"{LS_HOST}/api/projects/{project_id}/tasks/?page=1&page_size=200",
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            tasks_list = data if isinstance(data, list) else data.get("tasks", [])
            patched = 0
            for t in tasks_list:
                tdata = t.get("data") or {}
                vid = tdata.get("video_id")
                if not vid:
                    continue
                new_url = url_map.get(vid)
                if not new_url or tdata.get("video") == new_url:
                    continue
                new_data = dict(tdata)
                new_data["video"] = new_url
                r = session.patch(
                    f"{LS_HOST}/api/tasks/{t['id']}/",
                    json={"data": new_data},
                    headers=headers,
                    timeout=15,
                )
                if r.status_code in (200, 201):
                    patched += 1
                else:
                    print(f"    [patch {t['id']}] {r.status_code} {r.text[:200]}",
                          flush=True)
            print(f"    patched video URLs: {patched}", flush=True)

    # 9. Final task count + URL verification
    r = session.get(
        f"{LS_HOST}/api/projects/{project_id}/tasks/?page=1&page_size=200",
        timeout=30,
    )
    if r.status_code == 200:
        data = r.json()
        final_tasks = data if isinstance(data, list) else data.get("tasks", [])
        print(f"  [final] task count: {len(final_tasks)}", flush=True)
        # Verify first video URL
        if final_tasks:
            sample_url = (final_tasks[0].get("data") or {}).get("video", "")
            if sample_url:
                r = session.head(f"{LS_HOST}{sample_url}")
                print(f"  [verify] sample video {sample_url} -> {r.status_code}",
                      flush=True)

    print("\n=== Done ===", flush=True)
    print(f"Project URL: {LS_HOST}/projects/{project_id}/data", flush=True)
    print(f"Login: {EMAIL} / {PASSWORD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
