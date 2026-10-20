"""MammalNet 犬科(家犬)子集 + DogMo 预览素材下载脚本.

Owner: 数据实施员（P1 数据补充线）
日期: 2026-10-20
依据: 侦察报告（reports/p1-dataset-expansion-2026-10-20.json 中的 sources 段）

背景
----
MammalNet (Vision-CAIR/MammalNet, CVPR 2023) 的 S3 桶可直接匿名访问：
  - annotation.tar       1,512,456 B（taxonomy + detection_annotations.json + composition 划分）
  - trimmed_videos/<id>.mp4  逐文件对象（README 里的 trimmed_videos.tar.gz 已 403，
    桶内实际对象名是 trimmed_video.tar.gz，159 GB，不适合整包下载）
本脚本**只**下载家犬子集（genus=canis 且 keyword 不是 wolf/wolfdog），
按 (subset, keyword) 分层、按预算（默认 1.6 GB）抽样，逐文件并发分块下载。

DogMo (arXiv 2510.24117) 没有公开数据集发布渠道，只下载官网 47 个预览片段
（dataset 20 + dataset_pre 20 + results 7）和 2 个 GLB，作为**定性参考**，
不构成训练/评测集。

网络
----
- 所有请求显式绕过系统代理（ProxyHandler({})），符合项目网络排障约定。
- S3 单流只有 ~20 KB/s，必须高并发分块；实测 768 线程 × 128 KB ≈ 1.9 MB/s。
- 若 S3 不可达，脚本以非零码退出并打印可恢复指引（不伪造产物）。

用法
----
    # 全量（MammalNet 1.6 GB 预算 + DogMo 预览 ~209 MB）
    python scripts/download_mammalnet_canine.py --dataset all

    # 只列计划、不下载
    python scripts/download_mammalnet_canine.py --dataset mammalnet --dry-run

    # 指定预算与输出根目录
    python scripts/download_mammalnet_canine.py --dataset mammalnet \
        --output-root "D:/Desktop/PR论文/datasets" --max-gb 1.6

产出
----
    <output-root>/mammalnet_canine/
        annotations/          annotation.tar 解包内容
        videos/{train,val,test}/*.mp4
        manifest.json         每个文件的视频 ID / 分类 / 行为区间 / 字节 / sha256
        subset_annotations.json
        summary.json          下载统计（可达性、耗时、吞吐、预算）
        recovery.md           全量子集(9.29 GB)恢复指引
    <output-root>/dogmo_preview/
        preview/{dataset,dataset_pre,results}/*.mp4
        glbs/{pose,motion}.glb
        manifest.json
        summary.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(r"D:\Desktop\PR论文\datasets")

MN_ANNOT_URL = "https://mammalnet.s3.amazonaws.com/annotation.tar"
MN_VIDEO_BASE = "https://mammalnet.s3.amazonaws.com/"
MN_BUCKET = "https://mammalnet.s3.amazonaws.com/"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
DOGMO_PAGE = "https://pie-lab.cn/DogMo/"
DOGMO_BASE = "https://pie-lab.cn/DogMo/"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) k9-research/1.0"
CHUNK_BYTES = 128 * 1024
DOWNLOAD_WORKERS = 512

# 家犬判定：genus=canis 且 keyword 不落在野生犬科词表
WILD_KEYWORDS = {"wolf", "wolfdog"}
SUBSET_DIR = {"training": "train", "validation": "val", "testing": "test"}

PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)


def opener() -> urllib.request.OpenerDirector:
    """显式无代理 opener（项目网络约定：先绕代理直连）。"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def head_size(url: str, timeout: int = 60) -> Optional[int]:
    """取文件大小：先 HEAD，Content-Length 缺失/为 0 时退回 Range: bytes=0-0。"""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with opener().open(req, timeout=timeout) as r:
            size = int(r.headers.get("Content-Length") or 0)
            if size > 0:
                return size
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return None
    except Exception:
        pass

    # DogMo 的 GLB 响应不带 Content-Length（静态站由 CF 代理），用 206 的 Content-Range 兜底
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"}
    )
    try:
        with opener().open(req, timeout=timeout) as r:
            content_range = r.headers.get("Content-Range", "")
            r.read(1)
            if "/" in content_range:
                total = content_range.rsplit("/", 1)[-1]
                if total.isdigit() and int(total) > 0:
                    return int(total)
            size = int(r.headers.get("Content-Length") or 0)
            return size if size > 0 else None
    except Exception:
        return None


def head_etag(url: str, timeout: int = 60) -> Optional[str]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with opener().open(req, timeout=timeout) as r:
            etag = r.headers.get("ETag", "")
            return etag.strip('"') or None
    except Exception:
        return None


def fetch_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
            )
            with opener().open(req, timeout=120) as r:
                data = r.read()
                if r.status == 206:
                    return data
                # 服务器忽略 Range：整文件返回时只取需要的切片
                if len(data) >= end + 1:
                    return data[start : end + 1]
                raise IOError(f"unexpected status={r.status} len={len(data)}")
        except Exception as exc:  # noqa: BLE001 - 需要重试所有网络异常
            last = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"range fetch failed {start}-{end}: {last}")


def prepare_part(dest: Path, size: int) -> Path:
    """建立与目标等长的 .part 文件（分块随机写）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with open(part, "wb") as fh:
        fh.truncate(size)
    return part


def hash_and_finalize(part: Path, dest: Path, size: int, expected_md5: Optional[str], url: str) -> Tuple[str, str]:
    """校验 .part 长度与 MD5 后原子改名，返回 (md5, sha256)。"""
    if part.stat().st_size != size:
        raise IOError(f"size mismatch: {part.stat().st_size} != {size}")
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with open(part, "rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    digest = md5.hexdigest()
    # S3 ETag 等于单段上传的 MD5；带 "-" 的是 multipart，不可直接比较
    if expected_md5 and "-" not in expected_md5 and digest != expected_md5:
        raise IOError(f"md5 mismatch for {url}: {digest} != {expected_md5}")
    os.replace(part, dest)
    return digest, sha.hexdigest()


def download_parallel(
    url: str,
    dest: Path,
    size: int,
    workers: int = DOWNLOAD_WORKERS,
    chunk: int = CHUNK_BYTES,
    expected_md5: Optional[str] = None,
    label: str = "",
) -> Dict[str, object]:
    """单文件下载（内部转调 download_many，供标注包等独立文件使用）。"""
    results, failures = download_many(
        [{"url": url, "dest": dest, "size": size, "md5": expected_md5, "meta": {"label": label}}],
        workers=workers,
        chunk=chunk,
    )
    if failures:
        raise RuntimeError(failures[0]["error"])
    res = results[0]
    return {
        "bytes": res["size"],
        "md5": res["md5"],
        "sha256": res["sha256"],
        "cached": res["cached"],
        "elapsed": res.get("elapsed", 0.0),
    }


def download_many(
    items: List[Dict],
    workers: int = DOWNLOAD_WORKERS,
    chunk: int = CHUNK_BYTES,
    report_every: float = 20.0,
) -> Tuple[List[Dict], List[Dict]]:
    """跨文件并发分块下载。

    items: [{url, dest, size, md5(optional), meta}]
    给所有文件的 chunk 建一个全局任务池 —— 单文件并发对 <10 MB 片段并不划算，
    实测跨文件 768 并发可达 ~1.9 MB/s，而逐文件顺序下载只有 ~0.26 MB/s。

    Returns: (results, failures)
    """
    results: List[Dict] = []
    failures: List[Dict] = []
    pending: List[Dict] = []
    for item in items:
        dest: Path = item["dest"]
        if dest.exists() and dest.stat().st_size == item["size"]:
            md5 = hashlib.md5()
            sha = hashlib.sha256()
            with open(dest, "rb") as fh:
                for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                    md5.update(block)
                    sha.update(block)
            results.append({**item, "md5": md5.hexdigest(), "sha256": sha.hexdigest(), "cached": True})
            continue
        item["_part"] = prepare_part(dest, item["size"])
        pending.append(item)

    if not pending:
        return results, failures

    jobs = []
    for idx, item in enumerate(pending):
        size = item["size"]
        for lo in range(0, size, chunk):
            jobs.append((idx, lo, min(lo + chunk - 1, size - 1)))

    t0 = time.time()
    done = 0
    last = t0
    total_bytes = sum(i["size"] for i in pending)

    def run(job: Tuple[int, int, int]) -> int:
        idx, lo, hi = job
        item = pending[idx]
        data = fetch_range(item["url"], lo, hi)
        if len(data) != hi - lo + 1:
            raise IOError(f"short read {len(data)} != {hi - lo + 1}")
        with open(item["_part"], "r+b", buffering=0) as fh:
            fh.seek(lo)
            fh.write(data)
        return len(data)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for fut in cf.as_completed(futures):
            try:
                done += fut.result()
            except Exception:  # noqa: BLE001 - 失败文件在收敛阶段统一报告
                pass
            now = time.time()
            if now - last >= report_every:
                rate = done / max(now - t0, 1e-6)
                pct = 100 * done / max(total_bytes, 1)
                log(f"    {done/1e6:.1f}/{total_bytes/1e6:.1f} MB ({pct:.0f}%) {rate/1024:.0f} KB/s")
                last = now

    for item in pending:
        part: Path = item["_part"]
        try:
            md5, sha = hash_and_finalize(part, item["dest"], item["size"], item.get("md5"), item["url"])
            results.append(
                {
                    **{k: v for k, v in item.items() if k != "_part"},
                    "md5": md5,
                    "sha256": sha,
                    "cached": False,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"url": item["url"], "dest": str(item["dest"]), "error": str(exc)})
    log(f"    完成 {len(results)} 文件 / {sum(r['size'] for r in results)/1e6:.1f} MB，{time.time()-t0:.0f}s")
    return results, failures


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probe_clip(path: Path) -> Optional[Dict]:
    """ffprobe 取片段真实时长/分辨率/帧率。"""
    import subprocess

    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,codec_name",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        num, _, den = str(stream.get("avg_frame_rate", "0/1")).partition("/")
        fps = round(float(num) / float(den), 2) if den and float(den) else None
        return {
            "duration_s": round(float((data.get("format") or {}).get("duration") or 0), 3),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "fps": fps,
            "codec": stream.get("codec_name"),
        }
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# MammalNet
# --------------------------------------------------------------------------- #


def s3_list_trimmed_videos(cache: Path) -> Dict[str, int]:
    """列出 s3://mammalnet/trimmed_videos/ 全部对象，缓存为 {key: size}。"""
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        log(f"[mammalnet] 使用缓存的 S3 索引 {cache} ({len(data)} objects)")
        return {k: int(v) for k, v in data.items()}

    objects: Dict[str, int] = {}
    token: Optional[str] = None
    t0 = time.time()
    while True:
        query = "?list-type=2&prefix=trimmed_videos/&max-keys=1000"
        if token:
            query += "&continuation-token=" + urllib.parse.quote(token, safe="")
        req = urllib.request.Request(MN_BUCKET + query, headers={"User-Agent": USER_AGENT})
        with opener().open(req, timeout=90) as r:
            root = ET.fromstring(r.read())
        for item in root.findall("s3:Contents", S3_NS):
            key = item.find("s3:Key", S3_NS).text
            objects[key] = int(item.find("s3:Size", S3_NS).text)
        truncated = (root.findtext("s3:IsTruncated", "false", S3_NS) or "false") == "true"
        nxt = root.findtext("s3:NextContinuationToken", None, S3_NS)
        if not truncated or not nxt:
            break
        token = nxt
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(objects, indent=0, sort_keys=True), encoding="utf-8")
    log(f"[mammalnet] S3 索引完成: {len(objects)} objects, {time.time()-t0:.0f}s")
    return objects


def extract_annotation_tar(tar_path: Path, out_dir: Path) -> List[str]:
    """安全解包（拒绝绝对路径/..）；剥掉 tar 顶层的 annotation/ 前缀。"""
    written: List[str] = []
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"unsafe tar member: {name}")
            if name.startswith("annotation/"):
                name = name[len("annotation/") :]
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as dst:
                dst.write(src.read())
            written.append(name)
    return written


def is_domestic_dog(record: Dict) -> bool:
    for tax in record.get("taxnomy", []) or []:
        if str(tax.get("genus", "")).lower() == "canis":
            if str(tax.get("keyword", "")).lower() not in WILD_KEYWORDS:
                return True
    return False


def dog_keyword(record: Dict) -> str:
    for tax in record.get("taxnomy", []) or []:
        if str(tax.get("genus", "")).lower() == "canis":
            if str(tax.get("keyword", "")).lower() not in WILD_KEYWORDS:
                return str(tax.get("keyword", "")).lower()
    return "unknown"


def clip_files(video_id: str, index: Dict[str, int]) -> List[Tuple[str, int]]:
    """把 annotation 的视频 ID 映射到 trimmed_videos/<id>[_n].mp4 文件列表。"""
    out: List[Tuple[str, int]] = []
    for suffix in [""] + [f"_{i}" for i in range(1, 6)]:
        key = f"trimmed_videos/{video_id}{suffix}.mp4"
        if key in index:
            out.append((key, index[key]))
    return out


def select_stratified(
    candidates: Dict[Tuple[str, str], List[Tuple[str, int]]],
    budget: int,
) -> Tuple[List[Tuple[Tuple[str, str], str, int]], Dict[str, float]]:
    """按 (subset, keyword) 分层、按字节占比轮转选取，硬上限 budget。"""
    totals = {k: sum(sz for _, sz in v) for k, v in candidates.items()}
    grand = sum(totals.values()) or 1
    targets = {k: budget * totals[k] / grand for k in candidates}
    used = {k: 0 for k in candidates}
    picked: List[Tuple[Tuple[str, str], str, int]] = []
    cursors = {k: 0 for k in candidates}
    ordered = sorted(candidates.keys())
    selected_total = 0

    # 每层至少一个（取最小文件），保证 8 个品种 × 3 个 split 全覆盖
    for key in ordered:
        items = candidates[key]
        if not items:
            continue
        name, size = items[0]
        if selected_total + size <= budget:
            picked.append((key, name, size))
            used[key] += size
            selected_total += size
            cursors[key] = 1

    progressed = True
    while progressed:
        progressed = False
        for key in ordered:
            items = candidates[key]
            idx = cursors[key]
            if idx >= len(items):
                continue
            name, size = items[idx]
            if used[key] + size > targets[key]:
                continue
            if selected_total + size > budget:
                continue
            picked.append((key, name, size))
            used[key] += size
            selected_total += size
            cursors[key] = idx + 1
            progressed = True

    stats = {
        "strata": len(candidates),
        "selected_files": len(picked),
        "selected_bytes": selected_total,
        "budget_bytes": budget,
        "coverage": {f"{k[0]}/{k[1]}": used[k] for k in ordered},
    }
    return picked, stats


def run_mammalnet(output_root: Path, max_gb: float, dry_run: bool, workers: int) -> Dict:
    out = output_root / "mammalnet_canine"
    ann_dir = out / "annotations"
    cache = out / "_cache" / "s3_trimmed_index.json"
    out.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {
        "dataset": "mammalnet_canine",
        "output_dir": str(out),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # 1) 标注包
    ann_tar = out / "_cache" / "annotation.tar"
    ann_size = head_size(MN_ANNOT_URL)
    if ann_size is None:
        raise SystemExit(f"[mammalnet] annotation.tar 不可达: {MN_ANNOT_URL}")
    log(f"[mammalnet] annotation.tar {ann_size} B")
    if not dry_run:
        info = download_parallel(MN_ANNOT_URL, ann_tar, ann_size, workers=32, chunk=64 * 1024, label="annotation.tar")
        summary["annotation_tar"] = {"url": MN_ANNOT_URL, **info}
        written = extract_annotation_tar(ann_tar, ann_dir)
        summary["annotation_files"] = written
        log(f"[mammalnet] 解包 {len(written)} 个文件 -> {ann_dir}")

    ann_json = ann_dir / "detection_annotations.json"
    if not ann_json.exists():
        # tar 内层目录是 annotation/，解包后可能嵌套
        found = sorted(ann_dir.rglob("detection_annotations.json"))
        if found:
            ann_json = found[0]
    if not ann_json.exists():
        log(f"[mammalnet] 标注尚未落盘（期望 {ann_json}），结束")
        summary["status"] = "dry_run"
        return summary

    annotations = json.loads(ann_json.read_text(encoding="utf-8"))
    domestic = {vid: rec for vid, rec in annotations.items() if is_domestic_dog(rec)}
    log(f"[mammalnet] 视频 {len(annotations)}，家犬 {len(domestic)}")

    # 2) S3 索引（缓存）
    index = s3_list_trimmed_videos(cache)

    # 3) 分层候选
    candidates: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
    clip_meta: Dict[str, Dict] = {}
    for vid, rec in domestic.items():
        files = clip_files(vid, index)
        if not files:
            continue
        subset = SUBSET_DIR.get(str(rec.get("subset")), "unknown")
        key = (subset, dog_keyword(rec))
        clip_meta[vid] = rec
        for name, size in files:
            candidates.setdefault(key, []).append((name, size))
    for key in candidates:
        candidates[key].sort(key=lambda x: (x[1], x[0]))

    budget = int(max_gb * 1_000_000_000)
    picked, stats = select_stratified(candidates, budget)
    summary["selection"] = stats
    summary["full_domestic_subset"] = {
        "clips": len(clip_meta),
        "files": sum(len(v) for v in candidates.values()),
        "bytes": sum(sz for v in candidates.values() for _, sz in v),
        "hours": round(sum(float(r.get("duration", 0)) for r in clip_meta.values()) / 3600, 2),
    }
    log(
        f"[mammalnet] 选取 {stats['selected_files']} 文件 "
        f"{stats['selected_bytes']/1e9:.3f} GB / 预算 {max_gb:.2f} GB"
    )

    if dry_run:
        summary["status"] = "dry_run"
        for key in sorted(candidates):
            n = sum(1 for k, _, _ in picked if k == key)
            log(f"    {key[0]:10} {key[1]:14} files={n:3d} total={len(candidates[key]):3d}")
        return summary

    # 4) 并发取 ETag 作为 MD5 期望值（S3 单段上传 ETag == MD5）
    etags: Dict[str, Optional[str]] = {}
    with cf.ThreadPoolExecutor(max_workers=64) as pool:
        for name, etag in zip(
            [n for _, n, _ in picked],
            pool.map(lambda n: head_etag(MN_VIDEO_BASE + n), [n for _, n, _ in picked]),
        ):
            etags[name] = etag

    # 5) 跨文件并发下载
    items = []
    meta_by_file: Dict[str, Dict] = {}
    annotation_ids = sorted(clip_meta.keys(), key=len, reverse=True)
    for (subset, keyword), name, size in sorted(picked):
        video_id = Path(name).stem
        # YouTube ID 本身可能含 _数字 后缀（如 t387Sj7u_60），不能用正则剥；
        # 取能作为前缀匹配的最长 annotation ID。
        base_id = next((aid for aid in annotation_ids if video_id == aid or video_id.startswith(aid + "_")), video_id)
        dest = out / "videos" / subset / f"{video_id}.mp4"
        rec = clip_meta.get(base_id, {})
        meta_by_file[name] = {
            "file": str(dest.relative_to(out)).replace("\\", "/"),
            "s3_key": name,
            "video_id": base_id,
            "part": video_id if video_id != base_id else None,
            "subset": subset,
            "keyword": keyword,
            # 注意：annotation 的 duration 是**源 YouTube 视频**时长，不是裁剪后片段时长
            "source_duration_s": rec.get("duration"),
            "source_resolution": rec.get("resolution"),
            "source_fps": rec.get("fps"),
            "url": rec.get("url"),
            "taxonomy": rec.get("taxnomy"),
            "behaviors": rec.get("annotations"),
        }
        items.append(
            {"url": MN_VIDEO_BASE + name, "dest": dest, "size": size, "md5": etags.get(name), "key": name}
        )

    t0 = time.time()
    results, failures = download_many(items, workers=workers, chunk=CHUNK_BYTES)
    manifest = [
        {**meta_by_file[r["key"]], "bytes": r["size"], "md5": r["md5"], "sha256": r["sha256"]}
        for r in sorted(results, key=lambda r: r["key"])
    ]

    # 6) 用 ffprobe 补齐真实片段时长/分辨率（annotation 的 duration 是源视频时长）
    if shutil.which("ffprobe"):
        with cf.ThreadPoolExecutor(max_workers=24) as pool:
            probes = list(pool.map(lambda e: probe_clip(out / e["file"]), manifest))
        for entry, probe in zip(manifest, probes):
            entry["clip"] = probe
    else:
        log("[mammalnet] 未找到 ffprobe，manifest 缺 clip 实测字段")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    subset_ann = {vid: clip_meta[vid] for vid in {m["video_id"] for m in manifest} if vid in clip_meta}
    (out / "subset_annotations.json").write_text(
        json.dumps(subset_ann, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    summary.update(
        {
            "status": "ok" if not failures else "partial",
            "downloaded_files": len(manifest),
            "failed": failures[:50],
            "failed_count": len(failures),
            "downloaded_bytes": sum(m["bytes"] for m in manifest),
            "elapsed_s": round(time.time() - t0, 1),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    write_mammalnet_recovery(out, summary)
    return summary


def write_mammalnet_recovery(out: Path, summary: Dict) -> None:
    text = f"""# MammalNet 家犬子集 — 恢复指引

本目录是**受预算约束的分层抽样**，不是完整子集。当前内容：
- 文件数：{summary.get('downloaded_files', 0)}
- 字节数：{summary.get('downloaded_bytes', 0)}（预算 {summary.get('selection', {}).get('budget_bytes', 0)}）

完整家犬子集（genus=canis 排除 wolf/wolfdog）按 S3 索引测算：
- 1,129 个 clip / 1,198 个 mp4 文件 / 9.293 GB / 28.43 小时
- train 824 clip / 6.46 GB，val 92 clip / 0.74 GB，test 213 clip / 2.07 GB

## 恢复完整子集

```bash
# 1) 标注（1.5 MB，可直连）
curl -o annotation.tar https://mammalnet.s3.amazonaws.com/annotation.tar
tar -xf annotation.tar

# 2) 视频：逐文件对象，不要用 README 里的 trimmed_videos.tar.gz（403）
python scripts/download_mammalnet_canine.py --dataset mammalnet --max-gb 9.3

# 3) 或用 HF 镜像整包（159 GB，流式选择性解包，约 2 小时 @23 MB/s）
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
import tarfile, urllib.request
url = "https://hf-mirror.com/datasets/linxxx3/MammalNet/resolve/main/trimmed_video.tar.gz"
with urllib.request.urlopen(url) as r, tarfile.open(fileobj=r, mode="r|gz") as tf:
    for m in tf:
        if m.name.endswith(".mp4") and m.name.split("/")[-1].split(".")[0].rstrip("0123456789_") in WANTED:
            tf.extract(m, "mammalnet_full")
PY
```

S3 单流约 20 KB/s，必须高并发分块（脚本默认 512 线程 × 128 KB，实测 ≈1.9 MB/s）。

## 许可

MammalNet 标注 CC BY（论文/仓库）；视频源自 YouTube，作者声明只提供标注。
研究使用需引用 Chen et al., CVPR 2023；**不得再分发视频**。
"""
    (out / "recovery.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# DogMo
# --------------------------------------------------------------------------- #


def fetch_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener().open(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def dogmo_asset_urls() -> List[str]:
    """从官网 HTML 解析 dataset / results 片段，并派生 dataset_pre 变体。"""
    html = fetch_text(DOGMO_BASE)
    urls = sorted(
        set(re.findall(r"static/videos/(?:dataset|results)/[A-Za-z0-9_./-]+\.mp4", html))
    )
    pre = [
        u.replace("static/videos/dataset/", "static/videos/dataset_pre/")
        for u in urls
        if u.startswith("static/videos/dataset/")
    ]
    glbs = ["static/glbs/pose.glb", "static/glbs/motion.glb"]
    return urls + pre + glbs


def download_plain(url: str, dest: Path, timeout: int = 300) -> Dict[str, object]:
    """普通 GET 整文件下载（不发 Range / Accept-Encoding）。

    DogMo 的静态站在 Range 请求下会返回 *压缩实体* 的字节范围
    （Content-Range: bytes 0-0/6772757 对应 14,968,128 B 的 GLB），
    分块下载会写坏 GLB；因此 GLB 走整文件 GET。
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    size = 0
    t0 = time.time()
    with opener().open(req, timeout=timeout) as r, open(tmp, "wb") as fh:
        while True:
            block = r.read(1024 * 1024)
            if not block:
                break
            fh.write(block)
            md5.update(block)
            sha.update(block)
            size += len(block)
    os.replace(tmp, dest)
    return {
        "bytes": size,
        "md5": md5.hexdigest(),
        "sha256": sha.hexdigest(),
        "cached": False,
        "elapsed": round(time.time() - t0, 2),
    }


def run_dogmo(output_root: Path, dry_run: bool, workers: int = 16) -> Dict:
    out = output_root / "dogmo_preview"
    out.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {
        "dataset": "dogmo_preview",
        "output_dir": str(out),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "DogMo 无公开数据集渠道；本目录仅为官网预览素材，不可作为训练/评测集",
    }
    assets = dogmo_asset_urls()
    log(f"[dogmo] 官网解析到 {len(assets)} 个素材")

    sized = []
    for rel in assets:
        size = head_size(DOGMO_BASE + rel)
        if not size or size <= 0:
            log(f"    MISS {rel} (size unknown)")
            continue
        sized.append((rel, size))
    summary["planned_files"] = len(sized)
    summary["planned_bytes"] = sum(s for _, s in sized)
    log(f"[dogmo] 可下载 {len(sized)} 个 / {summary['planned_bytes']/1e6:.1f} MB")
    if dry_run:
        summary["status"] = "dry_run"
        return summary

    manifest = []
    failures = []
    t0 = time.time()

    def fetch(item: Tuple[str, int]) -> Tuple[str, int, Dict]:
        rel, size = item
        dest = out / rel.replace("static/videos/", "preview/").replace("static/glbs/", "glbs/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = DOGMO_BASE + rel
        if rel.endswith(".glb"):
            info = download_plain(url, dest)
        else:
            info = download_parallel(url, dest, size, workers=min(workers, 16), chunk=256 * 1024, label=Path(rel).name)
        return rel, int(info["bytes"]), {"path": dest, **info}

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, item): item for item in sized}
        for fut in cf.as_completed(futures):
            rel, size = futures[fut]
            try:
                rel, size, info = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append({"asset": rel, "error": str(exc)})
                log(f"    FAIL {rel}: {exc}")
                continue
            manifest.append(
                {
                    "file": str(info["path"].relative_to(out)).replace("\\", "/"),
                    "source_url": DOGMO_BASE + rel,
                    "bytes": size,
                    "md5": info["md5"],
                    "sha256": info["sha256"],
                    "kind": "glb" if rel.endswith(".glb") else rel.split("/")[2],
                }
            )
            log(f"    OK {rel} ({size/1e6:.1f} MB)")

    manifest.sort(key=lambda m: m["file"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    summary.update(
        {
            "status": "ok" if not failures else "partial",
            "downloaded_files": len(manifest),
            "downloaded_bytes": sum(m["bytes"] for m in manifest),
            "failed": failures,
            "elapsed_s": round(time.time() - t0, 1),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="下载 MammalNet 家犬子集与 DogMo 预览素材")
    parser.add_argument("--dataset", choices=["mammalnet", "dogmo", "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-gb", type=float, default=1.6, help="MammalNet 视频字节预算（GB，十进制）")
    parser.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    results = {}
    if args.dataset in ("mammalnet", "all"):
        results["mammalnet"] = run_mammalnet(args.output_root, args.max_gb, args.dry_run, args.workers)
    if args.dataset in ("dogmo", "all"):
        results["dogmo"] = run_dogmo(args.output_root, args.dry_run)

    print(json.dumps({k: v.get("status") for k, v in results.items()}, ensure_ascii=False))
    return 0 if all(v.get("status") in ("ok", "dry_run") for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
