"""Animal-CLIP 零样本/开放词表行为推理（P1 基线，K9 工作犬）。

设计要点（全部来自本轮实测，证据见 reports/p1-animalclip-zeroshot-2026-10-20.md）：
  1. 不调用上游 main.py / val.py —— 它们在 Windows 上必崩：
       main.py:231 init_process_group(backend='nccl')（Windows torch 无 NCCL）
       val.py:153/156 的 nu/tot 未初始化（UnboundLocalError）
       val.py:60 硬编码 num_labels=140
       datasets/tools.py:117 模块级硬编码作者本机路径
  2. 只复用两处上游逻辑：models/model_builder.py 的 XCLIP 结构 + build_model，
     以及 utils/tools.py 的 generate_text / get_relation。
  3. 缺失且无法在 Windows/py3.12 安装的依赖（apex / mmcv-full / detectron2 / fvcore /
     torch_geometric / transformers / torchstat / iopath）通过
     external/animal-clip/_k9_stubs.py 以 sys.modules 注入方式绕过，不改上游任何文件。
  4. 帧预处理严格对齐上游 val_pipeline（datasets/build.py:359-367）：
     Resize(短边 256, BICUBIC) -> CenterCrop(224) -> Normalize(
     mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_bgr=False)。
     注意上游在 0-255 尺度上做 Normalize，因此 tensor 不额外 /255。
  5. 抽帧对齐 SampleFrames(test_mode=True, clip_len=1, frame_interval=1)：
     offsets = floor(arange(n)*avg + avg/2)，avg = F/n。

用法：
    python scripts/eval_animalclip_zeroshot.py \
        --manifest <manifest.jsonl> --fragments-dir <dir> --out-json <out.json> \
        [--label-space native|dogcentric] [--limit N] [--device cuda]

退出码：
    0  推理完成，产物已写出（即使权重缺失也写 partial 报告，绝不伪造数字）
    2  参数/路径错误
    3  manifest 与视频目录不一致
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANIMAL_CLIP_ROOT = PROJECT_ROOT / "external" / "animal-clip"
WEIGHTS_DIR = ANIMAL_CLIP_ROOT / "weights"

# --- 权重清单 ---------------------------------------------------------------
# 来源：README.md:26 唯一入口 https://drive.google.com/drive/folders/1iNMta_pFjhHLNK3FRZLUigSt3ya7i8sU
# 文件与大小于 2026-10-20 通过 firecrawl 抓取该文件夹页面确认（6 个文件）。
WEIGHT_FILES = {
    "mmnet": ("mmnet_best.pth", 1_920_000_000, "1.92 GB"),
    "AK": ("AK_best.pth", 1_850_000_000, "1.85 GB"),
    "LoTE": ("LoTE_best.pth", 1_850_000_000, "1.85 GB"),
}
BACKBONE_FILE = "ViT-B-16.pt"
BACKBONE_BYTES = 350_837_078
BACKBONE_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
)

# 上游 datasets/build.py:40-41（img_norm_cfg）
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]
RESIZE_SHORT = 256
CROP = 224
NUM_FRAMES = 8

# 数据集 -> (标签表, 动物表, 关系表, 类别数, 权重 key)
DATASETS = {
    "mmnet": (
        "data_files/MammalNet/behavior_to_id.csv",
        "data_files/MammalNet/genus_to_id.csv",
        "data_files/MammalNet/relation.txt",
        12,
        "mmnet",
    ),
    "animal_kingdom": (
        "data_files/AK/label.csv",
        "data_files/AK/animal_label.csv",
        "data_files/AK/animal-action-relation.txt",
        140,
        "AK",
    ),
    "LoTE": (
        "data_files/LoTE-Animal/LoTE_label.csv",
        "data_files/LoTE-Animal/animal_label.csv",
        "data_files/LoTE-Animal/relation.txt",
        21,
        "LoTE",
    ),
}

# DogCentric (KTH) 官方 10 类（论文 Table I）。
DOGCENTRIC_LABELS = [
    "ball play",
    "car",
    "drink",
    "feed",
    "turn-left",
    "turn-right",
    "pet",
    "shake",
    "sniff",
    "walk",
]

# DogCentric 动作 -> MammalNet 12 类语义空间的显式映射（None = 无对应，审计用）。
DOGCENTRIC_TO_MMNET = {
    "ball play": None,
    "car": None,
    "drink": 0,   # drinks water
    "feed": 2,    # eats food
    "turn-left": None,
    "turn-right": None,
    "pet": None,
    "shake": 6,   # grooms/cleans itself or other animal（近似）
    "sniff": None,
    "walk": None,
}


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #
def _bootstrap_vendored_repo() -> None:
    for p in (str(ANIMAL_CLIP_ROOT / "_vendor"), str(ANIMAL_CLIP_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import _k9_stubs  # noqa: WPS433 — vendored path 注入后才可 import

    _k9_stubs.install()
    _k9_stubs.prepare_sys_path()


def _cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# 权重 / 网络状态
# --------------------------------------------------------------------------- #
def sha256_of(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def resolve_weights(dataset: str) -> dict:
    """返回数据集专用权重状态。文件不在就如实写 missing，绝不伪造。"""
    fname, expect_bytes, expect_label = WEIGHT_FILES[DATASETS[dataset][4]]
    path = WEIGHTS_DIR / fname
    info = {
        "expected_file": str(path),
        "expected_size_label": expect_label,
        "expected_size_bytes": expect_bytes,
        "source": "https://drive.google.com/drive/folders/1iNMta_pFjhHLNK3FRZLUigSt3ya7i8sU (README.md:26)",
    }
    if not path.exists():
        info.update(status="missing", actual_size_bytes=0, sha256=None)
        return info
    size = path.stat().st_size
    status = "present" if size >= 1_500_000_000 else "truncated"
    info.update(status=status, actual_size_bytes=size, sha256=sha256_of(path))
    return info


def resolve_backbone() -> dict:
    path = WEIGHTS_DIR / BACKBONE_FILE
    if not path.exists():
        return {"status": "missing", "path": str(path), "size_bytes": 0,
                "url": BACKBONE_URL, "sha256": None}
    size = path.stat().st_size
    return {
        "status": "present" if size == BACKBONE_BYTES else "size_mismatch",
        "path": str(path),
        "size_bytes": size,
        "expected_bytes": BACKBONE_BYTES,
        "url": BACKBONE_URL,
        "sha256": sha256_of(path),
    }


def network_probe(timeout: float = 6.0) -> dict:
    """当次实测权重托管方的 TCP 可达性 —— partial 结论的证据来源。"""
    import socket

    targets = [
        "drive.google.com", "drive.usercontent.google.com", "docs.google.com",
        "huggingface.co", "hf-mirror.com",
    ]
    out = {}
    for host in targets:
        t0 = time.time()
        try:
            s = socket.create_connection((host, 443), timeout=timeout)
            s.close()
            out[host] = {"reachable": True, "latency_ms": round((time.time() - t0) * 1000)}
        except Exception as exc:  # noqa: BLE001
            out[host] = {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------- #
# 解码 / 预处理
# --------------------------------------------------------------------------- #
def decode_uniform_frames(video: Path, num_frames: int = NUM_FRAMES):
    """等距抽 num_frames 帧，返回 (list[RGB HWC uint8], decode_info)。"""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 无法打开: {video}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0

        if total <= 0:  # 容器索引不可用 -> 顺序解码后等距取
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if not frames:
                raise RuntimeError(f"解码出 0 帧: {video}")
            idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
            return [frames[i] for i in idx], {
                "total_frames": len(frames), "fps": fps,
                "decoded": len(frames), "fallback": True,
            }

        avg = total / float(num_frames)
        idx = np.clip((np.arange(num_frames) * avg + avg / 2.0).astype(int), 0, total - 1)
        want = {int(i) for i in idx}
        last = max(want)
        cache: dict[int, object] = {}
        cur = 0
        while cur <= last:
            ok, fr = cap.read()
            if not ok:
                break
            if cur in want:
                cache[cur] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            cur += 1
        if not cache:
            raise RuntimeError(f"解码出 0 帧: {video}")
        avail_last = max(cache)
        frames_out = [cache[int(i)] if int(i) in cache else cache[avail_last] for i in idx]
        return frames_out, {
            "total_frames": total, "fps": fps,
            "decoded": len(cache), "fallback": False,
        }
    finally:
        cap.release()


def preprocess_frames(frames, size: int = CROP):
    """RGB HWC uint8 -> (T,3,224,224) float32；上游 caffe 常数在 0-255 尺度上归一化。"""
    import numpy as np
    import torch
    from PIL import Image

    return _preprocess(frames, MEAN, STD, size)


# OpenAI CLIP 的原生归一化（clip/clip.py:85 的 (0.4814,0.4578,0.4082)/(0.2686,0.2613,0.2758) ×255）。
# 基线必须用自己训练时的常数，否则对比不公平。
CLIP_MEAN = [122.7709, 116.7460, 104.0937]
CLIP_STD = [68.5005, 66.6325, 70.3233]


def preprocess_frames_clip(frames, size: int = CROP):
    """CLIP 基线专用预处理（CLIP 原生 mean/std，Resize 短边 224 + CenterCrop，与 clip._transform 一致）。"""
    import numpy as np
    import torch
    from PIL import Image

    return _preprocess(frames, CLIP_MEAN, CLIP_STD, size, resize_short=size)


def _preprocess(frames, mean, std, size, resize_short: int = RESIZE_SHORT):
    import numpy as np
    import torch
    from PIL import Image

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    out = []
    for fr in frames:
        img = Image.fromarray(fr.astype(np.uint8), "RGB")
        w, h = img.size
        short = min(w, h)
        scale = resize_short / float(short)
        nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
        img = img.resize((nw, nh), Image.BICUBIC)
        left, top = (nw - size) // 2, (nh - size) // 2
        img = img.crop((left, top, left + size, top + size))
        arr = (np.asarray(img, dtype=np.float32) - mean) / std
        out.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
    return torch.stack(out, dim=0)


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def build_animalclip(state_dict: dict, device: str):
    """由 state_dict 构建 XCLIP（T=8 / prompts_layers=3 / mit_layers=1 / use_cache=False）。

    use_cache=False 是必须的：`cache_text` 只对 140/12/21/14 类做白名单，
    其它类别数会走 view(n/50,50,512) 直接 RuntimeError（实测）。
    T 必须为 8：mit 的 positional_embedding 与 message token 形状写死。
    """
    from models import model_builder

    model = model_builder.build_model(
        dict(state_dict), T=NUM_FRAMES, prompts_layers=3, mit_layers=1, use_cache=False
    )
    return model.to(device).eval()


def load_animalclip_checkpoint(path: Path) -> tuple[dict, dict]:
    """读 checkpoint。utils/tools.py:254 存的是包装 dict，需取 ['model']。

    torch 2.11 默认 weights_only=True 会拒绝含 yacs CfgNode 的 pickle（实测
    UnpicklingError），因此必须显式 weights_only=False —— 这意味着信任该 pickle。
    """
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        meta = {k: ckpt.get(k) for k in ("epoch", "max_accuracy", "config")}
        sd = ckpt["model"]
    else:
        meta, sd = {}, ckpt
    sd = {k: v for k, v in sd.items() if not k.startswith("module.")}
    return sd, meta


def state_dict_report(model, sd: dict) -> dict:
    """strict=False 会静默吞掉不匹配，这里显式列出 missing/unexpected。"""
    mk, ck = set(model.state_dict()), set(sd)
    missing, unexpected = sorted(mk - ck), sorted(ck - mk)
    return {
        "model_keys": len(mk),
        "ckpt_keys": len(ck),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_sample": missing[:20],
        "unexpected_sample": unexpected[:20],
    }


def build_clip_baseline(device: str):
    """OpenAI CLIP ViT-B/16 对照基线。vendored clip.load 返回 (model, state_dict)。"""
    import clip

    model, _ = clip.load(str(WEIGHTS_DIR / BACKBONE_FILE), device=device, jit=False)
    return model.float().eval()


def clip_tokenize(texts):
    import clip

    return clip.tokenize(texts, truncate=True)


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
def build_edges(n_animals: int, n_actions: int):
    """canis 与全部行为连边 + 自环，与上游 get_relation 的构造方式一致。"""
    edges = []
    for a in range(n_animals):
        for c in range(n_actions):
            edges.append((a, n_animals + c))
            edges.append((n_animals + c, a))
    for k in range(n_animals + n_actions):
        edges.append((k, k))
    return edges


def run_animalclip(model, img, labels, animal_names, edges, animal_idx, device):
    import torch
    from utils.tools import generate_text

    # generate_text 期望 [[id, name], ...]（内部 `for i, c in data`）
    text = generate_text([[i, t] for i, t in enumerate(labels)]).to(device)
    animal_text = generate_text([[i, t] for i, t in enumerate(animal_names)]).to(device)
    animal_pred = torch.zeros(1, len(animal_names), device=device)
    animal_pred[0, animal_idx] = 1.0
    with torch.no_grad():
        logits, vf = model(img, text, animal_text, animal_pred, edges)
    return logits[0].float().cpu().tolist(), float(vf[0].norm())


def run_clip_baseline(model, img, labels, device):
    """CLIP 是图像模型：8 帧逐帧编码后取平均作为视频表征。"""
    import torch

    prompts = [f"a video of a dog {name}" for name in labels]
    with torch.no_grad():
        tf = model.encode_text(clip_tokenize(prompts).to(device))
        tf = tf / tf.norm(dim=-1, keepdim=True)
        imf = model.encode_image(img.to(device))
        imf = imf / imf.norm(dim=-1, keepdim=True)
        imf = imf.mean(dim=0, keepdim=True)
        imf = imf / imf.norm(dim=-1, keepdim=True)
        logits = (100.0 * imf @ tf.T).squeeze(0)
    return logits.float().cpu().tolist()


@dataclass
class FragResult:
    fragment_id: str
    video_path: str
    duration_s: float
    decode: dict = field(default_factory=dict)
    # label_space -> {"logits","probs","top1","top1_conf"}
    animalclip: dict = field(default_factory=dict)
    clip_baseline: dict = field(default_factory=dict)
    vf_norm: float = 0.0
    error: str | None = None
    clip_error: str | None = None


def softmax(xs):
    import math

    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def argmax_name(xs, names):
    k = max(range(len(xs)), key=lambda j: xs[j])
    return names[k], k


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def run_pipeline_self_test(sample_results, label_sets, animal_names, canis_idx,
                           labels_csv, relation_txt, device, backbone_info) -> dict:
    """用 OpenAI CLIP 骨干 + **随机初始化**的 Animal-CLIP 新增头跑完整 XCLIP.forward。

    目的只有一个：证明「读 mp4 → 抽 8 帧 → 预处理 → XCLIP.forward(image,text,
    animal_text,animal_pred,edges) → softmax」这条管线在本机真实可跑。
    随机头的输出没有任何语义意义，**不得**当作模型结果引用。
    """
    import torch

    out: dict = {
        "purpose": "管线连通性自检：随机初始化的 Animal-CLIP 头，输出无语义意义，"
                   "不得作为模型结果引用。",
        "n_clips": 0,
        "ok": False,
    }
    if backbone_info.get("status") != "present":
        out["error"] = f"CLIP 骨干不可用: {backbone_info.get('status')}"
        return out
    try:
        import clip as _clip

        # 注意：torch.jit.load 在 Windows 上对非 ASCII 绝对路径会 fopen 失败，
        # 必须传文件对象（clip.clip.load 内部也是这么做的）。
        with open(backbone_info["path"], "rb") as fh:
            sd = torch.jit.load(fh, map_location="cpu").state_dict()
        model = build_animalclip(sd, device)
        out["randomly_initialised_params_m"] = round(
            sum(p.numel() for p in model.parameters()) / 1e6, 1)
        out["note"] = ("骨干来自 OpenAI CLIP ViT-B/16；prompts_generator1/2、graph、mit、"
                       "prompts_visual_* 全部随机初始化（XCLIP.__init__ 默认行为）。")
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"build: {type(exc).__name__}: {exc}"
        return out

    try:
        from utils.tools import get_relation

        edges_native = (torch.tensor(get_relation(str(labels_csv), str(relation_txt))).to(device)
                        if "native" in label_sets else None)
        edges_dog = (torch.tensor(build_edges(len(animal_names), len(DOGCENTRIC_LABELS))).to(device)
                     if "dogcentric" in label_sets else None)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"edges: {type(exc).__name__}: {exc}"
        return out

    per_clip = []
    ok = 0
    for rec in sample_results:
        if not rec.decode:
            continue
        try:
            frames, _ = decode_uniform_frames(Path(rec.video_path))
            img = preprocess_frames(frames).unsqueeze(0).to(device)
            row = {"fragment_id": rec.fragment_id, "spaces": {}}
            for space, labels in label_sets.items():
                e = edges_native if space == "native" else edges_dog
                logits, vf = run_animalclip(model, img, labels, animal_names, e,
                                            canis_idx if canis_idx is not None else 0, device)
                probs = softmax(logits)
                row["spaces"][space] = {
                    "logits_shape": [1, len(logits)],
                    "logits_sum": round(sum(logits), 4),
                    "softmax_sum": round(sum(probs), 6),
                    "top1": labels[max(range(len(logits)), key=lambda j: logits[j])],
                    "finite": all(map(_is_finite, logits)),
                }
            row["video_feature_norm"] = round(vf, 4)
            per_clip.append(row)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            per_clip.append({"fragment_id": rec.fragment_id,
                             "error": f"{type(exc).__name__}: {exc}"})
    out["n_clips"] = ok
    out["clips"] = per_clip
    out["ok"] = ok == len(sample_results) and ok > 0
    out["checks"] = {
        "all_forwards_ran": out["ok"],
        "all_logits_finite": all(
            v for r in per_clip for sp in r.get("spaces", {}).values() for v in [sp["finite"]]
        ) if per_clip else False,
        "softmax_sums_to_one": all(
            abs(sp["softmax_sum"] - 1.0) < 1e-4
            for r in per_clip for sp in r.get("spaces", {}).values()
        ) if per_clip else False,
    }
    return out


def _is_finite(x: float) -> bool:
    import math

    return math.isfinite(x)


def summarize(results: list[FragResult], label_sets: dict[str, list[str]]) -> dict:
    """只做分布统计。没有 GT 就绝不算 acc / F1。"""
    out: dict = {
        "note": "208 段片段 label_status 全部为 rule_seed_pending（无行为标签），"
                "因此只给分布统计，不计算任何准确率/F1。",
    }
    for model_tag, attr in (("animalclip", "animalclip"), ("clip_baseline", "clip_baseline")):
        per_space = {}
        for space, labels in label_sets.items():
            recs = [getattr(r, attr).get(space) for r in results if getattr(r, attr).get(space)]
            if not recs:
                per_space[space] = {"n": 0, "note": "未产生该模型/标签空间的预测"}
                continue
            vals = [r["top1"] for r in recs]
            confs = [r["top1_conf"] for r in recs]
            c = collections.Counter(vals)
            per_space[space] = {
                "n": len(vals),
                "n_classes": len(labels),
                "n_classes_observed": len(c),
                "top1_counts": dict(c.most_common()),
                "top1_share": {k: round(v / len(vals), 4) for k, v in c.most_common()},
                "confidence": {
                    "mean": round(sum(confs) / len(confs), 4),
                    "min": round(min(confs), 4),
                    "max": round(max(confs), 4),
                    "median": round(sorted(confs)[len(confs) // 2], 4),
                },
            }
        out[model_tag] = per_space
    out["decode_errors"] = sum(1 for r in results if r.error)
    out["clip_errors"] = sum(1 for r in results if r.clip_error)
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def find_video(fragments_dir: Path, fid: str) -> Path | None:
    for cand in (fid, fid.replace("w42v-", "w25v-"), fid.replace("w25v-", "w42v-")):
        p = fragments_dir / f"{cand}.mp4"
        if p.exists():
            return p
    return None


def read_label_names(csv_path: Path) -> list[str]:
    import csv

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"空标签表: {csv_path}")
    col = "name" if "name" in rows[0] else list(rows[0])[-1]
    return [r[col] for r in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Animal-CLIP behaviour inference (K9)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fragments-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--dataset", default="mmnet", choices=sorted(DATASETS))
    ap.add_argument("--label-space", default="both", choices=["native", "dogcentric", "both"],
                    help="native=数据集自带标签表；dogcentric=DogCentric 10 类开放词表迁移；"
                         "both=同一遍抽帧跑两套标签（默认，省一半解码）")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--device", default="cuda" if _cuda_ok() else "cpu")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--self-test", type=int, default=0, metavar="N",
                    help="管线自检：用 OpenAI CLIP 骨干 + 随机初始化的 Animal-CLIP 新增头，"
                         "在前 N 段真实视频上跑完整 forward，仅证明管线打通。"
                         "结果写入 JSON 的 pipeline_self_test 字段，绝不计入结果/指标。")
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    fragments_dir = Path(args.fragments_dir).resolve()
    out_json = Path(args.out_json).resolve()
    if not manifest_path.exists() or not fragments_dir.is_dir():
        print(f"[FATAL] 输入不存在: manifest={manifest_path} dir={fragments_dir}", file=sys.stderr)
        return 2
    out_json.parent.mkdir(parents=True, exist_ok=True)

    _bootstrap_vendored_repo()
    import torch

    rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    missing = [r["fragment_id"] for r in rows if find_video(fragments_dir, r["fragment_id"]) is None]
    if missing:
        print(f"[FATAL] {len(missing)}/{len(rows)} 条 manifest 找不到视频，前 5: {missing[:5]}",
              file=sys.stderr)
        return 3

    weights = resolve_weights(args.dataset)
    backbone = resolve_backbone()
    label_rel, animal_rel, relation_rel, n_classes, _wkey = DATASETS[args.dataset]
    labels_csv = ANIMAL_CLIP_ROOT / label_rel
    animal_csv = ANIMAL_CLIP_ROOT / animal_rel
    relation_txt = ANIMAL_CLIP_ROOT / relation_rel

    native_actions = read_label_names(labels_csv)
    animal_names = read_label_names(animal_csv)
    canis_idx = animal_names.index("canis") if "canis" in animal_names else None

    if args.label_space == "both":
        label_sets = {"native": native_actions, "dogcentric": DOGCENTRIC_LABELS}
    elif args.label_space == "dogcentric":
        label_sets = {"dogcentric": DOGCENTRIC_LABELS}
    else:
        label_sets = {"native": native_actions}

    run_meta = {
        "script": str(Path(__file__).resolve()),
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset": args.dataset,
        "label_space": args.label_space,
        "label_sets": {k: len(v) for k, v in label_sets.items()},
        "native_labels": len(native_actions),
        "animal_labels": len(animal_names),
        "canis_index": canis_idx,
        "num_frames": NUM_FRAMES,
        "use_cache": False,
        "preprocess": {
            "resize_short_side": RESIZE_SHORT, "center_crop": CROP,
            "mean": MEAN, "std": STD, "color_order": "RGB",
            "interpolation": "BICUBIC", "scale": "0-255 (no /255)",
        },
        "frame_sampling": "SampleFrames(test_mode=True, clip_len=1, frame_interval=1)",
        "weights": weights,
        "backbone": backbone,
        "network_probe": network_probe(),
        "n_manifest": len(rows),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    # --- Animal-CLIP 权重 ---
    status, reason = "ok", None
    model = None
    sd_meta = {}
    if weights["status"] == "present":
        try:
            sd, sd_meta = load_animalclip_checkpoint(Path(weights["expected_file"]))
            model = build_animalclip(sd, args.device)
            run_meta["state_dict_report"] = state_dict_report(model, sd)
            run_meta["checkpoint_meta"] = {
                k: (str(v)[:200] if k == "config" else v) for k, v in sd_meta.items()
            }
            if run_meta["state_dict_report"]["missing_count"] > 0:
                status = "partial"
                reason = ("checkpoint 与 XCLIP 结构不完全匹配，missing="
                          f"{run_meta['state_dict_report']['missing_count']}；已记录，不算可用权重。")
        except Exception as exc:  # noqa: BLE001
            status, reason = "partial", f"权重加载失败: {type(exc).__name__}: {exc}"
    else:
        status = "partial"
        reason = (
            f"Animal-CLIP 数据集专用权重不可得（{weights['expected_file']} status="
            f"{weights['status']}）。README.md:26 的唯一托管方 Google Drive 在本机 TCP 层不可达，"
            "HuggingFace / hf-mirror / ModelScope / GitHub Releases / Gitee 均无镜像（本轮实测）。"
            "因此本次只产出 OpenAI CLIP ViT-B/16 对照基线；Animal-CLIP 列一律留空不伪造。"
        )

    # --- CLIP 基线 ---
    clip_model = None
    if not args.skip_baseline and backbone["status"] == "present":
        try:
            clip_model = build_clip_baseline(args.device)
            run_meta["clip_baseline"] = {"status": "loaded",
                                         "params_m": round(sum(p.numel() for p in clip_model.parameters()) / 1e6, 1)}
        except Exception as exc:  # noqa: BLE001
            run_meta["clip_baseline"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    else:
        run_meta["clip_baseline"] = {"status": "skipped", "backbone": backbone["status"]}

    edges = None
    if model is not None:
        if "native" in label_sets:
            from utils.tools import get_relation

            edges = torch.tensor(get_relation(str(labels_csv), str(relation_txt))).to(args.device)
        edges_dog = (torch.tensor(build_edges(len(animal_names), len(DOGCENTRIC_LABELS))).to(args.device)
                     if "dogcentric" in label_sets else None)

    results: list[FragResult] = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        fid = row["fragment_id"]
        video = find_video(fragments_dir, fid)
        rec = FragResult(fid, str(video), float(row.get("duration_s", 0.0)))
        try:
            frames, rec.decode = decode_uniform_frames(video)
        except Exception as exc:  # noqa: BLE001
            rec.error = f"decode: {type(exc).__name__}: {exc}"
            results.append(rec)
            continue

        img = preprocess_frames(frames)

        if model is not None:
            for space, labels in label_sets.items():
                e = edges if space == "native" else edges_dog
                try:
                    lg, vnorm = run_animalclip(
                        model, img.unsqueeze(0), labels, animal_names, e,
                        canis_idx if canis_idx is not None else 0, args.device)
                    probs = softmax(lg)
                    top1, _ = argmax_name(lg, labels)
                    rec.animalclip[space] = {
                        "logits": lg, "probs": probs, "top1": top1,
                        "top1_conf": probs[labels.index(top1)],
                    }
                    rec.vf_norm = vnorm
                except Exception as exc:  # noqa: BLE001
                    rec.error = f"animalclip[{space}]: {type(exc).__name__}: {exc}"

        if clip_model is not None:
            img_clip = preprocess_frames_clip(frames)
            for space, labels in label_sets.items():
                try:
                    lg = run_clip_baseline(clip_model, img_clip, labels, args.device)
                    probs = softmax(lg)
                    top1, _ = argmax_name(lg, labels)
                    rec.clip_baseline[space] = {
                        "logits": lg, "probs": probs, "top1": top1,
                        "top1_conf": probs[labels.index(top1)],
                    }
                except Exception as exc:  # noqa: BLE001
                    rec.clip_error = f"clip[{space}]: {type(exc).__name__}: {exc}"

        results.append(rec)
        if i % 20 == 0 or i == len(rows):
            ac = rec.animalclip.get("dogcentric", {}).get("top1", "-")
            cl = rec.clip_baseline.get("dogcentric", {}).get("top1", "-")
            print(f"[{i}/{len(rows)}] {fid} ac_dog={ac} clip_dog={cl} err={bool(rec.error)}",
                  flush=True)

    elapsed = time.time() - t0

    # --- 管线自检（可选，明确标注为"非结果"） --------------------------------
    self_test = None
    if args.self_test > 0:
        self_test = run_pipeline_self_test(
            results[: args.self_test], label_sets, animal_names, canis_idx,
            labels_csv, relation_txt, args.device, backbone)

    payload = {
        "schema": "k9.animalclip.zeroshot.v1",
        "status": status,
        "status_reason": reason,
        "run": run_meta,
        "labels": {k: v for k, v in label_sets.items()},
        "native_labels": native_actions,
        "dogcentric_labels": DOGCENTRIC_LABELS,
        "dogcentric_to_mmnet": DOGCENTRIC_TO_MMNET,
        "elapsed_s": round(elapsed, 2),
        "n_results": len(results),
        "results": [
            {
                "fragment_id": r.fragment_id,
                "video_path": r.video_path,
                "duration_s": r.duration_s,
                "decode": r.decode,
                "animalclip": r.animalclip or None,
                "clip_baseline": r.clip_baseline or None,
                "video_feature_norm": r.vf_norm,
                "error": r.error,
                "clip_error": r.clip_error,
            }
            for r in results
        ],
        "summary": summarize(results, label_sets),
        "pipeline_self_test": self_test,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {out_json}\n     status={status} n={len(results)} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
