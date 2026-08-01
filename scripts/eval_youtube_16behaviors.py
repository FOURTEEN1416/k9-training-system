"""Phase 2.0c: 1.2f 真实视频 16 行为定性验证.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 2.0c
依据: dev-docs/stages/phase-2.md §2.0c

目的:
    2.2 完成（16 行为规则引擎）后，用真实 YouTube 犬类视频重测行为识别能力。
    由于 YouTube 视频无 ground truth 标签，本脚本做定性验证（行为分布合理性），
    非定量准确率。定量准确率阻塞于 2.0b Label Studio 人工标注。

流程:
    1. 遍历 data/youtube_self_label/raw/*.mp4
    2. YOLO26-pose 推理 → keypoints 序列 (T, 24, 3)
    3. 16 行为规则引擎识别 → episodes
    4. 输出行为分布 + 代表性片段

用法:
    python scripts/eval_youtube_16behaviors.py
    python scripts/eval_youtube_16behaviors.py --limit 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.behavior.rule_engine import RuleEngine
from backend.ml.pose.inference import PoseInferenceEngine

YOUTUBE_DIR = PROJECT_ROOT / "data" / "youtube_self_label" / "raw"
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "train-2" / "weights" / "best.onnx"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description="1.2f YouTube 16 行为定性验证")
    parser.add_argument("--limit", type=int, default=None, help="采样上限")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO26-pose 模型路径")
    args = parser.parse_args()

    videos = sorted(YOUTUBE_DIR.glob("*.mp4"))
    if not videos:
        print(f"[ERROR] 未在 {YOUTUBE_DIR} 发现 YouTube 视频")
        return 1
    if args.limit:
        videos = videos[: args.limit]

    print(f"\n{'=' * 70}")
    print(f"1.2f YouTube 16 行为定性验证")
    print(f"视频数: {len(videos)}")
    print(f"模型: {args.model}")
    print(f"{'=' * 70}\n")

    # 加载模型（单例）
    model_path = str(args.model) if args.model.exists() else "yolo26n-pose.pt"
    print(f"[init] 加载 pose 模型: {model_path}")
    engine = PoseInferenceEngine(model_path=model_path, verbose=False)
    rule_engine = RuleEngine()
    print(f"[init] 加载完成\n")

    all_results = []
    total_episodes = 0
    behavior_dist: dict[str, int] = {}

    for i, video_path in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video_path.name}")
        t0 = time.time()
        try:
            result = engine.infer_video(video_path=video_path, save_output=False)
            kpts_seq = result.keypoints_sequence  # (T, 24, 3)
            fps = result.meta.get("fps", 30.0)
            duration = result.meta.get("duration_sec", 0.0)
            num_frames = len(result.frames)
            # 真实检测率 = 非零 box_conf 的帧占比（inference.py meta 不含 detection_rate）
            detected_frames = sum(1 for f in result.frames if f.box_conf > 0.0)
            detect_rate = detected_frames / num_frames if num_frames > 0 else 0.0

            episodes = rule_engine.recognize(kpts_seq, fps=fps)
            elapsed = time.time() - t0

            # 行为分布
            video_behaviors: dict[str, int] = {}
            for ep in episodes:
                b = ep.behavior
                video_behaviors[b] = video_behaviors.get(b, 0) + 1
                behavior_dist[b] = behavior_dist.get(b, 0) + 1

            total_episodes += len(episodes)
            all_results.append({
                "video": video_path.name,
                "duration_sec": round(duration, 1),
                "frames": num_frames,
                "fps": round(fps, 1),
                "detect_rate": round(detect_rate, 3),
                "episodes": len(episodes),
                "behaviors": video_behaviors,
                "elapsed_sec": round(elapsed, 1),
            })

            print(f"  时长: {duration:.1f}s | 帧: {num_frames} | 检测率: {detect_rate:.1%}")
            print(f"  行为: {len(episodes)} episodes | {video_behaviors}")
            print(f"  耗时: {elapsed:.1f}s\n")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [FAIL] {type(e).__name__}: {e} ({elapsed:.1f}s)\n")
            all_results.append({
                "video": video_path.name,
                "error": f"{type(e).__name__}: {e}",
                "elapsed_sec": round(elapsed, 1),
            })

    # 汇总
    print(f"{'=' * 70}")
    print(f"汇总")
    print(f"  视频数: {len(all_results)}")
    print(f"  总 episodes: {total_episodes}")
    print(f"  行为分布: {behavior_dist}")
    print(f"  行为种类: {len(behavior_dist)}")
    print(f"{'=' * 70}\n")

    # 写报告
    import json
    report_path = REPORTS_DIR / "phase-2.0c-youtube-16behaviors-validation.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2.0c — 1.2f YouTube 16 行为定性验证 报告\n\n")
        f.write(f"> 验证日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"> 视频数: {len(all_results)}\n")
        f.write(f"> 模型: {model_path}\n")
        f.write(f"> 验证类型: 定性（无 ground truth，定量准确率阻塞于 2.0b 人工标注）\n\n")
        f.write("## 汇总\n\n")
        f.write(f"- 总 episodes: {total_episodes}\n")
        f.write(f"- 行为种类: {len(behavior_dist)}\n")
        f.write(f"- 行为分布: {json.dumps(behavior_dist, ensure_ascii=False, indent=2)}\n\n")
        f.write("## 逐视频详情\n\n")
        f.write("| 视频 | 时长(s) | 帧 | 检测率 | episodes | 行为 | 耗时(s) |\n")
        f.write("|------|---------|----|--------|---------:|------|--------:|\n")
        for r in all_results:
            if "error" in r:
                f.write(f"| {r['video']} | - | - | - | - | ERROR: {r['error'][:40]} | {r['elapsed_sec']} |\n")
            else:
                behaviors_str = ", ".join(f"{k}:{v}" for k, v in sorted(r["behaviors"].items()))
                f.write(
                    f"| {r['video']} | {r['duration_sec']} | {r['frames']} | "
                    f"{r['detect_rate']:.1%} | {r['episodes']} | {behaviors_str} | {r['elapsed_sec']} |\n"
                )
        f.write("\n## 结论\n\n")
        f.write("- 本验证为定性验证（YouTube 视频无 ground truth 标签）\n")
        f.write("- 定量准确率（≥80%）阻塞于 2.0b Label Studio 人工标注\n")
        f.write("- kp_world 替代验证已通过（10/10 clips，10 种行为，见 reports/phase-2-prereq-1.2f-validation.md）\n")
        f.write("- 合成数据基线 92.9%（reports/phase-1.2f-validation.md）\n")

    print(f"报告已写入: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
