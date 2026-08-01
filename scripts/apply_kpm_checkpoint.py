"""从 checkpoint.h5 加载 keypoint-MoSeq 模型并生成 syllable 标注.

Phase 2 数据策略 Path 1 备用入口：
当 fit_model 训练中断或卡住时，从最近 checkpoint 直接 apply_model 生成 results.h5。

输入:
    - data/kpm_project/interpet4d_kpm/checkpoint.h5（fit_model 保存的 checkpoint）
    - data/interpet4d/smal_npy/*.npz（用于重新 format_data）

输出:
    - data/kpm_project/interpet4d_kpm/results.h5（含每个 clip 的 syllable 标注）

依据:
    - keypoint-MoSeq: https://github.com/dattalab/keypoint-moseq
    - kpm.load_checkpoint + kpm.apply_model API

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 2.0d（数据策略 Path 1）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# JAX 64-bit 精度必须在 import keypoint_moseq 之前启用
import jax
jax.config.update("jax_enable_x64", True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 复用 train_keypoint_moseq.py 的常量和数据加载函数
from scripts.train_keypoint_moseq import (
    KPT_NAMES, ANTERIOR_IDXS, POSTERIOR_IDXS,
    load_interpet4d_keypoints,
)

DEFAULT_KPM_DIR = PROJECT_ROOT / "data" / "kpm_project"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "interpet4d" / "smal_npy"
DEFAULT_MODEL_NAME = "interpet4d_kpm"


def apply_checkpoint(
    project_dir: Path,
    data_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    apply_iters: int = 50,
) -> None:
    """从 checkpoint 加载模型并 apply 到数据.

    Args:
        project_dir: kpm 项目目录
        data_dir: InterPet4D smal_npy 目录
        model_name: 模型名称（checkpoint.h5 所在子目录名）
        apply_iters: apply_model 推理迭代次数
    """
    try:
        import keypoint_moseq as kpm
    except ImportError:
        print("[ERROR] keypoint-moseq 未安装", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = project_dir / model_name / "checkpoint.h5"
    if not checkpoint_path.exists():
        print(f"[ERROR] checkpoint 不存在: {checkpoint_path}", file=sys.stderr)
        print("请先运行: python scripts/train_keypoint_moseq.py", file=sys.stderr)
        sys.exit(1)

    print(f"[load] 加载 checkpoint: {checkpoint_path}")
    print(f"       文件大小: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Step 1: 加载 checkpoint（含 model + data + metadata + iteration）
    model, data, metadata, iteration = kpm.load_checkpoint(
        project_dir=str(project_dir),
        model_name=model_name,
    )
    print(f"[load] checkpoint 已加载（iteration={iteration}）")
    print(f"[load] model keys: {list(model.keys())[:8]}...")
    print(f"[load] data Y shape: {data['Y'].shape}")
    print(f"[load] metadata: {len(metadata[0])} recordings")

    # 转换数据精度到 64-bit（与训练一致）
    from jax_moseq.utils.debugging import convert_data_precision
    data = convert_data_precision(data, x64=True)

    # Step 2: apply_model 生成 syllable 标注
    # apply_model 内部会 init_model(data, seed, params, hypparams) 重初始化
    # 但 apply_model 不传 noise_prior / anterior_idxs / posterior_idxs
    # 必须通过 kwargs 补全这些参数才能通过 _check_init_args 校验
    print(f"[apply] 运行 apply_model（num_iters={apply_iters}）...")
    t0 = time.time()
    results = kpm.apply_model(
        model=model,
        data=data,
        metadata=metadata,
        project_dir=str(project_dir),
        model_name=model_name,
        num_iters=apply_iters,
        ar_only=False,
        save_results=True,
        verbose=True,
        noise_prior=model["noise_prior"],
        anterior_idxs=ANTERIOR_IDXS,
        posterior_idxs=POSTERIOR_IDXS,
    )
    elapsed = time.time() - t0
    print(f"[apply] 完成，耗时 {elapsed:.1f}s")

    # Step 3: 统计 syllable 分布
    syllable_counts: dict[int, int] = {}
    results_path = project_dir / model_name / "results.h5"
    if results_path.exists():
        import h5py
        with h5py.File(str(results_path), "r") as f:
            clip_count = 0
            for clip_name in f.keys():
                if "syllable" in f[clip_name]:
                    syllables = np.asarray(f[clip_name]["syllable"]).flatten()
                    for s in syllables:
                        sid = int(s)
                        syllable_counts[sid] = syllable_counts.get(sid, 0) + 1
                    clip_count += 1
        print(f"\n[stats] {clip_count} clips, {len(syllable_counts)} syllables")
        if syllable_counts:
            total = sum(syllable_counts.values())
            print(f"[stats] 总帧数: {total}")
            print("[stats] Top 10 syllables:")
            for sid, count in sorted(syllable_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  syllable {sid:3d}: {count:6d} frames ({count / total * 100:.1f}%)")
    else:
        print(f"[WARN] results.h5 未生成: {results_path}", file=sys.stderr)

    print(f"\n[done] syllable 标注生成完成！")
    print(f"下一步: python scripts/map_syllables_to_behaviors.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 checkpoint 生成 syllable 标注")
    parser.add_argument(
        "--kpm-dir", type=Path, default=DEFAULT_KPM_DIR,
        help=f"kpm 项目目录（默认: {DEFAULT_KPM_DIR}）",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"InterPet4D smal_npy 目录（默认: {DEFAULT_DATA_DIR}）",
    )
    parser.add_argument(
        "--model-name", type=str, default=DEFAULT_MODEL_NAME,
        help=f"模型名称（默认: {DEFAULT_MODEL_NAME}）",
    )
    parser.add_argument(
        "--apply-iters", type=int, default=50,
        help="apply_model 推理迭代次数（默认 50）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("keypoint-MoSeq checkpoint → syllable 标注")
    print(f"kpm 目录: {args.kpm_dir}")
    print(f"数据目录: {args.data_dir}")
    print(f"模型名称: {args.model_name}")
    print(f"推理迭代: {args.apply_iters}")
    print("=" * 60)

    apply_checkpoint(
        project_dir=args.kpm_dir,
        data_dir=args.data_dir,
        model_name=args.model_name,
        apply_iters=args.apply_iters,
    )


if __name__ == "__main__":
    main()
