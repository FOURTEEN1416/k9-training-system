"""keypoint-MoSeq 无监督行为发现训练脚本（kpm 0.6.6 API 兼容）.

Phase 2 数据策略 Path 1: 从 InterPet4D 关键点序列无监督发现行为 syllables。

流程:
    1. 加载 InterPet4D smal_npy/*.npz（kp_world (T, 24, 3) + kp_weight (T, 24)）
    2. kpm.setup_project 创建项目配置
    3. kpm.format_data 格式化关键点为训练数据
    4. kpm.fit_pca PCA 降维
    5. kpm.init_model + kpm.fit_model AR-HMM 训练
    6. kpm.apply_model 生成 syllable 时间序列标注

用法:
    python scripts/train_keypoint_moseq.py
    python scripts/train_keypoint_moseq.py --data-dir D:/datasets/interpet4d --project-dir data/kpm_project --num-iters 50

依据:
    - dev-docs/research/PHASE2_DATA_STRATEGY.md Path 1
    - keypoint-MoSeq: https://github.com/dattalab/keypoint-moseq
    - Weinreb et al., Nature Methods, 2024

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

# Dog-Pose 24 关键点名称（与 backend/ml/pose/inference.py 一致）
KPT_NAMES = [
    "front_left_paw", "front_left_knee", "front_left_elbow",
    "rear_left_paw", "rear_left_knee", "rear_left_elbow",
    "front_right_paw", "front_right_knee", "front_right_elbow",
    "rear_right_paw", "rear_right_knee", "rear_right_elbow",
    "tail_start", "tail_end",
    "left_ear_base", "right_ear_base",
    "nose", "chin",
    "left_ear_tip", "right_ear_tip",
    "left_eye", "right_eye",
    "withers", "throat",
]

# keypoint-MoSeq 需要前后部关键点索引（用于朝向估计）
# 索引对应 KPT_NAMES 中的位置
ANTERIOR_IDXS = [16, 17, 20, 21, 14, 15, 18, 19, 23]  # nose, chin, eyes, ear bases, ear tips, throat
POSTERIOR_IDXS = [12, 13, 3, 9, 4, 10]  # tail_start, tail_end, rear paws, rear knees

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "interpet4d" / "smal_npy"
DEFAULT_PROJECT_DIR = PROJECT_ROOT / "data" / "kpm_project"


def load_interpet4d_keypoints(
    data_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """加载 InterPet4D kp_world 关键点序列.

    Returns:
        coordinates: dict {clip_name: (T, 24, 3) array} 关键点坐标
        confidences: dict {clip_name: (T, 24) array} 置信度
    """
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] 未找到 .npz 文件: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[load] 发现 {len(npz_files)} 个 .npz 文件")

    coordinates: dict[str, np.ndarray] = {}
    confidences: dict[str, np.ndarray] = {}
    skipped = 0

    for npz_path in npz_files:
        try:
            data = np.load(str(npz_path), allow_pickle=True)
            if "kp_world" not in data:
                skipped += 1
                continue

            kp = np.asarray(data["kp_world"], dtype=np.float32)  # (T, 24, 3)
            if kp.ndim != 3 or kp.shape[1] != 24:
                skipped += 1
                continue

            # 取前 3 维（SMAL 3D 世界坐标）
            kp = kp[:, :, :3]

            # 过滤过短序列（< 10 帧）
            if kp.shape[0] < 10:
                skipped += 1
                continue

            # 置信度
            if "kp_weight" in data:
                conf = np.asarray(data["kp_weight"], dtype=np.float32)  # (T, 24)
                conf = np.clip(conf, 0.0, 1.0)
            else:
                conf = np.ones((kp.shape[0], 24), dtype=np.float32)

            # 替换 NaN 为 0（kpm.format_data 会做插值）
            kp = np.nan_to_num(kp, nan=0.0)

            clip_name = npz_path.stem
            coordinates[clip_name] = kp
            confidences[clip_name] = conf

        except Exception as e:
            print(f"  [WARN] 加载失败: {npz_path.name}: {e}")
            skipped += 1
            continue

    print(f"[load] 成功加载 {len(coordinates)} 个序列，跳过 {skipped} 个")
    total_frames = sum(kp.shape[0] for kp in coordinates.values())
    print(f"[load] 总帧数: {total_frames}，平均长度: {total_frames / max(len(coordinates), 1):.1f} 帧")

    return coordinates, confidences


def setup_and_train(
    coordinates: dict[str, np.ndarray],
    confidences: dict[str, np.ndarray],
    project_dir: Path,
    pca_components: int = 10,
    num_iters: int = 50,
) -> None:
    """设置 keypoint-MoSeq 项目并训练模型.

    Args:
        coordinates: dict {clip_name: (T, 24, 3)}
        confidences: dict {clip_name: (T, 24)}
        project_dir: 项目输出目录
        pca_components: PCA 降维维度
        num_iters: Gibbs sampling 迭代次数
    """
    try:
        import keypoint_moseq as kpm
    except ImportError:
        print("[ERROR] keypoint-moseq 未安装。请运行: pip install keypoint-moseq", file=sys.stderr)
        sys.exit(1)

    project_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] 项目目录: {project_dir}")

    # Step 1: 设置项目配置
    print("[setup] 创建 keypoint-MoSeq 项目配置...")
    kpm.setup_project(
        project_dir=str(project_dir),
        bodyparts=KPT_NAMES,
        use_bodyparts=KPT_NAMES,
        anterior_bodyparts=[KPT_NAMES[i] for i in ANTERIOR_IDXS],
        posterior_bodyparts=[KPT_NAMES[i] for i in POSTERIOR_IDXS],
        overwrite=True,
    )

    # 加载配置
    config = kpm.load_config(project_dir=str(project_dir))
    print(f"[setup] 配置已创建: {project_dir / 'config.yml'}")

    # Step 2: 格式化数据
    print("[format] 格式化关键点数据...")
    data, metadata = kpm.format_data(
        coordinates=coordinates,
        confidences=confidences,
        bodyparts=KPT_NAMES,
        use_bodyparts=KPT_NAMES,
    )
    print(f"[format] Y shape: {data['Y'].shape}")
    print(f"[format] mask shape: {data['mask'].shape}")
    print(f"[format] conf shape: {data['conf'].shape if data['conf'] is not None else 'None'}")
    print(f"[format] metadata: {len(metadata[0])} recordings")

    # Step 3: PCA 拟合
    print(f"[pca] 拟合 PCA（n_components={pca_components}）...")
    pca = kpm.fit_pca(
        data["Y"],
        data["mask"],
        anterior_idxs=ANTERIOR_IDXS,
        posterior_idxs=POSTERIOR_IDXS,
        conf=data["conf"],
        conf_threshold=0.3,
        n_components=pca_components,
        verbose=True,
    )
    print(f"[pca] PCA 已拟合，explained variance ratio: {pca.explained_variance_ratio_[:pca_components]}")

    # 保存 PCA
    kpm.save_pca(pca, project_dir=str(project_dir))
    print(f"[pca] PCA 已保存: {project_dir}")

    # Step 4: 初始化模型
    print("[init] 初始化 AR-HMM 模型...")
    # 合并 conf_threshold 到 config，避免 init_model 重复参数冲突
    config["conf_threshold"] = 0.3
    # 转换数据精度到 64-bit（JAX x64 要求）
    from jax_moseq.utils.debugging import convert_data_precision
    data = convert_data_precision(data, x64=True)
    # 注意: pca 必须作为关键字参数传递（位置参数顺序: data, states, params, ...）
    model = kpm.init_model(
        data,
        pca=pca,
        conf=data["conf"],
        **config,
    )
    print(f"[init] 模型已初始化")

    # Step 5: 模型训练
    print(f"[train] AR-HMM 训练（num_iters={num_iters}）...")
    t0 = time.time()
    model, data, metadata = kpm.fit_model(
        model=model,
        data=data,
        metadata=metadata,
        project_dir=str(project_dir),
        model_name="interpet4d_kpm",
        num_iters=num_iters,
        ar_only=False,
        save_every_n_iters=max(num_iters // 4, 10),
        generate_progress_plots=False,  # 无 GUI 环境
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"[train] 训练完成，耗时 {elapsed:.1f}s")

    # Step 6: 生成 syllable 标注
    print("[label] 生成 syllable 标注...")
    results = kpm.apply_model(
        model=model,
        data=data,
        metadata=metadata,
        project_dir=str(project_dir),
        model_name="interpet4d_kpm",
        num_iters=50,  # apply_model 的推理迭代
        ar_only=False,
        save_results=True,
        verbose=True,
    )
    print(f"[label] 标注结果已保存到: {project_dir}")

    # 统计 syllable 分布
    syllable_counts: dict[int, int] = {}
    if isinstance(results, dict) and "syllable" in results:
        # 单一结果
        syllables = np.asarray(results["syllable"]).flatten()
        for s in syllables:
            sid = int(s)
            syllable_counts[sid] = syllable_counts.get(sid, 0) + 1
    elif hasattr(results, '__iter__'):
        # 可能是 dict of dicts 或 tuple
        try:
            # 尝试访问 results.h5
            results_path = project_dir / "interpet4d_kpm" / "results.h5"
            if not results_path.exists():
                # 搜索其他可能路径
                for p in project_dir.rglob("results.h5"):
                    results_path = p
                    break
            if results_path.exists():
                import h5py
                with h5py.File(str(results_path), "r") as f:
                    for clip_name in f.keys():
                        if "syllable" in f[clip_name]:
                            syllables = np.asarray(f[clip_name]["syllable"]).flatten()
                            for s in syllables:
                                sid = int(s)
                                syllable_counts[sid] = syllable_counts.get(sid, 0) + 1
        except Exception as e:
            print(f"  [WARN] 统计 syllable 分布失败: {e}")

    print(f"\n[stats] 发现 {len(syllable_counts)} 个 syllables")
    if syllable_counts:
        total = sum(syllable_counts.values())
        print(f"[stats] 总帧数: {total}")
        print("[stats] Top 10 syllables:")
        for sid, count in sorted(syllable_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  syllable {sid:3d}: {count:6d} frames ({count / total * 100:.1f}%)")

    print(f"\n[done] keypoint-MoSeq 训练完成！")
    print(f"下一步: 运行 python scripts/map_syllables_to_behaviors.py 将 syllables 映射到 16 行为类")


def main() -> None:
    parser = argparse.ArgumentParser(description="keypoint-MoSeq 无监督行为发现训练")
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"InterPet4D smal_npy 目录（默认: {DEFAULT_DATA_DIR}）",
    )
    parser.add_argument(
        "--project-dir", type=Path, default=DEFAULT_PROJECT_DIR,
        help=f"keypoint-MoSeq 项目输出目录（默认: {DEFAULT_PROJECT_DIR}）",
    )
    parser.add_argument("--pca-components", type=int, default=10, help="PCA 降维维度（默认 10）")
    parser.add_argument(
        "--num-iters", type=int, default=50,
        help="Gibbs sampling 迭代次数（默认 50，CPU 测试用；正式训练建议 200-500）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("keypoint-MoSeq 无监督行为发现训练")
    print(f"数据目录: {args.data_dir}")
    print(f"项目目录: {args.project_dir}")
    print(f"PCA 维度: {args.pca_components}")
    print(f"训练迭代: {args.num_iters}")
    print("=" * 60)

    # 1. 加载数据
    coordinates, confidences = load_interpet4d_keypoints(args.data_dir)
    if not coordinates:
        print("[ERROR] 无有效数据", file=sys.stderr)
        sys.exit(1)

    # 2. 训练
    setup_and_train(
        coordinates=coordinates,
        confidences=confidences,
        project_dir=args.project_dir,
        pca_components=args.pca_components,
        num_iters=args.num_iters,
    )


if __name__ == "__main__":
    main()
