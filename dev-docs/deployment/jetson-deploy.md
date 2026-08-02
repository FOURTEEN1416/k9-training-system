# Jetson 边缘部署手册

> 阶段: Phase 3.5
> Owner: 后端开发（`backend/`）+ ML 开发（`backend/ml/`）
> 依据: [RESEARCH_JETSON_DEPLOYMENT.md](../research/RESEARCH_JETSON_DEPLOYMENT.md) + [phase-3.md §3.5](../stages/phase-3.md)
> 状态: 3.5c 部分完成（抽帧策略 + TRT FP16 转换脚本已就绪，硬件待采购）
> 版本: v1.0（2026-08-02）

## 0. 当前状态摘要

| 子阶段 | 状态 | 已交付 | 待办 |
|--------|------|--------|------|
| 3.5a 调研 | ✅ | `RESEARCH_JETSON_DEPLOYMENT.md` (803 行) | — |
| 3.5b 环境配置 | ⏳ | — | 硬件采购 + JetPack 6.1 刷机 |
| 3.5c ONNX→TRT + FP16 + 抽帧 | 🔄 部分 | [frame_stride.py](../../backend/ml/pose/frame_stride.py) + [convert_trt_fp16.py](../../scripts/convert_trt_fp16.py) | Jetson 上实际 engine 转换 + 延迟测试 |
| 3.5d 平台抽象层 | ⏳ | 接口契约（调研报告 §6） | `InferenceBackend` + `AutoBackend` 实现 |
| 3.5e 延迟测试 | ⏳ | — | ≤ 0.5 min/min 视频实测 |

## 1. 已就绪资源

### 1.1 抽帧策略（[frame_stride.py](../../backend/ml/pose/frame_stride.py)）

`backend/ml/pose/frame_stride.py` 提供自适应抽帧 + 插值能力，已通过单元测试：

| 接口 | 用途 |
|------|------|
| `compute_infer_indices(total_frames, stride)` | 计算需推理的帧索引 `[0, stride, 2*stride, ..., last]` |
| `interpolate_keypoints(infer_kpts, infer_indices, total_frames)` | 线性/最近邻插值填充中间帧（关键点 + 检测框 + 置信度保守取 min） |
| `estimate_speedup(stride, total_frames)` | 估算加速比（stride=3 实际 2.99x，因末帧强制纳入） |
| `recommend_stride(target_speedup, total_frames, max_interpolation_error)` | 根据目标加速比 + 误差上限推荐 stride（含 `SPEEDUP_TOLERANCE=0.05` 边界处理） |

**经验误差表**（写入 `recommend_stride`）：

| stride | 推理量 | 误差 | 加速比 |
|--------|--------|------|--------|
| 1 | 100% | 0% | 1.00x |
| 2 | 50% | 3% | 1.99x |
| 3 | 33% | 5% | 2.99x |
| 4 | 25% | 8% | 3.99x |
| 5 | 20% | 10% | 4.99x |

### 1.2 TRT FP16 转换脚本（[convert_trt_fp16.py](../../scripts/convert_trt_fp16.py)）

`scripts/convert_trt_fp16.py::convert_onnx_to_trt` 在 Jetson 设备上执行：

- ONNX 解析 + builder 配置 + 动态 batch 优化 profile
- FP16 / INT8 精度切换（INT8 需额外校准数据集）
- workspace 默认 4 GiB（适配 Orin Nano 8GB 内存）
- 支持 YOLO26-pose + ST-GCN+BC + MotionBERT 三类 ONNX 模型

**调用方式**（在 Jetson 上）：

```bash
# YOLO26-pose
python scripts/convert_trt_fp16.py \
    --onnx data/models/yolo26_pose/best.onnx \
    --engine data/models/jetson/yolo26_pose_fp16.engine \
    --fp16

# ST-GCN+BC（固定 batch=1）
python scripts/convert_trt_fp16.py \
    --onnx data/models/stgcn_bc/stgcn_bc_dog24.onnx \
    --engine data/models/jetson/stgcn_bc_fp16.engine \
    --fp16 --min-batch 1 --opt-batch 1 --max-batch 1
```

### 1.3 已导出的 ONNX 模型

| 模型 | ONNX 路径 | 来源 | 用途 |
|------|-----------|------|------|
| YOLO26-pose | `data/models/yolo26_pose/best.onnx` | Phase 1.7 训练 | 24 关键点检测 |
| ST-GCN+BC | `data/models/stgcn_bc/stgcn_bc_dog24.onnx` | [export_onnx.py](../../backend/ml/behavior/stgcn_bc/export_onnx.py) epoch 21 (46.97%) | 22 类行为识别 |
| MotionBERT-Lite | `data/models/motionbert_dog24/motionbert_dog24.onnx` | Phase 3.3c 17→24 适配 | 2D→3D 姿态 lifting |

## 2. 部署前置条件

### 2.1 硬件清单（~$309）

| 项目 | 单价 | 数量 | 小计 |
|------|------|------|------|
| Jetson Orin Nano Super Developer Kit | $249 | 1 | $249 |
| 256GB NVMe SSD | ~$30 | 1 | $30 |
| 主动散热风扇（推荐） | ~$15 | 1 | $15 |
| 电源适配器（65W USB-C PD） | ~$15 | 1 | $15 |

**规格关键点**：
- AI 性能 67 INT8 TOPS，GPU 1024 CUDA + 32 Tensor Cores @ 1020 MHz
- 8GB LPDDR5 @ 102 GB/s（CPU/GPU 共享）
- **无 DLA**（Orin Nano 不支持 DLA 卸载，仅 GPU 推理）
- MAXN SUPER 模式特殊：`nvpmodel -m 2`（不是 `-m 0`）

### 2.2 软件栈（JetPack 6.1）

| 组件 | 版本 | 安装方式 |
|------|------|---------|
| JetPack | 6.1 | SDK Manager / SD 卡镜像 |
| TensorRT | 10.3+ | 随 JetPack 安装 |
| PyTorch | 2.10.0 | Jetson 专用 wheel（**非 pip 默认**） |
| torchvision | 0.25.0 | Jetson 专用 wheel |
| onnxruntime-gpu | 1.23.0 | aarch64 wheel（**PyPI 无 Jetson 二进制**） |
| cuDSS | 0.7.1 | deb 包（PyTorch 2.10.0 依赖） |
| ultralytics | latest | `pip install ultralytics[export]` |

## 3. Jetson 环境搭建

### 3.1 刷机 + 基础环境

```bash
# Step 1: 刷机 JetPack 6.1（SDK Manager 或 SD 卡镜像）

# Step 2: 基础环境
sudo apt update && sudo apt install python3-pip -y
pip install -U pip

# Step 3: 安装 ultralytics（含导出依赖）
pip install ultralytics[export]

# Step 4: 安装 PyTorch for Jetson（非 pip 默认版本）
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.10.0-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl

# Step 5: 安装 cuDSS（PyTorch 2.10.0 依赖）
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update && sudo apt-get -y install cudss

# Step 6: 安装 onnxruntime-gpu（aarch64 专用 wheel）
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl

# Step 7: 重启
sudo reboot
```

### 3.2 性能模式 + 监控

```bash
# Orin Nano Super 特殊：MAXN SUPER 模式
sudo nvpmodel -m 2
sudo jetson_clocks
sudo jetson_clocks --fan  # 风扇最大

# 监控工具
sudo pip3 install -U jetson-stats
jtop  # 实时监控 GPU/CPU/内存/温度

# 内存优化（减少 swap）
sudo sysctl -w vm.swappiness=10
```

### 3.3 Docker 容器化（可选）

```bash
# JetPack 6 官方预构建镜像
t=ultralytics/ultralytics:latest-jetson-jetpack6
sudo docker pull $t
sudo docker run -it --ipc=host --runtime=nvidia \
    -v /home/nvidia/models:/app/models \
    -v /run/jtop.sock:/run/jtop.sock \
    $t
```

## 4. 模型准备与转换

### 4.1 从 Windows 复制 ONNX 到 Jetson

```bash
# 在 Windows 上
scp data/models/yolo26_pose/best.onnx nvidia@jetson:/home/nvidia/models/
scp data/models/stgcn_bc/stgcn_bc_dog24.onnx nvidia@jetson:/home/nvidia/models/
scp data/models/motionbert_dog24/motionbert_dog24.onnx nvidia@jetson:/home/nvidia/models/
```

### 4.2 在 Jetson 上转换为 TRT FP16 引擎

```bash
cd /home/nvidia/k9-training-system

# YOLO26-pose（动态 batch 支持视频批量推理）
python scripts/convert_trt_fp16.py \
    --onnx models/yolo26_pose/best.onnx \
    --engine models/jetson/yolo26_pose_fp16.engine \
    --fp16 --dynamic-batch \
    --min-batch 1 --opt-batch 4 --max-batch 8

# ST-GCN+BC（滑动窗口单样本推理，固定 batch=1）
python scripts/convert_trt_fp16.py \
    --onnx models/stgcn_bc/stgcn_bc_dog24.onnx \
    --engine models/jetson/stgcn_bc_fp16.engine \
    --fp16 --min-batch 1 --opt-batch 1 --max-batch 1

# MotionBERT-Lite（27 帧窗口，固定 batch=1）
python scripts/convert_trt_fp16.py \
    --onnx models/motionbert_dog24/motionbert_dog24.onnx \
    --engine models/jetson/motionbert_fp16.engine \
    --fp16 --min-batch 1 --opt-batch 1 --max-batch 1
```

### 4.3 关键注意事项

1. **engine 文件不可跨平台**：Windows 上生成的 `.engine` 不能在 Jetson 上使用，必须在 Jetson 上从 `.onnx` 重新转换
2. **INT8 校准必须在目标设备**：校准结果因 GPU 架构而异，跨设备校准性能下降
3. **workspace 参数**：4 GiB 适合 Orin Nano 8GB，过大会报 `UNSUPPORTED_STATE`
4. **精度对比**：转换后用 dog-pose val 集对比 ONNX vs TRT 精度，mAP50-95 损失应 < 0.5%

## 5. 项目代码集成

### 5.1 平台抽象层（待 3.5d 实现）

按调研报告 §6 契约实现：

```
backend/ml/pose/
├── inference_backend.py    # InferenceBackend 抽象基类
├── onnx_backend.py         # OnnxRuntimeBackend（Windows 开发机）
├── tensorrt_backend.py     # TensorRtBackend（Jetson 部署）
└── auto_backend.py         # AutoBackend（自动选择）
```

**自动选择逻辑**：

```python
import platform as pf
is_jetson = "aarch64" in pf.machine().lower() or "tegra" in pf.uname().release.lower()

if is_jetson and os.path.exists(engine_path):
    backend = TensorRtBackend()  # 优先 TRT FP16
elif is_jetson:
    backend = OnnxRuntimeBackend()  # Jetson 降级 ONNX（慢但可用）
else:
    backend = OnnxRuntimeBackend()  # Windows 默认 ONNX Runtime
```

### 5.2 tasks.py 集成点

`backend/workers/tasks.py` 已有 `_resolve_stgcn_bc_path()` 优先 ONNX 回退 checkpoint 的逻辑，3.5d 实现时需扩展：

```python
# 当前（Phase 3.1e）
def _resolve_stgcn_bc_path() -> tuple[str, str]:
    onnx_path = "data/models/stgcn_bc/stgcn_bc_dog24.onnx"
    if os.path.exists(onnx_path):
        return onnx_path, "onnx"
    return checkpoint_path, "pytorch"

# 3.5d 扩展（待实现）
def _resolve_model_path(model_name: str) -> tuple[str, str]:
    """优先 TRT engine > ONNX > PyTorch checkpoint."""
    if _is_jetson():
        engine_path = f"data/models/jetson/{model_name}_fp16.engine"
        if os.path.exists(engine_path):
            return engine_path, "tensorrt"
    onnx_path = f"data/models/{model_name}/{model_name}.onnx"
    if os.path.exists(onnx_path):
        return onnx_path, "onnx"
    return f"data/models/{model_name}/best.pt", "pytorch"
```

### 5.3 抽帧策略集成

在 [tasks.py](../../backend/workers/tasks.py) 的 pose 推理循环中调用 frame_stride：

```python
from backend.ml.pose.frame_stride import (
    compute_infer_indices, interpolate_keypoints, recommend_stride
)

# 推理前：根据目标延迟推荐 stride
target_speedup = 2.0  # 目标 0.5x 实时
stride = recommend_stride(
    target_speedup=target_speedup,
    total_frames=total_frames,
    max_interpolation_error=0.05,
)
infer_indices = compute_infer_indices(total_frames, stride)

# 仅对 infer_indices 中的帧推理
infer_kpts = [backend.infer(frame[i]) for i in infer_indices]

# 插值填充中间帧
full_kpts = interpolate_keypoints(infer_kpts, infer_indices, total_frames)
```

## 6. 性能监控

### 6.1 jtop 实时监控

```bash
# 启动 jtop（TUI 实时监控）
jtop

# 关注指标:
# - GPU 利用率（应 > 80%）
# - GPU 频率（应锁定 1020 MHz）
# - 内存占用（应 < 4GB，留 4GB 给系统）
# - 温度（应 < 70°C，超过则检查散热）
# - 功耗（MAXN SUPER 模式 25W）
```

### 6.2 jetson-stats Python API

```python
from jtop import jtop

with jtop() as jetson:
    stats = jetson.stats
    print(f"GPU 频率: {stats.get('gpu', {}).get('freq', '?')} MHz")
    print(f"GPU 利用率: {stats.get('gpu', {}).get('val', '?')}%")
    print(f"温度: {stats.get('temp', {}).get('gpu', '?')}°C")
    print(f"功耗: {stats.get('power', '?')} mW")
```

## 7. 延迟优化策略

### 7.1 延迟达标路径

| 方案 | 推理量 | 预期延迟 | 是否达标 |
|------|--------|---------|---------|
| TRT FP16 + 全帧 | 100% | 0.137x（仅推理） | ✅ 远超 |
| TRT FP16 + 端到端 | 100% | 0.797x | ❌ 略超 |
| **TRT FP16 + stride=2** | 50% | **0.399x** | ✅ 达标 |
| TRT FP16 + stride=3 | 33% | 0.266x | ✅ 远超 |
| TRT INT8 + 批处理 + 流水线 | 100% | < 0.2x | ✅ 远超 |

**推荐**：Jetson Orin Nano Super + TRT FP16 + stride=2 即可达 ≤ 0.5 min/min 视频目标。

### 7.2 当前 Windows 基线（参考）

Phase 3.1e 端到端 SHADOW 模式实测（Windows RTX GPU）：
- 2700 帧 / 90s 视频 → 101.8s（1.13x）
- 瓶颈在 pose 推理 98s（96%），SHADOW 双轨仅占 2s（< 2%）
- Jetson 上预期：TRT FP16 + stride=2 → 35.9s（0.399x）

### 7.3 INT8 量化（可选，精度敏感场景慎用）

```bash
# 仅在 FP16 不达标时评估
yolo export model=best.onnx format=engine int8=True workspace=4 \
    data=dog-pose.yaml fraction=0.2 device=0

# 验证 INT8 精度（mAP50-95 损失应 < 5%）
yolo val model=best.engine data=dog-pose.yaml imgsz=640
```

**风险**：INT8 精度损失 3-5%，且必须在 Jetson 上校准（需 500+ 张代表性图像）。

## 8. 验证流程

### 8.1 转换后精度验证

```bash
# 在 Jetson 上验证 TRT engine vs ONNX 一致性
python scripts/verify_trt_fp16.py \
    --onnx models/yolo26_pose/best.onnx \
    --engine models/jetson/yolo26_pose_fp16.engine \
    --val-data data/dog-pose/val \
    --threshold 0.005  # mAP50-95 损失阈值
```

预期结果：
- mAP50-95 损失 < 0.5%
- 关键点坐标最大差异 < 1px
- 推理速度 ≥ 200 FPS（PyTorch 基线 64 FPS 的 3x+）

### 8.2 端到端延迟测试

```bash
# 在 Jetson 上运行 USPCA 端到端测试
python scripts/phase3_5_jetsone2e_test.py \
    --video data/test_videos/uspca_90s.mp4 \
    --scene uspca_patrol \
    --stride 2

# 预期:
# - 处理时间 ≤ 45s（90s 视频 × 0.5x）
# - verdict=pass
# - PDF 报告生成
# - jtop GPU 利用率 > 80%
```

### 8.3 跨平台一致性验证

```bash
# Windows 端
python scripts/phase2_6_e2e_test.py  # 基线 0.88x-1.13x

# Jetson 端
python scripts/phase3_5_jetsone2e_test.py  # 目标 ≤ 0.5x

# 对比评分结果差异（应 < 5%）
python scripts/compare_cross_platform.py \
    --windows-report reports/windows_e2e.json \
    --jetson-report reports/jetson_e2e.json
```

## 9. 待办事项（硬件到位后）

### 9.1 3.5b 环境配置

- [ ] 采购 Jetson Orin Nano Super + NVMe SSD + 散热风扇 + 电源
- [ ] 刷机 JetPack 6.1（SDK Manager 或 SD 卡镜像）
- [ ] 按 §3.1 安装软件栈
- [ ] 验证 `tensorrt` / `torch` / `onnxruntime` 导入正常
- [ ] `jtop` 监控工具就绪

### 9.2 3.5c 完整收尾

- [ ] 复制 3 个 ONNX 模型到 Jetson
- [ ] 执行 `convert_trt_fp16.py` 转换为 engine
- [ ] 精度验证（mAP50-95 损失 < 0.5%）
- [ ] 基准测试（jtop 监控）

### 9.3 3.5d 平台抽象层

- [ ] 实现 `backend/ml/pose/inference_backend.py::InferenceBackend`
- [ ] 实现 `backend/ml/pose/onnx_backend.py::OnnxRuntimeBackend`
- [ ] 实现 `backend/ml/pose/tensorrt_backend.py::TensorRtBackend`
- [ ] 实现 `backend/ml/pose/auto_backend.py::AutoBackend`
- [ ] tasks.py 集成 `_resolve_model_path()` 替代 `_resolve_stgcn_bc_path()`
- [ ] 抽帧策略集成到 pose 推理循环
- [ ] 单元测试（mock Jetson 平台检测）

### 9.4 3.5e 延迟测试

- [ ] 创建 `scripts/phase3_5_jetsone2e_test.py`
- [ ] USPCA 90s 视频端到端测试
- [ ] FCI-IGP 场景端到端测试
- [ ] 延迟 ≤ 0.5x 验收
- [ ] 跨平台一致性验证（Windows vs Jetson 评分差异 < 5%）
- [ ] 验收报告归档 `reports/phase-3.5-validation.md`

## 10. 故障排查

### 10.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `UNSUPPORTED_STATE` | workspace 过大 | `--workspace-size-gb 2`（降低到 2 GiB） |
| engine 跨设备失败 | TRT 版本/GPU 架构不同 | 必须在目标 Jetson 上重新转换 |
| PyTorch 导入失败 | pip 装了 x86_64 版本 | 用 Jetson 专用 wheel 重装 |
| onnxruntime 无 CUDA | PyPI 无 aarch64 Jetson 二进制 | 用 Ultralytics 官方 aarch64 wheel |
| GPU 频率不锁 | nvpmodel 未设置 | `sudo nvpmodel -m 2 && sudo jetson_clocks` |
| 温度过高 (>80°C) | 散热不足 | 加装主动散热风扇 + `jetson_clocks --fan` |
| 内存溢出 OOM | 模型 + 视频帧占用过高 | 降低 batch size 或启用 stride=3 |

### 10.2 性能调优检查清单

- [ ] `nvpmodel -m 2` 已设置（MAXN SUPER）
- [ ] `jetson_clocks` 已锁定最大频率
- [ ] `jetson_clocks --fan` 风扇最大
- [ ] `vm.swappiness=10` 已优化
- [ ] jtop 显示 GPU 利用率 > 80%
- [ ] jtop 显示温度 < 70°C
- [ ] jtop 显示功耗 ~25W（MAXN SUPER 模式）
- [ ] engine 文件为本机生成（非跨设备复制）

## 11. 参考

### 11.1 项目内部参考

| 文档 | 路径 | 用途 |
|------|------|------|
| Jetson 调研报告 | [RESEARCH_JETSON_DEPLOYMENT.md](../research/RESEARCH_JETSON_DEPLOYMENT.md) | 完整调研数据（803 行） |
| YOLO26-pose 部署深度报告 | [RESEARCH_YOLO26_DEPLOY_DEEP.md](../research/RESEARCH_YOLO26_DEPLOY_DEEP.md) | 性能基准 + INT8 量化数据 |
| Phase 3 阶段计划 | [phase-3.md §3.5](../stages/phase-3.md) | 子阶段任务定义 |
| 抽帧策略代码 | [frame_stride.py](../../backend/ml/pose/frame_stride.py) | 已就绪接口 |
| TRT 转换脚本 | [convert_trt_fp16.py](../../scripts/convert_trt_fp16.py) | 已就绪工具 |

### 11.2 官方文档

| 来源 | URL |
|------|-----|
| NVIDIA Jetson Orin Nano Super 产品页 | https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/ |
| Ultralytics Jetson 部署指南 | https://docs.ultralytics.com/guides/nvidia-jetson/ |
| Ultralytics TensorRT 集成 | https://docs.ultralytics.com/integrations/tensorrt/ |
| NVIDIA DeepStream Quickstart | https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html |
| PyTorch for Jetson 论坛 | https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 |

### 11.3 GitHub 仓库（调研报告 §8 已验证活跃度）

| 仓库 | 用途 |
|------|------|
| dusty-nv/jetson-inference | TRT 推理参考（2025-10-17 活跃） |
| NVIDIA-AI-IOT/jetson-containers | Docker 容器工具 |
| rbonghi/jetson_stats | 监控工具（2026-07-25 活跃） |
| NVIDIA-AI-IOT/torch2trt | PyTorch→TRT 转换（备选） |

---

*文档版本 v1.0（2026-08-02）。硬件到位后按 §9 待办事项推进，更新至 v1.1 含实测数据。*
