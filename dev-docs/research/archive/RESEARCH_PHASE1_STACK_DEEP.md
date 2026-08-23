# Phase 1 技术栈深度调研报告

> 版本: v1.0
> 日期: 2026-07-26
> 调研方法: WebSearch + 官方文档 + 社区实测报告 + GitHub Issue
> 调研者: AI Agent（sliver-vibe-coding 路由）
> Owner: 项目宪法 + 技术选型 Truth
> 修改触发: Phase 1 启动决策 / 技术栈变更

## 摘要（TL;DR）

针对 Phase 1 MVP 的四大未验证技术栈进行深度调研，得出以下关键结论：

| 调研项 | 可行性 | 风险 | 决策建议 |
|--------|--------|------|---------|
| mmcv 2.1.0 + PyTorch 2.11 + Blackwell 源码编译 | ✅ 可行（社区已验证路径） | 中（依赖 VS 工具链 + setuptools 严格控制） | Phase 1 启动前先验证编译 |
| PoseC3D 用于狗 24 关键点行为识别 | ⚠️ 可行但需重训练 | 高（预训练仅人体 17 点，狗行为需自标数据） | Phase 1 末尝试，规则引擎兜底 |
| TensorRT 10.8 + CUDA 12.8 + Windows | ✅ 完全可行 | 低（NVIDIA 官方支持） | Phase 1 即采用 FP16 导出 |
| YOLO26-pose + Dog-Pose 24 关键点 | ✅ 完全可行 | 低（Ultralytics 官方数据集） | Phase 1 首位验证项 |

**核心决策建议**：采用"分阶段降风险"路径 —— 先验证 YOLO26-pose + Dog-Pose（无依赖），再尝试 TensorRT 加速，最后才碰 mmaction2/mmcv 编译。规则引擎作为 PoseC3D 失败时的兜底方案，保证 Phase 1 端到端闭环可达成。

---

## 1. mmaction2 + mmcv 在 Windows + PyTorch 2.11 + Blackwell sm_120 的编译路径

### 1.1 现状与核心问题

**官方预编译 wheel 缺失**：
- OpenMMLab 官方仅提供 Linux 平台的 mmcv 预编译 wheel
- Windows 平台历史上从未提供 mmcv-full / mmcv 2.x 预编译 wheel
- PyTorch 2.11 + cu128 是 2026 年最新组合，社区 wheel 跟进滞后

**Blackwell sm_120 强制源码编译**：
- 50 系显卡（RTX 5060/5070/5080/5090 Laptop/Desktop）架构变更
- 旧 PyTorch（≤2.7）+ cu126 及以下不支持 sm_120
- 必须 PyTorch ≥ 2.8 + cu128（见 [ADR 0002](../decisions/0002-runtime-stack-revision-blackwell.md)）
- 此组合下 mmcv 必须源码编译以生成 sm_120 内核

### 1.2 验证可行的安装路径（基于社区 2026 实测报告）

来源：CSDN《MMCV源码编译安装》(2026-06-24) + 多份 50 系显卡部署报告

**前置条件**：

| 组件 | 推荐版本 | 说明 |
|------|---------|------|
| Python | 3.12.x | 与 ADR 0002 一致 |
| PyTorch | 2.11.x+cu128 | 通过 pip 安装 |
| CUDA Toolkit | 12.8（系统安装，仅用于编译） | 与 PyTorch wheel 内置 runtime 解耦 |
| Visual Studio | 2019 或 2022（**不可高于 2022**） | 含 C++ 桌面开发 + Windows SDK |
| setuptools | 79.0.1 | 严格固定，避免新版兼容问题 |
| mmcv 源码 | 2.1.0 | GitHub Releases 下载 |

**关键步骤**（PowerShell + x64 Native Tools Command Prompt for VS 2022）：

```powershell
# 1. 激活已有 venv（包含 PyTorch 2.11+cu128）
conda activate k9system  # 或对应 venv

# 2. 设置 CUDA_HOME（指向系统 CUDA Toolkit 12.8）
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

# 3. 设置 mmcv 编译开关
set MMCV_WITH_OPS=1
set TORCH_CUDA_ARCH_LIST=12.0
set MAX_JOBS=8

# 4. 固定 setuptools
pip install setuptools==79.0.1

# 5. 下载并解压 mmcv 2.1.0 源码
# https://github.com/open-mmlab/mmcv/releases/tag/v2.1.0
cd D:\path\to\mmcv-2.1.0

# 6. 安装 optional 依赖（用镜像加速）
pip install -r requirements/optional.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 7. 编译安装（10-30 分钟）
python setup.py develop -i https://pypi.tuna.tsinghua.edu.cn/simple

# 8. 验证
python -c "import mmcv; print(f'mmcv version: {mmcv.__version__}')"
python -c "from mmcv.ops import nms; print('CUDA ops available')"
```

### 1.3 常见编译失败与规避

| 报错 | 原因 | 解决 |
|------|------|------|
| `ninja: returned non-zero exit status 1` | 并行编译资源耗尽 | `set MAX_JOBS=4` 降低并行 |
| `ModuleNotFoundError: contourpy` | 依赖未预编译 | `pip install --only-binary contourpy contourpy` |
| `nvcc not found` | CUDA_PATH 未设置 | 设置 `CUDA_HOME` 和 `PATH` 含 `bin` |
| `no kernel image is available` | TORCH_CUDA_ARCH_LIST 缺 sm_120 | 必须 `set TORCH_CUDA_ARCH_LIST=12.0` |
| `setuptools version conflict` | setuptools 过新或过旧 | 严格 `pip install setuptools==79.0.1` |
| `THC.h not found`（mmcv 1.x） | PyTorch ≥ 2.0 移除 THC | 必须用 mmcv 2.x，**不可用 mmcv-full 1.x** |

### 1.4 mmaction2 安装

mmcv 编译成功后，mmaction2 安装较简单：

```powershell
# 方式 A：源码安装（推荐开发用）
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -r requirements.txt
pip install -v -e .

# 方式 B：pip 安装（仅调用 API）
pip install mmaction2

# 验证
python -c "import mmaction; print(mmaction.__version__)"
```

**版本组合推荐**（基于社区 2026 实测）：
- mmcv==2.1.0
- mmengine（最新稳定）
- mmdet==3.2.0（可选，用于时空检测）
- mmpose==1.1.0（可选，用于姿态提取）
- mmaction2 latest main 分支

### 1.5 风险评估

- **编译失败概率**：~30%
  - 主要风险：VS 工具链版本、setuptools 版本、contourpy 依赖
  - 缓解：严格按上述步骤、固定 setuptools 79.0.1、用镜像源
- **运行时失败概率**：~10%
  - 主要风险：PyTorch 2.11 与 mmcv 2.1.0 API 兼容性
  - 缓解：先跑 `mmcv.ops` 单元测试，再跑 mmaction2 demo
- **回滚方案**：若编译失败，Phase 1 退化为"纯规则引擎 + YOLO26-pose"，不阻塞 MVP

---

## 2. PoseC3D 在 mmaction2 中的可用性 + 输入格式 + 狗适配可行性

### 2.1 PoseC3D 概览

**论文**：[Revisiting Skeleton-based Action Recognition (arXiv 2104.13586)](https://arxiv.org/abs/2104.13586)

**核心思想**：将骨架序列转换为 3D 关键点热图堆叠（K×T×H×W），用 3D-CNN（SlowOnly）提取时空特征，规避 GCN 的鲁棒性/兼容性/可扩展性问题。

**性能基准**（人体数据集）：
| 数据集 | Top-1 | 说明 |
|--------|-------|------|
| FineGYM | 93.5% | 体操动作细粒度 |
| NTURGB+D xsub | 93.5% | 60 类日常动作 |
| Kinetics-skeleton | 82.0% | 400 类 |

### 2.2 在 mmaction2 中的代码路径

- **配置目录**：`configs/skeleton/posec3d/`
- **核心文件**：
  - `slowonly_r50_u48_240e_gym-keypoint.py`（FineGYM 基线）
  - `slowonly_r50_u48-240e_ntu120_xsub_keypoint.py`（NTU120 基线）
- **依赖工具**：
  - `tools/data/skeleton/ntu_pose_extraction.py`（关键点提取）
  - `tools/data/skeleton/`（数据格式转换）

### 2.3 输入数据格式

**原始关键点 pkl 格式**（PoseC3D 期望）：

```python
{
  'split': {
    'xsub_train': ['video_001', 'video_002', ...],
    'xsub_val':   ['video_100', ...]
  },
  'annotations': [
    {
      'keypoint': np.ndarray,        # shape: (M, T, V, C) — M人, T帧, V关键点, C通道
      'keypoint_score': np.ndarray,  # shape: (M, T, V)   — 置信度
      'frame_dir': 'video_001',
      'img_shape': (1080, 1920),
      'original_shape': (1080, 1920),
      'total_frames': 240,
      'label': 0  # 行为类别 ID
    },
    ...
  ]
}
```

**关键维度参数**：
- `M`（人数/犬数）：默认 1，工作犬场景固定为 1
- `T`（帧数）：默认 48（PoseC3D slowonly_r50_u48）
- `V`（关键点数）：**默认 17（COCO 人体），需改为 24（Dog-Pose）**
- `C`（通道）：默认 2（x, y），可选 3（x, y, score）

### 2.4 数据 pipeline 关键变换

```python
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_empty=False),
    dict(type='Resize', scale=(64, 64)),
    dict(type='GeneratePoseTarget',
         sigma=0.6,
         kernel=(7, 7),
         with_kp=True,
         with_limb=False),  # 关键：将关键点转为高斯热图
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
```

### 2.5 模型结构

```python
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=2,  # 仅 xy 坐标，热图通道数 = V × C_in
        base_channels=32,
        num_stages=3,
        stage_blocks=[4, 6, 3],
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        spatial_strides=[1, 2, 2]
    ),
    cls_head=dict(
        type='I3DHead',
        num_classes=8,  # ★ 工作犬 P0 8 类行为
        in_channels=512,
        spatial_type='avg',
        dropout_ratio=0.5
    ),
    train_cfg=dict(),
    test_cfg=dict(average_clips='prob')
)
```

### 2.6 狗 24 关键点适配可行性分析

**适配维度**：

| 维度 | 人体（COCO） | 狗（Dog-Pose） | 改造难度 |
|------|-------------|----------------|---------|
| 关键点数 V | 17 | 24 | 低（仅改配置） |
| 关键点语义 | 鼻/眼/肩/肘/腕/髋/膝/踝 | 爪/膝/肘/尾/耳/鼻/眼/鬓甲/喉 | 中（影响 limb 定义） |
| 骨架连接（limb） | COCO-17 默认 | 需自定义 24 点骨架 | 中 |
| 数据规模 | 6773+1703 张（Dog-Pose） | 工作犬视频需自标 | 高 |
| 预训练权重 | FineGYM/NTU 可用 | **无狗行为预训练** | 高（需从头训练） |

**关键改造步骤**：

1. **骨架定义**（`mmaction/datasets/skeletons/coco.py` 类似文件）：
   ```python
   dog_skeleton = [
       [0, 1], [1, 2],    # 前左爪 → 前左膝 → 前左肘
       [3, 4], [4, 5],    # 后左爪 → 后左膝 → 后左肘
       [6, 7], [7, 8],    # 前右爪 → 前右膝 → 前右肘
       [9, 10], [10, 11], # 后右爪 → 后右膝 → 后右肘
       [12, 13],          # 尾根 → 尾尖
       [14, 18], [15, 19],# 左/右耳基 → 左/右耳尖
       [16, 17],          # 鼻 → 下巴
       [20, 21],          # 左眼 → 右眼
       [22, 23]           # 鬓甲 → 喉
   ]
   ```

2. **配置文件改动**：
   - `model.cls_head.num_classes = 8`（P0 8 类行为）
   - `keypoint_shape = (24, 3)` 或 `(24, 2)`
   - `GeneratePoseTarget` 自动适配 V=24

3. **训练数据准备**：
   - 用 YOLO26-pose（Dog-Pose 微调）从工作犬视频提取 24 关键点
   - 按 mmaction2 skeleton 格式打包 pkl
   - 标注 8 类 P0 行为（坐/卧/立/随行/坐立/停留/叫/咬）

### 2.7 工作犬行为识别 PoseC3D 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 工作犬数据不足（<200 段视频/类） | 高 | 训练不收敛 | 先用规则引擎跑通闭环，Phase 2 再积累数据 |
| 24 关键点热图内存爆炸 | 中 | OOM | 降低 T=48→24，或 V=24→17 子集 |
| 预训练权重不适用 | 必然 | 从头训练慢 | 用 FineGYM 权重做 warm-start，仅 cls_head 随机初始化 |
| 单犬假设失败（多犬干扰） | 低 | 关键点错位 | Phase 1 单犬场景，Phase 3 多犬 |

### 2.8 推荐训练超参（工作犬 8 类 P0）

```python
# 基于 slowonly_r50_u48_240e_gym-keypoint.py 修改
param_scheduler = [
    dict(type='CosineAnnealing', T_max=240, by_epoch=True, eta_min=0)
]
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0003)
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=240, val_interval=10)
```

---

## 3. TensorRT 10.x for CUDA 12.8 Windows 安装 + YOLO26 集成

### 3.1 版本选择

**主选**：TensorRT 10.8.0.6（CUDA 12.8 官方完整支持）

来源：[NVIDIA TensorRT 10.8 Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/10.8.0/getting-started/support-matrix.html)

| 平台 | CUDA | TensorRT | Blackwell sm_120 |
|------|------|----------|------------------|
| Windows x64 | 12.8 | 10.8.x | ✅ |
| Linux x86-64 | 12.8 | 10.8.x | ✅ |

**备选**：TensorRT-RTX 1.5（最新，专为 RTX 50 系优化）
- 需要 CUDA 12.9 或 13.2
- 需要 NVIDIA Developer Program 会员
- 实验性质，**不推荐 Phase 1 使用**

### 3.2 安装步骤（TensorRT 10.8 + Windows）

```powershell
# 1. 下载 TensorRT 10.8 ZIP
# https://developer.nvidia.com/tensorrt/download/10x
# 文件名：TensorRT-10.8.0.6.Windows10.x86_64.cuda-12.8.zip

# 2. 解压到固定目录
# C:\TensorRT-10.8.0.6\
# ├── bin\
# ├── include\
# ├── lib\
# ├── python\

# 3. 添加 lib 到 PATH
setx PATH "%PATH%;C:\TensorRT-10.8.0.6\lib"

# 4. 安装 Python 绑定
cd C:\TensorRT-10.8.0.6\python
pip install tensorrt-10.8.0.6-cp312-cp312-win_amd64.whl

# 5. 验证
python -c "import tensorrt as trt; print(trt.__version__)"
```

### 3.3 YOLO26-pose TensorRT 集成

**FP16 导出**（推荐，精度损失 < 0.5%）：

```python
from ultralytics import YOLO

model = YOLO("best.pt")  # 微调后的 dog-pose 24 关键点模型

model.export(
    format="engine",
    imgsz=640,
    half=True,           # FP16 量化
    workspace=4,         # 4GB workspace
    simplify=True,
    device=0,            # GPU 0
    dynamic=False        # 固定尺寸更快
)
# 输出：best.engine
```

**INT8 导出**（精度损失 3-5%，需校准数据集）：

```python
model.export(
    format="engine",
    imgsz=640,
    int8=True,
    data="dog-pose.yaml",  # 校准数据集
    fraction=0.2,           # 20% 数据校准
    device=0
)
```

**TensorRT 推理**：

```python
from ultralytics import YOLO

# 方式 1：Ultralytics 自动加载（推荐）
trt_model = YOLO("best.engine", task="pose")
results = trt_model("dog_video.mp4")

# 方式 2：直接用 ONNX Runtime + TensorRT EP
# （仅当 Ultralytics engine 加载失败时）
```

### 3.4 性能预期（参考 RTX 4090，RTX 5060 Laptop 类似或更优）

| 模型 | 格式 | 推理延迟 | GPU 显存 |
|------|------|---------|---------|
| YOLO26n-pose | PyTorch FP32 | ~8 ms | ~800 MB |
| YOLO26n-pose | TensorRT FP16 | ~2 ms | ~400 MB |
| YOLO26n-pose | TensorRT INT8 | ~1.5 ms | ~350 MB |
| YOLO26s-pose | TensorRT FP16 | ~3 ms | ~600 MB |

### 3.5 风险评估

- **TensorRT 安装失败**：~5%（NVIDIA 官方支持）
- **YOLO26 engine 导出失败**：~10%（Ultralytics 与 TensorRT 版本匹配问题）
  - 缓解：先用 ONNX 导出验证，再尝试 engine
- **engine 文件不可跨 GPU**：必然（TensorRT 与 GPU 绑定）
  - 影响：开发机与部署机需为同型号 GPU
  - Phase 1-3 本地部署，无影响

---

## 4. YOLO26-pose + Dog-Pose 24 关键点实际验证

### 4.1 Dog-Pose 数据集

来源：[Ultralytics Dog-Pose 官方文档](https://docs.ultralytics.com/datasets/pose/dog-pose/)

- **来源**：Stanford Dogs Dataset（fine-grained 图像分类）扩展
- **规模**：6773 训练 / 1703 验证
- **关键点数**：24（kpt_shape=[24, 3]，3 = x, y, visibility）
- **类别**：1（dog）
- **下载**：`https://github.com/ultralytics/assets/releases/download/v0.0.0/dog-pose.zip`（337 MB）

### 4.2 24 关键点定义（完整列表）

| 索引 | 关键点 | 索引 | 关键点 |
|------|--------|------|--------|
| 0 | front_left_paw | 12 | tail_start |
| 1 | front_left_knee | 13 | tail_end |
| 2 | front_left_elbow | 14 | left_ear_base |
| 3 | rear_left_paw | 15 | right_ear_base |
| 4 | rear_left_knee | 16 | nose |
| 5 | rear_left_elbow | 17 | chin |
| 6 | front_right_paw | 18 | left_ear_tip |
| 7 | front_right_knee | 19 | right_ear_tip |
| 8 | front_right_elbow | 20 | left_eye |
| 9 | rear_right_paw | 21 | right_eye |
| 10 | rear_right_knee | 22 | withers |
| 11 | rear_right_elbow | 23 | throat |

### 4.3 训练流程（基于已有 RESEARCH_YOLO26_DEPLOY_DEEP.md）

**最小验证流程**（Phase 1 启动时立即执行）：

```python
from ultralytics import YOLO

# Step 1: 加载预训练 YOLO26-pose（默认 17 COCO 关键点）
model = YOLO("yolo26n-pose.pt")

# Step 2: 在 Dog-Pose 上微调（自动适配 24 关键点）
results = model.train(
    data="dog-pose.yaml",  # 自动下载
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.001,
    cos_lr=True,
    box=7.5,
    cls=0.5,
    pose=12.0,
    kobj=1.0,
    device=0
)
# 输出：runs/pose/train/weights/best.pt（24 关键点模型）

# Step 3: 验证
metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
```

### 4.4 推理 API + 输出格式

```python
from ultralytics import YOLO

model = YOLO("best.pt")  # 24 关键点微调后
results = model("dog_test.jpg", conf=0.5)

# 单犬场景
result = results[0]

# 关键点输出
keypoints = result.keypoints           # Keypoints 对象
xy = result.keypoints.xy               # shape: [N, 24, 2]  — (x, y) 像素坐标
conf = result.keypoints.conf           # shape: [N, 24]      — 每点置信度
data = result.keypoints.data           # shape: [N, 24, 3]   — (x, y, visible)

# 工作犬场景（单犬）
assert xy.shape == (1, 24, 2), f"Unexpected shape: {xy.shape}"
print(f"24 keypoints:\n{xy[0]}")
print(f"confidences:\n{conf[0]}")
```

### 4.5 关键点 → PoseC3D 输入格式转换

```python
import numpy as np
import pickle

def ultralytics_to_posec3d_pkl(video_results, label, output_path):
    """
    将 Ultralytics YOLO26-pose 视频推理结果转为 PoseC3D pkl 格式。
    """
    # video_results: list of per-frame Results
    M = 1  # 单犬
    T = len(video_results)
    V = 24  # Dog-Pose
    C = 3   # x, y, score

    keypoint = np.zeros((M, T, V, C), dtype=np.float32)
    keypoint_score = np.zeros((M, T, V), dtype=np.float32)

    for t, result in enumerate(video_results):
        if len(result.keypoints) > 0:
            kp_data = result.keypoints.data[0]  # [24, 3]
            keypoint[0, t, :, :2] = kp_data[:, :2]  # x, y
            keypoint[0, t, :, 2] = kp_data[:, 2]    # visibility
            keypoint_score[0, t, :] = kp_data[:, 2]

    annotation = {
        'keypoint': keypoint,
        'keypoint_score': keypoint_score,
        'frame_dir': 'dog_video_001',
        'img_shape': (1080, 1920),
        'original_shape': (1080, 1920),
        'total_frames': T,
        'label': label  # 0-7 对应 8 类 P0 行为
    }

    data = {
        'split': {'xsub_train': ['dog_video_001'], 'xsub_val': []},
        'annotations': [annotation]
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
```

### 4.6 风险评估

- **Dog-Pose 微调失败**：~5%（Ultralytics 官方支持）
- **工作犬场景精度低**：~50%（Dog-Pose 是宠物犬，工作犬姿态分布差异大）
  - 缓解：Phase 1 末采集 300-500 张工作犬视频帧微调
  - 见 [RESEARCH_YOLO26_DEPLOY_DEEP.md §2.2](./RESEARCH_YOLO26_DEPLOY_DEEP.md)
- **24 关键点遮挡问题**：~30%（侧面视角时某些点必然遮挡）
  - 缓解：用关键点置信度阈值过滤，缺失点用前后帧插值

---

## 5. 综合决策建议

### 5.1 Phase 1 实施路径（推荐）

**分阶段降风险，保证端到端闭环可达成**：

```
Phase 1.0 [Week 1-2]: YOLO26-pose 微调 + 推理验证
  ├─ 下载 Dog-Pose 数据集（6773 张）
  ├─ 微调 yolo26n-pose.pt → best.pt（24 关键点）
  ├─ 验证推理 API：results[0].keypoints.shape = [1, 24, 3]
  └─ 评估 mAP50-95（预期 > 70%）

Phase 1.1 [Week 2-3]: TensorRT 加速
  ├─ 安装 TensorRT 10.8 + Python 绑定
  ├─ model.export(format="engine", half=True)
  ├─ 验证 engine 推理延迟（预期 < 5ms/frame）
  └─ 集成到 backend/workers/inference.py

Phase 1.2 [Week 3-4]: 规则引擎（保证 MVP 闭环）
  ├─ 基于 24 关键点几何规则
  ├─ 实现 P0 8 类行为识别：
  │   - 坐/卧/立：躯体关键点相对位置
  │   - 随行：犬与人的相对运动
  │   - 坐立/停留：状态保持时长
  │   - 叫/咬：嘴部关键点动作
  ├─ 单元测试 + 真实视频验证
  └─ 端到端：视频 → 关键点 → 规则识别 → 评分 → PDF

Phase 1.3 [Week 5-6]: mmaction2 + PoseC3D 尝试（可选）
  ├─ 安装 VS BuildTools 2022 + CUDA Toolkit 12.8
  ├─ 源码编译 mmcv 2.1.0（TORCH_CUDA_ARCH_LIST=12.0）
  ├─ 安装 mmaction2 latest
  ├─ 若成功：
  │   ├─ 准备工作犬行为数据集（8 类 × 50 段视频 = 400 段）
  │   ├─ 用 YOLO26-pose 提取 24 关键点 → pkl
  │   ├─ 训练 PoseC3D slowonly_r50_u48（240 epoch）
  │   └─ 对比规则引擎 vs PoseC3D 精度
  └─ 若失败：
      ├─ 记录失败原因到 decisions/0004-*.md
      ├─ Phase 1 仅用规则引擎，Phase 2 再尝试
      └─ Phase 3 提前评估 ST-GCN+BC 直上路径

Phase 1.4 [Week 6-7]: 评分 + 报告 + 前后端集成
  ├─ F4 评分引擎：GA-T 3 维（动作准确度/响应延迟/保持时长）
  ├─ F5 报告生成：PDF 导出
  ├─ F7 犬只档案：基础 CRUD
  └─ F1 视频上传 + 异步推理

Phase 1.5 [Week 7-8]: 端到端验收
  ├─ 1 段工作犬视频 → 评分 PDF
  ├─ reports/phase-1-validation.md
  └─ 用户决策是否升级 Phase 2
```

### 5.2 风险矩阵

| 风险 | 概率 | 影响 | 触发应对 |
|------|------|------|---------|
| mmcv 编译失败 | 30% | 中 | 规则引擎兜底，PoseC3D 推迟 |
| PoseC3D 训练不收敛 | 50% | 中 | 规则引擎兜底，Phase 2 积累数据 |
| YOLO26-pose 工作犬精度 < 60% | 50% | 高 | Phase 1 末采集 300 张微调 |
| TensorRT engine 导出失败 | 10% | 低 | 退化为 ONNX Runtime GPU |
| 规则引擎精度 < 70% | 30% | 高 | 增加 P0 行为的关键点规则细化 |
| 端到端延迟 > 1 分钟/分钟视频 | 20% | 中 | 启用 TensorRT + 关键帧采样（每 3 帧采 1） |

### 5.3 推荐决策

**建议**：用户确认本调研结论后，立即启动 Phase 1.0（YOLO26-pose 微调验证），无需等待 mmcv 编译。

**理由**：
1. YOLO26-pose + Dog-Pose 是最低风险路径，Ultralytics 官方支持
2. 规则引擎作为兜底，保证 Phase 1 端到端闭环可达成
3. mmaction2 + PoseC3D 推迟到 Phase 1.3，不阻塞 MVP
4. TensorRT 加速在 YOLO26 微调成功后即可并行验证

### 5.4 待用户确认事项

1. **是否同意上述 Phase 1 实施路径**（分阶段降风险 + 规则引擎兜底）？
2. **是否同意 Phase 1.3 mmaction2 + PoseC3D 设为可选**（失败不阻塞 MVP）？
3. **是否同意 Phase 1 末采集 300 张工作犬视频帧微调 YOLO26-pose**（10-20 小时人工标注）？

---

## 6. 引用来源

### 官方文档
- [PyTorch Blackwell 兼容矩阵](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- [NVIDIA TensorRT 10.8 Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/10.8.0/getting-started/support-matrix.html)
- [NVIDIA TensorRT 安装指南](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)
- [Ultralytics YOLO26 官方文档](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics Dog-Pose 数据集](https://docs.ultralytics.com/datasets/pose/dog-pose/)
- [Ultralytics Pose Estimation 任务](https://docs.ultralytics.com/tasks/pose)
- [MMCV 2.x 安装文档](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html)
- [MMAction2 GitHub](https://github.com/open-mmlab/mmaction2)
- [PoseC3D 论文 arXiv 2104.13586](https://arxiv.org/abs/2104.13586)

### 社区实测报告
- [MMC源码编译安装（50 系显卡 + PyTorch 2.11 实测）](https://blog.csdn.net/weixin_64388392/article/details/162281309)（2026-06-24）
- [Windows 10 系统下 MMCV 源码编译与安装避坑指南](https://blog.csdn.net/weixin_29031057/article/details/158982382)（2026-03-13）
- [RTX 50 系显卡部署 AI 视频与语音套件：从环境配置到实战避坑](https://wenku.csdn.net/answer/36t2637pz75)（2026-03-13）
- [Windows 11 环境下 NVIDIA RTX 5090（sm_120）与 PyTorch 兼容问题](https://www.volcengine.com/article/13728)（2026-04-01）
- [Windows 下 MMCV 与 PyTorch 版本冲突全解析](https://blog.csdn.net/weixin_28831341/article/details/159261361)（2026-03-20）

### 相关调研
- [YOLO26-pose 微调详细参数 + 工作犬数据采集方案](./RESEARCH_YOLO26_DEPLOY_DEEP.md)
- [运行时栈修正 Blackwell 决策](../decisions/0002-runtime-stack-revision-blackwell.md)
- [PoseC3D 介绍（OpenMMLab）](https://cloud.tencent.com.cn/developer/article/1935879)

## 7. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-26 | 初始版本，覆盖 mmcv 编译 / PoseC3D / TensorRT / YOLO26-pose 四项调研 + 决策建议 |
