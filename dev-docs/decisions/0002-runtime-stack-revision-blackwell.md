# Decision 0002: 运行时栈修正（Blackwell sm_120 + CUDA 12.8 + PyTorch 2.11）

> 决策日期: 2026-07-26
> 决策状态: 已确认（用户确认 Step A+B 一起做，已基于实测硬件证据决策）
> 决策类型: 技术选型修正（foundation decision）
> 影响范围: technical-selection.md §5.1, runtime.md, Phase 0 全部基础设施

## 背景

项目立项时 technical-selection.md v1.0 基于 PRD 与调研文档，给出 `Python 3.12 + PyTorch 2.4 + CUDA 12.1` 的运行时组合。

接手审计（2026-07-26）发现 truth 与实测硬件存在严重偏差：

| 维度 | Truth 声明 | 实测 | 问题 |
|------|-----------|------|------|
| Python 版本 | 3.12 | 系统已装 3.10.11 | 不满足 PyTorch 2.7+ 最低 3.10 但 truth 自相矛盾 |
| PyTorch 版本 | 2.4 + CUDA 12.1 | - | CUDA 12.1 不支持 Blackwell sm_120，PyTorch 2.4 不含 sm_120 内核 |
| GPU 架构 | "TensorRT 10.x + CUDA 12.1" | RTX 5060 Laptop = Blackwell sm_120 | CUDA 12.1 仅支持到 sm_90（Ampere/Hopper），无法识别 sm_120 |
| NVIDIA 驱动 | 未声明 | 573.24 | ✅ 已满足 Blackwell 最低要求 550.90 |

**核心问题**：原 truth 在 RTX 5060 Laptop GPU 上**无法运行 GPU 推理**。PyTorch 2.4 + CUDA 12.1 wheel 不含 sm_120 内核，加载模型时会报 `no kernel image is available for execution on the device`。

## 调研证据

### 1. NVIDIA Blackwell 架构要求

- RTX 5060 Laptop GPU 基于 Blackwell 架构，计算能力 `sm_120`
- 官方要求：CUDA Toolkit ≥ 12.8，cuDNN ≥ 9.0，NVIDIA 驱动 ≥ 550.90
- 实测：本机 NVIDIA 驱动 573.24，CUDA 驱动支持版本 12.8 ✅

来源：
- https://developer.nvidia.com/cuda-toolkit-archive
- https://docs.nvidia.com/cudnn/index.html

### 2. PyTorch 官方兼容矩阵

来源：https://github.com/pytorch/pytorch/blob/main/RELEASE.md

| PyTorch 版本 | Python 范围 | 稳定 CUDA | 实验 CUDA | Blackwell sm_120 |
|------------|-----------|----------|----------|----------------|
| 2.7 | 3.9-3.13 | 11.8, 12.6 | 12.8 | ⚠️ 仅 cu128 实验 |
| 2.8 | 3.9-3.13 | 12.6, 12.8 | 12.9 | ✅ cu128 稳定 |
| 2.9 | 3.10-3.14 | 12.6, 12.8 | 13.0 | ✅ cu128 稳定 |
| 2.10 | 3.10-3.14 | 12.6, 12.8 | 13.0 | ✅ cu128 稳定 |
| 2.11 | 3.10-3.14 | 12.6, 12.8, 13.0 | 13.2 | ✅ cu128 稳定 |
| 2.12 | 3.10-3.14 | 12.6, 13.0 | 13.2 | ❌ 移除 cu128 稳定 |

### 3. 决策推导

- **必须**：PyTorch + cu128 wheel（CUDA 12.8 内置，支持 sm_120）
- **必须**：PyTorch ≥ 2.8（cu128 首次进入稳定）
- **必须**：PyTorch ≤ 2.11（2.12 移除 cu128 稳定）
- **最优**：PyTorch 2.11.x（最新稳定，cu128 + cu126 + cu130 三档稳定支持，Blackwell 完整覆盖）
- **Python**：3.12（覆盖 PyTorch 2.7-2.11 全档位，且 mmaction2 / ultralytics 主流推荐）

## 决策

### 主决策：运行时栈修正

| 组件 | 原选型 | 修正后 | 理由 |
|------|-------|-------|------|
| Python | 3.12 | 3.12.x | 保持不变 |
| PyTorch | 2.4 + CUDA 12.1 | **2.11.x + cu128** | Blackwell sm_120 必需 |
| CUDA Toolkit | 12.1 | **12.8（wheel 内置）** | PyTorch wheel 已 bundle CUDA runtime，无需单独安装 CUDA Toolkit |
| cuDNN | 未明确 | **9.17.1.4（wheel 内置）** | 随 PyTorch wheel 分发 |
| NVIDIA 驱动 | 未明确 | **≥ 550.90（实测 573.24）** | Blackwell 最低要求 |

### 安装方式

PyTorch 通过 pip 安装 cu128 wheel，**不需要**安装 CUDA Toolkit 与 cuDNN 到系统：

```bash
# 创建 Python 3.12 venv 后
pip install torch==2.11.x torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 验证脚本

```python
import torch
assert torch.cuda.is_available(), "CUDA 不可用"
assert torch.cuda.get_device_capability(0) == (12, 0), f"sm_120 期望失败: {torch.cuda.get_device_capability(0)}"
assert torch.version.cuda.startswith("12.8"), f"CUDA 12.8 期望失败: {torch.version.cuda}"
print(f"✅ PyTorch {torch.__version__} + CUDA {torch.version.cuda} + {torch.cuda.get_device_name(0)}")
```

## 不选择其他方案的理由

### 为什么不选 PyTorch 2.12（最新版）
- 2.12 移除了 cu128 稳定支持，只剩 cu126（不支持 Blackwell）和 cu130（最新但生态未跟上）
- 2.12 的 cu126 无法跑 sm_120

### 为什么不选 PyTorch Nightly
- Nightly 不稳定，且本决策是 foundation decision，必须用 stable
- PyTorch 2.11 stable 已完整支持 Blackwell，无需 Nightly

### 为什么不单独装 CUDA Toolkit 12.8
- PyTorch cu128 wheel 已 bundle 全部 CUDA runtime 库（cudart/cublas/cudnn）
- 单独装 CUDA Toolkit 仅在需要 nvcc 编译 C++/CUDA 扩展时才需要
- mmcv 含 C++/CUDA 扩展，如需源码编译再装 CUDA Toolkit 12.8（Phase 0 验证时决定）

### 为什么 Python 3.12 而非 3.13
- PyTorch 2.11 支持 3.10-3.14，3.13 也兼容
- 但 mmaction2 / mmcv / ultralytics 等扩展库对 3.13 的 wheel 覆盖仍在追赶
- 3.12 是最稳定选择

## 影响的 truth 文档

- ✅ `dev-docs/technical-selection.md` §5.1（已更新）
- ✅ `dev-docs/technical-selection.md` §2.1/§2.3/§6/§7/§8（已更新）
- ⏳ `dev-docs/runtime.md`（Phase 0 创建，需基于本决策）
- ⏳ `dev-docs/stages/phase-0.md`（Phase 0 创建，需包含本决策的验证步骤）

## 验证清单

Phase 0 安装后必须验证：

- [ ] Python 3.12.x 安装成功
- [ ] `python -c "import sys; print(sys.version_info)"` 输出 3.12.x
- [ ] PyTorch 2.11.x+cu128 安装成功
- [ ] `torch.cuda.is_available()` 返回 True
- [ ] `torch.cuda.get_device_capability(0)` 返回 `(12, 0)`
- [ ] `torch.cuda.get_device_name(0)` 返回 "NVIDIA GeForce RTX 5060 Laptop GPU"
- [ ] YOLO26-pose 模型加载成功（`YOLO("yolo26n-pose.pt")`）
- [ ] YOLO26-pose 在测试图像上推理成功
- [ ] mmaction2 1.2.x 安装成功（或记录兼容性问题）

## 回滚方案

如果 PyTorch 2.11+cu128 在 RTX 5060 Laptop 实测失败：

1. **回退 1**：PyTorch 2.10+cu128（次新稳定，cu128 同样支持）
2. **回退 2**：PyTorch 2.9+cu128
3. **回退 3**：评估 ONNX Runtime 路径（绕过 PyTorch 直跑 ONNX 模型）
4. **回退 4**：评估 CPU 推理（仅 Phase 0 验收用，不可用于生产）

## 不可逆边界

- 本决策不涉及任何不可逆操作
- PyTorch wheel 安装可随时卸载重装
- 不修改系统级 NVIDIA 驱动

## 相关文档

- `dev-docs/technical-selection.md` §5.1 完整技术栈
- `dev-docs/decisions/0001-chinese-path-migration-plan.md`（路径迁移已完成，本决策在新路径下生效）
- `dev-docs/runtime.md`（Phase 0 创建）
- https://github.com/pytorch/pytorch/blob/main/RELEASE.md（PyTorch 官方兼容矩阵）
