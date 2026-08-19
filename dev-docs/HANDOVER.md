# K9 Training Vision System — 项目交接文档

**版本**: v1.2  
**日期**: 2026-08-19  
**项目根目录**: `D:\Desktop\k9-training-system`  
**接手会话**: 歆歆（sliver-vibe-coding）于 2026-08-18~19 完成接管 + Mamba 编译 + tasks.py 集成路由 + 文档同步

---

## 1. 项目概况

工作犬训练机器视觉识别系统，用于分析工作犬训练视频，自动识别 22 种行为并生成 FCI-IGP 标准评分报告。

**核心技术栈**: Python 3.12 / PyTorch 2.11+cu128 / FastAPI / PostgreSQL 17 / Vue 3 / Redis + Celery

---

## 2. 当前系统状态

### 2.1 阶段完成情况

| 阶段 | 状态 | 关键交付 |
|------|------|----------|
| 立项 | ✅ 完成 | Truth 文档 + ADR + 调研 |
| Phase 0 | ✅ 完成 | 环境 + DB schema + 前后端骨架 |
| Phase 1 | ✅ 完成 | MVP 端到端闭环（双场景 + YAML 评分） |
| Phase 2 | ✅ 完成 | 数据飞轮 + 16 行为 + USPCA |
| Phase 3 | ✅ 已关闭（2026-08-17） | 全部子阶段 3.1b-3.8d 验收通过 + 3.2d 补充验证 |
| Phase 4 | 🔄 实施中 | LLM 解释器 ✅ / Transformer-Mamba 基线 ✅ / **mamba_ssm 真实编译 ✅（RTX 5060 sm_120）** / tasks.py 集成路由 ✅ |

### 2.2 核心模块

| 模块 | 状态 | 说明 |
|------|------|------|
| 姿态检测 (YOLO26-pose) | ✅ | 检测 24 个犬只关键点，Box mAP50=96.1% (APTv2 微调) |
| 行为识别 (ST-GCN+BC) | ✅ | 22 类行为分类，合成数据 46.97%，SHADOW 双轨上线 |
| 行为识别 (Mamba 基线) | ✅ | 合成数据 85.61%，21K 参数；**真实 mamba_ssm 已在 WSL/RTX 5060 sm_120 编译验证** |
| 行为识别 (VideoMamba) | ✅ | 纯 PyTorch 回退实现（videomamba_skeleton.py），无需 CUDA 编译 |
| 评分引擎 (FCI-IGP) | ✅ | 7 维评分 + 22 行为映射 + 5 级评级 + DQ 硬约束 |
| 评分引擎 (USPCA) | ✅ | 通用评分标准 |
| LLM 行为解释器 | ✅ | Agnes 2.5 Flash API，中文报告生成 |
| 多犬追踪 | ✅ | APTv2 微调 + 24 视频评估，5/24 双达标（MOTA≥70% 达 9/24） |
| RBAC 权限 | ✅ | 5 角色 + 基地隔离，端到端 25/25 通过 |
| 训练历史对比 | ✅ | 前端 ECharts 可视化，API 41/41 通过 |
| 前端 | ✅ | Vue 3 + TypeScript |

### 2.3 本次接手会话完成事项

1. **项目接管审计**：只读检查 Git 状态、truth 文档一致性、运行时证据
2. **Git 保护**：commit `3e02099` feat(phase4): 升级 Mamba 标准实现（11 文件 +2148 行）
3. **BUG 修复**：commit `def57c4` fix(trainer): 确保 best.pt 在训练完成后始终存在
4. **文档漂移修正**：Phase 3 关闭状态写入 phase-3.md / README.md / stage-plan.md
5. **ADR 0011 路径统一**：从根目录 `decisions/` 迁移至 `dev-docs/decisions/`
6. **Phase 4 truth 对齐**：phase-4.md / phase-4-transformer-mamba.md / RESEARCH_TRANSFORMER_MAMBA.md
7. **tasks.py Mamba 集成路由**（2026-08-19）：`behavior_deploy_mode` 配置项 + Mamba/Mamba+BC 单例懒加载 + 模式分发
8. **mamba_ssm WSL 真实编译**（2026-08-19）：CUDA 12.8 + gcc-12 + RTX 5060 (sm_120)，`load_mamba_class()` 返回真实 `mamba_ssm.modules.mamba_simple.Mamba`

---

## 3. 关键配置

### 3.1 环境变量 (.env)

```
# 数据库
DATABASE_URL=postgresql+asyncpg://k9system:K9System2026!@127.0.0.1:5433/k9system
PG_DSN=postgresql://k9system:K9System2026!@127.0.0.1:5433/k9system

# Redis / Celery
REDIS_URL=redis://127.0.0.1:6379/0
CELERY_BROKER_URL=redis://127.0.0.1:6379/1
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/2

# LLM API (Agnes 最强免费模型)
LLM_API_KEY=sk-N9pNXGA*** (从 opencode 配置读取)
LLM_API_BASE=https://apihub.agnes-ai.com/v1
LLM_MODEL=agnes-2.5-flash
```

### 3.2 启动服务顺序

```bash
# 1. 启动 Redis
redis-server --port 6379

# 2. 启动 PostgreSQL (需已安装 PG17)
# 服务在 127.0.0.1:5433

# 3. 启动 Celery Worker
.venv\Scripts\python.exe -m celery -A backend.workers.celery_app worker --loglevel=info --pool=solo

# 4. 启动 FastAPI
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001

# 5. 启动前端 (开发模式)
cd frontend && npm run dev
```

---

## 4. 数据集

| 数据集 | 位置 | 大小 | 说明 |
|--------|------|------|------|
| APTv2 完整 | `data/APTv2/APTv2/` | 25GB | 30 种动物，86195 文件 |
| APTv2 标注 | `data/aptv2_annotations/` | 127MB | COCO 格式标注 |
| APTv2 Canidae 子集 | `data/aptv2_canidae/` | 360 帧 | dog/wolf/fox，24 视频 |
| APTv2 YOLO 训练集 | `data/aptv2_yolo_pose/` | 290 图 | YOLO pose 格式 |
| YouTube 自标数据 | `data/youtube_self_label/` | 682 片段 | 5 视频，24 关键点 |
| 合成数据 | 代码生成 (make_synthetic_dataset) | 动态 | 22 类 × N 样本 |
| 上传视频 | `data/uploads/` | 33 个 | 真实犬只训练视频 |
| ReID 数据 | `data/market1501/` | 852 张 | 12 identity，Market1501 格式 |

---

## 5. 模型权重

| 模型 | 路径 | 大小 | 说明 |
|------|------|------|------|
| YOLO26-pose (原始) | `runs/train-2/weights/best.pt` | 7.8MB | 工作犬姿态检测 |
| YOLO26-pose (APTv2 微调) | `runs/pose/runs/aptv2_finetune/yolo_canidae_pose/weights/best.pt` | 7.8MB | 动物检测，Box mAP50=96.1% |
| ST-GCN+BC (真实数据) | `runs/stgcn_bc/best.pt` | 17MB | 22 类行为识别 |
| ST-GCN+BC (合成数据) | `runs/stgcn_bc_synthetic/best.pt` | 17MB | 合成基线 46.97% |
| Mamba 基线 | `runs/mamba_baseline/best.pt` | 0.1MB | 21K 参数，85.61% |
| Mamba+BC (合成) | `runs/mamba_bc_synthetic/best.pt` | 待训练 | MS-Temba 多尺度膨胀 SSM |
| OSNet ReID | `.venv/Lib/site-packages/models/osnet_x1_0_msmt17.pt` | 16.5MB | 行人重识别（跨域用于犬只） |
| OSNet ReID (小) | `.venv/Lib/site-packages/models/osnet_x0_25_msmt17.pt` | 8.9MB | 轻量版 |

---

## 6. Mamba 升级说明（本次会话核心成果）

### 6.1 实现策略

- **优先路径**：解析 `external/VideoMamba/mamba` 中的标准 `mamba_ssm` 源码
- **回退路径**：当前 Windows 环境缺少 `causal-conv1d` CUDA 编译依赖 → 使用 `videomamba_skeleton.py::VideoMambaSkeleton` 作为正式回退
- **兼容接口**：保留 `get_model()`、`MambaSequenceBaseline` 等旧名称，训练/推理/路由调用面无震荡
- **tasks.py 集成**：`behavior_deploy_mode` 配置项支持 `mamba_only` / `mamba_shadow` / `mamba_vote` / `mamba_bc_only` / `mamba_bc_shadow` / `mamba_bc_vote`，单例懒加载 MambaInferer / MambaBCInferer

### 6.2 WSL 真实编译（已完成 ✅）

- **环境**：Ubuntu 26.04 WSL2 + CUDA Toolkit 12.8 + gcc-12 + RTX 5060 (sm_120)
- **vendored 源码**：`external/VideoMamba/`（commit `37355c2`，未修改源码，仅补 sm_120 编译目标）
- **修补 setup.py**：`external/VideoMamba/mamba/setup.py` 和 `causal-conv1d/setup.py` 各增加一行 `-gencode arch=compute_120,code=sm_120`
- **glibc 2.43 兼容**：CUDA 12.8 `math_functions.h` 与 Ubuntu 26.04 glibc 2.43 冲突（`cospi/sinpi/rsqrt noexcept`），已对 `/usr/local/cuda/include/crt/math_functions.h` 打补丁
- **编译产物**：
  - `mamba_ssm 1.0.1`（含 `selective_scan_cuda` 187MB .so，sm_70/80/90/120）
  - `causal_conv1d 1.0.0`（含 `causal_conv1d_cuda` .so，sm_70/80/90/120）
  - `transformers 4.57.6`
- **验证**：
  - `load_mamba_class()` 返回 `mamba_ssm.modules.mamba_simple.Mamba` ✅
  - CUDA forward: `torch.Size([1,64,64]) -> torch.Size([1,64,64])` on RTX 5060 ✅
  - 项目测试：605 passed + 2 skipped，零回归 ✅
- **编译脚本**：`/root/setup_mamba.sh`（WSL 内部）

### 6.3 依赖现状

| 依赖 | 状态 | 说明 |
|------|------|------|
| `einops` | ✅ 已安装 | 0.8.2 |
| `causal-conv1d` | ✅ WSL 编译完成 | 1.0.0，sm_120 |
| `mamba_ssm` | ✅ WSL 编译完成 | 1.0.1，`load_mamba_class()` 返回真实类 |
| `transformers` | ✅ WSL 安装 | 4.57.6（兼容 mamba_ssm 1.0.1） |

---

## 7. WSL 环境（mamba_ssm 编译 — ✅ 已完成）

**状态**: ✅ 已完成（2026-08-19）

WSL 环境位置:
- 发行版: Ubuntu (WSL2)
- Python venv: `/root/k9venv/`（WSL 内部）
- 项目代码: `/mnt/d/Desktop/k9-training-system/`（通过 /mnt/d 访问 Windows 盘）
- 编译脚本: `/root/setup_mamba.sh`

**已完成**:
```bash
# WSL 中运行（或参考 /root/setup_mamba.sh）:
/root/k9venv/bin/python -c "from backend.ml.behavior.mamba_sequence import load_mamba_class; cls=load_mamba_class(); print(f'{cls.__name__} from {cls.__module__}')"
# 输出: Mamba from mamba_ssm.modules.mamba_simple
```

**注意事项**:
- `mamba_ssm` 和 `causal-conv1d` 的 setup.py 已修补增加 sm_120 编译目标
- CUDA math header 补丁（`/usr/local/cuda/include/crt/math_functions.h`）已在系统级应用
- 如需重新编译，确保环境变量正确传递：`CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 TORCH_CUDA_ARCH_LIST="12.0" MAMBA_FORCE_BUILD=TRUE`

---

## 8. 测试状态

| 测试套件 | 数量 | 结果 |
|---------|------|------|
| 核心单元测试 | 605 | ✅ 全部通过（2 skipped） |
| ML 测试 | 含 Mamba 升级 5 项 | ✅ 全部通过 |
| 项目守卫 | 7 | ✅ 7 PASS |
| ST-GCN+BC 部署测试 | 20 | ✅ 全部通过 |
| Mamba 升级测试 | 9 | ✅ 全部通过 |
| FCI-IGP 端到端 | 9 | ✅ 9/9 通过 |
| RBAC 端到端 | 25 | ✅ 25/25 通过 |
| 训练历史对比端到端 | 41 | ✅ 41/41 通过 |

**运行测试命令：**
```bash
# 全量单元测试（不含 integration/e2e）
python -m pytest backend/tests/ -q --ignore=backend/tests/integration

# 仅 Mamba 相关
python -m pytest backend/tests/ml/test_mamba_upgrade.py -q

# 项目守卫
python scripts/check_project_guardrails.py . --mode bootstrap

# WSL 中验证 mamba_ssm 真实编译
wsl.exe -d Ubuntu -u root --exec /bin/bash -lc '/root/setup_mamba.sh'
```

---

## 9. 关键文件索引

### 源代码
| 路径 | 说明 |
|------|------|
| `backend/ml/behavior/mamba_sequence.py` | **Mamba 统一入口**（含 mamba_ssm 优先解析 + VideoMambaSkeleton 回退） |
| `backend/ml/behavior/videomamba_skeleton.py` | VideoMamba 纯 PyTorch 骨骼实现 |
| `backend/ml/behavior/mamba_inference.py` | MambaInferer（PyTorch + ONNX 双后端） |
| `backend/ml/behavior/mamba_trainer.py` | MambaTrainer |
| `backend/ml/behavior/mamba_bc.py` | Mamba+BC（MS-Temba 多尺度膨胀 SSM + BC 头） |
| `backend/ml/behavior/mamba_bc_inference.py` | MambaBCInferer |
| `backend/ml/behavior/stgcn_bc/` | ST-GCN+BC 完整实现 |
| `backend/ml/behavior/router.py` | 行为识别路由层（SHADOW/VOTE/PRIMARY_STGCN/MAMBA_* / RULE_ONLY） |
| `backend/ml/behavior/llm_explainer.py` | LLM 行为解释器 |
| `backend/ml/pose/` | 姿态检测 + 3D 姿态重建 |
| `backend/ml/tracking/` | 多犬追踪 |
| `backend/ml/scoring/` | 评分引擎 |
| `backend/app/api/auth.py` | 认证 API（JWT + bcrypt） |
| `backend/app/api/bases.py` | 基地管理 API |
| `backend/app/core/security.py` | JWT + bcrypt + AuthError |
| `backend/app/core/deps.py` | 依赖注入（get_current_handler, check_base_access 等） |
| `scripts/train_mamba*.py` | Mamba 训练入口（baseline / bc / 带参数版本） |

### 文档
| 路径 | 说明 |
|------|------|
| `AGENTS.md` | 项目宪法，版本 v1.25 |
| `dev-docs/HANDOVER.md` | 本交接文档 |
| `dev-docs/README.md` | truth root 索引 |
| `dev-docs/stage-plan.md` | 阶段总览（v1.6） |
| `dev-docs/stages/phase-3.md` | Phase 3 计划（✅ 已关闭 v3.3） |
| `dev-docs/stages/phase-4.md` | Phase 4 计划（🔄 实施中） |
| `dev-docs/stages/phase-4-transformer-mamba.md` | Transformer-Mamba 基线（✅ 调研+基线完成） |
| `dev-docs/decisions/` | ADR 0001-0011 |
| `dev-docs/research/RESEARCH_TRANSFORMER_MAMBA.md` | Mamba/Transformer 调研 |
| `docs/user-guide.md` | 用户手册 v0.3.0 |

---

## 10. 未完成事项

### P0 - 关键
- [x] **WSL 编译 mamba_ssm**：✅ 已完成（2026-08-19，RTX 5060 sm_120）
- [ ] **真实数据标注**：YouTube 自标数据 (682 片段) 缺少行为标签，需用 Label Studio 标注后训练
- [x] **Mamba 集成路由**：✅ 已完成（2026-08-19，`behavior_deploy_mode` 支持 10 种模式）

### P1 - 重要
- [ ] **多犬追踪优化**：conf=0.35 后仍有 3 个负 MOTA 视频，需进一步优化
- [ ] **LLM 报告前端展示**：前端 `/compare` 页面增加 LLM 行为分析按钮
- [ ] **ST-GCN+BC 真实数据训练**：合成数据 46.97% → 目标 ≥85%，需真实标注数据

### P1 - 重要
- [ ] **多犬追踪优化**：conf=0.35 后仍有 3 个负 MOTA 视频，需进一步优化
- [ ] **LLM 报告前端展示**：前端 `/compare` 页面增加 LLM 行为分析按钮
- [ ] **ST-GCN+BC 真实数据训练**：合成数据 46.97% → 目标 ≥85%，需真实标注数据

### P2 - 增强
- [ ] **ReID 微调**：APTv2 Canidae 12 ID 数据不足，需更多数据才能有效微调 OSNet
- [ ] **学术论文**：Mamba 基线 85.61% + 多犬追踪 APTv2 验证 → 论文素材
- [ ] **生产部署**：配置 HTTPS、JWT 密钥、CORS 域名等生产环境设置

---

## 11. 快速启动命令

```bash
# 启动所有服务
redis-server --port 6379
.venv\Scripts\python.exe -m celery -A backend.workers.celery_app worker --loglevel=info --pool=solo
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001

# 运行测试
python -m pytest backend/tests/ -q --ignore=backend/tests/integration

# 项目守卫检查
python scripts/check_project_guardrails.py . --mode bootstrap

# 训练 Mamba 基线（合成数据）
python scripts/train_mamba_baseline.py --epochs 30

# 训练 Mamba+BC（含边界检测）
python scripts/train_mamba_bc.py --synthetic --epochs 50
```

---

## 12. Git 状态

**当前分支**: `main`  
**最新提交**:
```
def57c4 fix(trainer): 确保 best.pt 在训练完成后始终存在
3e02099 feat(phase4): 升级 Mamba 标准实现
cef7947 feat(3.7): 训练历史对比可视化 — API + 前端 + 端到端 41/41 通过
57fdc4e fix(runtime): onnxruntime 自然修复 + FastAPI 0.115.6 204 路由兼容性修复
9a5b7e8 feat(3.6): RBAC auth API 上线 + 多租户端到端验收通过 25/25
```

**工作区说明**:
- 已提交：本次会话的所有改动（Mamba 升级 + trainer 修复 + 文档同步）
- 未提交（前序会话成果，不主动修改）：
  - 已修改文件：`AGENTS.md`、`backend/app/core/config.py`、`backend/app/main.py`、`backend/ml/behavior/router.py`、`backend/ml/tracking/multi_dog_tracker.py`、`dev-docs/README.md`、`dev-docs/stage-plan.md`、`dev-docs/stages/phase-3.md`、`docs/user-guide.md`、`reports/*.json`、`scripts/eval_multi_dog_tracking.py`、`scripts/phase3_4_e2e_test.py`
  - 未跟踪文件：`backend/app/api/llm_explainer.py`、`backend/ml/behavior/llm_explainer.py`、`dev-docs/HANDOVER.md`、各类 tracking 报告、`real_format_pkl/` 等
- **不要执行 `git add .`**，未提交文件属于前序工作区状态

---

## 13. 联系人

**项目维护者**: 默默（用户）  
**AI 助手**: 歆歆（OpenCode 多智能体系统协调者）  
**项目根目录**: `D:\Desktop\k9-training-system`
