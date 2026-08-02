# Phase 1 — MVP 阶段计划

> 阶段: Phase 1 MVP
> 状态: ✅ 完成（2026-07-28 验收通过，带条件；v1.2 修订 1.6d/1.2f 前置条件达成，见 [reports/phase-1-validation.md](../../reports/phase-1-validation.md) + ADR 0005 v1.1）
> Owner: Phase 1 MVP
> 入口条件: Phase 0 验收通过 ✅ + 立项研讨完成 ✅
> 出口条件: 见 §6 验收清单（全部通过，详见验收报告）
> 时间约束: 3-4 周单人全职（实际 2026-07-26 → 2026-07-28 验收）
> 依据: [RESEARCH_PUPPY_SELECTION_BEHAVIOR.md](../research/RESEARCH_PUPPY_SELECTION_BEHAVIOR.md) + [RESEARCH_PUBLIC_WORKING_DOG_VIDEOS.md](../research/RESEARCH_PUBLIC_WORKING_DOG_VIDEOS.md) + [RESEARCH_SCORING_RULE_ENGINE.md](../research/RESEARCH_SCORING_RULE_ENGINE.md)

## 1. 阶段目标

**端到端双场景闭环**：
- 1 段科目测评视频 → 24 关键点 → 8 类行为识别 → 5 维评分 → PDF 报告
- 1 段幼犬选育视频 → 24 关键点 + 物体检测 → 选育信号 → 3 维评分 → PDF 报告
- YAML 评分卡可通过 API 动态读取/修改

**关键技术验证**：
- YOLO26-pose + Dog-Pose 24 关键点（✅ Phase 1.0 已完成 best.pt）
- TensorRT 10.8 FP16 加速（必达，失败降级 ONNX Runtime GPU）
- 规则引擎 P0 8 类行为（科目测评，必达）
- 选育信号提取器（幼犬选育，原型级）
- YAML 配置化评分引擎（双场景，必达）

## 2. 范围

### 2.1 包含

- **F1 视频上传**：本地文件上传（mp4/avi/mov）+ 测试类型字段
- **F2 姿态检测**：YOLO26-pose 24 关键点（✅ best.pt 已微调），单犬场景
- **F3a 选育信号提取器**：食物/玩具/胆量 3 维信号（原型级）
- **F3b 科目规则引擎**：8 类基础行为（坐/卧/立/随行/坐立/停留/叫/咬）
- **F4 评分引擎**：YAML 配置化（选育 3 维 + 科目 5 维）
- **F5 报告生成**：PDF 导出（reportlab）
- **F6 犬只档案**：基础 CRUD
- **F7 模型与评分卡管理**：版本注册 + 评分卡 CRUD + 热加载

### 2.2 不包含

- USPCA / FCI-IGP 标准（Phase 2/3）
- 16 类 P1 行为 + 6 类 P2 高级行为（Phase 2/3）
- 多犬追踪（Phase 3）
- 3D 姿态 / 多视角融合（Phase 3）
- ST-GCN+BC（Phase 3）
- LLM 评分解释 / RL 优化（Phase 4）
- RTSP 流接入（Phase 3）
- 用户权限 / 多租户（Phase 2）
- 基地视频采集（用户拿不到，用公开数据）

## 3. 子阶段任务分解

> **3-4 周时间约束**：按周划分，前序验收通过后才启动下一阶段。

### Phase 1.0 YOLO26-pose 微调 + 推理验证（✅ 已完成）

**Owner**: ML 开发（`backend/ml/pose/`）

- ✅ **1.0a** 下载 Dog-Pose 数据集（6773/1703 张，337 MB）
- ✅ **1.0b** 配置 `dog-pose.yaml`（含 24 关键点 flip_idx）
- ✅ **1.0c** 微调 `yolo26n-pose.pt` → `runs/train-2/weights/best.pt`（100 epoch 完成）
- ✅ **1.0d** 验证推理 API（shape=(T,24,3) 验证通过）
- ✅ **1.0g** 实现后端推理模块 `backend/ml/pose/inference.py`
- ✅ **1.0h** 单元测试通过（fast 12/12）

**最终指标**（[results.csv](file:///d:/Desktop/k9-training-system/runs/train-2/results.csv)）：
- Box mAP50: 0.9914 / mAP50-95: 0.8636
- Pose mAP50: 0.9239 / mAP50-95: 0.5167

**注**：原 1.0e（mAP50-95 ≥ 70%）阈值对 yolo26n 不现实，已弃用。原 1.0f（工作犬测试集）用户拿不到基地视频，改用公开数据集评估（见 Phase 1.5）。

### Phase 1.1 TensorRT FP16 加速（✅ 验收通过，2026-07-27）

**Owner**: ML 开发（`backend/ml/pose/`）

- ✅ **1.1a-c** 改用 PyPI 官方包 `pip install tensorrt==10.8.0.43`（比手动 ZIP 安装简单，无需配置 PATH）
- ✅ **1.1d** 导出 best.pt → best.engine（FP16，14.8 MB，构建 16 min）
- ⚠️ **1.1e** 延迟验证：TensorRT 17.05ms（未达 ≤5ms 目标）；ONNX Runtime GPU 13.39ms（生产推荐）
- ✅ **1.1f** 集成到 `inference.py`（支持 .pt / .engine / .onnx 三模式自动切换）
- ✅ **1.1g** ONNX Runtime GPU 回退方案已实现（比 TensorRT 快 21%）
- ⏳ **1.1h** 模型元数据入库（Phase 1.4 集成时完成）

**实测结论**：YOLO26n-pose 模型过小 + Blackwell 新架构，TensorRT 无加速收益。生产用 ONNX Runtime GPU（74.7 FPS），engine 保留备选。详见 §6.2 验收说明。

### Phase 1.2 科目规则引擎 P0 8 类行为

**Owner**: ML 开发（`backend/ml/behavior/`）

- **1.2a** 定义 24 关键点几何规则
  - **坐**：后躯关键点接近地面，前躯直立
  - **卧**：躯干关键点全部接近地面
  - **立**：四腿直立，躯干高于地面
  - **随行**：犬与人体检测框相对运动（用 YOLO26 默认 COCO person 类）
  - **坐立**：坐姿 + 头部抬起
  - **停留**：状态保持 ≥ N 帧（N=15，约 0.5s @ 30fps）
  - **叫**：嘴部关键点（nose ↔ chin）距离周期性变化
  - **咬**：嘴部 + 前爪动作组合
- **1.2b** 实现 `backend/ml/behavior/rule_engine.py`
  - 输入：关键点序列 `(T, 24, 3)`
  - 输出：`[(行为类别, 起始帧, 结束帧, 置信度), ...]`
- **1.2c** 单元测试：每类行为至少 1 个测试用例（合成关键点序列）
- **1.2d** 集成测试：1 段真实视频 → 行为识别结果
- **1.2e** 行为结果入库（`behaviors` 表）
- **1.2f** 准确率评估（用 DogMo 测试集，决定是否触发 Phase 1.3）
  - **若 < 80% → 触发 Phase 1.3 mmaction2 + PoseC3D**
  - **若 ≥ 80% → 跳过 Phase 1.3**

### Phase 1.3 mmaction2 + PoseC3D 尝试（条件触发）

**Owner**: ML 开发（`backend/ml/behavior/`）

**触发条件**: Phase 1.2f 规则引擎准确率 < 80%

- **1.3a** 安装 VS Buildtools 2022（C++ 桌面开发 + Windows SDK）
- **1.3b** 源码编译 mmcv 2.1.0
- **1.3c** 安装 mmaction2（`pip install -v -e .`）
- **1.3d** 若 1.3b/1.3c 失败：
  - 记录失败原因到 `decisions/0004-posec3d-postponed.md`
  - 跳过 1.3e-1.3h，Phase 1 仅用规则引擎
- **1.3e** 准备 PoseC3D 训练数据（DogMo pkl 格式，8 类）
- **1.3f** 训练 PoseC3D slowonly_r50_u48（240 epoch，8 类）
- **1.3g** 对比规则引擎 vs PoseC3D 精度
- **1.3h** 选择更高精度的方案作为评分输入

### Phase 1.4 评分引擎（YAML 配置化）+ PDF 报告 + 前后端集成

**Owner**: 后端开发 + 前端开发 + ML 开发（评分）

- ✅ **1.4a** **评分卡 YAML Schema 设计文档** `dev-docs/complex-features/scoring-card-schema.md`
  - 评分卡结构：dimensions / weight / rules / thresholds
  - 双场景评分卡：选育 3 维 + 科目 5 维
  - 热加载机制
- ✅ **1.4b** F4 评分引擎 `backend/ml/scoring/`
  - `engine.py`：评分引擎核心（单例 + mtime 热加载）
  - `schema.py`：Pydantic 数据模型
  - `conditions.py`：条件表达式解析
  - `configs/puppy_selection.yaml`：选育评分卡
  - `configs/obedience_trial.yaml`：科目评分卡
- ✅ **1.4c** F5 报告生成 `backend/app/services/report.py`
  - reportlab 模板：犬只信息 / 视频缩略图 / 行为时间线 / 评分表 / 关键帧截图
  - 输出：`reports/{video_id}.pdf`
- ✅ **1.4d** F1 视频上传 API + Celery 推理任务
  - `POST /api/videos/upload`（multipart + scene 字段：puppy_selection / obedience_trial）
  - Celery `ingest_video` 异步任务（姿态 → 行为/信号 → 评分 → PDF 全管线）
  - 状态轮询 `GET /api/videos/{id}/status`
  - 报告下载 `GET /api/videos/{id}/report`
  - 数据库扩展：`videos.scene` + `videos.report_path` + `behavior_class` 枚举新增 SIT_UP/STAY
  - 单元测试 `test_ingest_video.py`（26 用例，覆盖行为映射/信号提取/管线集成/任务配置）
- **1.4e** F6 犬只档案 CRUD（部分实现：list/create/get，缺 put/delete）
  - `POST/GET/PUT/DELETE /api/dogs`
- **1.4f** F7 模型与评分卡管理 API（部分实现：models list/get，缺 register/current/scoring configs）
  - `POST /api/models/register`（注册新模型版本）
  - `GET /api/models/current`（查询当前生产模型）
  - `GET /api/scoring/configs`（列出评分卡）
  - `GET /api/scoring/configs/{scene}`（获取评分卡）
  - `PUT /api/scoring/configs/{scene}`（更新评分卡，热加载）
  - `POST /api/scoring/evaluate`（评分）
- **1.4g** 前端页面实现（4 个 view 全部完成）
  - [x] UploadView.vue：拖拽上传 + 场景/犬只选择 + 状态轮询 + 跳转报告/下载 PDF
  - [x] ReportView.vue：视频元数据 + 7 维评分进度条 + PDF 内嵌预览 + 失败/处理中态
  - [x] HistoryView.vue：视频列表 + 统计卡片 + 场景/状态/犬只筛选 + 跳转报告
  - [x] AdminView.vue：4 Tab（模型管理/评分卡配置/犬只档案/系统信息）+ 注册 Modal + 评分卡 YAML 编辑器 + 试评分 + 犬只 CRUD Modal
  - [x] 类型检查 + 生产构建通过（vue-tsc --noEmit + vite build，4197 模块转译，无错误）
- **1.4h** 前后端联调（Vite proxy → FastAPI）

### Phase 1.5 物体检测集成（选育场景）

**Owner**: ML 开发（`backend/ml/behavior/`）

- **1.5a** YOLO26 默认 COCO 80 类检测集成
  - 球类（sports ball / baseball bat / tennis ball 等）
  - 食物类（banana / apple / sandwich / orange 等）
  - 玩具类（无直接类，用 sports ball 替代）
- **1.5b** 实现 `backend/ml/behavior/object_detector.py`
  - 输入：视频帧
  - 输出：[(物体类别, 检测框, 置信度), ...]
- **1.5c** 单元测试：球/食物检测准确率

### Phase 1.6 选育信号提取器

**Owner**: ML 开发（`backend/ml/behavior/`）

- **1.6a** 实现 `backend/ml/behavior/puppy_signals.py`
  - 食物欲望信号：approach_latency, approach_speed, sniff_duration
  - 玩具欲望信号：chase_latency, chase_speed, hold_duration
  - 胆量信号：retreat_distance, freeze_duration, recovery_time
- **1.6b** 单元测试：合成关键点 + 物体检测 → 信号字典
- **1.6c** 集成测试：1 段选育视频 → 信号字典
- **1.6d** 在 DogMo "Play With Toy" 序列上验证信号提取

### Phase 1.7 系统集成 + 端到端测试（✅ 验收通过，2026-07-28）

**Owner**: 全体

- ✅ **1.7a** 端到端冒烟测试：1 段科目测评视频 → 测评评分 PDF
  - video_id=17，10s 合成视频，处理耗时 3.6s，PDF 4028 字节
- ✅ **1.7b** 端到端冒烟测试：1 段选育视频 → 选育评分 PDF（含物体检测 + 9 信号）
  - video_id=18，15s 合成视频（球+犬形状），处理耗时 11.7s，PDF 3071 字节
- ✅ **1.7c** YAML 评分卡动态修改测试（PUT → 热加载 → 新评分对比）
  - 权重 0.4/0.4/0.2 → 0.6/0.2/0.2 修改生效；热加载机制（mtime 检测）验证通过
- ✅ **1.7d** 性能测试：延迟 ≤ 1 min / min 视频
  - obedience: 0.36x (3.6s/10s)；puppy: 0.78x (11.7s/15s)；均低于 1.0x 阈值
- ✅ **1.7e** 部署文档 `docs/deployment.md`（Windows 安装步骤 + NSSM 服务注册 + 故障排查）
- ✅ **1.7f** 用户手册 `docs/user-guide.md`（训导员使用指南 + 评分卡编辑 + FAQ）

### Phase 1.8 验收 + 学术整理（✅ 验收通过 2026-07-28，带条件）

**Owner**: 全体

- ✅ **1.8a** 验收报告 `reports/phase-1-validation.md`（15 章，§6.1-§6.8 全部证据归档）
- ✅ **1.8b** 学术副产物方向选定：**跨物种行为识别基准**（基于 InterPet4D + Animal Kingdom）
  - 潜在论文方向 3 个：规则引擎 / YAML 评分引擎 / 跨物种姿态迁移
  - 实验规划 P0/P1/P2 三优先级（见验收报告 §11.3）
- ✅ **1.8c** 学术实验整理：Phase 2 并行推进，InterPet4D 真实准确率为 P0 实验
- ✅ **1.8d** 用户决策：Phase 1 验收通过（**带条件**），1.6d/1.2f 真实验证作为 Phase 2 启动前置条件
- ✅ **1.8e** 升级决策记录到 [ADR 0005](../decisions/0005-phase-1-to-phase-2.md)

**验收说明**（2026-07-28）：
- Phase 1 §6.1-§6.7 全部通过，§6.8 三项（验收报告 / 学术方向 / 升级决策）全部完成
- "带条件"指 1.6d 真实序列验证 + 1.2f 真实数据复核作为 Phase 2 启动前置条件（不阻塞 Phase 1 验收）
- 数据集替代方案：DogMo 付费 → InterPet4D（HuggingFace 10.7 GB）+ YouTube 玩球视频，见 [ADR 0006](../decisions/0006-dogmo-open-alternative.md)
- Phase 2 启动脚本：`scripts/download_interpet4d.py` + `scripts/validate_phase2_prereq.py`（Phase 1.8 同步交付）

## 4. 技术决策（沿用 + Phase 1 新增）

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| YOLO26 变体 | n / s / m / l / x | **yolo26n-pose** | ✅ 已微调，速度优先 |
| 关键点数据集 | COCO / Dog-Pose / 自标 | **Dog-Pose（24 点）** | ✅ 已用 |
| TensorRT 量化 | FP16 / INT8 | **FP16** | 精度损失 < 0.5% |
| 行为识别主路径 | 规则 / PoseC3D / 双轨 | **规则为主，PoseC3D 可选** | 保证 MVP 闭环 |
| 评分引擎 | simple-rule-engine / 自研 / 硬编码 | **自研 YAML 配置化** | 训导员可改 + 工作犬场景特殊 |
| 评分卡配置 | JSON / YAML / DB | **YAML** | 可读性 + Git 版本控制 |
| 物体检测 | 自训 / COCO / OpenImages | **YOLO26 默认 COCO 80 类** | 3-4 周内无时间自训 |
| 评估数据 | 基地视频 / 公开数据 | **公开数据（DogMo + InterPet4D）** | 用户拿不到基地视频 |
| PDF 库 | reportlab / weasyprint / fpdf2 | **reportlab** | Phase 0 已锁定 |

## 5. 验证命令

```bash
# Phase 1.0 YOLO26-pose 评估（已完成）
.venv\Scripts\python.exe -m backend.ml.pose.train --eval-only

# Phase 1.1 TensorRT 导出
.venv\Scripts\python.exe -m backend.ml.pose.export --model best.pt --format engine --half

# Phase 1.2 规则引擎单元测试
.venv\Scripts\pytest.exe backend/tests/ml/test_rule_engine.py -v

# Phase 1.4 API 启动
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload

# Phase 1.4 Celery worker
.venv\Scripts\celery.exe -A backend.workers.celery_app worker -l info --pool solo

# Phase 1.4 前端
cd frontend && npm run dev

# Phase 1.7 端到端
.venv\Scripts\python.exe -m backend.cli.e2e --video data/sample.mp4 --output reports/e2e-test.pdf

# 项目守卫（Truth 变更时）
.venv\Scripts\python.exe scripts\check_project_guardrails.py . --mode bootstrap
```

## 6. 验收清单（出口条件）

### 6.1 姿态检测（Phase 1.0，✅ 已完成）

- ✅ Dog-Pose 数据集下载完成（6773/1703 张）
- ✅ `yolo26n-pose.pt` 微调完成，`best.pt` 生成
- ✅ 推理 API 输出 shape = (T, 24, 3)
- ✅ 1 段真实视频推理无报错
- ✅ `backend/ml/pose/inference.py` 单元测试通过

### 6.2 TensorRT 加速（Phase 1.1，✅ 验收通过）

- [x] TensorRT 10.8 安装成功，`import tensorrt` 可用（10.8.0.43 PyPI）
- [x] `best.engine` 导出成功（14.8 MB，FP16，构建耗时 16 min）+ ONNX 回退方案可用
- [x] 推理延迟实测（RTX 5060 Laptop GPU，imgsz=640，YOLO.predict 完整流程）：
  - `.pt` PyTorch: 17.05 ms/frame（58.6 FPS）
  - `.onnx` ONNX Runtime GPU: 13.39 ms/frame（74.7 FPS）← **生产推荐**
  - `.engine` TensorRT FP16: 17.05 ms/frame（58.6 FPS）
- [x] `inference.py` 支持 .pt / .engine / .onnx 三模式自动切换
- [ ] 模型元数据已入库（Phase 1.4 集成时完成）

**验收说明**（2026-07-27 实测）：
- 原目标 ≤5 ms/frame 未达成。原因：YOLO26n-pose 模型过小（3.3M 参数 / 9.1 GFLOPs），TensorRT kernel 优化收益不明显；RTX 5060 Blackwell 新架构 TensorRT 10.8 优化尚不充分；Ultralytics `predict()` 包含完整预处理/后处理开销。
- ONNX Runtime GPU 反而比 TensorRT 快 21%，已作为 Phase 1 生产推理后端。
- TensorRT engine 保留作为备选（Phase 2 大模型 / 批量推理场景再评估）。
- 延迟 13.39 ms/frame（74.7 FPS）满足 MVP 端到端 ≤1 min/min 视频需求（30fps 视频推理约 2 倍实时）。

### 6.3 科目规则引擎（Phase 1.2）

- [x] P0 8 类行为规则定义文档化
- [x] `rule_engine.py` 实现 8 类识别
- [x] 单元测试：8 类行为各 ≥ 1 测试用例通过
- [x] 真实视频集成测试无报错（`test_e2e_pipeline.py` 双场景通过，2026-07-27）
- [x] 行为结果入库（`behaviors` 表，端到端测试验证）
- [x] 准确率评估完成（决定是否触发 Phase 1.3）

**验收说明**（2026-07-28 评估）：
- 评估脚本: `scripts/eval_rule_engine.py`（支持 DogMo + 合成数据双路径）
- 评估结果: 合成数据 42 样本，准确率 92.9%（≥ 80% 阈值）
- Per-behavior: down/bite 100% F1；sit/stand/heel 中等；sit_up/stay/bark 精度偏低（规则重叠导致 FP）
- 局限性: 合成数据无法反映真实视频中的遮挡/模糊/姿态变异；DogMo 数据集需购买获取，真实评估待补
- **决策: 跳过 Phase 1.3 PoseC3D**（规则引擎达标，DogMo 真实评估后复核）
- 评估报告: `reports/phase-1.2f-validation.md`

### 6.4 PoseC3D（Phase 1.3，条件触发）

**触发条件**: Phase 1.3 仅在 Phase 1.2 准确率 < 80% 时启动

- [x] 若未触发：在验收报告中记录"规则引擎准确率 ≥ 80%，跳过 PoseC3D"
- [ ] 若触发且 mmcv 编译成功：mmaction2 安装 + PoseC3D 训练 + 精度对比报告
- [ ] 若触发但 mmcv 编译失败：失败原因记录到 `decisions/0004-posec3d-postponed.md`

**验收说明**（2026-07-28）：Phase 1.2f 合成评估准确率 92.9% ≥ 80%，Phase 1.3 未触发。DogMo 真实评估完成后需复核。

### 6.5 评分引擎 + 报告 + 前后端（Phase 1.4）

- [x] **`dev-docs/complex-features/scoring-card-schema.md` 评分卡 YAML Schema 文档**
- [x] 评分引擎核心实现（engine.py + schema.py + conditions.py）
- [x] 双场景评分卡 YAML（puppy_selection.yaml + obedience_trial.yaml）
- [x] 评分卡热加载机制（ScoringEngine 单例 + mtime 检测）
- [x] PDF 报告模板完成（含双场景评分）
- [x] F1 视频上传 API 通过测试（`POST /api/videos/upload` + 状态轮询 + 报告下载）
- [x] Celery `ingest_video` 推理任务全管线跑通（姿态 → 行为/信号 → 评分 → PDF）
- [x] `test_ingest_video.py` 26 用例通过（行为映射/信号提取/管线集成/任务配置）
- [x] F6 犬只档案 CRUD 通过测试（list/create/get/put/delete 全部实现 + 冒烟测试通过）
- [x] F7 模型与评分卡管理 API 通过测试（register/current/activate + scoring configs CRUD + evaluate 冒烟测试通过）
- [x] 4 个前端页面（Upload/Report/History/Admin）功能完整
- [x] 前后端联调通过

**验收说明**（2026-07-28 联调）：
- 集成测试脚本: `scripts/integration_test.py`（9 项测试全部通过）
- 测试覆盖: 健康检查 + dogs/videos CRUD + 评分引擎 API（双场景）+ 视频上传 + Celery 推理 + PDF 报告下载
- Vite proxy 验证: `/health` + `/api/scoring/configs` 通过代理可达
- conditions.py 短路求值 bug 修复: BoolOp 未实现短路，信号缺失时 `or` 表达式误判为 False

### 6.6 物体检测 + 选育信号（Phase 1.5 + 1.6）

- [x] YOLO26 COCO 物体检测集成（球/食物类）
- [x] `puppy_signals.py` 实现 3 维信号提取
- [x] 单元测试：合成数据 → 信号字典
- [x] 集成测试：1 段选育视频 → 信号字典
- [ ] DogMo "Play With Toy" 序列验证

**验收说明**（2026-07-28）：
- `object_detector.py`: YOLO26 COCO 80 类检测，筛选 person/dog/ball/food/toy；单例缓存 + 流式推理
- `puppy_signals.py`: 3 维 9 信号提取（食物欲望/猎物欲望/胆量），与 `puppy_selection.yaml` v1.1.0 对齐
- `tasks.py` 集成: `_run_puppy_pipeline` 调用 `extract_puppy_signals` + `ObjectDetector`
- `puppy_selection.yaml` 升级 v1.1.0: 新增 `sniff_duration` + `chase_speed` 信号
- 单元测试: `test_object_detector.py` + `test_puppy_signals.py` 全部通过
- DogMo "Play With Toy" 验证待 DogMo 数据集获取后补充

### 6.7 系统集成 + 端到端（Phase 1.7，✅ 验收通过 2026-07-28）

- [x] 1 段科目测评视频 → 测评评分 PDF 全流程跑通
- [x] 1 段选育视频 → 选育评分 PDF 全流程跑通
- [x] YAML 评分卡动态修改测试通过
- [x] 端到端延迟 ≤ 1 min / min 视频
- [x] 部署文档完成
- [x] 用户手册完成

**验收说明**（2026-07-28 端到端测试）：
- 测试脚本: `scripts/phase1_7_e2e_test.py`（7 项测试全部通过）
- 测试覆盖: 健康检查 + 犬只创建 + 模型预热 + 双场景端到端 + YAML 动态修改 + 延迟验证
- **1.7a 科目测评**: video_id=17，10s 合成视频，3.6s 处理完成，PDF 4028 字节
- **1.7b 幼犬选育**: video_id=18，15s 合成视频（球+犬形状触发物体检测），11.7s 处理完成，PDF 3071 字节
- **1.7c YAML 动态修改**: PUT 评分卡 API → Schema 校验 → 文件写入 → mtime 检测热加载，权重调整 0.4/0.4/0.2 → 0.6/0.2/0.2 验证通过
- **1.7d 延迟验证**: obedience 0.36x / puppy 0.78x，均 ≤ 1.0x 预算（模型预热后稳态测量）
- **1.6d 替代验证**: DogMo 数据集需购买，改用合成球+犬形状视频验证选育管线（物体检测 + 9 信号 + 评分 + PDF），DogMo 真实序列待数据集获取后补充
- 部署文档: `docs/deployment.md`（Windows 安装 + NSSM 服务 + 故障排查 + 数据备份）
- 用户手册: `docs/user-guide.md`（训导员使用指南 + 评分卡 YAML 编辑 + FAQ + API 速查）

### 6.8 验收 + 学术（Phase 1.8，✅ 验收通过 2026-07-28，带条件）

- [x] `reports/phase-1-validation.md` 验收报告归档（15 章完整）
- [x] 学术副产物方向选定（跨物种行为识别基准，InterPet4D + Animal Kingdom）
- [x] 用户确认升级 Phase 2（**带条件**：1.6d/1.2f 真实验证为 Phase 2 启动前置）

## 7. 出口决策

Phase 1 MVP 验收通过（**带条件**），见 `reports/phase-1-validation.md` §14。

升级决策记录到 [ADR 0005: Phase 1 → Phase 2 升级决策](../decisions/0005-phase-1-to-phase-2.md)。

**Phase 2 启动前置条件**（不阻塞 Phase 1 验收）：
1. ⏳ 1.6d 真实序列验证（InterPet4D kp_world + YouTube 玩球视频）
2. ⏳ 1.2f 真实数据复核（InterPet4D sit/down/stand/come 准确率，复核 Phase 1.3 跳过决策）

执行脚本：`scripts/download_interpet4d.py` + `scripts/validate_phase2_prereq.py --task 1.6d|1.2f`

## 8. 不可逆操作清单

本阶段涉及以下不可逆操作，需在执行前再次确认：

- ⏳ TensorRT 10.8 系统安装（PATH 修改）
- ⏳ **条件性**：VS BuildTools 2022 安装（约 5GB，仅当 Phase 1.3 触发）
- ⏳ **条件性**：mmcv 源码编译（GPU+CPU 资源消耗，仅当 Phase 1.3 触发）
- ⏳ Phase 1.4 数据库 migration（behaviors / models / scoring_configs 表）
- ⏳ DogMo / InterPet4D 数据集下载（公开数据，本地存储）

## 9. 风险矩阵

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 3-4 周做不完 | 40% | 高 | 砍幼犬选育，仅交付科目测评 |
| mmcv 编译失败 | 30% | 中 | 规则引擎兜底，Phase 2 评估 ST-GCN+BC 直上 |
| PoseC3D 训练不收敛 | 50% | 中 | 规则引擎兜底 |
| TensorRT engine 导出失败 | 10% | 低 | 退化为 ONNX Runtime GPU |
| 规则引擎准确率 < 80% | 50% | 中 | 触发 Phase 1.3 PoseC3D 尝试 |
| 端到端延迟 > 1 min/min 视频 | 20% | 中 | TensorRT + 关键帧采样 |
| 前后端联调接口不一致 | 40% | 低 | Phase 1.4 开始前先定 OpenAPI 契约 |
| 评分卡 YAML 设计不当 | 30% | 中 | 先做最简版本，基地反馈后迭代 |
| 公开数据集下载受限 | 20% | 中 | 退化为优酷视频 + 自标 |

## 10. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-26 | Phase 1 计划创建，基于 ADR 0003 |
| v1.1 | 2026-07-26 | 项目体检后修订：删除周期承诺；新增条件触发 |
| v2.0 | 2026-07-27 | 用户立项研讨后重写：3-4 周约束 + 双场景（选育 + 测评）+ YAML 评分引擎 + 公开数据策略 + 学术副产物；删除 70% mAP 阈值（不现实）；删除 1.0-extra 基地数据采集（用户拿不到） |
| v2.1 | 2026-07-27 | Phase 1.4a-d 验收：评分卡 Schema + 评分引擎（含热加载）+ PDF 报告 + 视频上传 API + Celery 推理任务 + 26 单元测试全部通过；标记 1.4e/f/g/h 未完成 |
| v2.2 | 2026-07-27 | Phase 1.4g 完成：4 个前端 view 全部重写（Upload/Report/History/Admin）+ 类型检查通过 + 生产构建通过（4197 模块，无错误）；新增评分查询 API 客户端 |
| v2.3 | 2026-07-28 | Phase 1.2f / 1.4h / 1.5 / 1.6 并行验收通过：1.2f 合成数据准确率 92.9% → 跳过 Phase 1.3 PoseC3D；1.4h 前后端联调 9 项集成测试通过（含 conditions.py 短路求值 bug 修复）；1.5 `object_detector.py` YOLO26 COCO 物体检测集成（球/食物/玩具/人/犬筛选 + 单例缓存 + 流式推理）；1.6 `puppy_signals.py` 3 维 9 信号提取 + `puppy_selection.yaml` v1.1.0 升级 + `tasks.py` 集成。待办：1.6d DogMo "Play With Toy" 真实序列验证（待数据集获取）。 |
| v2.4 | 2026-07-28 | Phase 1.7 验收通过：1.7a/b 双场景端到端冒烟测试（合成视频 → 推理 → PDF 报告，7/7 PASS）；1.7c YAML 评分卡动态修改测试通过（PUT → 热加载 → 评分对比）；1.7d 延迟验证 obedience 0.36x + puppy 0.78x（均 ≤ 1.0x）；1.7e 部署文档 `docs/deployment.md` 完成（Windows + NSSM + 故障排查）；1.7f 用户手册 `docs/user-guide.md` 完成（训导员指南 + 评分卡编辑 + FAQ）。1.6d DogMo 替代验证通过（合成球+犬形状视频跑通选育管线），真实数据集待购买后补做 1.2f 复核。 |
| v2.5 | 2026-07-28 | Phase 1.8 验收通过（带条件）：1.8a 验收报告 `reports/phase-1-validation.md` 归档（15 章）；1.8b 学术方向选定（跨物种行为识别基准，InterPet4D + Animal Kingdom）；1.8c 学术实验整理（Phase 2 并行，P0/P1/P2 三优先级）；1.8d 用户决策 Phase 1 验收通过（带条件）；1.8e 升级决策 ADR 0005 创建。Phase 2 启动前置条件：1.6d 真实序列验证（InterPet4D kp_world + YouTube 玩球视频）+ 1.2f 真实数据复核（InterPet4D sit/down/stand/come 准确率，复核 Phase 1.3 跳过决策）。配套脚本 `scripts/download_interpet4d.py` + `scripts/validate_phase2_prereq.py` 同步交付。 |
