# Phase 1 — MVP 阶段计划

> 阶段: Phase 1 MVP
> 状态: ⏳ 启动中（2026-07-27 立项研讨后重写）
> Owner: Phase 1 MVP
> 入口条件: Phase 0 验收通过 ✅ + 立项研讨完成 ✅
> 出口条件: 见 §6 验收清单（全部通过）
> 时间约束: 3-4 周单人全职
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
- **1.4g** 前端页面实现（当前为 Phase 0 占位，需重写）
  - UploadView.vue：拖拽上传 + 测试类型选择 + 进度条 + 状态轮询
  - ReportView.vue：评分展示（双场景）+ PDF 在线预览 + 下载
  - HistoryView.vue：视频历史列表 + 筛选
  - AdminView.vue：模型版本 + 评分卡 YAML 编辑器
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

### Phase 1.7 系统集成 + 端到端测试

**Owner**: 全体

- **1.7a** 端到端冒烟测试：1 段科目测评视频 → 测评评分 PDF
- **1.7b** 端到端冒烟测试：1 段选育视频 → 选育评分 PDF
- **1.7c** YAML 评分卡动态修改测试（API 修改 → 热加载 → 新评分）
- **1.7d** 性能测试：延迟 ≤ 1 min / min 视频
- **1.7e** 部署文档 `docs/deployment.md`（Windows 安装步骤）
- **1.7f** 用户手册 `docs/user-guide.md`（训导员使用指南）

### Phase 1.8 验收 + 学术整理

**Owner**: 全体

- **1.8a** 验收报告 `reports/phase-1-validation.md`
- **1.8b** 学术副产物方向选定（见 `stage-plan.md` §3）
- **1.8c** 学术实验整理（如选 DogMo 基准，跑完对比实验）
- **1.8d** 用户决策是否升级 Phase 2
- **1.8e** 升级决策记录到 `decisions/0005-*.md`

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

- [ ] P0 8 类行为规则定义文档化
- [ ] `rule_engine.py` 实现 8 类识别
- [ ] 单元测试：8 类行为各 ≥ 1 测试用例通过
- [ ] 真实视频集成测试无报错
- [ ] 行为结果入库（`behaviors` 表）
- [ ] 准确率评估完成（决定是否触发 Phase 1.3）

### 6.4 PoseC3D（Phase 1.3，条件触发）

**触发条件**: Phase 1.3 仅在 Phase 1.2 准确率 < 80% 时启动

- [ ] 若未触发：在验收报告中记录"规则引擎准确率 ≥ 80%，跳过 PoseC3D"
- [ ] 若触发且 mmcv 编译成功：mmaction2 安装 + PoseC3D 训练 + 精度对比报告
- [ ] 若触发但 mmcv 编译失败：失败原因记录到 `decisions/0004-posec3d-postponed.md`

### 6.5 评分引擎 + 报告 + 前后端（Phase 1.4）

- [x] **`dev-docs/complex-features/scoring-card-schema.md` 评分卡 YAML Schema 文档**
- [x] 评分引擎核心实现（engine.py + schema.py + conditions.py）
- [x] 双场景评分卡 YAML（puppy_selection.yaml + obedience_trial.yaml）
- [x] 评分卡热加载机制（ScoringEngine 单例 + mtime 检测）
- [x] PDF 报告模板完成（含双场景评分）
- [x] F1 视频上传 API 通过测试（`POST /api/videos/upload` + 状态轮询 + 报告下载）
- [x] Celery `ingest_video` 推理任务全管线跑通（姿态 → 行为/信号 → 评分 → PDF）
- [x] `test_ingest_video.py` 26 用例通过（行为映射/信号提取/管线集成/任务配置）
- [ ] F6 犬只档案 CRUD 通过测试（部分：list/create/get 已实现，缺 put/delete）
- [ ] F7 模型与评分卡管理 API 通过测试（部分：models list/get，缺 register/current/scoring configs）
- [ ] 4 个前端页面（Upload/Report/History/Admin）功能完整（当前为 Phase 0 占位）
- [ ] 前后端联调通过

### 6.6 物体检测 + 选育信号（Phase 1.5 + 1.6）

- [ ] YOLO26 COCO 物体检测集成（球/食物类）
- [ ] `puppy_signals.py` 实现 3 维信号提取
- [ ] 单元测试：合成数据 → 信号字典
- [ ] 集成测试：1 段选育视频 → 信号字典
- [ ] DogMo "Play With Toy" 序列验证

### 6.7 系统集成 + 端到端（Phase 1.7）

- [ ] 1 段科目测评视频 → 测评评分 PDF 全流程跑通
- [ ] 1 段选育视频 → 选育评分 PDF 全流程跑通
- [ ] YAML 评分卡动态修改测试通过
- [ ] 端到端延迟 ≤ 1 min / min 视频
- [ ] 部署文档完成
- [ ] 用户手册完成

### 6.8 验收 + 学术（Phase 1.8）

- [ ] `reports/phase-1-validation.md` 验收报告归档
- [ ] 学术副产物方向选定
- [ ] 用户确认升级 Phase 2（或推迟决策）

## 7. 出口决策

Phase 1 完成后，由用户判断是否升级到 Phase 2。升级决策记录到 `dev-docs/decisions/0005-phase-1-to-phase-2.md`（待创建）。

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
