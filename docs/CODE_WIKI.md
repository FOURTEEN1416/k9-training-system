# Code Wiki — 工作犬训练机器视觉识别系统

> 本文档为代码级开发参考，描述项目整体架构、模块职责、关键类与函数、依赖关系与运行方式。
> Truth 文档位于 `dev-docs/`（架构/选型/阶段计划/决策记录）。
> 生成日期：2026-08-01｜当前阶段：Phase 3 启动中

---

## 目录

1. [项目概览](#1-项目概览)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [后端模块详解](#4-后端模块详解)
   - 4.1 [FastAPI 应用层](#41-fastapi-应用层)
   - 4.2 [API 路由](#42-api-路由)
   - 4.3 [数据模型层（SQLAlchemy）](#43-数据模型层sqlalchemy)
   - 4.4 [服务层](#44-服务层)
   - 4.5 [Celery 异步任务](#45-celery-异步任务)
5. [ML 模块详解](#5-ml-模块详解)
   - 5.1 [姿态检测 pose](#51-姿态检测-pose)
   - 5.2 [行为识别 behavior](#52-行为识别-behavior)
   - 5.3 [评分引擎 scoring](#53-评分引擎-scoring)
   - 5.4 [多犬追踪 tracking](#54-多犬追踪-tracking)
6. [前端模块](#6-前端模块)
7. [数据库设计](#7-数据库设计)
8. [依赖关系](#8-依赖关系)
9. [端到端数据流](#9-端到端数据流)
10. [项目运行方式](#10-项目运行方式)
11. [测试体系](#11-测试体系)

---

## 1. 项目概览

**项目名称**：工作犬训练机器视觉识别系统（K9 Training Vision System）

**定位**：面向公安/海关工作犬训练场景的本地化机器视觉识别系统，实现「视频上传 → 姿态检测 → 行为识别 → 评分报告」端到端闭环，无需人工干预。

**核心能力**：
- **24 关键点姿态检测**：基于 YOLO26-pose + Dog-Pose 24 关键点体系（左侧肢/右侧肢/尾/耳/头/眼/鬐甲/喉咙）
- **22 种行为分类**：P0 基础 8 类 + P1 训练 8 类 + P2 高级 6 类（FCI-IGP）
- **7 维评分体系**：准确度/响应延迟/保持/搜索效率/注意力/胆量/步态
- **三大标准兼容**：GA-T（公安）/ USPCA（美国警犬认证）/ FCI-IGP（国际工作犬）
- **多场景管线**：幼犬选育 / 科目测评 / 工作犬综合 / USPCA 巡逻

**技术栈概览**：

| 层 | 选型 |
|----|------|
| 后端框架 | FastAPI 0.115 + Uvicorn（异步） |
| ORM | SQLAlchemy 2.0 async + asyncpg |
| 数据库 | PostgreSQL 17 + pgvector（向量检索） |
| 任务队列 | Celery 5.4 + Redis（异步推理） |
| ML | PyTorch 2.11+cu128 / ultralytics / ONNX Runtime / TensorRT |
| 前端 | Vue 3 + Vite + Naive UI + ECharts + Pinia |
| 部署 | 原生 Python venv（Windows 优先），Phase 3+ Jetson |

**架构风格**：单体应用 + 异步任务队列（Phase 0-3），微服务化留待 Phase 5。

**硬约束**：本地部署无云依赖｜数据不出本机｜Windows 优先｜精度优先于速度。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                  用户浏览器（Vue 3 + Naive UI）              │
│   上传视频 │ 查看评分报告 │ 历史对比 │ 评分卡管理 │ 模型管理   │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP REST + 轮询
┌──────────────────────────▼──────────────────────────────────┐
│                  FastAPI 后端（异步）                         │
│  videos / scores / dogs / models / scoring / annotations /   │
│  finetune / health                                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼───────┐  ┌───────▼───────┐  ┌──────▼────────┐
│   Celery      │  │ PostgreSQL 17  │  │   Redis       │
│  异步推理队列  │  │  + pgvector    │  │ Broker+缓存   │
│ ┌───────────┐ │  │ ┌────────────┐ │  └───────────────┘
│ │Pose 推理   │ │  │ │业务数据     │ │
│ │YOLO26-pose │ │  │ │关键点序列   │ │
│ │MotionBERT  │ │  │ │行为向量     │ │
│ ├───────────┤ │  │ └────────────┘ │
│ │行为识别    │ │  └────────────────┘
│ │规则引擎    │ │
│ │ST-GCN+BC  │ │
│ ├───────────┤ │
│ │评分引擎    │ │
│ │YAML 配置化 │ │
│ └───────────┘ │
└───────────────┘
```

**进程拓扑**（4 个独立进程）：

| 进程 | 作用 | 启动命令 |
|------|------|---------|
| FastAPI | API 服务 | `uvicorn backend.app.main:app` |
| Celery Worker | 异步推理 | `celery -A backend.workers.celery_app worker -l info --pool solo` |
| PostgreSQL | 主数据库 | Windows 服务 `postgresql-17`（端口 5433） |
| Redis | Broker + 缓存 | 端口 6379 |

**通信协议**：
- 前端 ↔ 后端：HTTP REST + 状态轮询（视频推理进度）
- 后端 ↔ Celery：Redis Broker
- Celery ↔ DB：SQLAlchemy 同步 Session（Celery 任务内）
- Celery ↔ ML：进程内调用（PyTorch）

---

## 3. 目录结构

```
k9-training-system/
├── AGENTS.md                  # Agent 行为宪法（项目硬约束 + Owner Map）
├── alembic.ini                # Alembic 迁移配置
├── backend/                   # 后端 + ML 核心代码
│   ├── alembic/               # 数据库迁移脚本
│   ├── app/                   # FastAPI 应用层
│   │   ├── api/               # API 路由（8 个 router）
│   │   ├── core/              # 配置 / 数据库引擎
│   │   ├── models/            # SQLAlchemy ORM 模型（10 张表）
│   │   ├── schemas/           # Pydantic 请求/响应 schema
│   │   ├── services/          # 业务服务（PDF 报告 / Label Studio）
│   │   └── main.py            # FastAPI 入口
│   ├── ml/                    # 机器学习模块
│   │   ├── pose/              # 姿态检测（YOLO26 + MotionBERT 3D）
│   │   ├── behavior/          # 行为识别（规则引擎 + ST-GCN+BC）
│   │   ├── scoring/           # 评分引擎（YAML 配置化）
│   │   └── tracking/          # 多犬追踪（BoxMOT + ReID）
│   ├── tests/                 # 单元 + 集成测试
│   ├── workers/               # Celery 任务定义
│   ├── .env.example           # 环境变量模板
│   └── requirements.txt       # Python 依赖
├── frontend/                  # Vue 3 前端
│   └── src/
│       ├── api/               # axios 封装 + 类型定义
│       ├── layouts/           # 布局组件
│       ├── router/            # Vue Router
│       ├── stores/            # Pinia 状态管理
│       └── views/             # 4 个页面（Upload/Report/History/Admin）
├── external/                  # 外部依赖（MotionBERT 源码 + Python 库）
│   └── MotionBERT/            # 3D 姿态 lifting 参考实现
├── dev-docs/                  # Truth 文档（架构/选型/阶段/决策）
├── docs/                      # 用户文档（部署/使用指南/本 Wiki）
├── data/                      # 数据目录（.gitignore，运行时生成）
│   ├── uploads/               # 上传视频
│   ├── generated/             # 生成的标注视频
│   └── models_weights/        # 模型权重
└── reports/                   # 生成的 PDF 评分报告
```

> `external/_pylibs/` 为 vendored Python 库（numpy/onnx），非项目代码。

---

## 4. 后端模块详解

### 4.1 FastAPI 应用层

入口文件：[main.py](file:///d:/Desktop/k9-training-system/backend/app/main.py)

```python
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",          # Swagger UI
    redoc_url="/redoc",        # ReDoc
    lifespan=lifespan,         # 启动时 ensure_dirs()
)
```

- **CORS**：开发期允许 Vite 端口（5173/8000）
- **路由注册**：8 个 router，前缀 `/api`（health 除外）
- **生命周期**：启动时通过 `settings.ensure_dirs()` 创建 `data/` 各子目录

**配置管理**：[config.py](file:///d:/Desktop/k9-training-system/backend/app/core/config.py) — `Settings(BaseSettings)` 类，配置来源优先级：环境变量 > `.env` 文件 > 默认值。通过 `@lru_cache` 单例 `get_settings()` 提供。配置项涵盖应用/数据库/Redis/路径/上传限制/CORS/Label Studio。

**数据库引擎**：[database.py](file:///d:/Desktop/k9-training-system/backend/app/core/database.py) — 异步引擎（asyncpg）+ `AsyncSessionLocal` 会话工厂 + `get_db()` FastAPI 依赖。另有 [database_sync.py](file:///d:/Desktop/k9-training-system/backend/app/core/database_sync.py) 提供同步 Session（Celery 任务用）。

### 4.2 API 路由

所有路由位于 [backend/app/api/](file:///d:/Desktop/k9-training-system/backend/app/api/)，统一使用 `APIRouter` + `DbSession = Annotated[AsyncSession, Depends(get_db)]` 依赖注入。

| Router | 文件 | 前缀 | 核心端点 |
|--------|------|------|---------|
| health | [health.py](file:///d:/Desktop/k9-training-system/backend/app/api/health.py) | - | `GET /health`、`GET /` |
| dogs | [dogs.py](file:///d:/Desktop/k9-training-system/backend/app/api/dogs.py) | `/api/dogs` | CRUD：`GET ""` / `POST ""` / `GET /{id}` / `PUT /{id}` / `DELETE /{id}` |
| videos | [videos.py](file:///d:/Desktop/k9-training-system/backend/app/api/videos.py) | `/api/videos` | `POST /upload`（上传+触发 Celery）、`GET ""`、`GET /{id}`、`GET /{id}/status`（轮询）、`GET /{id}/report`（下载 PDF） |
| scores | [scores.py](file:///d:/Desktop/k9-training-system/backend/app/api/scores.py) | `/api/scores` | `GET ""`、`GET /by-dog/{dog_id}`（按犬历史查询）、`GET /{score_id}` |
| models | [models.py](file:///d:/Desktop/k9-training-system/backend/app/api/models.py) | `/api/models` | `GET ""`、`GET /current?type=`、`POST /register`、`POST /{id}/activate`（切换激活版本） |
| scoring | [scoring.py](file:///d:/Desktop/k9-training-system/backend/app/api/scoring.py) | `/api/scoring` | `GET /configs`、`GET /configs/{scene}`、`PUT /configs/{scene}`（触发热加载）、`POST /evaluate`（实时评分预览） |
| annotations | [annotations.py](file:///d:/Desktop/k9-training-system/backend/app/api/annotations.py) | `/api/annotations` | 标注任务 CRUD + Label Studio 同步（`/tasks`、`/sync`、`/ls/health` 等） |
| finetune | [finetune.py](file:///d:/Desktop/k9-training-system/backend/app/api/finetune.py) | `/api/finetune` | `POST /trigger`（异步微调）、`GET /status`、`GET /pipeline`（数据飞轮状态） |

**关键设计**：
- **视频上传**：流式写盘（1MB chunk），超 2048MB 中断删除；UUID 存储文件名防冲突；上传后 `ingest_video.delay(video_id)` 异步派发
- **评分卡热加载**：`PUT` 写入 YAML 后，`ScoringEngine.get()` 通过 mtime 检测自动重载
- **错误处理**：统一 HTTPException + `{"detail": "..."}` 格式

### 4.3 数据模型层（SQLAlchemy）

所有模型位于 [backend/app/models/](file:///d:/Desktop/k9-training-system/backend/app/models/)，继承 `TimestampMixin, Base`（[base.py](file:///d:/Desktop/k9-training-system/backend/app/models/base.py) 提供 `id` + `created_at`/`updated_at`）。`__init__.py` 统一导出以供 Alembic autogenerate 注册元数据。

| 模型 | 文件 | 表名 | 职责 |
|------|------|------|------|
| `Handler` | [handler.py](file:///d:/Desktop/k9-training-system/backend/app/models/handler.py) | `handlers` | 训导员档案（角色 `UserRole`） |
| `Dog` | [dog.py](file:///d:/Desktop/k9-training-system/backend/app/models/dog.py) | `dogs` | 犬只档案（品种/性别/芯片号/训练阶段 `TrainingStage` P0/P1/P2） |
| `TrainingSession` | [training_session.py](file:///d:/Desktop/k9-training-system/backend/app/models/training_session.py) | `training_sessions` | 训练会话（含评分标准 `ScoringStandard`: GA-T/USPCA/FCI-IGP/CUSTOM） |
| `Video` | [video.py](file:///d:/Desktop/k9-training-system/backend/app/models/video.py) | `videos` | 训练视频元数据（状态机 `VideoStatus`: uploaded→processing→completed/failed，场景 scene） |
| `Keypoint` | [keypoint.py](file:///d:/Desktop/k9-training-system/backend/app/models/keypoint.py) | `keypoints` | 逐帧 24 关键点（JSON `[[x,y,conf],...]×24`） |
| `Behavior` | [behavior.py](file:///d:/Desktop/k9-training-system/backend/app/models/behavior.py) | `behaviors` | 行为识别结果（`BehaviorClass` 22 类枚举 + `BehaviorDetector` rule/posec3d/stgcn_bc） |
| `BehaviorVector` | [behavior_vector.py](file:///d:/Desktop/k9-training-system/backend/app/models/behavior_vector.py) | `behavior_vectors` | 行为嵌入向量（pgvector `Vector(256)`） |
| `Score` | [score.py](file:///d:/Desktop/k9-training-system/backend/app/models/score.py) | `scores` | 7 维评分（accuracy/duration/attention 必填，其余按阶段启用） |
| `MLModel` | [ml_model.py](file:///d:/Desktop/k9-training-system/backend/app/models/ml_model.py) | `ml_models` | ML 模型版本（`ModelType` pose/behavior/scoring + `ModelFramework` pytorch/onnx/tensorrt + `is_active`） |
| `AnnotationTask`/`Annotation` | [annotation.py](file:///d:/Desktop/k9-training-system/backend/app/models/annotation.py) | `annotation_tasks`/`annotations` | 数据飞轮标注任务与标注数据 |

**关系拓扑**：`Dog` 1—N `Video` N—1 `TrainingSession`；`Video` 1—N `Keypoint`/`Behavior`/`Score`/`AnnotationTask`；`Behavior` 1—1 `BehaviorVector`。

### 4.4 服务层

位于 [backend/app/services/](file:///d:/Desktop/k9-training-system/backend/app/services/)：

**ReportService** — [report.py](file:///d:/Desktop/k9-training-system/backend/app/services/report.py)
- `ReportInput` dataclass：犬只信息 + 视频信息 + `ScoringResult` + `list[BehaviorEpisode]`
- `generate_report(report_input, output_path)`：基于 reportlab platypus 生成 A4 PDF（标题/犬只表/视频表/评分摘要/维度明细/行为时间线/评分说明）

**LabelStudioClient** — [label_studio.py](file:///d:/Desktop/k9-training-system/backend/app/services/label_studio.py)
- 数据飞轮标注平台 HTTP 客户端
- 认证：Django session cookie（CSRF + form POST，非 JWT）
- 能力：`login()` / `list_projects()` / `list_tasks()` / `create_task()` / `upload_file()` / `export_annotations()`

### 4.5 Celery 异步任务

**Celery 实例**：[celery_app.py](file:///d:/Desktop/k9-training-system/backend/workers/celery_app.py)
- Broker/Backend：Redis（DB 1/2）
- Windows 兼容：`worker_pool="solo"`（不支持 prefork）
- `broker_visibility_timeout=3600`（视频处理耗时长）

**核心任务**：[tasks.py](file:///d:/Desktop/k9-training-system/backend/workers/tasks.py) — `ingest_video(video_id)`

```
@celery_app.task(bind=True, autoretry_for=(Exception,), max_retries=2)
def ingest_video(self, video_id: int) -> dict
```

**执行流程**（视频推理主入口）：
1. 加载 `Video` 元数据 → 标记 `PROCESSING`
2. **YOLO26-pose 推理**：`PoseInferenceEngine.infer_video()` → 逐帧 24 关键点
3. 写入 `keypoints` 表（批量 add）
4. **场景分支**（`video.scene`）：
   - `obedience_trial` → `_run_obedience_pipeline()`：`RuleEngine.recognize()` 16 行为 → signals → `ScoringEngine.evaluate()`
   - `puppy_selection` → `_run_puppy_pipeline()`：物体检测 + `extract_puppy_signals()` 9 信号 → 评分
   - `uspca_patrol` → `_run_uspca_pipeline()`：16 行为 → USPCA 5 维信号 → 评分
5. 写入 `behaviors` 表
6. 生成 PDF 报告 → `reports/{video_id}.pdf`
7. 标记 `COMPLETED` + `report_path`；异常 → `FAILED` + `error_message`

**单例优化**：`_pose_engine` 全局缓存避免每任务重载模型；模型路径优先 `.onnx` → `.pt` → 自动下载。

---

## 5. ML 模块详解

### 5.1 姿态检测 pose

位于 [backend/ml/pose/](file:///d:/Desktop/k9-training-system/backend/ml/pose/)。

**PoseInferenceEngine** — [inference.py](file:///d:/Desktop/k9-training-system/backend/ml/pose/inference.py)

2D 姿态检测核心引擎，封装 ultralytics YOLO26-pose。

```python
class PoseInferenceEngine:
    def __init__(self, model_path, device=0, imgsz=640, conf=0.25, iou=0.7)
    def infer_video(self, video_path, save_output=True) -> VideoInferenceResult
    def infer_image(self, image_path) -> np.ndarray  # (24, 3)
```

- **支持三种后端**：`.pt`（PyTorch）/ `.engine`（TensorRT FP16）/ `.onnx`（ONNX Runtime GPU），YOLO() 按后缀自动选择
- **关键修复**：`YOLO(model_path, task="pose")` 显式指定 task，避免 ONNX/Engine 模型回退到 detect 导致 keypoints=None
- **输出**：`VideoInferenceResult`（meta + `list[FrameResult]` + `keypoints_sequence` 属性 `(T, 24, 3)`）
- **多犬处理**：单犬场景取置信度最高的检测框
- **CLI**：`python -m backend.ml.pose.inference --video <path> --model <path>`

**Dog-Pose 24 关键点体系**（[inference.py](file:///d:/Desktop/k9-training-system/backend/ml/pose/inference.py) `KPT_NAMES` + [constants.py](file:///d:/Desktop/k9-training-system/backend/ml/behavior/constants.py)）：

| 索引 | 部位 |
|------|------|
| 0-5 | 左侧肢（前左爪/膝/肘 + 后左爪/膝/肘） |
| 6-11 | 右侧肢（前右爪/膝/肘 + 后右爪/膝/肘） |
| 12-13 | 尾（尾根/尾尖） |
| 14-15 | 耳根（左/右） |
| 16-17 | 头部中线（鼻/下巴） |
| 18-19 | 耳尖（左/右） |
| 20-21 | 眼（左/右） |
| 22 | 鬐甲 withers（肩峰，骨架根节点） |
| 23 | 喉咙 throat |

**MotionBERT 3D Lifting** — [motionbert/](file:///d:/Desktop/k9-training-system/backend/ml/pose/motionbert/)（Phase 3.3c）
- [model.py](file:///d:/Desktop/k9-training-system/backend/ml/pose/motionbert/model.py)：`DSTformerWrapper` 包装 `external/MotionBERT/lib/model/DSTformer.py`，支持 17→24 关键点权重迁移（pos_embed 丢弃重初始化，其余层 shape 匹配迁移）
- [inference.py](file:///d:/Desktop/k9-training-system/backend/ml/pose/motionbert/inference.py)：`MotionBERTLifter` 输入 2D `(T, 24, 2|3)` → 输出 3D `(T, 24, 3)`，支持 PyTorch / ONNX 双后端 + 滑动窗口推理
- [train.py](file:///d:/Desktop/k9-training-system/backend/ml/pose/motionbert/train.py) / [dataset.py](file:///d:/Desktop/k9-training-system/backend/ml/pose/motionbert/dataset.py)：InterPet4D 数据集微调
- [camera_projection.py](file:///d:/Desktop/k9-training-system/backend/ml/pose/camera_projection.py)：3D↔2D 相机投影
- [lifting_pairing.py](file:///d:/Desktop/k9-training-system/backend/ml/pose/lifting_pairing.py)：归一化/反归一化 + 滑动窗口

### 5.2 行为识别 behavior

位于 [backend/ml/behavior/](file:///d:/Desktop/k9-training-system/backend/ml/behavior/)。

**RuleEngine** — [rule_engine.py](file:///d:/Desktop/k9-training-system/backend/ml/behavior/rule_engine.py)

基于几何规则的 16 类行为识别引擎（可解释、低延迟、无需训练）。

```python
class RuleEngine:
    def recognize(self, keypoints_sequence: np.ndarray,  # (T, 24, 3)
                  boxes=None, person_boxes=None, target_boxes=None,
                  fps=30.0, enable_p1=True) -> list[BehaviorEpisode]
```

- **输入**：`(T, 24, 3)` 关键点序列（图像坐标，y 轴向下）
- **输出**：`list[BehaviorEpisode]`（behavior / start_frame / end_frame / confidence / metadata）
- **P0 基础 8 类**（Phase 1，已验收 92.9%）：
  - `sit`/`down`/`stand`：逐帧姿态分类（后腿折叠度 + 前腿伸直度 + 鬐甲高度比）→ 聚合连续相同姿态
  - `sit_up`：坐姿 + 头部抬起（nose.y < withers.y）
  - `stay`：连续 N 帧关键点位移 < 阈值
  - `heel`：持续移动 + 站立姿态
  - `bark`：nose-chin 距离序列标准差超阈值
  - `bite`：嘴部活跃 + 前爪快速移动
- **P1 训练 8 类**（Phase 2）：`track`/`alert_sit`/`alert_down`/`apprehend`/`escort`/`obstacle`/`recall`/`watch`
  - 完整版支持 `target_boxes`（示警/扑咬接近目标检测）和 `person_boxes`（押解/返回朝向人）
- **可配置阈值**：所有阈值通过构造参数可覆盖（YAML 评分卡亦可）

**便捷函数**：`recognize_behaviors(keypoints_sequence, **kwargs)` — 一次性调用。

**ObjectDetector** — [object_detector.py](file:///d:/Desktop/k9-training-system/backend/ml/behavior/object_detector.py)（Phase 1.5）
- YOLO26 COCO 80 类物体检测，用于幼犬选育场景（球类/食物/玩具/人）
- 单例缓存 `get_detector_singleton()`
- 输出 `VideoDetectionResult`（逐帧检测框序列）

**PuppySignals** — [puppy_signals.py](file:///d:/Desktop/k9-training-system/backend/ml/behavior/puppy_signals.py)（Phase 1.6）
- `extract_puppy_signals(kpts_seq, detections, fps, duration_sec)` → 9 信号字典
- 物体检测失败时降级为纯 pose 估算

**ST-GCN+BC** — [stgcn_bc/](file:///d:/Desktop/k9-training-system/backend/ml/behavior/stgcn_bc/)（Phase 3.1）
- [k9_graph.py](file:///d:/Desktop/k9-training-system/backend/ml/behavior/stgcn_bc/k9_graph.py)：`K9Graph` 24 节点犬类骨架拓扑（pyskl 兼容），根节点 WITHERS(22)
- [labels.py](file:///d:/Desktop/k9-training-system/backend/ml/behavior/stgcn_bc/labels.py)：22 类行为标签映射
- [data_adapter.py](file:///d:/Desktop/k9-training-system/backend/ml/behavior/stgcn_bc/data_adapter.py)：YOLO26 `(T,24,3)` ↔ pyskl 骨架格式转换 + 骨流/运动流计算

**常量定义** — [constants.py](file:///d:/Desktop/k9-training-system/backend/ml/behavior/constants.py)
- 24 关键点索引常量 + 分组（FRONT_PAWS / REAR_KNEES 等）
- 22 类行为常量（P0 8 + P1 8 + P2 6）+ 中文名映射 + 行为→科目映射

### 5.3 评分引擎 scoring

位于 [backend/ml/scoring/](file:///d:/Desktop/k9-training-system/backend/ml/scoring/)。

**ScoringEngine** — [engine.py](file:///d:/Desktop/k9-training-system/backend/ml/scoring/engine.py)

YAML 配置化评分引擎，支持热加载与可解释评分。

```python
class ScoringEngine:
    @classmethod
    def from_yaml(cls, path) -> ScoringEngine       # 直接加载
    @classmethod
    def get(cls, path) -> ScoringEngine              # 单例 + mtime 热加载（推荐）
    def evaluate(self, ctx: ScoringContext) -> ScoringResult
```

**核心特性**：
1. **YAML 配置化**：评分卡定义维度/规则/阈值/权重，训导员可改
2. **热加载**：`get()` 检测文件 mtime，修改后下次调用自动重载
3. **可解释**：每维度返回命中规则 ID + 标签 + 人类可读说明
4. **场景隔离**：4 场景（puppy_selection / obedience_trial / working_dog_trial / uspca_patrol）

**数据流**：`ScoringContext(signals, scene)` → `evaluate()` → `ScoringResult`

**评分逻辑**：
- 每维度按顺序匹配规则（`evaluate_condition` 条件表达式求值），首个命中生效
- 无命中用 `default`，再无则 0 分
- 聚合方式：`weighted_sum`（默认）/ `max` / `min`
- 判定：`pass`(≥70) / `borderline`(≥60) / `fail`

**Schema** — [schema.py](file:///d:/Desktop/k9-training-system/backend/ml/scoring/schema.py)（Pydantic 模型）
- `ScoringCardSpec`：完整评分卡（含权重和校验 = 1.0）
- `DimensionSpec` / `RuleSpec` / `DefaultSpec` / `ThresholdsSpec`
- `ScoringContext`（输入）/ `ScoringResult`（输出，含 `total_score` / `dimension_scores` / `verdict` / `explanation`）

**条件求值** — [conditions.py](file:///d:/Desktop/k9-training-system/backend/ml/conditions.py) — `evaluate_condition(expr, signals)` 安全表达式求值。

**评分卡 YAML** — [configs/](file:///d:/Desktop/k9-training-system/backend/ml/scoring/configs/)
- `uspca_patrol.yaml`（USPCA PDI 5 维）
- `puppy_selection.yaml` / `obedience_trial.yaml` / `working_dog_trial.yaml`

### 5.4 多犬追踪 tracking

位于 [backend/ml/tracking/](file:///d:/Desktop/k9-training-system/backend/ml/tracking/)（Phase 3.2b）。

**MultiDogTracker** — [multi_dog_tracker.py](file:///d:/Desktop/k9-training-system/backend/ml/tracking/multi_dog_tracker.py)
- 架构：YOLO26-pose 检测 → BoxMOT OccluBoost 追踪 → det_ind 关联关键点到 track_id
- 后端枚举 `TrackerBackend`：occluboost（默认，IDF1 最高）/ botsort / bytetrack / strongsort
- 单犬场景自动退化为 1 个 track_id（向后兼容 Phase 1/2）
- 接口：`track_video(video_path)` / `update_frame(dets, kps, img)`

**ReID 与 ID 切换监控**：
- [reid_extractor.py](file:///d:/Desktop/k9-training-system/backend/ml/tracking/reid_extractor.py)：OSNet/lmbn 特征提取
- [id_switch_monitor.py](file:///d:/Desktop/k9-training-system/backend/ml/tracking/id_switch_monitor.py)：多犬 ID 切换率监控（≥0.1 触发 ReID 微调）
- [reid_finetune_dataset.py](file:///d:/Desktop/k9-training-system/backend/ml/tracking/reid_finetune_dataset.py)：ReID 微调数据集构建
- [types.py](file:///d:/Desktop/k9-training-system/backend/ml/tracking/types.py)：`DogTrack` / `DogTrackFrame` / `MultiDogTrackingResult`

---

## 6. 前端模块

位于 [frontend/](file:///d:/Desktop/k9-training-system/frontend/)，Vue 3 + Composition API + TypeScript。

**技术栈**：
- 构建：Vite 6 + vue-tsc
- UI：Naive UI 2.40
- 图表：ECharts 5 + vue-echarts
- 状态：Pinia 2.3
- 路由：Vue Router 4.5
- HTTP：axios 1.7

**入口**：[main.ts](file:///d:/Desktop/k9-training-system/frontend/src/main.ts) — 注册 Pinia + Router + Naive UI

**路由** — [router/index.ts](file:///d:/Desktop/k9-training-system/frontend/src/router/index.ts)

| 路径 | 视图 | 功能 |
|------|------|------|
| `/upload` | [UploadView.vue](file:///d:/Desktop/k9-training-system/frontend/src/views/UploadView.vue) | 视频上传 + 场景选择 |
| `/report/:id?` | [ReportView.vue](file:///d:/Desktop/k9-training-system/frontend/src/views/ReportView.vue) | 评分报告 + PDF 下载 |
| `/history` | [HistoryView.vue](file:///d:/Desktop/k9-training-system/frontend/src/views/HistoryView.vue) | 历史记录对比 |
| `/admin` | [AdminView.vue](file:///d:/Desktop/k9-training-system/frontend/src/views/AdminView.vue) | 系统管理（模型/评分卡） |

**API 封装** — [api/](file:///d:/Desktop/k9-training-system/frontend/src/api/)
- [http.ts](file:///d:/Desktop/k9-training-system/frontend/src/api/http.ts)：axios 实例 + 响应拦截器（统一错误消息）
- [index.ts](file:///d:/Desktop/k9-training-system/frontend/src/api/index.ts)：`api` 对象封装所有端点（health/dogs/videos/models/scoring/scores）+ TypeScript 类型定义（`Video`/`Scene`/`ScoringResult` 等）

**状态管理** — [stores/](file:///d:/Desktop/k9-training-system/frontend/src/stores/)
- [app.ts](file:///d:/Desktop/k9-training-system/frontend/src/stores/app.ts)：全局应用状态
- [dogs.ts](file:///d:/Desktop/k9-training-system/frontend/src/stores/dogs.ts)：犬只列表状态

---

## 7. 数据库设计

**DBMS**：PostgreSQL 17（端口 5433）+ pgvector 0.8.0（向量检索扩展）

**ER 关系**：

```
handlers ──1:N── dogs ──1:N── training_sessions
                     │              │
                     └──1:N── videos ──┘
                                │
        ┌───────────┬───────────┼───────────┬──────────────┐
        ▼           ▼           ▼           ▼              ▼
   keypoints   behaviors    scores   annotation_tasks   (behavior_vectors)
                  │
                  └──1:1── behavior_vectors (pgvector 256 维)
```

**核心表说明**：

| 表 | 主键 | 关键字段 | 索引 |
|----|------|---------|------|
| `dogs` | id | name, breed, chip_id(unique), training_stage | chip_id |
| `videos` | id | status(枚举), scene, storage_path, report_path | status, dog_id, handler_id, session_id |
| `keypoints` | id | video_id(FK CASCADE), frame_idx, keypoints_json(JSON) | video_id |
| `behaviors` | id | video_id, behavior_class(枚举), detector(枚举), start/end_frame, confidence | video_id, behavior_class |
| `scores` | id | video_id, standard(枚举), accuracy/duration/attention + 4 可选维度, overall | video_id |
| `behavior_vectors` | id | behavior_id(FK unique), embedding `Vector(256)` | behavior_id |
| `ml_models` | id | type, framework, is_active, storage_path, metrics_json | type, is_active |

**迁移管理**：Alembic（[alembic.ini](file:///d:/Desktop/k9-training-system/alembic.ini) + [backend/alembic/env.py](file:///d:/Desktop/k9-training-system/backend/alembic/env.py)）
- `env.py` 导入 `backend.app.models.*` 注册元数据
- 从 `settings.pg_dsn` 覆盖 URL（同步 psycopg2 驱动）
- `compare_type=True` + `compare_server_default=True`

---

## 8. 依赖关系

### 8.1 Python 后端依赖（[requirements.txt](file:///d:/Desktop/k9-training-system/backend/requirements.txt)）

**Web 框架**：fastapi 0.115.6 / uvicorn[standard] 0.34 / python-multipart / websockets
**配置**：pydantic 2.10.4 / pydantic-settings 2.7.1
**数据库**：SQLAlchemy[asyncio] 2.0.36 / asyncpg 0.30 / psycopg2-binary 2.9.10 / alembic 1.14 / pgvector 0.5
**任务队列**：celery[redis] 5.4 / redis 5.2.1 / flower 2.0.1
**ML/CV**：numpy / pillow / opencv-python-headless 4.10 / ultralytics 8.4.107
**ONNX 推理**：onnxruntime-gpu 1.20.1 / onnx 1.22 / onnxslim
**TensorRT**：tensorrt 10.8（PyPI 官方包）
**数据处理**：scipy 1.15 / pandas 2.2.3
**报告**：reportlab 4.2.5 / matplotlib 3.10
**工具**：loguru / httpx / python-dateutil / tqdm / PyYAML
**测试**：pytest 8.3.4 / pytest-asyncio

**特殊安装（不通过 pip）**：
- PyTorch 2.11+cu128：`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`（Blackwell sm_120 必需 CUDA 12.8）
- TensorRT 10.8：NVIDIA 官方 ZIP + 手动 whl
- mmcv 2.1.0 / mmaction2：源码编译（Phase 1.3 可选，当前跳过）

### 8.2 前端依赖（[package.json](file:///d:/Desktop/k9-training-system/frontend/package.json)）

**dependencies**：vue 3.5 / vue-router 4.5 / pinia 2.3 / naive-ui 2.40 / echarts 5.5 / vue-echarts 7 / axios 1.7 / @vicons/ionicons5
**devDependencies**：vite 6 / vue-tsc 2.2 / typescript 5.7 / @vitejs/plugin-vue / @types/node

### 8.3 模块间依赖（关键调用链）

```
前端 api/index.ts
  └→ backend/app/api/* (FastAPI 路由)
       ├→ backend/app/services/* (业务逻辑)
       │    ├→ report.py → reportlab (PDF)
       │    └→ label_studio.py → requests (LS HTTP)
       ├→ backend/app/models/* (SQLAlchemy ORM)
       │    └→ backend/app/core/database.py → asyncpg → PostgreSQL
       └→ backend/workers/tasks.py (Celery 派发)
            └→ Redis Broker → Celery Worker
                 ├→ backend/ml/pose/inference.py (PoseInferenceEngine)
                 │    └→ ultralytics YOLO → ONNX/TensorRT
                 ├→ backend/ml/pose/motionbert/ (3D lifting, Phase 3)
                 │    └→ external/MotionBERT/lib/ (DSTformer)
                 ├→ backend/ml/behavior/rule_engine.py (RuleEngine)
                 │    └→ backend/ml/behavior/constants.py
                 ├→ backend/ml/behavior/object_detector.py (ObjectDetector)
                 ├→ backend/ml/behavior/puppy_signals.py
                 ├→ backend/ml/scoring/engine.py (ScoringEngine)
                 │    ├→ backend/ml/scoring/schema.py (Pydantic)
                 │    ├→ backend/ml/scoring/conditions.py (条件求值)
                 │    └→ backend/ml/scoring/configs/*.yaml (评分卡)
                 ├→ backend/ml/tracking/multi_dog_tracker.py (Phase 3)
                 │    └→ boxmot (OccluBoost)
                 └→ backend/app/services/report.py (PDF 生成)
```

---

## 9. 端到端数据流

### 9.1 视频推理主流程

```
1. 用户上传视频
   前端 UploadView → POST /api/videos/upload
   → 校验扩展名/大小 → UUID 存储到 data/uploads/
   → 创建 videos 记录(status=UPLOADED)
   → ingest_video.delay(video_id) 异步派发

2. Celery 推理任务（ingest_video）
   → 标记 status=PROCESSING
   → PoseInferenceEngine.infer_video()
       └→ YOLO26-pose 流式推理 → 逐帧 (24,3) 关键点
   → 批量写入 keypoints 表
   → 场景分支:
       obedience_trial:
         RuleEngine.recognize() → 16 行为 episodes
         → _signals_from_obedience_episodes() → 6 信号
         → ScoringEngine.get(obedience_trial.yaml).evaluate()
       puppy_selection:
         ObjectDetector.detect_video() → 物体检测
         → extract_puppy_signals() → 9 信号
         → ScoringEngine.get(puppy_selection.yaml).evaluate()
       uspca_patrol:
         RuleEngine.recognize() → 16 行为
         → _signals_from_uspca_episodes() → 9 信号
         → ScoringEngine.get(uspca_patrol.yaml).evaluate()
   → 写入 behaviors 表
   → generate_report() → reports/{video_id}.pdf
   → 标记 status=COMPLETED + report_path

3. 前端轮询结果
   GET /api/videos/{id}/status (1-2s 间隔)
   → status=completed 后 GET /api/videos/{id}/report 下载 PDF
```

### 9.2 数据飞轮闭环（Phase 2.1）

```
上传视频 → 创建标注任务(POST /api/annotations/tasks)
   → Label Studio 预标注（YOLO26-pose 自动标注）
   → 人工修正标注
   → 同步标注(POST /api/annotations/sync)
   → 触发微调(POST /api/finetune/trigger)
   → 激活新模型(POST /api/models/{id}/activate)
   → 回到上传（形成闭环）
```

### 9.3 评分卡管理流程

```
管理员 AdminView → GET /api/scoring/configs (列出)
   → PUT /api/scoring/configs/{scene} (修改 YAML)
       → Pydantic Schema 校验 → 写入文件
       → ScoringEngine.get() 下次调用 mtime 检测自动重载
   → POST /api/scoring/evaluate (实时预览评分)
```

---

## 10. 项目运行方式

### 10.1 环境准备

**硬件**：NVIDIA RTX 30/40/50 系列 8GB+ VRAM（Blackwell 架构需 CUDA 12.8）

**软件预装**（见 [docs/deployment.md](file:///d:/Desktop/k9-training-system/docs/deployment.md)）：
- Python 3.12（项目 venv）
- Node.js 20 LTS
- PostgreSQL 16+（端口 5433）
- Redis 7+（端口 6379）
- CUDA Toolkit 12.8+（Blackwell GPU）

### 10.2 后端启动

```powershell
# 1. 创建并激活 venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装 PyTorch（Blackwell 必需 cu128）
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# 3. 安装依赖
pip install -r backend/requirements.txt

# 4. 配置环境变量
cp backend/.env.example .env
# 编辑 .env 填入真实数据库密码

# 5. 数据库迁移
alembic upgrade head

# 6. 启动 FastAPI（开发模式）
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
# 访问 http://127.0.0.1:8000/docs 查看 Swagger
```

### 10.3 Celery Worker 启动

```powershell
# Windows 必须用 --pool solo（不支持 prefork）
celery -A backend.workers.celery_app worker -l info --pool solo

# 可选：Flower 监控
celery -A backend.workers.celery_app flower
```

### 10.4 前端启动

```powershell
cd frontend
npm install
npm run dev
# 访问 http://localhost:5173
```

### 10.5 单独运行 ML 推理（CLI）

```powershell
# YOLO26-pose 视频推理
python -m backend.ml.pose.inference --video data/sample.mp4 --model best.pt

# 端到端推理（通过 Celery 任务）
python -c "from backend.workers.tasks import ingest_video; ingest_video.apply(args=[video_id])"
```

### 10.6 关键配置项（`.env`）

```env
DATABASE_URL=postgresql+asyncpg://k9system:<PWD>@127.0.0.1:5433/k9system
PG_DSN=postgresql://k9system:<PWD>@127.0.0.1:5433/k9system
REDIS_URL=redis://127.0.0.1:6379/0
CELERY_BROKER_URL=redis://127.0.0.1:6379/1
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/2
APP_ENV=development
APP_PORT=8000
LS_URL=http://127.0.0.1:8080   # Label Studio（数据飞轮）
```

> 完整模板见 [backend/.env.example](file:///d:/Desktop/k9-training-system/backend/.env.example) 与 [dev-docs/runtime.md](file:///d:/Desktop/k9-training-system/dev-docs/runtime.md)

---

## 11. 测试体系

测试位于 [backend/tests/](file:///d:/Desktop/k9-training-system/backend/tests/)，遵循 [AGENTS.md](file:///d:/Desktop/k9-training-system/AGENTS.md) §4.3 约定。

### 11.1 测试结构

```
backend/tests/
├── conftest.py                          # pytest fixtures
├── integration/
│   └── test_e2e_pipeline.py             # 端到端集成测试（需 DB + GPU）
└── ml/
    ├── test_pose_inference.py           # YOLO26-pose 推理
    ├── test_pose_train.py               # 姿态模型训练
    ├── test_motionbert_24kp.py          # MotionBERT 24 关键点
    ├── test_3d_pose_pairing.py          # 3D 配对
    ├── test_rule_engine.py              # P0 规则引擎
    ├── test_rule_engine_p1.py           # P1 规则引擎
    ├── test_scoring_engine.py           # 评分引擎
    ├── test_object_detector.py          # 物体检测
    ├── test_puppy_signals.py            # 幼犬信号
    ├── test_stgcn_bc.py                 # ST-GCN+BC
    ├── test_multi_dog_tracking.py       # 多犬追踪
    ├── test_reid_id_switch.py           # ReID ID 切换
    └── test_ingest_video.py             # 视频入库
```

### 11.2 验证命令映射

| 验证目标 | 命令 | 频率 |
|---------|------|------|
| Python 单元测试 | `pytest backend/tests/` | 每次提交 |
| 前端单元测试 | `cd frontend && npm test` | 每次提交 |
| API 启动验证 | `uvicorn backend.app.main:app` | 开发期 |
| Celery 启动验证 | `celery -A backend.workers.celery_app worker -l info --pool solo` | 开发期 |
| 推理端到端 | `python -m backend.ml.pose.inference --video <path>` | 阶段验证 |
| E2E 集成测试 | `pytest backend/tests/integration/test_e2e_pipeline.py -v -m integration` | 阶段验证 |

> E2E 测试需真实 DB + GPU 模型，约 30-60s，默认不参与快速运行。

---

## 附：关键文档索引

| 文档 | 路径 | 用途 |
|------|------|------|
| Agent 宪法 | [AGENTS.md](file:///d:/Desktop/k9-training-system/AGENTS.md) | 项目硬约束 + Owner Map + 阶段映射 |
| 架构 Truth | [dev-docs/architecture.md](file:///d:/Desktop/k9-training-system/dev-docs/architecture.md) | 系统拓扑 + 模块划分 + ADR |
| 技术选型 | [dev-docs/technical-selection.md](file:///d:/Desktop/k9-training-system/dev-docs/technical-selection.md) | 技术栈决策 + 候选组合 |
| 运行时基线 | [dev-docs/runtime.md](file:///d:/Desktop/k9-training-system/dev-docs/runtime.md) | 硬件/软件版本基线 |
| 部署指南 | [docs/deployment.md](file:///d:/Desktop/k9-training-system/docs/deployment.md) | Windows 本地部署步骤 |
| 阶段计划 | [dev-docs/stages/](file:///d:/Desktop/k9-training-system/dev-docs/stages/) | Phase 0-3 详细计划 |
| 决策记录 | [dev-docs/decisions/](file:///d:/Desktop/k9-training-system/dev-docs/decisions/) | ADR 0003-0010 |
| 调研文档 | [dev-docs/research/](file:///d:/Desktop/k9-training-system/dev-docs/research/) | 技术调研（标准/开源/文献/对比） |
