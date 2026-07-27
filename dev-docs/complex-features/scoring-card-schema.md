# 评分卡 YAML Schema 设计文档

> Owner: Phase 1.4a 契约（见 `dev-docs/stages/phase-1.md` §1.4a）
> 依据: [RESEARCH_SCORING_RULE_ENGINE.md](../research/RESEARCH_SCORING_RULE_ENGINE.md)
> 状态: ✅ v1.0
> 日期: 2026-07-27

## 1. 目标

定义工作犬评分卡的 YAML Schema 契约，支持：
1. **双场景隔离**：幼犬选育（3 维）+ 科目测评（5 维）
2. **YAML 配置化**：训导员可改阈值/权重，不改代码
3. **热加载**：修改 YAML 后下次评分自动生效，无需重启
4. **可解释**：每条评分给出命中规则 + 标签
5. **版本控制**：YAML 进 Git，规则变更可追溯

## 2. 顶层 Schema

```yaml
scoring_engine:
  name: string           # 评分卡名称（人类可读）
  version: string        # 语义化版本（如 "1.0.0"）
  scene: string          # 场景标识（puppy_selection / obedience_trial）
  description: string    # 可选，评分卡说明
  dimensions:            # 评分维度列表（1-N 个）
    - id: string         # 维度标识（snake_case）
      name: string       # 维度名称（中文可读）
      weight: float      # 权重（0-1，所有维度权重之和应为 1.0）
      rules:             # 规则列表（按顺序匹配，首个命中生效）
        - id: string     # 规则标识
          condition: string  # 条件表达式（见 §3）
          score: int     # 命中分数（0-100）
          label: string  # 命中标签（高/中/低 等）
      default:           # 可选，所有规则未命中时的默认分
        score: int
        label: string
  thresholds:            # 总分阈值（可选，用于合格/淘汰判定）
    pass: int            # ≥ 此分为合格
    borderline: int      # ≥ 此分为基本合格
    fail: int            # < borderline 为淘汰
  aggregation: string    # 聚合方式：weighted_sum（默认）/ max / min
```

## 3. 条件表达式语法

### 3.1 支持的语法

条件表达式使用 Python 表达式子集，**只允许访问 signals 字典 + 基础运算**：

| 语法 | 示例 | 说明 |
|------|------|------|
| 变量引用 | `approach_latency` | 引用 signals 中的键 |
| 比较 | `approach_latency < 1.0` | `<`, `<=`, `>`, `>=`, `==`, `!=` |
| 逻辑 | `cond1 and cond2` | `and`, `or`, `not` |
| 算术 | `approach_speed * 3.6` | `+`, `-`, `*`, `/` |
| 括号 | `(a or b) and c` | 分组 |
| 常量 | `True`, `0.5`, `"high"` | 字面量 |
| 兜底 | `True` | 默认规则（永远命中） |

### 3.2 安全约束

- **禁止**：`import`、`exec`、`eval`、`__`、属性访问、函数调用
- **实现**：用 `eval()` + 受限全局命名空间（仅 `signals` + 基础运算符）
- **变量缺失**：表达式中引用的信号若不存在，条件视为 `False`（不命中）

### 3.3 示例

```yaml
- id: food_high
  condition: "approach_latency < 1.0 and approach_speed > 2.0"
  score: 90
  label: "高"
- id: food_default
  condition: "True"
  score: 40
  label: "中低"
```

## 4. 双场景评分卡定义

### 4.1 幼犬选育评分卡（3 维）

| 维度 ID | 名称 | 权重 | 输入信号 |
|---------|------|------|---------|
| `food_drive` | 食物欲望 | 0.40 | approach_latency, approach_speed, sniff_duration |
| `prey_drive` | 猎物欲望 | 0.40 | chase_latency, chase_speed, hold_duration |
| `courage` | 胆量 | 0.20 | retreat_distance, freeze_duration, recovery_time |

**信号来源**：`backend/ml/behavior/puppy_signals.py`（Phase 1.6 实现）

### 4.2 科目测评评分卡（5 维）

| 维度 ID | 名称 | 权重 | 输入信号 |
|---------|------|------|---------|
| `accuracy` | 准确度 | 0.30 | action_correct, action_count |
| `latency` | 延迟 | 0.20 | command_to_action_latency |
| `duration` | 保持 | 0.20 | action_duration |
| `attention` | 注意力 | 0.15 | focus_ratio |
| `gait` | 步态 | 0.15 | gait_score |

**信号来源**：`backend/ml/behavior/rule_engine.py`（Phase 1.2 已实现，输出 BehaviorEpisode 列表）

## 5. 评分引擎 API 契约

### 5.1 Python API

```python
from backend.ml.scoring import ScoringEngine, ScoringContext

# 加载评分卡（YAML）
engine = ScoringEngine.from_yaml("backend/ml/scoring/configs/puppy_selection.yaml")

# 构建上下文
ctx = ScoringContext(
    signals={
        "approach_latency": 0.8,    # 秒
        "approach_speed": 2.5,      # m/s
        # ... 其他信号
    },
    scene="puppy_selection",
    meta={"video_id": 123, "dog_id": 456},  # 可选元数据
)

# 评分
result = engine.evaluate(ctx)

# 结果字段
result.total_score          # float, 0-100
result.dimension_scores     # dict[str, int], 各维度得分
result.dimension_labels     # dict[str, str], 各维度标签
result.hit_rules            # dict[str, str], 各维度命中的规则 ID
result.passed               # bool, total_score >= thresholds.pass
result.verdict              # str, "pass" / "borderline" / "fail"
result.explanation          # list[str], 人类可读的评分说明
```

### 5.2 热加载机制

```python
# 单例 + mtime 检测：YAML 修改后下次 get() 自动重载
engine = ScoringEngine.get("backend/ml/scoring/configs/puppy_selection.yaml")
```

### 5.3 REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/scoring/configs` | 列出所有评分卡 |
| GET | `/api/scoring/configs/{scene}` | 获取某场景评分卡 YAML |
| PUT | `/api/scoring/configs/{scene}` | 更新评分卡（写 YAML，触发热加载） |
| POST | `/api/scoring/evaluate` | 评分（body: signals + scene） |

## 6. 文件布局

```
backend/ml/scoring/
├── __init__.py              # 导出 ScoringEngine, ScoringContext, ScoringResult
├── engine.py                # 评分引擎核心（~150 行）
├── schema.py                # Pydantic 数据模型（~80 行）
├── conditions.py            # 条件表达式安全解析（~100 行）
└── configs/
    ├── puppy_selection.yaml # 幼犬选育评分卡（3 维）
    └── obedience_trial.yaml # 科目测评评分卡（5 维）
```

## 7. 测试策略

- **单元测试**（`backend/tests/ml/test_scoring_engine.py`）：
  - 条件表达式解析（合法/非法语法）
  - 单维度评分（规则命中/默认/信号缺失）
  - 多维度加权聚合
  - 热加载（修改 YAML 后重载）
  - 双场景评分卡各 ≥ 1 个端到端测试
- **集成测试**：Phase 1.7 端到端冒烟测试覆盖

## 8. 与既有 truth 文档的差异

| 既有文档 | 原假设 | 本契约 | 需同步 |
|---------|--------|--------|--------|
| project-brief.md §7 维评分 | 7 维（准确度/延迟/保持/搜索效率/注意力/胆量/步态） | 选育 3 维 + 科目 5 维，分离 | ✅ 已在立项重写中同步 |
| architecture.md 评分模块 | 评分逻辑硬编码 | YAML 配置化 + 热加载 | ✅ 已在立项重写中同步 |

## 9. 学术副产物潜力

- **方向 A**：可解释 AI 评分卡 + 工作犬行为识别
- **贡献**：YAML 配置化评分引擎开源 + 工作犬行为信号 → 评分规则的标准化框架
- **目标会议**：ACM MM Application Track / AAAI Application Track

## 10. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-27 | 初始版本，基于 RESEARCH_SCORING_RULE_ENGINE.md 固化契约 |
