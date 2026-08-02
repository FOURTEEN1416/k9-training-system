"""评分卡管理 + 评分 API.

Owner: 后端开发 + ML 开发
Phase: 1.4f

路由:
    GET    /api/scoring/configs              列出所有评分卡
    GET    /api/scoring/configs/{scene}      获取评分卡 YAML 原文
    PUT    /api/scoring/configs/{scene}      更新评分卡 YAML（触发热加载）
    POST   /api/scoring/evaluate             对信号字典评分
"""

from pathlib import Path
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import settings
from backend.app.schemas.common import (
    ScoringConfigRead,
    ScoringConfigUpdate,
    ScoringEvaluateRequest,
    ScoringEvaluateResponse,
)
from backend.ml.scoring import ScoringContext, ScoringEngine
from backend.ml.scoring.conditions import ConditionError
from backend.ml.scoring.schema import ScoringCardSpec

router = APIRouter(prefix="/scoring", tags=["scoring"])


# 评分卡 YAML 路径
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCORING_CONFIGS_DIR = _PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs"

# 场景 → YAML 文件名映射
# Phase 1: puppy_selection / obedience_trial
# Phase 2.2e: working_dog_trial（16 行为 7 维）
# Phase 2.3b: uspca_patrol（USPCA PDI 5 维）
# Phase 3.4: fci_igp（FCI-IGP 国际工作犬 7 维 + DQ）
_SCENE_TO_FILE = {
    "puppy_selection": "puppy_selection.yaml",
    "obedience_trial": "obedience_trial.yaml",
    "working_dog_trial": "working_dog_trial.yaml",
    "uspca_patrol": "uspca_patrol.yaml",
    "fci_igp": "fci_igp.yaml",
}

_VALID_SCENES = set(_SCENE_TO_FILE.keys())


def _resolve_yaml_path(scene: str) -> Path:
    """根据 scene 返回 YAML 文件路径。"""
    if scene not in _SCENE_TO_FILE:
        raise HTTPException(
            status_code=400,
            detail=f"非法 scene: {scene}，允许: {sorted(_VALID_SCENES)}",
        )
    return _SCORING_CONFIGS_DIR / _SCENE_TO_FILE[scene]


# ============================================================
# 评分卡 CRUD
# ============================================================


@router.get("/configs", response_model=list[ScoringConfigRead])
async def list_scoring_configs() -> list[ScoringConfigRead]:
    """列出所有评分卡。"""
    result = []
    for scene, filename in _SCENE_TO_FILE.items():
        yaml_path = _SCORING_CONFIGS_DIR / filename
        content = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
        result.append(ScoringConfigRead(scene=scene, content=content))
    return result


@router.get("/configs/{scene}", response_model=ScoringConfigRead)
async def get_scoring_config(scene: str) -> ScoringConfigRead:
    """获取评分卡 YAML 原文。"""
    yaml_path = _resolve_yaml_path(scene)
    if not yaml_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"评分卡文件不存在: {yaml_path.name}",
        )
    content = yaml_path.read_text(encoding="utf-8")
    return ScoringConfigRead(scene=scene, content=content)


@router.put("/configs/{scene}", response_model=ScoringConfigRead)
async def update_scoring_config(scene: str, payload: ScoringConfigUpdate) -> ScoringConfigRead:
    """更新评分卡 YAML（触发热加载）。

    流程:
        1. 解析 YAML 校验语法
        2. 用 Pydantic 校验 Schema 合法性
        3. 写入文件（ScoringEngine 单例的 mtime 检测会自动重载）
    """
    yaml_path = _resolve_yaml_path(scene)

    # 1. YAML 语法校验
    try:
        parsed = yaml.safe_load(payload.content)
    except yaml.YAMLError as e:
        raise HTTPException(
            status_code=422,
            detail=f"YAML 语法错误: {e}",
        )

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=422,
            detail="YAML 顶层必须是字典",
        )

    # 2. Schema 校验（确保评分卡可用）
    # 评分卡 YAML 顶层结构为 {scoring_engine: {name, version, scene, dimensions, ...}}
    # 与 engine.py from_yaml() 保持一致:从 scoring_engine key 下取值再校验
    spec_data = parsed.get("scoring_engine", parsed) if isinstance(parsed, dict) else parsed
    try:
        ScoringCardSpec.model_validate(spec_data)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"评分卡 Schema 校验失败: {e}",
        )

    # 3. 写入文件
    yaml_path.write_text(payload.content, encoding="utf-8")

    return ScoringConfigRead(scene=scene, content=payload.content)


# ============================================================
# 评分
# ============================================================


@router.post("/evaluate", response_model=ScoringEvaluateResponse)
async def evaluate(payload: ScoringEvaluateRequest) -> ScoringEvaluateResponse:
    """对信号字典评分。

    用途: 前端实时预览评分结果（修改 signals → 看分数变化）
    """
    if payload.scene not in _VALID_SCENES:
        raise HTTPException(
            status_code=400,
            detail=f"非法 scene: {payload.scene}，允许: {sorted(_VALID_SCENES)}",
        )

    yaml_path = _resolve_yaml_path(payload.scene)
    try:
        engine = ScoringEngine.get(yaml_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"评分引擎加载失败: {e}",
        )

    ctx = ScoringContext(signals=payload.signals, scene=payload.scene)
    try:
        result = engine.evaluate(ctx)
    except ConditionError as e:
        raise HTTPException(
            status_code=422,
            detail=f"条件表达式错误: {e}",
        )

    return ScoringEvaluateResponse(
        total_score=result.total_score,
        verdict=result.verdict,
        passed=result.passed,
        dimension_labels=result.dimension_labels,
        dimension_scores=result.dimension_scores,
        explanation=result.explanation,
        scene=result.scene,
        card_name=result.card_name,
        card_version=result.card_version,
    )
