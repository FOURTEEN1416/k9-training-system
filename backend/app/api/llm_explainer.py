"""LLM行为解释器 API 端点 - Phase 4

Owner: 后端开发 (见 AGENTS.md §2.2)
Phase: 4.1
依据: ADR 0011 Phase 4 LLM行为解释器

提供 /api/llm/explain 端点，接受行为概率和评分，返回中文分析报告。
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...ml.behavior.llm_explainer import explain_behavior

router = APIRouter(prefix="/llm", tags=["LLM行为解释器"])


# 请求模型
class LLMExplainRequest(BaseModel):
    """LLM行为解释请求模型。"""
    video_id: str = "..."
    duration_sec: int = 90
    probabilities: list = [...]
    scores: list = [...]
    behaviors: list = None


# 响应模型
class LLMExplainResponse(BaseModel):
    """LLM行为解释响应模型。"""
    report: str
    generation_time: float
    model_version: str
    confidence: float


@router.post(
    "/explain",
    response_model=LLMExplainResponse,
    summary="生成行为分析报告",
    description="基于 ST-GCN+BC 22类行为概率和 FCI-IGP 7维评分生成中文行为分析报告",
)
async def explain(request: LLMExplainRequest):
    """生成行为分析报告。

    接受行为概率分布和FCI-IGP评分，调用外部 LLM API
    生成约200字的中文行为分析报告，包含行为概述、详细分析、评分解读、
    训练建议和风险提示等章节。

    返回的报告结构:
    1. 行为概述 - 简述最可能行为及重要性
    2. 详细行为分析 - Top 3行为特征分析
    3. 评分解读 - FCI-IGP评分解释
    4. 训练建议 - 每日/每周练习项目
    5. 风险提示 - 异常情况标注
    """
    try:
        result = explain_behavior(
            probabilities=request.probabilities,
            scores=request.scores,
            video_id=request.video_id,
            duration_sec=request.duration_sec,
            behaviors=request.behaviors,
        )
        return LLMExplainResponse(
            report=result["report"],
            generation_time=result["generation_time"],
            model_version=result["model_version"],
            confidence=result["confidence"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"生成行为分析报告时发生错误: {str(e)}",
        )
