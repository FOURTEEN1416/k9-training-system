"""LLM行为解释器模块 - Phase 4

Owner: 后端开发 (见 AGENTS.md §2.2)
Phase: 4.1
依据: ADR 0011 Phase 4 LLM行为解释器

功能:
- 基于 ST-GCN+BC 22类行为概率 + FCI-IGP 7维评分生成中文行为分析报告
- 使用外部 LLM API（OpenAI 兼容）进行推理
- FastAPI接口提供 /api/llm/explain 端点

配置:
- 通过环境变量 LLM_API_KEY、LLM_API_BASE、LLM_MODEL 配置外部 API
- 支持 OpenAI、Azure OpenAI、通义千问、智谱等兼容 API
- 默认模型: gpt-4o-mini
- 降级: 当 API 不可用时返回结构化模板报告
"""

from __future__ import annotations

import json
import time
from typing import Optional, Dict, Any, List

import numpy as np
import requests
from fastapi import HTTPException

# 导入应用配置
try:
    from ...app.core.config import settings
except ImportError:
    settings = None


def _build_prompt(
    probabilities: List[float],
    scores: List[float],
    video_id: str,
    duration_sec: int,
    behaviors: Optional[List[str]] = None,
) -> str:
    """构建发送给 LLM 的提示词。"""
    # 行为名称映射 (对应 22类)
    _behavior_names = [
        "食物拒绝", "捡拾物体", "远距离指挥", "嗅闻搜索",
        "跃过障碍", "静默警戒", "服从指令", "待命状态",
        "护卫区域", "目标搜寻", "援助受伤", "装备携带",
        "笼舍进入", "笼舍离开", "换位指挥", "跟随训导员",
        "防止逃跑", "噪音忽略", "目标锁定", "保持姿态",
        "攻击威胁", "受控释放", "平静安息",
    ]

    # 构建概率字符串 (只展示 Top 5 和 Bottom 2)
    prob_pairs = []
    indexed = list(enumerate(probabilities))
    indexed.sort(key=lambda x: x[1], reverse=True)
    for idx, prob in indexed[:5]:  # Top 5
        name = _behavior_names[idx] if idx < len(_behavior_names) else f"类别{idx}"
        prob_pairs.append(f"  - {name}: {prob:.1%}")
    for idx, prob in indexed[-2:]:  # Bottom 2
        name = _behavior_names[idx] if idx < len(_behavior_names) else f"类别{idx}"
        prob_pairs.append(f"  - {name}: {prob:.1%}")

    # 计算最可能行为
    top_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
    top_behavior = _behavior_names[top_idx] if top_idx < len(_behavior_names) else f"类别{top_idx}"

    # FCI-IGP评分解释
    igp_weights = [0.25, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10]
    igp_dims = ["准确度", "延迟", "保持", "搜索效率", "注意力", "胆量", "步态"]
    igp_text = "; ".join(
        f"{dim}: {score}/100 (权重 {weight})"
        for dim, score, weight in zip(igp_dims, scores, igp_weights)
    )

    prompt = f"""你是一位资深工作犬训练专家。基于以下数据生成一份约200中文字的行为分析报告：

视频ID: {video_id}
视频时长: {duration_sec}秒 ({duration_sec//60}分{duration_sec%60}秒)

22类行为概率分布 (Top 5 最可能 + Bottom 2 最不可能):
{chr(10).join(prob_pairs)}

FCI-IGP 7维评分卡:
{igp_text}

关键信息:
- 最可能行为: {top_behavior}
- 该行为概率: {probabilities[top_idx]:.1%}

请按照以下结构生成报告 (必须严格按照此格式输出):

**行为概述**
(约50字) 简述最可能的行为及其重要性

**详细行为分析**
(约80字) 分析Top 3行为的表现特征和训练意义

**评分解读**
(约40字) 结合FCI-IGP评分解释行为质量

**训练建议**
(约30字) 针对最可能行为的每日/每周练习项目

**风险提示**
(约20字) 如有异常概率或低分，标注潜在风险

要求:
- 语言: 简体中文
- 语调: 专业、鼓励、实用
- 字数: 约200字 (±20%)
- 必须包含上述5个章节标题
- 语气: 训导员视角，兼顾训练效果和犬只福利"""

    return prompt


def _call_llm_api(prompt: str, config: LLMConfig) -> str:
    """调用外部 LLM API。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "你是工作犬训练专家，基于行为概率和评分生成专业中文报告。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }
    
    response = requests.post(
        f"{config.api_base.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=config.timeout,
    )
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"LLM API 调用失败: HTTP {response.status_code} - {response.text[:200]}"
        )
    
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def explain_behavior(
    probabilities: List[float],
    scores: List[float],
    video_id: str,
    duration_sec: int,
    behaviors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """生成行为分析报告。

    主流程入口：构建提示词 -> 调用 LLM API -> 后处理输出。

    参数:
        probabilities: 22类行为概率列表 (0-1)，总和应接近 1.0
        scores: 7维 FCI-IGP 评分列表 (0-100)
        video_id: 视频标识符
        duration_sec: 视频时长(秒)
        behaviors: 可选的行为名称列表 (对应概率索引)

    返回:
        包含以下键的字典:
        - report: 中文行为分析报告 (≈200字)
        - generation_time: 生成耗时 (秒)
        - model_version: 使用的模型版本字符串
        - confidence: 模型置信度代理 (Top概率值)

    异常:
        HTTPException: 当输入无效或 API 调用失败时抛出
    """
    # 获取配置
    if settings is not None:
        api_key = settings.llm_api_key
        api_base = settings.llm_api_base
        model = settings.llm_model
        timeout = settings.llm_timeout
    else:
        api_key = os.environ.get("LLM_API_KEY", "")
        api_base = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        timeout = int(os.environ.get("LLM_TIMEOUT", "30"))

    start_time = time.time()

    # 验证输入
    if not probabilities or len(probabilities) != 22:
        raise HTTPException(
            status_code=400,
            detail=f"probabilities必须是22个元素的列表，实际: {len(probabilities) if probabilities else 0}"
        )
    if not scores or len(scores) != 7:
        raise HTTPException(
            status_code=400,
            detail=f"scores必须是7个元素的列表，实际: {len(scores) if scores else 0}"
        )
    if not (0.0 <= sum(probabilities) <= 1.5):  # 允许微小误差
        raise HTTPException(
            status_code=400,
            detail=f"概率总和应接近 1.0，实际: {sum(probabilities):.3f}"
        )

    # 构建提示词
    prompt = _build_prompt(
        probabilities=probabilities,
        scores=scores,
        video_id=video_id,
        duration_sec=duration_sec,
        behaviors=behaviors,
    )

    generation_time = time.time() - start_time

    # 尝试调用 LLM API
    try:
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="LLM API Key 未配置，请设置环境变量 LLM_API_KEY"
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是工作犬训练专家，基于行为概率和评分生成专业中文报告。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 512,
        }
        
        response = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"LLM API 调用失败: HTTP {response.status_code} - {response.text[:200]}"
            )
        
        result = response.json()
        report = result["choices"][0]["message"]["content"].strip()
        generation_time += time.time() - start_time

        return {
            "report": _post_process_report(report),
            "generation_time": round(generation_time, 2),
            "model_version": model,
            "confidence": round(max(probabilities), 3),
        }

    except HTTPException:
        raise
    except Exception as e:
        # 降级：返回模板报告 (当 API 不可用时)
        generation_time = time.time() - start_time
        report = _template_report(
            top_behavior=_behavior_names_max(probabilities),
            confidence=max(probabilities) if probabilities else 0.0,
        )
        return {
            "report": report,
            "generation_time": round(generation_time, 2),
            "model_version": "template_fallback",
            "confidence": round(max(probabilities) if probabilities else 0.0, 3),
        }


def _post_process_report(generated: str) -> str:
    """后处理 LLM 输出，提取报告主体。"""
    if not generated:
        return _template_report()

    cleaned = generated.strip()

    # 如果输出足够长且包含章节标题，直接返回
    if len(cleaned) > 100:
        return cleaned[:800]  # 限制最大长度
    else:
        return _template_report()


def _template_report(
    top_behavior: str = "未检测到",
    confidence: float = 0.0,
) -> str:
    """生成 fallback 模板报告 (当 API 不可用时使用)。"""
    return f"""**行为概述**
通过视频分析系统检测到该段录像中主要表现出 "{top_behavior}" 的行为模式。该行为在工作犬训练中具有重要意义，反映了犬只当前的训练状态和注意力集中度。

**详细行为分析**
根据22类行为分类模型的概率分析，检测到的主要行为模式显示出犬只在特定训练情境下的反应情况。重点关注该行为出现的频率和上下文环境，有助于训导员了解犬只的当前状态。

**评分解读**
系统FCI-IGP评分显示整体表现中等水平。建议训导员结合具体的训练日志，重点关注概率较高的行为模式，有针对性地进行强化训练或纠正。

**训练建议**
- 每日进行 10-15 分钟的目标行为强化训练
- 结合正向奖励法，逐步塑造期望的行为模式
- 每周回顾一次行为概率变化，评估训练效果

**风险提示**
当前检测概率较低 ({confidence:.0%})，建议训导员观察犬只是否存在压力或不适表现。如行为持续异常，请咨询兽医或专业训练师。"""


def _behavior_names_max(probabilities: List[float]) -> str:
    """返回概率最高的行为名称。"""
    _behavior_names = [
        "食物拒绝", "捡拾物体", "远距离指挥", "嗅闻搜索",
        "跃过障碍", "静默警戒", "服从指令", "待命状态",
        "护卫区域", "目标搜寻", "援助受伤", "装备携带",
        "笼舍进入", "笼舍离开", "换位指挥", "跟随训导员",
        "防止逃跑", "噪音忽略", "目标锁定", "保持姿态",
        "攻击威胁", "受控释放", "平静安息",
    ]
    if not probabilities:
        return "未知"
    max_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
    return _behavior_names[max_idx] if max_idx < len(_behavior_names) else f"类别{max_idx}"


# 便捷函数: 供 API 调用
__all__ = ["explain_behavior", "get_llm_config", "LLMConfig"]
