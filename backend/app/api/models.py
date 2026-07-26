"""ML 模型管理路由（Phase 0 占位）。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.models.ml_model import MLModel

router = APIRouter(prefix="/models", tags=["models"])


DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.get("")
async def list_models(db: DbSession, limit: int = 50, offset: int = 0) -> list[dict]:
    """列出模型版本。"""
    result = await db.execute(
        select(MLModel).order_by(MLModel.id.desc()).limit(limit).offset(offset)
    )
    return [
        {
            "id": m.id,
            "name": m.name,
            "version": m.version,
            "type": m.type.value,
            "framework": m.framework.value,
            "is_active": m.is_active,
        }
        for m in result.scalars().all()
    ]


@router.get("/{model_id}", response_model=None)
async def get_model(model_id: int, db: DbSession) -> dict:
    """获取单个模型。"""
    model = await db.get(MLModel, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return {
        "id": model.id,
        "name": model.name,
        "version": model.version,
        "type": model.type.value,
        "framework": model.framework.value,
        "is_active": model.is_active,
        "storage_path": model.storage_path,
        "metrics": model.metrics_json,
        "description": model.description,
        "created_at": model.created_at.isoformat() if model.created_at else None,
    }
