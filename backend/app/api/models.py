"""ML 模型管理路由。"""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.models.ml_model import MLModel, ModelFramework, ModelType
from backend.app.schemas.common import ModelCreate, ModelRead

router = APIRouter(prefix="/models", tags=["models"])


DbSession = Annotated[AsyncSession, Depends(get_db)]

_VALID_TYPES = {"pose", "behavior", "scoring"}
_VALID_FRAMEWORKS = {"pytorch", "onnx", "tensorrt"}


@router.get("", response_model=list[ModelRead])
async def list_models(
    db: DbSession,
    type: Optional[str] = Query(default=None, description="按类型筛选: pose/behavior/scoring"),
    limit: int = 50,
    offset: int = 0,
) -> list[MLModel]:
    """列出模型版本（可按类型筛选）。"""
    stmt = select(MLModel).order_by(MLModel.id.desc()).limit(limit).offset(offset)
    if type is not None:
        if type not in _VALID_TYPES:
            raise HTTPException(status_code=400, detail=f"非法 type: {type}")
        stmt = stmt.where(MLModel.type == ModelType(type))
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/current", response_model=ModelRead)
async def get_current_model(
    db: DbSession,
    type: str = Query(..., description="模型类型: pose/behavior/scoring"),
) -> MLModel:
    """查询当前激活的模型（按类型）。"""
    if type not in _VALID_TYPES:
        raise HTTPException(status_code=400, detail=f"非法 type: {type}")
    stmt = (
        select(MLModel)
        .where(MLModel.type == ModelType(type), MLModel.is_active.is_(True))
        .limit(1)
    )
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"无激活的 {type} 模型，请先注册并激活",
        )
    return model


@router.post("/register", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
async def register_model(payload: ModelCreate, db: DbSession) -> MLModel:
    """注册新模型版本。

    若 is_active=True，自动将同类型的其他模型置为 is_active=False。
    """
    if payload.type not in _VALID_TYPES:
        raise HTTPException(status_code=400, detail=f"非法 type: {payload.type}")
    if payload.framework not in _VALID_FRAMEWORKS:
        raise HTTPException(status_code=400, detail=f"非法 framework: {payload.framework}")

    model = MLModel(
        name=payload.name,
        version=payload.version,
        type=ModelType(payload.type),
        framework=ModelFramework(payload.framework),
        storage_path=payload.storage_path,
        is_active=payload.is_active,
        metrics_json=payload.metrics,
        description=payload.description,
    )
    db.add(model)

    # 若激活，取消同类型其他模型的激活
    if payload.is_active:
        await db.execute(
            update(MLModel)
            .where(
                MLModel.type == ModelType(payload.type),
                MLModel.is_active.is_(True),
            )
            .values(is_active=False)
        )

    await db.flush()
    await db.refresh(model)
    return model


@router.get("/{model_id}", response_model=ModelRead)
async def get_model(model_id: int, db: DbSession) -> MLModel:
    """获取单个模型。"""
    model = await db.get(MLModel, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return model


@router.post("/{model_id}/activate", response_model=ModelRead)
async def activate_model(model_id: int, db: DbSession) -> MLModel:
    """激活指定模型版本（取消同类型其他激活）。"""
    model = await db.get(MLModel, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # 取消同类型其他激活
    await db.execute(
        update(MLModel)
        .where(
            MLModel.type == model.type,
            MLModel.is_active.is_(True),
            MLModel.id != model_id,
        )
        .values(is_active=False)
    )
    model.is_active = True
    await db.flush()
    await db.refresh(model)
    return model
