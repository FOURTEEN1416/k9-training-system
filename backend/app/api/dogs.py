"""犬只管理路由。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.models.dog import Dog
from backend.app.schemas.common import DogCreate, DogRead, DogUpdate

router = APIRouter(prefix="/dogs", tags=["dogs"])


DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.get("", response_model=list[DogRead])
async def list_dogs(db: DbSession, limit: int = 50, offset: int = 0) -> list[Dog]:
    """列出犬只档案。"""
    result = await db.execute(
        select(Dog).order_by(Dog.id.desc()).limit(limit).offset(offset)
    )
    return list(result.scalars().all())


@router.post("", response_model=DogRead, status_code=status.HTTP_201_CREATED)
async def create_dog(payload: DogCreate, db: DbSession) -> Dog:
    """创建犬只档案。"""
    dog = Dog(**payload.model_dump())
    db.add(dog)
    await db.flush()
    await db.refresh(dog)
    return dog


@router.get("/{dog_id}", response_model=DogRead)
async def get_dog(dog_id: int, db: DbSession) -> Dog:
    """获取单个犬只档案。"""
    dog = await db.get(Dog, dog_id)
    if dog is None:
        raise HTTPException(status_code=404, detail=f"Dog {dog_id} not found")
    return dog


@router.put("/{dog_id}", response_model=DogRead)
async def update_dog(dog_id: int, payload: DogUpdate, db: DbSession) -> Dog:
    """更新犬只档案（partial update，仅更新非 None 字段）。"""
    dog = await db.get(Dog, dog_id)
    if dog is None:
        raise HTTPException(status_code=404, detail=f"Dog {dog_id} not found")

    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(dog, field, value)

    await db.flush()
    await db.refresh(dog)
    return dog


@router.delete("/{dog_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dog(dog_id: int, db: DbSession) -> None:
    """删除犬只档案。"""
    dog = await db.get(Dog, dog_id)
    if dog is None:
        raise HTTPException(status_code=404, detail=f"Dog {dog_id} not found")
    await db.delete(dog)

