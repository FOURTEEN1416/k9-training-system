"""犬只管理路由。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.deps import get_current_handler, require_roles
from backend.app.models.dog import Dog
from backend.app.models.handler import Handler, UserRole
from backend.app.schemas.common import DogCreate, DogRead, DogUpdate

router = APIRouter(
    prefix="/dogs",
    tags=["dogs"],
    dependencies=[Depends(get_current_handler)],
)


DbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentHandler = Annotated[Handler, Depends(get_current_handler)]
ManagerOrAdmin = Annotated[Handler, Depends(require_roles(UserRole.ADMIN, UserRole.MANAGER, UserRole.HANDLER, UserRole.RESEARCHER))]


@router.get("", response_model=list[DogRead])
async def list_dogs(db: DbSession, me: CurrentHandler, limit: int = 50, offset: int = 0) -> list[Dog]:
    """列出犬只档案（需 JWT）。"""
    result = await db.execute(
        select(Dog).order_by(Dog.id.desc()).limit(limit).offset(offset)
    )
    return list(result.scalars().all())


@router.post("", response_model=DogRead, status_code=status.HTTP_201_CREATED)
async def create_dog(payload: DogCreate, db: DbSession, me: ManagerOrAdmin) -> Dog:
    """创建犬只档案（需 JWT；VIEWER 不可写）。"""
    dog = Dog(**payload.model_dump())
    db.add(dog)
    await db.flush()
    await db.refresh(dog)
    return dog


@router.get("/{dog_id}", response_model=DogRead)
async def get_dog(dog_id: int, db: DbSession, me: CurrentHandler) -> Dog:
    """获取单个犬只档案（需 JWT）。"""
    dog = await db.get(Dog, dog_id)
    if dog is None:
        raise HTTPException(status_code=404, detail=f"Dog {dog_id} not found")
    return dog


@router.put("/{dog_id}", response_model=DogRead)
async def update_dog(dog_id: int, payload: DogUpdate, db: DbSession, me: ManagerOrAdmin) -> Dog:
    """更新犬只档案（需 JWT；VIEWER 不可写）。"""
    dog = await db.get(Dog, dog_id)
    if dog is None:
        raise HTTPException(status_code=404, detail=f"Dog {dog_id} not found")

    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(dog, field, value)

    await db.flush()
    await db.refresh(dog)
    return dog


@router.delete("/{dog_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_dog(
    dog_id: int,
    db: DbSession,
    me: Annotated[Handler, Depends(require_roles(UserRole.ADMIN, UserRole.MANAGER))],
) -> Response:
    """删除犬只档案（仅 ADMIN / MANAGER）。"""
    dog = await db.get(Dog, dog_id)
    if dog is None:
        raise HTTPException(status_code=404, detail=f"Dog {dog_id} not found")
    await db.delete(dog)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
