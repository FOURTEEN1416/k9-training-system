"""犬只管理路由。

隔离语义（2026-10-20 对抗验收修复）:
    端点此前仅做角色校验、未做基地/归属隔离（渗透脚本 P2-3 实测：跨基地 MANAGER
    可 PUT/DELETE 他人基地犬档案，返回 200/204）。现统一改走 core/deps 既有的
    ``check_dog_access`` / ``require_dog_access`` 原语，与 RBAC 设计保持一致：
        ADMIN/超管 → 全局；MANAGER → 本基地（含临时基地授权）；RESEARCHER/VIEWER → 本基地只读；
        HANDLER → 自有犬只 + 关联表（主训导员可写，副训导员只读）。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.deps import (
    check_dog_access,
    get_current_handler,
    require_roles,
    visible_dog_id_select,
)
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
    """列出犬只档案（需 JWT；非 ADMIN 仅见本基地 / 本人关联犬只）。"""
    visibility = visible_dog_id_select(me)
    stmt = select(Dog).order_by(Dog.id.desc()).limit(limit).offset(offset)
    if visibility is not None:
        stmt = stmt.where(Dog.id.in_(visibility))
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.post("", response_model=DogRead, status_code=status.HTTP_201_CREATED)
async def create_dog(payload: DogCreate, db: DbSession, me: ManagerOrAdmin) -> Dog:
    """创建犬只档案（需 JWT；VIEWER 不可写）。

    非 ADMIN 创建的犬只归属创建者所在基地（否则会创建出无人可见的孤儿档案）。
    """
    dog = Dog(**payload.model_dump())
    if not (me.is_superuser or me.role == UserRole.ADMIN):
        dog.home_base_id = me.base_id
    db.add(dog)
    await db.flush()
    await db.refresh(dog)
    return dog


@router.get("/{dog_id}", response_model=DogRead)
async def get_dog(dog_id: int, db: DbSession, me: CurrentHandler) -> Dog:
    """获取单个犬只档案（需 JWT；非授权范围返回 404，防枚举）。"""
    return await check_dog_access(db, me, dog_id, require_write=False)


@router.put("/{dog_id}", response_model=DogRead)
async def update_dog(dog_id: int, payload: DogUpdate, db: DbSession, me: ManagerOrAdmin) -> Dog:
    """更新犬只档案（需 JWT；须对目标犬只有写权限）。"""
    dog = await check_dog_access(db, me, dog_id, require_write=True)

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
    """删除犬只档案（仅 ADMIN / MANAGER；须对目标犬只有写权限）。"""
    dog = await check_dog_access(db, me, dog_id, require_write=True)
    await db.delete(dog)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
