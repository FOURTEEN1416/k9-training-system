"""基地管理 API 路由（Phase 3.6b）。

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 3.6b

端点:
    POST   /api/bases          创建基地（仅 ADMIN）
    GET    /api/bases          列出基地（ADMIN 全部；其他仅本基地）
    GET    /api/bases/{id}     基地详情（ADMIN 任何；其他仅本基地）
    PUT    /api/bases/{id}     更新基地（仅 ADMIN）
    DELETE /api/bases/{id}     禁用基地（仅 ADMIN，软删除 is_active=False）
"""
from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.deps import get_current_handler, require_roles
from backend.app.models.base_entity import Base_
from backend.app.models.handler import Handler, UserRole
from backend.app.schemas.common import BaseCreate, BaseRead, BaseUpdate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bases", tags=["bases"])

DbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentHandler = Annotated[Handler, Depends(get_current_handler)]
AdminOnly = Annotated[Handler, Depends(require_roles(UserRole.ADMIN))]


@router.post("", response_model=BaseRead, status_code=status.HTTP_201_CREATED)
async def create_base(
    payload: BaseCreate,
    db: DbSession,
    me: AdminOnly,
) -> Base_:
    """创建基地（仅 ADMIN 或超管）。"""
    # code 唯一性校验
    existing = await db.execute(select(Base_).where(Base_.code == payload.code))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail=f"基地代号 {payload.code} 已存在")
    # name 唯一性校验
    existing_name = await db.execute(select(Base_).where(Base_.name == payload.name))
    if existing_name.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail=f"基地名称 {payload.name} 已存在")

    base = Base_(
        name=payload.name,
        code=payload.code,
        description=payload.description,
        is_active=payload.is_active,
    )
    db.add(base)
    await db.flush()
    await db.refresh(base)
    logger.info(f"用户 {me.id} 创建基地 {base.id} ({base.code})")
    return base


@router.get("", response_model=list[BaseRead])
async def list_bases(
    db: DbSession,
    me: CurrentHandler,
    is_active: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Base_]:
    """列出基地。

    - ADMIN / 超管：可查看所有基地
    - 其他角色：仅可查看本基地（base_id 为 NULL 时返回空列表）
    """
    stmt = select(Base_).order_by(Base_.id.asc()).limit(limit).offset(offset)

    # 非 ADMIN 仅本基地
    if not (me.is_superuser or me.role == UserRole.ADMIN):
        if me.base_id is None:
            return []
        stmt = stmt.where(Base_.id == me.base_id)

    if is_active is not None:
        stmt = stmt.where(Base_.is_active.is_(is_active))

    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{base_id}", response_model=BaseRead)
async def get_base(
    base_id: int,
    db: DbSession,
    me: CurrentHandler,
) -> Base_:
    """获取基地详情。

    - ADMIN / 超管：可查看任何基地
    - 其他角色：仅可查看本基地
    """
    base = await db.get(Base_, base_id)
    if base is None:
        raise HTTPException(status_code=404, detail=f"基地 {base_id} 不存在")

    # 非 ADMIN 仅本基地
    if not (me.is_superuser or me.role == UserRole.ADMIN):
        if me.base_id != base_id:
            raise HTTPException(status_code=403, detail="无权查看其他基地")

    return base


@router.put("/{base_id}", response_model=BaseRead)
async def update_base(
    base_id: int,
    payload: BaseUpdate,
    db: DbSession,
    me: AdminOnly,
) -> Base_:
    """更新基地信息（仅 ADMIN）。"""
    base = await db.get(Base_, base_id)
    if base is None:
        raise HTTPException(status_code=404, detail=f"基地 {base_id} 不存在")

    # code 变更时校验唯一性
    if payload.code is not None and payload.code != base.code:
        existing = await db.execute(select(Base_).where(Base_.code == payload.code))
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail=f"基地代号 {payload.code} 已存在")
        base.code = payload.code

    # name 变更时校验唯一性
    if payload.name is not None and payload.name != base.name:
        existing = await db.execute(select(Base_).where(Base_.name == payload.name))
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail=f"基地名称 {payload.name} 已存在")
        base.name = payload.name

    if payload.description is not None:
        base.description = payload.description
    if payload.is_active is not None:
        base.is_active = payload.is_active

    await db.flush()
    await db.refresh(base)
    logger.info(f"用户 {me.id} 更新基地 {base_id}")
    return base


@router.delete("/{base_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def deactivate_base(
    base_id: int,
    db: DbSession,
    me: AdminOnly,
) -> Response:
    """禁用基地（软删除，仅 ADMIN）。

    设 is_active=False 而非物理删除，保留外键完整性。
    """
    base = await db.get(Base_, base_id)
    if base is None:
        raise HTTPException(status_code=404, detail=f"基地 {base_id} 不存在")

    base.is_active = False
    await db.flush()
    logger.info(f"用户 {me.id} 禁用基地 {base_id}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
