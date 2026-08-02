"""JWT 鉴权 + bcrypt 密码哈希工具。

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 3.6b

提供:
    - 密码哈希/验证（passlib CryptContext bcrypt）
    - JWT 创建/解析（PyJWT）
    - Token payload 数据类
    - 异常类型（AuthError 子类）

设计原则:
    - 所有敏感操作（密码验证 + JWT 签名）在此模块封装
    - 上层依赖项（deps.py）只做依赖注入，不直接处理密码/JWT
    - 异常统一为 AuthError 子类，便于 FastAPI exception_handler 转换为 401/403
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from passlib.context import CryptContext

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# 密码哈希
# ============================================================

# bcrypt cost factor 由 settings 控制（默认 12）
_pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.auth_bcrypt_rounds,
)


def hash_password(plain: str) -> str:
    """明文密码 → bcrypt 哈希。"""
    if not plain:
        raise ValueError("密码不能为空")
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: Optional[str]) -> bool:
    """验证明文密码与哈希是否匹配。

    Args:
        plain: 用户输入的明文密码
        hashed: DB 中存储的 bcrypt 哈希（NULL 视为未启用登录）

    Returns:
        True 密码匹配；False 密码不匹配或哈希为空。
    """
    if not hashed:
        # 历史数据无密码字段，禁止登录
        return False
    try:
        return _pwd_context.verify(plain, hashed)
    except Exception as e:
        logger.warning(f"密码验证异常: {e}", exc_info=False)
        return False


# ============================================================
# JWT
# ============================================================


@dataclass
class TokenPayload:
    """JWT payload 数据类。"""

    sub: int  # handler.id（subject）
    role: str  # UserRole.value
    base_id: Optional[int]  # 所属基地 ID（NULL = 全局）
    is_superuser: bool
    exp: datetime  # 过期时间
    iat: datetime  # 签发时间

    def to_dict(self) -> dict:
        return {
            "sub": self.sub,
            "role": self.role,
            "base_id": self.base_id,
            "is_superuser": self.is_superuser,
            "exp": self.exp,
            "iat": self.iat,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenPayload":
        return cls(
            sub=int(d["sub"]),
            role=str(d["role"]),
            base_id=d.get("base_id"),
            is_superuser=bool(d.get("is_superuser", False)),
            exp=datetime.fromtimestamp(d["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(d["iat"], tz=timezone.utc),
        )


def create_access_token(
    handler_id: int,
    role: str,
    base_id: Optional[int],
    is_superuser: bool = False,
    expires_minutes: Optional[int] = None,
) -> str:
    """签发 JWT access token。

    Args:
        handler_id: 用户 ID（写入 sub claim）
        role: 角色（UserRole.value）
        base_id: 所属基地 ID
        is_superuser: 是否超管
        expires_minutes: 过期分钟数（None 用 settings 默认值）

    Returns:
        JWT 字符串
    """
    minutes = expires_minutes if expires_minutes is not None else settings.auth_access_token_expire_minutes
    now = datetime.now(tz=timezone.utc)
    exp = now + timedelta(minutes=minutes)
    payload = TokenPayload(
        sub=handler_id,
        role=role,
        base_id=base_id,
        is_superuser=is_superuser,
        exp=exp,
        iat=now,
    )
    return jwt.encode(
        payload.to_dict(),
        settings.auth_secret_key,
        algorithm=settings.auth_algorithm,
    )


class AuthError(Exception):
    """鉴权异常基类。"""

    status_code: int = 401

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class InvalidTokenError(AuthError):
    """JWT 无效或已过期。"""

    def __init__(self, message: str = "无效或已过期的令牌"):
        super().__init__(message, status_code=401)


class InsufficientPermissionError(AuthError):
    """权限不足。"""

    def __init__(self, message: str = "权限不足"):
        super().__init__(message, status_code=403)


class AuthenticationFailedError(AuthError):
    """认证失败（邮箱/密码错误）。"""

    def __init__(self, message: str = "邮箱或密码错误"):
        super().__init__(message, status_code=401)


def decode_access_token(token: str) -> TokenPayload:
    """解析 + 验证 JWT access token。

    Args:
        token: JWT 字符串（不含 "Bearer " 前缀）

    Returns:
        TokenPayload

    Raises:
        InvalidTokenError: token 无效/过期/签名错误
    """
    if not token:
        raise InvalidTokenError("令牌为空")
    try:
        payload = jwt.decode(
            token,
            settings.auth_secret_key,
            algorithms=[settings.auth_algorithm],
        )
    except jwt.ExpiredSignatureError:
        raise InvalidTokenError("令牌已过期，请重新登录")
    except jwt.InvalidTokenError as e:
        raise InvalidTokenError(f"令牌解析失败: {e}")
    return TokenPayload.from_dict(payload)
