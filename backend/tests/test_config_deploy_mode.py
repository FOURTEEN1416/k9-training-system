"""P0-4 行为部署模式默认值的配置测试。

Owner: 后端开发
Phase: P0-4（production-readiness-gap-plan.md P0-4：
      「behavior_deploy_mode 默认值定为 rule_only（生产未达标前）」）

语义:
    - 未显式配置 → production 默认 rule_only；development/test 默认 shadow
    - 显式配置在任何环境都被尊重
"""

from __future__ import annotations

import secrets

from backend.app.core.config import Settings

_STRONG_SECRET = secrets.token_urlsafe(32)
_REALISH_DSN = "postgresql+asyncpg://k9system:RotatedPass123!@127.0.0.1:5433/k9system"
_REALISH_PG = "postgresql://k9system:RotatedPass123!@127.0.0.1:5433/k9system"


def _settings(**overrides) -> Settings:
    kwargs = dict(
        auth_secret_key=_STRONG_SECRET,
        database_url=_REALISH_DSN,
        pg_dsn=_REALISH_PG,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


class TestDeployModeDefault:
    def test_production_defaults_rule_only(self):
        s = _settings(app_env="production")
        assert s.behavior_deploy_mode == "rule_only"

    def test_development_defaults_shadow(self):
        s = _settings(app_env="development")
        assert s.behavior_deploy_mode == "shadow"

    def test_test_env_defaults_shadow(self):
        s = _settings(app_env="test")
        assert s.behavior_deploy_mode == "shadow"

    def test_explicit_value_wins_in_production(self):
        s = _settings(app_env="production", behavior_deploy_mode="shadow")
        assert s.behavior_deploy_mode == "shadow"

    def test_explicit_value_wins_in_development(self):
        s = _settings(app_env="development", behavior_deploy_mode="rule_only")
        assert s.behavior_deploy_mode == "rule_only"
