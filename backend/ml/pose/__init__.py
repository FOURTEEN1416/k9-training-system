# K9 Training System - 狗姿态检测模块
# Owner: ML 开发（见 AGENTS.md §2.2）
# Phase: 1.0

"""姿态检测模块初始化。

包含 ultralytics check_font 兼容性 patch：
- 问题: ultralytics check_font 用大小写敏感子串匹配查找系统字体，
  Windows 系统 Arial.ttf 实际为 arial.ttf（小写），匹配失败触发下载。
- 沙箱: TRAE 沙箱禁止写入 C:\\Users\\<user>\\AppData\\Roaming\\Ultralytics\\，
  导致 safe_download(Arial.ttf) 抛 ConnectionError，训练中断。
- 修复: monkey-patch check_font，大小写不敏感匹配系统字体；
  若仍未找到，返回 None（训练可继续，仅标注图无字体）。
"""
from __future__ import annotations

from pathlib import Path


def _patch_ultralytics_check_font() -> None:
    """Patch ultralytics.utils.checks.check_font，兼容 Windows + 沙箱环境。"""
    try:
        import ultralytics.utils.checks as _ul_checks
        from ultralytics.utils import USER_CONFIG_DIR, ASSETS_URL
    except ImportError:
        return  # ultralytics 未安装，跳过

    if getattr(_ul_checks.check_font, "_k9_patched", False):
        return  # 已 patch

    def _k9_check_font(font: str = "Arial.ttf"):
        """K9 patch: 大小写不敏感匹配系统字体，失败时返回 None 不下载。"""
        from matplotlib import font_manager

        name = Path(font).name
        # 1. USER_CONFIG_DIR 已存在字体（之前可能下载成功）
        cfg_file = USER_CONFIG_DIR / name
        if cfg_file.exists():
            return cfg_file

        # 2. 系统字体（大小写不敏感匹配）
        name_lower = name.lower()
        for sys_font in font_manager.findSystemFonts():
            # sys_font 是完整路径，取 basename 比较
            if Path(sys_font).name.lower() == name_lower:
                return sys_font

        # 3. 不下载（避免沙箱权限错误），返回 None
        # ultralytics 内部对返回 None 有兜底（用 PIL 默认字体）
        return None

    _k9_check_font._k9_patched = True  # type: ignore[attr-defined]
    _ul_checks.check_font = _k9_check_font


# 模块导入时自动 patch
_patch_ultralytics_check_font()
