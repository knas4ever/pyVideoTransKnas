"""Localization helpers dedicated to the Gradio WebUI.

Desktop/Qt translations live in videotrans/language. Keeping the WebUI catalog
separate prevents a partial WebUI locale from being advertised as a desktop
locale.
"""

import json
import os
from functools import cache
from pathlib import Path



def normalize_locale(value):
    """Normalize common locale aliases used by the WebUI."""
    return {
        "fr": "fr_FR",
        "fr-FR": "fr_FR",
        "zh": "zh_CN",
        "zh-CN": "zh_CN",
        "en": "en_US",
        "en-US": "en_US",
    }.get(value, value)


@cache
def _load_catalog(locale):
    from videotrans.configure.config import ROOT_DIR
    """Load only WebUI-owned translations for locale."""
    locale = normalize_locale(locale)
    path = Path(f"{ROOT_DIR}/videotrans/webui_languages/{locale}.json")
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _startup_locale():
    """Honor PYVIDEOTRANS_LANG, then the saved UI language, then Chinese."""
    env_lang = os.environ.get("PYVIDEOTRANS_LANG")
    if env_lang:
        return normalize_locale(env_lang)
    try:
        cfg_path = Path(__file__).resolve().parents[1] / "videotrans" / "cfg.json"
        if cfg_path.is_file():
            saved = json.loads(cfg_path.read_text(encoding="utf-8")).get("lang") or ""
            if saved:
                return normalize_locale(saved)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return "zh_CN"


WEBUI_LOCALE = _startup_locale()
_CATALOG = _load_catalog(WEBUI_LOCALE)


def tr(key, *args):
    """Translate a WebUI key, falling back to the key itself."""
    value = _CATALOG.get(key, key)
    return value.format(*args) if args else value
