"""
pyVideoTrans WebUI — Gradio-based web interface for video translation.

TODO:
    拆分 webui.py 各个部分到 videotrans/webui_codes/ 中

Usage:
    uv run webui.py
    # or
    uv run python webui.py

Requires: uv sync --extra webui
"""

import sys
import json
import queue
import re
import threading
import time
import asyncio
import traceback
from pathlib import Path
from typing import List

from videotrans.configure._default_params import channel_default_params
from videotrans.ui.advset_keys import notices, ComboBox_List, ComboBox_Data

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---------------------------------------------------------------------------
# WebUI localization
# ---------------------------------------------------------------------------
# WebUI translations are intentionally isolated from the desktop/Qt catalogs.
from videotrans.webui_i18n import tr  # noqa: E402

# ---------------------------------------------------------------------------
# 初始化 videotrans 环境
# ---------------------------------------------------------------------------
from videotrans.configure import config, constants  # noqa: E402

# A download callback reports "file.pt:88.00%" about once a second. Keep that
# as one Execution Log line so a multi-gigabyte model does not push the
# percent out of the text box.
_DOWNLOAD_PROGRESS = re.compile(r"^(?P<head>.*):(?:\d+\.\d+%|\d+\.\d+MB)$")


def _append_exec_log(lines, msg, clock):
    text = f"[{clock}] {msg}"
    match = _DOWNLOAD_PROGRESS.match(str(msg).strip())
    if match and lines:
        previous = lines[-1].split("] ", 1)[-1]
        prev = _DOWNLOAD_PROGRESS.match(previous.strip())
        if prev and prev.group("head") == match.group("head"):
            lines[-1] = text
            return "\n".join(lines)
    lines.append(text)
    return "\n".join(lines)

config.init_run()

from videotrans.configure.config import ROOT_DIR, TEMP_DIR, app_cfg, params, settings
from videotrans.configure.constants import FASTER_MODELS_DICT, DEEPGRAM_ASR_MODELS, OPENAI_WHISPER_MODELS, \
    FUN_ASR_MODELS, DEFAULT_STYLE
from videotrans import recognition, translator, tts
from videotrans.util import tools
from videotrans.util.gpus import getset_gpu
from videotrans.util.help_ffmpeg import runffmpeg
from videotrans.util._ffmpeg_audio import test_clip_args, test_clip_target
from videotrans.task._speaker_refs import speaker_cap_to_nums
from videotrans.util.help_role import role_menu

# ---------------------------------------------------------------------------
# params / settings 持久化路径
# ---------------------------------------------------------------------------
PARAMS_JSON = Path(ROOT_DIR) / "videotrans" / "params.json"
SETTINGS_JSON = Path(ROOT_DIR) / "videotrans" / "cfg.json"


def _load_params() -> dict:
    """从 params.json 加载"""
    try:
        if PARAMS_JSON.exists():
            return json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_params(data: dict):
    """Merge into params.json. A partial write used to replace the whole file."""
    current = _load_params()
    if not isinstance(current, dict):
        current = {}
    current.update(data)
    PARAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_JSON.write_text(
        json.dumps(current, ensure_ascii=False), encoding="utf-8"
    )
    params.getset_params(current)
    _user_params.clear()
    _user_params.update(current)


def _load_settings() -> dict:
    try:
        if SETTINGS_JSON.exists():
            return json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def apply_settings_update(current, update):
    """Merge one saved section into the existing cfg. A section used to replace the file."""
    base = dict(current) if isinstance(current, dict) else {}
    base.update(update)
    return base


def _save_settings(data: dict):
    current = apply_settings_update(_load_settings(), data)
    SETTINGS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_JSON.write_text(
        json.dumps(current, ensure_ascii=False), encoding="utf-8"
    )
    settings.parse_init(data)
    _user_settings.clear()
    _user_settings.update(current)


def setting_widget_value(key, current):
    """Value a advanced-settings widget should show for one cfg entry."""
    raw = current.get(key, "")
    if isinstance(raw, bool):
        text = "true" if raw else "false"
    else:
        text = str(raw)
    if key in ComboBox_List:
        options = ComboBox_Data.get(key) or []
        if text in options:
            return text
        return options[0] if options else text
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


# 加载当前配置
_user_params = _load_params()
_user_settings = _load_settings()

# ---------------------------------------------------------------------------
# 渠道名称列表
# ---------------------------------------------------------------------------
RECOGN_NAMES: List[str] = recognition.RECOGN_NAME_LIST
TRANSLATE_NAMES: List[str] = translator.TRANSLASTE_NAME_LIST
TTS_NAMES: List[str] = tts.TTS_NAME_LIST
LANGNAME_DICT: dict = translator.LANGNAME_DICT

# ---------------------------------------------------------------------------
# 可选渠道索引
# ---------------------------------------------------------------------------
SELECTABLE_RECOGN = {0, 1, 2, 3, 4}
DEFAULT_RECOGN = recognition.OPENAI_WHISPER
SELECTABLE_TRANSLATE = {0, 1, 2, translator.HYMT2_INDEX}
DEFAULT_TRANSLATE = translator.HYMT2_INDEX
SELECTABLE_TTS = {0, 1, 3, 4, 5, 6, 7, 8, 31}
DEFAULT_TTS = tts.CHATTERBOX_TTS

FASTER_MODEL_NAMES = list(FASTER_MODELS_DICT.keys())
DEFAULT_MODEL = "large-v3"

LANG_DISPLAY_NAMES = list(LANGNAME_DICT.values())
LANG_CHOICES = [(name, code) for code, name in LANGNAME_DICT.items()]
DEFAULT_SOURCE_LANG = "en"
DEFAULT_TARGET_LANG = "da"

SUBTITLE_TYPES = {
    tr("No subtitles"): 0,
    tr("Embed hard subtitles"): 1,
    tr("Embed soft subtitles"): 2,
    tr("Embed hard subtitles (bilingual)"): 3,
    tr("Embed soft subtitles (bilingual)"): 4,
}
DEFAULT_SUBTITLE_TYPE = tr("Embed hard subtitles")
PUNC_OPTIONS = {tr("Default punctuation"): 0, tr("Restore punctuation"): 1, tr("Remove punctuation"): 2}
LOOP_BGM_OPTIONS = {tr("Truncate background audio"): 0, tr("Loop background audio"): 1}

# ---------------------------------------------------------------------------
# ASS 字幕样式
# ---------------------------------------------------------------------------
ASS_JSON_FILE = f"{ROOT_DIR}/videotrans/ass.json"


def _parse_ass_color(c):
    if not c.startswith("&H") or not c.endswith("&"):
        return "#ffffff"
    h = c[2:-1].upper()
    if len(h) == 6:
        return f"#{int(h[4:6], 16):02x}{int(h[2:4], 16):02x}{int(h[0:2], 16):02x}"
    elif len(h) == 8:
        return f"#{int(h[6:8], 16):02x}{int(h[4:6], 16):02x}{int(h[2:4], 16):02x}"
    return "#ffffff"


def _to_ass_color(h):
    h = h.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"&H00{b:02X}{g:02X}{r:02X}&"
    return "&H00FFFFFF&"


def _load_ass_style():
    try:
        if Path(ASS_JSON_FILE).exists():
            return json.loads(Path(ASS_JSON_FILE).read_text(encoding="utf-8"))
    except Exception:
        pass
    return DEFAULT_STYLE.copy()


def _save_ass_style(s):
    Path(ASS_JSON_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(ASS_JSON_FILE).write_text(
        json.dumps(s, indent=4, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _lang_code_from_display(d):
    for code, name in LANGNAME_DICT.items():
        if name == d:
            return code
    return d


def _tts_index_from_display(d):
    for i, name in enumerate(TTS_NAMES):
        if name == d:
            return i
    return 0


def _recogn_index_from_display(d):
    for i, name in enumerate(RECOGN_NAMES):
        if name == d:
            return i
    return 0


def _translate_index_from_display(d):
    for i, name in enumerate(TRANSLATE_NAMES):
        if name == d:
            return i
    return 0


def _format_rate(v):
    return f"+{v}%" if v >= 0 else f"{v}%"


def _format_pitch(v):
    return f"+{v}Hz" if v >= 0 else f"{v}Hz"


def _speaker_summary(cache_folder, target_dir):
    from collections import Counter

    for folder in (cache_folder, target_dir):
        path = Path(folder) / "speaker.json"
        if not path.is_file():
            continue
        try:
            ids = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(ids, list) or not ids:
            continue
        counts = Counter(str(item) for item in ids if item)
        if counts:
            text = ", ".join(f"{name} ({count})" for name, count in counts.items())
            return tr("Speakers found: {}", text)
    return tr("No speaker labels were produced. Each line is cloned from its own audio.")


def _safe_get(key, default=""):
    """从 _user_params 读取值，支持 str/int/float/bool"""
    v = _user_params.get(key, default)
    if v is None:
        return default
    return v


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _param_bool(key, default=False):
    if key not in _user_params:
        return default
    return _as_bool(_user_params.get(key), default)


def _param_int(key, default=0):
    try:
        return int(_user_params.get(key, default))
    except (TypeError, ValueError):
        return default


def _saved_index(key, names, default):
    idx = _param_int(key, default)
    if idx < 0 or idx >= len(names):
        return default if 0 <= default < len(names) else 0
    return idx


def _percent_int(key, default=0):
    raw = str(_user_params.get(key, default)).replace("%", "").replace("Hz", "").strip()
    try:
        return int(float(raw))
    except ValueError:
        return default


def _pick_role(roles, current):
    if not roles:
        return "No"
    if current in roles:
        return current
    if "clone" in roles:
        return "clone"
    return roles[0]


def _option_label(options, idx):
    keys = list(options)
    if 0 <= idx < len(keys):
        return keys[idx]
    return keys[0]


def _models_for_recogn(idx):
    """Model names for an ASR channel. disabled=True when the channel has no model list."""
    disabled = False
    if idx in [recognition.FASTER_WHISPER, recognition.Faster_Whisper_XXL, recognition.WHISPERX_API]:
        models = constants.WHISPER_MODELS.split(",")
    elif idx == recognition.OPENAI_WHISPER:
        models = OPENAI_WHISPER_MODELS.split(",")
    elif idx == recognition.Deepgram:
        models = DEEPGRAM_ASR_MODELS.split(",")
    elif idx == recognition.Whisper_CPP:
        models = constants.WHISPER_CPP_MODELS.split(",")
    elif idx == recognition.WHISPER_NET:
        models = constants.WHISPER_NET_MODELS.split(",")
    elif idx == recognition.QWENASR:
        models = ["1.7B", "0.6B"]
    elif idx == recognition.HUGGINGFACE_ASR:
        models = list(recognition.HUGGINGFACE_ASR_MODELS.keys())
    elif idx == recognition.FUNASR_CN:
        models = FUN_ASR_MODELS.split(",")
    else:
        models = list(FASTER_MODEL_NAMES)
        disabled = True
    models = [m.strip() for m in models if str(m).strip()]
    return models, disabled

"""
# ---------------------------------------------------------------------------
# 渠道设置面板定义
# 1. translator/recognition/tts 3个包中 _constants.py 中  ID_NAME_DICT 遍历，获取均有 win 属性的项，用win值 到  
#   videotrans.configure._default_params.py  channel_default_params(win.name) 换取该项所需默认值dict, 
# 2. 返回的 dict中 字符串、数字、布尔值(checkbox)、select(_model)结尾的值
# eg.
# channel_default_params("deepseek")
 return 
 {
            "deepseek_key": "",  # SK 密钥
            "deepseek_thinking": False,  # 默认不启用思考
            "deepseek_model": DEEPSEEK_MODELS.split(',')[0],
            "deepseek_max_token": 16384,  # 最大输出64k
        }
# ---------------------------------------------------------------------------
"""
CHANNEL_SETTINGS={}
_HAD_WIN=[]
from videotrans.translator import ID_NAME_DICT as trans_idname
from videotrans.recognition import ID_NAME_DICT as stt_idname
from videotrans.tts import ID_NAME_DICT as tts_idname


def _add_channel(it,title):

    pas=channel_default_params(it.win)
    if not pas:return
    _HAD_WIN.append(it.win)
    tmp={
        "category":title,
        "fields":[]
    }
    for k,v in pas.items():
        k_split=k.split('_',1)
        k_type="text"
        choices=[]
        if v in [True,False]:
            k_type="checkbox"
        elif k_split[-1].endswith('model'):
            k_type='select'
            choices=settings.get(k,'').split(',')
        elif k=='gptsovits_role':
            k_type="textarea"
        elif isinstance(v,int|float):
            k_type="number"

        tmp['fields'].append(
            {
                "key": k,
                "label":k_split[-1],
                "type": k_type,
                "default": v,
                "placeholder": k,
                "choices":choices
            }
        )
    CHANNEL_SETTINGS[it.name]=tmp

for it in trans_idname.values():
    if not it.win or it.win in _HAD_WIN:continue
    _add_channel(it,tr("Subtitle Translation Channels"))

# print(f'{CHANNEL_SETTINGS=}')
for it in stt_idname.values():
    if not it.win or it.win in _HAD_WIN:continue
    _add_channel(it,tr("ASR Channels"))

for it in tts_idname.values():
    if not it.win or it.win in _HAD_WIN:continue
    _add_channel(it,tr("TTS Channel"))

# ---------------------------------------------------------------------------
# ASS 样式编辑器（纯 Gradio）
# ---------------------------------------------------------------------------
def build_ass_editor():
    import gradio as gr

    style = _load_ass_style()

    with gr.Accordion(tr("Hard Subtitle Style Editor"), open=False):
        gr.Markdown(
            tr("Click 'Save Style' after modifying, style will apply to all tasks with embedded hard subtitles."))
        with gr.Tabs():
            with gr.Tab(tr("Primary Subtitle")):
                with gr.Row():
                    ass_fontname = gr.Textbox(
                        label=tr("Font Name"), value=style.get("Fontname", "Arial")
                    )
                    ass_fontsize = gr.Slider(
                        label=tr("Font Size"),
                        minimum=1,
                        maximum=200,
                        value=style.get("Fontsize", 16),
                        step=1,
                    )
                with gr.Row():
                    ass_primary_color = gr.ColorPicker(
                        label=tr("Primary Color"),
                        value=_parse_ass_color(
                            style.get("PrimaryColour", "&H00FFFFFF&")
                        ),
                    )
                    ass_outline_color = gr.ColorPicker(
                        label=tr("Outline Color"),
                        value=_parse_ass_color(
                            style.get("OutlineColour", "&H00000000&")
                        ),
                    )
                    ass_back_color = gr.ColorPicker(
                        label=tr("Background Color"),
                        value=_parse_ass_color(style.get("BackColour", "&H00000000&")),
                    )
                with gr.Row():
                    ass_bold = gr.Checkbox(
                        label=tr("Bold"), value=bool(style.get("Bold", 0))
                    )
                    ass_italic = gr.Checkbox(
                        label=tr("Italic"), value=bool(style.get("Italic", 0))
                    )
                    ass_underline = gr.Checkbox(
                        label=tr("Underline"), value=bool(style.get("Underline", 0))
                    )
                    ass_strikeout = gr.Checkbox(
                        label=tr("Strikeout"), value=bool(style.get("StrikeOut", 0))
                    )
            with gr.Tab(tr("Bottom Subtitle (Bilingual)")):
                with gr.Row():
                    ass_bottom_fontname = gr.Textbox(
                        label=tr("Font Name"),
                        value=style.get("Bottom_Fontname", "Arial"),
                    )
                    ass_bottom_fontsize = gr.Slider(
                        label=tr("Font Size"),
                        minimum=1,
                        maximum=200,
                        value=style.get("Bottom_Fontsize", 16),
                        step=1,
                    )
                with gr.Row():
                    ass_bottom_primary_color = gr.ColorPicker(
                        label=tr("Primary Color"),
                        value=_parse_ass_color(
                            style.get("Bottom_PrimaryColour", "&H00FFFFFF&")
                        ),
                    )
                    ass_bottom_outline_color = gr.ColorPicker(
                        label=tr("Outline Color"),
                        value=_parse_ass_color(
                            style.get("Bottom_OutlineColour", "&H00000000&")
                        ),
                    )
                    ass_bottom_back_color = gr.ColorPicker(
                        label=tr("Background Color"),
                        value=_parse_ass_color(
                            style.get("Bottom_BackColour", "&H00000000&")
                        ),
                    )
                with gr.Row():
                    ass_bottom_bold = gr.Checkbox(
                        label=tr("Bold"), value=bool(style.get("Bottom_Bold", 0))
                    )
                    ass_bottom_italic = gr.Checkbox(
                        label=tr("Italic"), value=bool(style.get("Bottom_Italic", 0))
                    )
            with gr.Tab(tr("Global Style")):
                with gr.Row():
                    ass_border_style = gr.Dropdown(
                        label=tr("Border Style"),
                        choices=[tr("Outline"), tr("Opaque Background")],
                        value=tr("Outline")
                        if style.get("BorderStyle", 1) == 1
                        else tr("Opaque Background"),
                    )
                    ass_outline = gr.Slider(
                        label=tr("Outline Width"),
                        minimum=0.0,
                        maximum=10.0,
                        value=style.get("Outline", 0.5),
                        step=0.1,
                    )
                    ass_shadow = gr.Slider(
                        label=tr("Shadow"),
                        minimum=0.0,
                        maximum=10.0,
                        value=style.get("Shadow", 0.5),
                        step=0.1,
                    )
                with gr.Row():
                    ass_scale_x = gr.Slider(
                        label=tr("Scale X %"),
                        minimum=1,
                        maximum=1000,
                        value=style.get("ScaleX", 100),
                        step=1,
                    )
                    ass_scale_y = gr.Slider(
                        label=tr("Scale Y %"),
                        minimum=1,
                        maximum=1000,
                        value=style.get("ScaleY", 100),
                        step=1,
                    )
                    ass_spacing = gr.Slider(
                        label=tr("Letter Spacing"),
                        minimum=-100,
                        maximum=100,
                        value=style.get("Spacing", 0),
                        step=1,
                    )
                    ass_angle = gr.Slider(
                        label=tr("Rotation Angle"),
                        minimum=-360,
                        maximum=360,
                        value=style.get("Angle", 0),
                        step=1,
                    )
                with gr.Row():
                    ass_margin_l = gr.Slider(
                        label=tr("Left Margin"),
                        minimum=0,
                        maximum=1000,
                        value=style.get("MarginL", 10),
                        step=1,
                    )
                    ass_margin_r = gr.Slider(
                        label=tr("Right Margin"),
                        minimum=0,
                        maximum=1000,
                        value=style.get("MarginR", 10),
                        step=1,
                    )
                    ass_margin_v = gr.Slider(
                        label=tr("Vertical Margin"),
                        minimum=0,
                        maximum=1000,
                        value=style.get("MarginV", 10),
                        step=1,
                    )
                ass_alignment = gr.Dropdown(
                    label=tr("Alignment"),
                    choices=[
                        tr("Bottom Left"),
                        tr("Bottom Center"),
                        tr("Bottom Right"),
                        tr("Middle Left"),
                        tr("Center"),
                        tr("Middle Right"),
                        tr("Top Left"),
                        tr("Top Center"),
                        tr("Top Right"),
                    ],
                    value={
                        1: tr("Bottom Left"),
                        2: tr("Bottom Center"),
                        3: tr("Bottom Right"),
                        4: tr("Middle Left"),
                        5: tr("Center"),
                        6: tr("Middle Right"),
                        7: tr("Top Left"),
                        8: tr("Top Center"),
                        9: tr("Top Right"),
                    }.get(style.get("Alignment", 2), tr("Bottom Center")),
                )
        with gr.Row():
            ass_save_btn = gr.Button(tr("Save Style"), variant="primary")
            ass_reset_btn = gr.Button(tr("Restore Default"))
            ass_status = gr.Textbox(label=tr("Status"), interactive=False, visible=True)

        def save_ass_style(
                fontname,
                fontsize,
                primary_color,
                outline_color,
                back_color,
                bold,
                italic,
                underline,
                strikeout,
                bottom_fontname,
                bottom_fontsize,
                bottom_primary_color,
                bottom_outline_color,
                bottom_back_color,
                bottom_bold,
                bottom_italic,
                border_style,
                outline,
                shadow,
                scale_x,
                scale_y,
                spacing,
                angle,
                margin_l,
                margin_r,
                margin_v,
                alignment,
        ):
            am = {
                tr("Bottom Left"): 1,
                tr("Bottom Center"): 2,
                tr("Bottom Right"): 3,
                tr("Middle Left"): 4,
                tr("Center"): 5,
                tr("Middle Right"): 6,
                tr("Top Left"): 7,
                tr("Top Center"): 8,
                tr("Top Right"): 9,
            }
            _save_ass_style(
                {
                    "Name": "Default",
                    "Fontname": fontname,
                    "Bottom_Fontname": bottom_fontname,
                    "Fontsize": int(fontsize),
                    "Bottom_Fontsize": int(bottom_fontsize),
                    "PrimaryColour": _to_ass_color(primary_color),
                    "Bottom_PrimaryColour": _to_ass_color(bottom_primary_color),
                    "SecondaryColour": "&H00FFFFFF&",
                    "OutlineColour": _to_ass_color(outline_color),
                    "BackColour": _to_ass_color(back_color),
                    "Bold": 1 if bold else 0,
                    "Italic": 1 if italic else 0,
                    "Bottom_SecondaryColour": "&H00FFFFFF&",
                    "Bottom_OutlineColour": _to_ass_color(bottom_outline_color),
                    "Bottom_BackColour": _to_ass_color(bottom_back_color),
                    "Bottom_Bold": 1 if bottom_bold else 0,
                    "Bottom_Italic": 1 if bottom_italic else 0,
                    "Underline": 1 if underline else 0,
                    "StrikeOut": 1 if strikeout else 0,
                    "ScaleX": int(scale_x),
                    "ScaleY": int(scale_y),
                    "Spacing": int(spacing),
                    "Angle": int(angle),
                    "BorderStyle": 1 if border_style == tr("Outline") else 3,
                    "Outline": float(outline),
                    "Shadow": float(shadow),
                    "Alignment": am.get(alignment, 2),
                    "MarginL": int(margin_l),
                    "MarginR": int(margin_r),
                    "MarginV": int(margin_v),
                    "Encoding": 1,
                }
            )
            return tr("Style saved")

        def reset_ass_style():
            _save_ass_style(DEFAULT_STYLE.copy())
            s = DEFAULT_STYLE
            return (
                s["Fontname"],
                s["Fontsize"],
                _parse_ass_color(s["PrimaryColour"]),
                _parse_ass_color(s["OutlineColour"]),
                _parse_ass_color(s["BackColour"]),
                bool(s["Bold"]),
                bool(s["Italic"]),
                bool(s["Underline"]),
                bool(s["StrikeOut"]),
                s["Bottom_Fontname"],
                s["Bottom_Fontsize"],
                _parse_ass_color(s["Bottom_PrimaryColour"]),
                _parse_ass_color(s["Bottom_OutlineColour"]),
                _parse_ass_color(s["Bottom_BackColour"]),
                bool(s["Bottom_Bold"]),
                bool(s["Bottom_Italic"]),
                tr("Outline") if s["BorderStyle"] == 1 else tr("Opaque Background"),
                s["Outline"],
                s["Shadow"],
                s["ScaleX"],
                s["ScaleY"],
                s["Spacing"],
                s["Angle"],
                s["MarginL"],
                s["MarginR"],
                s["MarginV"],
                {
                    1: tr("Bottom Left"),
                    2: tr("Bottom Center"),
                    3: tr("Bottom Right"),
                    4: tr("Middle Left"),
                    5: tr("Center"),
                    6: tr("Middle Right"),
                    7: tr("Top Left"),
                    8: tr("Top Center"),
                    9: tr("Top Right"),
                }.get(s["Alignment"], tr("Bottom Center")),
                tr("Restored default style"),
            )

        ass_save_btn.click(
            fn=save_ass_style,
            inputs=[
                ass_fontname,
                ass_fontsize,
                ass_primary_color,
                ass_outline_color,
                ass_back_color,
                ass_bold,
                ass_italic,
                ass_underline,
                ass_strikeout,
                ass_bottom_fontname,
                ass_bottom_fontsize,
                ass_bottom_primary_color,
                ass_bottom_outline_color,
                ass_bottom_back_color,
                ass_bottom_bold,
                ass_bottom_italic,
                ass_border_style,
                ass_outline,
                ass_shadow,
                ass_scale_x,
                ass_scale_y,
                ass_spacing,
                ass_angle,
                ass_margin_l,
                ass_margin_r,
                ass_margin_v,
                ass_alignment,
            ],
            outputs=[ass_status],
        )

        ass_reset_btn.click(
            fn=reset_ass_style,
            inputs=[],
            outputs=[
                ass_fontname,
                ass_fontsize,
                ass_primary_color,
                ass_outline_color,
                ass_back_color,
                ass_bold,
                ass_italic,
                ass_underline,
                ass_strikeout,
                ass_bottom_fontname,
                ass_bottom_fontsize,
                ass_bottom_primary_color,
                ass_bottom_outline_color,
                ass_bottom_back_color,
                ass_bottom_bold,
                ass_bottom_italic,
                ass_border_style,
                ass_outline,
                ass_shadow,
                ass_scale_x,
                ass_scale_y,
                ass_spacing,
                ass_angle,
                ass_margin_l,
                ass_margin_r,
                ass_margin_v,
                ass_alignment,
                ass_status,
            ],
        )


# ---------------------------------------------------------------------------
# 渠道设置面板构建
# ---------------------------------------------------------------------------
def build_channel_settings():
    """构建所有渠道设置面板"""
    import gradio as gr

    # 按 category 分组
    categories = {}
    for name, cfg in CHANNEL_SETTINGS.items():
        cat = cfg["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, cfg))

    gr.Markdown(tr("### Channel Settings"))
    gr.Markdown(
        tr(
            "Configure API URLs and keys for channels. **Saved settings are shared with desktop version (sp.exe)** in `videotrans/params.json`."
        )
    )

    with gr.Tabs():
        for cat_name, channels in categories.items():
            with gr.Tab(cat_name):
                for ch_name, ch_cfg in channels:
                    with gr.Accordion(ch_name, open=False):
                        fields = []
                        for f in ch_cfg["fields"]:
                            val = str(_safe_get(f["key"], f.get("default", "")))

                            if f['type']=='number':
                                tb=gr.Number(
                                    label=f["label"],
                                    value=val,
                                    placeholder=f.get("placeholder", ""),
                                    interactive=True,
                                )
                            elif f['type']=='checkbox':
                                tb=gr.Checkbox(
                                    label=f["label"],
                                    value=val,
                                    interactive=True,
                                )
                            elif f['type']=='select':
                                tb=gr.Dropdown(
                                    choices=f.get('choices',[]),
                                    label=f["label"],
                                    value=val,
                                    interactive=True,
                                )
                            elif f['type']=='textarea':
                                tb=gr.TextArea(
                                    label=f["label"],
                                    value=val,
                                    placeholder=f.get("placeholder", ""),
                                    interactive=True,
                                )
                            else:
                                tb = gr.Textbox(
                                    label=f["label"],
                                    value=val,
                                    placeholder=f.get("placeholder", ""),
                                    interactive=True,
                                )
                            fields.append((f["key"], tb))

                        save_btn = gr.Button(tr("Save"), size="sm")
                        status = gr.Textbox(
                            label="", interactive=False, visible=True, show_label=False
                        )

                        # 使用闭包捕获当前值
                        def make_save_handler(field_keys, field_widgets):
                            def handler(*values):
                                data = {}
                                for k, v in zip(field_keys, values):
                                    data[k] = v
                                _save_params(data)
                                return tr("Saved")

                            return handler

                        save_btn.click(
                            fn=make_save_handler(
                                [f[0] for f in fields], [f[1] for f in fields]
                            ),
                            inputs=[f[1] for f in fields],
                            outputs=[status],
                        )

        # === 参考音频 Tab ===
        with gr.Tab(tr("Set Reference Audio")):
            gr.Markdown(tr("### Voice Clone Reference Audio Settings"))
            gr.Markdown(
                tr(
                    "Configure reference audio for voice cloning. One per line: `filename.wav#spoken text`\n- Audio files must be placed in `{}/f5-tts/`\n- File format must be wav\n- Use `#` to separate filename and text",
                    ROOT_DIR,
                )
            )

            ref_audio_text = gr.Textbox(
                label=tr("Reference Audio List"),
                value=str(_safe_get("f5tts_role", "")),
                placeholder=tr(
                    "myaudio1.wav#You say all is void, yet why keep your eyes closed\nmyaudio2.wav#Hello, this is a test audio"
                ),
                lines=8,
                interactive=True,
            )

            ref_audio_save = gr.Button(tr("Save Reference Audio"), variant="primary")
            ref_audio_status = gr.Markdown("", visible=False)

            def save_ref_audio(text):
                text = text.strip()
                if not text:
                    return gr.Markdown(tr("Please enter reference audio info"), visible=True)

                lines = text.split("\n")
                errors = []
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("#")
                    if len(parts) != 2:
                        errors.append(
                            tr("Line {} format error: must separate filename and text with #", i + 1)
                        )
                        continue

                    filename = parts[0].strip()
                    f5tts_dir = Path(ROOT_DIR) / "f5-tts"

                    # 检查文件是否存在（支持带/不带 .wav 后缀）
                    if (
                            not (f5tts_dir / filename).exists()
                            and not (f5tts_dir / f"{filename}.wav").exists()
                    ):
                        errors.append(
                            tr(
                                "Line {}: file `{}` does not exist in f5-tts/",
                                i + 1,
                                filename,
                            )
                        )
                        continue

                    # 自动补全 .wav 后缀
                    if (
                            not filename.endswith(".wav")
                            and (f5tts_dir / f"{filename}.wav").exists()
                    ):
                        lines[i] = f"{filename}.wav#{parts[1].strip()}"

                if errors:
                    return gr.Markdown(
                        tr("Save failed:\n") + "\n".join(errors), visible=True
                    )

                role_text = "\n".join(line for line in lines if line.strip())
                _save_params({"f5tts_role": role_text})
                return gr.Markdown(tr("Reference audio saved"), visible=True)

            ref_audio_save.click(
                fn=save_ref_audio,
                inputs=[ref_audio_text],
                outputs=[ref_audio_status],
            )


# ---------------------------------------------------------------------------
# 高级选项设置面板
# ---------------------------------------------------------------------------

# 全局 widget 注册表
_all_widgets = {}
# Page load restores widgets from disk. Ignore dropdown events until that
# restore has finished, so the old default cannot be written back over it.
_advanced_ready = False


def _w(key, label, tip="", area=False):
    """创建一个设置项：标题在上，组件在下"""
    import gradio as gr

    val = str(_user_settings.get(key, ""))
    with gr.Column():
        label_text = f"**{label}**" + (f"\n<sub>{tip}</sub>" if tip else "")
        gr.Markdown(label_text)
        if key in ComboBox_List:
            options = ComboBox_Data.get(key, [val])
            w = gr.Dropdown(
                choices=options,
                value=val if val in options else (options[0] if options else None),
                label="",
                interactive=True,
                show_label=False,
            )
            if key == "uvr_models":
                def _save_bgm_model(value):
                    global _advanced_ready
                    if not _advanced_ready or not value:
                        return
                    chosen = str(value)
                    if chosen not in options:
                        return
                    if str(_load_settings().get("uvr_models", "")) == chosen:
                        return
                    _save_settings({"uvr_models": chosen})

                w.change(fn=_save_bgm_model, inputs=[w], outputs=[])
        elif val.lower() in ("true", "false"):
            w = gr.Checkbox(
                value=val.lower() == "true",
                label="",
                show_label=False,
                interactive=True,
            )
        else:
            w = gr.Textbox(
                value=val,
                label=None,
                show_label=False,
                lines=3 if area else 1,
                interactive=True,
            )
    _all_widgets[key] = w


def _save_section(section_key, keys):
    """为指定分区创建保存按钮和状态显示"""
    import gradio as gr

    with gr.Row():
        save_btn = gr.Button(
            tr("Save {}", tr(f'{section_key}_head')),
            variant="primary",
            size="sm",
        )
        status = gr.Markdown("", visible=False)

    def _make_handler(k_list):
        def handler(*values):
            data = {}
            for k, v in zip(k_list, values):
                data[k] = str(v)
            _save_settings(data)
            return gr.Markdown(tr("Saved"), visible=True)

        return handler

    save_btn.click(
        fn=_make_handler(keys), inputs=[_all_widgets[k] for k in keys], outputs=[status]
    )


def _advanced_form_values():
    global _advanced_ready
    current = _load_settings()
    _user_settings.clear()
    _user_settings.update(current)
    _advanced_ready = True
    return [setting_widget_value(key, current) for key in _all_widgets]


# ---------------------------------------------------------------------------
# 高级选项设置面板（紧凑网格布局）
# ---------------------------------------------------------------------------

def build_advanced_settings():
    import gradio as gr

    gr.Markdown(
        tr(
            "Configure global advanced parameters. **Saved settings are shared with desktop version (sp.exe)** in `videotrans/cfg.json`.\nNote: Some parameters require restarting software to take effect."
        )
    )

    for k, v in notices.items():

        # ---- 通用设置 ----
        with gr.Accordion(tr(f'{k}_head'), open=True):
            for k2 in v:
                with gr.Row():
                    if k2.startswith('initial_prompt_'):
                        _text = tr("Initial prompt for the Whisper model for speech", tr(k2.split('_')[2]))
                        _w(k2, _text, _text)

                    else:
                        _w(k2, tr(f'{k2}_title'), tr(f'{k2}_notice'))
        _save_section(
            k, v
        )


# ---------------------------------------------------------------------------
# UI 构建
# ---------------------------------------------------------------------------
def build_ui():
    import gradio as gr

    with gr.Blocks(title="pyVideoTrans WebUI") as app:
        gr.Markdown(
            tr("""
# pyVideoTrans Video Translation WebUI
> [This interface only implements partial features. For full features, please use the desktop app (sp.exe or sp.py)](https://pyvideotrans.com)
>
>  [Documentation](https://pyvideotrans.com) |
>  [Open Source](https://github.com/jianchang512/pyvideotrans) |
>  [Help & Issues](https://bbs.pyvideotrans.com)
----
        """)
        )

        with gr.Tabs():
            # === Tab 1: 视频翻译 ===
            with gr.Tab(tr("Video Translation"), id="translate"):
                _init_recogn = _saved_index("recogn_type", RECOGN_NAMES, DEFAULT_RECOGN)
                _init_translate = _saved_index(
                    "translate_type", TRANSLATE_NAMES, DEFAULT_TRANSLATE
                )
                _init_tts = _saved_index("tts_type", TTS_NAMES, DEFAULT_TTS)
                _init_models, _init_models_off = _models_for_recogn(_init_recogn)
                _init_model = _user_params.get("model_name") or DEFAULT_MODEL
                if _init_model not in _init_models:
                    _init_model = (
                        DEFAULT_MODEL if DEFAULT_MODEL in _init_models else _init_models[0]
                    )
                prev_recogn = gr.State(value=RECOGN_NAMES[_init_recogn])
                prev_translate = gr.State(value=TRANSLATE_NAMES[_init_translate])
                prev_tts = gr.State(value=TTS_NAMES[_init_tts])

                with gr.Row():
                    with gr.Column(scale=3):
                        input_file = gr.Video(
                            label=tr("Select Video File"), interactive=True
                        )

                        recogn_choice = gr.Dropdown(
                            choices=RECOGN_NAMES,
                            value=RECOGN_NAMES[_init_recogn],
                            label=tr("ASR Channel"),
                            interactive=True,
                        )
                        model_choice = gr.Dropdown(
                            choices=_init_models,
                            value=_init_model,
                            label=tr("Model"),
                            interactive=not _init_models_off,
                        )

                        translate_choice = gr.Dropdown(
                            choices=TRANSLATE_NAMES,
                            value=TRANSLATE_NAMES[_init_translate],
                            label=tr("Translation Channel"),
                            interactive=True,
                        )
                        source_lang = gr.Dropdown(
                            choices=LANG_CHOICES,
                            value=_lang_code_from_display(
                                _user_params.get("source_language", DEFAULT_SOURCE_LANG)
                            ),
                            label=tr("Source Language"),
                            interactive=True,
                        )
                        target_lang = gr.Dropdown(
                            choices=[("-", "-")] + LANG_CHOICES,
                            value=_lang_code_from_display(
                                _user_params.get("target_language", DEFAULT_TARGET_LANG)
                            ),
                            label=tr("Target Language"),
                            interactive=True,
                        )

                        tts_choice = gr.Dropdown(
                            choices=TTS_NAMES,
                            value=TTS_NAMES[_init_tts],
                            label=tr("TTS Channel"),
                            interactive=True,
                        )
                        # 根据已加载的TTS渠道和目标语言预填充角色列表
                        _init_tts_idx = _init_tts
                        _init_target = _user_params.get(
                            "target_language", DEFAULT_TARGET_LANG
                        )
                        _init_langcode = (
                            _lang_code_from_display(_init_target)
                            if _init_target and _init_target != "-"
                            else None
                        )
                        try:
                            _init_roles = role_menu(
                                _init_tts_idx, langcode=_init_langcode
                            )
                            if not _init_roles:
                                _init_roles = ["No"]
                        except Exception:
                            _init_roles = ["No"]
                        _init_role_val = _pick_role(
                            _init_roles, _user_params.get("voice_role", "clone")
                        )
                        voice_role = gr.Dropdown(
                            choices=_init_roles,
                            value=_init_role_val,
                            label=tr("Voice Role"),
                            interactive=True,
                        )
                        per_speaker_voice = gr.Checkbox(
                            label=tr("Each speaker keeps their own voice"),
                            value=_param_bool("per_speaker_voice", True),
                            info=tr("Local diarization, then one cloned voice per speaker."),
                        )
                        voice_per_sentence = gr.Checkbox(
                            label=tr("One voice per sentence"),
                            value=_param_bool("voice_per_sentence", True),
                            info=tr(
                                "Each sentence is cloned from its own audio. The same person can change voice from line to line."
                            ),
                        )

                        with gr.Row():
                            voice_autorate = gr.Checkbox(
                                label=tr("Audio Speedup"),
                                value=_param_bool("voice_autorate", False),
                            )
                            video_autorate = gr.Checkbox(
                                label=tr("Video Slowdown"),
                                value=_param_bool("video_autorate", False),
                            )
                        with gr.Row():
                            voice_rate = gr.Slider(
                                minimum=-50,
                                maximum=50,
                                value=int(
                                    str(_user_params.get("voice_rate", "0")).replace(
                                        "%", ""
                                    )
                                ),
                                step=1,
                                label=tr("Voice Speed (%)"),
                            )
                            volume_rate = gr.Slider(
                                minimum=-95,
                                maximum=100,
                                value=int(
                                    str(_user_params.get("volume", "0")).replace(
                                        "%", ""
                                    )
                                ),
                                step=1,
                                label=tr("Volume (%)"),
                            )
                            pitch_rate = gr.Slider(
                                minimum=-100,
                                maximum=100,
                                value=int(
                                    str(_user_params.get("pitch", "0")).replace(
                                        "Hz", ""
                                    )
                                ),
                                step=1,
                                label=tr("Pitch (Hz)"),
                            )
                        subtitle_type = gr.Dropdown(
                            choices=list(SUBTITLE_TYPES.keys()),
                            value=_option_label(
                                SUBTITLE_TYPES, _param_int("subtitle_type", 0)
                            ),
                            label=tr("Subtitle Embedding Type"),
                            interactive=True,
                        )
                        build_ass_editor()

                        with gr.Accordion(tr("More Settings"), open=False):
                            with gr.Row():
                                remove_noise = gr.Checkbox(
                                    label=tr("Noise Reduction"),
                                    value=_param_bool("remove_noise", False),
                                )
                                fix_punc = gr.Dropdown(
                                    choices=list(PUNC_OPTIONS.keys()),
                                    value=_option_label(
                                        PUNC_OPTIONS, _param_int("fix_punc", 0)
                                    ),
                                    label=tr("Punctuation Processing"),
                                    interactive=True,
                                )
                            with gr.Row():
                                is_separate = gr.Checkbox(
                                    label=tr("Separate Vocals and BGM"),
                                    value=_param_bool("is_separate", True),
                                )
                                embed_bgm = gr.Checkbox(
                                    label=tr("Re-embed Background Audio"),
                                    value=_param_bool("embed_bgm", True),
                                )
                            with gr.Row():
                                loop_bgm = gr.Dropdown(
                                    choices=list(LOOP_BGM_OPTIONS.keys()),
                                    value=_option_label(
                                        LOOP_BGM_OPTIONS, _param_int("loop_backaudio", 0)
                                    ),
                                    label=tr("Background Audio Processing"),
                                    interactive=True,
                                )
                                backaudio_volume = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=float(
                                        _user_params.get(
                                            "backaudio_volume",
                                            settings.get("backaudio_volume", 0.8),
                                        )
                                    ),
                                    step=0.1,
                                    label=tr("Background Audio Volume"),
                                )

                        cuda_accel = gr.Checkbox(
                            label=tr("Enable CUDA Acceleration"),
                            value=_param_bool("is_cuda", True),
                        )
                        clear_output = gr.Checkbox(
                            label=tr("Clear previous output"),
                            value=_param_bool("clear_cache", False),
                            info=tr(
                                "Deletes subtitles and files from the last run of this video. Leave it off to keep them."
                            ),
                        )
                        speaker_ref_threads = gr.Slider(
                            minimum=0,
                            maximum=16,
                            value=int(_user_params.get("speaker_ref_threads", 0) or 0),
                            step=1,
                            label=tr("Speaker reference threads"),
                            info=tr(
                                "0 follows the CPU, up to 8 threads. A higher number overrides that."
                            ),
                        )
                        speaker_cap = gr.Slider(
                            minimum=0,
                            maximum=30,
                            value=int(_user_params.get("speaker_cap", 8) or 0),
                            step=1,
                            label=tr("Maximum speakers"),
                            info=tr(
                                "0 means no limit. 8 keeps the film to at most 8 voices."
                            ),
                        )
                        test_minutes = gr.Slider(
                            minimum=0,
                            maximum=60,
                            value=int(_user_params.get("test_minutes", 0) or 0),
                            step=1,
                            label=tr("Test length in minutes"),
                            info=tr(
                                "0 uses the whole film. 10 runs only the first 10 minutes."
                            ),
                        )
                        with gr.Row():
                            spoken_srt = gr.File(
                                label=tr("Import spoken SRT"),
                                file_types=[".srt"],
                                type="filepath",
                            )
                            target_srt = gr.File(
                                label=tr("Import target language SRT"),
                                file_types=[".srt"],
                                type="filepath",
                            )
                        gr.Markdown(
                            tr(
                                "A spoken file skips speech recognition. A target file skips translation. The two files need the same number of lines."
                            )
                        )
                        channel_warning = gr.Markdown("", visible=False)

                        start_btn = gr.Button(
                            tr("Start"), variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        log_output = gr.Textbox(
                            label=tr("Execution Log"), lines=20, interactive=False
                        )
                        video_preview = gr.Video(
                            label=tr("Video Preview"), interactive=False
                        )
                        result_files = gr.File(
                            label=tr("Output Files (Click to Download)"), interactive=False
                        )

                # 渠道验证并更新模型列表
                def validate_recogn(choice, prev, current_model):
                    idx = _recogn_index_from_display(choice)

                    _rs = recognition.is_input_api(recogn_type=idx, return_str=True)
                    if _rs is not True:
                        msg = tr("Channel '{}' is currently unavailable and has been rolled back").format(choice)
                        gr.Warning(msg)
                        return prev, f"⚠️ {msg}", gr.update()

                    # 根据渠道更新模型下拉框
                    print(f'{idx=}')
                    models, disabled = _models_for_recogn(idx)

                    if models:
                        if current_model in models:
                            default_val = current_model
                        elif DEFAULT_MODEL in models:
                            default_val = DEFAULT_MODEL
                        else:
                            default_val = models[0]
                        return (
                            choice,
                            "",
                            gr.update(
                                choices=models,
                                value=default_val,
                                interactive=not disabled,
                            ),
                        )
                    return choice, "", gr.update(interactive=False)

                def validate_translate(choice, prev):
                    idx = _translate_index_from_display(choice)
                    _rs = translator.is_allow_translate(
                        translate_type=idx, return_str=True
                    )
                    if _rs is not True:
                        msg = tr("Channel '{}' is currently unavailable and has been rolled back").format(choice)
                        gr.Warning(msg)
                        return prev, f"⚠️ {msg}"
                    return choice, ""

                def tts_change_handler(choice, prev, target_display, current_role):
                    idx = _tts_index_from_display(choice)
                    warning = ""
                    _rs = tts.is_input_api(tts_type=idx, return_str=True)
                    if _rs is not True:
                        msg = tr("Channel '{}' is currently unavailable and has been rolled back").format(choice)
                        gr.Warning(msg)
                        choice = prev
                        warning = f"⚠️ {msg}"
                    tts_idx = _tts_index_from_display(choice)
                    lang_code = _lang_code_from_display(target_display)
                    try:
                        roles = role_menu(tts_idx, langcode=lang_code)
                        if not roles:
                            roles = ["No"]
                    except Exception:
                        roles = ["No"]
                    return (
                        choice,
                        gr.update(choices=roles, value=_pick_role(roles, current_role)),
                        warning,
                    )

                recogn_choice.change(
                    fn=validate_recogn,
                    inputs=[recogn_choice, prev_recogn, model_choice],
                    outputs=[recogn_choice, channel_warning, model_choice],
                )
                translate_choice.change(
                    fn=validate_translate,
                    inputs=[translate_choice, prev_translate],
                    outputs=[translate_choice, channel_warning],
                )
                tts_choice.change(
                    fn=tts_change_handler,
                    inputs=[tts_choice, prev_tts, target_lang, voice_role],
                    outputs=[tts_choice, voice_role, channel_warning],
                )

                def update_voice_roles(tts_display, target_display, current_role):
                    tts_idx = _tts_index_from_display(tts_display)
                    lang_code = _lang_code_from_display(target_display)
                    try:
                        roles = role_menu(tts_idx, langcode=lang_code)
                        if not roles:
                            roles = ["No"]
                    except Exception:
                        roles = ["No"]
                    return gr.update(choices=roles, value=_pick_role(roles, current_role))

                target_lang.change(
                    fn=update_voice_roles,
                    inputs=[tts_choice, target_lang, voice_role],
                    outputs=[voice_role],
                )

                # 执行翻译
                _BTN_RUNNING = gr.update(value=tr("Processing..."), interactive=False)
                _BTN_IDLE = gr.update(value=tr("Start"), interactive=True)

                def run_translation(
                        file_path,
                        recogn_display,
                        model_name,
                        translate_display,
                        source_display,
                        target_display,
                        tts_display,
                        voice_role_name,
                        voice_autorate_val,
                        video_autorate_val,
                        voice_rate_val,
                        volume_rate_val,
                        pitch_rate_val,
                        subtitle_type_name,
                        remove_noise_val,
                        fix_punc_name,
                        is_separate_val,
                        embed_bgm_val,
                        loop_bgm_name,
                        backaudio_volume_val,
                        cuda_val,
                        per_speaker_val,
                        voice_per_sentence_val,
                        clear_output_val,
                        speaker_ref_threads_val,
                        speaker_cap_val,
                        test_minutes_val,
                        spoken_srt_file,
                        target_srt_file,
                ):
                    print(f"{file_path=}")
                    if not file_path:
                        yield tr("Please select a video or audio file first"), None, [], _BTN_IDLE
                        return
                    app_cfg.current_status = "ing"
                    # 清空上次的日志、预览和输出，显示执行中状态
                    yield "", None, [], _BTN_RUNNING

                    log_lines = []

                    def log(msg):
                        return _append_exec_log(log_lines, msg, time.strftime("%H:%M:%S"))

                    recogn_idx = _recogn_index_from_display(recogn_display)
                    translate_idx = _translate_index_from_display(translate_display)
                    tts_idx = _tts_index_from_display(tts_display)
                    saved_role = voice_role_name
                    per_speaker = bool(per_speaker_val)
                    local_clone = set(tts.SUPPORT_CLONE) & set(tts.LOCAL_BUILTIN)
                    switched_tts = False
                    if per_speaker and tts_idx not in local_clone:
                        tts_idx = tts.CHATTERBOX_TTS
                        switched_tts = True
                    if per_speaker:
                        voice_role_name = "clone"
                    source_code = _lang_code_from_display(source_display)
                    target_code = _lang_code_from_display(target_display)
                    subtitle_val = SUBTITLE_TYPES.get(subtitle_type_name, 0)
                    fix_punc_val = PUNC_OPTIONS.get(fix_punc_name, 0)
                    loop_bgm_val = LOOP_BGM_OPTIONS.get(loop_bgm_name, 0)
                    try:
                        _save_params({
                            "recogn_type": recogn_idx,
                            "model_name": model_name,
                            "translate_type": translate_idx,
                            "source_language": source_code,
                            "target_language": target_code,
                            "tts_type": _tts_index_from_display(tts_display),
                            "voice_role": saved_role,
                            "per_speaker_voice": per_speaker,
                            "voice_per_sentence": bool(voice_per_sentence_val),
                            "voice_autorate": bool(voice_autorate_val),
                            "video_autorate": bool(video_autorate_val),
                            "voice_rate": _format_rate(int(voice_rate_val)),
                            "volume": _format_rate(int(volume_rate_val)),
                            "pitch": _format_pitch(int(pitch_rate_val)),
                            "subtitle_type": subtitle_val,
                            "remove_noise": bool(remove_noise_val),
                            "fix_punc": fix_punc_val,
                            "is_separate": bool(is_separate_val),
                            "embed_bgm": bool(embed_bgm_val),
                            "loop_backaudio": loop_bgm_val,
                            "backaudio_volume": float(backaudio_volume_val),
                            "is_cuda": bool(cuda_val),
                            "clear_cache": bool(clear_output_val),
                            "speaker_ref_threads": int(speaker_ref_threads_val or 0),
                            "speaker_cap": int(speaker_cap_val or 0),
                            "test_minutes": int(test_minutes_val or 0),
                        })
                    except Exception:
                        traceback.print_exc()

                    try:
                        app_cfg.exit_soft = False
                        app_cfg.exec_mode = "cli"

                        getset_gpu()
                        test_minutes_int = int(test_minutes_val or 0)
                        if test_minutes_int > 0:
                            clip_dir = Path(TEMP_DIR) / "test-clips"
                            clip_dir.mkdir(parents=True, exist_ok=True)
                            clip_path = test_clip_target(file_path, clip_dir, test_minutes_int)
                            source_path = Path(file_path)
                            if (
                                not clip_path.is_file()
                                or clip_path.stat().st_mtime < source_path.stat().st_mtime
                            ):
                                runffmpeg(test_clip_args(source_path, clip_path, test_minutes_int))
                            file_path = clip_path.as_posix()
                            yield (
                                log(tr("Testing the first {} minutes.", test_minutes_int)),
                                None,
                                [],
                                _BTN_RUNNING,
                            )
                        _file_obj = tools.format_video(
                            Path(file_path).absolute().as_posix()
                        )
                        _nospacebasename = (
                            _file_obj["basename"].replace(" ", "-").replace(".", "-")
                        )
                        _cache_folder = f"{TEMP_DIR}/{_file_obj['uuid']}"
                        app_cfg.rm_uuid(_file_obj["uuid"])
                        _target_dir = f"{ROOT_DIR}/output/{_nospacebasename}"
                        _file_obj["target_dir"] = _target_dir
                        Path(_cache_folder).mkdir(parents=True, exist_ok=True)
                        Path(_target_dir).mkdir(parents=True, exist_ok=True)

                        from dataclasses import asdict

                        common_params = {
                            "name": file_path,
                            "cache_folder": _cache_folder,
                        }
                        common_params.update(asdict(_file_obj))
                        yield (
                            log(tr("Source File: {}", Path(file_path).name)),
                            None,
                            [],
                            _BTN_RUNNING,
                        )

                        vtv_params = {
                            "source_language_code": source_code,
                            "target_language_code": target_code,
                            "recogn_type": recogn_idx,
                            "model_name": model_name,
                            "is_cuda": cuda_val,
                            "remove_noise": remove_noise_val,
                            "enable_diariz": per_speaker,
                            "nums_diariz": speaker_cap_to_nums(speaker_cap_val) if per_speaker else 0,
                            "detect_language": source_code,
                            "rephrase": 0,
                            "fix_punc": fix_punc_val,
                            "tts_type": tts_idx,
                            "voice_role": voice_role_name,
                            "voice_rate": _format_rate(int(voice_rate_val)),
                            "volume": _format_rate(int(volume_rate_val)),
                            "pitch": _format_pitch(int(pitch_rate_val)),
                            "voice_autorate": voice_autorate_val,
                            "video_autorate": video_autorate_val,
                            "align_sub_audio": True,
                            "voice_per_sentence": bool(voice_per_sentence_val),
                            "translate_type": translate_idx,
                            "is_separate": is_separate_val,
                            "recogn2pass": False,
                            "subtitle_type": subtitle_val,
                            "clear_cache": bool(clear_output_val),
                            "embed_bgm": embed_bgm_val,
                            "loop_backaudio": loop_bgm_val,
                            "backaudio_volume": backaudio_volume_val,
                            "background_music": "",
                        }
                        params_dict = {**common_params, **vtv_params}

                        yield (
                            log(
                                tr(
                                    "ASR: {}  Translation: {}  TTS: {}",
                                    RECOGN_NAMES[recogn_idx],
                                    TRANSLATE_NAMES[translate_idx],
                                    TTS_NAMES[tts_idx],
                                )
                            ),
                            None,
                            [],
                            _BTN_RUNNING,
                        )
                        yield (
                            log(
                                tr(
                                    "Language: {} -> {}  Role: {}",
                                    source_code,
                                    target_code,
                                    voice_role_name,
                                )
                            ),
                            None,
                            [],
                            _BTN_RUNNING,
                        )
                        if bool(voice_per_sentence_val):
                            yield (
                                log(tr("Each sentence keeps its own cloned voice.")),
                                None,
                                [],
                                _BTN_RUNNING,
                            )
                        yield log(""), None, [], _BTN_RUNNING

                        yield log(tr("Starting video translation...")), None, [], _BTN_RUNNING
                        from videotrans.task.trans_create import TransCreate
                        from videotrans.task.taskcfg import TaskCfgVTT

                        trk = TransCreate(cfg=TaskCfgVTT(**params_dict))
                        trk.do_diarize = per_speaker
                        if bool(clear_output_val):
                            yield (
                                log(tr("Previous output and cache were deleted.")),
                                None,
                                [],
                                _BTN_RUNNING,
                            )
                        from videotrans.task._import_srt import install_subtitle_files
                        from videotrans.task._reference_srt import load_reference_uploads

                        reference_pairs = load_reference_uploads(spoken_srt_file, target_srt_file)
                        if reference_pairs is not None:
                            trk.reference_pairs = reference_pairs
                            Path(trk.cfg.target_sub).unlink(missing_ok=True)
                            yield (
                                log(
                                    tr(
                                        "The uploaded subtitles are a reference for the translator, {} lines.",
                                        len(reference_pairs),
                                    )
                                ),
                                None,
                                [],
                                _BTN_RUNNING,
                            )
                            imported = {"spoken": None, "target": None}
                        else:
                            try:
                                imported = install_subtitle_files(
                                    spoken_srt_file,
                                    target_srt_file,
                                    trk.cfg.source_sub,
                                    trk.cfg.target_sub,
                                    limit_ms=test_minutes_int * 60 * 1000 if test_minutes_int > 0 else None,
                                    drop_paths=(
                                        f"{_target_dir}/speaker.json",
                                        f"{_cache_folder}/speaker.json",
                                        f"{_target_dir}/words.json",
                                        f"{_cache_folder}/words.json",
                                    ),
                                )
                            except ValueError as exc:
                                yield (
                                    log(tr("The subtitle file has no lines inside this run: {}", exc)),
                                    None,
                                    [],
                                    _BTN_IDLE,
                                )
                                return
                        if imported["spoken"] is not None or imported["target"] is not None:
                            trk.force_new_dubbing = True
                            yield (
                                log(tr("The selected subtitles are used, and the voices are generated again.")),
                                None,
                                [],
                                _BTN_RUNNING,
                            )
                        if imported["spoken"] is not None:
                            trk.keep_imported_lines = True
                            yield (
                                log(tr("Using the uploaded spoken subtitles, {} lines.", imported["spoken"])),
                                None,
                                [],
                                _BTN_RUNNING,
                            )
                        if imported["target"] is not None:
                            yield (
                                log(tr("Using the uploaded target subtitles, {} lines.", imported["target"])),
                                None,
                                [],
                                _BTN_RUNNING,
                            )
                        if (
                            imported["spoken"] is not None
                            and imported["target"] is not None
                            and imported["spoken"] != imported["target"]
                        ):
                            yield (
                                log(
                                    tr(
                                        "The spoken file has {} lines and the target file has {} lines.",
                                        imported["spoken"],
                                        imported["target"],
                                    )
                                ),
                                None,
                                [],
                                _BTN_RUNNING,
                            )
                        app_cfg.line_roles = {}
                        if per_speaker:
                            note = tr("Per-speaker voice: local diarization, then one cloned voice per speaker.")
                            if switched_tts:
                                note += " " + tr("This TTS cannot clone locally, so this run uses Chatterbox.")
                            yield log(note), None, [], _BTN_RUNNING

                        stages = [
                            (tr("Stage 1/8: Preprocessing..."), "prepare", tr("Preprocessing completed")),
                            (tr("Stage 2/8: Speech recognition..."), "recogn", tr("Speech recognition completed")),
                            (
                                tr("Stage 3/8: Speaker diarization..."),
                                "diariz",
                                tr("Speaker diarization completed"),
                            ),
                            (tr("Stage 4/8: Subtitle translation..."), "trans", tr("Subtitle translation completed")),
                            (
                                tr("Stage 5/8: Dubbing generation..."),
                                "dubbing",
                                tr("Dubbing generation completed"),
                            ),
                            (tr("Stage 6/8: Audio-video alignment..."), "align", tr("Audio-video alignment completed")),
                            (
                                tr("Stage 7/8: Second-pass recognition..."),
                                "recogn2pass",
                                tr("Second-pass recognition completed"),
                            ),
                            (
                                tr("Stage 8/8: Final assembly..."),
                                "assembling",
                                tr("Final assembly completed"),
                            ),
                        ]
                        app_cfg.web_log_queue = queue.Queue()

                        def _drain_web_log():
                            while True:
                                try:
                                    text = app_cfg.web_log_queue.get_nowait()
                                except queue.Empty:
                                    return
                                yield text

                        def _run_stage(method_name):
                            error = []

                            def work():
                                try:
                                    getattr(trk, method_name)()
                                except Exception as exc:
                                    error.append(exc)

                            thread = threading.Thread(target=work, daemon=True)
                            thread.start()
                            try:
                                while thread.is_alive():
                                    for text in _drain_web_log():
                                        yield log(text), None, [], _BTN_RUNNING
                                    thread.join(0.4)
                            finally:
                                # Closing the page raises GeneratorExit at the yield.
                                # Wait for this stage so a second click cannot start
                                # another pipeline while the first one is still running.
                                thread.join()
                            for text in _drain_web_log():
                                yield log(text), None, [], _BTN_RUNNING
                            if error:
                                exc = error[0]
                                raise exc.with_traceback(exc.__traceback__)

                        for stage_name, method, done_msg in stages:
                            yield log(stage_name), None, [], _BTN_RUNNING
                            yield from _run_stage(method)
                            if method == "diariz" and per_speaker:
                                yield log(_speaker_summary(_cache_folder, _target_dir)), None, [], _BTN_RUNNING
                            if method != "assembling":
                                yield log(f"✓ {done_msg}"), None, [], _BTN_RUNNING

                        trk.task_done()
                        yield log(tr("Video synthesis completed")), None, [], _BTN_RUNNING
                        yield log(tr("All tasks completed!")), None, [], _BTN_RUNNING

                        output_files, video_preview_path = [], None

                        if target_path.exists():
                            for f in sorted(target_path.rglob("*")):
                                if f.is_file():
                                    if (
                                            f.suffix.lower() == ".mp4"
                                            and video_preview_path is None
                                    ):
                                        video_preview_path = str(f)
                                    else:
                                        output_files.append(str(f))
                        if not output_files and video_preview_path is None:
                            for f in sorted(Path(_cache_folder).rglob("*")):
                                if f.is_file():
                                    if (
                                            f.suffix.lower() == ".mp4"
                                            and video_preview_path is None
                                    ):
                                        video_preview_path = str(f)
                                    elif f.suffix.lower() in (
                                            ".mkv",
                                            ".wav",
                                            ".srt",
                                            ".txt",
                                            ".mp3",
                                    ):
                                        output_files.append(str(f))
                        # 添加当天日志文件到输出列表
                        import datetime

                        log_file = (
                                Path(ROOT_DIR)
                                / "logs"
                                / f"{datetime.datetime.now().strftime('%Y%m%d')}.log"
                        )
                        if log_file.exists():
                            output_files.append(str(log_file))

                        yield (
                            log(tr("Output Directory: {}", _target_dir)),
                            video_preview_path,
                            output_files,
                            _BTN_IDLE,
                        )

                    except Exception as e:
                        tb = traceback.format_exc()
                        yield (
                            log(tr("Execution error: {}\n\n{}", str(e), tb)),
                            None,
                            [],
                            _BTN_IDLE,
                        )
                    finally:
                        app_cfg.web_log_queue = None

                start_btn.click(
                    fn=run_translation,
                    inputs=[
                        input_file,
                        recogn_choice,
                        model_choice,
                        translate_choice,
                        source_lang,
                        target_lang,
                        tts_choice,
                        voice_role,
                        voice_autorate,
                        video_autorate,
                        voice_rate,
                        volume_rate,
                        pitch_rate,
                        subtitle_type,
                        remove_noise,
                        fix_punc,
                        is_separate,
                        embed_bgm,
                        loop_bgm,
                        backaudio_volume,
                        cuda_accel,
                        per_speaker_voice,
                        voice_per_sentence,
                        clear_output,
                        speaker_ref_threads,
                        speaker_cap,
                        test_minutes,
                        spoken_srt,
                        target_srt,
                    ],
                    outputs=[log_output, video_preview, result_files, start_btn],
                )

                def _restore_saved_form():
                    fresh = _load_params()
                    if isinstance(fresh, dict) and fresh:
                        _user_params.clear()
                        _user_params.update(fresh)
                    recogn_idx = _saved_index("recogn_type", RECOGN_NAMES, DEFAULT_RECOGN)
                    translate_idx = _saved_index(
                        "translate_type", TRANSLATE_NAMES, DEFAULT_TRANSLATE
                    )
                    tts_idx = _saved_index("tts_type", TTS_NAMES, DEFAULT_TTS)
                    models, models_off = _models_for_recogn(recogn_idx)
                    saved_model = _user_params.get("model_name") or DEFAULT_MODEL
                    if saved_model not in models:
                        saved_model = (
                            DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)
                        )
                    source_code = _lang_code_from_display(
                        _user_params.get("source_language", DEFAULT_SOURCE_LANG)
                    )
                    target_raw = _user_params.get("target_language", DEFAULT_TARGET_LANG)
                    target_code = (
                        _lang_code_from_display(target_raw)
                        if target_raw not in (None, "", "-")
                        else "-"
                    )
                    try:
                        roles = role_menu(
                            tts_idx,
                            langcode=target_code if target_code not in (None, "", "-") else None,
                        ) or ["No"]
                    except Exception:
                        roles = ["No"]
                    role_val = _pick_role(roles, _user_params.get("voice_role", "clone"))
                    try:
                        back_vol = float(_user_params.get("backaudio_volume", 0.6))
                    except (TypeError, ValueError):
                        back_vol = 0.6
                    return (
                        RECOGN_NAMES[recogn_idx],
                        gr.update(
                            choices=models,
                            value=saved_model,
                            interactive=not models_off,
                        ),
                        TRANSLATE_NAMES[translate_idx],
                        source_code,
                        target_code,
                        TTS_NAMES[tts_idx],
                        gr.update(choices=roles, value=role_val),
                        _param_bool("voice_autorate", False),
                        _param_bool("video_autorate", False),
                        _percent_int("voice_rate", 0),
                        _percent_int("volume", 0),
                        _percent_int("pitch", 0),
                        _option_label(SUBTITLE_TYPES, _param_int("subtitle_type", 0)),
                        _param_bool("remove_noise", False),
                        _option_label(PUNC_OPTIONS, _param_int("fix_punc", 0)),
                        _param_bool("is_separate", True),
                        _param_bool("embed_bgm", True),
                        _option_label(LOOP_BGM_OPTIONS, _param_int("loop_backaudio", 0)),
                        back_vol,
                        _param_bool("is_cuda", True),
                        _param_bool("per_speaker_voice", True),
                        _param_bool("voice_per_sentence", True),
                        _param_bool("clear_cache", False),
                        int(_user_params.get("speaker_ref_threads", 0) or 0),
                        int(_user_params.get("speaker_cap", 8) or 0),
                        int(_user_params.get("test_minutes", 0) or 0),
                        RECOGN_NAMES[recogn_idx],
                        TRANSLATE_NAMES[translate_idx],
                        TTS_NAMES[tts_idx],
                    )

                app.load(
                    fn=_restore_saved_form,
                    outputs=[
                        recogn_choice,
                        model_choice,
                        translate_choice,
                        source_lang,
                        target_lang,
                        tts_choice,
                        voice_role,
                        voice_autorate,
                        video_autorate,
                        voice_rate,
                        volume_rate,
                        pitch_rate,
                        subtitle_type,
                        remove_noise,
                        fix_punc,
                        is_separate,
                        embed_bgm,
                        loop_bgm,
                        backaudio_volume,
                        cuda_accel,
                        per_speaker_voice,
                        voice_per_sentence,
                        clear_output,
                        speaker_ref_threads,
                        speaker_cap,
                        test_minutes,
                        prev_recogn,
                        prev_translate,
                        prev_tts,
                    ],
                )

            # === Tab 2: 渠道设置 ===
            with gr.Tab(tr("Channel Settings"), id="settings"):
                build_channel_settings()

            # === Tab 3: 高级选项 ===
            with gr.Tab(tr("Advanced Options"), id="advanced"):
                build_advanced_settings()
                if _all_widgets:
                    app.load(
                        fn=_advanced_form_values,
                        outputs=list(_all_widgets.values()),
                    )

    return app


if __name__ == "__main__":
    try:
        import argparse
        import gradio as gr

        parser = argparse.ArgumentParser(description="pyVideoTrans WebUI")
        parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
        parser.add_argument("--port", type=int, default=7860, help="Port number")
        parser.add_argument(
            "--share", action="store_true", help="Create a public Gradio link"
        )
        args = parser.parse_args()
        app = build_ui()
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            inbrowser=True,
            # /app/output is a symlink to the mounted /data volume.
            allowed_paths=[str(ROOT_DIR), "/data", "/tmp"],
            theme=gr.themes.Soft(),
            css="""
        *, *::before, *::after {
            font-family: "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", sans-serif !important;
        }
        h1{text-align:center}
        input, textarea, select, button, label, .gr-textbox, .gr-dropdown, .gr-checkbox {
            font-family: "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", sans-serif !important;
        }
    """,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(tr("\nLaunch failed: {}", e))
