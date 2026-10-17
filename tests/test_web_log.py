import queue

from webui import _append_exec_log
from videotrans.configure.base import BaseCon
from videotrans.configure.config import app_cfg


def test_download_progress_replaces_the_previous_line():
    lines = []
    _append_exec_log(lines, "Stage 2/8: Speech recognition...", "08:19:01")
    prefix = "Downloading... Please wait if progress freezes. large-v2.pt"
    _append_exec_log(lines, f"{prefix}:10.00%", "08:19:02")
    _append_exec_log(lines, f"{prefix}:88.00%", "08:20:02")
    assert lines == [
        "[08:19:01] Stage 2/8: Speech recognition...",
        "[08:20:02] Downloading... Please wait if progress freezes. large-v2.pt:88.00%",
    ]


def test_other_log_lines_stay():
    lines = []
    _append_exec_log(lines, "Dubbing 1/3", "08:19:01")
    _append_exec_log(lines, "Dubbing 2/3", "08:19:02")
    assert len(lines) == 2


def test_signal_reaches_the_web_log_and_skips_subtitle_dumps():
    previous_mode = app_cfg.exec_mode
    previous_queue = getattr(app_cfg, "web_log_queue", None)
    notes = queue.Queue()
    app_cfg.exec_mode = "cli"
    app_cfg.web_log_queue = notes
    try:
        worker = BaseCon()
        worker.signal(text="large-v2.pt:12.00%")
        worker.signal(text="1\n00:00:01,000 --> 00:00:02,000\nHello\n", type="replace_subtitle")
        assert notes.get_nowait() == "large-v2.pt:12.00%"
        assert notes.empty()
    finally:
        app_cfg.exec_mode = previous_mode
        app_cfg.web_log_queue = previous_queue


def test_download_callback_is_throttled_but_always_emits_100():
    previous_mode = app_cfg.exec_mode
    previous_queue = getattr(app_cfg, "web_log_queue", None)
    notes = queue.Queue()
    app_cfg.exec_mode = "cli"
    app_cfg.web_log_queue = notes
    try:
        worker = BaseCon()
        worker.last_down_time = 0
        worker._process_callback("large-v2.pt:10.00%")
        worker._process_callback("large-v2.pt:10.50%")
        worker._process_callback("large-v2.pt:100.00%")
        assert notes.get_nowait().endswith("large-v2.pt:10.00%")
        assert notes.get_nowait().endswith("large-v2.pt:100.00%")
        assert notes.empty()
    finally:
        app_cfg.exec_mode = previous_mode
        app_cfg.web_log_queue = previous_queue
