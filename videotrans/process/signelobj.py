import multiprocessing
import os
import threading
import time

from videotrans.configure.config import app_cfg, settings, logger

def pool_init_worker():
    import sys
    class NullWriter:
        def write(self, *args, **kwargs): pass
        def flush(self, *args, **kwargs): pass
        @staticmethod
        def isatty(): return False
        @staticmethod
        def fileno(): return -1
        @staticmethod
        def read(*args, **kwargs): return ""
        def close(self): pass
        @property
        def encoding(self): return "utf-8"
        @property
        def closed(self): return False

    _null_writer = NullWriter()
    for _stream in ['stdout', 'stderr', 'stdin', '__stdout__', '__stderr__', '__stdin__']:
        if getattr(sys, _stream, None) is None:
            setattr(sys, _stream, _null_writer)


def _task_worker_entry(func, kwargs, result_q):
    pool_init_worker()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        result_q.put(("ok", func(**kwargs)))
    except Exception as e:
        result_q.put(("err", f"{type(e).__name__}: {e}"))


class OneShotFuture:
    """One process per task. The result queue is read while the child is alive.

    A 2-hour transcript is larger than the pipe buffer. Waiting for the child
    to exit before reading left both sides blocked.
    """

    def __init__(self, proc, result_q):
        self.proc = proc
        self.result_q = result_q
        self._cached = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._collect_bg, daemon=True)
        self._thread.start()

    def done(self):
        if self.proc.is_alive():
            return False
        self._ready.wait(timeout=5)
        return True

    def result(self, timeout=None):
        self._ready.wait(None if timeout is None else timeout)
        if self._cached is None:
            return None, "Subprocess crashed hard (Segmentation Fault/OOM)"
        return self._cached

    def _collect_bg(self):
        kind, payload = "dead", None
        try:
            kind, payload = self.result_q.get()
        except Exception:
            kind, payload = "dead", None
        self.proc.join(timeout=30)
        try:
            self.result_q.close()
        except Exception:
            pass
        if kind == "ok":
            self._cached = payload
        elif kind == "err":
            self._cached = (None, f"Subprocess Error: {payload}")
        else:
            self._cached = (None, "Subprocess crashed hard (Segmentation Fault/OOM)")
        self._ready.set()


# ==========================================
# 全局单例管理器
# ==========================================

class GlobalProcessManager:
    _executor_cpu = None
    _executor_gpu = None



    @classmethod
    def get_gpu_process_nums(cls):
        cpu_count = max(int(os.cpu_count()),1)
        try:
            process_max_gpu = max( int(float(settings.get('process_max_gpu', 0))) ,1)
        except (TypeError, ValueError):
            process_max_gpu = 1

        return int(min(process_max_gpu, 8, cpu_count))


    @classmethod
    def _submit(cls, func, kwargs):
        ctx = multiprocessing.get_context("spawn")
        result_q = ctx.Queue()
        proc = ctx.Process(target=_task_worker_entry, args=(func, kwargs, result_q))
        proc.start()
        return OneShotFuture(proc, result_q)

    @classmethod
    def submit_task_cpu(cls, func, **kwargs):
        return cls._submit(func, kwargs)

    @classmethod
    def submit_task_gpu(cls, func, **kwargs):
        return cls._submit(func, kwargs)

    @classmethod
    def shutdown(cls):
        if cls._executor_cpu:
            cls._executor_cpu.close()
            cls._executor_cpu.join()
            cls._executor_cpu = None
        if cls._executor_gpu:
            cls._executor_gpu.close()
            cls._executor_gpu.join()
            cls._executor_gpu = None
