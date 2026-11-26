"""后台扫描线程：在 QThread 中运行 scan_core.run_scan，界面通过定时器读取进度。"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal

from scan_core import SAT_LEVEL_V, ScanParams, run_scan


def format_error(e: Exception) -> str:
    code = getattr(e, "error_code", None)
    msg = str(e).strip()
    return f"[错误代码 {code}] {msg}" if code is not None else f"{type(e).__name__}: {msg}"


class ScanThread(QThread):
    scan_done = Signal(str)      # "完成" / "已停止"
    scan_failed = Signal(str)    # 错误信息

    def __init__(self, backend, params: ScanParams, img: np.ndarray, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.params = params
        self.img = img
        self._stop = False
        self.count = 0
        self.sat_count = 0
        self.current = None      # (iy, ix, x, y, v, peak)

    def request_stop(self):
        self._stop = True

    def _on_point(self, iy, ix, x, y, v, peak):
        self.count += 1
        if peak >= SAT_LEVEL_V:
            self.sat_count += 1
        self.current = (iy, ix, x, y, v, peak)

    def run(self):
        try:
            status = run_scan(self.backend, self.params, self.img,
                              lambda: self._stop, self._on_point)
            self.scan_done.emit(status)
        except Exception as e:
            self.scan_failed.emit(format_error(e))
