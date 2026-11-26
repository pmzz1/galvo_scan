import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from daq_backend import Backend, SimBackend  # noqa: E402
from scan_core import ScanParams, axes, run_scan  # noqa: E402

FAST = dict(settle_ms=0, row_wait_ms=0)


def make(p, **kw):
    x, y = axes(p)
    img = np.zeros((len(y), len(x)))
    b = SimBackend(samples_per_point=p.samples_per_point, realistic_timing=False, record=True,
                   safety_limit_v=p.safety_limit_v, seed=1, **kw)
    return b, img


def test_full_scan_user_example(monkeypatch):
    monkeypatch.setattr("scan_core.FIRST_POINT_EXTRA_MS", 0)
    p = ScanParams(**FAST)
    b, img = make(p)
    status = run_scan(b, p, img, lambda: False)
    assert status == "完成"
    assert img.shape == (41, 21)
    assert np.isfinite(img).all()
    assert 0 <= img.min() and img.max() <= 2.0
    assert b.writes[-1] == (0.0, 0.0)            # 结束回零
    assert b.open_count == 1 and b.close_count == 1
    assert max(max(abs(x), abs(y)) for x, y in b.writes) <= 2.0 + 1e-9


def test_ramp_to_start_is_gradual(monkeypatch):
    monkeypatch.setattr("scan_core.FIRST_POINT_EXTRA_MS", 0)
    p = ScanParams(x_start=-5, x_stop=-4, y_start=5, y_stop=5.2, x_step=0.5, y_step=0.1, **FAST)
    b, img = make(p)
    run_scan(b, p, img, lambda: False)
    pts = np.array([(0.0, 0.0)] + b.writes)
    first_scan_idx = next(i for i, w in enumerate(b.writes) if w == (-5.0, 5.0))
    steps = np.abs(np.diff(pts[: first_scan_idx + 2], axis=0)).max(axis=1)
    assert steps.max() <= Backend.ramp_step_v + 1e-9   # 到起点的斜坡每步 ≤ 0.05 V
    assert b.writes[-1] == (0.0, 0.0)


def test_serpentine_pixels_match_pattern(monkeypatch):
    monkeypatch.setattr("scan_core.FIRST_POINT_EXTRA_MS", 0)
    from daq_backend import sample_pattern
    p = ScanParams(serpentine=True, **FAST)
    b, img = make(p, noise_v=0.0)
    run_scan(b, p, img, lambda: False)
    x, y = axes(p)
    X, Y = np.meshgrid(x, y)
    assert np.allclose(img, sample_pattern(X, Y), atol=1e-9)   # 每个像素放在正确位置


def test_stop_midway(monkeypatch):
    monkeypatch.setattr("scan_core.FIRST_POINT_EXTRA_MS", 0)
    p = ScanParams(**FAST)
    b, img = make(p)
    n = {"k": 0}

    def stop():
        n["k"] += 1
        return n["k"] > 100

    assert run_scan(b, p, img, stop) == "已停止"
    assert np.isfinite(img).sum() == 100
    assert b.writes[-1] == (0.0, 0.0) and b.close_count == 1


def test_error_still_parks_and_closes(monkeypatch):
    monkeypatch.setattr("scan_core.FIRST_POINT_EXTRA_MS", 0)

    class Broken(SimBackend):
        def _read(self):
            if len(self.writes) > 60:
                raise RuntimeError("模拟硬件故障")
            return super()._read()

    p = ScanParams(**FAST)
    x, y = axes(p)
    b = Broken(realistic_timing=False, record=True)
    with pytest.raises(RuntimeError):
        run_scan(b, p, np.zeros((len(y), len(x))), lambda: False)
    assert b.writes[-1] == (0.0, 0.0) and b.close_count == 1


def test_safety_limit_refuses_output():
    b = SimBackend(realistic_timing=False, record=True, safety_limit_v=1.0)
    b.open()
    with pytest.raises(ValueError):
        b.write_xy(1.5, 0)
    with pytest.raises(ValueError):
        b.ramp_to(0, -1.2)
    assert b.writes == []
    with pytest.raises(ValueError):
        SimBackend(realistic_timing=False, safety_limit_v=50).write_xy(10.5, 0)   # 硬限 10 V


def test_invalid_params_raise_before_output():
    p = ScanParams(x_step=0)
    b = SimBackend(realistic_timing=False, record=True)
    with pytest.raises(ValueError):
        run_scan(b, p, np.zeros((1, 1)), lambda: False)
    assert b.writes == [] and b.open_count == 0


def test_realistic_timing_settle(monkeypatch):
    """稳定时间确实被执行：20 点 × 2 ms ≥ 40 ms。"""
    import time
    monkeypatch.setattr("scan_core.FIRST_POINT_EXTRA_MS", 0)
    p = ScanParams(x_start=0, x_stop=0.9, x_step=0.1, y_start=0, y_stop=0.1, y_step=0.1,
                   settle_ms=2, row_wait_ms=2)
    b, img = make(p)
    t = time.perf_counter()
    run_scan(b, p, img, lambda: False)
    assert time.perf_counter() - t >= 0.040


def test_thread_stop_flag(monkeypatch):
    monkeypatch.setattr("scan_core.FIRST_POINT_EXTRA_MS", 0)
    p = ScanParams(settle_ms=1, row_wait_ms=1)
    b, img = make(p)
    ev = threading.Event()
    res = {}
    th = threading.Thread(target=lambda: res.setdefault("s", run_scan(b, p, img, ev.is_set)))
    th.start()
    ev.set()
    th.join(5)
    assert res["s"] == "已停止" and b.writes[-1] == (0.0, 0.0)
