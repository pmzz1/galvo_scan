import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_core import (MAX_POINTS, ScanParams, axes, estimate_seconds, load_csv, make_axis,  # noqa: E402
                       save_csv, scan_order, validate)


def test_axis_user_example():
    x = make_axis(-1, 1, 0.1)
    y = make_axis(-2, 2, 0.1)
    assert len(x) == 21 and len(y) == 41
    assert x[0] == pytest.approx(-1) and x[-1] == pytest.approx(1)
    assert y[0] == pytest.approx(-2) and y[-1] == pytest.approx(2)


def test_axis_not_divisible_never_exceeds_stop():
    x = make_axis(-1, 1, 0.3)
    assert len(x) == 7
    assert x[-1] == pytest.approx(0.8)
    assert x.max() <= 1 + 1e-12


def test_axis_reversed_and_degenerate():
    assert np.allclose(make_axis(1, -1, 0.5), make_axis(-1, 1, 0.5))
    assert len(make_axis(0.5, 0.5, 0.1)) == 1
    assert len(make_axis(0, 1, 5)) == 1
    with pytest.raises(ValueError):
        make_axis(0, 1, 0)


@pytest.mark.parametrize("step,n", [(0.01, 201), (0.05, 41), (0.2, 11), (0.001, 2001)])
def test_axis_float_robust(step, n):
    assert len(make_axis(-1, 1, step)) == n


def test_validate_ok_default():
    assert validate(ScanParams()) == []


def test_validate_errors():
    assert validate(ScanParams(x_step=0))
    assert validate(ScanParams(y_step=-0.1))
    assert validate(ScanParams(x_stop=10.5))
    assert validate(ScanParams(safety_limit_v=1.5))      # Y ±2 超出 1.5
    assert validate(ScanParams(safety_limit_v=12))
    assert validate(ScanParams(samples_per_point=1))
    assert validate(ScanParams(sample_rate=3e6))
    assert validate(ScanParams(settle_ms=-1))
    assert validate(ScanParams(x_start=-10, x_stop=10, y_start=-10, y_stop=10, x_step=0.01, y_step=0.01))
    assert validate(ScanParams(x_start=math.nan))


def test_scan_order_raster_and_serpentine():
    r = list(scan_order(3, 2, False))
    assert r == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    s = list(scan_order(3, 3, True))
    assert s[:3] == [(0, 0), (0, 1), (0, 2)]
    assert s[3:6] == [(1, 2), (1, 1), (1, 0)]
    assert s[6:] == [(2, 0), (2, 1), (2, 2)]
    for nx, ny in ((5, 4), (1, 7), (6, 1)):
        for serp in (False, True):
            seq = list(scan_order(nx, ny, serp))
            assert len(seq) == nx * ny and len(set(seq)) == nx * ny


def test_estimate_positive_and_scales():
    a = estimate_seconds(ScanParams())
    b = estimate_seconds(ScanParams(x_step=0.05))
    assert a > 0 and b > a * 1.8


def test_csv_roundtrip(tmp_path):
    p = ScanParams(x_step=0.5, y_step=1.0)
    x, y = axes(p)
    img = np.arange(len(x) * len(y), dtype=float).reshape(len(y), len(x)) / 10
    img[1, 2] = np.nan
    f = tmp_path / "a.csv"
    save_csv(f, img, x, y, {"测试": 1})
    img2, x2, y2 = load_csv(f)
    assert np.allclose(x2, x) and np.allclose(y2, y)
    assert np.allclose(img2, img, equal_nan=True)
    assert f.read_text(encoding="utf-8-sig").startswith("# 振镜扫描成像数据")


def test_max_points_constant():
    assert MAX_POINTS == 1_000_000


def test_normalize_basic():
    from scan_core import normalize
    img = np.array([[1.0, 0.5, np.nan], [0.2, 0.0, 0.3]])
    bg = np.array([[2.0, 1.0, 1.0], [0.001, 1.0, np.nan]])
    t = normalize(img, bg, dark=0.0, min_bg=0.002)
    assert t[0, 0] == pytest.approx(0.5) and t[0, 1] == pytest.approx(0.5)
    assert np.isnan(t[0, 2])          # 信号未扫描
    assert np.isnan(t[1, 0])          # 背景太弱
    assert t[1, 1] == 0.0
    assert np.isnan(t[1, 2])          # 背景缺失
    td = normalize(np.array([1.1]), np.array([2.1]), dark=0.1)
    assert td[0] == pytest.approx(0.5)


def test_same_grid():
    from scan_core import same_grid
    x = make_axis(-1, 1, 0.1); y = make_axis(-2, 2, 0.1)
    assert same_grid(x, y, x.copy(), y.copy())
    assert not same_grid(x, y, make_axis(-1, 1, 0.2), y)
    assert not same_grid(x, y, x + 0.05, y)
