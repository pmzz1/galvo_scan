"""界面冒烟测试：无界面模式下用模拟后端完整扫描一次，并可保存截图。

设置环境变量 GALVO_SCREENSHOT=路径 时保存窗口截图。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pg = pytest.importorskip("pyqtgraph")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    pg.setConfigOptions(imageAxisOrder="row-major")
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


@pytest.fixture(autouse=True)
def slot_errors(monkeypatch):
    """Qt 槽函数中的异常会交给 sys.excepthook；收集起来，测试结束时断言为空。"""
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    yield errors
    assert not errors, f"槽函数异常: {errors[0][1]!r}"


def test_window_init_with_saved_settings(app, tmp_path, monkeypatch):
    """模拟真实启动：存在 settings.json 时初始化不能报错。"""
    import json
    import main_window
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"params": {"x_step": 0.2}, "sim": False, "x": "ao0"}), encoding="utf-8")
    monkeypatch.setattr(main_window, "SETTINGS_FILE", f)
    w = main_window.MainWindow()
    app.processEvents()
    assert w.sp_dx.value() == pytest.approx(0.2)
    w.close()


def _wait(app, cond, timeout=60):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


def test_points_update_while_typing(app, tmp_path, monkeypatch):
    """在输入框里打字（不按回车）时，点数和预计时间要立即更新。"""
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    import main_window
    monkeypatch.setattr(main_window, "SETTINGS_FILE", tmp_path / "settings.json")
    w = main_window.MainWindow()
    w.show()
    assert "X 21 × Y 41" in w.lb_points.text()
    eta0 = w.lb_eta.text()

    le = w.sp_x1.lineEdit()                       # X终止：1 → 3
    le.selectAll()
    QTest.keyClicks(le, "3")
    app.processEvents()
    assert "X 41 × Y 41" in w.lb_points.text()

    le = w.sp_dy.lineEdit()                       # Y步长：0.1 → 0.05
    le.selectAll()
    QTest.keyClicks(le, "0.05")
    app.processEvents()
    assert "X 41 × Y 81" in w.lb_points.text()
    assert w.lb_eta.text() != eta0
    w.close()


def test_gui_sim_scan(app, tmp_path, monkeypatch):
    import main_window
    monkeypatch.setattr(main_window, "SETTINGS_FILE", tmp_path / "settings.json")
    w = main_window.MainWindow()
    w.resize(1500, 900)
    w.show()
    w.cb_sim.setChecked(True)
    w.sp_x0.setValue(-1); w.sp_x1.setValue(1); w.sp_dx.setValue(0.05)
    w.sp_y0.setValue(-2); w.sp_y1.setValue(2); w.sp_dy.setValue(0.05)
    w.cb_mode.setCurrentIndex(1)
    w.sp_settle.setValue(0); w.sp_roww.setValue(0)
    assert "41 × Y 81" in w.lb_points.text()

    w.start_scan()
    assert w.thread is not None
    assert _wait(app, lambda: w.thread is None, 120), "扫描未在时限内完成"
    assert w.img.shape == (81, 41) and np.isfinite(w.img).all()
    assert w.last_pos == (0.0, 0.0)
    assert "完成" in w.lb_status.text()

    # 定位面板
    w.sp_mx.setValue(0.3); w.sp_my.setValue(0.6)
    w.move_manual()
    assert w.manual is not None and w.manual.position == (0.3, 0.6)
    assert "PD:" in w.lb_live.text()
    w.marker.setData([0.3], [0.6])

    # 保存
    csv = tmp_path / "t.csv"; png = tmp_path / "t.png"
    w.save_csv_dialog(str(csv)); w.save_png_dialog(str(png))
    assert csv.stat().st_size > 1000 and png.stat().st_size > 1000

    app.processEvents()
    shot = os.environ.get("GALVO_SCREENSHOT")
    if shot:
        w.grab().save(shot)

    # 回零并关闭
    w.go_zero()
    assert w.last_pos == (0.0, 0.0) and w.manual is None
    w.close()


def test_gui_stop_button(app, tmp_path, monkeypatch):
    import main_window
    monkeypatch.setattr(main_window, "SETTINGS_FILE", tmp_path / "settings.json")
    w = main_window.MainWindow()
    w.cb_sim.setChecked(True)
    w.sp_dx.setValue(0.01); w.sp_dy.setValue(0.01)      # 201×401 点，足够长
    w.start_scan()
    _wait(app, lambda: w.thread is not None and w.thread.count > 50, 30)
    w.stop_scan()
    assert _wait(app, lambda: w.thread is None, 30)
    assert "已停止" in w.lb_status.text()
    assert w.last_pos == (0.0, 0.0)
    w.close()
