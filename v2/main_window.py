"""主界面：参数面板 + 定位面板 + 实时强度图 + 数据表格。"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from daq_backend import TERMINAL_CHOICES, NidaqBackend, SimBackend, list_devices
from scan_core import (DEG_PER_V, HARD_LIMIT_V, SAT_LEVEL_V, ScanParams, axes,
                       estimate_seconds, load_csv, normalize, same_grid, save_csv, validate)
from scan_worker import ScanThread, format_error

APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "settings.json"
BG_FILE = APP_DIR / "background.npz"
DEFAULT_DATA_DIR = APP_DIR / "data"
RAW_LEVELS = (0.0, 2.0)
NORM_LEVELS = (0.0, 1.2)       # 相除（透过率）
SUB_LEVELS = (-2.0, 2.0)       # 相减（信号 − 背景）


def dspin(value, lo, hi, decimals=4, step=0.1, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    w.setSuffix(suffix)
    w.setKeyboardTracking(True)   # 边输入边更新点数和预计时间
    w.setMinimumWidth(95)
    return w


def ispin(value, lo, hi, suffix=""):
    w = QtWidgets.QSpinBox()
    w.setRange(lo, hi)
    w.setValue(value)
    w.setSuffix(suffix)
    w.setKeyboardTracking(True)   # 边输入边更新点数和预计时间
    w.setMinimumWidth(95)
    return w


class ArrayModel(QtCore.QAbstractTableModel):
    """数据表格：方向与强度图一致（最上一行为 Y 最大）。"""

    def __init__(self):
        super().__init__()
        self.img = np.zeros((0, 0))
        self.x = np.zeros(0)
        self.y = np.zeros(0)

    def set_array(self, img, x, y):
        self.beginResetModel()
        self.img, self.x, self.y = img, x, y
        self.endResetModel()

    def set_values(self, img):
        """更换显示的数值（形状不变时不重建表格）。"""
        if img.shape != self.img.shape:
            self.set_array(img, self.x, self.y)
        else:
            self.img = img
            self.refresh()

    def refresh(self):
        r, c = self.img.shape
        if r and c:
            self.dataChanged.emit(self.index(0, 0), self.index(r - 1, c - 1))

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.img.shape[0]

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.img.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and index.isValid():
            v = self.img[self.img.shape[0] - 1 - index.row(), index.column()]
            return "" if np.isnan(v) else f"{v:.4f}"
        if role == QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return f"X={self.x[section]:.4g}" if section < len(self.x) else None
        idx = len(self.y) - 1 - section
        return f"Y={self.y[idx]:.4g}" if 0 <= idx < len(self.y) else None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("振镜扫描成像  (USB-6363)")
        self.resize(1400, 860)

        self.thread: ScanThread | None = None
        self.scan_backend = None
        self.manual = None                 # 定位面板使用的后端会话
        self.last_pos = (0.0, 0.0)         # 振镜当前（最后写入）位置
        self.img = np.full((1, 1), np.nan)
        self.x = np.zeros(1)
        self.y = np.zeros(1)
        self.scan_params: ScanParams | None = None
        self.t_start = 0.0
        self.devices: list[dict] = []
        self.scan_meta: dict = {}
        # 背底（用于扣背底）：bg[iy, ix] 与坐标轴 bg_x / bg_y
        self.bg: np.ndarray | None = None
        self.bg_x = np.zeros(0)
        self.bg_y = np.zeros(0)
        self.bg_info = ""
        self._mode = None                  # 当前显示模式 "raw" / "sub"（相减）/ "div"（相除）
        self._levels = {"raw": RAW_LEVELS, "sub": SUB_LEVELS, "div": NORM_LEVELS}

        # 定时器先创建：构建界面/加载设置时可能触发需要它们的槽函数
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._on_timer)
        self.live_timer = QtCore.QTimer(self)
        self.live_timer.setInterval(200)
        self.live_timer.timeout.connect(self._live_read)

        self._build_ui()
        self._load_settings()
        self._load_background_file()
        if getattr(self, "_restore_norm", False) and self.bg is not None:
            self.cb_norm.blockSignals(True); self.cb_norm.setChecked(True); self.cb_norm.blockSignals(False)
        self._on_bgmode_changed()
        self.cb_sim_blank.setEnabled(self.cb_sim.isChecked())
        self.cb_sim.toggled.connect(self.cb_sim_blank.setEnabled)
        self.refresh_devices(initial=True)
        self._update_estimate()
        self._refresh_view()
        self.timer.start()

    # ================================================================ UI
    def _build_ui(self):
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)

        # ---- 硬件
        g = QtWidgets.QGroupBox("硬件")
        f = QtWidgets.QGridLayout(g)
        self.cb_sim = QtWidgets.QCheckBox("模拟模式（不接硬件）")
        self.cb_sim_blank = QtWidgets.QCheckBox("模拟：拿掉样品（扫背底）")
        self.cb_sim_blank.setToolTip("仅模拟模式有效，用于练习扣背底")
        self.cb_dev = QtWidgets.QComboBox()
        self.bt_refresh = QtWidgets.QPushButton("刷新")
        self.cb_x = QtWidgets.QComboBox(); self.cb_x.setEditable(True)
        self.cb_y = QtWidgets.QComboBox(); self.cb_y.setEditable(True)
        self.cb_pd = QtWidgets.QComboBox(); self.cb_pd.setEditable(True)
        self.cb_term = QtWidgets.QComboBox(); self.cb_term.addItems(list(TERMINAL_CHOICES))
        f.addWidget(self.cb_sim, 0, 0, 1, 2)
        f.addWidget(self.cb_sim_blank, 0, 2, 1, 2)
        f.addWidget(QtWidgets.QLabel("设备"), 1, 0); f.addWidget(self.cb_dev, 1, 1, 1, 2); f.addWidget(self.bt_refresh, 1, 3)
        f.addWidget(QtWidgets.QLabel("X 振镜"), 2, 0); f.addWidget(self.cb_x, 2, 1)
        f.addWidget(QtWidgets.QLabel("Y 振镜"), 2, 2); f.addWidget(self.cb_y, 2, 3)
        f.addWidget(QtWidgets.QLabel("PD 输入"), 3, 0); f.addWidget(self.cb_pd, 3, 1)
        f.addWidget(QtWidgets.QLabel("接线"), 3, 2); f.addWidget(self.cb_term, 3, 3)
        self.bt_refresh.clicked.connect(lambda: self.refresh_devices())
        self.cb_dev.currentIndexChanged.connect(self._fill_channels)
        self.cb_sim.toggled.connect(self._close_manual)
        lv.addWidget(g)

        # ---- 扫描范围
        g = QtWidgets.QGroupBox("扫描范围（电压）")
        f = QtWidgets.QGridLayout(g)
        self.sp_x0 = dspin(-1, -HARD_LIMIT_V, HARD_LIMIT_V, suffix=" V")
        self.sp_x1 = dspin(1, -HARD_LIMIT_V, HARD_LIMIT_V, suffix=" V")
        self.sp_dx = dspin(0.1, 0.0001, 20, step=0.01, suffix=" V")
        self.sp_y0 = dspin(-2, -HARD_LIMIT_V, HARD_LIMIT_V, suffix=" V")
        self.sp_y1 = dspin(2, -HARD_LIMIT_V, HARD_LIMIT_V, suffix=" V")
        self.sp_dy = dspin(0.1, 0.0001, 20, step=0.01, suffix=" V")
        for c, t in enumerate(("", "起始", "终止", "步长")):
            f.addWidget(QtWidgets.QLabel(t), 0, c)
        f.addWidget(QtWidgets.QLabel("X"), 1, 0)
        f.addWidget(self.sp_x0, 1, 1); f.addWidget(self.sp_x1, 1, 2); f.addWidget(self.sp_dx, 1, 3)
        f.addWidget(QtWidgets.QLabel("Y"), 2, 0)
        f.addWidget(self.sp_y0, 2, 1); f.addWidget(self.sp_y1, 2, 2); f.addWidget(self.sp_dy, 2, 3)
        self.cb_mode = QtWidgets.QComboBox(); self.cb_mode.addItems(["单向", "蛇形"])
        self.lb_angle = QtWidgets.QLabel()
        f.addWidget(QtWidgets.QLabel("扫描方式"), 3, 0); f.addWidget(self.cb_mode, 3, 1)
        f.addWidget(self.lb_angle, 3, 2, 1, 2)
        lv.addWidget(g)

        # ---- 时序与采样
        g = QtWidgets.QGroupBox("时序与采样")
        f = QtWidgets.QGridLayout(g)
        self.sp_settle = dspin(1, 0, 10000, decimals=2, step=0.5, suffix=" ms")
        self.sp_roww = dspin(5, 0, 10000, decimals=2, step=1, suffix=" ms")
        self.sp_ns = ispin(10, 2, 1_000_000)
        self.sp_rate = dspin(100000, 1, 2_000_000, decimals=0, step=10000, suffix=" Hz")
        self.sp_lim = dspin(10, 0.01, HARD_LIMIT_V, decimals=3, step=0.5, suffix=" V")
        f.addWidget(QtWidgets.QLabel("稳定时间"), 0, 0); f.addWidget(self.sp_settle, 0, 1)
        f.addWidget(QtWidgets.QLabel("换行等待"), 0, 2); f.addWidget(self.sp_roww, 0, 3)
        f.addWidget(QtWidgets.QLabel("每点采样数"), 1, 0); f.addWidget(self.sp_ns, 1, 1)
        f.addWidget(QtWidgets.QLabel("采样率"), 1, 2); f.addWidget(self.sp_rate, 1, 3)
        f.addWidget(QtWidgets.QLabel("电压安全上限"), 2, 0); f.addWidget(self.sp_lim, 2, 1)
        lv.addWidget(g)

        # ---- 定位面板
        g = QtWidgets.QGroupBox("定位（手动移动振镜 / 实时读 PD）")
        f = QtWidgets.QGridLayout(g)
        self.sp_mx = dspin(0, -HARD_LIMIT_V, HARD_LIMIT_V, step=0.01, suffix=" V")
        self.sp_my = dspin(0, -HARD_LIMIT_V, HARD_LIMIT_V, step=0.01, suffix=" V")
        self.bt_move = QtWidgets.QPushButton("移动到此点")
        self.cb_live = QtWidgets.QCheckBox("实时读数")
        self.lb_live = QtWidgets.QLabel("PD: -- V")
        self.lb_live.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.bt_use_cursor = QtWidgets.QPushButton("取图上点击位置")
        f.addWidget(QtWidgets.QLabel("X"), 0, 0); f.addWidget(self.sp_mx, 0, 1)
        f.addWidget(QtWidgets.QLabel("Y"), 0, 2); f.addWidget(self.sp_my, 0, 3)
        f.addWidget(self.bt_move, 1, 0, 1, 2); f.addWidget(self.cb_live, 1, 2); f.addWidget(self.bt_use_cursor, 1, 3)
        f.addWidget(self.lb_live, 2, 0, 1, 4)
        self.bt_move.clicked.connect(self.move_manual)
        self.cb_live.toggled.connect(self._toggle_live)
        self.bt_use_cursor.clicked.connect(self._use_clicked_point)
        lv.addWidget(g)

        # ---- 运行
        g = QtWidgets.QGroupBox("运行")
        f = QtWidgets.QGridLayout(g)
        self.bt_start = QtWidgets.QPushButton("开始扫描")
        self.bt_start.setStyleSheet("font-weight: bold; padding: 6px;")
        self.bt_stop = QtWidgets.QPushButton("停止"); self.bt_stop.setEnabled(False)
        self.bt_zero = QtWidgets.QPushButton("回零")
        self.bt_csv = QtWidgets.QPushButton("保存 CSV")
        self.bt_png = QtWidgets.QPushButton("保存 PNG")
        self.cb_auto = QtWidgets.QCheckBox("扫描结束自动保存")
        self.ed_dir = QtWidgets.QLineEdit(str(DEFAULT_DATA_DIR))
        self.bt_dir = QtWidgets.QPushButton("…"); self.bt_dir.setFixedWidth(30)
        f.addWidget(self.bt_start, 0, 0, 1, 2); f.addWidget(self.bt_stop, 0, 2); f.addWidget(self.bt_zero, 0, 3)
        f.addWidget(self.bt_csv, 1, 0, 1, 2); f.addWidget(self.bt_png, 1, 2, 1, 2)
        f.addWidget(self.cb_auto, 2, 0, 1, 4)
        dl = QtWidgets.QHBoxLayout(); dl.addWidget(QtWidgets.QLabel("保存文件夹")); dl.addWidget(self.ed_dir); dl.addWidget(self.bt_dir)
        f.addLayout(dl, 3, 0, 1, 4)
        self.bt_start.clicked.connect(self.start_scan)
        self.bt_stop.clicked.connect(self.stop_scan)
        self.bt_zero.clicked.connect(self.go_zero)
        self.bt_csv.clicked.connect(lambda: self.save_csv_dialog())
        self.bt_png.clicked.connect(lambda: self.save_png_dialog())
        self.bt_dir.clicked.connect(self._choose_dir)
        lv.addWidget(g)

        # ---- 扣背底
        g = QtWidgets.QGroupBox("扣背底")
        f = QtWidgets.QGridLayout(g)
        self.bt_bg_set = QtWidgets.QPushButton("当前图存为背底")
        self.bt_bg_load = QtWidgets.QPushButton("载入背底 CSV")
        self.bt_bg_clear = QtWidgets.QPushButton("清除背底")
        self.cb_norm = QtWidgets.QCheckBox("显示扣背底结果")
        self.cb_bgmode = QtWidgets.QComboBox()
        self.cb_bgmode.addItems(["相减：信号 − 背底", "相除：信号 ÷ 背底（透过率）"])
        self.cb_bgmode.setToolTip("相减：去掉固定的背景光/偏置；\n"
                                  "相除：同时消除 PD 在不同位置收光多少不一样的影响，结果为透过率")
        self.sp_dark = dspin(0.0, -1.0, 2.0, decimals=4, step=0.001, suffix=" V")
        self.bt_dark = QtWidgets.QPushButton("测暗电平（先挡光）")
        self.sp_bgmin = dspin(0.002, 0.0, 2.0, decimals=4, step=0.001, suffix=" V")
        self.sp_bgmin.setToolTip("相除时：背底扣暗后低于此值的点（PD 几乎收不到光）不计算，显示为空")
        self.lb_bg = QtWidgets.QLabel("背底：无"); self.lb_bg.setWordWrap(True)
        f.addWidget(self.bt_bg_set, 0, 0, 1, 2); f.addWidget(self.bt_bg_load, 0, 2); f.addWidget(self.bt_bg_clear, 0, 3)
        f.addWidget(self.cb_norm, 1, 0, 1, 2); f.addWidget(self.cb_bgmode, 1, 2, 1, 2)
        f.addWidget(QtWidgets.QLabel("暗电平"), 2, 0); f.addWidget(self.sp_dark, 2, 1); f.addWidget(self.bt_dark, 2, 2, 1, 2)
        f.addWidget(QtWidgets.QLabel("背底下限"), 3, 0); f.addWidget(self.sp_bgmin, 3, 1)
        f.addWidget(self.lb_bg, 4, 0, 1, 4)
        self.bt_bg_set.clicked.connect(self.set_background)
        self.bt_bg_load.clicked.connect(lambda: self.load_background_dialog())
        self.bt_bg_clear.clicked.connect(self.clear_background)
        self.cb_norm.toggled.connect(self._on_norm_toggled)
        self.cb_bgmode.currentIndexChanged.connect(self._on_bgmode_changed)
        self.bt_dark.clicked.connect(self.measure_dark)
        self.sp_dark.valueChanged.connect(self._refresh_view)
        self.sp_bgmin.valueChanged.connect(self._refresh_view)
        lv.addWidget(g)

        # ---- 状态
        g = QtWidgets.QGroupBox("状态")
        f = QtWidgets.QGridLayout(g)
        self.lb_points = QtWidgets.QLabel()
        self.lb_eta = QtWidgets.QLabel()
        self.pb = QtWidgets.QProgressBar(); self.pb.setRange(0, 1000); self.pb.setFormat("%p%")
        self.lb_cur = QtWidgets.QLabel("当前: --")
        self.lb_sat = QtWidgets.QLabel("● 未饱和"); self.lb_sat.setStyleSheet("color: green;")
        self.lb_status = QtWidgets.QLabel("就绪"); self.lb_status.setWordWrap(True)
        f.addWidget(self.lb_points, 0, 0, 1, 2); f.addWidget(self.lb_eta, 0, 2, 1, 2)
        f.addWidget(self.pb, 1, 0, 1, 4)
        f.addWidget(self.lb_cur, 2, 0, 1, 3); f.addWidget(self.lb_sat, 2, 3)
        f.addWidget(self.lb_status, 3, 0, 1, 4)
        lv.addWidget(g)
        lv.addStretch(1)

        scroll = QtWidgets.QScrollArea(); scroll.setWidget(left); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(470)

        # ---- 右侧：强度图 + 表格
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.plot = self.glw.addPlot()
        self.plot.setLabel("bottom", "X 电压", units="V")
        self.plot.setLabel("left", "Y 电压", units="V")
        self.plot.showGrid(x=False, y=False)
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        self.cbar = pg.ColorBarItem(values=(0.0, 2.0), colorMap=pg.colormap.get("inferno"),
                                    label="光强 (V)", interactive=True)
        self.cbar.setImageItem(self.img_item, insert_in=self.plot)
        self.marker = pg.ScatterPlotItem(size=12, symbol="+", pen=pg.mkPen("c", width=2), brush=None)
        self.plot.addItem(self.marker)
        self.click_marker = pg.ScatterPlotItem(size=10, symbol="o", pen=pg.mkPen("g", width=2), brush=None)
        self.plot.addItem(self.click_marker)
        self.clicked_xy = None
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_move)
        self.plot.scene().sigMouseClicked.connect(self._on_mouse_click)

        self.cb_autolevel = QtWidgets.QCheckBox("色标自动范围")
        self.cb_aspect = QtWidgets.QCheckBox("X/Y 等比例显示")
        self.cb_aspect.setChecked(True)
        self.cb_aspect.toggled.connect(self._apply_aspect)
        self.lb_hover = QtWidgets.QLabel("鼠标移到图上查看数值；单击选点，可用于“定位”")
        top = QtWidgets.QHBoxLayout(); top.addWidget(self.cb_autolevel); top.addWidget(self.cb_aspect)
        top.addStretch(1); top.addWidget(self.lb_hover)
        img_tab = QtWidgets.QWidget(); il = QtWidgets.QVBoxLayout(img_tab)
        il.addLayout(top); il.addWidget(self.glw)

        self.model = ArrayModel()
        self.table = QtWidgets.QTableView(); self.table.setModel(self.model)
        self.table.horizontalHeader().setDefaultSectionSize(72)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.tab_note = QtWidgets.QLabel()
        tbl_tab = QtWidgets.QWidget(); tl = QtWidgets.QVBoxLayout(tbl_tab)
        tl.addWidget(self.tab_note); tl.addWidget(self.table)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(img_tab, "强度图")
        self.tabs.addTab(tbl_tab, "数据表格")

        split = QtWidgets.QSplitter()
        split.addWidget(scroll); split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        for w in (self.sp_x0, self.sp_x1, self.sp_dx, self.sp_y0, self.sp_y1, self.sp_dy,
                  self.sp_settle, self.sp_roww, self.sp_ns, self.sp_rate, self.sp_lim):
            w.valueChanged.connect(self._update_estimate)
        for w in (self.sp_ns, self.sp_rate, self.sp_lim):
            w.valueChanged.connect(self._close_manual)
        for w in (self.cb_x, self.cb_y, self.cb_pd):
            w.currentTextChanged.connect(self._close_manual)
        self.cb_term.currentIndexChanged.connect(self._close_manual)

        self.param_widgets = [self.cb_sim, self.cb_dev, self.bt_refresh, self.cb_x, self.cb_y, self.cb_pd,
                              self.cb_term, self.sp_x0, self.sp_x1, self.sp_dx, self.sp_y0, self.sp_y1,
                              self.sp_dy, self.cb_mode, self.sp_settle, self.sp_roww, self.sp_ns,
                              self.sp_rate, self.sp_lim, self.bt_move, self.cb_live, self.bt_zero,
                              self.bt_start, self.bt_use_cursor, self.bt_bg_set, self.bt_bg_load,
                              self.bt_bg_clear, self.bt_dark]

    # ================================================================ 设备
    def refresh_devices(self, initial=False):
        keep = self.cb_dev.currentData() or getattr(self, "_saved_dev", None)
        self.devices, msg = list_devices()
        self.cb_dev.blockSignals(True)
        self.cb_dev.clear()
        for d in self.devices:
            tag = "（仿真）" if d["simulated"] else ""
            self.cb_dev.addItem(f'{d["name"]}  {d["product"]}{tag}', d["name"])
        self.cb_dev.blockSignals(False)
        if keep:
            i = self.cb_dev.findData(keep)
            if i >= 0:
                self.cb_dev.setCurrentIndex(i)
        self._fill_channels()
        if not self.devices:
            self.cb_sim.setChecked(True)
            self._set_status(f"{msg}，已切换到模拟模式。接好 USB-6363 后点“刷新”。")
        else:
            if initial or self.cb_sim.isChecked():
                self.cb_sim.setChecked(False)      # 检测到设备时默认使用真实硬件
            self._set_status(f"找到 {len(self.devices)} 个设备：" +
                             "，".join(f'{d["name"]}（{d["product"]}）' for d in self.devices))

    def _fill_channels(self):
        name = self.cb_dev.currentData()
        dev = next((d for d in self.devices if d["name"] == name), None)
        ao = dev["ao"] if dev else ["ao0", "ao1"]
        ai = dev["ai"] if dev else ["ai0"]
        for cb, items, want in ((self.cb_x, ao, getattr(self, "_saved_x", "ao0")),
                                (self.cb_y, ao, getattr(self, "_saved_y", "ao1")),
                                (self.cb_pd, ai, getattr(self, "_saved_pd", "ai0"))):
            cur = cb.currentText() or want
            cb.blockSignals(True)
            cb.clear(); cb.addItems(items)
            cb.setCurrentText(cur if cur else want)
            cb.blockSignals(False)

    def _full_chan(self, short: str) -> str:
        short = short.strip()
        if "/" in short:
            return short
        dev = self.cb_dev.currentData() or "Dev1"
        return f"{dev}/{short}"

    def _make_backend(self, p: ScanParams):
        if self.cb_sim.isChecked():
            return SimBackend(samples_per_point=p.samples_per_point, sample_rate=p.sample_rate,
                              safety_limit_v=p.safety_limit_v, start_pos=self.last_pos,
                              with_sample=not self.cb_sim_blank.isChecked())
        if not self.cb_dev.currentData():
            raise RuntimeError("没有可用的 NI 设备，请接好 USB-6363 后点“刷新”，或勾选模拟模式")
        return NidaqBackend(self._full_chan(self.cb_x.currentText()), self._full_chan(self.cb_y.currentText()),
                            self._full_chan(self.cb_pd.currentText()),
                            terminal=TERMINAL_CHOICES[self.cb_term.currentText()],
                            samples_per_point=p.samples_per_point, sample_rate=p.sample_rate,
                            safety_limit_v=p.safety_limit_v, start_pos=self.last_pos)

    # ================================================================ 参数
    def params(self) -> ScanParams:
        return ScanParams(
            x_start=self.sp_x0.value(), x_stop=self.sp_x1.value(), x_step=self.sp_dx.value(),
            y_start=self.sp_y0.value(), y_stop=self.sp_y1.value(), y_step=self.sp_dy.value(),
            serpentine=self.cb_mode.currentIndex() == 1,
            settle_ms=self.sp_settle.value(), row_wait_ms=self.sp_roww.value(),
            samples_per_point=self.sp_ns.value(), sample_rate=self.sp_rate.value(),
            safety_limit_v=self.sp_lim.value())

    def _update_estimate(self):
        p = self.params()
        errs = validate(p)
        lo_x, hi_x = sorted((p.x_start, p.x_stop)); lo_y, hi_y = sorted((p.y_start, p.y_stop))
        self.lb_angle.setText(f"角度 X {lo_x * DEG_PER_V:g}~{hi_x * DEG_PER_V:g}°  "
                              f"Y {lo_y * DEG_PER_V:g}~{hi_y * DEG_PER_V:g}°")
        if errs:
            self.lb_points.setText("<span style='color:red'>参数无效</span>")
            self.lb_eta.setText("")
            self.lb_points.setToolTip("\n".join(errs))
            return
        x, y = axes(p)
        self.lb_points.setToolTip("")
        self.lb_points.setText(f"点数 X {len(x)} × Y {len(y)} = {len(x) * len(y)}")
        self.lb_eta.setText(f"预计 {self._fmt_time(estimate_seconds(p))}")

    @staticmethod
    def _fmt_time(s: float) -> str:
        s = max(0.0, s)
        return f"{s:.1f} s" if s < 120 else f"{int(s // 60)} 分 {int(s % 60)} 秒"

    # ================================================================ 扫描
    def start_scan(self):
        if self.thread is not None:
            return
        p = self.params()
        errs = validate(p)
        if errs:
            QtWidgets.QMessageBox.warning(self, "参数无效", "\n".join(errs))
            return
        self._close_manual()
        try:
            backend = self._make_backend(p)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "无法开始", str(e))
            return
        self.x, self.y = axes(p)
        self.img = np.full((len(self.y), len(self.x)), np.nan)
        self.scan_params = p
        self.scan_meta = self._meta(p)
        self._setup_image()
        self.model.set_array(self.img, self.x, self.y)
        self._update_bg_label()

        self.scan_backend = backend
        self.thread = ScanThread(backend, p, self.img, self)
        self.thread.scan_done.connect(lambda s: self._on_scan_end(s, None))
        self.thread.scan_failed.connect(lambda m: self._on_scan_end("出错", m))
        self.t_start = time.perf_counter()
        self._set_running(True)
        self._set_status("扫描中…")
        self._save_settings()
        self.thread.start()

    def stop_scan(self):
        if self.thread is not None:
            self.thread.request_stop()
            self._set_status("正在停止…")

    def _on_scan_end(self, status: str, err: str | None):
        th = self.thread
        if th is None:
            return
        th.wait()
        self._on_timer()                   # 最后刷新一次
        self.last_pos = self.scan_backend.position
        elapsed = time.perf_counter() - self.t_start
        n = th.count
        sat = th.sat_count
        self.thread = None
        self.scan_backend = None
        self._set_running(False)
        self._refresh_view()
        self._update_bg_label()
        self.marker.setData([], [])
        msg = f"{status}：已扫描 {n}/{self.img.size} 点，用时 {self._fmt_time(elapsed)}"
        if sat:
            msg += f"，其中 {sat} 点饱和"
        self._set_status(msg)
        if n and self.cb_auto.isChecked():
            try:
                self._autosave()
            except Exception as e:
                self._set_status(msg + f"；自动保存失败：{e}")
        if err:
            self._set_status(f"扫描出错（振镜已回零）：{err}")
            QtWidgets.QMessageBox.critical(self, "扫描出错", err + "\n\n振镜已尝试回到 0 V。")

    def _set_running(self, running: bool):
        for w in self.param_widgets:
            w.setEnabled(not running)
        self.bt_stop.setEnabled(running)

    def _meta(self, p: ScanParams) -> dict:
        x, y = axes(p)
        m = {"模式": "模拟" if self.cb_sim.isChecked() else "NI 硬件",
             "设备": self.cb_dev.currentData() or "-",
             "X通道": self.cb_x.currentText(), "Y通道": self.cb_y.currentText(),
             "PD通道": self.cb_pd.currentText(), "接线": self.cb_term.currentText(),
             "X范围(V)": f"{x[0]:g} ~ {x[-1]:g}  步长 {p.x_step:g}  点数 {len(x)}",
             "Y范围(V)": f"{y[0]:g} ~ {y[-1]:g}  步长 {p.y_step:g}  点数 {len(y)}",
             "扫描方式": "蛇形" if p.serpentine else "单向",
             "稳定时间(ms)": p.settle_ms, "换行等待(ms)": p.row_wait_ms,
             "每点采样数": p.samples_per_point, "采样率(Hz)": p.sample_rate,
             "振镜标定": f"1 V = {DEG_PER_V:g} 度"}
        return m

    # ================================================================ 图像刷新
    def _setup_image(self):
        self.img_item.setImage(np.zeros_like(self.img), autoLevels=False)
        self._apply_rect()
        self.img_item.setLevels(self.cbar.levels())
        self._apply_aspect()
        self.plot.autoRange()

    def _apply_aspect(self, *_):
        # X、Y 标定相同（1 V = 2°），等比例显示时图像形状与实际扫描区域一致
        self.plot.setAspectLocked(self.cb_aspect.isChecked(), ratio=1.0)
        self.plot.autoRange()

    def _apply_rect(self):
        p = self.scan_params
        if p is None:
            return
        dx, dy = p.x_step, p.y_step      # 每个像素以坐标点为中心
        self.img_item.setRect(QtCore.QRectF(self.x[0] - dx / 2, self.y[0] - dy / 2,
                                            len(self.x) * dx, len(self.y) * dy))

    def _refresh_image(self):
        mode = self._bg_mode() if self._norm_active() else "raw"
        self._apply_mode(mode)
        data = self._display_array()
        lo, hi = self.cbar.levels()
        if self.cb_autolevel.isChecked() and np.isfinite(data).any():
            lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
            if hi - lo < 1e-6:
                hi = lo + 1e-3
            self.cbar.setLevels((lo, hi))
        disp = np.where(np.isnan(data), lo, data)
        self.img_item.setImage(disp, autoLevels=False)
        self._apply_rect()
        self.img_item.setLevels((lo, hi))
        return data

    def _refresh_view(self, *_):
        """刷新强度图和数据表格（扫描中由定时器调用，平时在切换显示时调用）。"""
        if self.scan_params is None:
            self._apply_mode(self._bg_mode() if self.cb_norm.isChecked() else "raw")
            return
        data = self._refresh_image()
        self.model.set_values(data)

    # ================================================================ 扣背底
    def _bg_mode(self) -> str:
        return "div" if self.cb_bgmode.currentIndex() == 1 else "sub"

    def _on_bgmode_changed(self, *_):
        div = self._bg_mode() == "div"
        for w in (self.sp_dark, self.bt_dark, self.sp_bgmin):
            w.setEnabled(div)                # 相减时暗电平互相抵消，不需要
        self._refresh_view()

    def _bg_matches(self) -> bool:
        return self.bg is not None and same_grid(self.bg_x, self.bg_y, self.x, self.y)

    def _norm_active(self) -> bool:
        return self.cb_norm.isChecked() and self.scan_params is not None and self._bg_matches()

    def _display_array(self) -> np.ndarray:
        if self._norm_active():
            if self._bg_mode() == "sub":
                return self.img - self.bg
            return normalize(self.img, self.bg, self.sp_dark.value(), self.sp_bgmin.value())
        return self.img

    def _apply_mode(self, mode: str):
        """切换色标范围、标签和表格说明。两种模式各自记住色标范围。"""
        if mode == self._mode:
            return
        if self._mode is not None:
            self._levels[self._mode] = tuple(self.cbar.levels())
        self._mode = mode
        self.cbar.setLevels(self._levels[mode])
        label = {"raw": "光强 (V)", "sub": "扣背底后光强 (V)", "div": "透过率"}[mode]
        self.cbar.getAxis("left").setLabel(label)
        unit = {"raw": "单位 V",
                "sub": "数值为 信号 − 背底，单位 V",
                "div": "数值为透过率（信号÷背底，已扣暗电平），空白表示背底太弱无法计算"}[mode]
        self.tab_note.setText(f"表格方向与强度图一致：最上一行为 Y 最大，列从左到右 X 增大。{unit}。")

    def _update_bg_label(self):
        if self.bg is None:
            self.lb_bg.setText("背底：无")
            return
        txt = (f"背底：{len(self.bg_x)}×{len(self.bg_y)} 点，X {self.bg_x[0]:g}~{self.bg_x[-1]:g} V，"
               f"Y {self.bg_y[0]:g}~{self.bg_y[-1]:g} V，最大 {np.nanmax(self.bg):.4f} V。{self.bg_info}")
        if self.cb_norm.isChecked() and self.scan_params is not None and not self._bg_matches():
            txt += "<br><span style='color:red'>当前图的扫描范围/步长与背底不一致，暂时显示原始光强。"
            txt += "请用相同参数重新扫背底。</span>"
        self.lb_bg.setText(txt)

    def _set_background(self, img, x, y, info: str):
        self.bg = np.array(img, dtype=float)
        self.bg_x = np.array(x, dtype=float)
        self.bg_y = np.array(y, dtype=float)
        self.bg_info = info
        try:
            np.savez(BG_FILE, bg=self.bg, x=self.bg_x, y=self.bg_y, info=np.array(info))
        except Exception as e:
            self._set_status(f"背底已设置，但保存到 {BG_FILE.name} 失败：{e}")
        self._update_bg_label()
        self._refresh_view()

    def set_background(self):
        if self.thread is not None:
            return
        if not self._has_data():
            QtWidgets.QMessageBox.information(self, "没有数据", "请先扫描一张不放样品的空白图，再点“当前图存为背底”。")
            return
        n_nan = int(np.isnan(self.img).sum())
        if n_nan:
            r = QtWidgets.QMessageBox.question(
                self, "背底不完整", f"当前图有 {n_nan} 个点未扫描，这些点将无法扣背底。仍然存为背底吗？")
            if r != QtWidgets.QMessageBox.Yes:
                return
        self._set_background(self.img, self.x, self.y, f"设置于 {datetime.now():%m-%d %H:%M}")
        self._set_status("已将当前图存为背底。放上样品、用相同参数扫描，勾选“显示扣背底结果”即可。")

    def load_background_dialog(self, path: str | None = None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择背底 CSV（本程序保存的原始数据）",
                                                            str(self._data_dir()), "CSV (*.csv)")
            if not path:
                return
        try:
            img, x, y = load_csv(path)
            if img.shape != (len(y), len(x)):
                raise ValueError("数据行列数与坐标不一致")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "载入失败", f"无法读取 {path}：{e}")
            return
        self._set_background(img, x, y, f"来自 {Path(path).name}")
        self._set_status(f"已载入背底 {path}")

    def clear_background(self):
        self.bg = None
        self.bg_x = self.bg_y = np.zeros(0)
        self.bg_info = ""
        try:
            BG_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        self._update_bg_label()
        self._refresh_view()

    def _load_background_file(self):
        try:
            d = np.load(BG_FILE, allow_pickle=False)
            self.bg, self.bg_x, self.bg_y = d["bg"], d["x"], d["y"]
            self.bg_info = str(d["info"]) + "（上次保存）"
        except Exception:
            self.bg = None
        self._update_bg_label()

    def _on_norm_toggled(self, on: bool):
        if on and self.bg is None:
            QtWidgets.QMessageBox.information(
                self, "还没有背底",
                "使用方法：\n1. 拿掉样品，扫描一张空白图；\n2. 点“当前图存为背底”；\n"
                "3. 放上样品，用相同的范围和步长扫描；\n4. 勾选“显示扣背底结果”。")
            self.cb_norm.blockSignals(True); self.cb_norm.setChecked(False); self.cb_norm.blockSignals(False)
            return
        self._update_bg_label()
        self._refresh_view()

    def measure_dark(self):
        """挡住激光后测 PD 暗电平：连续读 20 次取平均。"""
        if self.thread is not None:
            return
        try:
            b = self._ensure_manual()
            vals = [b.read_point()[0] for _ in range(20)]
            self.sp_dark.setValue(float(np.mean(vals)))
            self._set_status(f"暗电平 = {np.mean(vals):.5f} V（20 次平均，标准差 {np.std(vals) * 1000:.3f} mV）")
        except Exception as e:
            self._close_manual()
            QtWidgets.QMessageBox.critical(self, "测量失败", format_error(e))

    def _on_timer(self):
        th = self.thread
        if th is None:
            return
        data = self._refresh_image()
        total = self.img.size
        self.pb.setValue(int(1000 * th.count / total) if total else 0)
        elapsed = time.perf_counter() - self.t_start
        if th.count:
            remain = elapsed / th.count * (total - th.count)
            self.lb_eta.setText(f"已用 {self._fmt_time(elapsed)}，剩余约 {self._fmt_time(remain)}")
        if th.current:
            iy, ix, x, y, v, peak = th.current
            self.lb_cur.setText(f"当前 X={x:.4f} V  Y={y:.4f} V  PD={v:.4f} V")
            self.marker.setData([x], [y])
            self._show_sat(th.sat_count)
        if self.tabs.currentIndex() == 1:
            self.model.set_values(data)

    def _show_sat(self, count: int):
        if count:
            self.lb_sat.setText(f"● 饱和 {count} 点"); self.lb_sat.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.lb_sat.setText("● 未饱和"); self.lb_sat.setStyleSheet("color: green;")

    # ================================================================ 鼠标
    def _pixel_at(self, scene_pos):
        if self.scan_params is None or not self.plot.sceneBoundingRect().contains(scene_pos):
            return None
        mp = self.plot.vb.mapSceneToView(scene_pos)
        p = self.scan_params
        ix = int(round((mp.x() - self.x[0]) / p.x_step))
        iy = int(round((mp.y() - self.y[0]) / p.y_step))
        if 0 <= ix < len(self.x) and 0 <= iy < len(self.y):
            return iy, ix
        return None

    def _on_mouse_move(self, pos):
        px = self._pixel_at(pos)
        if px is None:
            return
        iy, ix = px
        v = self.img[iy, ix]
        vs = "未扫描" if np.isnan(v) else f"{v:.4f} V"
        txt = f"X={self.x[ix]:.4f} V  Y={self.y[iy]:.4f} V  光强={vs}"
        if self._norm_active():
            r = self._display_array()[iy, ix]
            rs = "--" if np.isnan(r) else (f"{r:.4f} V" if self._bg_mode() == "sub" else f"{r:.3f}")
            name = "扣背底后" if self._bg_mode() == "sub" else "透过率"
            txt += f"  背底={self.bg[iy, ix]:.4f} V  {name}={rs}"
        self.lb_hover.setText(txt)

    def _on_mouse_click(self, ev):
        px = self._pixel_at(ev.scenePos())
        if px is None:
            return
        iy, ix = px
        self.clicked_xy = (float(self.x[ix]), float(self.y[iy]))
        self.click_marker.setData([self.clicked_xy[0]], [self.clicked_xy[1]])

    def _use_clicked_point(self):
        if self.clicked_xy is None:
            self._set_status("请先在强度图上单击一个点")
            return
        self.sp_mx.setValue(self.clicked_xy[0])
        self.sp_my.setValue(self.clicked_xy[1])

    # ================================================================ 定位面板
    def _ensure_manual(self):
        if self.manual is None:
            p = self.params()
            b = self._make_backend(p)
            b.open()
            self.manual = b
        return self.manual

    def _close_manual(self, *_):
        self.live_timer.stop()
        if self.cb_live.isChecked():
            self.cb_live.blockSignals(True); self.cb_live.setChecked(False); self.cb_live.blockSignals(False)
        if self.manual is not None:
            self.last_pos = self.manual.position
            try:
                self.manual.close()
            except Exception:
                pass
            self.manual = None

    def move_manual(self):
        if self.thread is not None:
            return
        x, y = self.sp_mx.value(), self.sp_my.value()
        lim = self.sp_lim.value()
        if abs(x) > lim or abs(y) > lim:
            QtWidgets.QMessageBox.warning(self, "超出安全上限", f"目标电压超出安全上限 ±{lim:g} V")
            return
        try:
            b = self._ensure_manual()
            b.ramp_to(x, y)
            self.last_pos = b.position
            v, peak = b.read_point()
            self._show_live(v, peak)
            self.marker.setData([x], [y])
            self._set_status(f"振镜已移动到 X={x:g} V, Y={y:g} V（{x * DEG_PER_V:g}°, {y * DEG_PER_V:g}°）")
        except Exception as e:
            self._close_manual()
            QtWidgets.QMessageBox.critical(self, "移动失败", format_error(e))

    def _toggle_live(self, on: bool):
        if on:
            try:
                self._ensure_manual()
            except Exception as e:
                self.cb_live.blockSignals(True); self.cb_live.setChecked(False); self.cb_live.blockSignals(False)
                QtWidgets.QMessageBox.critical(self, "无法读取", format_error(e))
                return
            self.live_timer.start()
        else:
            self.live_timer.stop()

    def _live_read(self):
        if self.manual is None:
            self.live_timer.stop()
            return
        try:
            v, peak = self.manual.read_point()
            self._show_live(v, peak)
        except Exception as e:
            self._close_manual()
            self._set_status(f"实时读数出错：{format_error(e)}")

    def _show_live(self, v, peak):
        color = "red" if peak >= SAT_LEVEL_V else "black"
        self.lb_live.setText(f"<span style='color:{color}'>PD: {v:.4f} V（峰值 {peak:.4f} V）"
                             f"{'  饱和!' if peak >= SAT_LEVEL_V else ''}</span>")

    def go_zero(self):
        if self.thread is not None:
            return
        try:
            b = self._ensure_manual()
            b.ramp_to(0.0, 0.0)
            self.last_pos = (0.0, 0.0)
            self.sp_mx.setValue(0.0); self.sp_my.setValue(0.0)
            self.marker.setData([0.0], [0.0])
            self._set_status("振镜已回零")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "回零失败", format_error(e))
        finally:
            self._close_manual()

    # ================================================================ 保存
    def _has_data(self):
        return self.scan_params is not None and np.isfinite(self.img).any()

    def _timestamp_name(self):
        return f"扫描_{datetime.now():%Y%m%d_%H%M%S}"

    def _data_dir(self) -> Path:
        d = Path(self.ed_dir.text().strip() or DEFAULT_DATA_DIR)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_data(self, path):
        """保存原始光强 CSV；若当前显示扣背底结果，另存一份 *_扣背底.csv。返回保存的文件列表。"""
        path = Path(path)
        save_csv(path, self.img, self.x, self.y, self.scan_meta)
        files = [path]
        if self._norm_active():
            meta = dict(self.scan_meta)
            if self._bg_mode() == "sub":
                meta.update({"数据类型": "扣背底（相减）= 信号 - 背底，单位 V", "背底": self.bg_info})
            else:
                meta.update({"数据类型": "扣背底（相除）= 透过率 = (信号-暗电平)/(背底-暗电平)", "背底": self.bg_info,
                             "暗电平(V)": self.sp_dark.value(), "背底下限(V)": self.sp_bgmin.value()})
            npath = path.with_name(path.stem + "_扣背底.csv")
            save_csv(npath, self._display_array(), self.x, self.y, meta)
            files.append(npath)
        return files

    def _autosave(self):
        base = self._data_dir() / self._timestamp_name()
        files = self._save_data(base.with_suffix(".csv"))
        self.save_png(base.with_suffix(".png"))
        self._set_status(self.lb_status.text() + f"；已自动保存 {len(files)} 个 CSV 和 PNG 到 {base.parent}")

    def save_csv_dialog(self, path: str | None = None):
        if not self._has_data():
            self._set_status("没有可保存的数据")
            return
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "保存 CSV", str(self._data_dir() / (self._timestamp_name() + ".csv")), "CSV (*.csv)")
            if not path:
                return
        files = self._save_data(path)
        self._set_status("已保存 " + "，".join(str(f) for f in files))

    def save_png(self, path):
        self.glw.grab().save(str(path))

    def save_png_dialog(self, path: str | None = None):
        if not self._has_data():
            self._set_status("没有可保存的图像")
            return
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "保存 PNG", str(self._data_dir() / (self._timestamp_name() + ".png")), "PNG (*.png)")
            if not path:
                return
        self.save_png(path)
        self._set_status(f"已保存 {path}")

    def _choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择保存文件夹", self.ed_dir.text())
        if d:
            self.ed_dir.setText(d)

    # ================================================================ 设置持久化
    def _save_settings(self):
        s = {"params": self.params().to_dict(), "sim": self.cb_sim.isChecked(),
             "device": self.cb_dev.currentData(), "x": self.cb_x.currentText(), "y": self.cb_y.currentText(),
             "pd": self.cb_pd.currentText(), "term": self.cb_term.currentText(),
             "auto": self.cb_auto.isChecked(), "dir": self.ed_dir.text(),
             "autolevel": self.cb_autolevel.isChecked(), "last_pos": list(self.last_pos),
             "norm": self.cb_norm.isChecked(), "dark": self.sp_dark.value(), "bg_min": self.sp_bgmin.value(),
             "bg_mode": self.cb_bgmode.currentIndex()}
        try:
            SETTINGS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_settings(self):
        try:
            s = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        p = s.get("params", {})
        for key, w in (("x_start", self.sp_x0), ("x_stop", self.sp_x1), ("x_step", self.sp_dx),
                       ("y_start", self.sp_y0), ("y_stop", self.sp_y1), ("y_step", self.sp_dy),
                       ("settle_ms", self.sp_settle), ("row_wait_ms", self.sp_roww),
                       ("samples_per_point", self.sp_ns), ("sample_rate", self.sp_rate),
                       ("safety_limit_v", self.sp_lim)):
            if key in p:
                w.setValue(p[key])
        self.cb_mode.setCurrentIndex(1 if p.get("serpentine") else 0)
        self._saved_dev = s.get("device")
        self._saved_x, self._saved_y, self._saved_pd = s.get("x", "ao0"), s.get("y", "ao1"), s.get("pd", "ai0")
        self.cb_x.setCurrentText(self._saved_x); self.cb_y.setCurrentText(self._saved_y)
        self.cb_pd.setCurrentText(self._saved_pd)
        if s.get("term") in TERMINAL_CHOICES:
            self.cb_term.setCurrentText(s["term"])
        self.cb_auto.setChecked(bool(s.get("auto", False)))
        if s.get("dir"):
            self.ed_dir.setText(s["dir"])
        self.cb_autolevel.setChecked(bool(s.get("autolevel", False)))
        lp = s.get("last_pos")
        if isinstance(lp, list) and len(lp) == 2:
            self.last_pos = (float(lp[0]), float(lp[1]))
        if "dark" in s:
            self.sp_dark.setValue(float(s["dark"]))
        if "bg_min" in s:
            self.sp_bgmin.setValue(float(s["bg_min"]))
        self._restore_norm = bool(s.get("norm", False))
        if s.get("bg_mode") in (0, 1):
            self.cb_bgmode.setCurrentIndex(int(s["bg_mode"]))

    # ================================================================ 其他
    def _set_status(self, text: str):
        self.lb_status.setText(text)

    def closeEvent(self, ev: QtGui.QCloseEvent):
        if self.thread is not None:
            self.thread.request_stop()
            self.thread.wait(10000)       # run_scan 的 finally 会回零并释放任务
            if self.scan_backend is not None:
                self.last_pos = self.scan_backend.position
            self.thread = None
        if self.manual is not None:
            try:
                self.manual.ramp_to(0.0, 0.0)
            except Exception:
                pass
            self._close_manual()
        self._save_settings()
        super().closeEvent(ev)
