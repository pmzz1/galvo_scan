"""扫描核心逻辑：参数、坐标轴、校验、扫描顺序、扫描主循环、数据保存。

本模块不依赖 Qt 和 NI 驱动，便于单元测试。
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Iterator

import numpy as np

HARD_LIMIT_V = 10.0        # 6363 AO 硬件量程 ±10 V
MAX_POINTS = 1_000_000     # 单幅图最大点数，防止步长填错
AI_MIN_V = 0.0             # PD 输出范围
AI_MAX_V = 2.0
SAT_LEVEL_V = 1.98         # 采样峰值 ≥ 此值视为饱和
DEG_PER_V = 2.0            # 振镜标定：1 V = 2°
FIRST_POINT_EXTRA_MS = 100.0  # 第一个点额外稳定时间
USB_OVERHEAD_MS = 1.5      # 每点 USB 通信开销经验值（用于估算时间）


@dataclass
class ScanParams:
    x_start: float = -1.0
    x_stop: float = 1.0
    x_step: float = 0.1
    y_start: float = -2.0
    y_stop: float = 2.0
    y_step: float = 0.1
    serpentine: bool = False
    settle_ms: float = 1.0
    row_wait_ms: float = 5.0
    samples_per_point: int = 10
    sample_rate: float = 100_000.0
    safety_limit_v: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- 坐标 / 校验
def make_axis(start: float, stop: float, step: float) -> np.ndarray:
    """从 min(start,stop) 到 max(start,stop)，按 step 生成坐标；最后一点不超过终止值。"""
    if not (step > 0) or not math.isfinite(step):
        raise ValueError("步长必须大于 0")
    lo, hi = sorted((float(start), float(stop)))
    n = int(math.floor((hi - lo) / step + 1e-6)) + 1
    return lo + np.arange(n, dtype=float) * step


def validate(p: ScanParams) -> list[str]:
    """返回错误信息列表；为空表示参数有效。"""
    errs: list[str] = []
    for name, v in (("X步长", p.x_step), ("Y步长", p.y_step)):
        if not (math.isfinite(v) and v > 0):
            errs.append(f"{name}必须大于 0")
    if not (0 < p.safety_limit_v <= HARD_LIMIT_V):
        errs.append(f"电压安全上限必须在 0~{HARD_LIMIT_V:g} V 之间")
    lim = min(p.safety_limit_v, HARD_LIMIT_V)
    for name, v in (("X起始", p.x_start), ("X终止", p.x_stop),
                    ("Y起始", p.y_start), ("Y终止", p.y_stop)):
        if not math.isfinite(v) or abs(v) > lim + 1e-12:
            errs.append(f"{name} = {v:g} V 超出安全上限 ±{lim:g} V")
    if p.samples_per_point < 2:
        errs.append("每点采样数至少为 2")
    if not (0 < p.sample_rate <= 2_000_000):
        errs.append("采样率必须在 0~2 MHz 之间")
    if p.settle_ms < 0 or p.row_wait_ms < 0:
        errs.append("等待时间不能为负")
    if not errs:
        nx = len(make_axis(p.x_start, p.x_stop, p.x_step))
        ny = len(make_axis(p.y_start, p.y_stop, p.y_step))
        if nx * ny > MAX_POINTS:
            errs.append(f"总点数 {nx}×{ny}={nx * ny} 超过上限 {MAX_POINTS}")
    return errs


def axes(p: ScanParams) -> tuple[np.ndarray, np.ndarray]:
    return (make_axis(p.x_start, p.x_stop, p.x_step),
            make_axis(p.y_start, p.y_stop, p.y_step))


def scan_order(nx: int, ny: int, serpentine: bool) -> Iterator[tuple[int, int]]:
    """按扫描顺序给出 (iy, ix)。蛇形模式下奇数行反向。"""
    for iy in range(ny):
        cols = range(nx - 1, -1, -1) if (serpentine and iy % 2 == 1) else range(nx)
        for ix in cols:
            yield iy, ix


def estimate_seconds(p: ScanParams) -> float:
    x, y = axes(p)
    per_point_ms = p.settle_ms + p.samples_per_point / p.sample_rate * 1000.0 + USB_OVERHEAD_MS
    return (len(x) * len(y) * per_point_ms + len(y) * p.row_wait_ms + FIRST_POINT_EXTRA_MS) / 1000.0


# ---------------------------------------------------------------- 计时
def precise_sleep(seconds: float) -> None:
    """亚毫秒精度等待：先 sleep，最后 1.5 ms 自旋。"""
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    if seconds > 0.002:
        time.sleep(seconds - 0.0015)
    while time.perf_counter() < end:
        pass


# ---------------------------------------------------------------- 扫描主循环
PointCallback = Callable[[int, int, float, float, float, float], None]


def run_scan(backend, p: ScanParams, img: np.ndarray,
             should_stop: Callable[[], bool],
             on_point: PointCallback | None = None) -> str:
    """执行一次扫描，结果写入 img[iy, ix]（形状 ny×nx）。

    backend 需提供 open / close / write_xy / ramp_to / read_point。
    无论正常结束、停止还是异常，最后都会斜坡回到 (0, 0) 并关闭任务。
    返回 "完成" 或 "已停止"；硬件错误以异常抛出。
    """
    errs = validate(p)
    if errs:
        raise ValueError("；".join(errs))
    x, y = axes(p)
    if img.shape != (len(y), len(x)):
        raise ValueError(f"图像数组形状应为 {(len(y), len(x))}")
    img[:] = np.nan

    status = "完成"
    backend.open()
    try:
        prev_iy = -1
        first = True
        for iy, ix in scan_order(len(x), len(y), p.serpentine):
            if should_stop():
                status = "已停止"
                break
            row_start = iy != prev_iy
            if first:
                backend.ramp_to(x[ix], y[iy])      # 从当前位置缓慢移到起点
            else:
                backend.write_xy(x[ix], y[iy])
            wait_ms = p.row_wait_ms if row_start else p.settle_ms
            if first:
                wait_ms += FIRST_POINT_EXTRA_MS
            precise_sleep(wait_ms / 1000.0)
            v, peak = backend.read_point()
            img[iy, ix] = v
            if on_point is not None:
                on_point(iy, ix, float(x[ix]), float(y[iy]), v, peak)
            prev_iy = iy
            first = False
    finally:
        try:
            backend.ramp_to(0.0, 0.0)
        finally:
            backend.close()
    return status


# ---------------------------------------------------------------- 数据保存
def save_csv(path, img: np.ndarray, x: np.ndarray, y: np.ndarray, meta: dict | None = None) -> None:
    """CSV 格式：# 开头的参数行；随后一行为 X 坐标；之后每行 = Y 坐标 + 该行光强。"""
    lines = [f"# 振镜扫描成像数据  保存时间: {datetime.now():%Y-%m-%d %H:%M:%S}"]
    for k, v in (meta or {}).items():
        lines.append(f"# {k}: {v}")
    lines.append("# 第一行为 X 电压(V)，第一列为 Y 电压(V)，其余为 PD 平均电压(V)；nan 表示未扫描")
    lines.append("Y\\X," + ",".join(f"{v:.6g}" for v in x))
    for iy, yv in enumerate(y):
        row = ",".join("nan" if np.isnan(v) else f"{v:.6f}" for v in img[iy])
        lines.append(f"{yv:.6g},{row}")
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def load_csv(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 save_csv 保存的文件，返回 (img, x, y)。"""
    with open(path, encoding="utf-8-sig") as f:
        rows = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    x = np.array([float(v) for v in rows[0].split(",")[1:]])
    y, data = [], []
    for ln in rows[1:]:
        parts = ln.split(",")
        y.append(float(parts[0]))
        data.append([float(v) for v in parts[1:]])
    return np.array(data), x, np.array(y)
