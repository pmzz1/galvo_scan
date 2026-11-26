"""硬件后端：NI-DAQmx（USB-6363）和模拟后端。

所有输出电压都经过安全检查；ramp_to 用小步长缓慢移动振镜。
"""
from __future__ import annotations

import math

import numpy as np

from scan_core import AI_MAX_V, AI_MIN_V, HARD_LIMIT_V, precise_sleep

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TaskMode, TerminalConfiguration
    NIDAQMX_AVAILABLE = True
except Exception:  # pragma: no cover - 未安装 nidaqmx 时
    NIDAQMX_AVAILABLE = False

# 界面显示名 -> nidaqmx 枚举名
TERMINAL_CHOICES = {"RSE（单端）": "RSE", "差分": "DIFF", "NRSE": "NRSE", "默认": "DEFAULT"}


class Backend:
    ramp_step_v = 0.05      # 斜坡移动每步最大电压变化
    ramp_dwell_s = 0.0005   # 斜坡每步停留时间

    def __init__(self, safety_limit_v: float = 10.0, start_pos=(0.0, 0.0)):
        self.safety_limit_v = min(float(safety_limit_v), HARD_LIMIT_V)
        self.position = (float(start_pos[0]), float(start_pos[1]))
        self.is_open = False

    # ---- 生命周期
    def open(self):
        if not self.is_open:
            self._open()
            self.is_open = True

    def close(self):
        if self.is_open:
            try:
                self._close()
            finally:
                self.is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- 输出
    def _check(self, x: float, y: float, limit: float):
        for name, v in (("X", x), ("Y", y)):
            if not math.isfinite(v) or abs(v) > limit + 1e-9:
                raise ValueError(f"{name} 电压 {v:g} V 超出限制 ±{limit:g} V，已拒绝输出")

    def write_xy(self, x: float, y: float, _limit: float | None = None):
        x, y = float(x), float(y)
        self._check(x, y, self.safety_limit_v if _limit is None else _limit)
        self._write(x, y)
        self.position = (x, y)

    def ramp_to(self, x: float, y: float):
        """从当前位置按小步长直线移动到 (x, y)。目标点受安全上限约束。"""
        x, y = float(x), float(y)
        self._check(x, y, self.safety_limit_v)
        x0, y0 = self.position
        n = max(1, math.ceil(max(abs(x - x0), abs(y - y0)) / self.ramp_step_v))
        for k in range(1, n + 1):
            t = k / n
            # 中间点只检查硬件限值（起点可能在旧的安全上限之外）
            self.write_xy(x0 + (x - x0) * t, y0 + (y - y0) * t,
                          _limit=None if k == n else HARD_LIMIT_V)
            if k < n:
                precise_sleep(self.ramp_dwell_s)

    # ---- 输入
    def read_point(self) -> tuple[float, float]:
        """返回 (平均电压, 峰值电压)。"""
        return self._read()

    # ---- 子类实现
    def _open(self): ...
    def _close(self): ...
    def _write(self, x: float, y: float): ...
    def _read(self) -> tuple[float, float]: ...


class NidaqBackend(Backend):
    """USB-6363：一个两通道 AO 任务（X, Y）+ 一个有限采样 AI 任务（PD）。任务只创建一次。"""

    def __init__(self, x_chan: str, y_chan: str, pd_chan: str, terminal: str = "RSE",
                 samples_per_point: int = 10, sample_rate: float = 100_000.0,
                 safety_limit_v: float = 10.0, start_pos=(0.0, 0.0)):
        super().__init__(safety_limit_v, start_pos)
        self.x_chan, self.y_chan, self.pd_chan = x_chan, y_chan, pd_chan
        self.terminal = terminal
        self.n = int(samples_per_point)
        self.rate = float(sample_rate)
        self._ao = None
        self._ai = None

    def _open(self):
        if not NIDAQMX_AVAILABLE:
            raise RuntimeError("未安装 nidaqmx 库")
        try:
            self._ao = nidaqmx.Task()
            self._ao.ao_channels.add_ao_voltage_chan(self.x_chan, min_val=-HARD_LIMIT_V, max_val=HARD_LIMIT_V)
            self._ao.ao_channels.add_ao_voltage_chan(self.y_chan, min_val=-HARD_LIMIT_V, max_val=HARD_LIMIT_V)
            self._ao.start()  # 按需输出：每次 write 立即生效

            self._ai = nidaqmx.Task()
            self._ai.ai_channels.add_ai_voltage_chan(
                self.pd_chan, terminal_config=getattr(TerminalConfiguration, self.terminal),
                min_val=AI_MIN_V, max_val=AI_MAX_V)
            self._ai.timing.cfg_samp_clk_timing(self.rate, sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=self.n)
            self._ai.control(TaskMode.TASK_COMMIT)  # 提交后每点启停更快
        except Exception:
            self._close()
            raise

    def _close(self):
        for t in (self._ai, self._ao):
            if t is not None:
                try:
                    t.close()
                except Exception:
                    pass
        self._ai = self._ao = None

    def _write(self, x, y):
        self._ao.write([x, y])  # 两通道各 1 个采样

    def _read(self):
        self._ai.start()
        try:
            d = np.asarray(self._ai.read(number_of_samples_per_channel=self.n,
                                         timeout=self.n / self.rate + 2.0), dtype=float)
        finally:
            self._ai.stop()
        return float(d.mean()), float(d.max())


def sample_pattern(x, y):
    """模拟样品的透射光强（V）：亮背景 + 网格线 + 圆盘 + 一条斜边。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    edge = lambda d, w: 0.5 * (1 + np.tanh(d / w))           # 平滑边缘
    t = np.ones_like(x * y)
    gx = np.abs(x / 0.5 - np.round(x / 0.5)) * 0.5            # 到最近网格线的距离
    gy = np.abs(y / 0.5 - np.round(y / 0.5)) * 0.5
    t *= 1 - 0.7 * edge(0.025 - np.minimum(gx, gy), 0.008)
    r = np.hypot(x - 0.3, y - 0.6)
    t *= 1 - 0.5 * edge(0.45 - r, 0.03)
    t *= 1 - 0.3 * edge(-(x + y + 1.5), 0.05)
    return 1.7 * t


class SimBackend(Backend):
    """不接硬件时使用：返回模拟样品的光强并叠加噪声。"""

    def __init__(self, samples_per_point: int = 10, sample_rate: float = 100_000.0,
                 noise_v: float = 0.005, realistic_timing: bool = True, record: bool = False,
                 safety_limit_v: float = 10.0, start_pos=(0.0, 0.0), seed: int | None = None):
        super().__init__(safety_limit_v, start_pos)
        self.n = int(samples_per_point)
        self.rate = float(sample_rate)
        self.noise_v = noise_v
        self.realistic_timing = realistic_timing
        self.record = record
        self.writes: list[tuple[float, float]] = []
        self.open_count = 0
        self.close_count = 0
        self._rng = np.random.default_rng(seed)

    def _open(self):
        self.open_count += 1

    def _close(self):
        self.close_count += 1

    def _write(self, x, y):
        if self.record:
            self.writes.append((x, y))

    def _read(self):
        if self.realistic_timing:
            precise_sleep(self.n / self.rate + 0.0005)
        base = float(sample_pattern(*self.position))
        d = np.clip(base + self._rng.normal(0.0, self.noise_v, self.n), -0.1, 2.1)
        return float(d.mean()), float(d.max())


def list_devices() -> tuple[list[dict], str]:
    """枚举 NI-DAQmx 设备。返回 (设备列表, 提示信息)。"""
    if not NIDAQMX_AVAILABLE:
        return [], "未安装 nidaqmx 库"
    try:
        from nidaqmx.system import System
        devs = []
        for d in System.local().devices:
            short = lambda names: [n.split("/")[-1] for n in names]
            try:
                ao = short(d.ao_physical_chans.channel_names)
                ai = short(d.ai_physical_chans.channel_names)
            except Exception:
                ao, ai = [], []
            devs.append({"name": d.name, "product": d.product_type,
                         "simulated": bool(getattr(d, "is_simulated", False)), "ao": ao, "ai": ai})
        return devs, "" if devs else "未检测到 NI-DAQmx 设备"
    except Exception as e:
        return [], f"读取设备列表失败：{e}"
