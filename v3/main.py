"""振镜扫描成像程序入口。双击 运行.bat 启动。"""
import sys
import traceback
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))


def _excepthook(exc_type, exc, tb):
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        with open(APP_DIR / "error.log", "a", encoding="utf-8") as f:
            f.write(f"\n==== {datetime.now():%Y-%m-%d %H:%M:%S}\n{text}")
    except Exception:
        pass
    try:
        from PySide6 import QtWidgets
        if QtWidgets.QApplication.instance():
            QtWidgets.QMessageBox.critical(None, "程序错误", text[-2000:])
    except Exception:
        pass


def main():
    sys.excepthook = _excepthook
    import pyqtgraph as pg
    from PySide6 import QtWidgets
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    from main_window import MainWindow
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
