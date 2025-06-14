# ======================================================================
#  hdr_reconstructor.py  --  LDR→HDR 交互式重建工具 (PyQt6)
# ======================================================================

import sys
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QPointF, QSizeF, QRectF, pyqtSignal, QEvent
from PyQt6.QtGui import (
    QPainter,
    QPen,
    QBrush,
    QColor,
    QPixmap,
    QImage,
    QAction,
    QPainterPath,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QFileDialog,
    QComboBox,
    QCheckBox,
)


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


class CurveWidget(QWidget):
    curveChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(256, 256)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.points = [QPointF(0.0, 0.0), QPointF(1.0, 1.0)]
        self.selected = self.dragging = -1
        self.padding = 18
        self.grid_pen = QPen(QColor(70, 70, 70), 1, Qt.PenStyle.DashLine)
        self.border_pen = QPen(QColor(90, 90, 90), 1)
        self.curve_pen = QPen(QColor(0, 170, 255), 2)
        self.point_brush = QBrush(QColor(0, 170, 255))
        self.sel_brush = QBrush(QColor(250, 120, 0))
        self.point_pen = QPen(QColor("white"), 1)

    def _segments(self):
        pts = sorted(self.points, key=lambda p: p.x())
        if len(pts) < 3:
            return []
        pad = [pts[0]] + pts + [pts[-1]]
        res = []
        for i in range(1, len(pad) - 2):
            p0, p1, p2, p3 = pad[i - 1 : i + 3]
            c1 = p1 + (p2 - p0) / 6.0
            c2 = p2 - (p3 - p1) / 6.0
            res.append((p1, c1, c2, p2))
        return res

    def _lut(self):
        pts = sorted(self.points, key=lambda p: p.x())
        if len(pts) < 2:
            return np.arange(256, dtype=np.uint8)
        segs = self._segments()
        if not segs:
            x = [p.x() for p in pts]
            y = [p.y() for p in pts]
            xs = np.linspace(0, 1, 256)
            ys = np.interp(xs, x, y)
        else:
            lx, ly = [], []
            for p1, c1, c2, p2 in segs:
                for i in range(101):
                    t = i / 100
                    u = 1 - t
                    p = u**3 * p1 + 3 * u**2 * t * c1 + 3 * u * t**2 * c2 + t**3 * p2
                    if not lx or p.x() > lx[-1]:
                        lx.append(p.x())
                        ly.append(p.y())
            xs = np.linspace(0, 1, 256)
            ys = np.interp(xs, lx, ly)
        return (ys * 255).clip(0, 255).astype(np.uint8)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(self.border_pen)
        p.setBrush(QColor(40, 40, 40))
        p.drawRect(self.rect())
        w, h = self.width() - 2 * self.padding, self.height() - 2 * self.padding
        p.translate(self.padding, self.padding)

        p.setPen(self.grid_pen)
        for i in range(1, 4):
            p.drawLine(0, i * h // 4, w, i * h // 4)
            p.drawLine(i * w // 4, 0, i * w // 4, h)

        pts = sorted(self.points, key=lambda p_: p_.x())
        scr = [QPointF(p_.x() * w, (1 - p_.y()) * h) for p_ in pts]
        path = QPainterPath()
        path.moveTo(scr[0])
        if len(pts) < 3:
            for s in scr[1:]:
                path.lineTo(s)
        else:
            for p1, c1, c2, p2 in self._segments():
                path.cubicTo(
                    QPointF(c1.x() * w, (1 - c1.y()) * h),
                    QPointF(c2.x() * w, (1 - c2.y()) * h),
                    QPointF(p2.x() * w, (1 - p2.y()) * h),
                )
        p.setPen(self.curve_pen)
        p.drawPath(path)

        for i, pp in enumerate(pts):
            px, py = pp.x() * w, (1 - pp.y()) * h
            p.setBrush(self.sel_brush if i == self.selected else self.point_brush)
            p.setPen(self.point_pen)
            r = 8 if i == self.selected else 5
            p.drawEllipse(QPointF(px, py), r, r)

    def _at(self, pos):
        w, h = self.width() - 2 * self.padding, self.height() - 2 * self.padding
        for i, p in enumerate(self.points):
            px = p.x() * w + self.padding
            py = (1 - p.y()) * h + self.padding
            if (pos - QPointF(px, py)).manhattanLength() < 12:
                return i
        return -1

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            idx = self._at(e.position())
            self.selected = idx
            if idx != -1:
                self.dragging = idx
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            self.update()

    def mouseMoveEvent(self, e):
        if self.dragging != -1:
            h = self.height() - 2 * self.padding
            y = 1 - max(0, min(h, e.position().y() - self.padding)) / h
            self.points[self.dragging].setY(y)
            self.curveChanged.emit()
            self.update()

    def mouseReleaseEvent(self, _):
        self.dragging = -1
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            w, h = self.width() - 2 * self.padding, self.height() - 2 * self.padding
            nx = max(0, min(w, e.position().x() - self.padding)) / w
            ny = 1 - max(0, min(h, e.position().y() - self.padding)) / h
            self.points.append(QPointF(nx, ny))
            self.selected = len(self.points) - 1
            self.curveChanged.emit()
            self.update()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.selected > 1:
            self.points.pop(self.selected)
            self.selected = -1
            self.curveChanged.emit()
            self.update()

    def lut(self):
        return self._lut()


class HDRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LDR → HDR 重建工具")
        self.setGeometry(100, 100, 1200, 700)
        self.pil = None
        self.hdr = None
        self.pixmap = None
        self.zoom = 1.0
        self.offset = QPointF()
        self.panning = False
        self.last = QPointF()
        self._ui()
        self.img_label.installEventFilter(self)

    def _ui(self):
        menu = self.menuBar().addMenu("文件")
        act = QAction("打开图片...", self)
        act.triggered.connect(self.open_file)
        menu.addAction(act)

        central = QWidget()
        root = QHBoxLayout(central)
        split = QSplitter(Qt.Orientation.Horizontal)

        self.img_label = QLabel("从“文件”菜单中加载一张图片")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("background:#2b2b2b;")
        self.img_label.setMinimumSize(400, 400)
        self.img_label.setCursor(Qt.CursorShape.OpenHandCursor)
        split.addWidget(self.img_label)

        ctrl_w = QWidget()
        ctrl_l = QVBoxLayout(ctrl_w)
        ctrl_w.setMinimumWidth(320)
        ctrl_w.setMaximumWidth(460)

        self.method_box = QComboBox()
        self.method_box.addItems(["手动曲线", "Inverse Reinhard", "Gamma 2.2"])
        self.view_box = QComboBox()
        self.view_box.addItems(["HDR (scaled)", "Tone-mapped"])
        self.false_color = QCheckBox("显示伪色彩")
        self.curve = CurveWidget()

        ctrl_l.addWidget(self.method_box)
        ctrl_l.addWidget(self.view_box)
        ctrl_l.addWidget(self.false_color)
        ctrl_l.addWidget(self.curve)
        split.addWidget(ctrl_w)
        split.setSizes([820, 380])
        root.addWidget(split)
        self.setCentralWidget(central)

        self.curve.curveChanged.connect(self.process)
        self.method_box.currentIndexChanged.connect(self.process)
        self.view_box.currentIndexChanged.connect(self.process)
        self.false_color.stateChanged.connect(self.process)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "", "图片文件 (*.png *.jpg *.jpeg *.tif *.bmp)"
        )
        if not path:
            return
        try:
            self.pil = Image.open(path).convert("RGB")
            self.fit_view()
            self.process()
        except Exception as e:
            self.img_label.setText(f"加载错误: {e}")

    def _ldr_to_hdr(self):
        if self.pil is None:
            return None
        ldr_srgb = np.asarray(self.pil, np.float32) / 255.0
        ldr_lin = srgb_to_linear(ldr_srgb)
        mode = self.method_box.currentText()
        if mode == "手动曲线":
            lut = self.curve.lut().astype(np.float32) / 255.0
            idx = (ldr_lin * 255).astype(np.uint8)
            hdr = lut[idx] * 5.0
        elif mode == "Inverse Reinhard":
            eps = 1e-4
            hdr = np.clip(ldr_lin, 0, 1 - eps) / (1 - np.clip(ldr_lin, 0, 1 - eps))
        else:
            hdr = np.power(ldr_srgb, 2.2)
        return hdr

    def _tone_map(self, hdr):
        return hdr / (1 + hdr)

    def _false_color(self, img):
        v = img.mean(axis=-1).flatten()
        r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
        return np.stack((r, g, b), 1).reshape(*img.shape)

    def _to_pixmap(self, rgb):
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def process(self):
        hdr = self._ldr_to_hdr()
        if hdr is None:
            return
        self.hdr = hdr

        vis_mode = self.view_box.currentText()
        if vis_mode.startswith("HDR"):
            mx = hdr.max() if hdr.max() > 0 else 1.0
            vis = (hdr / mx).clip(0, 1)
        else:
            vis = self._tone_map(hdr)

        if self.false_color.isChecked():
            vis_rgb = (self._false_color(vis) * 255).astype(np.uint8)
        else:
            vis_rgb = (vis * 255).clip(0, 255).astype(np.uint8)
        self.pixmap = self._to_pixmap(vis_rgb)
        self.update_view()

    def update_view(self):
        if self.pixmap is None:
            return
        canvas = QPixmap(self.img_label.size())
        canvas.fill(QColor("#2b2b2b"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        size = self.pixmap.size() * self.zoom
        pos = (
            QPointF(
                (self.img_label.width() - size.width()) / 2,
                (self.img_label.height() - size.height()) / 2,
            )
            + self.offset
        )
        painter.drawPixmap(
            QRectF(pos, QSizeF(size)), self.pixmap, QRectF(self.pixmap.rect())
        )
        painter.end()
        self.img_label.setPixmap(canvas)

    def fit_view(self):
        self.zoom = 1.0
        self.offset = QPointF()
        self.update_view()

    def eventFilter(self, src, ev):
        if src is self.img_label:
            if ev.type() == QEvent.Type.Wheel and self.pixmap:
                old = self.zoom
                self.zoom *= 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
                self.zoom = max(0.05, min(self.zoom, 50))
                mp = ev.position()
                cp = QPointF(self.img_label.rect().center())
                self.offset -= (mp - cp - self.offset) * (self.zoom / old - 1)
                self.update_view()
                return True
            if ev.type() == QEvent.Type.MouseButtonPress:
                if ev.button() == Qt.MouseButton.LeftButton:
                    self.panning = True
                    self.last = ev.position()
                    self.img_label.setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True
                if ev.button() == Qt.MouseButton.RightButton:
                    self.fit_view()
                    return True
            if ev.type() == QEvent.Type.MouseMove and self.panning:
                self.offset += ev.position() - self.last
                self.last = ev.position()
                self.update_view()
                return True
            if (
                ev.type() == QEvent.Type.MouseButtonRelease
                and ev.button() == Qt.MouseButton.LeftButton
            ):
                self.panning = False
                self.img_label.setCursor(Qt.CursorShape.OpenHandCursor)
                return True
        return super().eventFilter(src, ev)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update_view()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = HDRApp()
    win.show()
    sys.exit(app.exec())