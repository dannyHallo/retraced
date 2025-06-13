# =================================================================
#  scanner_advanced.py -- 高级交互式色调曲线图像查看器 (PyQt6)
# =================================================================
#  - [功能升级] 色调曲线从折线升级为平滑的贝塞尔曲线，调整更自然。
#  - [交互优化] 图像预览区平移操作从鼠标中键更改为更常用的左键拖动。
#  - [修复] 彻底解决在缩放视图时因坐标类型不匹配导致的 TypeError。
#  - [优化] 曲线控制点（包括新增点）现在只能垂直拖动，操作更直观。
#  - [改进] 打开图片时默认缩放以完整显示 (Fit to View)。
#  - [功能] 支持动态添加/删除曲线控制点 (双击添加，Delete键删除)。
#  - [功能] 图像预览区支持滚轮缩放、左键平移、右键重置视图。
# =================================================================

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QCheckBox, QFileDialog, QLabel, QSplitter
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QAction, QPainterPath, QCursor
)
from PyQt6.QtCore import (
    Qt, QPointF, pyqtSignal, QEvent
)
from PIL import Image
import numpy as np

# ----------------------------------------------------------------------
#  支持动态增删节点和贝塞尔曲线的高级曲线控件
# ----------------------------------------------------------------------
class CurveWidget(QWidget):
    curveChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(256, 256)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.points = [QPointF(0.0, 0.0), QPointF(1.0, 1.0)]
        
        self.selected_point_index = -1
        self.dragging_point_index = -1
        self.padding = 20

        self.grid_pen = QPen(QColor(60, 60, 60), 1, Qt.PenStyle.DashLine)
        self.border_pen = QPen(QColor(80, 80, 80), 1)
        self.curve_pen = QPen(QColor(0, 150, 255), 2)
        self.point_brush = QBrush(QColor(0, 150, 255))
        self.point_pen = QPen(QColor("white"), 1)
        self.selected_point_brush = QBrush(QColor(255, 100, 0))

    def _calculate_bezier_segments(self):
        """
        Calculates the Bezier curve segments based on the control points.
        Uses Catmull-Rom spline algorithm to ensure the curve passes through the points.
        Returns a list of tuples, where each tuple is (start_point, control_point1, control_point2, end_point).
        """
        sorted_points = sorted(self.points, key=lambda p: p.x())
        if len(sorted_points) < 3:
            return [] # Not enough points for a Catmull-Rom spline

        segments = []
        # Pad points for boundary conditions to calculate tangents at endpoints
        padded = [sorted_points[0]] + sorted_points + [sorted_points[-1]]

        for i in range(1, len(padded) - 2):
            p0 = padded[i-1]
            p1 = padded[i]
            p2 = padded[i+1]
            p3 = padded[i+2]

            # Convert Catmull-Rom points to Bezier control points.
            # The tension is 0.5, which corresponds to a divisor of 6.0.
            c1 = p1 + (p2 - p0) / 6.0
            c2 = p2 - (p3 - p1) / 6.0
            segments.append((p1, c1, c2, p2))
            
        return segments

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(self.border_pen)
        painter.setBrush(QColor(35, 35, 35))
        painter.drawRect(self.rect())
        
        painter.save()
        w = self.width() - 2 * self.padding
        h = self.height() - 2 * self.padding
        painter.translate(self.padding, self.padding)
        
        # Draw grid
        painter.setPen(self.grid_pen)
        for i in range(1, 4):
            painter.drawLine(0, i * h // 4, w, i * h // 4)
            painter.drawLine(i * w // 4, 0, i * w // 4, h)
        
        painter.setPen(self.curve_pen)
        path = QPainterPath()
        sorted_points = sorted(self.points, key=lambda p: p.x())
        screen_points = [QPointF(p.x() * w, (1 - p.y()) * h) for p in sorted_points]

        # --- [MODIFIED] Draw Bezier curve or straight lines ---
        if len(sorted_points) < 3:
            # Draw straight lines if less than 3 points
            if screen_points:
                path.moveTo(screen_points[0])
                for i in range(1, len(screen_points)):
                    path.lineTo(screen_points[i])
        else:
            # Draw a smooth Bezier curve
            segments = self._calculate_bezier_segments()
            if segments:
                p_start, _, _, _ = segments[0]
                path.moveTo(QPointF(p_start.x() * w, (1 - p_start.y()) * h))
                for p1, c1, c2, p2 in segments:
                    sc1 = QPointF(c1.x() * w, (1 - c1.y()) * h)
                    sc2 = QPointF(c2.x() * w, (1 - c2.y()) * h)
                    sp2 = QPointF(p2.x() * w, (1 - p2.y()) * h)
                    path.cubicTo(sc1, sc2, sp2)
        
        painter.drawPath(path)
        
        # Draw control points
        for i, p in enumerate(self.points):
            px = p.x() * w
            py = (1 - p.y()) * h
            if i == self.selected_point_index:
                painter.setBrush(self.selected_point_brush)
                painter.setPen(self.point_pen)
                painter.drawEllipse(QPointF(px, py), 8, 8)
            else:
                painter.setBrush(self.point_brush)
                painter.setPen(self.point_pen)
                painter.drawEllipse(QPointF(px, py), 5, 5)
        painter.restore()

    def _get_point_at_pos(self, pos):
        w = self.width() - 2 * self.padding
        h = self.height() - 2 * self.padding
        for i, p in enumerate(self.points):
            px = p.x() * w + self.padding
            py = (1 - p.y()) * h + self.padding
            if (pos - QPointF(px, py)).manhattanLength() < 12:
                return i
        return -1

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point_index = self._get_point_at_pos(event.position())
            self.selected_point_index = point_index
            if point_index != -1:
                self.dragging_point_index = point_index
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            self.update()

    def mouseMoveEvent(self, event):
        if self.dragging_point_index != -1:
            h = self.height() - 2 * self.padding
            norm_y = 1.0 - max(0, min(h, event.position().y() - self.padding)) / h
            self.points[self.dragging_point_index].setY(norm_y)
            self.update()
            self.curveChanged.emit()

    def mouseReleaseEvent(self, event):
        self.dragging_point_index = -1
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            w = self.width() - 2 * self.padding
            h = self.height() - 2 * self.padding
            norm_x = max(0, min(w, event.position().x() - self.padding)) / w
            norm_y = 1.0 - max(0, min(h, event.position().y() - self.padding)) / h
            new_point = QPointF(norm_x, norm_y)
            self.points.append(new_point)
            self.selected_point_index = len(self.points) - 1
            self.update()
            self.curveChanged.emit()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.selected_point_index > 1:
                del self.points[self.selected_point_index]
                self.selected_point_index = -1
                self.update()
                self.curveChanged.emit()

    def get_lut(self) -> np.ndarray:
        """
        Generates a 256-entry Look-Up Table (LUT) from the curve.
        If the curve is a Bezier, it samples the curve at high resolution
        to create an accurate LUT. Otherwise, it uses linear interpolation.
        """
        sorted_points = sorted(self.points, key=lambda p: p.x())
        if len(sorted_points) < 2:
            return np.arange(256, dtype=np.uint8)

        # --- [MODIFIED] Generate LUT from Bezier or linear curve ---
        segments = self._calculate_bezier_segments()
        if not segments: # Use linear interpolation for 2 points
            x_coords = [p.x() for p in sorted_points]
            y_coords = [p.y() for p in sorted_points]
            input_values = np.linspace(0.0, 1.0, 256)
            output_values = np.interp(input_values, x_coords, y_coords)
        else: # Sample the Bezier curve for higher accuracy
            fine_x, fine_y = [], []
            for p1, c1, c2, p2 in segments:
                # Sample each segment. 100 steps is a good balance of accuracy and performance.
                for i in range(101):
                    t = i / 100.0
                    inv_t = 1.0 - t
                    # The Bezier formula
                    pt = (inv_t**3 * p1) + \
                         (3 * inv_t**2 * t * c1) + \
                         (3 * inv_t * t**2 * c2) + \
                         (t**3 * p2)
                    
                    # Avoid adding duplicate points at segment boundaries
                    if not fine_x or pt.x() > fine_x[-1]:
                         fine_x.append(pt.x())
                         fine_y.append(pt.y())

            input_values = np.linspace(0.0, 1.0, 256)
            output_values = np.interp(input_values, fine_x, fine_y)

        lut = (output_values * 255).clip(0, 255).astype(np.uint8)
        return lut

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高级图像扫描仪")
        self.setGeometry(100, 100, 1200, 700)
        self.pil_image = None
        self.processed_pixmap = None
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.is_panning = False
        self.last_pan_pos = QPointF(0, 0)
        self.init_ui()
        self.image_label.installEventFilter(self)

    def init_ui(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("文件")
        open_action = QAction("打开图片...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.image_label = QLabel("请从“文件”菜单打开一张图片")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background-color: #2b2b2b;")
        # --- [MODIFIED] Set cursor to indicate panning is possible ---
        self.image_label.setCursor(Qt.CursorShape.OpenHandCursor)
        splitter.addWidget(self.image_label)
        
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_widget.setMinimumWidth(300)
        controls_widget.setMaximumWidth(450)
        
        self.invert_check = QCheckBox("反相 (Invert)")
        self.curve_widget = CurveWidget()
        
        controls_layout.addWidget(self.invert_check)
        controls_layout.addWidget(self.curve_widget)
        splitter.addWidget(controls_widget)
        splitter.setSizes([800, 400])
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)
        
        self.invert_check.stateChanged.connect(self.process_image)
        self.curve_widget.curveChanged.connect(self.process_image)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "图片文件 (*.png *.jpg *.jpeg *.tif *.bmp)")
        if path:
            try:
                self.pil_image = Image.open(path).convert("L")
                self.process_image()
                self.fit_to_view()
            except Exception as e:
                print(f"打开文件失败: {e}")
                self.image_label.setText("无法加载此图片。")

    def process_image(self):
        if self.pil_image is None: return
        arr = np.asarray(self.pil_image, np.uint8)
        if self.invert_check.isChecked(): arr = 255 - arr
        lut = self.curve_widget.get_lut()
        arr = lut[arr]
        height, width = arr.shape
        q_image = QImage(arr.data, width, height, width, QImage.Format.Format_Grayscale8)
        self.processed_pixmap = QPixmap.fromImage(q_image)
        self.update_display()

    def update_display(self):
        if self.processed_pixmap is None: return
        label_pixmap = QPixmap(self.image_label.size())
        label_pixmap.fill(QColor("#2b2b2b"))
        painter = QPainter(label_pixmap)
        
        scaled_size = self.processed_pixmap.size() * self.zoom_factor
        target_pos = QPointF(
            (self.image_label.width() - scaled_size.width()) / 2,
            (self.image_label.height() - scaled_size.height()) / 2
        ) + self.pan_offset
        
        # Use toPoint() for the final drawing coordinate to avoid TypeError
        painter.drawPixmap(target_pos.toPoint(), self.processed_pixmap.scaled(scaled_size))
        painter.end()
        self.image_label.setPixmap(label_pixmap)

    def fit_to_view(self):
        if self.processed_pixmap is None: return
        label_size = self.image_label.size()
        pixmap_size = self.processed_pixmap.size()
        if pixmap_size.isEmpty(): return
        
        zoom_x = label_size.width() / pixmap_size.width()
        zoom_y = label_size.height() / pixmap_size.height()
        self.zoom_factor = min(zoom_x, zoom_y)
        self.pan_offset = QPointF(0, 0)
        self.update_display()

    def eventFilter(self, source, event):
        if source is self.image_label:
            if event.type() == QEvent.Type.Wheel:
                delta = event.angleDelta().y()
                old_zoom = self.zoom_factor
                self.zoom_factor *= 1.15 if delta > 0 else 1 / 1.15
                self.zoom_factor = max(0.02, min(self.zoom_factor, 50.0))
                
                mouse_pos = event.position()
                center_pos = QPointF(self.image_label.rect().center())
                
                mouse_vec = mouse_pos - center_pos - self.pan_offset
                self.pan_offset -= mouse_vec * (self.zoom_factor / old_zoom - 1)
                self.update_display()
                return True

            # --- [MODIFIED] Use Left Mouse Button for Panning ---
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.is_panning = True
                    self.last_pan_pos = event.position()
                    self.image_label.setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True
                if event.button() == Qt.MouseButton.RightButton:
                    self.fit_to_view()
                    return True
            
            if event.type() == QEvent.Type.MouseMove and self.is_panning:
                delta = event.position() - self.last_pan_pos
                self.pan_offset += delta
                self.last_pan_pos = event.position()
                self.update_display()
                return True

            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                if self.is_panning:
                    self.is_panning = False
                    self.image_label.setCursor(Qt.CursorShape.OpenHandCursor)
                    return True
        
        return super().eventFilter(source, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())