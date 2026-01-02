import sys
from typing import Optional, Callable, List
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QColorDialog, QComboBox,
    QSlider, QSpinBox, QProgressBar, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QTransform
from PyQt5.QtCore import Qt, QPoint
import cv2
import numpy as np
import numpy.typing as npt
from color_analysis import analyze_image
from image_rendering import (
    apply_gaussian_blur, 
    create_alpha_overlay_image, 
    create_layer_overlay_image,
    create_result_image_with_thresholds
)
from batch_processing import BatchProcessor, validate_batch_parameters


class ZoomableLabel(QLabel):
    """可縮放和拖曳的圖片標籤"""
    def __init__(self, text: str = "", parent: Optional[QWidget] = None, sync_callback: Optional[Callable] = None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        # 移除背景色設定，使用透明方格背景
        self.setScaledContents(False)  # 不自動縮放內容
        self.setMinimumSize(100, 100)  # 設置最小尺寸防止縮小到0
        
        self.pixmap_original: Optional[QPixmap] = None  # 原始 pixmap
        self.scale_factor: float = 1.0      # 縮放因子
        self.offset: QPoint = QPoint(0, 0)   # 偏移量
        self.dragging: bool = False        # 是否正在拖曳
        self.last_pos: QPoint = QPoint()     # 上次滑鼠位置
        self.sync_callback: Optional[Callable] = sync_callback  # 同步回調函數
        self.is_syncing: bool = False      # 防止無限循環的標誌
        
    def create_checkerboard_background(self, width: int, height: int, square_size: int = 20) -> QPixmap:
        """創建白灰方格透明背景"""
        from PyQt5.QtGui import QPainter
        
        # 確保尺寸至少為 1
        width = max(1, width)
        height = max(1, height)
        
        canvas = QPixmap(width, height)
        painter = QPainter(canvas)
        
        # 定義兩種顏色
        color1 = QColor(255, 255, 255)  # 白色
        color2 = QColor(220, 220, 220)  # 淺灰色
        
        # 繪製方格
        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                # 根據位置選擇顏色（棋盤模式）
                if (x // square_size + y // square_size) % 2 == 0:
                    painter.fillRect(x, y, square_size, square_size, color1)
                else:
                    painter.fillRect(x, y, square_size, square_size, color2)
        
        painter.end()
        return canvas
        
    def set_image(self, pixmap):
        """設置圖片"""
        # 只有在圖片真的改變時才重置縮放和偏移
        if self.pixmap_original is None or self.pixmap_original.size() != pixmap.size():
            # 新圖片或尺寸改變時重置
            self.pixmap_original = pixmap
            self.scale_factor = 1.0
            self.offset = QPoint(0, 0)
        else:
            # 只是更新內容，保持當前的縮放和偏移
            self.pixmap_original = pixmap
        
        self.update_display()
    
    def update_display(self):
        """更新顯示"""
        if self.pixmap_original is None:
            # 沒有圖片時只顯示方格背景
            canvas_width = max(self.width(), 1)
            canvas_height = max(self.height(), 1)
            canvas = self.create_checkerboard_background(canvas_width, canvas_height)
            QLabel.setPixmap(self, canvas)
            return
        
        # 計算縮放後的大小（轉換為整數）
        scaled_width = int(self.pixmap_original.width() * self.scale_factor)
        scaled_height = int(self.pixmap_original.height() * self.scale_factor)
        
        # 確保尺寸至少為1
        scaled_width = max(1, scaled_width)
        scaled_height = max(1, scaled_height)
        
        scaled_pixmap = self.pixmap_original.scaled(
            scaled_width,
            scaled_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # 創建一個與 label 當前大小相同的 pixmap
        canvas_width = max(self.width(), 1)
        canvas_height = max(self.height(), 1)
        
        # 創建帶方格背景的畫布
        canvas = self.create_checkerboard_background(canvas_width, canvas_height)
        
        from PyQt5.QtGui import QPainter
        painter = QPainter(canvas)
        
        # 計算繪製位置（居中 + 偏移）
        x = (canvas_width - scaled_pixmap.width()) // 2 + self.offset.x()
        y = (canvas_height - scaled_pixmap.height()) // 2 + self.offset.y()
        
        painter.drawPixmap(x, y, scaled_pixmap)
        painter.end()
        
        # 使用 QLabel 的內部方法設置 pixmap，不觸發 resize
        QLabel.setPixmap(self, canvas)
    
    def sizeHint(self):
        """覆蓋 sizeHint 防止自動調整大小"""
        return self.size()
    
    def minimumSizeHint(self):
        """覆蓋 minimumSizeHint 防止自動調整大小"""
        return self.minimumSize()
    
    def wheelEvent(self, event):
        """滾輪縮放 - 以檢視窗中心為縮放中心"""
        if self.pixmap_original is None or self.is_syncing:
            return
        
        # 獲取滾輪滾動量
        delta = event.angleDelta().y()
        
        # 計算縮放因子變化
        zoom_in_factor = 1.1
        zoom_out_factor = 0.9
        
        old_scale = self.scale_factor
        
        if delta > 0:  # 放大
            self.scale_factor *= zoom_in_factor
        else:  # 縮小
            self.scale_factor *= zoom_out_factor
        
        # 限制縮放範圍
        self.scale_factor = max(0.1, min(10.0, self.scale_factor))
        
        # 計算縮放後需要調整的偏移量，使縮放中心為檢視窗中心
        if self.scale_factor != old_scale and self.pixmap_original is not None:
            # 計算縮放比例變化
            scale_ratio = self.scale_factor / old_scale
            
            # 調整偏移量，使圖片保持在檢視窗中心縮放
            # 當放大時，偏移量需要向中心收縮
            # 當縮小時，偏移量需要向外擴展
            self.offset.setX(int(self.offset.x() * scale_ratio))
            self.offset.setY(int(self.offset.y() * scale_ratio))
        
        self.update_display()
        
        # 同步到另一個視圖
        if self.sync_callback:
            self.sync_callback(self.scale_factor, self.offset)
    
    def mousePressEvent(self, event):
        """開始拖曳"""
        if event.button() == Qt.LeftButton and self.pixmap_original is not None:
            self.dragging = True
            self.last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        """拖曳移動"""
        if self.dragging and not self.is_syncing:
            delta = event.pos() - self.last_pos
            self.offset += delta
            self.last_pos = event.pos()
            self.update_display()
            
            # 同步到另一個視圖
            if self.sync_callback:
                self.sync_callback(self.scale_factor, self.offset)
    
    def mouseReleaseEvent(self, event):
        """結束拖曳"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
    
    def sync_from_other(self, scale_factor, offset):
        """從另一個視圖同步狀態"""
        self.is_syncing = True  # 設置標誌防止無限循環
        self.scale_factor = scale_factor
        self.offset = QPoint(offset)
        self.update_display()
        self.is_syncing = False  # 重置標誌
    
    def resizeEvent(self, event):
        """視窗大小改變時重新顯示"""
        super().resizeEvent(event)
        if not self.is_syncing:
            self.update_display()

class ColorBlock(QLabel):
    """可點擊的顏色顯示方塊"""
    def __init__(self, color_name="Foreground", parent=None, callback=None, remove_callback=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self.color = QColor(128, 128, 128)
        self.setStyleSheet(f"background-color: {self.color.name()}; border: 1px solid black;")
        self.setToolTip(color_name)
        self.callback = callback
        self.remove_callback = remove_callback

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 左鍵：開啟顏色選擇器
            color = QColorDialog.getColor(self.color, self, "選擇顏色")
            if color.isValid():
                self.color = color
                self.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
                if self.callback:
                    self.callback()
        elif event.button() == Qt.RightButton:
            # 右鍵：刪除顏色方塊
            if self.remove_callback:
                self.remove_callback(self)

class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Color Analysis UI")
        self.setGeometry(200, 100, 1000, 700)
        self.image = None
        self.original_filename = ""  # 記錄原始檔案名稱
        self.current_result_img = None  # 儲存當前的分析結果圖像
        self.batch_processor = None  # 批量處理執行緒

        # ---- 左側影像顯示區（可縮放和拖曳，並且同步）----
        self.label_original = ZoomableLabel("原始圖片", sync_callback=self.sync_to_result)
        self.label_result = ZoomableLabel("預覽圖片", sync_callback=self.sync_to_original)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.label_original)
        left_layout.addWidget(self.label_result)

        # ---- 右側控制區 ----
        self.btn_import = QPushButton("匯入圖片")
        self.btn_import.clicked.connect(self.import_image)

        # 顏色通道選擇下拉清單
        channel_label = QLabel("顏色通道:")
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(['RGB', '飽和', '亮度', '色相'])
        self.channel_combo.currentTextChanged.connect(self.update_analysis)

        # 上半部檢視模式選擇
        original_view_label = QLabel("上半部檢視模式:")
        self.original_view_combo = QComboBox()
        self.original_view_combo.addItems(['正常模式', '高斯模糊模式'])
        self.original_view_combo.currentTextChanged.connect(self.update_display_modes)

        # 高斯模糊參數控制項（初始隱藏）
        self.blur_kernel_label = QLabel("模糊核心大小:")
        self.blur_kernel_slider = QSlider(Qt.Horizontal)
        self.blur_kernel_slider.setRange(1, 50)
        self.blur_kernel_slider.setValue(15)
        self.blur_kernel_slider.valueChanged.connect(self.update_blur_parameters)
        
        self.blur_kernel_spinbox = QSpinBox()
        self.blur_kernel_spinbox.setRange(1, 99)
        self.blur_kernel_spinbox.setValue(15)
        self.blur_kernel_spinbox.valueChanged.connect(self.update_blur_parameters)
        
        # 連接滑桿和數字框
        self.blur_kernel_slider.valueChanged.connect(self.blur_kernel_spinbox.setValue)
        self.blur_kernel_spinbox.valueChanged.connect(self.blur_kernel_slider.setValue)
        
        self.blur_kernel_layout = QHBoxLayout()
        self.blur_kernel_layout.addWidget(self.blur_kernel_slider)
        self.blur_kernel_layout.addWidget(self.blur_kernel_spinbox)
        
        # 初始隱藏高斯模糊參數
        self.blur_kernel_label.setVisible(False)
        self.blur_kernel_slider.setVisible(False)
        self.blur_kernel_spinbox.setVisible(False)

        # 下半部檢視模式選擇
        result_view_label = QLabel("下半部檢視模式:")
        self.result_view_combo = QComboBox()
        self.result_view_combo.addItems(['預設模式', '不透明度模式', '圖層模式'])
        self.result_view_combo.currentTextChanged.connect(self.update_display_modes)

        # 前景閾值控制項
        fg_threshold_label = QLabel("前景閾值 (%):")
        self.fg_threshold_slider = QSlider(Qt.Horizontal)
        self.fg_threshold_slider.setRange(0, 100)
        self.fg_threshold_slider.setValue(20)
        self.fg_threshold_slider.valueChanged.connect(self.update_analysis)
        
        self.fg_threshold_spinbox = QSpinBox()
        self.fg_threshold_spinbox.setRange(0, 100)
        self.fg_threshold_spinbox.setValue(20)
        self.fg_threshold_spinbox.valueChanged.connect(self.update_analysis)
        
        # 連接滑桿和數字框
        self.fg_threshold_slider.valueChanged.connect(self.fg_threshold_spinbox.setValue)
        self.fg_threshold_spinbox.valueChanged.connect(self.fg_threshold_slider.setValue)
        
        fg_threshold_layout = QHBoxLayout()
        fg_threshold_layout.addWidget(self.fg_threshold_slider)
        fg_threshold_layout.addWidget(self.fg_threshold_spinbox)

        # 背景閾值控制項
        bg_threshold_label = QLabel("背景閾值 (%):")
        self.bg_threshold_slider = QSlider(Qt.Horizontal)
        self.bg_threshold_slider.setRange(0, 100)
        self.bg_threshold_slider.setValue(80)
        self.bg_threshold_slider.valueChanged.connect(self.update_analysis)
        
        self.bg_threshold_spinbox = QSpinBox()
        self.bg_threshold_spinbox.setRange(0, 100)
        self.bg_threshold_spinbox.setValue(80)
        self.bg_threshold_spinbox.valueChanged.connect(self.update_analysis)
        
        # 連接滑桿和數字框
        self.bg_threshold_slider.valueChanged.connect(self.bg_threshold_spinbox.setValue)
        self.bg_threshold_spinbox.valueChanged.connect(self.bg_threshold_slider.setValue)
        
        bg_threshold_layout = QHBoxLayout()
        bg_threshold_layout.addWidget(self.bg_threshold_slider)
        bg_threshold_layout.addWidget(self.bg_threshold_spinbox)

        # 去雜點參數控制項
        noise_removal_label = QLabel("去雜點面積:")
        self.noise_removal_slider = QSlider(Qt.Horizontal)
        self.noise_removal_slider.setRange(0, 100)
        self.noise_removal_slider.setValue(0)
        # 移除自動更新連接
        
        self.noise_removal_spinbox = QSpinBox()
        self.noise_removal_spinbox.setRange(0, 9999)  # 提高上限到 9999
        self.noise_removal_spinbox.setValue(0)
        # 移除自動更新連接
        
        # 連接滑桿和數字框（保持同步，但不觸發渲染）
        self.noise_removal_slider.valueChanged.connect(self.noise_removal_spinbox.setValue)
        self.noise_removal_spinbox.valueChanged.connect(self.sync_noise_removal_slider)
        
        noise_removal_layout = QHBoxLayout()
        noise_removal_layout.addWidget(self.noise_removal_slider)
        noise_removal_layout.addWidget(self.noise_removal_spinbox)
        
        # 新增去雜點渲染按鈕
        self.btn_render_noise_removal = QPushButton("套用去雜點")
        self.btn_render_noise_removal.clicked.connect(self.render_with_noise_removal)

        # 去空洞參數控制項
        hole_removal_label = QLabel("去空洞面積:")
        self.hole_removal_slider = QSlider(Qt.Horizontal)
        self.hole_removal_slider.setRange(0, 1000)
        self.hole_removal_slider.setValue(0)
        # 移除自動更新連接
        
        self.hole_removal_spinbox = QSpinBox()
        self.hole_removal_spinbox.setRange(0, 9999)  # 提高上限到 9999
        self.hole_removal_spinbox.setValue(0)
        # 移除自動更新連接
        
        # 連接滑桿和數字框（保持同步，但不觸發渲染）
        self.hole_removal_slider.valueChanged.connect(self.hole_removal_spinbox.setValue)
        self.hole_removal_spinbox.valueChanged.connect(self.sync_hole_removal_slider)
        
        hole_removal_layout = QHBoxLayout()
        hole_removal_layout.addWidget(self.hole_removal_slider)
        hole_removal_layout.addWidget(self.hole_removal_spinbox)
        
        # 新增去空洞渲染按鈕
        self.btn_render_hole_removal = QPushButton("套用去空洞")
        self.btn_render_hole_removal.clicked.connect(self.render_with_hole_removal)

        # 邊緣處理參數控制項
        edge_label = QLabel("邊緣處理:")
        
        # 擴張參數
        dilate_label = QLabel("擴張:")
        self.dilate_slider = QSlider(Qt.Horizontal)
        self.dilate_slider.setRange(0, 20)
        self.dilate_slider.setValue(0)
        
        self.dilate_spinbox = QSpinBox()
        self.dilate_spinbox.setRange(0, 999)  # 提高上限到 999
        self.dilate_spinbox.setValue(0)
        
        # 連接擴張滑桿和數字框
        self.dilate_slider.valueChanged.connect(self.dilate_spinbox.setValue)
        self.dilate_spinbox.valueChanged.connect(self.sync_dilate_slider)
        
        dilate_layout = QHBoxLayout()
        dilate_layout.addWidget(dilate_label)
        dilate_layout.addWidget(self.dilate_slider)
        dilate_layout.addWidget(self.dilate_spinbox)
        
        # 侵蝕參數
        erode_label = QLabel("侵蝕:")
        self.erode_slider = QSlider(Qt.Horizontal)
        self.erode_slider.setRange(0, 20)
        self.erode_slider.setValue(0)
        
        self.erode_spinbox = QSpinBox()
        self.erode_spinbox.setRange(0, 999)  # 提高上限到 999
        self.erode_spinbox.setValue(0)
        
        # 連接侵蝕滑桿和數字框
        self.erode_slider.valueChanged.connect(self.erode_spinbox.setValue)
        self.erode_spinbox.valueChanged.connect(self.sync_erode_slider)
        
        erode_layout = QHBoxLayout()
        erode_layout.addWidget(erode_label)
        erode_layout.addWidget(self.erode_slider)
        erode_layout.addWidget(self.erode_spinbox)

        # 新增邊緣處理渲染按鈕
        self.btn_render_edge_processing = QPushButton("套用邊緣處理")
        self.btn_render_edge_processing.clicked.connect(self.render_with_edge_processing)

        # 裁切座標控制項
        crop_label = QLabel("裁切座標:")
        
        # x1 座標控制
        x1_label = QLabel("X1:")
        self.x1_slider = QSlider(Qt.Horizontal)
        self.x1_slider.setRange(0, 1000)
        self.x1_slider.setValue(0)
        self.x1_slider.valueChanged.connect(self.update_analysis)
        
        self.x1_spinbox = QSpinBox()
        self.x1_spinbox.setRange(0, 99999)
        self.x1_spinbox.setValue(0)
        self.x1_spinbox.valueChanged.connect(self.update_analysis)
        
        self.x1_slider.valueChanged.connect(self.x1_spinbox.setValue)
        self.x1_spinbox.valueChanged.connect(self.x1_slider.setValue)
        
        x1_layout = QHBoxLayout()
        x1_layout.addWidget(x1_label)
        x1_layout.addWidget(self.x1_slider)
        x1_layout.addWidget(self.x1_spinbox)

        # y1 座標控制
        y1_label = QLabel("Y1:")
        self.y1_slider = QSlider(Qt.Horizontal)
        self.y1_slider.setRange(0, 1000)
        self.y1_slider.setValue(0)
        self.y1_slider.valueChanged.connect(self.update_analysis)
        
        self.y1_spinbox = QSpinBox()
        self.y1_spinbox.setRange(0, 99999)
        self.y1_spinbox.setValue(0)
        self.y1_spinbox.valueChanged.connect(self.update_analysis)
        
        self.y1_slider.valueChanged.connect(self.y1_spinbox.setValue)
        self.y1_spinbox.valueChanged.connect(self.y1_slider.setValue)
        
        y1_layout = QHBoxLayout()
        y1_layout.addWidget(y1_label)
        y1_layout.addWidget(self.y1_slider)
        y1_layout.addWidget(self.y1_spinbox)

        # x2 座標控制
        x2_label = QLabel("X2:")
        self.x2_slider = QSlider(Qt.Horizontal)
        self.x2_slider.setRange(1, 1000)
        self.x2_slider.setValue(1000)
        self.x2_slider.valueChanged.connect(self.update_analysis)
        
        self.x2_spinbox = QSpinBox()
        self.x2_spinbox.setRange(1, 99999)
        self.x2_spinbox.setValue(1000)
        self.x2_spinbox.valueChanged.connect(self.update_analysis)
        
        self.x2_slider.valueChanged.connect(self.x2_spinbox.setValue)
        self.x2_spinbox.valueChanged.connect(self.x2_slider.setValue)
        
        x2_layout = QHBoxLayout()
        x2_layout.addWidget(x2_label)
        x2_layout.addWidget(self.x2_slider)
        x2_layout.addWidget(self.x2_spinbox)

        # y2 座標控制
        y2_label = QLabel("Y2:")
        self.y2_slider = QSlider(Qt.Horizontal)
        self.y2_slider.setRange(1, 1000)
        self.y2_slider.setValue(1000)
        self.y2_slider.valueChanged.connect(self.update_analysis)
        
        self.y2_spinbox = QSpinBox()
        self.y2_spinbox.setRange(1, 99999)
        self.y2_spinbox.setValue(1000)
        self.y2_spinbox.valueChanged.connect(self.update_analysis)
        
        self.y2_slider.valueChanged.connect(self.y2_spinbox.setValue)
        self.y2_spinbox.valueChanged.connect(self.y2_slider.setValue)
        
        y2_layout = QHBoxLayout()
        y2_layout.addWidget(y2_label)
        y2_layout.addWidget(self.y2_slider)
        y2_layout.addWidget(self.y2_spinbox)

        # 初始前景/背景顏色方塊（使用清單支持多個顏色）
        self.fg_color_blocks = []
        self.bg_color_blocks = []
        self.fg_color_block = ColorBlock("前景色", callback=self.update_analysis, remove_callback=self.remove_fg_color)
        self.bg_color_block = ColorBlock("背景色", callback=self.update_analysis, remove_callback=self.remove_bg_color)
        self.fg_color_blocks.append(self.fg_color_block)
        self.bg_color_blocks.append(self.bg_color_block)

        # （只使用歐氏距離法，移除算法選擇）

        # 三個額外的顏色顯示框（無互動功能）
        self.color_block_1 = QLabel()
        self.color_block_1.setFixedSize(40, 40)
        self.color_block_1.setStyleSheet("background-color: #888888; border: 1px solid black;")

        self.color_block_2 = QLabel()
        self.color_block_2.setFixedSize(40, 40)
        self.color_block_2.setStyleSheet("background-color: #888888; border: 1px solid black;")

        self.color_block_3 = QLabel()
        self.color_block_3.setFixedSize(40, 40)
        self.color_block_3.setStyleSheet("background-color: #888888; border: 1px solid black;")

        # 右側：匯入 + 通道選擇 + 閾值控制 + 前景/背景顏色群
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.btn_import)
        
        # 顏色通道選擇
        right_layout.addWidget(channel_label)
        right_layout.addWidget(self.channel_combo)
        right_layout.addSpacing(10)
        
        # 檢視模式選擇
        right_layout.addWidget(original_view_label)
        right_layout.addWidget(self.original_view_combo)
        right_layout.addWidget(self.blur_kernel_label)
        right_layout.addLayout(self.blur_kernel_layout)
        right_layout.addWidget(result_view_label)
        right_layout.addWidget(self.result_view_combo)
        right_layout.addSpacing(10)
        
        # 閾值控制
        right_layout.addWidget(fg_threshold_label)
        right_layout.addLayout(fg_threshold_layout)
        right_layout.addWidget(bg_threshold_label)
        right_layout.addLayout(bg_threshold_layout)
        
        # 去雜點控制
        right_layout.addWidget(noise_removal_label)
        right_layout.addLayout(noise_removal_layout)
        right_layout.addWidget(self.btn_render_noise_removal)
        
        # 去空洞控制
        right_layout.addWidget(hole_removal_label)
        right_layout.addLayout(hole_removal_layout)
        right_layout.addWidget(self.btn_render_hole_removal)
        right_layout.addSpacing(10)
        
        # 邊緣處理控制
        right_layout.addWidget(edge_label)
        right_layout.addLayout(dilate_layout)
        right_layout.addLayout(erode_layout)
        right_layout.addWidget(self.btn_render_edge_processing)
        right_layout.addSpacing(10)
        
        # 裁切控制
        right_layout.addWidget(crop_label)
        right_layout.addLayout(x1_layout)
        right_layout.addLayout(y1_layout)
        right_layout.addLayout(x2_layout)
        right_layout.addLayout(y2_layout)
        right_layout.addSpacing(10)

        # 前景顏色區塊（水平排列）
        fg_label = QLabel("前景顏色群:")
        right_layout.addWidget(fg_label)
        self.fg_layout = QHBoxLayout()
        self.fg_layout.addWidget(self.fg_color_block)
        add_fg_btn = QPushButton("新增前景色")
        add_fg_btn.clicked.connect(self.add_fg_color)
        right_layout.addLayout(self.fg_layout)
        right_layout.addWidget(add_fg_btn)

        # 背景顏色區塊（水平排列）
        bg_label = QLabel("背景顏色群:")
        right_layout.addWidget(bg_label)
        self.bg_layout = QHBoxLayout()
        self.bg_layout.addWidget(self.bg_color_block)
        add_bg_btn = QPushButton("新增背景色")
        add_bg_btn.clicked.connect(self.add_bg_color)
        right_layout.addLayout(self.bg_layout)
        right_layout.addWidget(add_bg_btn)

        # 輸出按鈕
        self.btn_export = QPushButton("輸出圖片")
        self.btn_export.clicked.connect(self.export_image)
        right_layout.addWidget(self.btn_export)

        # 批量處理區域
        batch_label = QLabel("批量處理:")
        right_layout.addWidget(batch_label)
        
        # 批量處理按鈕
        self.btn_batch_process = QPushButton("選擇資料夾進行批量處理")
        self.btn_batch_process.clicked.connect(self.start_batch_processing)
        right_layout.addWidget(self.btn_batch_process)
        
        # 批量處理進度條
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        right_layout.addWidget(self.batch_progress)
        
        # 批量處理狀態標籤
        self.batch_status_label = QLabel("")
        self.batch_status_label.setVisible(False)
        right_layout.addWidget(self.batch_status_label)
        
        # 停止批量處理按鈕
        self.btn_stop_batch = QPushButton("停止批量處理")
        self.btn_stop_batch.clicked.connect(self.stop_batch_processing)
        self.btn_stop_batch.setVisible(False)
        right_layout.addWidget(self.btn_stop_batch)

        # 參數匯出/讀取按鈕
        params_layout = QHBoxLayout()
        self.btn_export_params = QPushButton("匯出參數")
        self.btn_export_params.clicked.connect(self.export_parameters)
        self.btn_import_params = QPushButton("讀取參數")
        self.btn_import_params.clicked.connect(self.import_parameters)
        params_layout.addWidget(self.btn_export_params)
        params_layout.addWidget(self.btn_import_params)
        right_layout.addLayout(params_layout)

        # 保留三個靜態顏色顯示框
        # right_layout.addSpacing(10)
        # right_layout.addWidget(self.color_block_1)
        # right_layout.addWidget(self.color_block_2)
        # right_layout.addWidget(self.color_block_3)
        right_layout.addStretch(1)

        # ---- 主佈局 ----
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)

    def sync_to_result(self, scale_factor, offset):
        """將原始圖片的縮放和偏移同步到結果圖片"""
        self.label_result.sync_from_other(scale_factor, offset)
    
    def sync_to_original(self, scale_factor, offset):
        """將結果圖片的縮放和偏移同步到原始圖片"""
        self.label_original.sync_from_other(scale_factor, offset)
    
    def sync_noise_removal_slider(self, value):
        """同步去雜點數字框到滑桿，但只在滑桿範圍內才更新滑桿"""
        if value <= self.noise_removal_slider.maximum():
            self.noise_removal_slider.setValue(value)
        # 數字框的值總是會更新，即使超過滑桿範圍
    
    def sync_hole_removal_slider(self, value):
        """同步去空洞數字框到滑桿，但只在滑桿範圍內才更新滑桿"""
        if value <= self.hole_removal_slider.maximum():
            self.hole_removal_slider.setValue(value)
        # 數字框的值總是會更新，即使超過滑桿範圍
    
    def sync_dilate_slider(self, value):
        """同步擴張數字框到滑桿，但只在滑桿範圍內才更新滑桿"""
        if value <= self.dilate_slider.maximum():
            self.dilate_slider.setValue(value)
        # 數字框的值總是會更新，即使超過滑桿範圍
    
    def sync_erode_slider(self, value):
        """同步侵蝕數字框到滑桿，但只在滑桿範圍內才更新滑桿"""
        if value <= self.erode_slider.maximum():
            self.erode_slider.setValue(value)
        # 數字框的值總是會更新，即使超過滑桿範圍
    
    def import_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇圖片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            # 記錄原始檔案名稱（不包含副檔名）
            import os
            self.original_filename = os.path.splitext(os.path.basename(path))[0]
            
            # 使用 IMREAD_UNCHANGED 來保留 alpha 通道
            self.image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            
            # 檢查是否有 alpha 通道，如果沒有則添加一個完全不透明的 alpha 通道
            if len(self.image.shape) == 3 and self.image.shape[2] == 3:
                # BGR 圖像，添加 alpha 通道
                height, width = self.image.shape[:2]
                alpha_channel = np.full((height, width, 1), 255, dtype=np.uint8)
                self.image = np.concatenate([self.image, alpha_channel], axis=2)
            elif len(self.image.shape) == 2:
                # 灰階圖像，轉換為 BGRA
                height, width = self.image.shape
                bgr_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
                alpha_channel = np.full((height, width, 1), 255, dtype=np.uint8)
                self.image = np.concatenate([bgr_image, alpha_channel], axis=2)
            
            # 設置裁切座標的預設值為圖片的完整尺寸
            height, width = self.image.shape[:2]
            
            # 更新滑桿和數字框的範圍
            self.x1_slider.setMaximum(width - 1)
            self.x1_spinbox.setMaximum(width - 1)
            self.x2_slider.setMaximum(width)
            self.x2_spinbox.setMaximum(width)
            self.y1_slider.setMaximum(height - 1)
            self.y1_spinbox.setMaximum(height - 1)
            self.y2_slider.setMaximum(height)
            self.y2_spinbox.setMaximum(height)
            
            # 設置預設值
            self.x1_slider.setValue(0)
            self.y1_slider.setValue(0)
            self.x2_slider.setValue(width)
            self.y2_slider.setValue(height)
            
            self.display_image(self.image, self.label_original)
            self.update_analysis()

    def add_fg_color(self):
        """新增一個前景顏色方塊（採用目前預設色）"""
        new_block = ColorBlock("前景色", callback=self.update_analysis, remove_callback=self.remove_fg_color)
        self.fg_color_blocks.append(new_block)
        self.fg_layout.addWidget(new_block)

    def add_bg_color(self):
        """新增一個背景顏色方塊（採用目前預設色）"""
        new_block = ColorBlock("背景色", callback=self.update_analysis, remove_callback=self.remove_bg_color)
        self.bg_color_blocks.append(new_block)
        self.bg_layout.addWidget(new_block)

    def remove_fg_color(self, color_block):
        """刪除前景顏色方塊"""
        if color_block in self.fg_color_blocks:
            self.fg_color_blocks.remove(color_block)
            self.fg_layout.removeWidget(color_block)
            color_block.setParent(None)
            color_block.deleteLater()
            self.update_analysis()

    def remove_bg_color(self, color_block):
        """刪除背景顏色方塊"""
        if color_block in self.bg_color_blocks:
            self.bg_color_blocks.remove(color_block)
            self.bg_layout.removeWidget(color_block)
            color_block.setParent(None)
            color_block.deleteLater()
            self.update_analysis()

    def update_analysis(self):
        """當前景色或背景色改變時，更新分析結果（不包含去雜點）"""
        if self.image is None:
            return
        
        # 獲取選定的顏色通道和閾值參數
        selected_channel = self.channel_combo.currentText()
        fg_threshold = self.fg_threshold_slider.value()
        bg_threshold = self.bg_threshold_slider.value()
        # 不使用去雜點參數，設為 0
        noise_removal_area = 0
        
        # 獲取裁切座標
        x1 = self.x1_slider.value()
        y1 = self.y1_slider.value()
        x2 = self.x2_slider.value()
        y2 = self.y2_slider.value()
        
        # 確保座標有效性
        height, width = self.image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        # 將圖片轉換為 RGB 並轉為浮點數
        if self.image.shape[2] == 4:
            # BGRA 圖像，提取 BGR 和 alpha 通道
            img_bgr = self.image[:, :, :3]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
            # 提取 alpha 通道用於後續處理
            original_alpha = self.image[:, :, 3]
        else:
            # BGR 圖像，正常處理
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB).astype(np.float64)
            original_alpha = None
        
        # 裁切原始圖片並顯示
        if self.image.shape[2] == 4:
            cropped_original = self.image[y1:y2, x1:x2]
        else:
            cropped_original = self.image[y1:y2, x1:x2]
        self.display_original_image(cropped_original)
        
        # 使用分析函數，包含閾值、裁切參數，但不包含去雜點和邊緣處理
        crop_coords = (x1, y1, x2, y2)
        # 不使用邊緣處理參數，設為 0
        dilate_size = 0
        erode_size = 0
        # 提取裁切後的 alpha 通道（如果有的話）
        cropped_alpha = original_alpha[y1:y2, x1:x2] if original_alpha is not None else None
        result_img = analyze_image(img_rgb, self.fg_color_blocks, self.bg_color_blocks, selected_channel, fg_threshold, bg_threshold, crop_coords, noise_removal_area, dilate_size, erode_size, cropped_alpha)
        
        # 儲存當前結果圖像用於檢視模式切換
        self.current_result_img = result_img
        
        # 顯示結果
        if self.image.shape[2] == 4:
            cropped_original_display = cropped_original[:, :, :3]  # 只取 BGR 部分用於顯示
        else:
            cropped_original_display = cropped_original
        self.display_result_image(cropped_original_display, result_img)

    def render_with_noise_removal(self):
        """按需渲染，包含去雜點功能"""
        if self.image is None:
            return
        
        # 獲取選定的顏色通道和閾值參數
        selected_channel = self.channel_combo.currentText()
        fg_threshold = self.fg_threshold_slider.value()
        bg_threshold = self.bg_threshold_slider.value()
        noise_removal_area = self.noise_removal_slider.value()  # 使用去雜點參數
        
        # 獲取裁切座標
        x1 = self.x1_slider.value()
        y1 = self.y1_slider.value()
        x2 = self.x2_slider.value()
        y2 = self.y2_slider.value()
        
        # 確保座標有效性
        height, width = self.image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        # 將圖片轉換為 RGB 並轉為浮點數
        if self.image.shape[2] == 4:
            # BGRA 圖像，提取 BGR 和 alpha 通道
            img_bgr = self.image[:, :, :3]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
            # 提取 alpha 通道用於後續處理
            original_alpha = self.image[:, :, 3]
        else:
            # BGR 圖像，正常處理
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB).astype(np.float64)
            original_alpha = None
        
        # 使用分析函數，包含所有參數（包括去雜點）
        crop_coords = (x1, y1, x2, y2)
        dilate_size = self.dilate_slider.value()
        erode_size = self.erode_slider.value()
        # 提取裁切後的 alpha 通道（如果有的話）
        cropped_alpha = original_alpha[y1:y2, x1:x2] if original_alpha is not None else None
        result_img = analyze_image(img_rgb, self.fg_color_blocks, self.bg_color_blocks, selected_channel, fg_threshold, bg_threshold, crop_coords, noise_removal_area, dilate_size, erode_size, cropped_alpha)
        
        # 儲存當前結果圖像用於檢視模式切換
        self.current_result_img = result_img
        
        # 顯示結果
        cropped_original = self.image[y1:y2, x1:x2]
        if self.image.shape[2] == 4:
            cropped_original_display = cropped_original[:, :, :3]  # 只取 BGR 部分用於顯示
        else:
            cropped_original_display = cropped_original
        self.display_result_image(cropped_original_display, result_img)

    def render_with_hole_removal(self):
        """按需渲染，包含去空洞功能"""
        if self.image is None:
            return
        
        # 獲取選定的顏色通道和閾值參數
        selected_channel = self.channel_combo.currentText()
        fg_threshold = self.fg_threshold_slider.value()
        bg_threshold = self.bg_threshold_slider.value()
        noise_removal_area = self.noise_removal_slider.value()
        hole_removal_area = self.hole_removal_slider.value()  # 使用去空洞參數
        
        # 獲取裁切座標
        x1 = self.x1_slider.value()
        y1 = self.y1_slider.value()
        x2 = self.x2_slider.value()
        y2 = self.y2_slider.value()
        
        # 確保座標有效性
        height, width = self.image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        # 將圖片轉換為 RGB 並轉為浮點數
        if self.image.shape[2] == 4:
            # BGRA 圖像，提取 BGR 和 alpha 通道
            img_bgr = self.image[:, :, :3]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
            # 提取 alpha 通道用於後續處理
            original_alpha = self.image[:, :, 3]
        else:
            # BGR 圖像，正常處理
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB).astype(np.float64)
            original_alpha = None
        
        # 使用分析函數，包含所有參數（包括去空洞）
        crop_coords = (x1, y1, x2, y2)
        dilate_size = self.dilate_slider.value()
        erode_size = self.erode_slider.value()
        # 提取裁切後的 alpha 通道（如果有的話）
        cropped_alpha = original_alpha[y1:y2, x1:x2] if original_alpha is not None else None
        result_img = analyze_image(img_rgb, self.fg_color_blocks, self.bg_color_blocks, selected_channel, fg_threshold, bg_threshold, crop_coords, noise_removal_area, dilate_size, erode_size, cropped_alpha, hole_removal_area)
        
        # 儲存當前結果圖像用於檢視模式切換
        self.current_result_img = result_img
        
        # 顯示結果
        cropped_original = self.image[y1:y2, x1:x2]
        if self.image.shape[2] == 4:
            cropped_original_display = cropped_original[:, :, :3]  # 只取 BGR 部分用於顯示
        else:
            cropped_original_display = cropped_original
        self.display_result_image(cropped_original_display, result_img)

    def render_with_edge_processing(self):
        """按需渲染，包含邊緣處理功能"""
        if self.image is None:
            return
        
        # 獲取選定的顏色通道和閾值參數
        selected_channel = self.channel_combo.currentText()
        fg_threshold = self.fg_threshold_slider.value()
        bg_threshold = self.bg_threshold_slider.value()
        noise_removal_area = self.noise_removal_slider.value()
        dilate_size = self.dilate_slider.value()  # 使用邊緣處理參數
        erode_size = self.erode_slider.value()    # 使用邊緣處理參數
        
        # 獲取裁切座標
        x1 = self.x1_slider.value()
        y1 = self.y1_slider.value()
        x2 = self.x2_slider.value()
        y2 = self.y2_slider.value()
        
        # 確保座標有效性
        height, width = self.image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        # 裁切原始圖片並顯示
        cropped_original = self.image[y1:y2, x1:x2]
        self.display_original_image(cropped_original)  # 傳遞完整的圖像包括 alpha 通道
        
        # 將圖片轉換為 RGB 並轉為浮點數
        if self.image.shape[2] == 4:
            # BGRA 圖像，提取 BGR 和 alpha 通道
            img_bgr = self.image[:, :, :3]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
            # 提取 alpha 通道用於後續處理
            original_alpha = self.image[:, :, 3]
        else:
            # BGR 圖像，正常處理
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB).astype(np.float64)
            original_alpha = None
        
        # 使用分析函數，包含所有參數（包括邊緣處理）
        crop_coords = (x1, y1, x2, y2)
        # 提取裁切後的 alpha 通道（如果有的話）
        cropped_alpha = original_alpha[y1:y2, x1:x2] if original_alpha is not None else None
        result_img = analyze_image(img_rgb, self.fg_color_blocks, self.bg_color_blocks, selected_channel, fg_threshold, bg_threshold, crop_coords, noise_removal_area, dilate_size, erode_size, cropped_alpha)
        
        # 儲存當前結果圖像用於檢視模式切換
        self.current_result_img = result_img
        
        # 顯示結果
        if self.image.shape[2] == 4:
            cropped_original_display = cropped_original[:, :, :3]  # 只取 BGR 部分用於結果顯示
        else:
            cropped_original_display = cropped_original
        self.display_result_image(cropped_original_display, result_img)

    def display_image(self, img, label):
        """顯示影像在 ZoomableLabel 上"""
        # 檢查圖像是否有 alpha 通道
        if len(img.shape) == 3 and img.shape[2] == 4:
            # BGRA 圖像，轉換為 RGBA 用於顯示
            rgba_image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            h, w, ch = rgba_image.shape
            bytes_per_line = ch * w
            qimg = QImage(rgba_image.data, w, h, bytes_per_line, QImage.Format_RGBA8888).copy()
        else:
            # BGR 圖像，正常處理
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        pix = QPixmap.fromImage(qimg)
        
        # 使用 ZoomableLabel 的 set_image 方法
        if isinstance(label, ZoomableLabel):
            label.set_image(pix)
        else:
            # 兼容舊方法
            pix = pix.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pix)

    def display_original_image(self, img):
        """根據選擇的檢視模式顯示原始圖像"""
        view_mode = self.original_view_combo.currentText()
        
        if view_mode == '高斯模糊模式':
            # 應用高斯模糊，使用當前參數
            kernel_size = self.blur_kernel_slider.value()
            blurred_img = apply_gaussian_blur(img, kernel_size=kernel_size)
            self.display_image(blurred_img, self.label_original)
        else:
            # 正常模式
            self.display_image(img, self.label_original)

    def display_result_image(self, original_img, result_img):
        """根據選擇的檢視模式顯示結果圖像"""
        view_mode = self.result_view_combo.currentText()
        
        if view_mode == '不透明度模式':
            # 創建不透明度模式圖像（RGBA）
            rgba_img = create_alpha_overlay_image(
                cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), 
                result_img
            )
            # 轉換為可顯示的格式（需要轉回 BGR）
            # 對於 RGBA，我們需要特殊處理
            self.display_rgba_image(rgba_img, self.label_result)
            
        elif view_mode == '圖層模式':
            # 創建圖層模式圖像
            layer_img = create_layer_overlay_image(
                cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), 
                result_img
            )
            # 轉回 BGR 並顯示
            layer_bgr = cv2.cvtColor(layer_img, cv2.COLOR_RGB2BGR)
            self.display_image(layer_bgr, self.label_result)
            
        else:
            # 正常模式(黑/綠)
            result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            self.display_image(result_bgr, self.label_result)

    def display_rgba_image(self, rgba_img, label):
        """顯示 RGBA 圖像"""
        h, w, ch = rgba_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgba_img.data, w, h, bytes_per_line, QImage.Format_RGBA8888).copy()
        pix = QPixmap.fromImage(qimg)
        
        # 使用 ZoomableLabel 的 set_image 方法
        if isinstance(label, ZoomableLabel):
            label.set_image(pix)
        else:
            # 兼容舊方法
            pix = pix.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pix)

    def update_display_modes(self):
        """當檢視模式改變時更新顯示"""
        # 檢查是否選擇了高斯模糊模式，決定是否顯示模糊參數
        blur_mode_selected = self.original_view_combo.currentText() == '高斯模糊模式'
        self.blur_kernel_label.setVisible(blur_mode_selected)
        self.blur_kernel_slider.setVisible(blur_mode_selected)
        self.blur_kernel_spinbox.setVisible(blur_mode_selected)
        
        if self.image is None or self.current_result_img is None:
            return
        
        # 重新顯示當前的圖像
        # 獲取當前的裁切座標
        x1 = self.x1_slider.value()
        y1 = self.y1_slider.value()
        x2 = self.x2_slider.value()
        y2 = self.y2_slider.value()
        
        # 確保座標有效性
        height, width = self.image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        cropped_original = self.image[y1:y2, x1:x2]
        
        # 更新顯示
        self.display_original_image(cropped_original)  # 傳遞完整的圖像包括 alpha 通道
        if self.image.shape[2] == 4:
            cropped_original_display = cropped_original[:, :, :3]  # 只取 BGR 部分用於結果顯示
        else:
            cropped_original_display = cropped_original
        self.display_result_image(cropped_original_display, self.current_result_img)

    def update_blur_parameters(self):
        """當高斯模糊參數改變時更新顯示"""
        if self.image is None or self.original_view_combo.currentText() != '高斯模糊模式':
            return
        
        # 獲取當前的裁切座標
        x1 = self.x1_slider.value()
        y1 = self.y1_slider.value()
        x2 = self.x2_slider.value()
        y2 = self.y2_slider.value()
        
        # 確保座標有效性
        height, width = self.image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        cropped_original = self.image[y1:y2, x1:x2]
        
        # 只更新原始圖像顯示
        self.display_original_image(cropped_original)  # 傳遞完整的圖像包括 alpha 通道

    def export_image(self):
        if self.image is None:
            return
        
        # 產生預設檔案名稱
        default_filename = f"{self.original_filename}-rmbg.png" if self.original_filename else "output-rmbg.png"
        
        # 取得裁切座標
        x1 = self.x1_spinbox.value()
        y1 = self.y1_spinbox.value()
        x2 = self.x2_spinbox.value()
        y2 = self.y2_spinbox.value()
        # 裁切原始圖片
        img_cropped = self.image[y1:y2, x1:x2].copy()
        
        # 處理圖像數據，支持 alpha 通道
        if self.image.shape[2] == 4:
            # BGRA 圖像，提取 BGR 和 alpha 通道
            img_bgr_cropped = img_cropped[:, :, :3]
            original_alpha_cropped = img_cropped[:, :, 3]
            img_rgb_cropped = cv2.cvtColor(img_bgr_cropped, cv2.COLOR_BGR2RGB).astype(np.float64)
        else:
            # BGR 圖像，正常處理
            img_bgr_cropped = img_cropped
            original_alpha_cropped = None
            img_rgb_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB).astype(np.float64)
        
        # 取得預覽圖像（分析後結果），包含去雜點和去空洞參數
        selected_channel = self.channel_combo.currentText()
        fg_threshold = self.fg_threshold_slider.value()
        bg_threshold = self.bg_threshold_slider.value()
        noise_removal_area = self.noise_removal_slider.value()  # 輸出時使用去雜點參數
        dilate_size = self.dilate_slider.value()
        erode_size = self.erode_slider.value()
        hole_removal_area = self.hole_removal_slider.value()  # 輸出時使用去空洞參數
        result_img = analyze_image(img_rgb_cropped, self.fg_color_blocks, self.bg_color_blocks, selected_channel, fg_threshold, bg_threshold, None, noise_removal_area, dilate_size, erode_size, original_alpha_cropped, hole_removal_area)
        
        # 將 G 通道轉為 alpha 通道
        alpha = result_img[:, :, 1]
        rgb = cv2.cvtColor(img_bgr_cropped, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([rgb, alpha])
        # 儲存檔案
        path, _ = QFileDialog.getSaveFileName(self, "儲存圖片", default_filename, "PNG Files (*.png)")
        if path:
            cv2.imwrite(path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    def export_parameters(self):
        """匯出右方面板的所有參數為 JSON 檔案"""
        import json
        
        # 收集所有參數
        parameters = {
            "channel": self.channel_combo.currentText(),
            "fg_threshold": self.fg_threshold_slider.value(),
            "bg_threshold": self.bg_threshold_slider.value(),
            "noise_removal_area": self.noise_removal_slider.value(),
            "hole_removal_area": self.hole_removal_slider.value(),
            "dilate_size": self.dilate_slider.value(),
            "erode_size": self.erode_slider.value(),
            "original_view_mode": self.original_view_combo.currentText(),
            "result_view_mode": self.result_view_combo.currentText(),
            "blur_kernel_size": self.blur_kernel_slider.value(),
            "crop_coordinates": {
                "x1": self.x1_slider.value(),
                "y1": self.y1_slider.value(),
                "x2": self.x2_slider.value(),
                "y2": self.y2_slider.value()
            },
            "fg_colors": [],
            "bg_colors": []
        }
        
        # 收集前景顏色
        for block in self.fg_color_blocks:
            color = block.color
            parameters["fg_colors"].append({
                "r": color.red(),
                "g": color.green(),
                "b": color.blue()
            })
        
        # 收集背景顏色
        for block in self.bg_color_blocks:
            color = block.color
            parameters["bg_colors"].append({
                "r": color.red(),
                "g": color.green(),
                "b": color.blue()
            })
        
        # 儲存檔案
        path, _ = QFileDialog.getSaveFileName(self, "匯出參數", "parameters.json", "JSON Files (*.json)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(parameters, f, indent=2, ensure_ascii=False)

    def import_parameters(self):
        """從 JSON 檔案讀取參數並套用到右方面板"""
        import json
        
        path, _ = QFileDialog.getOpenFileName(self, "讀取參數", "", "JSON Files (*.json)")
        if not path:
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                parameters = json.load(f)
            
            # 套用通道設定
            if "channel" in parameters:
                index = self.channel_combo.findText(parameters["channel"])
                if index >= 0:
                    self.channel_combo.setCurrentIndex(index)
            
            # 套用閾值設定
            if "fg_threshold" in parameters:
                self.fg_threshold_slider.setValue(parameters["fg_threshold"])
            if "bg_threshold" in parameters:
                self.bg_threshold_slider.setValue(parameters["bg_threshold"])
            if "noise_removal_area" in parameters:
                self.noise_removal_slider.setValue(parameters["noise_removal_area"])
            if "hole_removal_area" in parameters:
                self.hole_removal_slider.setValue(parameters["hole_removal_area"])
            
            # 套用邊緣處理設定
            if "dilate_size" in parameters:
                self.dilate_slider.setValue(parameters["dilate_size"])
            if "erode_size" in parameters:
                self.erode_slider.setValue(parameters["erode_size"])
            
            # 套用檢視模式設定
            if "original_view_mode" in parameters:
                index = self.original_view_combo.findText(parameters["original_view_mode"])
                if index >= 0:
                    self.original_view_combo.setCurrentIndex(index)
            if "result_view_mode" in parameters:
                index = self.result_view_combo.findText(parameters["result_view_mode"])
                if index >= 0:
                    self.result_view_combo.setCurrentIndex(index)
            if "blur_kernel_size" in parameters:
                self.blur_kernel_slider.setValue(parameters["blur_kernel_size"])
            
            # 套用裁切座標（需要檢查範圍）
            if "crop_coordinates" in parameters:
                coords = parameters["crop_coordinates"]
                
                # 如果有圖片，檢查座標範圍
                if self.image is not None:
                    height, width = self.image.shape[:2]
                    
                    # 確保座標在有效範圍內
                    x1 = max(0, min(coords.get("x1", 0), width - 1))
                    y1 = max(0, min(coords.get("y1", 0), height - 1))
                    x2 = max(x1 + 1, min(coords.get("x2", width), width))
                    y2 = max(y1 + 1, min(coords.get("y2", height), height))
                    
                    self.x1_slider.setValue(x1)
                    self.y1_slider.setValue(y1)
                    self.x2_slider.setValue(x2)
                    self.y2_slider.setValue(y2)
                else:
                    # 沒有圖片時直接套用
                    self.x1_slider.setValue(coords.get("x1", 0))
                    self.y1_slider.setValue(coords.get("y1", 0))
                    self.x2_slider.setValue(coords.get("x2", 1000))
                    self.y2_slider.setValue(coords.get("y2", 1000))
            
            # 清除現有顏色方塊並重新建立
            # 清除前景顏色
            for block in self.fg_color_blocks[1:]:  # 保留第一個
                self.fg_layout.removeWidget(block)
                block.setParent(None)
                block.deleteLater()
            self.fg_color_blocks = [self.fg_color_blocks[0]]  # 只保留第一個
            
            # 清除背景顏色  
            for block in self.bg_color_blocks[1:]:  # 保留第一個
                self.bg_layout.removeWidget(block)
                block.setParent(None)
                block.deleteLater()
            self.bg_color_blocks = [self.bg_color_blocks[0]]  # 只保留第一個
            
            # 套用前景顏色
            if "fg_colors" in parameters and parameters["fg_colors"]:
                # 設定第一個方塊的顏色
                first_fg_color = parameters["fg_colors"][0]
                self.fg_color_blocks[0].color = QColor(first_fg_color["r"], first_fg_color["g"], first_fg_color["b"])
                self.fg_color_blocks[0].setStyleSheet(f"background-color: {self.fg_color_blocks[0].color.name()}; border: 1px solid black;")
                
                # 新增其餘的前景顏色方塊
                for color_data in parameters["fg_colors"][1:]:
                    new_block = ColorBlock("前景色", callback=self.update_analysis, remove_callback=self.remove_fg_color)
                    new_block.color = QColor(color_data["r"], color_data["g"], color_data["b"])
                    new_block.setStyleSheet(f"background-color: {new_block.color.name()}; border: 1px solid black;")
                    self.fg_color_blocks.append(new_block)
                    self.fg_layout.addWidget(new_block)
            
            # 套用背景顏色
            if "bg_colors" in parameters and parameters["bg_colors"]:
                # 設定第一個方塊的顏色
                first_bg_color = parameters["bg_colors"][0]
                self.bg_color_blocks[0].color = QColor(first_bg_color["r"], first_bg_color["g"], first_bg_color["b"])
                self.bg_color_blocks[0].setStyleSheet(f"background-color: {self.bg_color_blocks[0].color.name()}; border: 1px solid black;")
                
                # 新增其餘的背景顏色方塊
                for color_data in parameters["bg_colors"][1:]:
                    new_block = ColorBlock("背景色", callback=self.update_analysis, remove_callback=self.remove_bg_color)
                    new_block.color = QColor(color_data["r"], color_data["g"], color_data["b"])
                    new_block.setStyleSheet(f"background-color: {new_block.color.name()}; border: 1px solid black;")
                    self.bg_color_blocks.append(new_block)
                    self.bg_layout.addWidget(new_block)
            
            # 更新分析結果
            self.update_analysis()
            
        except Exception as e:
            # 可以加入錯誤處理，例如顯示訊息框
            print(f"讀取參數檔案時發生錯誤: {e}")

    def get_current_parameters(self) -> dict:
        """獲取當前所有參數"""
        return {
            "channel": self.channel_combo.currentText(),
            "fg_threshold": self.fg_threshold_slider.value(),
            "bg_threshold": self.bg_threshold_slider.value(),
            "noise_removal_area": self.noise_removal_slider.value(),
            "hole_removal_area": self.hole_removal_slider.value(),
            "dilate_size": self.dilate_slider.value(),
            "erode_size": self.erode_slider.value(),
            "original_view_mode": self.original_view_combo.currentText(),
            "result_view_mode": self.result_view_combo.currentText(),
            "blur_kernel_size": self.blur_kernel_slider.value(),
            "x1": self.x1_slider.value(),
            "y1": self.y1_slider.value(),
            "x2": self.x2_slider.value(),
            "y2": self.y2_slider.value()
        }

    def start_batch_processing(self):
        """開始批量處理"""
        # 選擇輸入資料夾
        folder_path = QFileDialog.getExistingDirectory(self, "選擇要批量處理的資料夾")
        if not folder_path:
            return
        
        # 獲取當前參數
        parameters = self.get_current_parameters()
        
        # 驗證參數
        is_valid, message = validate_batch_parameters(parameters, self.fg_color_blocks, self.bg_color_blocks)
        if not is_valid:
            QMessageBox.warning(self, "參數錯誤", f"無法進行批量處理：{message}")
            return
        
        # 顯示確認對話框
        reply = QMessageBox.question(
            self, 
            "確認批量處理", 
            f"即將對資料夾 '{folder_path}' 中的所有圖片進行批量處理。\n\n"
            f"當前設定：\n"
            f"- 顏色通道：{parameters['channel']}\n"
            f"- 前景閾值：{parameters['fg_threshold']}%\n"
            f"- 背景閾值：{parameters['bg_threshold']}%\n"
            f"- 去雜點面積：{parameters['noise_removal_area']}\n"
            f"- 去空洞面積：{parameters['hole_removal_area']}\n"
            f"- 裁切範圍：({parameters['x1']}, {parameters['y1']}) - ({parameters['x2']}, {parameters['y2']})\n\n"
            f"處理結果將輸出到：'{folder_path}-rmbg'\n\n"
            f"是否繼續？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 創建並啟動批量處理執行緒
        self.batch_processor = BatchProcessor(folder_path, parameters, self.fg_color_blocks, self.bg_color_blocks)
        self.batch_processor.progress_updated.connect(self.update_batch_progress)
        self.batch_processor.processing_finished.connect(self.batch_processing_finished)
        
        # 顯示進度UI
        self.batch_progress.setVisible(True)
        self.batch_status_label.setVisible(True)
        self.btn_stop_batch.setVisible(True)
        self.btn_batch_process.setEnabled(False)
        
        # 啟動處理
        self.batch_processor.start()

    def update_batch_progress(self, current: int, total: int, filename: str):
        """更新批量處理進度"""
        self.batch_progress.setMaximum(total)
        self.batch_progress.setValue(current)
        self.batch_status_label.setText(f"正在處理 ({current}/{total}): {filename}")

    def batch_processing_finished(self, success: bool, message: str):
        """批量處理完成"""
        # 隱藏進度UI
        self.batch_progress.setVisible(False)
        self.batch_status_label.setVisible(False)
        self.btn_stop_batch.setVisible(False)
        self.btn_batch_process.setEnabled(True)
        
        # 顯示結果訊息
        if success:
            QMessageBox.information(self, "批量處理完成", message)
        else:
            QMessageBox.warning(self, "批量處理失敗", message)
        
        # 清理執行緒
        self.batch_processor = None

    def stop_batch_processing(self):
        """停止批量處理"""
        if self.batch_processor:
            reply = QMessageBox.question(
                self, 
                "確認停止", 
                "確定要停止批量處理嗎？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.batch_processor.stop_processing()
                self.batch_status_label.setText("正在停止處理...")
                self.btn_stop_batch.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageApp()
    window.show()
    sys.exit(app.exec_())
