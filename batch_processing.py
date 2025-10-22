"""
批量處理模組
包含批量圖像處理功能
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Any
from PyQt5.QtCore import QThread, pyqtSignal
from color_analysis import analyze_image, ChannelType
import numpy.typing as npt


class BatchProcessor(QThread):
    """批量處理執行緒"""
    
    # 信號定義
    progress_updated = pyqtSignal(int, int, str)  # current, total, filename
    processing_finished = pyqtSignal(bool, str)   # success, message
    
    def __init__(self, input_folder: str, parameters: dict, fg_color_blocks: List[Any], bg_color_blocks: List[Any]):
        super().__init__()
        self.input_folder = input_folder
        self.parameters = parameters
        self.fg_color_blocks = fg_color_blocks
        self.bg_color_blocks = bg_color_blocks
        self.should_stop = False
        
    def stop_processing(self):
        """停止處理"""
        self.should_stop = True
        
    def run(self):
        """執行批量處理"""
        try:
            # 創建輸出資料夾
            folder_name = os.path.basename(self.input_folder.rstrip(os.sep))
            parent_folder = os.path.dirname(self.input_folder)
            output_folder = os.path.join(parent_folder, f"{folder_name}-rmbg")
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # 獲取所有支援的圖片檔案
            image_files = self.get_image_files(self.input_folder)
            
            if not image_files:
                self.processing_finished.emit(False, "在選定資料夾中沒有找到支援的圖片檔案")
                return
            
            total_files = len(image_files)
            processed_count = 0
            
            for i, image_path in enumerate(image_files):
                if self.should_stop:
                    break
                    
                try:
                    filename = os.path.basename(image_path)
                    self.progress_updated.emit(i + 1, total_files, filename)
                    
                    # 處理單張圖片
                    success = self.process_single_image(image_path, output_folder)
                    
                    if success:
                        processed_count += 1
                        
                except Exception as e:
                    print(f"處理圖片 {image_path} 時發生錯誤: {e}")
                    continue
            
            if self.should_stop:
                self.processing_finished.emit(False, "處理已被用戶取消")
            else:
                self.processing_finished.emit(True, f"批量處理完成！成功處理 {processed_count}/{total_files} 張圖片")
                
        except Exception as e:
            self.processing_finished.emit(False, f"批量處理失敗: {str(e)}")
    
    def get_image_files(self, folder_path: str) -> List[str]:
        """獲取資料夾中所有支援的圖片檔案"""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_extensions:
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)
    
    def adjust_crop_coordinates(self, img_height: int, img_width: int, crop_coords: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """調整裁切座標以適應圖片尺寸"""
        x1, y1, x2, y2 = crop_coords
        
        # 確保座標在有效範圍內
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(x1 + 1, min(x2, img_width))
        y2 = max(y1 + 1, min(y2, img_height))
        
        return x1, y1, x2, y2
    
    def process_single_image(self, input_path: str, output_folder: str) -> bool:
        """處理單張圖片"""
        try:
            # 讀取圖片
            img = cv2.imread(input_path)
            if img is None:
                print(f"無法讀取圖片: {input_path}")
                return False
            
            height, width = img.shape[:2]
            
            # 調整裁切座標
            original_crop = (
                self.parameters.get("x1", 0),
                self.parameters.get("y1", 0),
                self.parameters.get("x2", width),
                self.parameters.get("y2", height)
            )
            
            adjusted_crop = self.adjust_crop_coordinates(height, width, original_crop)
            x1, y1, x2, y2 = adjusted_crop
            
            # 裁切圖片
            img_cropped = img[y1:y2, x1:x2].copy()
            
            # 轉換為 RGB 格式
            img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB).astype(np.float64)
            
            # 執行分析
            channel_type = self.parameters.get("channel", "RGB")
            fg_threshold = self.parameters.get("fg_threshold", 20.0)
            bg_threshold = self.parameters.get("bg_threshold", 80.0)
            noise_removal_area = self.parameters.get("noise_removal_area", 0)
            dilate_size = self.parameters.get("dilate_size", 0)
            erode_size = self.parameters.get("erode_size", 0)
            
            result_img = analyze_image(
                img_rgb, 
                self.fg_color_blocks, 
                self.bg_color_blocks, 
                channel_type,
                fg_threshold, 
                bg_threshold, 
                None,  # 已經裁切過了
                noise_removal_area,
                dilate_size,
                erode_size
            )
            
            # 創建 RGBA 圖片（將綠色通道作為 alpha）
            alpha = result_img[:, :, 1]
            rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
            rgba = np.dstack([rgb, alpha])
            
            # 生成輸出檔案路徑
            filename = os.path.basename(input_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{name}.png")
            
            # 保存圖片（轉換為 BGRA 格式給 OpenCV）
            cv2.imwrite(output_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            
            return True
            
        except Exception as e:
            print(f"處理圖片 {input_path} 時發生錯誤: {e}")
            return False


def validate_batch_parameters(parameters: dict, fg_color_blocks: List[Any], bg_color_blocks: List[Any]) -> Tuple[bool, str]:
    """驗證批量處理參數"""
    
    # 檢查必要參數
    required_params = ["channel", "fg_threshold", "bg_threshold", "noise_removal_area", "dilate_size", "erode_size"]
    for param in required_params:
        if param not in parameters:
            return False, f"缺少必要參數: {param}"
    
    # 檢查顏色方塊
    if not fg_color_blocks:
        return False, "至少需要一個前景顏色"
    
    if not bg_color_blocks:
        return False, "至少需要一個背景顏色"
    
    # 檢查閾值範圍
    if not (0 <= parameters["fg_threshold"] <= 100):
        return False, "前景閾值必須在 0-100 範圍內"
    
    if not (0 <= parameters["bg_threshold"] <= 100):
        return False, "背景閾值必須在 0-100 範圍內"
    
    if parameters["fg_threshold"] >= parameters["bg_threshold"]:
        return False, "前景閾值必須小於背景閾值"
    
    # 檢查邊緣處理參數範圍
    if not (0 <= parameters["dilate_size"] <= 20):
        return False, "擴張大小必須在 0-20 範圍內"
    
    if not (0 <= parameters["erode_size"] <= 20):
        return False, "侵蝕大小必須在 0-20 範圍內"
    
    return True, "參數驗證通過"