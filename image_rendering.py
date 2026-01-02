"""
圖像渲染模組
包含各種結果圖像的生成和渲染功能
"""

import cv2
import numpy as np
from numba import jit
from typing import Tuple
import numpy.typing as npt


def apply_morphological_operations(mask: npt.NDArray[np.uint8], dilate_size: int, erode_size: int, use_custom_operations: bool = True) -> npt.NDArray[np.uint8]:
    """
    對遮罩應用擴張和侵蝕操作
    
    擴張：針對每個像素，在擴張範圍內找到數值最高的像素
    侵蝕：針對每個像素，在侵蝕範圍內找到數值最低的像素
    
    Parameters:
    - mask: 二值遮罩 (0-255)
    - dilate_size: 擴張操作的核心大小
    - erode_size: 侵蝕操作的核心大小
    - use_custom_operations: 是否使用自定義操作（預設True）
    
    Returns:
    - 處理後的遮罩
    """
    original_mask = mask.copy()  # 保存原始圖像
    result_mask = mask.copy()
    
    if use_custom_operations:
        # 先處理擴張（基於原始圖像）
        if dilate_size > 0:
            result_mask = _custom_dilate_numba(original_mask, dilate_size)
        
        # 再處理侵蝕（基於擴張結果，或者如果沒有擴張則基於原始圖像）
        if erode_size > 0:
            result_mask = _custom_erode_numba(result_mask, erode_size)
            result_mask = _custom_erode_numba(result_mask, erode_size)
    else:
        # 傳統的形態學操作
        if dilate_size > 0:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size*2+1, dilate_size*2+1))
            result_mask = cv2.dilate(result_mask, kernel_dilate, iterations=1)
        
        if erode_size > 0:
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size*2+1, erode_size*2+1))
            result_mask = cv2.erode(result_mask, kernel_erode, iterations=1)
    
    return result_mask


@jit(nopython=True, cache=True)
def _custom_dilate_numba(mask: npt.NDArray[np.uint8], kernel_size: int) -> npt.NDArray[np.uint8]:
    """
    自定義擴張操作：針對每個像素，將其值設為擴張範圍內原始影像的最高值
    
    邏輯：對於輸出影像的每個像素位置(x,y)，
         在原始影像中查看(x,y)周圍 kernel_size 範圍內的所有像素，
         取其中的最大值作為輸出影像(x,y)位置的新值
         
    這樣可以讓高值區域向外擴散，覆蓋低值區域
    """
    h, w = mask.shape
    result = np.zeros_like(mask)
    
    # 對輸出影像的每個像素位置
    for y in range(h):
        for x in range(w):
            max_val = mask[y, x]  # 從當前像素值開始，而不是從0開始
            found_any = False
            
            # 在原始影像中搜索當前位置周圍的範圍
            for dy in range(-kernel_size, kernel_size + 1):
                for dx in range(-kernel_size, kernel_size + 1):
                    source_y = y + dy
                    source_x = x + dx
                    
                    # 檢查來源位置是否在影像邊界內
                    if 0 <= source_y < h and 0 <= source_x < w:
                        # 使用圓形核心（更自然的擴張效果）
                        distance_sq = dy * dy + dx * dx
                        if distance_sq <= kernel_size * kernel_size:
                            # 從原始影像中取值
                            source_val = mask[source_y, source_x]
                            if source_val > max_val:
                                max_val = source_val
                            found_any = True
            
            # 如果沒有找到任何有效像素，保持原值
            if found_any:
                result[y, x] = max_val
            else:
                result[y, x] = mask[y, x]
    
    return result


@jit(nopython=True, cache=True)
def _custom_erode_numba(mask: npt.NDArray[np.uint8], kernel_size: int) -> npt.NDArray[np.uint8]:
    """
    自定義侵蝕操作：針對每個像素，將其值設為侵蝕範圍內原始影像的最低值
    
    邏輯：對於輸出影像的每個像素位置(x,y)，
         在原始影像中查看(x,y)周圍 kernel_size 範圍內的所有像素，
         取其中的最小值作為輸出影像(x,y)位置的新值
         
    這樣可以讓低值區域向外擴散，縮小高值區域
    """
    h, w = mask.shape
    result = np.zeros_like(mask)
    
    # 對輸出影像的每個像素位置
    for y in range(h):
        for x in range(w):
            min_val = mask[y, x]  # 從當前像素值開始，而不是從255開始
            found_any = False
            
            # 在原始影像中搜索當前位置周圍的範圍
            for dy in range(-kernel_size, kernel_size + 1):
                for dx in range(-kernel_size, kernel_size + 1):
                    source_y = y + dy
                    source_x = x + dx
                    
                    # 檢查來源位置是否在影像邊界內
                    if 0 <= source_y < h and 0 <= source_x < w:
                        # 使用圓形核心（更自然的侵蝕效果）
                        distance_sq = dy * dy + dx * dx
                        if distance_sq <= kernel_size * kernel_size:
                            # 從原始影像中取值
                            source_val = mask[source_y, source_x]
                            if source_val < min_val:
                                min_val = source_val
                            found_any = True
            
            # 如果沒有找到任何有效像素，保持原值
            if found_any:
                result[y, x] = min_val
            else:
                result[y, x] = mask[y, x]
    
    return result


@jit(nopython=True, cache=True)
def _create_result_image_numba(
    intensity_2d: npt.NDArray[np.float64],
    h: int,
    w: int,
    fg_threshold: float,
    bg_threshold: float
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """使用 numba 優化的結果圖像生成"""
    result_img = np.zeros((h, w, 3), dtype=np.uint8)
    alpha_channel = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            val = intensity_2d[y, x]
            
            if val <= fg_threshold:
                # 前景閾值內：純綠色
                result_img[y, x, 0] = 0    # R
                result_img[y, x, 1] = 255  # G
                result_img[y, x, 2] = 0    # B
                alpha_channel[y, x] = 255
            elif val >= bg_threshold:
                # 背景閾值內：純黑色
                result_img[y, x, 0] = 0    # R
                result_img[y, x, 1] = 0    # G
                result_img[y, x, 2] = 0    # B
                alpha_channel[y, x] = 0
            else:
                # 中間區域：線性插值
                # 從綠色 (0,255,0) 到黑色 (0,0,0)
                ratio = (val - fg_threshold) / (bg_threshold - fg_threshold)
                green_intensity = int(255 * (1.0 - ratio))
                result_img[y, x, 0] = 0
                result_img[y, x, 1] = green_intensity
                result_img[y, x, 2] = 0
                alpha_channel[y, x] = green_intensity
    
    return result_img, alpha_channel


@jit(nopython=True, cache=True)
def _update_result_with_cleaned_alpha_numba(
    result_img: npt.NDArray[np.uint8],
    cleaned_alpha: npt.NDArray[np.uint8],
    h: int,
    w: int
) -> npt.NDArray[np.uint8]:
    """使用 numba 優化的 alpha 通道更新，正確處理所有 alpha 值"""
    for y in range(h):
        for x in range(w):
            alpha_val = cleaned_alpha[y, x]
            if alpha_val == 0:
                # 完全透明：設為黑色
                result_img[y, x, 0] = 0
                result_img[y, x, 1] = 0
                result_img[y, x, 2] = 0
            else:
                # 根據 alpha 值設定綠色強度
                result_img[y, x, 0] = 0
                result_img[y, x, 1] = alpha_val  # 綠色通道等於 alpha 值
                result_img[y, x, 2] = 0
    return result_img


@jit(nopython=True, cache=True)
def _create_alpha_overlay_numba(
    original_img: npt.NDArray[np.uint8],
    green_channel: npt.NDArray[np.uint8],
    h: int,
    w: int
) -> npt.NDArray[np.uint8]:
    """使用 numba 優化的不透明度模式圖像生成"""
    result = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA
    
    for y in range(h):
        for x in range(w):
            # 複製原圖的 RGB
            result[y, x, 0] = original_img[y, x, 0]  # R
            result[y, x, 1] = original_img[y, x, 1]  # G
            result[y, x, 2] = original_img[y, x, 2]  # B
            # 使用綠色通道作為 alpha
            result[y, x, 3] = green_channel[y, x]    # A
    
    return result


@jit(nopython=True, cache=True)
def _create_layer_overlay_numba(
    original_img: npt.NDArray[np.uint8],
    green_channel: npt.NDArray[np.uint8],
    h: int,
    w: int
) -> npt.NDArray[np.uint8]:
    """使用 numba 優化的圖層模式圖像生成"""
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            # 原圖作為底圖
            orig_r = original_img[y, x, 0]
            orig_g = original_img[y, x, 1]
            orig_b = original_img[y, x, 2]
            
            # 綠色圖層的 alpha（綠色通道 * 0.5）
            green_alpha = int(green_channel[y, x] * 0.5)
            green_alpha_norm = green_alpha / 255.0
            
            # Alpha 混合：result = orig * (1 - alpha) + green * alpha
            result[y, x, 0] = int(orig_r * (1.0 - green_alpha_norm) + 0 * green_alpha_norm)      # R
            result[y, x, 1] = int(orig_g * (1.0 - green_alpha_norm) + 255 * green_alpha_norm)   # G
            result[y, x, 2] = int(orig_b * (1.0 - green_alpha_norm) + 0 * green_alpha_norm)     # B
    
    return result


def create_result_image_with_thresholds(
    intensity: npt.NDArray[np.float64], 
    h: int, 
    w: int,
    fg_threshold: float,
    bg_threshold: float,
    noise_removal_area: int = 100,
    dilate_size: int = 0,
    erode_size: int = 0,
    hole_removal_area: int = 0
) -> npt.NDArray[np.uint8]:
    """生成結果圖像，支援前景背景閾值、去雜點、邊緣處理和空洞移除
    Args:
        intensity: 強度值陣列，範圍 [0, 100]
        h: 圖像高度
        w: 圖像寬度
        fg_threshold: 前景閾值 (0-100)
        bg_threshold: 背景閾值 (0-100)
        noise_removal_area: 去雜點最小面積
        dilate_size: 擴張核心大小
        erode_size: 侵蝕核心大小
        hole_removal_area: 去空洞最小面積
    Returns:
        結果圖像陣列，形狀為 (h, w, 3)
    """
    from noise_removal import remove_noise_components, remove_hole_components  # 避免循環導入
    
    intensity_2d = intensity.reshape(h, w)
    
    # 使用 numba 優化的圖像生成
    result_img, alpha_channel = _create_result_image_numba(intensity_2d, h, w, fg_threshold, bg_threshold)
    
    # 應用去雜點處理
    if noise_removal_area > 0:
        cleaned_alpha = remove_noise_components(alpha_channel, noise_removal_area)
    else:
        cleaned_alpha = alpha_channel
    
    # 應用去空洞處理
    if hole_removal_area > 0:
        cleaned_alpha = remove_hole_components(cleaned_alpha, hole_removal_area)
    
    # 應用邊緣處理（擴張和侵蝕）
    if dilate_size > 0 or erode_size > 0:
        cleaned_alpha = apply_morphological_operations(cleaned_alpha, dilate_size, erode_size)
    
    # 使用 numba 優化的 alpha 更新（只有在進行了處理時才更新）
    if noise_removal_area > 0 or hole_removal_area > 0 or dilate_size > 0 or erode_size > 0:
        result_img = _update_result_with_cleaned_alpha_numba(result_img, cleaned_alpha, h, w)
    
    return result_img


def apply_gaussian_blur(img: npt.NDArray[np.uint8], kernel_size: int = 5) -> npt.NDArray[np.uint8]:
    """應用高斯模糊
    Args:
        img: 輸入圖像
        kernel_size: 核心大小（奇數）
    Returns:
        模糊後的圖像
    """
    # 確保核心大小為奇數
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 使用 OpenCV 的高斯模糊（更高品質）
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def create_alpha_overlay_image(
    original_img: npt.NDArray[np.uint8],
    result_img: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """創建不透明度模式圖像
    Args:
        original_img: 原始圖像 (H, W, 3)
        result_img: 分析結果圖像 (H, W, 3)
    Returns:
        不透明度模式圖像 (H, W, 4) RGBA
    """
    h, w = original_img.shape[:2]
    green_channel = result_img[:, :, 1]  # 取綠色通道
    
    # 使用 numba 優化函數
    return _create_alpha_overlay_numba(original_img, green_channel, h, w)


def create_layer_overlay_image(
    original_img: npt.NDArray[np.uint8],
    result_img: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """創建圖層模式圖像
    Args:
        original_img: 原始圖像 (H, W, 3)
        result_img: 分析結果圖像 (H, W, 3)
    Returns:
        圖層模式圖像 (H, W, 3)
    """
    h, w = original_img.shape[:2]
    green_channel = result_img[:, :, 1]  # 取綠色通道
    
    # 使用 numba 優化函數
    return _create_layer_overlay_numba(original_img, green_channel, h, w)