"""
顏色分析計算模組
包含顏色通道轉換和距離計算功能
"""

import numpy as np
from numba import jit
from typing import Tuple, List, Literal, Any
import numpy.typing as npt
from image_rendering import create_result_image_with_thresholds

# 定義顏色通道類型
ChannelType = Literal['RGB', '飽和', '亮度', '色相']


@jit(nopython=True, cache=True)
def _convert_to_luminance_numba(rgb_normalized: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """使用 numba 優化的亮度轉換"""
    luminance = 0.299 * rgb_normalized[:, 0] + 0.587 * rgb_normalized[:, 1] + 0.114 * rgb_normalized[:, 2]
    return (luminance * 255.0).reshape(-1, 1)


@jit(nopython=True, cache=True)
def _convert_to_saturation_numba(rgb_normalized: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """使用 numba 優化的飽和度轉換"""
    n = rgb_normalized.shape[0]
    saturation = np.zeros(n)
    
    for i in range(n):
        max_val = max(rgb_normalized[i, 0], rgb_normalized[i, 1], rgb_normalized[i, 2])
        min_val = min(rgb_normalized[i, 0], rgb_normalized[i, 1], rgb_normalized[i, 2])
        
        if max_val == 0:
            saturation[i] = 0
        else:
            saturation[i] = (max_val - min_val) / max_val
    
    return (saturation * 255.0).reshape(-1, 1)


@jit(nopython=True, cache=True)
def _convert_to_hue_numba(rgb_normalized: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """使用 numba 優化的色相轉換"""
    n = rgb_normalized.shape[0]
    hue = np.zeros(n)
    
    for i in range(n):
        r, g, b = rgb_normalized[i, 0], rgb_normalized[i, 1], rgb_normalized[i, 2]
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        if diff == 0:
            hue[i] = 0
        elif max_val == r:
            hue[i] = ((g - b) / diff) % 6
        elif max_val == g:
            hue[i] = (b - r) / diff + 2
        else:  # max_val == b
            hue[i] = (r - g) / diff + 4
    
    # 轉換為 0-255 範圍
    hue = (hue * 60 / 360 * 255.0).reshape(-1, 1)
    return hue


def convert_to_channel(rgb_array: npt.NDArray[np.float64], channel_type: ChannelType) -> npt.NDArray[np.float64]:
    """將 RGB 陣列轉換為指定的顏色通道
    Args:
        rgb_array: 形狀為 (N, 3) 的 RGB 陣列
        channel_type: 'RGB', '飽和', '亮度', '色相'
    Returns:
        轉換後的陣列，形狀根據通道類型而定
    """
    if channel_type == 'RGB':
        return rgb_array
    
    # 正規化到 [0, 1] 範圍
    rgb_normalized = rgb_array / 255.0
    
    if channel_type == '亮度':
        # 使用 numba 優化的亮度轉換
        return _convert_to_luminance_numba(rgb_normalized)
    
    elif channel_type == '飽和':
        # 使用 numba 優化的飽和度轉換
        return _convert_to_saturation_numba(rgb_normalized)
    
    elif channel_type == '色相':
        # 使用 numba 優化的色相轉換
        return _convert_to_hue_numba(rgb_normalized)
    
    return rgb_array


@jit(nopython=True, cache=True)
def _calculate_rgb_distances_numba(
    pixels_channel: npt.NDArray[np.float64],
    fg_colors_channel: npt.NDArray[np.float64],
    bg_colors_channel: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """使用 numba 優化的 RGB 距離計算"""
    num_pixels = pixels_channel.shape[0]
    num_fg = fg_colors_channel.shape[0]
    num_bg = bg_colors_channel.shape[0]
    
    dist_fg = np.full(num_pixels, np.inf)
    dist_bg = np.full(num_pixels, np.inf)
    
    # 計算前景距離
    for p in range(num_pixels):
        for f in range(num_fg):
            diff_r = pixels_channel[p, 0] - fg_colors_channel[f, 0]
            diff_g = pixels_channel[p, 1] - fg_colors_channel[f, 1]
            diff_b = pixels_channel[p, 2] - fg_colors_channel[f, 2]
            distance = np.sqrt(diff_r**2 + diff_g**2 + diff_b**2)
            if distance < dist_fg[p]:
                dist_fg[p] = distance
    
    # 計算背景距離
    for p in range(num_pixels):
        for b in range(num_bg):
            diff_r = pixels_channel[p, 0] - bg_colors_channel[b, 0]
            diff_g = pixels_channel[p, 1] - bg_colors_channel[b, 1]
            diff_b = pixels_channel[p, 2] - bg_colors_channel[b, 2]
            distance = np.sqrt(diff_r**2 + diff_g**2 + diff_b**2)
            if distance < dist_bg[p]:
                dist_bg[p] = distance
    
    return dist_fg, dist_bg


@jit(nopython=True, cache=True)
def _calculate_single_channel_distances_numba(
    pixels_channel: npt.NDArray[np.float64],
    fg_colors_channel: npt.NDArray[np.float64],
    bg_colors_channel: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """使用 numba 優化的單通道距離計算"""
    num_pixels = pixels_channel.shape[0]
    num_fg = fg_colors_channel.shape[0]
    num_bg = bg_colors_channel.shape[0]
    
    dist_fg = np.full(num_pixels, np.inf)
    dist_bg = np.full(num_pixels, np.inf)
    
    # 計算前景距離
    for p in range(num_pixels):
        for f in range(num_fg):
            distance = abs(pixels_channel[p, 0] - fg_colors_channel[f, 0])
            if distance < dist_fg[p]:
                dist_fg[p] = distance
    
    # 計算背景距離
    for p in range(num_pixels):
        for b in range(num_bg):
            distance = abs(pixels_channel[p, 0] - bg_colors_channel[b, 0])
            if distance < dist_bg[p]:
                dist_bg[p] = distance
    
    return dist_fg, dist_bg


def calculate_color_distances(
    pixels: npt.NDArray[np.float64], 
    fg_colors: npt.NDArray[np.float64], 
    bg_colors: npt.NDArray[np.float64], 
    channel_type: ChannelType
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """計算像素到前景和背景顏色的距離
    Args:
        pixels: 像素陣列 (P, 3)
        fg_colors: 前景顏色陣列 (F, 3)
        bg_colors: 背景顏色陣列 (B, 3)
        channel_type: 顏色通道類型
    Returns:
        dist_fg, dist_bg: 到前景和背景的最小距離
    """
    # 轉換到指定的顏色通道
    pixels_channel = convert_to_channel(pixels, channel_type)
    fg_colors_channel = convert_to_channel(fg_colors, channel_type)
    bg_colors_channel = convert_to_channel(bg_colors, channel_type)
    
    # 使用 numba 優化的距離計算
    if channel_type == 'RGB':
        # RGB 通道使用歐氏距離
        dist_fg, dist_bg = _calculate_rgb_distances_numba(pixels_channel, fg_colors_channel, bg_colors_channel)
    else:
        # 單通道使用絕對差值
        dist_fg, dist_bg = _calculate_single_channel_distances_numba(pixels_channel, fg_colors_channel, bg_colors_channel)
    
    return dist_fg, dist_bg


def analyze_image(
    img_rgb: npt.NDArray[np.float64], 
    fg_color_blocks: List[Any], 
    bg_color_blocks: List[Any], 
    channel_type: ChannelType,
    fg_threshold: float = 20.0,
    bg_threshold: float = 80.0,
    crop_coords: Tuple[int, int, int, int] = None,
    noise_removal_area: int = 100,
    dilate_size: int = 0,
    erode_size: int = 0,
    original_alpha: npt.NDArray[np.uint8] = None
) -> npt.NDArray[np.uint8]:
    """分析圖像並生成結果
    Args:
        img_rgb: RGB 圖像陣列，形狀為 (H, W, 3)
        fg_color_blocks: 前景顏色方塊清單
        bg_color_blocks: 背景顏色方塊清單
        channel_type: 顏色通道類型
        fg_threshold: 前景閾值 (0-100)
        bg_threshold: 背景閾值 (0-100)
        crop_coords: 裁切座標 (x1, y1, x2, y2)，如果為 None 則不裁切
        noise_removal_area: 去雜點最小面積
        dilate_size: 擴張核心大小
        erode_size: 侵蝕核心大小
        original_alpha: 原始圖像的 alpha 通道，如果有的話
    Returns:
        result_img: 分析結果圖像，形狀為 (H, W, 3)
    """
    # 如果有裁切座標，先裁切圖像
    if crop_coords is not None:
        x1, y1, x2, y2 = crop_coords
        img_rgb = img_rgb[y1:y2, x1:x2, :]
    
    h, w, _ = img_rgb.shape
    pixels = img_rgb.reshape(-1, 3)  # shape (P,3)
    
    # 如果有原始 alpha 通道，創建像素遮罩
    alpha_mask = None
    if original_alpha is not None:
        # 將 alpha 通道重塑為 1D 陣列
        alpha_mask = original_alpha.reshape(-1) > 0  # alpha > 0 的像素才參與計算
    
    # 建立前景顏色矩陣和背景顏色矩陣
    if len(fg_color_blocks) > 0:
        fg_colors_rgb = np.array([[b.color.red(), b.color.green(), b.color.blue()] for b in fg_color_blocks], dtype=np.float64)
    else:
        # 前景群為空時使用白色
        fg_colors_rgb = np.array([[255.0, 255.0, 255.0]], dtype=np.float64)
        
    if len(bg_color_blocks) > 0:
        bg_colors_rgb = np.array([[b.color.red(), b.color.green(), b.color.blue()] for b in bg_color_blocks], dtype=np.float64)
    else:
        # 背景群為空時使用黑色
        bg_colors_rgb = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    
    # 計算距離
    dist_fg, dist_bg = calculate_color_distances(pixels, fg_colors_rgb, bg_colors_rgb, channel_type)
    
    # 計算距離比值：前景距離 / (前景距離 + 背景距離)
    # 避免除以零的情況
    total_dist = dist_fg + dist_bg + 1e-8
    fg_ratio = dist_fg / total_dist
    
    # 轉換為百分比 (0-100)
    intensity_percent = fg_ratio * 100.0
    
    # 如果有 alpha 遮罩，將 alpha 為 0 的像素設置為背景（100%）
    if alpha_mask is not None:
        intensity_percent[~alpha_mask] = 100.0  # alpha 為 0 的像素設為背景
    
    # 限制範圍到 [0, 100]
    intensity_percent = np.clip(intensity_percent, 0.0, 100.0)
    
    # 生成結果圖像
    result_img = create_result_image_with_thresholds(intensity_percent, h, w, fg_threshold, bg_threshold, noise_removal_area, dilate_size, erode_size)
    
    return result_img