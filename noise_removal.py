"""
去雜點處理模組
包含連通區塊分析和小面積區域移除功能
"""

import cv2
import numpy as np
from numba import jit
import numpy.typing as npt


@jit(nopython=True, cache=True)
def _remove_small_components_numba(
    labels: npt.NDArray[np.int32], 
    stats: npt.NDArray[np.int32], 
    alpha_channel: npt.NDArray[np.uint8],
    min_area: int
) -> npt.NDArray[np.uint8]:
    """使用 numba 優化的小區塊移除函數"""
    h, w = labels.shape
    cleaned_alpha = alpha_channel.copy()
    num_labels = stats.shape[0]
    
    # 檢查每個連通區塊（跳過背景 label=0）
    for label in range(1, num_labels):
        area = stats[label, 4]  # cv2.CC_STAT_AREA 的索引是 4
        
        # 如果區塊面積小於閾值，將該區塊設為透明
        if area < min_area:
            for y in range(h):
                for x in range(w):
                    if labels[y, x] == label:
                        cleaned_alpha[y, x] = 0
    
    return cleaned_alpha


def remove_noise_components(alpha_channel: npt.NDArray[np.uint8], min_area: int) -> npt.NDArray[np.uint8]:
    """移除小於指定面積的連通區塊
    Args:
        alpha_channel: Alpha 通道陣列 (H, W)
        min_area: 最小面積閾值
    Returns:
        處理後的 Alpha 通道陣列
    """
    # Early return 如果 min_area <= 0，直接返回原始圖像
    if min_area <= 0:
        return alpha_channel.copy()
    
    # 創建二值化圖像（前景區域）
    # 假設 alpha > 0 為前景區域
    binary_mask = (alpha_channel > 0).astype(np.uint8) * 255
    
    # 尋找連通區塊
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # 如果沒有前景區域，直接返回
    if num_labels <= 1:
        return alpha_channel.copy()
    
    # 使用 numba 優化的函數處理
    cleaned_alpha = _remove_small_components_numba(labels, stats, alpha_channel, min_area)
    
    return cleaned_alpha