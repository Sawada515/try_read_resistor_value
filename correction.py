#!/bin/env python3

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from cv2.typing import MatLike
from typing import cast, List, TypedDict, Tuple, Optional
from enum import Enum, auto

IMG_INPUT_DIR:str = "./output_resistors"
IMG_OUTPUT_DIR:str = "./results"

MIN_EXPECTED_ASPECT_AFTER_ALIGNMENT:float = 2.5

def main():
    save_dir = Path(IMG_OUTPUT_DIR)
    
    for f in save_dir.glob("*"):
        if f.is_file():
            f.unlink()
    
    input_path = Path(IMG_INPUT_DIR)

    files = list(input_path.rglob("*.bmp"))

    for file in files:
        print(f"start {str(file)}")

        img: MatLike | None = cv2.imread(str(file))
        if img is None:
            print(f"{str(file)} is None")

            continue
    
        ret = check_resistor_roi_quality(cast(NDArray[np.uint8], img))
        if ret == False:
            continue
        
        # 2. ROIクリッピング (5%カット)
        clipped_roi = clip_resistor_roi(cast(NDArray[np.uint8], img))
        if clipped_roi is None:
            continue
            
        h, w = clipped_roi.shape[:2]

        hsv = cv2.cvtColor(clipped_roi, cv2.COLOR_BGR2HSV)
        
        h, s, v = cv2.split(hsv)

        lab = cv2.cvtColor(clipped_roi, cv2.COLOR_BGR2Lab)

        l, a, b = cv2.split(lab)

        specular_mask = create_specular_mask(cast(NDArray[np.uint8], clipped_roi))
        if specular_mask is None:
            continue

        debug_img = []

        debug_img.append(clipped_roi)
        
        debug_img.append(cv2.cvtColor(h, cv2.COLOR_GRAY2BGR))
        debug_img.append(cv2.cvtColor(s, cv2.COLOR_GRAY2BGR))
        debug_img.append(cv2.cvtColor(v, cv2.COLOR_GRAY2BGR))

        debug_img.append(cv2.cvtColor(l, cv2.COLOR_GRAY2BGR))
        debug_img.append(cv2.cvtColor(a, cv2.COLOR_GRAY2BGR))
        debug_img.append(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR))

        debug_img.append(cv2.cvtColor(specular_mask, cv2.COLOR_GRAY2BGR))

        debug_imgs = cv2.vconcat(debug_img)

        debug_imgs = cv2.resize(debug_imgs, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)

        cv2.imshow("debug", debug_imgs)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            
            break

        cv2.destroyAllWindows()

       
def check_resistor_roi_quality(roi: NDArray[np.uint8] | None) -> bool:
    """check_resistor_roi_quality

    Args1
        roi (np.ndarray): 角度補正済みROI

    Returns:
        bool: True -> 処理続行 False -> 引数として渡したROIを破棄
    """

    if roi is None:
        return False

    h, w = roi.shape[:2]
    if h < 10 or w < 30:
        return False
    
    area = h * w
    if area < 600:
        return False
    
    aspect = w / h
    if aspect < MIN_EXPECTED_ASPECT_AFTER_ALIGNMENT:
        return False
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    projection_y = np.sum(gray > 30, axis=1)
    raw_is_valid_mask = projection_y > (w * 0.2)

    valid_ys = np.where(raw_is_valid_mask)[0]
    if len(valid_ys) == 0:
        return False
    
    diffs = np.diff(valid_ys)

    split_indices = np.where(diffs > 1)[0] + 1
    
    segments = np.split(valid_ys, split_indices)

    max_continuous_hight = max(len(seg) for seg in segments)
    if max_continuous_hight < (h * 0.4):
        return False
    
    return True

def clip_resistor_roi(roi: NDArray[np.uint8] | None) -> NDArray[np.uint8] | None:
    """clip_resistor_roi
    YOLOで検出した画像の4辺のごみをなくすために5%カット

    Args:
        roi (np.ndarray): クオリティチェック後の画像

    Returns:
        ndarray: 4辺を5%カットした後の画像
    """
    if roi is None:
        return None

    h, w = roi.shape[:2]

    trim_w = int(w * 0.05)
    trim_h = int(h * 0.05)

    return roi[trim_h : h-trim_h, trim_w : w-trim_w]

def get_glare_l_threshold(l_roi: NDArray[np.float32] | None) -> float | None:
    if l_roi is None:
        return None

    hist = cv2.calcHist([l_roi], [0], None, [256], [0, 256])
    body_peak_index = np.argmax(hist)

    MARGIN_UINT8 = 53.5
    ABSOLUTE_FLOOR_UINT8 = 130.0
    
    dynamic_limit_uint8 = body_peak_index + MARGIN_UINT8
    final_limit_uint8 = max(dynamic_limit_uint8, ABSOLUTE_FLOOR_UINT8)

    min_limit = final_limit_uint8 * (100.0 / 255.0)

    l_roi_u8 = (l_roi * (255.0 / 100.0)) .clip(0, 255).astype(np.uint8)   
    triangle_threshold_u8, _ = cv2.threshold(l_roi_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

    triangle_threshold = triangle_threshold_u8 * (100.0 / 255.0) 

    if triangle_threshold < min_limit:
        FALLBACK_PERCENTILE = 99.8

        percentile_threshold = np.percentile(l_roi, FALLBACK_PERCENTILE)

        return float(np.fmax(percentile_threshold, min_limit))
    
    return float(triangle_threshold)

def get_s_threshold(s_roi: NDArray[np.float32] | None) -> float | None:
    if s_roi is None:
        return None
    
    s_u8 = (s_roi * 255.0).clip(0, 255).astype(np.uint8)

    otsu_threshold, _ = cv2.threshold(s_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    otsu_threshold = otsu_threshold / 255.0

    MIN_LIMIT = 0.10
    MAX_LIMIT = 0.30
    
    return np.fmax(np.fmin(otsu_threshold, MAX_LIMIT), MIN_LIMIT)

def get_ab_threshold(
    a_roi: NDArray[np.float32],
    b_roi: NDArray[np.float32],
    l_mask: NDArray[np.uint8]
) -> float:
    valid_indices = (l_mask > 0)
    if not np.any(valid_indices):
        return 20.0
    
    a_valid = a_roi[valid_indices]
    b_valid = b_roi[valid_indices]

    distances = np.sqrt(a_valid**2 + b_valid**2)

    distance_u8 = np.clip(distances, 0, 100).astype(np.uint8)

    if len(distance_u8) < 10:
        return 15.0

    otsu_val, _ = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    MIN_LIMIT = 8.0
    MAX_LIMIT = 20.0
    
    dynamic_threshold = float(otsu_val)

    return min(max(dynamic_threshold, MIN_LIMIT), MAX_LIMIT)

def create_specular_mask(
    clipped_roi: NDArray[np.uint8] | None
) -> NDArray[np.uint8] | None:

    if clipped_roi is None:
        return None
    
    roi_f = clipped_roi.astype(np.float32) / 255.0

    hsv_roi_f = cast(NDArray[np.float32], cv2.cvtColor(roi_f, cv2.COLOR_BGR2HSV))
    lab_roi_f = cast(NDArray[np.float32], cv2.cvtColor(roi_f, cv2.COLOR_BGR2LAB))

    s_roi = hsv_roi_f[:, :, 1]

    l_roi = lab_roi_f[:, :, 0]
    a_roi = lab_roi_f[:, :, 1]
    b_roi = lab_roi_f[:, :, 2]

    ret = get_glare_l_threshold(l_roi)
    if ret is None:
        return None

    l_threshold = ret

    ret = get_s_threshold(s_roi)
    if ret is None:
        return None
    
    s_threshold = ret
    
    _, mask_l = cv2.threshold(l_roi, l_threshold, 1.0, cv2.THRESH_BINARY)
    _, mask_s = cv2.threshold(s_roi, s_threshold, 1.0, cv2.THRESH_BINARY_INV)

    a_centered = a_roi
    b_centered = b_roi

    destination_ab = np.sqrt(a_centered**2 + b_centered**2)

    ab_threshold = get_ab_threshold(b_roi, a_roi, cast(NDArray[np.uint8], mask_l))
    
    _, mask_ab = cv2.threshold(destination_ab, ab_threshold, 1.0, cv2.THRESH_BINARY_INV)

    mask_l = (mask_l * 255).astype(np.uint8)
    mask_s = (mask_s * 255).astype(np.uint8)
    mask_ab = (mask_ab * 255).astype(np.uint8)

    candidates = cv2.bitwise_and(mask_l, mask_s)
    candidates = cv2.bitwise_and(candidates, mask_ab)
    
    h, w = clipped_roi.shape[:2]
    roi_area = h * w

    k_x = max(3, int(w * 0.02))
    k_y = max(1, int(h * 0.01))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_x | 1, k_y | 1))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(candidates)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < (roi_area * 0.001):
            continue
            
        aspect = float(w) / float(h)
        if aspect < 0.8:
            continue
        
        cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    return cast(NDArray[np.uint8], mask)
    

if __name__ == "__main__":
    main()
        
