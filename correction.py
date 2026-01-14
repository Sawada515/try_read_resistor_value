#!/bin/env python3

from pathlib import Path

import cv2
import numpy as np

IMG_INPUT_DIR = "./output_resistors"
IMG_OUTPUT_DIR = "./results"

MIN_EXPECTED_ASPECT_AFTER_ALIGNMENT = 2.5

def main():
    save_dir = Path(IMG_OUTPUT_DIR)
    
    for f in save_dir.glob("*"):
        if f.is_file():
            f.unlink()
    
    input_path = Path(IMG_INPUT_DIR)

    files = list(input_path.rglob("*.bmp"))

    for file in files:
        print(f"start {str(file)}")

        img = cv2.imread(str(file))
        if img is None:
            print(f"{str(file)} is None")

            continue
    
        check_resistor_roi_quality(img)

def check_resistor_roi_quality(roi: np.ndarray) -> bool:
    """check_resistor_roi_quality

    Args:
        roi (np.ndarray): 角度補正済みROI

    Returns:
        bool: True -> 処理続行 False -> 引数として渡したROIを破棄
    """

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

def clip_resistor_roi(roi: np.ndarray) -> ndarray:
    """clip_resistor_roi
    YOLOで検出した画像の4辺のごみをなくすために5%カット

    Args:
        roi (np.ndarray): クオリティチェック後の画像

    Returns:
        ndarray: 4辺を5%カットした後の画像
    """

if __name__ == "__main__":
    main()
        
