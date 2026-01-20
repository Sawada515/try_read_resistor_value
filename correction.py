#!/bin/env python3

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
from numpy.typing import NDArray
from cv2.typing import MatLike
from typing import cast, List, TypedDict, Tuple, Optional, Dict, Any
from enum import Enum, auto
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d, grey_opening

#セグメント分割時に使用 データクラス
@dataclass
class SegmentStats:
    index: int
    y_top: int
    y_bottom: int
    valid_pixels_ratio: float
    glare_mask_ratio: float
    weight: float
    ab_variance: float
    z_score: float
    quality_score: float

@dataclass
class BackGroundStats:
    color_lab: Tuple[float, float, float]
    occupancy_ratio: float
    ab_variance: float
    resistor_body_color_mask: NDArray[np.uint8]
    is_reliable: bool

@dataclass
class DetectBand:
    x: int
    band_width: int
    mean_color_lab: Tuple[float, float, float]
    ssd_score: float

IMG_INPUT_DIR:str = "./output_resistors"
IMG_OUTPUT_DIR:str = "./result"

#定数
MIN_EXPECTED_ASPECT_AFTER_ALIGNMENT:float = 2.5

# ヒストグラムのピーク(ボディの明るさ)から、どれくらい離れればグレアとみなすか
#L_PEAK_MARGIN: float = 20.0  
L_PEAK_MARGIN: float = 10.0  
# どんなに暗い画像でも、このL値を下回るものはグレアと判定しない（絶対下限）
L_ABSOLUTE_MIN: float = 50.0 

# 理想的なグレアはS=0だが、境界部分のボケや圧縮ノイズを考慮して広めに取る。
# 最終判定は L(高) AND ab(無彩色) との組み合わせで行うため、ここはリコール重視で甘めに設定。
S_THRESHOLD_MAX: float = 0.25

# MAD (Median Absolute Deviation) に掛ける係数。正規分布の3σ相当などを狙う。
AB_MAD_FACTOR: float = 4.0
# ab距離の閾値のハードリミット（最小2.0はノイズ許容、最大15.0は有彩色バンド除外）
AB_DIST_MIN: float = 2.0
AB_DIST_MAX: float = 15.0

# 画像幅に対するカーネル幅の比率（経験値）。
# 将来的には「抵抗器の物理直径(mm) × 解像度(px/mm)」から算出することが望ましい。
MORPH_KERNEL_RATIO_W: float = 0.015
MORPH_KERNEL_MAX_PX: int = 5
MORPH_KERNEL_MIN_PX: int = 3

# 最終的な候補領域内の最大輝度が、全体の上位0.5%輝度の何割以上か
L_CHECK_RATIO: float = 0.90
# 最終チェック時の絶対下限輝度
#L_CHECK_MIN: float = 70.0
L_CHECK_MIN: float = 50.0

#分割したセグメントの重み 中央のセグメントほど優先
Y_CENTER_WEIGHTS = {0: 0.7, 1: 0.85, 2: 1.0, 3: 0.85, 4: 0.7}

#ab分散閾値
AB_VARIANCE_MIN_SAFE: float = 100.0
AB_VARIANCE_MAX_SAFE: float = 400.0

MIN_SPLIT_SEGMENT: int = 5
GLARE_MAX_SOFT: float = 0.15

AB_VALIANCE_CLIP = 800.0

SN_RATIO_LOW: float = 3.0
SN_RATIO_HIGH: float = 10.0

def main():
    save_dir = Path(IMG_OUTPUT_DIR)
    
    for f in save_dir.glob("*"):
        if f.is_file():
            f.unlink()
    
    input_path = Path(IMG_INPUT_DIR)

    files = list(input_path.rglob("*.bmp"))

    files.sort(key=lambda f: int(re.findall(r'\d+', f.stem)[-1]))

    if files is None:
        print("dir is None")

    for file in files:
        print(f"start {str(file)}")

        img: MatLike | None = cv2.imread(str(file))
        if img is None:
            print(f"{str(file)} is None")

            continue
    
        ret = check_resistor_roi_quality(cast(NDArray[np.uint8], img))
        if ret == False:
            continue
        
        clipped_roi = clip_resistor_roi(cast(NDArray[np.uint8], img))
        if clipped_roi is None:
            continue

        print(f"clipped_roi size : {clipped_roi.shape[:2]}")
            
        h, w = clipped_roi.shape[:2]

        hsv = cv2.cvtColor(clipped_roi, cv2.COLOR_BGR2HSV)

        lab = cv2.cvtColor(clipped_roi, cv2.COLOR_BGR2Lab)

        l, a, b = cv2.split(lab)

        specular_mask = create_specular_mask(cast(NDArray[np.uint8], clipped_roi))
        if specular_mask is None:
            print("specular mask in None")

            continue

        lab_segment, valid_pixels_ratio = select_best_segment(cast(NDArray[np.uint8], clipped_roi), specular_mask)
        if lab_segment is None:
            continue

        lab_one_line = compress_segment_to_one_line(lab_segment)

        back_ground_stats: BackGroundStats = analyze_resistor_background(lab_one_line, valid_pixels_ratio)

        print(
            f"{back_ground_stats.color_lab}, {back_ground_stats.occupancy_ratio}, {back_ground_stats.ab_variance}, {back_ground_stats.is_reliable}"
        )

        result_bands, debug_data = detect_bands(lab_one_line, back_ground_stats)

        print("band_result start")
        for r_b in result_bands:
            print(r_b)
        print("band_result finish")

        correction_result_bands = refine_band_candidates(result_bands, w)

        points = [(x.x) for x in correction_result_bands]

        l_segment_u8 = (lab_segment[:, :, 0] * (255.0 / 100.0)).astype(np.uint8)
        a_segment_u8 = (lab_segment[:, :, 1] + 128.0).astype(np.uint8)
        b_segment_u8 = (lab_segment[:, :, 2] + 128.0).astype(np.uint8)

        lab_segment_u8 = cv2.merge([l_segment_u8, a_segment_u8, b_segment_u8])

        l_one_line_u8 = (lab_one_line[:, :, 0] * (255.0 / 100.0)).astype(np.uint8)
        a_one_line_u8 = (lab_one_line[:, :, 1] + 128.0).astype(np.uint8)
        b_one_line_u8 = (lab_one_line[:, :, 2] + 128.0).astype(np.uint8)

        lab_one_line_u8 = cv2.merge([l_one_line_u8, a_one_line_u8, b_one_line_u8])

        debug_add_points = lab_one_line_u8.copy()
        debug_add_points = cv2.cvtColor(debug_add_points, cv2.COLOR_LAB2BGR)

        debug_add_points = cv2.resize(debug_add_points, (w, h // 2), interpolation=cv2.INTER_NEAREST)

        for p in points:
            cv2.circle(debug_add_points, (p, h // 4), 2, (255, 0, 0), -1)

        lab_one_line_u8 = cv2.resize(lab_one_line_u8, (w, 2), interpolation=cv2.INTER_NEAREST)

        l_resistor_back_ground_u8 = np.clip(back_ground_stats.color_lab[0] * (255.0 / 100.0), 0, 255).astype(np.uint8)
        a_resistor_back_ground_u8 = np.clip(back_ground_stats.color_lab[1] + 128.0, 0, 255).astype(np.uint8)
        b_resistor_back_ground_u8 = np.clip(back_ground_stats.color_lab[2] + 128.0, 0, 255).astype(np.uint8)

        #w 1px h 1px
        lab_resistor_back_ground_u8 = np.array([[[l_resistor_back_ground_u8, a_resistor_back_ground_u8, b_resistor_back_ground_u8]]], dtype=np.uint8)

        lab_resistor_back_ground_u8 = cv2.resize(lab_resistor_back_ground_u8, (w, 2), interpolation=cv2.INTER_NEAREST)

        debug_img = []

        debug_img.append(clipped_roi)

        debug_img.append(cv2.cvtColor(specular_mask, cv2.COLOR_GRAY2BGR))

        debug_img.append(cv2.cvtColor(lab_segment_u8, cv2.COLOR_LAB2BGR))

        debug_img.append(np.zeros_like(lab_one_line_u8))
        
        debug_img.append(cv2.cvtColor(lab_one_line_u8, cv2.COLOR_LAB2BGR))

        debug_img.append(np.zeros_like(lab_one_line_u8))

        debug_img.append(cv2.cvtColor(lab_resistor_back_ground_u8, cv2.COLOR_LAB2BGR))

        debug_img.append(debug_add_points)

        debug_imgs = cv2.vconcat(debug_img)

        debug_imgs = cv2.resize(debug_imgs, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)


        save_name = file.name
        save_path =  save_dir / save_name

        cv2.imwrite(str(save_path), debug_imgs)

        print(f"save {save_path}")

       
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

    trim_w = int(w * 0.01)
    trim_h = int(h * 0.05)

    return roi[trim_h : h-trim_h, trim_w : w-trim_w]

def estimate_glare_l_threshold_peak(l_roi: NDArray[np.float32]) -> float:
    hist_bins = 100
    l_large_max = 100.0
    
    hist = cv2.calcHist([l_roi], [0], None, [hist_bins], [0.0, l_large_max])

    peak_index = int(np.argmax(hist))

    bin_width = l_large_max / float(hist_bins)
    peak_l_value = float(peak_index) * bin_width
    
    base_threshold = peak_l_value + L_PEAK_MARGIN
    
    p90 = float(np.percentile(l_roi, 90.0))

    tmp_l_threshold = max(base_threshold, L_ABSOLUTE_MIN)

    return min(tmp_l_threshold, p90)

def get_adaptive_white_point(
    a_roi: NDArray[np.float32],
    b_roi: NDArray[np.float32],
    s_roi: NDArray[np.float32]
) -> Tuple[float, float]:
    low_s_mask = s_roi < 0.25
    
    if np.count_nonzero(low_s_mask) < 10:
        return 128.0, 128.0
    
    a_reference = float(np.median(a_roi[low_s_mask]))
    b_reference = float(np.median(b_roi[low_s_mask]))

    return a_reference, b_reference

def get_chroma_threshold(chroma_dist: NDArray[np.float32]) -> float:
    median = np.median(chroma_dist)

    absolute_deviation = np.abs(chroma_dist - median)
    mad = np.median(absolute_deviation)

    threshold = median + (AB_MAD_FACTOR * mad)

    return float(np.clip(threshold, AB_DIST_MIN, AB_DIST_MAX))

def create_specular_mask(
    clipped_roi: NDArray[np.uint8] | None
) -> NDArray[np.uint8] | None:
    if clipped_roi is None:
        return None
    
    roi_f = clipped_roi.astype(np.float32) / 255.0

    h, w = roi_f.shape[:2]

    roi_f = cv2.bilateralFilter(roi_f, d=5, sigmaColor=50, sigmaSpace=50)

    hsv_f = cv2.cvtColor(roi_f, cv2.COLOR_BGR2HSV)
    lab_f = cv2.cvtColor(roi_f, cv2.COLOR_BGR2LAB)

    s_roi = cast(NDArray[np.float32], hsv_f[:, :, 1])

    l_roi = cast(NDArray[np.float32], lab_f[:, :, 0])
    a_roi = cast(NDArray[np.float32], lab_f[:, :, 1])
    b_roi = cast(NDArray[np.float32], lab_f[:, :, 2])

    s_threshold = min(S_THRESHOLD_MAX, np.percentile(s_roi, 20.0))

    l_threshold = estimate_glare_l_threshold_peak(l_roi)

    a_reference, b_reference = get_adaptive_white_point(a_roi, b_roi, s_roi)

    chroma_distention = np.sqrt((a_roi - a_reference)**2 + (b_roi - b_reference)**2)

    ab_threshold = get_chroma_threshold(chroma_distention)

    mask_s = s_roi < s_threshold

    mask_l = l_roi > l_threshold
    mask_ab = chroma_distention < ab_threshold

    candidates = np.logical_and(mask_l, mask_s)
    candidates = np.logical_and(candidates, mask_ab)

    candidates = (candidates * 255).astype(np.uint8)

    k_w_calc = int(float(w) * MORPH_KERNEL_RATIO_W)
    k_x = min(MORPH_KERNEL_MAX_PX, max(MORPH_KERNEL_MIN_PX, k_w_calc))

    k_y = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_x | 1, k_y))
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))  #1/4[W]のみを想定 pxが20x50程度のため
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(candidates)

    roi_area = float(h * w)

    p99 = float(np.percentile(l_roi, 99.0))
    glare_peak_floor = max(p99 * L_CHECK_RATIO, L_CHECK_MIN)

    for contrast in contours:
        x, y, contrast_w, contrast_h = cv2.boundingRect(contrast)

        if (float(contrast_w * contrast_h)) < (roi_area * 0.0001):
            continue
        
        aspect = float(contrast_w) / float(contrast_h)
        if aspect < 0.1:
            continue
        
        if contrast_h > (h * 0.8):
            continue
        
        temp_mask = np.zeros_like(mask)
        cv2.drawContours(temp_mask, [contrast], -1, 255, -1)

        _, max_val, _, _ = cv2.minMaxLoc(l_roi, mask=temp_mask)
        if max_val < glare_peak_floor:
            continue
        
        cv2.drawContours(mask, [contrast], -1, 255, -1)

    return cast(NDArray[np.uint8], mask)

def select_best_segment(
    clipped_roi: NDArray[np.uint8] | None,
    glare_mask: NDArray[np.uint8] | None
) -> tuple[NDArray[np.float32] | None, float]:
   
    if clipped_roi is None or glare_mask is None:
       return None, 0.0

    roi_f = clipped_roi.astype(np.float32) / 255.0

    h, w = roi_f.shape[:2]
    roi_area = h * w

    NUM_SEGMENTS = max(1, h // MIN_SPLIT_SEGMENT)

    glare_mask_ratio = cv2.countNonZero(glare_mask) / roi_area

    lab_f = cv2.cvtColor(roi_f, cv2.COLOR_BGR2LAB)

    segment_h = h / NUM_SEGMENTS

    center_index = (NUM_SEGMENTS - 1) / 2.0
    max_dist_norm = NUM_SEGMENTS / 2.0

    stats_list: List[SegmentStats] = []
    ab_variance_list: List[float] = []

    for i in range(NUM_SEGMENTS):
        y_top = int(i * segment_h)
    
        if i < NUM_SEGMENTS - 1:
            y_bottom = int((i + 1) * segment_h)
        else:
            y_bottom = h
        
        if y_bottom <= y_top:
            continue
        
        glare_mask_seg = glare_mask[y_top:y_bottom, :]
        lab_f_seg = lab_f[y_top:y_bottom, :]

        segment_area = (y_bottom - y_top) * w
        
        glare_seg_pixels = cv2.countNonZero(glare_mask_seg)
        valid_pixels = segment_area - glare_seg_pixels

        glare_pixels_ratio = glare_seg_pixels / segment_area
        valid_pixels_ratio = valid_pixels / segment_area if segment_area > 0 else 0.0

        valid_glare_mask = cv2.bitwise_not(glare_mask_seg)

        if valid_pixels > 0:
            mean, std_deviation = cv2.meanStdDev(lab_f_seg, mask=valid_glare_mask)

            a = std_deviation[1][0] ** 2
            b = std_deviation[2][0] ** 2

            ab_variance = a + b
        else:
            ab_variance = 9999.0    #ペナルティ
        
        if ab_variance < 5000.0:    #まともな値のみを採用
            ab_variance_list.append(ab_variance)

        dist_from_center = abs(i - center_index)

        weight = 1.0 - 0.4 * (dist_from_center / max_dist_norm)

        stats_list.append(SegmentStats(i, y_top, y_bottom, float(valid_pixels_ratio), float(glare_pixels_ratio), weight, float(ab_variance), 0, 0))

        #最もいいセグメントを選択
        #優先順位 中央付近 有効ピクセル数 = roi_area * 0.2に近ければ近いほどOK((roi_are * 0.2) * 0.8)程度でOKとしておく
        #分散 後で
        #有効ピクセル数でソート
    
    if len(ab_variance_list) > 1:
        ab_variance_median = np.median(ab_variance_list)

        # mad : 中央絶対偏差
        ab_variance_mad = np.median(np.abs(np.array(ab_variance_list) - ab_variance_median))

        if ab_variance_mad < 1e-6:
            ab_variance_mad = 1.0
    else:
        ab_variance_median = 0.0
        ab_variance_mad = 1.0

    glare_ratios = [s.glare_mask_ratio for s in stats_list]
    glare_reference = np.percentile(glare_ratios, 20)
    
    for l in stats_list:
        #Quality Weight via Robust Z
        if l.ab_variance < AB_VALIANCE_CLIP:
            z_score = abs(l.ab_variance - ab_variance_median) / ab_variance_mad
        else:
            z_score = 999.0

        l.z_score = float(z_score)

#        if l.glare_mask_ratio > 0.05:
#            glare_factor = 0.0
#        elif l.glare_mask_ratio > 0.0:
#            glare_factor = 0.5
#        else:
#            glare_factor = 1.0

        glare_factor = np.exp(-(l.glare_mask_ratio - glare_reference) / (glare_reference + 1e-3))
        if l.glare_mask_ratio < glare_reference:
            glare_factor = 1.0
        
        if l.z_score > 3.5:
            l.quality_score = 0.0
        else:
            ab_variance_factor = np.exp(-0.5 * l.z_score)
        
            l.quality_score = l.valid_pixels_ratio * l.weight * ab_variance_factor * glare_factor
        
    stats_list.sort(key=lambda x: x.quality_score, reverse=True)

    #debug
    for l in stats_list:
        print(l)

    return cast(NDArray[np.float32], lab_f[stats_list[0].y_top:stats_list[0].y_bottom, :]), stats_list[0].valid_pixels_ratio

def compress_segment_to_one_line(segment_lab: NDArray[np.float32]) -> NDArray[np.float32]:
    h, w, c = segment_lab.shape

    one_line = np.zeros((1, w, c), dtype=np.float32)
    
    cols = np.transpose(segment_lab, (1, 0, 2))

    for x in range(w):
        col_pixels = cols[x]

        l_vals = col_pixels[:, 0]
        
        med_l = np.median(l_vals)
        
        abs_diff = np.abs(l_vals - med_l)
        mad_l = np.median(abs_diff)
        
        if mad_l < 1e-6:
            mad_l = 1.0
        
        # 係数 0.6745 は正規分布換算用
        modified_z_score = 0.6745 * abs_diff / mad_l
        
        valid_mask = modified_z_score <= 3.5
        
        if np.count_nonzero(valid_mask) < 3:
            final_pixels = col_pixels
        else:
            final_pixels = col_pixels[valid_mask]

        one_line[0, x, 0] = np.median(final_pixels[:, 0])
        one_line[0, x, 1] = np.median(final_pixels[:, 1])
        one_line[0, x, 2] = np.median(final_pixels[:, 2])

    return one_line

def analyze_resistor_background(
    lab_one_line: NDArray[np.float32],
    valid_pixels_ratio: float
) -> BackGroundStats:

    w = lab_one_line.shape[1]

    pixels = lab_one_line[0]

    ab_pixels = pixels[:, 1:]

    #diff: (W, 1, 2) - (1, W, 2) -> (W, W, 2)
    diff = ab_pixels[:, np.newaxis, :] - ab_pixels[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))

    k_neighbor_index = int(w * valid_pixels_ratio * 0.3)
    if k_neighbor_index < 1:
        k_neighbor_index = 1
    if k_neighbor_index >= w:
        k_neighbor_index = w - 1
    
    sorted_dists = np.sort(dist_matrix, axis=1)

    tightness_scores = sorted_dists[:, k_neighbor_index]
    peak_index = int(np.argmin(tightness_scores))

    peak_dist = sorted_dists[peak_index]

    core_limit = int(w * 0.4)
    if core_limit < 1:
        core_limit = 1
    
    back_ground_spread = np.median(peak_dist[: core_limit])

    color_threshold = max(3.0, back_ground_spread * 2.5)

    dist_from_peak = dist_matrix[peak_index]

    back_ground_mask_bool = dist_from_peak < color_threshold
    back_ground_mask = back_ground_mask_bool.astype(np.uint8) * 255
    
    occupancy = float(np.count_nonzero(back_ground_mask)) / float(w)

    back_ground_pixels_lab = pixels[back_ground_mask_bool]

    if len(back_ground_pixels_lab) > 0:
        back_ground_mean_l = float(np.median(back_ground_pixels_lab[:, 0]))
        back_ground_mean_a = float(np.median(back_ground_pixels_lab[:, 1]))
        back_ground_mean_b = float(np.median(back_ground_pixels_lab[:, 2]))

        a_variance = np.var(back_ground_pixels_lab[:, 1])
        b_variance = np.var(back_ground_pixels_lab[:, 2])

        ab_variance = float(a_variance + b_variance) 
    else:
        back_ground_mean_l = 0.0
        back_ground_mean_a = 0.0
        back_ground_mean_b = 0.0

        ab_variance = 9999.0
    
    is_reliable = (occupancy >= 0.40) and (ab_variance < 200.0)

    mask_reshaped = back_ground_mask.reshape(1, w)

    result: BackGroundStats = BackGroundStats(
        (back_ground_mean_l, back_ground_mean_a, back_ground_mean_b),
        occupancy,
        ab_variance,
        mask_reshaped,
        is_reliable
    )

    return result

def detect_bands(
    lab_one_line: NDArray[np.float32],
    back_ground_stats: BackGroundStats,
    visualize: bool = False
) -> Tuple[List[DetectBand], dict]:

    w_img = lab_one_line.shape[1]

    pixels = lab_one_line[0]

    #SSD
    pixel_ab = pixels[:, 1:]
    back_ground_ab = np.array(back_ground_stats.color_lab[1:])

    diff_sq_ab = np.sum((pixel_ab - back_ground_ab)**2, axis=1)

    back_ground_l = back_ground_stats.color_lab[0]
    pixel_l = pixels[:, 0]

    delta_l = back_ground_l - pixel_l

    diff_l_dark = np.maximum(0, delta_l)

    diff_l_bright = np.minimum(np.maximum(0, -delta_l), 30.0)

    diff_sq_total = diff_sq_ab + (1.0 * diff_l_dark) ** 2 + (0.8 * diff_l_bright) ** 2
        
    ssd_signal = np.sqrt(diff_sq_total)

    #scale
    ssd_pre_smooth = gaussian_filter1d(ssd_signal, sigma=1.0)

    raw_mad = np.median(np.abs(ssd_pre_smooth - np.median(ssd_pre_smooth)))
    if raw_mad < 1.0:
        raw_mad = 1.0
    
    prov_peaks, _ = find_peaks(ssd_pre_smooth, prominence=raw_mad * 3.0, distance=3)

    if len(prov_peaks) > 1:
        peak_diffs = np.diff(prov_peaks)

        median_peak_dist = np.median(peak_diffs)
    else:
        median_peak_dist = max(10.0, w_img / 15.0)
    
    #ベースライン補正
    base_line_window = int(median_peak_dist * 1.3) | 1

    base_line = grey_opening(ssd_signal, size=base_line_window)

    signal_corrected = ssd_signal - base_line
    signal_smooth = gaussian_filter1d(signal_corrected, sigma=1.0)

    #パラメータ
    back_ground_std = np.sqrt(back_ground_stats.ab_variance)
    signal_mad = np.median(np.abs(signal_smooth - np.median(signal_smooth)))

    noise_floor = max(back_ground_std * 1.2, signal_mad * 1.5)

    #Savitzky-Golay
    sg_window_len = int(median_peak_dist * 0.3) | 1
    if sg_window_len < 5:
        sg_window_len = 5
        
    match_tolerance = max(5.0, median_peak_dist * 0.4)

    #強度ピーク Amplitude
    signal_a = savgol_filter(signal_smooth, window_length=sg_window_len, polyorder=2, deriv=0)

    peaks_a, props_a = find_peaks(signal_a, height=noise_floor, prominence=noise_floor * 0.3, distance=sg_window_len * 0.6)

    #形状ピーク Derivative Zero-Crossing
    deriv_window = sg_window_len
    signal_b_deriv = savgol_filter(signal_smooth, window_length=deriv_window, polyorder=2, deriv=1)

    #print(f"debug signal_b_deriv = {signal_b_deriv}")

    deriv_median = np.median(signal_b_deriv)
    deriv_abs_dev = np.abs(signal_b_deriv - deriv_median)

    deriv_mad = np.median(deriv_abs_dev)
    if deriv_mad < 1e-6:
        #0.6745 正規分布換算
        deriv_mad = np.std(signal_b_deriv) * 0.6745

    slope_epsilon = 0.01

    zero_crossings_b = []

    for i in range(1, w_img - 1):
        d_prev = signal_b_deriv[i - 1]
        d_next = signal_b_deriv[i + 1]

        #print(f"debug d_prev = {d_prev}, d_next = {d_next}")

        if d_prev > 0 and d_next < 0:
            #if abs(d_prev - d_next) > slope_epsilon:
            if d_prev > slope_epsilon or d_next < -slope_epsilon:
                zero_crossings_b.append(i)
    
    zero_crossings_b = np.array(zero_crossings_b)

    #print(f"debug : zero_crossings_b count = {len(zero_crossings_b)}")

    #統合
    result_bands = []
    
    if len(peaks_a) > 0:
        widths, width_heights, left_ips, right_ips = peak_widths(signal_a, peaks_a, rel_height=0.6)
    
    #print(f"debug : peaks_a = {peaks_a}")
        
    for i, x_a in enumerate(peaks_a):
        #print("debug info : code[for i, x_a in enumerate(peaks_a):]")
        #print(f"debug zero_crossings_b = {zero_crossings_b}")

        valid_b = False

        x_left = left_ips[i]
        x_right = right_ips[i]
        
        if len(zero_crossings_b) > 0:
            is_within_width = (zero_crossings_b >= x_left) & (zero_crossings_b <= x_right)
            if np.any(is_within_width):
                valid_b = True
        
        if valid_b:
            index_start = max(0, int(np.floor(left_ips[i])))
            index_end = min(w_img, int(np.ceil(right_ips[i])))

            band_width = index_end - index_start

            if band_width < 2:
                continue
        
            band_pixels = pixels[index_start:index_end]

            mean_l = float(np.median(band_pixels[:, 0]))
            mean_a = float(np.median(band_pixels[:, 1]))
            mean_b = float(np.median(band_pixels[:, 2]))

            result_bands.append(DetectBand(
                int(x_a),
                int(band_width),
                (mean_l, mean_a, mean_b),
                float(signal_a[x_a])
            ))
    
    debug_data = {
        "ssd_original": ssd_signal,
        "baseline": base_line,
        "signal_corrected": signal_corrected,
        "signal_a": signal_a,
        "signal_b_deriv": signal_b_deriv,
        "peaks_a": peaks_a,
        "zeros_b": zero_crossings_b,
        "noise_floor": noise_floor,
        "slope_epsilon": slope_epsilon,
        "median_dist": median_peak_dist # 確認用
    }

    return result_bands, debug_data

def scan_bands_directional(
    candidate_bands: List[DetectBand],
    pitch: float,
    is_reverse: bool
) -> List[DetectBand]:
    
    if is_reverse:
        work_list = list(reversed(candidate_bands))
    else:
        work_list = list(candidate_bands)
    
    valid_chain = [work_list[0]]
    last_x = work_list[0].x

    for i in range(1, len(work_list)):
        b = work_list[i]
        dist = abs(b.x - last_x)

        if pitch == 0:
            ratio = 0
        else:
            ratio = dist / pitch
        
        if 0.75 <= ratio <= 2.5:   #金のバンドをわざと離している抵抗器もあるので離れているのは有効とみなす
            valid_chain.append(b)

            last_x = b.x
    
    if is_reverse:
        valid_chain.reverse()
    
    return valid_chain

def refine_band_candidates(
    bands: List[DetectBand],
    roi_width: int
) -> List[DetectBand]:
    
    count = len(bands)

    if count == 0:
        return []
    
    bands.sort(key=lambda b: b.x)

    if count == 1:
        band = bands[0]

        TOLERANCE: float = 0.15

        margin = roi_width * TOLERANCE
        if abs(band.x - roi_width / 2) < margin:
            return [band]
        else:
            return []
    if count == 2:
        return bands
    
    x_coords = np.array([b.x for b in bands])
    diffs = np.diff(x_coords)
    median_pitch = np.median(diffs)

    average_width = int(np.mean([b.band_width for b in bands]))
    most_small_width = [b for b in bands if b.band_width < average_width * 0.5]
    if most_small_width is not []:
        for most_small in most_small_width:
            bands.remove(most_small)

    forward_result = scan_bands_directional(bands, float(median_pitch), is_reverse=False)
    backward_result = scan_bands_directional(bands, float(median_pitch), is_reverse=True)

    if len(forward_result) >= len(backward_result):
        refine_bands = forward_result
    else:
        refine_bands = backward_result

    target_side:str = ""
    
    if len(refine_bands) == 3:
        if len(refine_bands) > 1:
            refine_band_x_coords = np.array([b.x for b in refine_bands])

            result_pitch = np.mean(np.diff(refine_band_x_coords))
        else:
            result_pitch = median_pitch

        first_band_x = refine_bands[0].x
        last_band_x = refine_bands[-1].x

        left_margin = first_band_x
        right_margin = roi_width - last_band_x

        predicted_band = None

        if left_margin > right_margin:
            target_side = "left"
        else:
            target_side = "right"
    
        if target_side == "left":
            pred_x = int(float(first_band_x) - result_pitch)

            if pred_x > 0:
                predicted_band = DetectBand(
                    pred_x, average_width, (0,0,0), 0.0
                )
            else:
                print(f"fail left {pred_x}, {roi_width}, {target_side}, {left_margin}, {right_margin}")
        elif target_side == "right":
            pred_x = int(float(last_band_x) + result_pitch)

            if pred_x < roi_width:
                print(f">> Extrapolating RIGHT band at {pred_x}")
                predicted_band = DetectBand(
                    pred_x, average_width, (0,0,0), 0.0
                )
            else:
                print(f"fail right {pred_x}, {roi_width}")

        if predicted_band:
            refine_bands.append(predicted_band)
            refine_bands.sort(key=lambda b: b.x)

    return refine_bands


if __name__ == "__main__":
    main()
        
