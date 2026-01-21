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
    energy_score: float

IMG_INPUT_DIR:str = "./output_resistors"
IMG_OUTPUT_DIR:str = "./result"

#定数
MIN_EXPECTED_ASPECT_AFTER_ALIGNMENT:float = 2.5

Z_SCORE_THRESHOLD = 1.0

# ノイズモデル: Noise ≈ sqrt( ReadNoise^2 + (Gain * Signal)^2 )
# SHOT_NOISE_FACTOR: 信号強度に比例するノイズ成分 (例: 0.05 = 輝度の5%程度のゆらぎを許容)
SHOT_NOISE_FACTOR = 0.001
# READ_NOISE_EPSILON: 暗部でのゼロ除算防止および最低限のフロアノイズ
READ_NOISE_EPSILON = 2.0

# 無彩色判定の半径 (0-127 scale)
LAB_CHROMA_LIMIT = 30.0

# 局所平均を計算する窓サイズ比率
LOCAL_WINDOW_RATIO = 0.15

MIN_SPLIT_SEGMENT: int = 5

AB_VALIANCE_CLIP = 800.0

#WEIGHT_CHROMA:float = 2.0       # 色差(a,b)重視
WEIGHT_LUMA_DARK:float = 1.2    # 輝度(L)は標準
WEIGHT_LUMA_BRIGHT:float = 0.1  # テカリは抑制

# 幾何パラメータ (Geometry)
# 最小バンド幅: 物理的な最小サイズ（抵抗器の規格や解像度から決まる定数）
# ここでは画像幅の1%としているが、本来は固定ピクセル数が望ましい
MIN_BAND_WIDTH_RATIO:float = 0.01
BASELINE_WINDOW_FACTOR:int = 5   # ベースライン推定窓は最小バンド幅の何倍か

# 閾値パラメータ (Thresholds)
MAD_FACTOR:float = 1.0             # ノイズフロア推定係数
SYSTEM_NOISE_FACTOR:float = 0.6
PROMINENCE_RATIO:float = 1.5       # S/N比として、ノイズの何倍の突出を信号とするか

# 色差判定 (JND: Just Noticeable Difference)
# CIE76においてDelta E < 2.3 は「目視で差が識別できない」とされる
# ここでは安全マージンを取り 3.0 を閾値とする
JAD_AB_THRESHOLD:float = 2.3
#SHADOW_LUMA_THRESHOLD:float = 20.0
SHADOW_LUMA_THRESHOLD:float = 12.5

def main():
    file_count = 0
    
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

        result_bands, debug_data = detect_bands(lab_one_line, back_ground_stats)

        print("band_result start")
        for r_b in result_bands:
            print(r_b)
        print("band_result finish")

        correction_result_bands = refine_band_candidates(result_bands, w)

        #points = [(x.x) for x in correction_result_bands]
        points = [(x.x) for x in result_bands]
        #print(f"w = {w}, h = {h}")
        #print(f"points = {points}")

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


        save_name = f"{file_count:03d}.bmp"
        save_path =  save_dir / save_name

        cv2.imwrite(str(save_path), debug_imgs)

        print(f"save {save_path}")

        file_count += 1

       
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

    trim_w = int(w * 0.005)
    trim_h = int(h * 0.05)

    return roi[trim_h : h-trim_h, trim_w : w-trim_w]


def create_specular_mask(
    clipped_roi: NDArray[np.uint8] | None
) -> NDArray[np.uint8] | None:
    if clipped_roi is None:
        return None
    
    if clipped_roi is None:
        return None

    h, w = clipped_roi.shape[:2]

    # ---------------------------------------------------------
    # 1. Color Space Conversion (Raw Data)
    # ---------------------------------------------------------
    # 前処理フィルタはかけず、ピーク形状を保存する
    lab = cv2.cvtColor(clipped_roi, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    
    l_float = l_ch.astype(np.float32)

    # ---------------------------------------------------------
    # 2. Chromaticity Constraint (Physics)
    # ---------------------------------------------------------
    # 中心(128)からのユークリッド距離で彩度を評価
    a_centered = a_ch.astype(np.float32) - 128.0
    b_centered = b_ch.astype(np.float32) - 128.0
    chroma = np.sqrt(a_centered**2 + b_centered**2)
    
    # Gate 1: 色度による制約 (無彩色であること)
    mask_achromatic = chroma < LAB_CHROMA_LIMIT

    # ---------------------------------------------------------
    # 3. Statistical Peak Detection (Signal-Dependent Z-score)
    # ---------------------------------------------------------
    # 窓サイズ決定
    ksize = int(w * LOCAL_WINDOW_RATIO)
    if ksize % 2 == 0: ksize += 1
    ksize = max(3, ksize)

    # モーメント計算 (BoxFilter)
    # E[X] (局所平均)
    mu = cv2.boxFilter(l_float, -1, (ksize, ksize))
    # E[X^2]
    mu2 = cv2.boxFilter(l_float**2, -1, (ksize, ksize))
    
    # Variance = E[X^2] - (E[X])^2
    variance = mu2 - mu**2
    variance = np.maximum(variance, 0)
    sigma_local = np.sqrt(variance)
    
    # --- ノイズモデルの適用 (修正箇所) ---
    # 定数フロアではなく、信号強度(mu)に依存する項を追加
    # Denominator^2 = σ_local^2 + (k * μ)^2 + ε^2
    # 明部では (k * μ) が支配的になり、Z-scoreの過敏な反応を抑える
    # 暗部では ε が支配的になり、ゼロ除算を防ぐ
    noise_model = (SHOT_NOISE_FACTOR * mu)**2 + READ_NOISE_EPSILON**2
    denominator = np.sqrt(sigma_local**2 + noise_model)
    
    # Z-score = (Signal - Mean) / Estimated_Noise_spread
    z_score = (l_float - mu) / denominator
    
    # Gate 2: 統計的有意性 (局所ピーク判定)
    mask_peak = z_score > Z_SCORE_THRESHOLD

    # ---------------------------------------------------------
    # 4. Integration
    # ---------------------------------------------------------
    # 統計条件(Peak) AND 物理条件(Achromatic)
    candidates = np.logical_and(mask_peak, mask_achromatic)
    candidates_u8 = (candidates * 255).astype(np.uint8)

    # モルフォロジー演算は行わない (微小なピーク情報を保存するため)

    # ---------------------------------------------------------
    # 5. Geometry Check
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(candidates_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(candidates_u8)
    
    # 像として成立する最小面積 (ホットピクセル除去)
    MIN_PIXEL_AREA = 2.0 

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area < MIN_PIXEL_AREA:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw == 0 or ch == 0: continue
        
        # 形状フィルタ (極端なノイズ除去)
        aspect = float(cw) / float(ch)
        if aspect < 0.1:
            continue
            
        cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    print(f"L range: {l_float.min():.1f} - {l_float.max():.1f}")
    print(f"Chroma range: {chroma.min():.1f} - {chroma.max():.1f} (Limit: {LAB_CHROMA_LIMIT})")
    print(f"Z-score range: {z_score.min():.2f} - {z_score.max():.2f} (Thresh: {Z_SCORE_THRESHOLD})")
    print(f"Sigma range: {sigma_local.min():.1f} - {sigma_local.max():.1f}")

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
    l_plane = segment_lab[:, :, 0]

    med_l = np.median(l_plane, axis=0)

    # ブロードキャストにより (H, W) - (W,) の計算が一発で行われます
    abs_diff = np.abs(l_plane - med_l)

    mad_l = np.median(abs_diff, axis=0)

    # ゼロ除算対策: MADが極小の場合は 1.0 に置換 (np.whereで条件分岐もベクトル化)
    mad_l = np.where(mad_l < 1e-6, 1.0, mad_l)

    modified_z_score = 0.6745 * abs_diff / mad_l

    valid_mask = modified_z_score <= 3.5

    # 列ごとの有効画素数をカウント -> shape: (W,)
    valid_counts = np.sum(valid_mask, axis=0)
    
    # フォールバックすべき列のフラグ -> shape: (W,)
    fallback_cols = valid_counts < 3

    # fallback_cols[np.newaxis, :] で (1, W) にして (H, W) と演算
    final_mask = valid_mask | fallback_cols[np.newaxis, :]

    masked_data = segment_lab.copy()
    masked_data[~final_mask] = np.nan

    # np.nanmedian を使い、NaNを無視して縦方向(axis=0)の中央値を計算
    # keepdims=True で (1, W, C) の形状を維持
    one_line = np.nanmedian(masked_data, axis=0, keepdims=True)

    return one_line.astype(np.float32)

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

def calculate_dynamic_chroma_weight(ab_variance: float) -> float:
    # 基準となる標準偏差 (経験的に、綺麗な抵抗器背景は sigma=5.0~8.0 程度)
    REFERENCE_SIGMA = 10.0
    
    # 現在の背景の標準偏差
    current_sigma = np.sqrt(ab_variance)
    if current_sigma < 1.0: current_sigma = 1.0

    # 基準より綺麗なら重みが増え、汚ければ減る
    # ベースを 1.5 とし、最大 2.0倍 程度までブーストさせるイメージ
    weight = 1.5 * (REFERENCE_SIGMA / current_sigma)

    # クランプ処理
    # 最小 1.5: どんなに汚くても「金」などを検出するために最低限の色評価は必要
    # 最大 3.5: 綺麗ならかなり強気攻めてOK
    return float(np.clip(weight, 1.5, 3.5))

def detect_bands(
    lab_one_line: NDArray[np.float32],
    back_ground_stats: BackGroundStats,
    visualize: bool = False
) -> Tuple[List[DetectBand], dict]:

    w_img = lab_one_line.shape[1]
    pixels = lab_one_line[0]

    # --- 1. エネルギー信号計算 (Weighted Euclidean Distance) ---
    # 旧: ssd_signal (名称誤り) -> 新: energy_signal
    # 定義: 背景ベクトルからの「重み付き距離」
    
    # Chroma (Color distance squared)
    pixel_ab = pixels[:, 1:]
    back_ground_ab = np.array(back_ground_stats.color_lab[1:])
    diff_sq_ab = np.sum((pixel_ab - back_ground_ab)**2, axis=1)

    # Luma (Brightness distance squared with asymmetric potential)
    back_ground_l = back_ground_stats.color_lab[0]
    pixel_l = pixels[:, 0]
    delta_l = back_ground_l - pixel_l

    dynamic_weight_chroma = calculate_dynamic_chroma_weight(back_ground_stats.ab_variance)
    
    term_chroma = dynamic_weight_chroma * diff_sq_ab

    term_luma_dark = (WEIGHT_LUMA_DARK * np.maximum(0, delta_l)) ** 2
    term_luma_bright = (WEIGHT_LUMA_BRIGHT * np.minimum(np.maximum(0, -delta_l), 50.0)) ** 2

    # Unit: Distance in weighted Lab space
    energy_signal = np.sqrt(term_chroma + term_luma_dark + term_luma_bright)

    # --- 2. ベースライン補正 ---
    # 物理的な「最小バンド幅」を基準に、それより大きなうねりをベースラインとして除去
    min_band_width_px = max(3, int(w_img * MIN_BAND_WIDTH_RATIO))
    
    baseline_window_size = (min_band_width_px * BASELINE_WINDOW_FACTOR) | 1
    base_line = grey_opening(energy_signal, size=baseline_window_size)
    
    signal_corrected = energy_signal - base_line
    
    # --- 3. ノイズ除去 (Gaussian Filter) ---
    # 修正: SGフィルタ(多項式近似)は廃止。
    # 目的は「センサ起因の高周波ノイズ除去」であり、バンド幅(幾何)とは無関係。
    # sigma=1.0 は一般的なセンサノイズ抑制の標準値。
    signal_smooth = gaussian_filter1d(signal_corrected, sigma=1.0)

    # --- 4. ノイズフロア推定 (MAD) ---
    # ベースライン補正済みのため、信号の中央値は0付近と仮定できるが、
    # 念のため median を引いた偏差を見る (Robust MAD)
    signal_mad = np.median(np.abs(signal_smooth - np.median(signal_smooth)))
    if signal_mad < 1e-6: signal_mad = 1e-6

    # 最小システムノイズのガード (背景分散由来)
    back_ground_std = np.sqrt(back_ground_stats.ab_variance)
    system_noise_floor = back_ground_std * 0.8
    
    estimated_noise_floor = signal_mad * MAD_FACTOR
    final_noise_floor = max(estimated_noise_floor, system_noise_floor)

    # --- 5. ピーク検出 (Prominence Only) ---
    # 修正: height制限を撤廃 (二重拘束の解消)。
    # 「絶対的な高さ」ではなく「周囲のノイズフロアに対してどれだけ突出しているか」のみで判定する。
    min_prominence = final_noise_floor * PROMINENCE_RATIO

    peaks, properties = find_peaks(
        signal_smooth,
        prominence=min_prominence,     # 相対高さのみを見る
        width=min_band_width_px * 0.5, # 物理的な幅制約
        rel_height=0.75
    )

    result_bands = []
    
    if len(peaks) > 0:
        left_ips = properties["left_ips"]
        right_ips = properties["right_ips"]
        
        back_ground_l = back_ground_stats.color_lab[0]
        back_ground_a = back_ground_stats.color_lab[1]
        back_ground_b = back_ground_stats.color_lab[2]

        for i, x_peak in enumerate(peaks):
            x_left = left_ips[i]
            x_right = right_ips[i]
            
            index_start = max(0, int(np.floor(x_left)))
            index_end = min(w_img, int(np.ceil(x_right)))
            band_width = index_end - index_start
            
            if band_width < 2: continue

            band_pixels = pixels[index_start:index_end]
            mean_l = float(np.median(band_pixels[:, 0]))
            mean_a = float(np.median(band_pixels[:, 1]))
            mean_b = float(np.median(band_pixels[:, 2]))

            # --- 影(Shadow)判定 ---
            # a,b 平面での距離 (ユークリッド距離)
            dist_ab = np.sqrt((mean_a - back_ground_a)**2 + (mean_b - back_ground_b)**2)

            delta_l_band = back_ground_l - mean_l
            
            is_low_chroma_diff = (dist_ab < JAD_AB_THRESHOLD)
            is_darker = (delta_l_band > 0)
            
            is_shadow = is_low_chroma_diff and is_darker and (delta_l_band < SHADOW_LUMA_THRESHOLD)

            if is_shadow:
                continue

            result_bands.append(DetectBand(
                x=int(x_peak),
                band_width=int(band_width),
                mean_color_lab=(mean_l, mean_a, mean_b),
                energy_score=float(signal_smooth[x_peak]) 
            ))

    debug_data = {
        "energy_signal": energy_signal, # 変数名変更を反映
        "baseline": base_line,
        "signal_smooth": signal_smooth,
        "noise_floor": final_noise_floor,
        "peaks": peaks
    }

    print(debug_data)

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
        
        #print(f"result pitch = {result_pitch}")
    
        if target_side == "left":
            pred_x = int(float(first_band_x) - result_pitch)

            if pred_x > 0:
                predicted_band = DetectBand(
                    int(result_pitch // 2), average_width, (0,0,0), 0.0
                )
            else:
                pass
                #print(f"fail left {pred_x}, {roi_width}, {target_side}, {left_margin}, {right_margin}")
        elif target_side == "right":
            pred_x = int(float(last_band_x) + result_pitch)

            if pred_x < roi_width:
                predicted_band = DetectBand(
                    int(roi_width - ((result_pitch // 3) * 2)), average_width, (0,0,0), 0.0
                )
            else:
                pass
                #print(f"fail right {pred_x}, {roi_width}")

        if predicted_band:
            refine_bands.append(predicted_band)
            refine_bands.sort(key=lambda b: b.x)

    return refine_bands


if __name__ == "__main__":
    main()
