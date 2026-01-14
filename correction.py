#!/bin#!/bin/env python3

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from cv2.typing import MatLike
from typing import cast, List, TypedDict, Tuple, Optional
from enum import Enum, auto

IMG_INPUT_DIR:str = "./output_resistors"
IMG_OUTPUT_DIR:str = "./results"

MAX_BASE_SATURATION_SQ: float = 2500.0 
MAX_REFLECTION_SATURATION_SQ: float = 400.0

MIN_EXPECTED_ASPECT_AFTER_ALIGNMENT:float = 2.5

class ClusterTag(Enum):
    BASE_CANDIDATE = auto()
    REFLECTION = auto()
    NOISE = auto()
    UNCERTAIN = auto()
    
class ClusterStat(TypedDict):
    id: int
    center_ab: NDArray[np.float32]
    ratio: float    #占有率
    mean_L: float   #平均輝度
    var_ab: float   #色の分散
    dist_from_origin: float
    tag: ClusterTag #排他タグ

class GlobalImageStats(TypedDict):
    l_mean: float
    l_std: float
    global_mse: float
    reflection_threshold: float
    noise_var_threshold: float

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

        # ---------------------------------------------------------
        # パイプライン実行 (手動構成)
        # ※ analyze_color_clusters_pipeline は入力がBGR前提だが、
        #    preprocess_for_color_analysis はLabを返すため、ここで個別に繋ぐ
        # ---------------------------------------------------------

        # Step A: 前処理 (Lab空間での平滑化)
        lab_roi = preprocess_for_color_analysis(clipped_roi)
        if lab_roi is None: continue

        # Step B: 正規化 (Auto-WB対策) 
        # get_normalized_lab はBGR->Lab変換を含むため使わず、ここで直接計算する
        l_flat = lab_roi[:, :, 0].reshape((-1)).astype(np.float32)
        ab_channels = lab_roi[:, :, 1:3]
        data_full = ab_channels.reshape((-1, 2)).astype(np.float32)
        
        ab_bias = np.median(data_full, axis=0)
        data_normalized = data_full - ab_bias

        # Step C: クラスタリング
        kmeans_res = execute_kmeans_clustering(data_normalized, k=12, sample_step=4)
        if kmeans_res is None:
            print(f"  -> K-means Failed: {file.name}")
            continue
        
        centers, labels, nearest_dist_sq = kmeans_res

        # Step D: 統計算出
        raw_stats, global_stats = calculate_raw_stats(
            l_flat, nearest_dist_sq, labels, centers, k=12
        )

        # Step E: 分類 (タグ付け)
        classified_stats = classify_clusters(raw_stats, global_stats)

        # ---------------------------------------------------------
        # 結果の可視化 (Visualization)
        # ---------------------------------------------------------
        
        # 1. 量子化画像の再構築 (確認用)
        # 正規化された色(centers)にバイアス(ab_bias)を足して元の色に戻す
        centers_denorm = centers + ab_bias
        centers_uint8 = np.clip(centers_denorm, 0, 255).astype(np.uint8)
        
        # ラベルマップを画像化
        label_map = labels.reshape((h, w))
        ab_reconstructed = centers_uint8[label_map] # (H, W, 2)
        
        # Lチャンネルは元の画像を維持して結合
        lab_reconstructed = np.zeros_like(lab_roi)
        lab_reconstructed[:, :, 0] = lab_roi[:, :, 0]
        lab_reconstructed[:, :, 1:3] = ab_reconstructed
        img_quantized = cv2.cvtColor(lab_reconstructed, cv2.COLOR_Lab2BGR)

        # 2. 分析結果マスク (Semantic Mask)
        # タグごとに色を変えて塗りつぶす
        # BASE: 緑, REFLECTION: 赤, NOISE: 黄, UNCERTAIN: グレー
        img_analysis = np.zeros((h, w, 3), dtype=np.uint8)
        
        TAG_COLORS = {
            ClusterTag.BASE_CANDIDATE: (0, 255, 0),   # Green
            ClusterTag.REFLECTION:     (0, 0, 255),   # Red
            ClusterTag.NOISE:          (0, 255, 255), # Yellow
            ClusterTag.UNCERTAIN:      (128, 128, 128) # Gray
        }

        # 各クラスタIDに対応する色を決定
        # id -> color のルックアップテーブル作成
        color_lut = np.zeros((12, 3), dtype=np.uint8)
        for stat in classified_stats:
            cid = stat['id']
            color = TAG_COLORS.get(stat['tag'], (0, 0, 0))
            color_lut[cid] = color
            
        # ファンシーインデックスで一括着色
        img_analysis = color_lut[label_map]

        # ---------------------------------------------------------
        # 保存
        # ---------------------------------------------------------
        # 左から [元画像] [量子化] [判定結果] を連結
        combined = cv2.hconcat([clipped_roi, img_quantized, img_analysis])
        
        # 視認性のため4倍に拡大 (Nearest Neighborでドット感を維持)
        combined_large = cv2.resize(combined, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

        output_filename = save_dir / f"result_{file.name}"
        cv2.imwrite(str(output_filename), combined_large)
        print(f"  -> Saved: {output_filename.name}")
        
def check_resistor_roi_quality(roi: NDArray[np.uint8] | None) -> bool:
    """check_resistor_roi_quality

    Args:
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

def preprocess_for_reflection(roi: NDArray[np.uint8] | None) -> MatLike | None:
    """preprocess_for_reflaction
    エッジを残してノイズ除去を実行
    バイラテラルフィルタ

    Args:
        roi (NDArray[np.uint8] | None): クオリティチェック・クリップ済みのROI

    Returns:
        NDArray: バイラテラルフィルタをかけた後のROI
    """

    if roi is None:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    v = hsv[:, :, 2]

    v_filtered = cv2.bilateralFilter(v, d=5, sigmaColor=40.0, sigmaSpace=40.0)

    hsv[:, :, 2] = v_filtered
    
    return hsv

def preprocess_for_color_analysis(roi: NDArray[np.uint8] | None) -> MatLike | None:
    """preprocess_for_color_analysis
    エッジを残してノイズ除去を実行
    バイラテラルフィルタ

    Args:
        roi (NDArray[np.uint8] | None): クオリティチェック・クリップ済みのROI

    Returns:
        NDArray: バイラテラルフィルタをかけた後のROI
    """

    if roi is None:
        return None

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)

    ab = lab[:, :, 1:3]

    ab_filtered = cv2.bilateralFilter(ab, d=3, sigmaColor=15.0, sigmaSpace=15.0)

    lab[:, :, 1:3] = ab_filtered
    
    return lab

def get_normalized_lab(
    filtered_roi: MatLike | None
) -> Optional[Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]]:
    """get_normalized_lab

    Args:
        filtered_roi (MatLike): _description_

    Returns:
        Optional[Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]]: _description_
    """
    
    if filtered_roi is None:
        return None
    
    lab_image = cv2.cvtColor(filtered_roi, cv2.COLOR_BGR2Lab)

    l_flat = lab_image[:, :, 0].reshape((-1)).astype(np.float32)

    ab_channels = lab_image[:, :, 1:3]

    data_full = ab_channels.reshape((-1, 2)).astype(np.float32)

    ab_bias = np.median(data_full, axis=0)
    
    data_normalized = data_full - ab_bias

    return l_flat, data_normalized, ab_bias

def execute_kmeans_clustering(
    data_normalized: NDArray[np.float32] | None,
    k: int,
    sample_step: int
) -> Optional[Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32]]]:
    """
    K-meansを実行し、ラベルと「最近傍距離」のみを返す。
    巨大な距離行列 (N, K) は関数内で破棄し、メモリを節約する。
    
    Returns:
        Optional: 失敗時は None を返す (呼び出し側でハンドリング必須)
        Success: (centers, labels_full, nearest_dist_sq)
            - centers: (K, 2)
            - labels_full: (N,)
            - nearest_dist_sq: (N,) 各画素の所属クラスタ中心までの距離の二乗
    """
    if data_normalized is None:
        return None

    h_times_w = data_normalized.shape[0]
    
    # ガード: サンプルステップ適正化
    safe_step = max(1, min(sample_step, h_times_w // 1000))
    data_sampled = data_normalized[::safe_step]
    
    if data_sampled.shape[0] < k:
        return None

    # K-means実行
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers_mat = cv2.kmeans(data_sampled, k, cast(MatLike, None), criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    centers: NDArray[np.float32] = centers_mat # type: ignore

    dists_sq_matrix = np.sum((data_normalized[:, None] - centers[None, :])**2, axis=2)
    
    labels_full = np.argmin(dists_sq_matrix, axis=1).astype(np.int32)
    nearest_dist_sq = np.min(dists_sq_matrix, axis=1) # (N,) 所属クラスタまでの距離だけでいい
    
    # dists_sq_matrix はここで解放される
    return centers, labels_full, nearest_dist_sq

def calculate_raw_stats(
    l_flat: NDArray[np.float32],
    nearest_dist_sq: NDArray[np.float32], # (N,) のみ受け取る
    labels_full: NDArray[np.int32],
    centers: NDArray[np.float32],
    k: int
) -> Tuple[List[ClusterStat], GlobalImageStats]:
    """
    統計算出。巨大な行列を受け取らず、最近傍距離ベクトルのみを使用する。
    """
    total_pixels = l_flat.shape[0]
    cluster_stats: List[ClusterStat] = []
    
    l_mean = float(np.mean(l_flat))
    l_std = float(max(np.std(l_flat), 1e-5))
    
    # Global MSE計算も軽量化済み
    global_mse = float(max(np.mean(nearest_dist_sq), 10.0))
    
    global_stats: GlobalImageStats = {
        "l_mean": l_mean,
        "l_std": l_std,
        "global_mse": global_mse,
        "reflection_threshold": l_mean + (2.0 * l_std),
        "noise_var_threshold": global_mse * 2.5
    }
    
    for i in range(k):
        mask_flat = (labels_full == i)
        count = np.count_nonzero(mask_flat)
        if count == 0: continue
            
        ratio = float(count / total_pixels)
        mean_l = float(np.mean(l_flat[mask_flat]))
        
        # クラスタ内分散: すでに計算済みの nearest_dist_sq を使うだけ
        var_ab = float(np.mean(nearest_dist_sq[mask_flat]))
        
        dist_from_origin = float(np.sum(centers[i]**2))
        
        stat: ClusterStat = {
            "id": i,
            "center_ab": centers[i],
            "ratio": ratio,
            "mean_L": mean_l,
            "var_ab": var_ab,
            "dist_from_origin": dist_from_origin,
            "tag": ClusterTag.UNCERTAIN
        }
        cluster_stats.append(stat)
        
    return cluster_stats, global_stats

def classify_clusters(
    cluster_stats: List[ClusterStat],
    global_stats: GlobalImageStats
) -> List[ClusterStat]:
    """
    統計に基づきタグ付けを行う。
    """
    for stat in cluster_stats:
        # REFLECTION
        if (stat['mean_L'] > global_stats['reflection_threshold'] and 
            stat['dist_from_origin'] < MAX_REFLECTION_SATURATION_SQ):
            stat['tag'] = ClusterTag.REFLECTION
            
        # NOISE
        elif (stat['ratio'] < 0.005 and 
              stat['var_ab'] > global_stats['noise_var_threshold']):
            stat['tag'] = ClusterTag.NOISE

    cluster_stats.sort(key=lambda x: x["ratio"], reverse=True)

    # BASE_CANDIDATE (複数許容)
    for stat in cluster_stats:
        if stat['tag'] != ClusterTag.UNCERTAIN:
            continue
            
        if (stat['ratio'] >= 0.10 and 
            stat['var_ab'] <= global_stats['global_mse'] * 1.5 and
            stat['dist_from_origin'] <= MAX_BASE_SATURATION_SQ):
            
            stat['tag'] = ClusterTag.BASE_CANDIDATE
    
    return cluster_stats

def analyze_color_clusters_pipeline(
    filtered_bgr: MatLike, 
    k: int = 12
) -> Optional[Tuple[NDArray[np.int32], List[ClusterStat], NDArray[np.float32]]]:
    """
    統合パイプライン。K-means失敗時は空の結果を返す。
    """
    # 1. 変換 & 正規化
    ret = get_normalized_lab(filtered_bgr)
    if ret is not None:
        l_flat, data_normalized, ab_bias = ret
    else:
        return None
    
    # 2. クラスタリング (Optional対応)
    kmeans_result = execute_kmeans_clustering(data_normalized, k, sample_step=4)
    
    # 失敗時のハンドリング (明示的)
    if kmeans_result is None:
        h, w = filtered_bgr.shape[:2]
        # 空の結果を返して、呼び出し元でスキップさせる
        return np.zeros((h, w), dtype=np.int32), [], ab_bias

    centers, labels, nearest_dist_sq = kmeans_result

    # 3. 統計 (Lightweight)
    stats, global_stats = calculate_raw_stats(l_flat, nearest_dist_sq, labels, centers, k)
    
    # 4. 分類
    classified_stats = classify_clusters(stats, global_stats)
    
    h, w = filtered_bgr.shape[:2]
    label_map = labels.reshape((h, w))
    
    return label_map, classified_stats, ab_bias
    

if __name__ == "__main__":
    main()
        
