import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

def order_points(pts):
    """
    4つの座標を [左上, 右上, 右下, 左下] の順に並び替える関数
    (ユーザー指定のオリジナルロジックを維持)
    """
    rect = np.zeros((4, 2), dtype="float32")

    # 左上(sが最小)と右下(sが最大)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上(diffが最小)と左下(diffが最大)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    """
    4点の座標に基づいて画像を切り出し、水平に補正する
    (ユーザー指定のオリジナルロジックを維持)
    """
    # 座標を並び替え
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 幅の最大値を計算
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 高さの最大値を計算
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 変換後の座標（真上から見た長方形）
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 射影変換行列を計算して適用
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def process_resistors_obb(model_path, input_dir, output_dir, conf_threshold=0.5):
    # --- GPU設定 ---
    if torch.cuda.is_available():
        device = 0
        print(f"🚀 GPUを使用します: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ GPUを使用しません (CPUモード)")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading OBB model: {model_path}")
    model = YOLO(model_path, task="obb")

    # 画像検索
    files = list(input_path.rglob("*.bmp"))
    print(f"📂 対象画像数: {len(files)} 枚")

    total_count = 0

    for file_path in files:
        # 画像読み込み
        original_img = cv2.imread(str(file_path))
        if original_img is None: continue

        # 推論実行
        results = model.predict(source=str(file_path), conf=conf_threshold, device=device, verbose=False)
        result = results[0]

        # 検出がなかった場合はスキップ
        if result.obb is None or len(result.obb) == 0:
            print(f"⚠️ No resistor detected: {file_path.name}")
            continue

        # --- 変更箇所: 検出個数分ループ処理に変更 ---
        # 検出された全てのOBB座標を取得 (N, 4, 2)
        obb_list = result.obb.xyxyxyxy.cpu().numpy()

        for i, pts in enumerate(obb_list):
            try:
                # 補正・切り出し実行 (元の関数を使用)
                warped_img = four_point_transform(original_img, pts)

                if warped_img is None or warped_img.size == 0:
                    continue

                # --- 向きの最終調整 (元のロジックを維持) ---
                # 縦長なら回転
                h, w = warped_img.shape[:2]
                if h > w:
                    warped_img = cv2.rotate(warped_img, cv2.ROTATE_90_CLOCKWISE)

                # --- 保存処理 ---
                # 複数検出に対応するため、ファイル名に連番を付与
                # 例: image.bmp -> image_0.bmp, image_1.bmp
                save_name = f"{file_path.stem}_{i}{file_path.suffix}"
                save_full_path = output_path / save_name
                
                cv2.imwrite(str(save_full_path), warped_img)
                # print(f"Saved: {save_name}") # ログが多すぎる場合はコメントアウト
                total_count += 1

            except Exception as e:
                print(f"Error processing {file_path.name} (ID: {i}): {e}")

    print(f"\n完了: 合計 {total_count} 個の抵抗を処理しました。")

if __name__ == "__main__":
    # --- 設定 ---
    MODEL_PATH = "best.pt"
    INPUT_DIR = "./raw_images"
    OUTPUT_DIR = "./results"
    CONFIDENCE = 0.5

    process_resistors_obb(MODEL_PATH, INPUT_DIR, OUTPUT_DIR, CONFIDENCE)
