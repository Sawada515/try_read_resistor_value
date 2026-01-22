import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample

# --- 設定 ---
CSV_PATH = "dataset_color_features.csv"
MODEL_SAVE_PATH = "resistor_color_svm.pkl"

# 特徴量カラム
FEATURE_COLS = ["mean_l", "mean_a", "mean_b", "mean_c", "std_l"]

# クラス定義 (ユーザー仕様)
COLOR_NAMES = {
    0: "BLACK",
    1: "BROWN",
    2: "RED",
    3: "ORANGE",
    4: "YELLOW",
    5: "GREEN",
    6: "BLUE",
    7: "PURPLE",
    8: "GRAY",
    9: "WHITE",
    10: "GOLD_DARK",
    11: "GOLD_LITE"
}

def train_svm_with_balancing():
    # 1. データ読み込み
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"エラー: {CSV_PATH} が見つかりません。")
        return

    # ラベルが空の行を削除 & 整数変換
    df = df.dropna(subset=['label_id'])
    try:
        df['label_id'] = df['label_id'].astype(int)
    except ValueError:
        print("エラー: label_id に数値以外のデータが含まれています。")
        return

    # 指定外のクラスIDが含まれていないかチェック
    valid_ids = set(COLOR_NAMES.keys())
    present_ids = set(df['label_id'].unique())
    unknown_ids = present_ids - valid_ids
    if unknown_ids:
        print(f"【警告】定義にないラベルIDが含まれています: {unknown_ids}")
        print("これらは除外せず学習されますが、COLOR_NAMESに登録されていません。")

    print("--- 水増し前のデータ分布 ---")
    print(df['label_id'].value_counts().sort_index().rename(index=COLOR_NAMES))

    if len(present_ids) < 2:
        print("\n【エラー】ラベルが1種類しかありません。学習には2種類以上のデータが必要です。")
        return

    # 2. データの水増し (Upsampling)
    # 最多クラスのデータ数に合わせる
    max_samples = df['label_id'].value_counts().max()
    
    df_upsampled_list = []
    
    for label in present_ids:
        df_class = df[df['label_id'] == label]
        
        # データ数がmax_samples未満なら水増し
        if len(df_class) < max_samples:
            df_class_upsampled = resample(
                df_class,
                replace=True,     # 重複許可
                n_samples=max_samples,
                random_state=42
            )
            df_upsampled_list.append(df_class_upsampled)
        else:
            # 既に最多ならそのまま追加
            df_upsampled_list.append(df_class)
    
    df_balanced = pd.concat(df_upsampled_list)
    
    # 3. データ分割
    X = df_balanced[FEATURE_COLS]
    y = df_balanced['label_id']

    # Stratify分割で学習/テストの比率を維持
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. パイプライン構築 (標準化 + SVM)
    # StandardScaler: 平均0, 分散1に変換 (SVMの収束と精度に必須)
    clf = make_pipeline(
        StandardScaler(),
        SVC(
            kernel='rbf',      # 非線形分離
            C=10.0,            # 誤分類ペナルティ大 (境界を厳しく設定)
            gamma='scale',     # カーネル係数の自動調整
            probability=True,  # 確率推定を有効化
            random_state=42
        )
    )

    # 5. 学習
    print(f"\n学習中... (データ数: {len(X_train)})")
    clf.fit(X_train, y_train)
    print("学習完了。")

    # 6. 評価
    print("\n--- 評価結果 ---")
    y_pred = clf.predict(X_test)
    
    print(f"正解率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")

    # レポート用クラス名リスト
    target_names = [COLOR_NAMES.get(i, f"ID_{i}") for i in sorted(present_ids)]
    
    print("\n詳細レポート:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Resistor Colors)')
    plt.show()

    # 7. 保存
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"\nモデルを保存しました: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_svm_with_balancing()
