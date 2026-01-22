import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample # データを水増しするためのツール

# --- 設定 ---
CSV_PATH = "dataset_features.csv"
MODEL_SAVE_PATH = "band_binary_classifier.pkl"

# 特徴量カラム
FEATURE_COLUMNS = [
    "band_rate", "symmetry", "log_norm", "std_l", "energy"
]

def train_model_with_oversampling():
    # 1. データ読み込み
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"エラー: {CSV_PATH} が見つかりません。")
        return

    # ラベル欠損行を削除
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    print(f"--- 水増し前のデータ数 ---")
    print(df['label'].value_counts())

    # ラベル種別チェック
    if df['label'].nunique() < 2:
        print("\n【警告】ラベルが1種類しかありません。0(Band)と1(Shadow)の両方が必要です。")
        return

    # 2. データの水増し (Upsampling)
    # 多数派(バンド=0)と少数派(影=1)に分ける
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    # 少数派を多数派と同じ数になるまでランダムに複製(replace=True)
    df_minority_upsampled = resample(
        df_minority, 
        replace=True,     # サンプルを重複させる
        n_samples=len(df_majority), # 多数派と同じ数まで増やす
        random_state=42
    )

    # 結合
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    print(f"\n--- 水増し後のデータ数 (学習に使用) ---")
    print(df_upsampled['label'].value_counts())

    # 3. 特徴量(X)と正解(y)に分割
    X = df_upsampled[FEATURE_COLUMNS]
    y = df_upsampled['label']

    # 4. 学習用とテスト用に分割 (7:3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 5. 決定木の学習
    # データ数が均衡になったので class_weight='balanced' は不要
    clf = DecisionTreeClassifier(
        max_depth=4,            
        min_samples_leaf=5,     
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    print("\n学習完了。")

    # 6. 精度評価
    y_pred = clf.predict(X_test)
    print("\n--- 評価結果 ---")
    # 水増しデータでの評価なのでスコアは高く出やすいが、
    # 重要なのは「Shadow(1)のRecallが出るようになったか」
    print(classification_report(y_test, y_pred, target_names=['Band (0)', 'Shadow/Noise (1)']))

    print("混同行列:")
    print(confusion_matrix(y_test, y_pred))

    # 7. 特徴量重要度の可視化
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [FEATURE_COLUMNS[i] for i in indices])
    plt.tight_layout()
    plt.show()

    # 8. 決定木の可視化
    plt.figure(figsize=(16, 8))
    plot_tree(
        clf, 
        feature_names=FEATURE_COLUMNS,
        class_names=['Band', 'Shadow/Noise'],
        filled=True, 
        rounded=True,
        fontsize=10
    )
    plt.title("Decision Tree Logic")
    plt.show()

    # 9. モデル保存
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"\nモデルを保存しました: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model_with_oversampling()
