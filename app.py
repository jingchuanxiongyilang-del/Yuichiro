from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import joblib
from datetime import timedelta
import plotly.graph_objects as go
import base64
import traceback

app = Flask(__name__)

# --- アプリケーションの基本設定 ---
# 各種ファイルが格納されているディレクトリ名
DATA_DIR = "stock_data"
MODEL_DIR = "models"
# 予測に使用する過去の日数（モデルの入力形状に合わせる）
LOOK_BACK = 60
# 予測に使用する特徴量（モデル学習時と同一にする）
FEATURES = ["終値", "出来高", "SMA_20", "SMA_50", "RSI_14"]

# --- 予測対象の銀行株リスト（ティッカーコード: 銘柄名） ---
STOCK_LIST = {
    "8306.T": "三菱UFJ銀行",
    "8411.T": "みずほ銀行",
    "8316.T": "三井住友銀行",
    "8308.T": "りそな銀行",
}


# --- データ読み込みと前処理を行う関数 ---
# --- データ読み込みと前処理を行う関数 ---
def load_and_prepare_data(file_path):
    """CSVファイルを読み込み、必要なカラムのデータ型を整える（最終修正版）"""
    # インデックスを文字列として読み込む
    df = pd.read_csv(file_path, index_col=0)

    # インデックスを強制的に日付型に変換。変換できないものはNaT(Not a Time)にする
    df.index = pd.to_datetime(df.index, errors="coerce")

    # ★★★ この行が重要です ★★★
    # インデックスがNaTになった行（日付として不正だった行）を正しく削除する
    df.dropna(inplace=True)

    # 有効なデータが残っているかチェック
    if df.empty:
        raise ValueError(
            f"ファイル '{os.path.basename(file_path)}' から有効な日付データを読み込めませんでした。"
        )

    # カラム名を日本語に統一
    df.rename(columns={"Close": "終値", "Volume": "出来高"}, inplace=True)

    # 必要なカラムが存在することを確認し、数値型に変換
    for col in ["終値", "出来高"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # 欠損値を前方・後方の値で補完
            df[col].ffill(inplace=True)
            df[col].bfill(inplace=True)
        else:
            # 万が一カラムが存在しない場合は0で埋める
            df[col] = 0
    return df


# --- テクニカル指標を計算する関数（特徴量エンジニアリング） ---
def feature_engineering(data):
    """データフレームにテクニカル指標を追加する"""
    df = data.copy()
    # 移動平均線 (SMA)
    df["SMA_20"] = df["終値"].rolling(window=20).mean()
    df["SMA_50"] = df["終値"].rolling(window=50).mean()

    # 相対力指数 (RSI)
    delta = df["終値"].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # 計算初期に発生する欠損値を後方の値で埋める
    df.fillna(method="bfill", inplace=True)
    return df


# --- 予測結果を可視化するグラフを作成する関数 ---
def create_plot(predict_data, title):
    """Plotlyを使用して実績と予測をプロットしたグラフ画像を生成する"""
    fig = go.Figure()

    # 実績値のプロット（折れ線グラフ）
    fig.add_trace(
        go.Scatter(
            x=predict_data["historical_dates"],
            y=predict_data["historical_values"],
            mode="lines",
            name="実績値",
            line={"color": "blue", "width": 2.5},
        )
    )

    # 翌日予測値のプロット（赤い点）
    fig.add_trace(
        go.Scatter(
            x=[predict_data["predicted_date"]],
            y=[predict_data["predicted_value"]],
            mode="markers",
            name="翌日予測",
            marker={"size": 10, "color": "red", "symbol": "circle"},
        )
    )

    # グラフのレイアウト設定
    fig.update_layout(
        title=title,
        xaxis_title="日付",
        yaxis_title="株価 (円)",
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.7)"},
        margin={"t": 50, "l": 60, "r": 30, "b": 50},
        xaxis=dict(
        tickformat="%Y-%m-%d",  # ★ 日付の表示フォーマットを指定
        tickmode="auto",        # ★自動で見やすい間隔に
        nticks=8,              # ★表示する目盛り数を制御
        tickangle=45            # ★ 45度傾けて見やすく
        )

    )

    # グラフをPNG画像に変換し、HTMLで表示できるようにBase64エンコードする
    img_bytes = fig.to_image(format="png", width=800, height=500)
    return base64.b64encode(img_bytes).decode("utf8")


# --- メインのWebページ処理 ---
@app.route("/", methods=["GET", "POST"])
def index():
    # テンプレートに渡すための初期コンテキスト
    context = {"stock_list": STOCK_LIST, "form_values": request.form, "results": None}

    # ユーザーがフォームを送信（POSTリクエスト）した場合の処理
    if request.method == "POST":
        try:
            ticker = request.form.get("ticker")
            if not ticker:
                raise ValueError("銘柄が選択されていません。")

            # --- 1. 必要なファイルのパスを生成 ---
            model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
            scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
            data_path = os.path.join(DATA_DIR, f"{ticker}_{STOCK_LIST[ticker]}.csv")

            # ファイルの存在チェック
            if (
                not os.path.exists(model_path)
                or not os.path.exists(scaler_path)
                or not os.path.exists(data_path)
            ):
                raise FileNotFoundError(
                    f"必要なファイル（モデル、スケーラー、データ）が見つかりません。"
                )

            # --- 2. モデルとデータの読み込み ---
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            df_raw = load_and_prepare_data(data_path)
            df_feat = feature_engineering(df_raw)

            # --- 3. 予測の実行 ---
            # モデルに入力する直近60日分の特徴量データを取得
            last_60_days_features = df_feat[FEATURES].tail(LOOK_BACK).values
            # データをスケーリング（0〜1の範囲に変換）
            scaled_input = scaler.transform(last_60_days_features)
            # モデルが要求する形状 (1, 60, 5) に変形
            X = np.reshape(scaled_input, (1, LOOK_BACK, len(FEATURES)))

            # 翌日の終値を予測
            predicted_scaled_value = model.predict(X, verbose=0)

            # --- 4. 予測結果を元のスケールに戻す ---
            # スケーリングを逆変換するためにダミー配列を作成
            target_col_index = FEATURES.index("終値")
            dummy_array = np.zeros((1, len(FEATURES)))
            dummy_array[0, target_col_index] = predicted_scaled_value[0, 0]

            # スケールを逆変換して、人間が読める株価に戻す
            predicted_value = scaler.inverse_transform(dummy_array)[0, target_col_index]

            # --- 5. 表示用データの準備 ---
            # 予測日の計算（データの最終日の翌日）
            next_date = df_raw.index[-1] + timedelta(days=1)

            # グラフ描画用のデータを作成
            plot_payload = {
    "historical_dates": df_raw.index.strftime("%Y-%m-%d"),  # ← 日付を文字列に変換
    "historical_values": df_raw["終値"],
    "predicted_date": next_date.strftime("%Y-%m-%d"),       # ← 予測日も文字列に変換
    "predicted_value": float(predicted_value),
}


            # 画面表示用の統計情報を計算
            last_seq = df_raw["終値"].values[-LOOK_BACK:]
            context["results"] = {
                "reason": {
                    "trend": "上昇" if last_seq[-1] > last_seq[0] else "下降",
                    "trend_delta": f"{float(last_seq[-1] - last_seq[0]):.2f}",
                    "volatility": f"{float(np.std(last_seq) / np.mean(last_seq)):.2%}",
                    "predicted_next": f"{float(predicted_value):.2f}",
                    "predicted_date": next_date.strftime("%Y-%m-%d"),
                },
                "plot_url": create_plot(
                    plot_payload, f"{STOCK_LIST[ticker]} の翌日終値予測"
                ),
            }

        except Exception as e:
            # エラーが発生した場合の処理
            traceback.print_exc()  # コンソールに詳細なエラーを出力
            context["error"] = f"エラーが発生しました: {e}"

    # テンプレート(index.html)を描画してユーザーに返す
    return render_template("index.html", **context)


# --- アプリケーションの実行 ---
if __name__ == "__main__":
    app.run(debug=True)