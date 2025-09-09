import pandas as pd  # データ分析のプロフェッショナルツール (Excelのスーパー版)
import numpy as np  # 数値計算、特に配列を効率的に扱うツール
import yfinance as yf  # Yahoo Financeから株価データをダウンロードするツール
from datetime import datetime, timedelta  # 日付や時間を扱うツール
import os  # ファイルやフォルダを操作するOSの機能を使うツール
from sklearn.preprocessing import (
    MinMaxScaler,
)  # データを特定の範囲(0~1)に変換するツール
from tensorflow.keras.models import (
    Sequential,
)  # ニューラルネットワークのモデルを層を積み重ねて作るツール
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
)  # モデルの層の種類 (LSTM, 全結合, ドロップアウト)
import joblib  # 学習済みのPythonオブジェクト(今回はScaler)を保存/読み込みするツール


# --- :gear: 2. 設定項目 ---
# このセクションをカスタマイズすることで、対象銘柄や学習の挙動を自由に変更できます。

# 【データ取得関連】
# 取得対象の銘柄リスト (キー: ティッカーシンボル, 値: 会社名)
TICKERS = {
    "8306.T": "三菱UFJ銀行",
    "8411.T": "みずほ銀行",
    "8316.T": "三井住友銀行",
    "8308.T": "りそな銀行",
}
# データ取得期間（過去何年分のデータを学習に使うか）
YEARS_TO_FETCH = 5
# データ保存先ディレクトリ名
DATA_DIR = "stock_data"

# 【モデル学習関連】
# 完成したモデルの保存先ディレクトリ名
MODEL_DIR = "models"
# 過去何日分のデータを使って翌日を予測するか（AIが振り返る日数）
LOOK_BACK = 60
# 予測に使用する「特徴量」のリスト。AIに与えるヒントの種類。
FEATURES = ["終値", "出来高", "SMA_20", "SMA_50", "RSI_14"]


# --- :inbox_tray: 3. データ収集関数 ---
def collect_stock_data(tickers, years):
    """
    yfinanceライブラリを使い、指定された銘柄の株価データを取得し、
    CSVファイルとしてローカルに保存する関数。
    """
    print("--- STEP 1: 株価データの収集を開始 ---")
    # 保存先ディレクトリがなければ作成する
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"ディレクトリ '{DATA_DIR}' を作成しました。")

    # データ取得期間を定義 (今日からN年前まで)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    # 設定した銘柄リストの各銘柄に対してループ処理
    for ticker, name in tickers.items():
        try:
            print(f"【{name} ({ticker})】のデータを取得中...")
            # yfinanceを使ってデータをダウンロード
            data = yf.download(ticker, start=start_date, end=end_date)
            # データが空でなければ処理を続行
            if data.empty:
                print(f"警告: {name} ({ticker}) のデータが取得できませんでした。")
                continue

            # 列名を英語から日本語に分かりやすく変更
            data = data.rename(
                columns={
                    "Open": "始値",
                    "High": "高値",
                    "Low": "安値",
                    "Close": "終値",
                    "Adj Close": "調整後終値",
                    "Volume": "出来高",
                }
            )
            # ファイルパスを生成し、CSVとして保存
            file_path = os.path.join(DATA_DIR, f"{ticker}_{name}.csv")
            data.to_csv(file_path)
            print(f"-> データを '{file_path}' に保存しました。")
        except Exception as e:
            print(f"エラー: {name} ({ticker}) のデータ取得中にエラーが発生: {e}")
    print("--- STEP 1: データ収集完了 ---\n")


# --- :sparkles: 4. データ前処理 兼 特徴量エンジニアリング関数 ---
def preprocess_and_feature_engineering(data):
    """
    データフレームを受け取り、AIが学習しやすいようにデータを整形し、
    テクニカル指標などの新しい特徴量を追加する関数。
    """
    print("-> データの前処理と特徴量エンジニアリングを実行中...")

    # データクレンジング: 欠損値などを補完し、データを数値型に統一
    for col_jp, col_en in [("終値", "Close"), ("出来高", "Volume")]:
        if col_jp not in data.columns and col_en in data.columns:
            data.rename(columns={col_en: col_jp}, inplace=True)
        if col_jp in data.columns:
            data[col_jp] = pd.to_numeric(data[col_jp], errors="coerce")  # 文字をNaNに
            data[col_jp].ffill(inplace=True)  # 前の値でNaNを埋める
            data[col_jp].bfill(inplace=True)  # 後ろの値でNaNを埋める

    if "終値" not in data.columns or "出来高" not in data.columns:
        print("-> 必須列（終値、出来高）が揃っていないため、処理を中断します。")
        return pd.DataFrame()

    # 特徴量エンジニアリング: 元のデータから予測に役立つヒントを計算して追加
    # SMA (単純移動平均線): 過去の価格の平均。現在のトレンド（上昇/下降）をAIに教える。
    data["SMA_20"] = data["終値"].rolling(window=20).mean()
    data["SMA_50"] = data["終値"].rolling(window=50).mean()

    # RSI (相対力指数): 価格の勢いや過熱感（買われすぎ/売られすぎ）をAIに教える。
    delta = data["終値"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # 計算過程で発生したNaN（計算不能）行を削除し、クリーンなデータにする
    data.dropna(inplace=True)
    return data


# --- :brain: 5. モデル学習・保存関数 ---
def train_and_save_models():
    """
    データディレクトリ内の各CSVファイルに対して、データセット作成、モデル構築、
    学習、評価、保存の一連のフローを実行する関数。
    """
    print("--- STEP 2: モデルの学習と保存を開始 ---")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"ディレクトリ '{MODEL_DIR}' を作成しました。")

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print(
            f"エラー: '{DATA_DIR}' に学習データが見つかりません。STEP 1を先に実行してください。"
        )
        return

    # 各CSVファイル（銘柄ごと）にループ処理
    for file_name in csv_files:
        ticker, company_name = file_name.replace(".csv", "").split("_", 1)
        print(f"\n--- 【{company_name} ({ticker})】のモデル学習を開始 ---")
        file_path = os.path.join(DATA_DIR, file_name)

        # ■ 5-1. 堅牢なデータ読み込みと日付の整備
        # CSVを読み込み、日付として認識できない行があってもエラーにせず、安全に処理する。
        # 1. 最初にインデックス（日付列）を文字列として読み込む
        data = pd.read_csv(file_path, index_col=0)
        # 2. インデックスを強制的に日付型に変換。変換できないものはNaT(Not a Time)という特殊な値に置き換える
        data.index = pd.to_datetime(data.index, errors="coerce")
        # 3. NaTになった行（日付として認識できなかった行）をデータフレームから完全に削除する
        original_rows = len(data)
        data.dropna(inplace=True)
        if len(data) < original_rows:
            print(f"-> 日付形式が不正な {original_rows - len(data)} 行を削除しました。")

        # 有効なデータが残っているかチェック
        if not isinstance(data.index, pd.DatetimeIndex) or len(data) == 0:
            print(
                f"警告: ファイル '{file_name}' から有効な日付データを読み込めませんでした。スキップします。"
            )
            continue

        # ■ 5-2. 前処理と特徴量エンジニアリングの実行
        data = preprocess_and_feature_engineering(data)
        if data.empty or len(data) < LOOK_BACK:
            print(
                f"警告: ファイル '{file_name}' は前処理後のデータが不足しているため、スキップします。"
            )
            continue

        # ■ 5-3. データの正規化
        # 各特徴量のスケールを0~1の範囲に統一する。AIの学習効率と精度を高めるための重要な下準備。
        feature_data = data[FEATURES].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(feature_data)
        target_column_index = FEATURES.index("終値")

        # ■ 5-4. データセットの作成 (スライディングウィンドウ)
        # 「過去60日分のデータ(x)」と「その翌日の終値(y)」のペアを大量に作成する。
        x_data, y_data_price = [], []
        for i in range(len(scaled_data) - LOOK_BACK):
            x_data.append(scaled_data[i : i + LOOK_BACK])
            y_data_price.append(scaled_data[i + LOOK_BACK, target_column_index])
        x_data, y_data_price = np.asarray(x_data), np.asarray(y_data_price)

        # ■ 5-5. 訓練データとテストデータへの分割
        # データを学習用(80%)と性能評価用(20%)に分ける。AIの「本当の実力」を測るために不可欠。
        train_size = int(x_data.shape[0] * 0.8)
        x_train, y_train_price = x_data[:train_size], y_data_price[:train_size]
        x_test, y_test_price = x_data[train_size:], y_data_price[train_size:]
        print(
            f"-> データセット準備完了 | 訓練データ数: {len(x_train)}, テストデータ数: {len(x_test)}"
        )

        # ■ 5-6. LSTMモデルの構築
        # 時系列データのパターンを記憶するのが得意なLSTM層を2段重ねたモデルを定義する。
        model = Sequential(
            [
                LSTM(
                    units=50,
                    return_sequences=True,
                    input_shape=(x_train.shape[1], x_train.shape[2]),
                ),
                Dropout(0.2),  # 過学習を防ぐための「忘れ」の仕組み
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),  # 最終的な出力を形成する層
                Dense(units=1),  # 予測値は1つ（終値）なので出力は1
            ]
        )
        model.compile(optimizer="adam", loss="mean_squared_error")

        # ■ 5-7. モデルの学習
        # 訓練データを使って、モデルに「予測→答え合わせ→修正」を繰り返させる。
        print("-> モデルの学習を開始します...")
        model.fit(x_train, y_train_price, batch_size=32, epochs=50, verbose=0)
        print("-> モデルの学習が完了しました。")

        # ■ 5-8. モデルの評価
        # 学習に使っていないテストデータで、モデルの真の実力を評価する。
        loss = model.evaluate(x_test, y_test_price, verbose=0)
        print(f"-> テストデータでの性能評価 | 損失 (MSE): {loss:.6f}")

        # ■ 5-9. モデルとスケーラーの保存
        # 後で予測に使えるように、学習済みの「モデル」と「正規化のルール(scaler)」を保存する。
        model.save(os.path.join(MODEL_DIR, f"{ticker}_model.h5"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl"))
        print(f"-> モデルとスケーラーを '{MODEL_DIR}' に保存しました。")

    print("\n--- STEP 2: 全てのモデル学習が完了しました ---")


# --- :rocket: 6. メイン処理実行ブロック ---
# このスクリプトが直接実行された場合に、以下の処理を順番に実行する司令塔。
if __name__ == "__main__":
    # STEP 1: 株価データを収集する
    collect_stock_data(TICKERS, YEARS_TO_FETCH)

    # STEP 2: 収集したデータを使ってモデルを学習・保存する
    train_and_save_models()

    print("\n全工程が正常に終了しました。")