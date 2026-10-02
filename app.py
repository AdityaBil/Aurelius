import os
import os
import json
import traceback

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
CORS(app)

# ── Load model artifacts ─────────────────────────────────────────────────────
model = None
scaler = None
model_config = {}
runtime_features = []
runtime_lookback = None
runtime_threshold = None
runtime_model_name = None
finbert_tok = None
finbert_mdl = None
finbert_ok = False

def load_artifacts():
    global model, scaler, model_config, runtime_features
    global runtime_lookback, runtime_threshold, runtime_model_name
    global finbert_tok, finbert_mdl, finbert_ok

    config_path = os.path.join(BASE_DIR, 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            model_config = json.load(f)
        print(f"  [OK] model_config.json loaded")
    else:
        model_config = {
            'lookback': 25,
            'forward_days': 3,
            'best_threshold': 0.5,
            'features': []
        }
        print("  [WARN] model_config.json not found, using defaults")

    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(BASE_DIR, 'scaler_sentiment.pkl')
    if os.path.exists(scaler_path):
        import joblib
        scaler = joblib.load(scaler_path)
        print(f"  [OK] scaler loaded from {os.path.basename(scaler_path)}")

    for model_name in ['stock_model.keras', 'best_model.keras']:
        model_path = os.path.join(BASE_DIR, model_name)
        if os.path.exists(model_path):
            try:
                import tensorflow as tf
                from tensorflow import keras

                def focal_loss(alpha=0.25, gamma=2.0):
                    def loss(y_true, y_pred):
                        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
                        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
                        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
                        focal = alpha * (1 - pt) ** gamma * bce
                        return tf.reduce_mean(focal)
                    return loss

                class MultiHeadSelfAttention(keras.layers.Layer):
                    def __init__(self, d_model, num_heads, **kwargs):
                        super().__init__(**kwargs)
                        self.h = num_heads
                        self.dk = d_model // num_heads
                        self.d = d_model
                        self.wq = keras.layers.Dense(d_model)
                        self.wk = keras.layers.Dense(d_model)
                        self.wv = keras.layers.Dense(d_model)
                        self.wo = keras.layers.Dense(d_model)

                    def build(self, input_shape):
                        self.wq.build(input_shape)
                        self.wk.build(input_shape)
                        self.wv.build(input_shape)
                        self.wo.build((input_shape[0], input_shape[1], self.d))
                        super().build(input_shape)

                    def call(self, x, training=False):
                        batch_size = tf.shape(x)[0]

                        def split_heads(t):
                            t = tf.reshape(t, (batch_size, -1, self.h, self.dk))
                            return tf.transpose(t, [0, 2, 1, 3])

                        q = split_heads(self.wq(x))
                        k = split_heads(self.wk(x))
                        v = split_heads(self.wv(x))
                        attn = tf.nn.softmax(
                            tf.matmul(q, k, transpose_b=True) /
                            tf.math.sqrt(tf.cast(self.dk, tf.float32)),
                            axis=-1,
                        )
                        out = tf.reshape(
                            tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3]),
                            (batch_size, -1, self.d),
                        )
                        return self.wo(out)

                    def get_config(self):
                        cfg = super().get_config()
                        cfg.update({'d_model': self.d, 'num_heads': self.h})
                        return cfg

                class TransformerBlock(keras.layers.Layer):
                    def __init__(self, d_model, num_heads, d_ff, drop, **kwargs):
                        super().__init__(**kwargs)
                        self.attn = MultiHeadSelfAttention(d_model, num_heads)
                        self.ffn = keras.Sequential([
                            keras.layers.Dense(d_ff),
                            keras.layers.Activation(keras.activations.gelu),
                            keras.layers.Dropout(drop),
                            keras.layers.Dense(d_model),
                        ])
                        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
                        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
                        self.drop1 = keras.layers.Dropout(drop)
                        self.drop2 = keras.layers.Dropout(drop)
                        self.d_model = d_model
                        self.num_heads = num_heads
                        self.d_ff = d_ff
                        self.drop = drop

                    def build(self, input_shape):
                        self.attn.build(input_shape)
                        self.ffn.build(input_shape)
                        self.norm1.build(input_shape)
                        self.norm2.build(input_shape)
                        super().build(input_shape)

                    def call(self, x, training=False):
                        x = x + self.drop1(
                            self.attn(self.norm1(x), training=training),
                            training=training,
                        )
                        return x + self.drop2(
                            self.ffn(self.norm2(x), training=training),
                            training=training,
                        )

                    def get_config(self):
                        cfg = super().get_config()
                        cfg.update({'d_model': self.d_model, 'num_heads': self.num_heads,
                                    'd_ff': self.d_ff, 'drop': self.drop})
                        return cfg

                class ChannelAttention(keras.layers.Layer):
                    def __init__(self, channels, reduction=8, **kwargs):
                        super().__init__(**kwargs)
                        self.fc1 = keras.layers.Dense(max(channels // reduction, 4), activation='relu')
                        self.fc2 = keras.layers.Dense(channels, activation='sigmoid')
                        self.channels = channels
                        self.reduction = reduction

                    def build(self, input_shape):
                        pooled_shape = (input_shape[0], input_shape[-1])
                        self.fc1.build(pooled_shape)
                        self.fc2.build((input_shape[0], max(self.channels // self.reduction, 4)))
                        super().build(input_shape)

                    def call(self, x, training=False):
                        gap = tf.reduce_mean(x, axis=1)
                        scale = tf.expand_dims(self.fc2(self.fc1(gap)), axis=1)
                        return x * scale

                    def get_config(self):
                        cfg = super().get_config()
                        cfg.update({'channels': self.channels, 'reduction': self.reduction})
                        return cfg

                custom_objects = {
                    'MultiHeadSelfAttention': MultiHeadSelfAttention,
                    'TransformerBlock': TransformerBlock,
                    'ChannelAttention': ChannelAttention,
                    'loss_fn': focal_loss()
                }
                candidate = keras.models.load_model(
                    model_path,
                    custom_objects=custom_objects,
                    compile=False,
                )
                input_shape = candidate.input_shape
                model_lookback = int(input_shape[1])
                model_features = int(input_shape[2])
                scaler_features = getattr(scaler, 'n_features_in_', None)
                if scaler_features is not None and scaler_features != model_features:
                    print(
                        f"  [WARN] Skipping {model_name}: model expects "
                        f"{model_features} features but scaler has {scaler_features}"
                    )
                    continue

                model = candidate
                runtime_model_name = model_name
                runtime_lookback = model_lookback
                if hasattr(scaler, 'feature_names_in_'):
                    runtime_features = list(scaler.feature_names_in_)
                else:
                    runtime_features = model_config.get('features', [])[:model_features]

                config_matches_model = (
                    len(model_config.get('features', [])) == model_features and
                    model_config.get('lookback') == model_lookback
                )
                runtime_threshold = (
                    float(model_config.get('best_threshold', 0.5))
                    if config_matches_model else 0.5
                )
                if not config_matches_model:
                    print(
                        "  [WARN] model_config.json does not match the selected "
                        f"{model_name}; using runtime shape and threshold 0.5"
                    )
                print(
                    f"  [OK] Model loaded: {model_name} "
                    f"(lookback={model_lookback}, features={model_features})"
                )
                break
            except Exception as e:
                print(f"  [WARN] Could not load {model_name}: {e}")

    finbert_dir = os.path.join(BASE_DIR, 'finbert')
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        src = finbert_dir if os.path.exists(finbert_dir) else "ProsusAI/finbert"
        finbert_tok = AutoTokenizer.from_pretrained(src)
        finbert_mdl = AutoModelForSequenceClassification.from_pretrained(src)
        finbert_ok = True
        print(f"  [OK] FinBERT loaded from: {src}")
    except Exception as e:
        print(f"  [WARN] FinBERT not available: {e}")


load_artifacts()


# ── Feature engineering ───────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_features(df, sentiment_score=0.0):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = df['High'] / (df['Low'] + 1e-9)
    df['Close_Open_Ratio'] = df['Close'] / (df['Open'] + 1e-9)
    for w in [5, 10, 20, 50]:
        df[f'MA_{w}'] = df['Close'].rolling(w).mean()
    df['MA_5_10_cross'] = df['MA_5'] - df['MA_10']
    df['MA_10_20_cross'] = df['MA_10'] - df['MA_20']
    df['MA_20_50_cross'] = df['MA_20'] - df['MA_50']
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['RSI_7'] = compute_rsi(df['Close'], 7)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = bb_mid + 2 * bb_std
    df['BB_lower'] = bb_mid - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (bb_mid + 1e-10)
    df['BB_pos'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)
    df['Volatility_5'] = df['Return'].rolling(5).std()
    df['Volatility_20'] = df['Return'].rolling(20).std()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume'].rolling(10).mean() + 1e-10)
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = (df['Close'] - low14) / (high14 - low14 + 1e-10) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['ATR_ratio'] = atr / (df['Close'] + 1e-10)
    df['CCI'] = (df['Close'] - df['Close'].rolling(20).mean()) / (0.015 * df['Close'].rolling(20).std() + 1e-10)
    df['Williams_R'] = -100 * (high14 - df['Close']) / (high14 - low14 + 1e-10)
    for w in [5, 10, 20]:
        df[f'Momentum_{w}'] = df['Close'] / (df['Close'].shift(w) + 1e-9) - 1

    rolling_high_20 = df['Close'].rolling(20).max()
    df['dist_from_high_20'] = df['Close'] / (rolling_high_20 + 1e-9) - 1

    down_bar = (df['Close'].diff() < 0).astype(int)
    groups = (down_bar != down_bar.shift()).cumsum()
    df['consec_down'] = down_bar.groupby(groups).cumsum() * down_bar

    atr14 = tr.rolling(14).mean()
    df['candle_body_ratio'] = (df['Close'] - df['Open']) / (atr14 + 1e-9)

    df['ma_alignment'] = (
        (df['MA_5'] > df['MA_10']).astype(int) +
        (df['MA_10'] > df['MA_20']).astype(int) +
        (df['MA_20'] > df['MA_50']).astype(int) +
        (df['MA_5'] > df['MA_50']).astype(int)
    ).astype(float)
    df['ema_slope_5'] = df['MA_5'].pct_change(3)
    df['ema_slope_20'] = df['MA_20'].pct_change(5)
    df['rsi_momentum'] = df['RSI_14'] - df['RSI_14'].shift(3)

    rolling_low_20 = df['Close'].rolling(20).min()
    df['price_range_pos'] = (df['Close'] - rolling_low_20) / (
        rolling_high_20 - rolling_low_20 + 1e-9
    )
    df['up_days_5'] = (df['Close'].diff() > 0).astype(float).rolling(5).sum()

    vpt_raw = (df['Return'] * df['Volume']).cumsum()
    vpt_std = vpt_raw.rolling(20).std().replace(0, np.nan)
    df['vpt_norm'] = (vpt_raw - vpt_raw.rolling(20).mean()) / (vpt_std + 1e-9)
    df['gap_signal'] = (df['Open'] - df['Close'].shift()) / (df['Close'].shift() + 1e-10)
    df['above_ma50'] = (df['Close'] > df['MA_50']).astype(float)
    df['sentiment'] = sentiment_score
    return df


def _normalize_price_frame(df):
    """Return the OHLCV columns in the format used by the feature pipeline."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        wanted = {'Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Datetime'}
        flattened = []
        for column in df.columns:
            parts = [str(part) for part in column if str(part) != '']
            flattened.append(next((part for part in parts if part in wanted), parts[0]))
        df.columns = flattened

    if 'Date' not in df.columns and 'Datetime' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        elif df.index.name:
            df = df.reset_index()

    if 'Datetime' in df.columns and 'Date' not in df.columns:
        df = df.rename(columns={'Datetime': 'Date'})
    if 'Date' not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: 'Date'})

    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if any(column not in df.columns for column in required):
        return pd.DataFrame()

    df = df[required].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True).dt.tz_localize(None)
    for column in required[1:]:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df.dropna().sort_values('Date').drop_duplicates('Date').reset_index(drop=True)


def fetch_price_history(symbol):
    """Fetch enough daily OHLCV data despite occasional Yahoo API failures."""
    import requests

    candidates = []
    try:
        import yfinance as yf
        candidates.append(yf.Ticker(symbol).history(period='1y', auto_adjust=False))
    except Exception as exc:
        print(f"  [WARN] Yahoo Ticker.history failed for {symbol}: {exc}")

    try:
        import yfinance as yf
        candidates.append(yf.download(
            symbol,
            period='1y',
            interval='1d',
            auto_adjust=False,
            progress=False,
            threads=False,
        ))
    except Exception as exc:
        print(f"  [WARN] Yahoo download failed for {symbol}: {exc}")

    for candidate in candidates:
        normalized = _normalize_price_frame(candidate)
        if len(normalized) >= 30:
            return normalized

    # Yahoo's chart endpoint does not require yfinance's cookie/crumb cache and
    # is a useful fallback in hosted containers where that cache is unavailable.
    from urllib.parse import quote
    for host in ('query1.finance.yahoo.com', 'query2.finance.yahoo.com'):
        try:
            url = f'https://{host}/v8/finance/chart/{quote(symbol, safe="")}'
            response = requests.get(
                url,
                params={'range': '1y', 'interval': '1d', 'events': 'history'},
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=20,
            )
            response.raise_for_status()
            result = (response.json().get('chart', {}).get('result') or [None])[0]
            if not result:
                continue
            quotes = (result.get('indicators', {}).get('quote') or [{}])[0]
            timestamps = result.get('timestamp') or []
            candidate = pd.DataFrame({
                'Date': pd.to_datetime(timestamps, unit='s', utc=True).tz_localize(None),
                'Open': quotes.get('open', []),
                'High': quotes.get('high', []),
                'Low': quotes.get('low', []),
                'Close': quotes.get('close', []),
                'Volume': quotes.get('volume', []),
            })
            normalized = _normalize_price_frame(candidate)
            if len(normalized) >= 30:
                return normalized
        except Exception as exc:
            print(f"  [WARN] Yahoo chart fallback failed for {symbol} on {host}: {exc}")

    return pd.DataFrame()


def get_sentiment(headlines):
    if not finbert_ok or not headlines:
        return 0.0
    try:
        import torch
        scores = []
        for h in headlines[:10]:
            inputs = finbert_tok(h, return_tensors='pt', truncation=True, max_length=128)
            with torch.no_grad():
                logits = finbert_mdl(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].tolist()
            scores.append(probs[0] - probs[1])
        return float(np.mean(scores))
    except Exception:
        return 0.0


def get_headlines(symbol):
    try:
        from GoogleNews import GoogleNews
        googlenews = GoogleNews(lang='en', period='7d')
        googlenews.search(f'{symbol} stock')
        results = googlenews.results() or []
        headlines = [item.get('title', '') for item in results[:15]]
        headlines = [title.strip() for title in headlines if title and title.strip()]
        if headlines:
            return headlines
    except Exception as e:
        print(f"  [WARN] GoogleNews unavailable: {e}")

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        return [n.get('title', '') or n.get('content', {}).get('title', '') for n in news[:15] if n]
    except Exception:
        return []


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model': model is not None,
        'scaler': scaler is not None,
        'finbert': finbert_ok,
        'runtime_model': runtime_model_name,
        'runtime_lookback': runtime_lookback,
        'runtime_features': len(runtime_features),
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/model_info')
def model_info():
    return jsonify({
        'config': model_config,
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'runtime_model': runtime_model_name,
        'runtime_lookback': runtime_lookback,
        'runtime_features': runtime_features
    })


@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(force=True) or {}
        symbol = str(body.get('symbol', '')).strip().upper()
        if not symbol:
            return jsonify({'error': 'symbol is required'}), 400
        if len(symbol) > 10:
            return jsonify({'error': 'invalid symbol'}), 400

        df = fetch_price_history(symbol)
        if df is None or len(df) < 30:
            return jsonify({'error': f'Not enough data for {symbol}. Check the ticker symbol.'}), 400

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        headlines = get_headlines(symbol)
        sentiment_score = get_sentiment(headlines)

        df_feat = compute_features(df, sentiment_score)
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        feature_cols = runtime_features or model_config.get('features', [])
        if not feature_cols:
            feature_cols = [c for c in df_feat.columns if c not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        missing = [c for c in feature_cols if c not in df_feat.columns]
        for m in missing:
            df_feat[m] = 0.0

        lookback = runtime_lookback or model_config.get('lookback', 25)
        threshold = runtime_threshold if runtime_threshold is not None else model_config.get('best_threshold', 0.5)

        if model is not None and scaler is not None:
            feat_data = df_feat[feature_cols].values
            if len(feat_data) < lookback:
                return jsonify({'error': f'Need at least {lookback} days of data'}), 400

            n_feat = feat_data.shape[1]
            n_scaler = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else n_feat
            n_model = int(model.input_shape[-1])
            if n_feat != n_scaler or n_feat != n_model:
                return jsonify({'error': 'Model, scaler, and feature configuration are incompatible'}), 503

            feat_scaled = scaler.transform(feat_data)
            seq = feat_scaled[-lookback:]
            seq = seq.reshape(1, lookback, -1)
            prob_up = float(model.predict(seq, verbose=0)[0][0])
        else:
            rsi = float(df_feat['RSI_14'].iloc[-1]) if 'RSI_14' in df_feat.columns else 50.0
            macd = float(df_feat['MACD'].iloc[-1]) if 'MACD' in df_feat.columns else 0.0
            prob_up = 0.5 + (rsi - 50) / 200 + macd / 100 + sentiment_score * 0.1
            prob_up = float(np.clip(prob_up, 0.3, 0.75))

        signal = 'UP' if prob_up >= threshold else 'DOWN'
        confidence = prob_up * 100 if signal == 'UP' else (1 - prob_up) * 100

        last = df_feat.iloc[-1]
        rsi_val = float(last.get('RSI_14', np.nan)) if 'RSI_14' in last.index else None
        macd_val = float(last.get('MACD', np.nan)) if 'MACD' in last.index else None
        bb_width = float(last.get('BB_width', np.nan)) if 'BB_width' in last.index else None
        vol_ratio = float(last.get('Volume_Ratio', np.nan)) if 'Volume_Ratio' in last.index else None

        recent_rows = []
        for _, row in df.tail(60).iterrows():
            recent_rows.append({
                'Date': row['Date'].strftime('%Y-%m-%d'),
                'Open': round(float(row['Open']), 4),
                'High': round(float(row['High']), 4),
                'Low': round(float(row['Low']), 4),
                'Close': round(float(row['Close']), 4),
                'Volume': int(row['Volume'])
            })

        return jsonify({
            'symbol': symbol,
            'signal': signal,
            'probability_up': round(prob_up, 4),
            'probability_down': round(1 - prob_up, 4),
            'probability': round(confidence, 2),
            'confidence': round(confidence, 2),
            'latest_price': round(float(df['Close'].iloc[-1]), 2),
            'current_price': round(float(df['Close'].iloc[-1]), 2),
            'rsi': round(rsi_val, 2) if rsi_val is not None and not np.isnan(rsi_val) else None,
            'macd': round(macd_val, 4) if macd_val is not None and not np.isnan(macd_val) else None,
            'bb_width': round(bb_width, 4) if bb_width is not None and not np.isnan(bb_width) else None,
            'volume_ratio': round(vol_ratio, 4) if vol_ratio is not None and not np.isnan(vol_ratio) else None,
            'sentiment_score': round(sentiment_score, 4),
            'headlines': headlines[:15],
            'recent_data': recent_rows,
            'timestamp': datetime.utcnow().isoformat(),
            'model_used': 'neural_network' if model is not None else 'heuristic'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
