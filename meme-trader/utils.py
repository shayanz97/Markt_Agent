# utils.py
import ccxt, pandas as pd, numpy as np, ta
from typing import List, Tuple

exchange = ccxt.kucoin()

def fetch_ohlcv_df(symbol: str, timeframe: str='15m', limit: int=20000):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['time','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.sort_values('time', inplace=True, ignore_index=True)
    return df

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma_diff'] = (df['ma20'] - df['ma50']) / (df['ma50'] + 1e-9)
    df['vol_change'] = df['volume'].pct_change()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    stoch = ta.momentum.StochRSIIndicator(df['close'])
    df['stochrsi_k'] = stoch.stochrsi_k()
    df['stochrsi_d'] = stoch.stochrsi_d()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_pct'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap_like'] = (tp * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-9)
    df['vwap_dist'] = (df['close'] - df['vwap_like']) / (df['vwap_like'] + 1e-9)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    return df

def compute_max_future_return_array(prices: np.ndarray, horizons: List[int]) -> np.ndarray:
    n = len(prices)
    max_ret = np.zeros(n, dtype=float)
    max_h = max(horizons)
    for t in range(n):
        best = 0.0
        for h in horizons:
            if t + h < n:
                best = max(best, (np.max(prices[t+1:t+1+h]) - prices[t]) / (prices[t] + 1e-9))
        max_ret[t] = best
    return max_ret

def build_sequences(df, features: List[str], seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get sequences of features, labels, raw_max_ret aligned with sequence-end index
    prices = df['close'].values
    max_ret = compute_max_future_return_array(prices, horizons)
    X, y, raw_ret = [], [], []
    for i in range(seq_len, len(df)):
        X.append(df[features].iloc[i-seq_len:i].values)
        y.append(1 if max_ret[i] > pos_threshold else 0)
        raw_ret.append(max_ret[i])
    return np.array(X), np.array(y), np.array(raw_ret)
