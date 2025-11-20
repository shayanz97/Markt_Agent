# backtest.py
import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from utils import compute_max_future_return_array
import json

# load artifacts
model = tf.keras.models.load_model("transformer_pnl_model.keras", compile=False)  # compile False; we'll use predict
scaler = joblib.load("scaler.pkl")
meta = np.load("meta.npy", allow_pickle=True)

# load your X_val/X_test arrays or recompute/separate them
# For brevity assume you saved X_val_scaled, y_val, raw_val, X_test_scaled, y_test, raw_test during train step
X_val = np.load("X_val_scaled.npy")
y_val = np.load("y_val.npy")
X_test = np.load("X_test_scaled.npy")
y_test = np.load("y_test.npy")

probs_val = model.predict(X_val).flatten()
probs_test = model.predict(X_test).flatten()

# evaluation/backtest util — simplified (use the earlier detailed backtest if you prefer)
def simulate_trades(probs, y_true, meta_list, thresholds, pos_threshold=0.01, slippage=0.0005, fee=0.001,
                    initial_capital=10000.0, risk_per_trade=0.01):
    out = []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        trades = []
        capital = initial_capital
        for i, p in enumerate(preds):
            if p == 0: continue
            m = meta_list[i]
            df = m['df']; idx = m['seq_end_idx']
            # entry = next open if exists else next close
            if idx+1 < len(df):
                entry = df['open'].iat[idx+1]
            else:
                entry = df['close'].iat[idx]
            # find TP within max horizon
            futs = df['close'].iat[idx+1: idx+1+max(horizons)]
            if len(futs)==0: continue
            target = entry*(1+pos_threshold)
            exit_p = futs[futs >= target].values
            if len(exit_p) > 0:
                exit_price = exit_p[0]
            else:
                exit_price = futs.values[-1]
            entry_eff = entry*(1+slippage+fee)
            exit_eff = exit_price*(1-slippage-fee)
            qty = (capital * risk_per_trade)/entry
            pnl = (exit_eff-entry_eff)*qty
            capital += pnl
            trades.append(pnl)
        out.append({'threshold': thr, 'final_capital': capital, 'n_trades': len(trades), 'total_pnl': sum(trades)})
    return out

thresholds = np.arange(0.01, 0.96, 0.01)
res_val = simulate_trades(probs_val, y_val, meta_val, thresholds)
# pick best threshold by final_capital
best = max(res_val, key=lambda r: r['final_capital'])
print("Best threshold on val:", best)
# apply to test (same threshold)
res_test = simulate_trades(probs_test, y_test, meta_test, [best['threshold']])
print("Test result:", res_test)
