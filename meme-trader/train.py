# train.py
import os, numpy as np, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from utils import fetch_ohlcv_df, featurize, compute_max_future_return_array

# CONFIG
coins = ["DOGE/USDT","SHIB/USDT","PEPE/USDT","BONK/USDT"]
timeframe = '15m'
limit = 34560
seq_len = 16
horizons = [3,6,12]
pos_threshold = 0.01
min_vol = 0.0005

batch_size = 128
epochs = 40
lr = 1e-3
R_SCALE = 0.02

features = [
 'open','high','low','close','volume','rsi','atr','ma_diff','vol_change',
 'macd','macd_signal','macd_hist','stochrsi_k','stochrsi_d','obv','bb_pct','vwap_dist'
]

# build dataset across coins (per-coin chronological splits)
X_all, y_all, rawret_all = [], [], []
meta = []
for coin in coins:
    df = featurize(fetch_ohlcv_df(coin, timeframe, limit))
    # compute raw max-ret aligned
    raw_max = compute_max_future_return_array(df['close'].values, horizons)
    # construct sequences
    for i in range(seq_len, len(df)):
        # optional volatility filter
        vol_rel = df['atr'].iat[i] / (df['close'].iat[i] + 1e-9)
        if vol_rel < min_vol:
            continue
        X_all.append(df[features].iloc[i-seq_len:i].values)
        y_all.append(1 if raw_max[i] > pos_threshold else 0)
        rawret_all.append(raw_max[i])
        meta.append({'coin': coin, 'idx': i})
X = np.array(X_all); y = np.array(y_all); rawret = np.array(rawret_all)
# train/val/test chronological split (stratified per coin would be more complex; using global chronological)
n = len(X); i1 = int(0.7*n); i2 = int(0.85*n)
X_train, y_train, raw_train = X[:i1], y[:i1], rawret[:i1]
X_val,   y_val,   raw_val   = X[i1:i2], y[i1:i2], rawret[i1:i2]
X_test,  y_test,  raw_test  = X[i2:], y[i2:], rawret[i2:]

# scaler: fit on flattened train
ns = X_train.shape[2]
scaler = StandardScaler().fit(X_train.reshape(-1, ns))
X_train_scaled = scaler.transform(X_train.reshape(-1, ns)).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, ns)).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, ns)).reshape(X_test.shape)

# prepare combined y for pnl-aware loss
def prepare_combined_y(y_binary, raw_max_ret, R_SCALE=R_SCALE):
    r_norm = np.tanh(raw_max_ret / R_SCALE).astype('float32')
    return np.vstack([y_binary.astype('float32'), r_norm]).T

y_train_comb = prepare_combined_y(y_train, raw_train)
y_val_comb   = prepare_combined_y(y_val,   raw_val)

# build transformer
def positional_encoding_layer(x):
    seq_len = x.shape[1]; d_model = x.shape[2]
    pos = tf.range(seq_len)[:, tf.newaxis]
    i = tf.range(d_model)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000., (2 * (i//2)) / tf.cast(d_model, tf.float32))
    angle_rads = tf.cast(pos, tf.float32) * angle_rates
    s = tf.concat([tf.sin(angle_rads[:, 0::2]), tf.cos(angle_rads[:, 1::2])], axis=-1)
    return x + s

inp = Input(shape=(seq_len, ns))
x = layers.Dense(128)(inp)
# add pos enc
pos_enc = layers.Lambda(lambda z: positional_encoding_layer(z))(x)
x = pos_enc
for _ in range(2):
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    attn = layers.Dropout(0.2)(attn)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    ff = layers.Dense(256, activation='relu')(x)
    ff = layers.Dense(128)(ff)
    ff = layers.Dropout(0.2)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1, activation='sigmoid')(x)
model = Model(inp, out)

# pnl-aware loss (same as provided earlier)
import tensorflow.keras.backend as K
def expected_pnl_plus_bce_loss(alpha=1.0, beta=0.5):
    def loss_fn(y_true, y_pred):
        eps = K.epsilon()
        y_lab = K.clip(y_true[:, 0], 0.0, 1.0)
        r_norm = K.clip(y_true[:, 1], -1.0, 1.0)
        p = K.clip(y_pred[:, 0], eps, 1.0-eps)
        expected_pnl = K.mean(p * r_norm)
        bce = K.mean(K.binary_crossentropy(y_lab, p))
        return -alpha * expected_pnl + beta * bce
    return loss_fn

loss_fn = expected_pnl_plus_bce_loss(alpha=1.0, beta=0.5)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_fn, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
model.summary()

callbacks = [EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
             ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5),
             ModelCheckpoint("best_transformer_pnl.keras", monitor='val_auc', save_best_only=True, mode='max')]

model.fit(X_train_scaled, y_train_comb, validation_data=(X_val_scaled, y_val_comb),
          epochs=epochs, batch_size=batch_size, callbacks=callbacks)

# save artifacts
model.save("transformer_pnl_model.keras")
joblib.dump(scaler, "scaler.pkl")
np.save("meta.npy", np.array(meta))
print("Saved model + scaler + meta")
