"""Baselines, gradient boosting, and the LSTM."""

import os
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.features import CAL_COLS, LAG_COLS, future_feature_columns, past_feature_columns

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def block_starts(start, end, horizon, step=None):
    """Origins whose full horizon still falls on or before end."""
    step = horizon if step is None else step
    starts = []
    cur = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while cur + pd.Timedelta(days=horizon - 1) <= end:
        starts.append(cur)
        cur = cur + pd.Timedelta(days=step)
    return starts


def review_starts(start, end, step):
    starts = []
    cur = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while cur <= end:
        starts.append(cur)
        cur = cur + pd.Timedelta(days=step)
    return starts


def naive_forecast(history, horizon):
    return np.repeat(float(history[-1]), horizon)


def seasonal_naive_forecast(history, horizon, season=7):
    hist = list(np.asarray(history, dtype=float))
    out = []
    for _ in range(horizon):
        out.append(hist[-season])
        hist.append(out[-1])
    return np.asarray(out, dtype=float)


def moving_average_forecast(history, horizon, window=7):
    value = float(np.mean(history[-window:]))
    return np.repeat(value, horizon)


def _history_series(history, n_days):
    values = np.asarray(history, dtype=float)[-n_days:]
    index = pd.date_range(end="2017-01-01", periods=len(values), freq="D")
    # The dates are only there so statsmodels sees a regular weekly index.
    # The forecast itself does not depend on the calendar labels.
    return pd.Series(values, index=index)


def holt_winters_forecast(history, horizon, n_days):
    y = _history_series(history, n_days)
    try:
        model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, use_brute=False)
        pred = np.asarray(fit.forecast(horizon), dtype=float)
    except Exception as exc:
        print(f"    Holt-Winters failed ({exc}); using seasonal naive.")
        pred = seasonal_naive_forecast(history, horizon)
    return np.clip(pred, 0, None)


def sarima_forecast(history, horizon, n_days):
    """A small seasonal ARIMA. Fixed orders, not an automatic search.

    Order (1, 1, 1) x (0, 1, 1, 7) is the weekly airline-style model. It is
    fit on the last year only, because a weekly model does not need the full
    history and the fit is much slower on the whole sample.
    """
    y = _history_series(history, n_days)
    try:
        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=50, method="lbfgs")
        pred = np.asarray(fit.forecast(horizon), dtype=float)
    except Exception as exc:
        print(f"    SARIMA failed ({exc}); using seasonal naive.")
        pred = seasonal_naive_forecast(history, horizon)
    return np.clip(pred, 0, None)


def _append_forecast_rows(rows, g, origin, prediction, model_name, store, family):
    future = g.loc[g["date"] >= origin].sort_values("date").head(len(prediction))
    for step, (_, row) in enumerate(future.iterrows()):
        if not bool(row["in_sample"]) or pd.isna(row["sales"]):
            # Still keep rows past the sample if the caller asked for them:
            # inventory needs the horizon even when actual sales are unknown.
            actual = np.nan if pd.isna(row["sales"]) else float(row["sales"])
        else:
            actual = float(row["sales"])
        rows.append(
            {
                "model": model_name,
                "store_nbr": int(store),
                "family": family,
                "origin": pd.Timestamp(origin),
                "date": pd.Timestamp(row["date"]),
                "horizon_day": step + 1,
                "actual": actual,
                "prediction": float(prediction[step]),
                "in_sample": bool(row["in_sample"]),
            }
        )


def forecast_history_models(panel, origins, horizon, stat_history_days, which=None):
    """Refit the statistical baselines at each forecast origin."""
    specs = {
        "naive": lambda hist: naive_forecast(hist, horizon),
        "seasonal_naive": lambda hist: seasonal_naive_forecast(hist, horizon),
        "moving_average": lambda hist: moving_average_forecast(hist, horizon),
        "holt_winters": lambda hist: holt_winters_forecast(hist, horizon, stat_history_days),
        "sarima": lambda hist: sarima_forecast(hist, horizon, stat_history_days),
    }
    if which is not None:
        specs = {name: specs[name] for name in which}
    rows = []
    grouped = list(panel.groupby(["store_nbr", "family"], sort=False))
    for name, predict in specs.items():
        print(f"  forecasting {name} at {len(origins)} origins")
        for (store, family), g in grouped:
            g = g.sort_values("date")
            for origin in origins:
                hist = g.loc[(g["date"] < origin) & g["sales"].notna(), "sales"].to_numpy(dtype=float)
                if len(hist) < 28:
                    continue
                if name in ("holt_winters", "sarima"):
                    print(f"    {name} store {store} {family} origin {pd.Timestamp(origin).date()}")
                pred = predict(hist)
                _append_forecast_rows(rows, g, origin, pred, name, store, family)
    return pd.DataFrame(rows)


def lag_dict(history):
    hist = np.asarray(history, dtype=float)
    return {
        "lag_7": hist[-7],
        "lag_14": hist[-14],
        "lag_28": hist[-28],
        "roll_mean_7": float(hist[-7:].mean()),
        "roll_std_7": float(hist[-7:].std(ddof=1)),
        "roll_mean_28": float(hist[-28:].mean()),
        "roll_std_28": float(hist[-28:].std(ddof=1)),
    }


def assert_lags_match(panel):
    """The recursive forecaster and the training columns must use the same lags."""
    store = int(panel["store_nbr"].iloc[0])
    family = panel["family"].iloc[0]
    g = panel.loc[(panel["store_nbr"] == store) & (panel["family"] == family) & panel["in_sample"]]
    g = g.sort_values("date").reset_index(drop=True)
    sales = g["sales"].to_numpy(dtype=float)
    t = 40
    built = lag_dict(sales[:t])
    row = g.loc[t]
    for key, value in built.items():
        if abs(value - float(row[key])) > 1e-3:
            raise AssertionError(f"Lag mismatch on {key}: {value} vs {row[key]}")


HGB_BASE = LAG_COLS + ["onpromotion", "oil_lag1"] + CAL_COLS


def fit_hgb(panel, val_start, dummy_cols, params):
    cols = HGB_BASE + dummy_cols
    train = panel.loc[(panel["date"] < val_start) & panel["in_sample"]].dropna(subset=cols + ["sales"])
    model = HistGradientBoostingRegressor(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        min_samples_leaf=20,
        random_state=params["seed"],
    )
    model.fit(train[cols].to_numpy(dtype=float), train["sales"].to_numpy(dtype=float))
    return model, cols


def forecast_hgb(model, columns, panel, origins, horizon, dummy_cols):
    rows = []
    for (store, family), g in panel.groupby(["store_nbr", "family"], sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        sales = g["sales"].to_numpy(dtype=float)
        date_to_i = {pd.Timestamp(d): i for i, d in enumerate(g["date"])}
        for origin in origins:
            origin = pd.Timestamp(origin)
            if origin not in date_to_i:
                continue
            i0 = date_to_i[origin]
            if i0 < 28:
                continue
            work = sales.copy()
            last_oil = float(g.loc[i0 - 1, "oil"])
            preds = []
            for step in range(horizon):
                t = i0 + step
                if t >= len(g):
                    break
                feats = lag_dict(work[:t])
                row = g.loc[t]
                feats["onpromotion"] = float(row["onpromotion"])
                feats["oil_lag1"] = last_oil
                for col in CAL_COLS:
                    feats[col] = float(row[col])
                for col in dummy_cols:
                    feats[col] = float(row[col])
                x = np.array([[feats[c] for c in columns]], dtype=float)
                yhat = max(0.0, float(model.predict(x)[0]))
                work[t] = yhat
                preds.append(yhat)
            _append_forecast_rows(rows, g, origin, np.asarray(preds), "hist_gradient_boosting", store, family)
    return pd.DataFrame(rows)


def make_sequences(panel, window, horizon, dummy_cols, val_start, val_end):
    """Sliding windows. A window is training data only if its whole target is before validation."""
    past_cols = past_feature_columns(dummy_cols)
    future_cols = future_feature_columns()
    x_past, x_future, y, splits = [], [], [], []

    for _, g in panel.groupby(["store_nbr", "family"], sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        if g[past_cols].isna().any().any():
            # Leading lag rows and the post-sample extension can contain null sales.
            # Sequences are only started where the past window is complete.
            pass
        values_past = g[past_cols].to_numpy(dtype=np.float32)
        values_future = g[future_cols].to_numpy(dtype=np.float32)
        target = g["sales_scaled"].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()
        for i in range(window, len(g) - horizon + 1):
            past = values_past[i - window : i]
            fut = values_future[i : i + horizon].copy()
            tgt = target[i : i + horizon]
            if np.isnan(past).any() or np.isnan(fut).any() or np.isnan(tgt).any():
                continue
            # Future oil is not known at the origin. Carry forward the last price.
            oil_pos = future_cols.index("oil_scaled")
            fut[:, oil_pos] = past[-1, past_cols.index("oil_scaled")]
            first = pd.Timestamp(dates[i])
            last = pd.Timestamp(dates[i + horizon - 1])
            if last < val_start:
                split = "train"
            elif first >= val_start and last <= val_end:
                split = "val"
            else:
                continue
            x_past.append(past)
            x_future.append(fut)
            y.append(tgt)
            splits.append(split)

    splits = np.asarray(splits)
    data = {
        "past": np.asarray(x_past, dtype=np.float32),
        "future": np.asarray(x_future, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "split": splits,
    }
    return data, past_cols, future_cols


def build_lstm(window, n_past, horizon, n_future, units, n_layers, dropout, lr, seed):
    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    past_in = keras.Input(shape=(window, n_past), name="past")
    fut_in = keras.Input(shape=(horizon, n_future), name="future")

    x = keras.layers.LSTM(units, return_sequences=(n_layers > 1))(past_in)
    x = keras.layers.Dropout(dropout)(x)
    if n_layers > 1:
        x = keras.layers.LSTM(max(units // 2, 8))(x)
        x = keras.layers.Dropout(dropout)(x)

    # Promotions and the calendar over the horizon are known when the order is placed.
    flat = keras.layers.Flatten()(fut_in)
    flat = keras.layers.Dense(32, activation="relu")(flat)
    x = keras.layers.Concatenate()([x, flat])
    out = keras.layers.Dense(horizon)(x)

    model = keras.Model([past_in, fut_in], out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def train_lstm(model, data, epochs, batch, patience):
    from tensorflow import keras

    train = data["split"] == "train"
    val = data["split"] == "val"
    class _EpochLog(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            print(
                f"    epoch {epoch + 1}: loss={logs.get('loss', float('nan')):.4f} "
                f"val_loss={logs.get('val_loss', float('nan')):.4f}"
            )

    callbacks = [
        _EpochLog(),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]
    history = model.fit(
        [data["past"][train], data["future"][train]],
        data["y"][train],
        validation_data=(
            [data["past"][val], data["future"][val]],
            data["y"][val],
        ),
        epochs=epochs,
        batch_size=batch,
        shuffle=True,
        verbose=0,
        callbacks=callbacks,
    )
    return history.history


def _inverse(scaler, values):
    pred = scaler.inverse_transform(np.asarray(values, dtype=float).reshape(-1, 1)).ravel()
    return np.clip(pred, 0, None)


def forecast_lstm(model, panel, origins, horizon, window, dummy_cols, scalers):
    past_cols = past_feature_columns(dummy_cols)
    future_cols = future_feature_columns()
    rows = []
    for (store, family), g in panel.groupby(["store_nbr", "family"], sort=False):
        g = g.sort_values("date").set_index("date")
        scaler = scalers["sales"][(int(store), family)]
        for origin in origins:
            origin = pd.Timestamp(origin)
            hist_end = origin - pd.Timedelta(days=1)
            hist_start = hist_end - pd.Timedelta(days=window - 1)
            fut_idx = pd.date_range(origin, periods=horizon, freq="D")
            if hist_start not in g.index or fut_idx[-1] not in g.index:
                continue
            past = g.loc[hist_start:hist_end, past_cols].to_numpy(dtype=np.float32)
            fut = g.loc[fut_idx, future_cols].to_numpy(dtype=np.float32).copy()
            if past.shape[0] != window or fut.shape[0] != horizon:
                continue
            if np.isnan(past).any() or np.isnan(fut).any():
                continue
            fut[:, future_cols.index("oil_scaled")] = past[-1, past_cols.index("oil_scaled")]
            pred_scaled = model.predict([past[None, ...], fut[None, ...]], verbose=0)[0]
            pred = _inverse(scaler, pred_scaled)
            _append_forecast_rows(rows, g.reset_index(), origin, pred, "lstm", store, family)
    return pd.DataFrame(rows)


def search_lstm(panel, scalers, dummy_cols, bounds, search, epochs, batch, patience, seed, horizon, model_path):
    """Train each small config and keep the one with the best validation RMSLE.

    The winning model is written to disk inside the loop. The next candidate
    calls clear_session(), which would invalidate a model kept only in memory.
    """
    import tensorflow as tf
    from tensorflow import keras

    from src.evaluate import rmsle

    val_origins = block_starts(bounds["val_start"], bounds["val_end"], horizon)
    results = []
    best = None

    for cfg in search:
        print(f"  LSTM search {cfg}")
        keras.backend.clear_session()
        tf.random.set_seed(seed)
        data, _, _ = make_sequences(
            panel,
            cfg["window"],
            horizon,
            dummy_cols,
            bounds["val_start"],
            bounds["val_end"],
        )
        n_train = int((data["split"] == "train").sum())
        n_val = int((data["split"] == "val").sum())
        print(f"    sequences train={n_train} val={n_val}")
        model = build_lstm(
            cfg["window"],
            data["past"].shape[-1],
            horizon,
            data["future"].shape[-1],
            cfg["units"],
            cfg["layers"],
            cfg["dropout"],
            cfg["lr"],
            seed,
        )
        history = train_lstm(model, data, epochs, batch, patience)
        val_fc = forecast_lstm(
            model, panel, val_origins, horizon, cfg["window"], dummy_cols, scalers
        )
        scored = val_fc.loc[val_fc["actual"].notna()]
        score = rmsle(scored["actual"], scored["prediction"])
        row = {
            **cfg,
            "val_rmsle": score,
            "epochs_ran": len(history["loss"]),
            "final_train_loss": float(history["loss"][-1]),
            "best_val_loss": float(np.min(history["val_loss"])),
            "n_train_sequences": n_train,
            "n_val_sequences": n_val,
        }
        results.append(row)
        print(f"    val RMSLE={score:.4f} epochs={row['epochs_ran']}")
        if best is None or score < best["val_rmsle"]:
            model.save(model_path)
            best = {"val_rmsle": score, "history": history, "cfg": cfg}
        del model

    return best, pd.DataFrame(results)
