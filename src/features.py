"""Calendar, lag, and scaling features. Lags use only earlier days."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CAL_COLS = [
    "holiday_national",
    "holiday_regional",
    "holiday_local",
    "is_event",
    "is_payday",
    "is_day_after_payday",
    "is_closed",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "dom_sin",
    "dom_cos",
]

LAG_COLS = [
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_28",
    "roll_std_28",
]


def add_calendar(df):
    """Day-of-week, month, payday, and sin/cos encodings."""
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["days_in_month"] = out["date"].dt.days_in_month
    # Public-sector wages in Ecuador are paid on the 15th and on the last day.
    out["is_payday"] = ((out["day"] == 15) | (out["day"] == out["days_in_month"])).astype(int)
    # Shoppers often spend the wage on the payday itself and the next day.
    out["is_day_after_payday"] = ((out["day"] == 16) | (out["day"] == 1)).astype(int)

    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dom_sin"] = np.sin(2 * np.pi * out["day"] / 31)
    out["dom_cos"] = np.cos(2 * np.pi * out["day"] / 31)
    return out


def add_lags(df):
    """Lag and rolling features shifted so day t does not see sales from day t.

    Rolling windows are applied to sales.shift(1). A plain rolling mean on
    sales would include the day being predicted and leak the target.
    """
    parts = []
    for _, g in df.groupby(["store_nbr", "family"], sort=False):
        g = g.sort_values("date").copy()
        sales = g["sales"]
        g["lag_7"] = sales.shift(7)
        g["lag_14"] = sales.shift(14)
        g["lag_28"] = sales.shift(28)
        past = sales.shift(1)
        g["roll_mean_7"] = past.rolling(7).mean()
        g["roll_std_7"] = past.rolling(7).std()
        g["roll_mean_28"] = past.rolling(28).mean()
        g["roll_std_28"] = past.rolling(28).std()
        g["oil_lag1"] = g["oil"].shift(1)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_dummies(df):
    """One-hot store and family columns for the shared models."""
    out = df.copy()
    store_ids = sorted(int(s) for s in out["store_nbr"].unique())
    families = sorted(out["family"].unique())
    dummy_cols = []
    for store in store_ids:
        col = f"store_{store}"
        out[col] = (out["store_nbr"] == store).astype(np.float32)
        dummy_cols.append(col)
    for family in families:
        slug = "".join(ch if ch.isalnum() else "_" for ch in family)
        col = f"family_{slug}"
        out[col] = (out["family"] == family).astype(np.float32)
        dummy_cols.append(col)
    return out, dummy_cols


def _fit_minmax(values):
    scaler = MinMaxScaler()
    arr = np.asarray(values, dtype=float).reshape(-1, 1)
    # A constant series has no range. Keep the scaler from dividing by zero.
    if np.nanmax(arr) - np.nanmin(arr) < 1e-8:
        scaler.fit(np.array([[0.0], [1.0]]))
        return scaler
    scaler.fit(arr)
    return scaler


def scale_for_lstm(df, val_start):
    """Scale using the training period only. Returns the frame and the scalers.

    Each series gets its own sales and promotion scaler. A single oil scaler is
    fit on training dates. Validation and test rows are transformed with those
    same scalers, never refit.
    """
    out = df.copy()
    train_rows = out["date"] < val_start
    oil_scaler = _fit_minmax(out.loc[train_rows, "oil"])
    out["oil_scaled"] = oil_scaler.transform(out[["oil"]]).ravel()

    sales_scalers = {}
    promo_scalers = {}
    out["sales_scaled"] = np.nan
    out["promo_scaled"] = np.nan

    parts = []
    for (store, family), g in out.groupby(["store_nbr", "family"], sort=False):
        g = g.sort_values("date").copy()
        fit_mask = (g["date"] < val_start) & g["in_sample"]
        sales_scaler = _fit_minmax(g.loc[fit_mask, "sales"])
        promo_scaler = _fit_minmax(g.loc[fit_mask, "onpromotion"])
        known = g["sales"].notna()
        g.loc[known, "sales_scaled"] = sales_scaler.transform(g.loc[known, ["sales"]]).ravel()
        g["promo_scaled"] = promo_scaler.transform(g[["onpromotion"]]).ravel()
        sales_scalers[(int(store), family)] = sales_scaler
        promo_scalers[(int(store), family)] = promo_scaler
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    scalers = {"sales": sales_scalers, "promo": promo_scalers, "oil": oil_scaler}
    return out, scalers


def past_feature_columns(dummy_cols):
    return ["sales_scaled", "promo_scaled", "oil_scaled"] + CAL_COLS + dummy_cols


def future_feature_columns():
    # Oil is overwritten at forecast time with the last known price.
    return ["promo_scaled", "oil_scaled"] + CAL_COLS
