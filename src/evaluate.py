"""Forecast metrics, comparison tables, and residual plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mae(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def rmsle(y, yhat):
    """Kaggle metric. Predictions are clipped at zero before the log."""
    y = np.clip(np.asarray(y, dtype=float), 0, None)
    yhat = np.clip(np.asarray(yhat, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(yhat) - np.log1p(y)) ** 2)))


def smape(y, yhat):
    """Symmetric MAPE in percent. A day with both values at zero contributes 0."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = np.abs(y) + np.abs(yhat)
    diff = np.abs(y - yhat)
    out = np.zeros(len(y), dtype=float)
    mask = denom > 1e-8
    out[mask] = 2.0 * diff[mask] / denom[mask]
    return float(np.mean(out) * 100)


def mape(y, yhat):
    """MAPE in percent, computed only where actual sales are positive."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.abs(y) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def metric_row(y, yhat):
    return {
        "MAE": mae(y, yhat),
        "RMSE": rmse(y, yhat),
        "MAPE": mape(y, yhat),
        "sMAPE": smape(y, yhat),
        "RMSLE": rmsle(y, yhat),
        "n": int(len(y)),
    }


def comparison_tables(forecasts):
    """Overall and per-family tables. forecasts has model, family, actual, prediction."""
    overall = []
    for model, g in forecasts.groupby("model"):
        row = metric_row(g["actual"], g["prediction"])
        row["model"] = model
        overall.append(row)
    overall = pd.DataFrame(overall).set_index("model").sort_values("RMSLE")

    by_family = []
    for (model, family), g in forecasts.groupby(["model", "family"]):
        row = metric_row(g["actual"], g["prediction"])
        row["model"] = model
        row["family"] = family
        by_family.append(row)
    by_family = pd.DataFrame(by_family).sort_values(["family", "RMSLE"])

    by_series = []
    for (model, store, family), g in forecasts.groupby(["model", "store_nbr", "family"]):
        row = metric_row(g["actual"], g["prediction"])
        row["model"] = model
        row["store_nbr"] = int(store)
        row["family"] = family
        by_series.append(row)
    by_series = pd.DataFrame(by_series)
    return overall, by_family, by_series


def plot_metrics(overall, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    table = overall.sort_values("RMSLE")
    ax.barh(table.index.astype(str), table["RMSLE"], color="#3b6ea5")
    ax.set_xlabel("RMSLE (lower is better)")
    ax.set_title("Test RMSLE by model")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_forecasts(forecasts, path, model="lstm"):
    subset = forecasts.loc[forecasts["model"] == model].copy()
    if subset.empty:
        return
    families = list(subset["family"].unique())
    stores = sorted(subset["store_nbr"].unique())
    fig, axes = plt.subplots(len(families), len(stores), figsize=(4 * len(stores), 2.4 * len(families)), sharex=True)
    if len(families) == 1 and len(stores) == 1:
        axes = np.array([[axes]])
    elif len(families) == 1:
        axes = np.array([axes])
    elif len(stores) == 1:
        axes = np.array([[ax] for ax in axes])
    for i, family in enumerate(families):
        for j, store in enumerate(stores):
            ax = axes[i, j]
            g = subset.loc[(subset["family"] == family) & (subset["store_nbr"] == store)].sort_values("date")
            ax.plot(g["date"], g["actual"], color="black", lw=1.2, label="actual")
            ax.plot(g["date"], g["prediction"], color="#c45c26", lw=1.2, label=model)
            ax.set_title(f"Store {store} · {family}", fontsize=9)
            ax.tick_params(labelsize=7)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"Test forecasts vs actual ({model})", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(forecasts, path, model="lstm"):
    g = forecasts.loc[forecasts["model"] == model].copy()
    if g.empty:
        return
    g["residual"] = g["actual"] - g["prediction"]
    g["dow"] = pd.to_datetime(g["date"]).dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].hist(g["residual"], bins=30, color="#3b6ea5")
    axes[0].set_title("Residual histogram")
    axes[0].set_xlabel("actual − forecast")
    axes[1].scatter(g["prediction"], g["residual"], s=8, alpha=0.4, color="#3b6ea5")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Residual vs forecast")
    axes[1].set_xlabel("forecast")
    means = g.groupby("dow")["residual"].mean().reindex(order)
    axes[2].bar(range(7), means.values, color="#3b6ea5")
    axes[2].set_xticks(range(7))
    axes[2].set_xticklabels([d[:3] for d in order])
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Mean residual by weekday")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {
        "residual_mean": float(g["residual"].mean()),
        "residual_std": float(g["residual"].std()),
        "mean_residual_by_dow": {k: float(v) for k, v in means.dropna().items()},
    }


def plot_loss(history, path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(history["loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE on scaled sales")
    ax.set_title("LSTM training and validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def self_check():
    y = np.array([1.0, 2.0, 0.0])
    assert abs(mae(y, y)) < 1e-9
    assert abs(rmsle(y, y)) < 1e-9
    assert abs(mae(np.array([1.0, 2.0]), np.array([1.0, 4.0])) - 1.0) < 1e-9
    assert abs(smape(np.array([0.0]), np.array([0.0]))) < 1e-9
