"""Run EDA, baselines, the LSTM search, evaluation, and the inventory simulation."""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    ARTIFACT_DIR,
    DATA_DIR,
    HOLDING_COST,
    HORIZON,
    LEAD_TIME,
    LSTM_BATCH,
    LSTM_EPOCHS,
    LSTM_PATIENCE,
    LSTM_SEARCH,
    N_FAMILIES,
    N_STORES,
    REVIEW_PERIOD,
    SEED,
    SERVICE_LEVEL,
    STAT_HISTORY_DAYS,
    STOCKOUT_COST,
    TEST_DAYS,
    VAL_DAYS,
)
from src.data import choose_subset, load_raw, profile_raw, build_panel, split_dates
from src.evaluate import (
    comparison_tables,
    metric_row,
    plot_forecasts,
    plot_loss,
    plot_metrics,
    plot_residuals,
    self_check,
)
from src.features import add_calendar, add_dummies, add_lags, scale_for_lstm
from src.inventory import ASSUMPTIONS, run_inventory
from src.models import (
    assert_lags_match,
    block_starts,
    fit_hgb,
    forecast_hgb,
    forecast_history_models,
    forecast_lstm,
    review_starts,
    search_lstm,
    set_seeds,
)
from src.plots import run_eda

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_json_ready(payload), indent=2))


def _score_origins(forecasts, origins):
    origin_set = set(pd.Timestamp(d) for d in origins)
    g = forecasts.loc[
        forecasts["origin"].map(pd.Timestamp).isin(origin_set) & forecasts["actual"].notna()
    ].copy()
    return g


def main():
    set_seeds(SEED)
    self_check()
    art = ARTIFACT_DIR
    for folder in ["figures", "metrics", "forecasts", "inventory", "models", "eda", "processed"]:
        (art / folder).mkdir(parents=True, exist_ok=True)

    print("Loading data from", DATA_DIR)
    raw = load_raw(DATA_DIR)
    profile = profile_raw(raw)
    _write_json(art / "eda" / "profile.json", profile)
    (art / "eda" / "holidays_info.txt").write_text(profile["holidays"]["info"])
    print("Train shape", profile["train"]["shape"], "missing days", profile["train"]["missing_days"])
    print(raw["holidays"].head(3).to_string(index=False))

    print("EDA")
    eda_stats = run_eda(raw, art / "figures")
    _write_json(art / "eda" / "summary.json", eda_stats)

    store_ids, families, family_sales, store_sales = choose_subset(raw["train"], N_STORES, N_FAMILIES)
    print("Stores", store_ids)
    print("Families", families)
    subset_info = {
        "stores": store_ids,
        "families": families,
        "store_sales": {int(k): float(v) for k, v in store_sales.items()},
        "family_sales": {str(k): float(v) for k, v in family_sales.items()},
    }
    _write_json(art / "eda" / "subset.json", subset_info)

    print("Building panel")
    panel = build_panel(raw, store_ids, families)
    panel = add_calendar(panel)
    panel = add_lags(panel)
    panel, dummy_cols = add_dummies(panel)
    bounds = split_dates(panel, VAL_DAYS, TEST_DAYS)
    print("Split", {k: str(v.date()) for k, v in bounds.items()})
    panel, scalers = scale_for_lstm(panel, bounds["val_start"])
    assert_lags_match(panel)

    first_sale = (
        panel.loc[(panel["in_sample"]) & (panel["sales"] > 0)]
        .groupby(["store_nbr", "family"])["date"]
        .min()
    )
    print("First positive sales date per series:")
    print(first_sale.to_string())

    keep_cols = [
        "date", "store_nbr", "family", "sales", "onpromotion", "oil",
        "holiday_national", "holiday_regional", "holiday_local", "is_event",
        "is_payday", "is_day_after_payday", "is_closed", "city", "state",
        "store_type", "cluster", "in_sample",
    ]
    panel.loc[panel["in_sample"], keep_cols].to_parquet(art / "processed" / "panel.parquet", index=False)
    joblib.dump(scalers, art / "models" / "scalers.joblib")

    val_origins = block_starts(bounds["val_start"], bounds["val_end"], HORIZON)
    test_accuracy_origins = block_starts(bounds["test_start"], bounds["test_end"], HORIZON)
    test_rolling_origins = block_starts(bounds["test_start"], bounds["test_end"], HORIZON, step=7)
    test_review_origins = review_starts(bounds["test_start"], bounds["test_end"], REVIEW_PERIOD)
    model_origins = sorted(set(val_origins + test_review_origins))
    print("Validation origins", [str(d.date()) for d in val_origins])
    print("Test review origins", [str(d.date()) for d in test_review_origins])

    fast = forecast_history_models(
        panel,
        model_origins,
        HORIZON,
        STAT_HISTORY_DAYS,
        which=["naive", "seasonal_naive", "moving_average", "holt_winters", "sarima"],
    )

    print("Gradient boosting search")
    hgb_grid = [
        {"max_depth": 4, "learning_rate": 0.1, "max_iter": 200, "seed": SEED},
        {"max_depth": 8, "learning_rate": 0.05, "max_iter": 250, "seed": SEED},
    ]
    hgb_rows = []
    best_hgb = None
    best_hgb_score = None
    from src.evaluate import rmsle

    for params in hgb_grid:
        model, cols = fit_hgb(panel, bounds["val_start"], dummy_cols, params)
        val_fc = forecast_hgb(model, cols, panel, val_origins, HORIZON, dummy_cols)
        score = rmsle(val_fc["actual"], val_fc["prediction"])
        print(f"  HGB {params} val RMSLE={score:.4f}")
        hgb_rows.append({**params, "val_rmsle": score})
        if best_hgb_score is None or score < best_hgb_score:
            best_hgb_score = score
            best_hgb = (model, cols, val_fc)
    hgb_model, hgb_cols, hgb_val = best_hgb
    hgb_test = forecast_hgb(hgb_model, hgb_cols, panel, test_review_origins, HORIZON, dummy_cols)
    joblib.dump(hgb_model, art / "models" / "hgb.joblib")
    pd.DataFrame(hgb_rows).to_csv(art / "metrics" / "hgb_search.csv", index=False)

    print("LSTM search")
    model_path = art / "models" / "lstm.keras"
    best_lstm, search_table = search_lstm(
        panel,
        scalers,
        dummy_cols,
        bounds,
        LSTM_SEARCH,
        LSTM_EPOCHS,
        LSTM_BATCH,
        LSTM_PATIENCE,
        SEED,
        HORIZON,
        model_path,
    )
    search_table.to_csv(art / "metrics" / "lstm_search.csv", index=False)
    _write_json(art / "models" / "lstm_history.json", best_lstm["history"])
    _write_json(art / "models" / "lstm_config.json", best_lstm["cfg"])
    plot_loss(best_lstm["history"], art / "figures" / "09_lstm_loss.png")

    from tensorflow import keras

    lstm = keras.models.load_model(model_path)
    lines = []
    lstm.summary(print_fn=lines.append)
    (art / "models" / "lstm_summary.txt").write_text("\n".join(lines))
    lstm_val = forecast_lstm(
        lstm, panel, val_origins, HORIZON, best_lstm["cfg"]["window"], dummy_cols, scalers
    )
    lstm_test = forecast_lstm(
        lstm, panel, test_review_origins, HORIZON, best_lstm["cfg"]["window"], dummy_cols, scalers
    )

    val_forecasts = pd.concat([fast, hgb_val, lstm_val], ignore_index=True)
    val_forecasts = val_forecasts.loc[val_forecasts["origin"].map(pd.Timestamp).isin(set(val_origins))]
    test_forecasts = pd.concat([fast, hgb_test, lstm_test], ignore_index=True)
    test_forecasts = test_forecasts.loc[
        test_forecasts["origin"].map(pd.Timestamp).isin(set(test_review_origins))
    ]
    # fast contains validation origins too; keep test reviews only in the test file.
    val_forecasts.to_csv(art / "forecasts" / "validation_forecasts.csv", index=False)
    test_forecasts.to_csv(art / "forecasts" / "test_forecasts.csv", index=False)

    accuracy = _score_origins(test_forecasts, test_accuracy_origins)
    overall, by_family, by_series = comparison_tables(accuracy)
    overall.to_csv(art / "metrics" / "overall.csv")
    by_family.to_csv(art / "metrics" / "by_family.csv", index=False)
    by_series.to_csv(art / "metrics" / "by_series.csv", index=False)
    print("Test metrics")
    print(overall.round(4).to_string())

    rolling_rows = []
    for origin in test_rolling_origins:
        block = _score_origins(test_forecasts, [origin])
        for model, g in block.groupby("model"):
            row = metric_row(g["actual"], g["prediction"])
            row["model"] = model
            row["origin"] = str(pd.Timestamp(origin).date())
            rolling_rows.append(row)
    rolling = pd.DataFrame(rolling_rows)
    rolling.to_csv(art / "metrics" / "rolling_origin.csv", index=False)
    if len(rolling):
        rolling_mean = rolling.groupby("model")[["MAE", "RMSE", "MAPE", "sMAPE", "RMSLE"]].mean()
        rolling_mean.to_csv(art / "metrics" / "rolling_origin_mean.csv")
        print("Rolling-origin mean RMSLE")
        print(rolling_mean["RMSLE"].sort_values().round(4).to_string())

    plot_metrics(overall, art / "figures" / "10_model_rmsle.png")
    plot_forecasts(accuracy, art / "figures" / "11_forecast_vs_actual_lstm.png", model="lstm")
    plot_forecasts(accuracy, art / "figures" / "11b_forecast_vs_actual_seasonal_naive.png", model="seasonal_naive")
    residual_stats = plot_residuals(accuracy, art / "figures" / "12_residuals_lstm.png", model="lstm")
    _write_json(art / "metrics" / "lstm_residuals.json", residual_stats or {})

    best_name = overall["RMSLE"].idxmin()
    lstm_rmsle = float(overall.loc["lstm", "RMSLE"]) if "lstm" in overall.index else None
    seasonal_rmsle = float(overall.loc["seasonal_naive", "RMSLE"]) if "seasonal_naive" in overall.index else None

    inventory_models = [
        "seasonal_naive",
        "moving_average",
        "holt_winters",
        "sarima",
        "hist_gradient_boosting",
        "lstm",
    ]
    print("Inventory simulation")
    summary, daily, z = run_inventory(
        test_forecasts,
        val_forecasts,
        panel,
        bounds,
        inventory_models,
        LEAD_TIME,
        REVIEW_PERIOD,
        SERVICE_LEVEL,
        HOLDING_COST,
        STOCKOUT_COST,
    )
    summary.to_csv(art / "inventory" / "summary.csv", index=False)
    daily.to_csv(art / "inventory" / "daily.csv", index=False)
    pooled = summary.groupby("model")[
        ["total_cost", "holding_cost", "stockout_cost", "lost_sales", "demand"]
    ].sum()
    # Average inventory is a level, so it is averaged across series. Costs are summed.
    pooled["avg_inventory"] = summary.groupby("model")["avg_inventory"].mean()
    pooled["fill_rate"] = 1 - pooled["lost_sales"] / pooled["demand"]
    pooled["stockout_rate"] = summary.groupby("model")["stockout_rate"].mean()
    pooled.to_csv(art / "inventory" / "pooled.csv")
    print(pooled.round(3).to_string())

    meta = {
        "bounds": {k: str(v.date()) for k, v in bounds.items()},
        "stores": store_ids,
        "families": families,
        "horizon": HORIZON,
        "val_days": VAL_DAYS,
        "test_days": TEST_DAYS,
        "best_lstm": best_lstm["cfg"],
        "best_lstm_val_rmsle": best_lstm["val_rmsle"],
        "best_test_model_by_rmsle": str(best_name),
        "lstm_test_rmsle": lstm_rmsle,
        "seasonal_naive_test_rmsle": seasonal_rmsle,
        "lstm_beats_seasonal_naive": bool(lstm_rmsle < seasonal_rmsle),
        "service_z": z,
        "lead_time": LEAD_TIME,
        "review_period": REVIEW_PERIOD,
        "holding_cost": HOLDING_COST,
        "stockout_cost": STOCKOUT_COST,
        "assumptions": ASSUMPTIONS,
        "dummy_cols": dummy_cols,
    }
    _write_json(art / "metadata.json", meta)
    print("Done. Best test model by RMSLE:", best_name)


if __name__ == "__main__":
    main()
