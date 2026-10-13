"""Safety stock, reorder point, and a periodic-review inventory simulation."""

import numpy as np
import pandas as pd
from scipy.stats import norm

ASSUMPTIONS = [
    "Recorded sales are treated as demand. The dataset has no inventory, so historical stockouts are censored and true demand is unobserved.",
    "Service level is 95% (one-sided normal). Safety stock is z * sigma_error * sqrt(lead_time), with sigma_error equal to the standard deviation of daily validation forecast errors for that model and series.",
    "Lead time is 7 days and the review period is 7 days. An order placed on a review day arrives 7 days later.",
    "The order-up-to level is the forecast of demand over lead time plus review period, plus safety stock. The reorder point is the forecast over the lead time, plus safety stock.",
    "The policy is periodic review: every 7 days, order enough to lift the inventory position (on hand + on order) up to the order-up-to level.",
    "The simulation starts with on-hand inventory equal to that policy's own order-up-to level, so the comparison is not an artefact of starting from zero.",
    "Holding cost is 0.02 per unit per day and stockout cost is 0.50 per unit of unmet sales. These are a cost index, not Favorita's accounting costs. Units differ across families (weight versus count), so a summed cost is only a rough total.",
    "Promotions after the last observed day are assumed to be zero when a review looks past the end of the sample. No fixed order cost and no shelf-life limit are modelled.",
]


def service_z(level):
    return float(norm.ppf(level))


def safety_stock(sigma, lead_time, z):
    return float(z * sigma * np.sqrt(lead_time))


def _cover(forecast, length):
    forecast = np.asarray(forecast, dtype=float)
    if len(forecast) >= length:
        return forecast[:length]
    if len(forecast) == 0:
        return np.zeros(length)
    pad = np.repeat(forecast[-1], length - len(forecast))
    return np.concatenate([forecast, pad])


def reorder_point(forecast, lead_time, ss):
    return float(np.sum(_cover(forecast, lead_time)) + ss)


def order_up_to(forecast, lead_time, review, ss):
    return float(np.sum(_cover(forecast, lead_time + review)) + ss)


def error_sigma(val_forecasts, model, store, family):
    g = val_forecasts.loc[
        (val_forecasts["model"] == model)
        & (val_forecasts["store_nbr"] == store)
        & (val_forecasts["family"] == family)
    ].dropna(subset=["actual", "prediction"])
    if len(g) < 2:
        return 0.0
    return float(np.std(g["actual"].to_numpy() - g["prediction"].to_numpy(), ddof=1))


def simulate_series(dates, actual, forecasts_by_origin, sigma, lead, review, z, holding_cost, stockout_cost):
    """Simulate one series. dates and actual cover the test window only."""
    dates = list(pd.to_datetime(dates))
    actual = np.clip(np.asarray(actual, dtype=float), 0, None)
    origins = sorted(pd.Timestamp(d) for d in forecasts_by_origin)
    origin_set = set(origins)
    first = origins[0]
    ss = safety_stock(sigma, lead, z)
    S0 = order_up_to(forecasts_by_origin[first], lead, review, ss)
    rop0 = reorder_point(forecasts_by_origin[first], lead, ss)

    on_hand = S0
    pipeline = []
    rows = []
    lost_total = 0.0
    demand_total = 0.0
    holding_total = 0.0

    for t, day in enumerate(dates):
        arrived = sum(qty for due, qty in pipeline if due == t)
        pipeline = [(due, qty) for due, qty in pipeline if due != t]
        on_hand += arrived

        ordered = 0.0
        if day in origin_set:
            forecast = np.asarray(forecasts_by_origin[day], dtype=float)
            target = order_up_to(forecast, lead, review, ss)
            on_order = sum(qty for due, qty in pipeline if due > t)
            ordered = max(0.0, target - (on_hand + on_order))
            if ordered > 0:
                pipeline.append((t + lead, ordered))

        demand = actual[t]
        if on_hand >= demand:
            sold = demand
            lost = 0.0
            on_hand -= demand
        else:
            sold = on_hand
            lost = demand - on_hand
            on_hand = 0.0

        demand_total += demand
        lost_total += lost
        holding_total += on_hand
        rows.append(
            {
                "date": day,
                "on_hand": on_hand,
                "demand": demand,
                "sold": sold,
                "lost_sales": lost,
                "order_qty": ordered,
                "received": arrived,
            }
        )

    n = len(dates)
    fill = 1.0 if demand_total == 0 else 1.0 - lost_total / demand_total
    stockout_days = sum(1 for r in rows if r["lost_sales"] > 1e-8)
    summary = {
        "safety_stock": ss,
        "reorder_point": rop0,
        "order_up_to": S0,
        "sigma_error": sigma,
        "fill_rate": float(fill),
        "stockout_rate": float(stockout_days / n) if n else 0.0,
        "avg_inventory": float(holding_total / n) if n else 0.0,
        "holding_cost": float(holding_cost * holding_total),
        "stockout_cost": float(stockout_cost * lost_total),
        "total_cost": float(holding_cost * holding_total + stockout_cost * lost_total),
        "lost_sales": float(lost_total),
        "demand": float(demand_total),
    }
    return pd.DataFrame(rows), summary


def run_inventory(test_forecasts, val_forecasts, panel, bounds, models, lead, review, service_level, holding_cost, stockout_cost):
    """Compare inventory policies driven by each forecast model."""
    z = service_z(service_level)
    test_start = pd.Timestamp(bounds["test_start"])
    test_end = pd.Timestamp(bounds["test_end"])
    daily_frames = []
    summaries = []

    series_keys = (
        test_forecasts[["store_nbr", "family"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    for store, family in series_keys:
        actual = panel.loc[
            (panel["store_nbr"] == store)
            & (panel["family"] == family)
            & (panel["date"] >= test_start)
            & (panel["date"] <= test_end)
            & panel["in_sample"]
        ].sort_values("date")
        for model in models:
            fc = test_forecasts.loc[
                (test_forecasts["model"] == model)
                & (test_forecasts["store_nbr"] == store)
                & (test_forecasts["family"] == family)
            ]
            by_origin = {}
            for origin, g in fc.groupby("origin"):
                g = g.sort_values("date")
                by_origin[pd.Timestamp(origin)] = g["prediction"].to_numpy(dtype=float)
            if not by_origin:
                continue
            sigma = error_sigma(val_forecasts, model, store, family)
            daily, summary = simulate_series(
                actual["date"],
                actual["sales"],
                by_origin,
                sigma,
                lead,
                review,
                z,
                holding_cost,
                stockout_cost,
            )
            daily["store_nbr"] = int(store)
            daily["family"] = family
            daily["model"] = model
            summary["store_nbr"] = int(store)
            summary["family"] = family
            summary["model"] = model
            daily_frames.append(daily)
            summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    return summary_df, daily_df, z
