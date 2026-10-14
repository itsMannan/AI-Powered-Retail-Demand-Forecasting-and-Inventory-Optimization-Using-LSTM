"""Exploratory plots and the summary statistics behind the written findings."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import build_holiday_flags

plt.switch_backend("Agg")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_eda(raw, fig_dir):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    train = raw["train"]
    stores = raw["stores"]
    oil = raw["oil"]
    holidays = raw["holidays"]
    transactions = raw["transactions"]

    daily = train.groupby("date", as_index=False)["sales"].sum()
    daily["roll28"] = daily["sales"].rolling(28, min_periods=7).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["date"], daily["sales"], color="#9aa7b5", lw=0.6, label="daily total")
    ax.plot(daily["date"], daily["roll28"], color="#1f4e79", lw=1.6, label="28-day mean")
    ax.set_title("Total unit sales")
    ax.set_ylabel("units")
    ax.legend()
    _save(fig, fig_dir / "01_total_sales.png")

    family = train.groupby("family")["sales"].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(family.index, family.values, color="#3b6ea5")
    ax.set_title("Total unit sales by product family")
    ax.set_xlabel("units")
    _save(fig, fig_dir / "02_sales_by_family.png")

    store_totals = train.groupby("store_nbr")["sales"].sum().rename("sales").reset_index()
    store_totals = store_totals.merge(stores, on="store_nbr")
    by_type = store_totals.groupby("store_type")["sales"].mean().sort_values()
    by_cluster = store_totals.groupby("cluster")["sales"].mean().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(by_type.index.astype(str), by_type.values, color="#3b6ea5")
    axes[0].set_title("Mean store sales by store type")
    axes[0].set_ylabel("total units per store")
    axes[1].bar(by_cluster.index.astype(str), by_cluster.values, color="#3b6ea5")
    axes[1].set_title("Mean store sales by cluster")
    axes[1].set_xlabel("cluster")
    _save(fig, fig_dir / "03_store_type_cluster.png")

    daily["dow"] = daily["date"].dt.day_name()
    daily["month"] = daily["date"].dt.month
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_mean = daily.groupby("dow")["sales"].mean().reindex(dow_order)
    month_mean = daily.groupby("month")["sales"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(7), dow_mean.values, color="#3b6ea5")
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels([d[:3] for d in dow_order])
    axes[0].set_title("Average daily total by weekday")
    axes[1].bar(month_mean.index, month_mean.values, color="#3b6ea5")
    axes[1].set_title("Average daily total by month")
    axes[1].set_xlabel("month")
    _save(fig, fig_dir / "04_dow_month.png")

    store_day = train.groupby(["date", "store_nbr"], as_index=False)["sales"].sum()
    flags = build_holiday_flags(holidays, stores)
    store_day = store_day.merge(flags, on=["date", "store_nbr"], how="left")
    for col in ["holiday_national", "holiday_regional", "holiday_local", "is_event"]:
        store_day[col] = store_day[col].fillna(0)
    store_day["any_holiday"] = (
        (store_day["holiday_national"] + store_day["holiday_regional"] + store_day["holiday_local"]) > 0
    ).astype(int)

    holiday_means = {
        "ordinary day": float(store_day.loc[store_day["any_holiday"] == 0, "sales"].mean()),
        "any matched holiday": float(store_day.loc[store_day["any_holiday"] == 1, "sales"].mean()),
        "national holiday": float(store_day.loc[store_day["holiday_national"] == 1, "sales"].mean()),
        "regional holiday": float(store_day.loc[store_day["holiday_regional"] == 1, "sales"].mean()),
        "local holiday": float(store_day.loc[store_day["holiday_local"] == 1, "sales"].mean()),
        "event": float(store_day.loc[store_day["is_event"] == 1, "sales"].mean()),
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(holiday_means), list(holiday_means.values()), color="#3b6ea5")
    ax.set_ylabel("mean store-day sales")
    ax.set_title("Sales on holidays versus ordinary days")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, fig_dir / "05_holidays.png")

    train = train.copy()
    train["promo"] = np.where(train["onpromotion"] > 0, "on promotion", "not on promotion")
    promo_overall = train.groupby("promo")["sales"].mean()
    top_families = family.sort_values(ascending=False).head(8).index
    promo_family = (
        train.loc[train["family"].isin(top_families)]
        .groupby(["family", "promo"])["sales"]
        .mean()
        .unstack()
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    promo_family.plot(kind="barh", ax=ax, color=["#9aa7b5", "#c45c26"])
    ax.set_xlabel("mean daily sales")
    ax.set_title("Promotion and average sales, largest families")
    _save(fig, fig_dir / "06_promotions.png")

    oil_daily = oil.dropna().set_index("date")["dcoilwtico"]
    weekly_sales = daily.set_index("date")["sales"].resample("W").sum()
    weekly_oil = oil_daily.resample("W").mean()
    both = pd.concat([weekly_sales.rename("sales"), weekly_oil.rename("oil")], axis=1).dropna()
    oil_corr = float(both["sales"].corr(both["oil"]))
    oil_diff_corr = float(both["sales"].diff().corr(both["oil"].diff()))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(both["oil"], both["sales"], s=12, alpha=0.6, color="#3b6ea5")
    ax.set_xlabel("weekly mean oil price")
    ax.set_ylabel("weekly total sales")
    ax.set_title(f"Oil price and sales (correlation {oil_corr:.2f})")
    _save(fig, fig_dir / "07_oil.png")

    store_day["day"] = store_day["date"].dt.day
    by_dom = store_day.groupby("day")["sales"].mean()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(by_dom.index, by_dom.values, marker="o", color="#1f4e79")
    ax.axvline(15, color="#c45c26", ls="--", lw=1, label="15th (payday)")
    ax.set_xlabel("day of month")
    ax.set_ylabel("mean store-day sales")
    ax.set_title("Sales across the month")
    ax.legend()
    _save(fig, fig_dir / "08_payday.png")

    quake_rows = holidays.loc[holidays["description"].str.contains("Terremoto", case=False)]
    quake_date = pd.Timestamp(quake_rows["date"].min()) if len(quake_rows) else pd.Timestamp("2016-04-16")
    before = daily.loc[
        (daily["date"] >= quake_date - pd.Timedelta(days=14))
        & (daily["date"] < quake_date),
        "sales",
    ].mean()
    after = daily.loc[
        (daily["date"] >= quake_date) & (daily["date"] < quake_date + pd.Timedelta(days=14)),
        "sales",
    ].mean()

    tx = transactions.merge(store_day[["date", "store_nbr", "sales"]], on=["date", "store_nbr"], how="inner")
    tx_corr = float(tx["transactions"].corr(tx["sales"]))

    total_sales = float(train["sales"].sum())
    top5_share = float(family.sort_values(ascending=False).head(5).sum() / family.sum())
    weekend = float(dow_mean.loc[["Saturday", "Sunday"]].mean())
    weekday = float(dow_mean.loc[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]].mean())
    dom = store_day["date"].dt.day
    dim = store_day["date"].dt.days_in_month
    is_payday = (dom == 15) | (dom == dim)
    payday_mean = float(store_day.loc[is_payday, "sales"].mean())
    other_mean = float(store_day.loc[~is_payday, "sales"].mean())
    day_after = (dom == 16) | (dom == 1)
    day_after_mean = float(store_day.loc[day_after, "sales"].mean())

    stats = {
        "total_sales": total_sales,
        "top5_family_share": top5_share,
        "busiest_dow": str(dow_mean.idxmax()),
        "quietest_dow": str(dow_mean.idxmin()),
        "dow_means": {k: float(v) for k, v in dow_mean.items()},
        "weekend_vs_weekday_ratio": weekend / weekday,
        "peak_month": int(month_mean.idxmax()),
        "quiet_month": int(month_mean.idxmin()),
        "month_means": {int(k): float(v) for k, v in month_mean.items()},
        "store_type_mean_sales": {str(k): float(v) for k, v in by_type.items()},
        "holiday_means": holiday_means,
        "promo_means": {str(k): float(v) for k, v in promo_overall.items()},
        "oil_weekly_corr": oil_corr,
        "oil_weekly_diff_corr": oil_diff_corr,
        "payday_mean": payday_mean,
        "day_after_payday_mean": day_after_mean,
        "other_day_mean": other_mean,
        "earthquake_date": str(quake_date.date()),
        "sales_14d_before_earthquake": float(before),
        "sales_14d_after_earthquake": float(after),
        "transactions_sales_corr": tx_corr,
        "day_of_month_means": {int(k): float(v) for k, v in by_dom.items()},
    }
    return stats
