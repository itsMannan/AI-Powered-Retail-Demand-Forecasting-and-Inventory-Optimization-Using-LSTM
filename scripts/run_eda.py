"""Print the data checks and findings, and save the exploratory PNG charts.

This is the exploration step. Modelling is in scripts/run_pipeline.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import ARTIFACT_DIR, DATA_DIR, N_FAMILIES, N_STORES
from src.data import choose_subset, frame_info, load_raw, profile_raw
from src.plots import run_eda


def main():
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)

    print("DATA_DIR =", DATA_DIR)
    raw = load_raw(DATA_DIR)

    holidays = raw["holidays"]
    print("\nHoliday file, first rows")
    print(holidays.head().to_string(index=False))
    print()
    print(frame_info(holidays))
    print("transferred counts")
    print(holidays["transferred"].value_counts().to_string())
    print("type counts")
    print(holidays["type"].value_counts().to_string())

    profile = profile_raw(raw)
    print()
    for name, info in profile.items():
        if not isinstance(info, dict) or "shape" not in info:
            continue
        print(
            f"{name:15} shape={info['shape']} "
            f"duplicate_rows={info['duplicate_rows']} missing={info['missing']}"
        )
    print("train dates", profile["train"]["date_min"], "->", profile["train"]["date_max"])
    print("missing days", profile["train"]["missing_days"])
    print("duplicate store-family-date keys", profile["train"]["duplicate_keys"])
    print(
        "oil null prices",
        profile["oil"]["null_prices"],
        "dates absent from the oil file",
        profile["oil"]["dates_absent_from_file"],
    )
    print(
        "official test dates (no sales labels)",
        profile["test"]["date_min"],
        "->",
        profile["test"]["date_max"],
    )

    fig_dir = ARTIFACT_DIR / "figures"
    stats = run_eda(raw, fig_dir)
    print("\nCharts saved to", fig_dir)

    stores, families, _, _ = choose_subset(raw["train"], N_STORES, N_FAMILIES)
    print(
        f"1. The training panel has {profile['train']['shape'][0]:,} rows, "
        f"{profile['train']['n_stores']} stores and {profile['train']['n_families']} families, "
        f"from {profile['train']['date_min']} to {profile['train']['date_max']}. "
        f"The only missing dates are {', '.join(profile['train']['missing_days'])}. "
        "Those are Christmas days, when the stores were closed, so the dates are absent rather than stored as zero."
    )
    print(
        f"2. Sales are highly concentrated. The five largest families account for "
        f"{stats['top5_family_share']:.0%} of unit sales. The busiest weekday in the company total is "
        f"{stats['busiest_dow']} and the quietest is {stats['quietest_dow']}. "
        f"Weekend days average {stats['weekend_vs_weekday_ratio']:.2f} times a weekday. "
        f"The strongest month is month {stats['peak_month']} and the weakest is month {stats['quiet_month']}."
    )
    print(
        "3. Store type and cluster are not just labels. Mean total sales per store by type: "
        + ", ".join(f"{k}={v:,.0f}" for k, v in stats["store_type_mean_sales"].items())
        + "."
    )
    h = stats["holiday_means"]
    print(
        f"4. A matched holiday is not automatically a high-sales day. "
        f"Mean store-day sales are {h['ordinary day']:,.0f} on ordinary days and "
        f"{h['national holiday']:,.0f} on national holidays "
        f"({h['national holiday'] / h['ordinary day'] - 1:+.1%} versus an ordinary day). "
        "Christmas itself is missing from the file, so this comparison does not include the closure."
    )
    p = stats["promo_means"]
    print(
        f"5. Promotions line up with higher sales, but that is not a clean causal effect: "
        f"promoted rows average {p.get('on promotion', float('nan')):,.1f} units and "
        f"other rows average {p.get('not on promotion', float('nan')):,.1f}. "
        "Popular families are promoted more often."
    )
    print(
        f"6. Weekly oil prices and weekly total sales have a correlation of {stats['oil_weekly_corr']:.2f}. "
        f"Week-to-week changes are almost uncorrelated ({stats['oil_weekly_diff_corr']:.2f}), "
        "so the level correlation is mostly the shared multi-year trend: oil fell while recorded sales grew. "
        f"Store-day sales average {stats['payday_mean']:,.0f} on the 15th or the last day of the month, "
        f"{stats['day_after_payday_mean']:,.0f} on the next day, and {stats['other_day_mean']:,.0f} on other days. "
        "The calendar plot shows the clearer spike on the 1st and 2nd, after wages paid at month end, rather than on the 15th itself."
    )
    print(
        f"7. The earthquake on {stats['earthquake_date']} is a real shock in this sample. "
        f"Mean daily total sales were {stats['sales_14d_before_earthquake']:,.0f} in the 14 days before and "
        f"{stats['sales_14d_after_earthquake']:,.0f} in the 14 days after. "
        "A weekly seasonal model has no way to represent that break unless it is given an event flag."
    )
    print(
        f"Modelling subset: stores {stores} and families {families}. "
        "These are the largest series by unit sales. The full grid is 54 stores by 33 families, "
        "which is more than a CPU LSTM needs in order to compare methods. "
        f"Transactions correlate {stats['transactions_sales_corr']:.2f} with store-day sales, "
        "but they are recorded on the same day as sales, so they are not used as a predictor."
    )


if __name__ == "__main__":
    main()
