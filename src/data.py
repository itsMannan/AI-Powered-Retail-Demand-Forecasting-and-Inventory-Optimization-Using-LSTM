"""Load the Favorita files and build a daily store-family panel."""

import io
from pathlib import Path

import numpy as np
import pandas as pd

from config import EXTEND_DAYS


def _as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def load_raw(data_dir):
    """Read every competition file. DATA_DIR can point somewhere else."""
    data_dir = Path(data_dir)
    required = [
        "train.csv",
        "test.csv",
        "stores.csv",
        "oil.csv",
        "transactions.csv",
        "holidays_events.csv",
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {data_dir}. Run scripts/download_data.py "
            "or set DATA_DIR to the folder that holds the Kaggle files."
        )

    train = pd.read_csv(
        data_dir / "train.csv",
        dtype={"store_nbr": "int16", "sales": "float32", "onpromotion": "int16"},
        parse_dates=["date"],
    )
    test = pd.read_csv(
        data_dir / "test.csv",
        dtype={"store_nbr": "int16", "onpromotion": "int16"},
        parse_dates=["date"],
    )
    stores = pd.read_csv(data_dir / "stores.csv", dtype={"store_nbr": "int16"})
    oil = pd.read_csv(data_dir / "oil.csv", parse_dates=["date"])
    transactions = pd.read_csv(
        data_dir / "transactions.csv",
        dtype={"store_nbr": "int16", "transactions": "int32"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])

    train["family"] = train["family"].astype(str)
    test["family"] = test["family"].astype(str)
    stores["store_nbr"] = stores["store_nbr"].astype(int)
    stores = stores.rename(columns={"type": "store_type"})
    holidays["transferred"] = _as_bool(holidays["transferred"])
    return {
        "train": train,
        "test": test,
        "stores": stores,
        "oil": oil,
        "transactions": transactions,
        "holidays": holidays,
    }


def frame_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def missing_calendar_days(dates):
    dates = pd.to_datetime(pd.Series(dates).drop_duplicates()).sort_values()
    full = pd.date_range(dates.min(), dates.max(), freq="D")
    missing = full.difference(pd.DatetimeIndex(dates))
    return list(missing)


def profile_raw(raw):
    """Shapes, dtypes, missing values, duplicates, and date coverage."""
    profile = {}
    for name, df in raw.items():
        profile[name] = {
            "shape": list(df.shape),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing": {col: int(n) for col, n in df.isna().sum().items()},
            "duplicate_rows": int(df.duplicated().sum()),
            "head": df.head(5).astype(str).to_dict(orient="records"),
            "info": frame_info(df),
        }

    train = raw["train"]
    key = ["date", "store_nbr", "family"]
    profile["train"]["duplicate_keys"] = int(train.duplicated(key).sum())
    profile["train"]["date_min"] = str(train["date"].min().date())
    profile["train"]["date_max"] = str(train["date"].max().date())
    profile["train"]["n_stores"] = int(train["store_nbr"].nunique())
    profile["train"]["n_families"] = int(train["family"].nunique())
    profile["train"]["n_dates"] = int(train["date"].nunique())
    missing_days = missing_calendar_days(train["date"])
    profile["train"]["missing_days"] = [str(d.date()) for d in missing_days]

    oil = raw["oil"]
    profile["oil"]["date_min"] = str(oil["date"].min().date())
    profile["oil"]["date_max"] = str(oil["date"].max().date())
    profile["oil"]["null_prices"] = int(oil["dcoilwtico"].isna().sum())
    oil_span = pd.date_range(oil["date"].min(), oil["date"].max(), freq="D")
    profile["oil"]["dates_absent_from_file"] = int(len(oil_span) - oil["date"].nunique())

    test = raw["test"]
    profile["test"]["date_min"] = str(test["date"].min().date())
    profile["test"]["date_max"] = str(test["date"].max().date())
    profile["holidays"]["type_counts"] = raw["holidays"]["type"].value_counts().to_dict()
    profile["holidays"]["locale_counts"] = raw["holidays"]["locale"].value_counts().to_dict()
    profile["holidays"]["transferred_true"] = int(raw["holidays"]["transferred"].sum())
    return profile


def prepare_oil(oil, all_dates):
    """Fill missing oil prices with the last quoted price.

    WTI is not quoted on weekends and some holidays, and the file also leaves
    those prices blank. The price a store would have known on a missing day is
    the previous quote, so forward-fill is the right default. Back-fill is only
    there for the short gap before the first quote (1 Jan 2013 is blank).
    """
    series = oil.set_index("date")["dcoilwtico"].sort_index()
    series = series.reindex(pd.DatetimeIndex(all_dates)).ffill().bfill()
    if series.isna().any():
        raise ValueError("Oil prices still contain nulls after forward/back fill.")
    return series.rename("oil")


def build_holiday_flags(holidays, stores):
    """National, regional, and local holiday flags, with transfers removed.

    A row with transferred=True is the original date of a holiday that the
    government moved. That date is an ordinary day. The celebration itself is
    the matching row with type 'Transfer'. Work Day rows are makeup working
    days, so they are not holidays. Events (for example the 2016 earthquake)
    are kept as a separate flag.
    """
    h = holidays.copy()
    h = h.loc[~h["transferred"]].copy()
    h = h.loc[h["type"] != "Work Day"].copy()
    h["is_event_row"] = (h["type"] == "Event").astype(int)
    h["is_holiday_row"] = (h["type"] != "Event").astype(int)

    pieces = []

    def _add(frame, flag_name, store_ids):
        if frame.empty:
            return
        dates = frame.loc[frame["keep"] == 1, ["date"]].drop_duplicates()
        if dates.empty:
            return
        crossed = dates.merge(pd.DataFrame({"store_nbr": store_ids}), how="cross")
        crossed[flag_name] = 1
        pieces.append(crossed)

    stores = stores.copy()
    stores["store_nbr"] = stores["store_nbr"].astype(int)
    all_ids = stores["store_nbr"].tolist()

    national = h.loc[h["locale"] == "National"].copy()
    national["keep"] = national["is_holiday_row"]
    _add(national, "holiday_national", all_ids)
    national["keep"] = national["is_event_row"]
    _add(national, "is_event", all_ids)

    regional = h.loc[h["locale"] == "Regional"]
    for state, group in regional.groupby("locale_name"):
        ids = stores.loc[stores["state"] == state, "store_nbr"].tolist()
        group = group.copy()
        group["keep"] = group["is_holiday_row"]
        _add(group, "holiday_regional", ids)
        group["keep"] = group["is_event_row"]
        _add(group, "is_event", ids)

    local = h.loc[h["locale"] == "Local"]
    for city, group in local.groupby("locale_name"):
        ids = stores.loc[stores["city"] == city, "store_nbr"].tolist()
        group = group.copy()
        group["keep"] = group["is_holiday_row"]
        _add(group, "holiday_local", ids)
        group["keep"] = group["is_event_row"]
        _add(group, "is_event", ids)

    flag_cols = ["holiday_national", "holiday_regional", "holiday_local", "is_event"]
    if not pieces:
        return pd.DataFrame(columns=["date", "store_nbr"] + flag_cols)

    flags = pd.concat(pieces, ignore_index=True)
    for col in flag_cols:
        if col not in flags.columns:
            flags[col] = 0
    flags = flags.groupby(["date", "store_nbr"], as_index=False)[flag_cols].max()
    return flags


def assert_holiday_rules(flags, stores):
    """Check the transferred-holiday example from the file description."""
    quito = int(stores.loc[stores["city"] == "Quito", "store_nbr"].iloc[0])
    cuenca = int(stores.loc[stores["city"] == "Cuenca", "store_nbr"].iloc[0])
    cotopaxi = int(stores.loc[stores["state"] == "Cotopaxi", "store_nbr"].iloc[0])
    pichincha = int(stores.loc[stores["state"] == "Pichincha", "store_nbr"].iloc[0])
    indexed = flags.set_index(["date", "store_nbr"])

    def flag(date, store, col):
        key = (pd.Timestamp(date), store)
        if key not in indexed.index:
            return 0
        return int(indexed.loc[key, col])

    # Independencia de Guayaquil was moved from 9 Oct 2012 to 12 Oct 2012.
    assert flag("2012-10-09", quito, "holiday_national") == 0
    assert flag("2012-10-12", quito, "holiday_national") == 1
    # Regional holidays apply only in that province.
    assert flag("2012-04-01", cotopaxi, "holiday_regional") == 1
    assert flag("2012-04-01", pichincha, "holiday_regional") == 0
    # Local holidays apply only in that city.
    assert flag("2012-04-12", cuenca, "holiday_local") == 1
    assert flag("2012-04-12", quito, "holiday_local") == 0


def choose_subset(train, n_stores, n_families):
    family_sales = train.groupby("family")["sales"].sum().sort_values(ascending=False)
    store_sales = train.groupby("store_nbr")["sales"].sum().sort_values(ascending=False)
    families = family_sales.head(n_families).index.tolist()
    stores = [int(s) for s in store_sales.head(n_stores).index.tolist()]
    return stores, families, family_sales, store_sales


def _trim_leading_zeros(g, min_run=30):
    """Drop a long opening run of exact zeros.

    Some families (notably PRODUCE) are recorded as zero for a long stretch at
    the start of the sample and then suddenly become large. That pattern is a
    change in what was recorded, and it should not be treated as real demand.
    Internal zeros are kept.
    """
    sales = g["sales"].to_numpy()
    i = 0
    while i < len(sales) and sales[i] == 0:
        i += 1
    if i >= min_run:
        return g.iloc[i:].copy()
    return g


def build_panel(raw, store_ids, families):
    """Daily panel for the chosen stores and families, with oil and holidays."""
    train = raw["train"]
    mask = train["store_nbr"].isin(store_ids) & train["family"].isin(families)
    panel = train.loc[mask, ["date", "store_nbr", "family", "sales", "onpromotion"]].copy()
    panel["store_nbr"] = panel["store_nbr"].astype(int)
    panel["in_sample"] = True

    stores = raw["stores"]
    panel = panel.merge(
        stores[["store_nbr", "city", "state", "store_type", "cluster"]],
        on="store_nbr",
        how="left",
    )

    # Christmas is missing for every series because the stores were closed.
    # Put those days back as zero sales so that a 7-day lag stays on the same weekday.
    pieces = []
    meta_cols = ["city", "state", "store_type", "cluster"]
    for (store, family), g in panel.groupby(["store_nbr", "family"], sort=False):
        g = g.sort_values("date")
        meta = g.iloc[0]
        observed = g.set_index("date")
        full_idx = pd.date_range(observed.index.min(), observed.index.max(), freq="D")
        out = observed.reindex(full_idx)
        out["store_nbr"] = store
        out["family"] = family
        for col in meta_cols:
            out[col] = meta[col]
        # Christmas was missing because the stores were closed. Filling those
        # days with zero sales keeps them in the sample.
        out["in_sample"] = True
        out["sales"] = out["sales"].fillna(0.0)
        out["onpromotion"] = out["onpromotion"].fillna(0).astype(int)
        out.index.name = "date"
        pieces.append(out.reset_index())

    panel = pd.concat(pieces, ignore_index=True)
    trimmed = []
    for _, g in panel.groupby(["store_nbr", "family"], sort=False):
        trimmed.append(_trim_leading_zeros(g))
    panel = pd.concat(trimmed, ignore_index=True)

    sample_end = panel.loc[panel["in_sample"], "date"].max()
    extra_idx = pd.date_range(sample_end + pd.Timedelta(days=1), periods=EXTEND_DAYS, freq="D")
    extra_rows = []
    meta = (
        panel[["store_nbr", "family", "city", "state", "store_type", "cluster"]]
        .drop_duplicates()
    )
    for row in meta.itertuples(index=False):
        for day in extra_idx:
            extra_rows.append(
                {
                    "date": day,
                    "store_nbr": int(row.store_nbr),
                    "family": row.family,
                    "sales": np.nan,
                    "onpromotion": 0,
                    "city": row.city,
                    "state": row.state,
                    "store_type": row.store_type,
                    "cluster": row.cluster,
                    "in_sample": False,
                }
            )
    if extra_rows:
        panel = pd.concat([panel, pd.DataFrame(extra_rows)], ignore_index=True)

    all_dates = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    oil_df = prepare_oil(raw["oil"], all_dates).rename_axis("date").reset_index()
    panel = panel.merge(oil_df, on="date", how="left")

    flags = build_holiday_flags(raw["holidays"], stores)
    assert_holiday_rules(flags, stores)
    panel = panel.merge(flags, on=["date", "store_nbr"], how="left")
    for col in ["holiday_national", "holiday_regional", "holiday_local", "is_event"]:
        panel[col] = panel[col].fillna(0).astype(int)

    # Closed on Christmas. This is a known calendar fact, not something inferred
    # from the sales value, so it is safe to use as a feature.
    panel["is_closed"] = ((panel["date"].dt.month == 12) & (panel["date"].dt.day == 25)).astype(int)
    panel = panel.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    return panel


def split_dates(panel, val_days, test_days):
    dates = np.sort(panel.loc[panel["in_sample"], "date"].unique())
    if len(dates) <= val_days + test_days:
        raise ValueError("Not enough dates for the requested validation and test windows.")
    test_start = pd.Timestamp(dates[-test_days])
    val_start = pd.Timestamp(dates[-(val_days + test_days)])
    test_end = pd.Timestamp(dates[-1])
    val_end = pd.Timestamp(dates[-(test_days + 1)])
    return {
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
    }
