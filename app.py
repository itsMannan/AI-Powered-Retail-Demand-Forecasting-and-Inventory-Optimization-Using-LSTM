"""Streamlit dashboard for the saved Favorita forecasts.

The app reads artifacts produced by scripts/run_pipeline.py. It does not
import TensorFlow, so it can start without reloading the training stack.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


@st.cache_data
def load_inputs():
    meta = json.loads((ART / "metadata.json").read_text())
    panel = pd.read_parquet(ART / "processed" / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    forecasts = pd.read_csv(ART / "forecasts" / "test_forecasts.csv", parse_dates=["date", "origin"])
    overall = pd.read_csv(ART / "metrics" / "overall.csv")
    by_family = pd.read_csv(ART / "metrics" / "by_family.csv")
    inventory = pd.read_csv(ART / "inventory" / "summary.csv")
    daily = pd.read_csv(ART / "inventory" / "daily.csv", parse_dates=["date"])
    pooled = pd.read_csv(ART / "inventory" / "pooled.csv")
    return meta, panel, forecasts, overall, by_family, inventory, daily, pooled


def main():
    st.set_page_config(page_title="Favorita demand forecasts", layout="wide")
    st.title("Favorita demand forecasts")
    st.caption(
        "Store-family sales for a subset of Corporación Favorita. "
        "Forecasts and inventory results are loaded from saved files."
    )

    if not (ART / "metadata.json").exists():
        st.error("No saved results yet. Run `python scripts/run_pipeline.py` first.")
        st.stop()

    meta, panel, forecasts, overall, by_family, inventory, daily, pooled = load_inputs()
    accuracy_origins = set()
    # The scored test window is two non-overlapping horizons, not every review date.
    test_start = pd.Timestamp(meta["bounds"]["test_start"])
    horizon = int(meta["horizon"])
    accuracy_origins.add(test_start)
    accuracy_origins.add(test_start + pd.Timedelta(days=horizon))

    stores = sorted(int(s) for s in panel["store_nbr"].unique())
    families = sorted(panel["family"].unique())

    st.sidebar.header("Filters")
    store = st.sidebar.selectbox("Store", stores)
    family = st.sidebar.selectbox("Product family", families)
    date_min = panel["date"].min().date()
    date_max = panel["date"].max().date()
    date_range = st.sidebar.slider(
        "Date range",
        min_value=date_min,
        max_value=date_max,
        value=(date_min, date_max),
    )
    horizon_choice = st.sidebar.slider("Forecast horizon (days)", 1, horizon, horizon)
    st.sidebar.caption(
        f"Modelled stores {meta['stores']} and families {', '.join(meta['families'])}. "
        f"Test window {meta['bounds']['test_start']} to {meta['bounds']['test_end']}."
    )

    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1])
    history = panel.loc[
        (panel["store_nbr"] == store)
        & (panel["family"] == family)
        & (panel["date"] >= start)
        & (panel["date"] <= end)
    ].sort_values("date")

    tab_history, tab_forecast, tab_metrics, tab_inventory = st.tabs(
        ["Historical sales", "Forecast vs actual", "Model comparison", "Inventory"]
    )

    with tab_history:
        st.subheader(f"Store {store} · {family}")
        if history.empty:
            st.warning("No sales in this date range.")
        else:
            chart = history.copy()
            chart["28-day mean"] = chart["sales"].rolling(28, min_periods=7).mean()
            long = chart.melt(
                id_vars="date",
                value_vars=["sales", "28-day mean"],
                var_name="series",
                value_name="units",
            )
            fig = px.line(long, x="date", y="units", color="series", template="plotly_white")
            fig.update_layout(height=420, legend_title="")
            st.plotly_chart(fig, use_container_width=True)
            promo_share = float((history["onpromotion"] > 0).mean())
            holiday_share = float(
                (
                    history["holiday_national"]
                    + history["holiday_regional"]
                    + history["holiday_local"]
                ).gt(0).mean()
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean daily sales", f"{history['sales'].mean():,.1f}")
            c2.metric("Days on promotion", f"{promo_share:.0%}")
            c3.metric("Days on a matched holiday", f"{holiday_share:.0%}")
        with st.expander("Full-sample exploratory charts"):
            for name in [
                "01_total_sales.png",
                "02_sales_by_family.png",
                "04_dow_month.png",
                "05_holidays.png",
                "06_promotions.png",
                "07_oil.png",
                "08_payday.png",
            ]:
                path = ART / "figures" / name
                if path.exists():
                    st.image(str(path))

    with tab_forecast:
        models = sorted(forecasts["model"].unique())
        default_model = "lstm" if "lstm" in models else models[0]
        model = st.selectbox("Model", models, index=models.index(default_model))
        scored = forecasts.loc[
            (forecasts["store_nbr"] == store)
            & (forecasts["family"] == family)
            & (forecasts["model"] == model)
            & (forecasts["origin"].isin(list(accuracy_origins)))
            & (forecasts["horizon_day"] <= horizon_choice)
            & forecasts["actual"].notna()
        ].sort_values("date")
        scored = scored.loc[(scored["date"] >= start) & (scored["date"] <= end)]
        st.subheader(f"{model} against actual sales")
        if scored.empty:
            st.warning("No scored forecast rows for this filter. The test window is the last 28 days of the sample.")
        else:
            # A blank row between origins stops the line from bridging a gap
            # when the horizon is shorter than the gap between forecast blocks.
            pieces = []
            for _, block in scored.groupby("origin", sort=True):
                pieces.append(block)
                gap = block.iloc[[-1]].copy()
                gap["date"] = gap["date"] + pd.Timedelta(hours=12)
                gap["actual"] = float("nan")
                gap["prediction"] = float("nan")
                pieces.append(gap)
            plot_df = pd.concat(pieces, ignore_index=True).iloc[:-1]
            fig = go.Figure()
            fig.add_scatter(x=plot_df["date"], y=plot_df["actual"], name="actual", line=dict(color="black"), connectgaps=False)
            fig.add_scatter(x=plot_df["date"], y=plot_df["prediction"], name=model, line=dict(color="#c45c26"), connectgaps=False)
            fig.update_layout(template="plotly_white", height=420, yaxis_title="units")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(scored[["date", "origin", "horizon_day", "actual", "prediction"]], hide_index=True)
        loss_path = ART / "figures" / "09_lstm_loss.png"
        if loss_path.exists():
            st.caption(
                f"LSTM config chosen on validation RMSLE: {meta.get('best_lstm')}. "
                "The loss curve is on scaled sales."
            )
            st.image(str(loss_path))

    with tab_metrics:
        st.subheader("Test metrics")
        st.caption(
            "Non-overlapping 14-day forecasts covering the last 28 days. "
            "MAPE ignores days with zero sales. sMAPE and RMSLE do not. Lower is better."
        )
        show = overall.copy()
        for col in ["MAE", "RMSE", "MAPE", "sMAPE", "RMSLE"]:
            if col in show.columns:
                show[col] = show[col].map(lambda v: round(float(v), 3))
        st.dataframe(show, hide_index=True)
        if "RMSLE" in overall.columns:
            ranked = overall.sort_values("RMSLE")
            fig = px.bar(
                ranked,
                x="RMSLE",
                y="model",
                orientation="h",
                template="plotly_white",
                title="Test RMSLE",
            )
            # Lowest RMSLE at the top, so the best model is visible without scrolling.
            fig.update_yaxes(
                categoryorder="array",
                categoryarray=ranked.sort_values("RMSLE", ascending=False)["model"].tolist(),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.subheader(f"By family, with {family} highlighted in the table")
        family_table = by_family.copy()
        for col in ["MAE", "RMSE", "MAPE", "sMAPE", "RMSLE"]:
            if col in family_table.columns:
                family_table[col] = family_table[col].map(lambda v: round(float(v), 3))
        st.dataframe(family_table, hide_index=True)
        one = by_family.loc[by_family["family"] == family]
        if not one.empty and "RMSLE" in one.columns:
            fig = px.bar(one.sort_values("RMSLE"), x="RMSLE", y="model", orientation="h", template="plotly_white")
            fig.update_layout(title=f"RMSLE for {family}")
            st.plotly_chart(fig, use_container_width=True)

    with tab_inventory:
        st.subheader("Safety stock and simulated cost")
        st.caption(
            "Periodic review every 7 days, lead time 7 days, 95% service level. "
            "Costs are an index (holding 0.02 per unit-day, stockout 0.50 per unit), not Favorita's accounts."
        )
        series = inventory.loc[(inventory["store_nbr"] == store) & (inventory["family"] == family)].copy()
        if series.empty:
            st.warning("No inventory result for this series.")
        else:
            nice = series[
                [
                    "model",
                    "sigma_error",
                    "safety_stock",
                    "reorder_point",
                    "order_up_to",
                    "fill_rate",
                    "stockout_rate",
                    "avg_inventory",
                    "total_cost",
                ]
            ].copy()
            for col in nice.columns:
                if col != "model":
                    nice[col] = nice[col].map(lambda v: round(float(v), 3))
            st.dataframe(nice, hide_index=True)
            path = daily.loc[(daily["store_nbr"] == store) & (daily["family"] == family)]
            compare = path.loc[path["model"].isin(["lstm", "seasonal_naive"])]
            if not compare.empty:
                fig = px.line(
                    compare,
                    x="date",
                    y="on_hand",
                    color="model",
                    template="plotly_white",
                    title="On-hand inventory over the test window",
                )
                st.plotly_chart(fig, use_container_width=True)
        st.subheader("Pooled across the modelled series")
        st.caption("Costs and lost sales are summed. Average inventory is the mean across series. Fill rate uses the summed units.")
        st.dataframe(pooled.round(3), hide_index=True)
        if {"model", "total_cost"}.issubset(set(pooled.columns)):
            base = pooled.loc[pooled["model"] == "seasonal_naive", "total_cost"]
            lstm_cost = pooled.loc[pooled["model"] == "lstm", "total_cost"]
            if len(base) and len(lstm_cost):
                delta = float(base.iloc[0] - lstm_cost.iloc[0])
                st.metric(
                    "LSTM cost minus seasonal-naive cost, summed units",
                    f"{-delta:,.1f}",
                    help="Negative means the LSTM policy costs less on this index. Units are not comparable across families.",
                )
        with st.expander("Assumptions and limitations"):
            for line in meta.get("assumptions", []):
                st.write("- " + line)


if __name__ == "__main__":
    main()
