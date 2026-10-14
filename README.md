# AI-powered retail demand forecasting (Favorita)

A university portfolio project. It forecasts daily store-family sales for Corporación Favorita with an LSTM, compares that model with simple and statistical baselines, and runs the forecasts through a small inventory simulation.

The data are the Kaggle competition [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). `holidays_events.csv` is included in this repository. The other files are downloaded by the script below from a public Hugging Face copy of the same CSVs, because the Kaggle API needs a personal token. Point `DATA_DIR` at your own folder if you already have the files.

## Layout

```
config.py                 paths, seeds, and modelling choices
app.py                    Streamlit dashboard
src/data.py               load, profile, oil fill, holiday flags, panel
src/features.py           calendar, lags, scaling
src/models.py             baselines, gradient boosting, LSTM
src/evaluate.py           MAE, RMSE, MAPE, sMAPE, RMSLE, plots
src/inventory.py          safety stock, reorder point, simulation
src/plots.py              exploratory charts
scripts/download_data.py
scripts/run_eda.py        data checks and the written findings
scripts/run_pipeline.py
artifacts/                saved forecasts, metrics, figures, model
```

## Setup

Python 3.10 or newer. These commands were run with Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_pipeline.py
streamlit run app.py
```

`DATA_DIR` defaults to `data/raw`. Override it with `DATA_DIR=/path/to/csvs python scripts/run_pipeline.py`.

Exploratory checks are a plain Python script. It prints the holiday file, the data profile, and the findings, and it saves PNG charts:

```bash
python scripts/run_eda.py
```

The pipeline takes roughly ten minutes on a CPU. It writes `artifacts/`, including `models/lstm.keras`. The dashboard only reads those files, so it does not import TensorFlow.

## What was modelled

The raw training grid is 54 stores by 33 families (about 3.0 million rows, 1 January 2013 to 15 August 2017). The models use the three largest stores (44, 45, and 47, all type A in Quito) and the five largest families (`GROCERY I`, `BEVERAGES`, `PRODUCE`, `CLEANING`, `DAIRY`). Those families are about 79% of unit sales. Exploratory charts still use the full panel.

The split is chronological. Validation is 24 May 2017 to 18 July 2017 (56 days). The test window is 19 July 2017 to 15 August 2017 (28 days). The forecast horizon is 14 days, which matches a 7-day lead time plus a 7-day review. Random k-fold is not used.

## Result in one paragraph

On the test window, a histogram gradient boosting model fit to lag features has the lowest RMSLE (0.130). The LSTM is second (0.153) and beats seasonal naive (0.184), Holt-Winters (0.158), and SARIMA (0.155). The LSTM is the best model only for `CLEANING`. On `GROCERY I` it is worse than seasonal naive. The inventory simulation was actually run: under the stated cost index the LSTM policy is slightly cheaper than seasonal naive and slightly more expensive than gradient boosting, with a lower fill rate than either. Details, tables, and the assumptions are in [REPORT.md](REPORT.md).

## How to explain this project in an interview

**The decision.** Grocery ordering loses money on both sides: leftover stock and empty shelves. The Kaggle series is daily sales by store and product family, which is already the grain of a replenishment forecast. I did not model all 1,782 series. I kept the three largest Quito stores and the five largest families so that an LSTM, a SARIMA, and a walk-forward inventory simulation could all be fit on a laptop and compared on the same dates.

**The leakage choices.** A rolling mean that includes today would score the model on the answer. Lags are `shift(7/14/28)`, and rolling stats are computed on `shift(1)`. The scaler is fit on dates before 24 May 2017 only. Future oil prices are replaced with the last known price. Promotions on the horizon are kept, because a retailer sets the promotion plan before the order. Transactions are left out: they are recorded on the same day as sales. Christmas is missing from the file; I put those days back as zero so that a 7-day lag stays on the same weekday. A transferred holiday is dropped on its original date. The 9 October 2012 Guayaquil independence day is the check: it is not a holiday, and 12 October, the transfer date, is.

**Why not random k-fold.** Adjacent days are highly dependent, and the series has a trend. A random fold would train on August and test on May. The score would look better than any forecast you could have issued at the time.

**What won.** Gradient boosting on the lag features won on RMSLE, MAE, and MAPE. The LSTM, a single 64-unit layer, 28-day window, and 14-day output, beat the seasonal naive and the two statsmodels baselines overall, and it was best on `CLEANING`. It lost on `GROCERY I`, which is the noisiest large series. The residual plot shows the miss: Saturday forecasts are too high and Thursday forecasts are too low, so the multi-step head is not locking onto the weekday cycle as tightly as a model that is handed `lag_7` directly. The training curve does not show a large overfit. Train and validation loss fall together and finish close. The test RMSLE (0.153) is still worse than the validation RMSLE (0.125), so the July-August window was a bit harder than the validation window.

**The inventory part.** This is a simulation, not an optimizer. Safety stock is `z * sigma_error * sqrt(7)` at a 95% service level, with sigma taken from that model's own validation errors. Every 7 days the policy orders up to forecast demand over 14 days plus safety stock. The simulation starts each policy at its own order-up-to level. Recorded sales are treated as demand, which is wrong if the stores were already stocking out: the dataset has no inventory, so those stockouts are censored. Under a holding cost of 0.02 per unit-day and a stockout cost of 0.50 per unit, the LSTM's lower inventory saves about 4% versus seasonal naive on the summed cost index, and the fill rate falls from 99.9% to 98.7%. If the stockout penalty were about 0.90 instead of 0.50, that saving would disappear. I would not tell a buyer that the LSTM "optimizes inventory."

**What I would do next.** Quantile or conformal intervals, so safety stock comes from a predictive distribution rather than a normal formula. A per-family model for `GROCERY I`. Shelf life for produce. More than three stores, once the weekday residual is understood.

## Honest claims checklist

| Claim | Can you say it? |
| --- | --- |
| Exploratory analysis of the full Favorita training panel, including missing Christmas days | Yes |
| Holiday flags respect locale and treat a transferred date as an ordinary day | Yes, and there is an assertion for the Guayaquil example |
| Lag and rolling features are built without using the target day | Yes |
| Scalers are fit on the training period only | Yes |
| Evaluation is chronological, with a rolling-origin check | Yes |
| The LSTM beats seasonal naive, Holt-Winters, and SARIMA on test RMSLE for this 15-series subset | Yes |
| The LSTM is the best model in the comparison | No. Gradient boosting is better overall and on four of the five families |
| The project reports a Kaggle leaderboard score | No. The official test file has no sales labels. The score is an internal holdout |
| Inventory was simulated and costed under stated assumptions | Yes |
| The project optimizes inventory or finds an optimal policy | No |
| Promotions or oil prices were shown to cause sales | No. The associations are descriptive, and the oil correlation is a shared trend |
| Sales are the same thing as demand | No. Historical stockouts are censored |
