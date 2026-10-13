"""Paths and modelling choices shared by the pipeline and the dashboard."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data" / "raw"))
ARTIFACT_DIR = ROOT / "artifacts"

SEED = 42

# Top stores and families by total unit sales. The full panel is 54 x 33 series,
# which is more than this project needs in order to compare forecasting methods.
N_STORES = 3
N_FAMILIES = 5

# Chronological holdout. Validation is long enough to estimate forecast-error
# sigma for safety stock. The test window is two forecast horizons.
VAL_DAYS = 56
TEST_DAYS = 28
HORIZON = 14
DEFAULT_WINDOW = 28

# Inventory policy. Horizon = lead time + review period, so one LSTM forecast
# covers the protection period of a weekly review.
LEAD_TIME = 7
REVIEW_PERIOD = 7
SERVICE_LEVEL = 0.95
# These are not Favorita's costs. They are a stated cost index per sales unit.
HOLDING_COST = 0.02
STOCKOUT_COST = 0.50

# Weekly statistical models are fit on the last year of history before the origin.
STAT_HISTORY_DAYS = 364

LSTM_EPOCHS = 20
LSTM_BATCH = 64
LSTM_PATIENCE = 4

# Extra calendar days after the sample so the last replenishment review can
# look a full horizon ahead. Those days are not scored.
EXTEND_DAYS = 14

LSTM_SEARCH = [
    {"units": 32, "layers": 1, "lr": 1e-3, "window": 28, "dropout": 0.2},
    {"units": 64, "layers": 1, "lr": 1e-3, "window": 28, "dropout": 0.2},
    {"units": 32, "layers": 2, "lr": 1e-3, "window": 28, "dropout": 0.2},
    {"units": 32, "layers": 1, "lr": 5e-4, "window": 28, "dropout": 0.2},
    {"units": 32, "layers": 1, "lr": 1e-3, "window": 56, "dropout": 0.2},
]
