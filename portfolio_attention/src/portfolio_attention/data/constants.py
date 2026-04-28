"""Data-domain constants for scenario panel loading."""

from __future__ import annotations

import re

BASE_REQUIRED_COLUMNS = [
    "stock_id",
    "t",
    "characteristic_1",
    "characteristic_2",
    "characteristic_3",
    "MKT",
    "SMB",
    "HML",
    "price",
]
OPTIONAL_RETURN_COLUMN = "return"
STOCK_FEATURE_COLUMNS = [
    "characteristic_1",
    "characteristic_2",
    "characteristic_3",
    "price",
]
PRICE_FEATURE_INDEX = STOCK_FEATURE_COLUMNS.index("price")
VALID_PRICE_NORMALIZATION_MODES = ("none", "relative_to_anchor")
RELATIVE_TO_ANCHOR_PRICE_NORMALIZATION_MODE = "relative_to_anchor"
ANCHOR_PRICE_EPSILON = 1e-12
MARKET_FEATURE_COLUMNS = ["MKT", "SMB", "HML"]
NUMERIC_COLUMNS = STOCK_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS + [OPTIONAL_RETURN_COLUMN]
LOADABLE_COLUMNS = BASE_REQUIRED_COLUMNS + [OPTIONAL_RETURN_COLUMN]
SCENARIO_FILE_PATTERN = re.compile(
    r"^(?P<state>.+?)_(?P<n>\d+)_(?P<t>\d+)_PL_(?P<scenario_index>\d+)\.parquet$"
)
