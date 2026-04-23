"""Build contemporaneously aligned returns and prices."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from toy_ff_generator.characteristics import FIRM_CHARACTERISTIC_COLUMNS


def build_panel(
    firm_characteristics_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    epsilon_df: pd.DataFrame,
    factor_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge component outputs into a long panel with same-period inputs."""

    missing_columns = [
        column_name
        for column_name in FIRM_CHARACTERISTIC_COLUMNS
        if column_name not in firm_characteristics_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "firm_characteristics_df is missing required columns "
            f"{missing_columns}. Expected {FIRM_CHARACTERISTIC_COLUMNS}."
        )

    panel_df = firm_characteristics_df.merge(beta_df, on=["stock_id", "t"], how="inner")
    panel_df = panel_df.merge(alpha_df, on="stock_id", how="inner")
    panel_df = panel_df.merge(epsilon_df, on=["stock_id", "t"], how="inner")
    panel_df = panel_df.merge(factor_df, on="t", how="inner")

    return panel_df[
        [
            "stock_id",
            "t",
            "state",
            *FIRM_CHARACTERISTIC_COLUMNS,
            "alpha",
            "beta_mkt",
            "beta_smb",
            "beta_hml",
            "MKT",
            "SMB",
            "HML",
            "epsilon",
        ]
    ].copy()


def compute_raw_returns(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute contemporaneous raw returns `r_{i,t}` from the merged panel."""

    result_df = panel_df.copy()
    result_df["raw_return"] = (
        result_df["alpha"]
        + result_df["beta_mkt"] * result_df["MKT"]
        + result_df["beta_smb"] * result_df["SMB"]
        + result_df["beta_hml"] * result_df["HML"]
        + result_df["epsilon"]
    )
    return result_df


def clip_returns(panel_df: pd.DataFrame, limit_down: float, limit_up: float) -> pd.DataFrame:
    """Clip raw returns into the observable `return` series."""

    result_df = panel_df.copy()
    result_df["return"] = result_df["raw_return"].clip(lower=limit_down, upper=limit_up)
    return result_df


def generate_prices(
    panel_df: pd.DataFrame,
    initial_prices: Mapping[str, float],
    time_columns: Sequence[str],
) -> pd.DataFrame:
    """Generate price paths from clipped returns."""

    result_df = panel_df.copy()
    time_order = {time_label: idx for idx, time_label in enumerate(time_columns)}
    result_df["_time_order"] = result_df["t"].map(time_order)
    result_df = result_df.sort_values(["stock_id", "_time_order"]).copy()

    stock_order = result_df["stock_id"].drop_duplicates().tolist()
    stock_count = len(stock_order)
    time_count = len(time_columns)
    is_dense_panel = len(result_df) == stock_count * time_count

    if is_dense_panel:
        initial_price_vector = np.asarray(
            [float(initial_prices[stock_id]) for stock_id in stock_order],
            dtype=float,
        )[:, np.newaxis]
        returns_matrix = result_df["return"].to_numpy(dtype=float).reshape(stock_count, time_count)
        price_matrix = initial_price_vector * np.cumprod(1.0 + returns_matrix, axis=1)
        result_df["price"] = price_matrix.reshape(stock_count * time_count)
    else:
        prices: list[float] = []
        for stock_id, stock_panel in result_df.groupby("stock_id", sort=False):
            current_price = float(initial_prices[stock_id])
            for clipped_return in stock_panel["return"].tolist():
                current_price = current_price * (1.0 + float(clipped_return))
                prices.append(current_price)
        result_df["price"] = prices

    return result_df.drop(columns="_time_order")
