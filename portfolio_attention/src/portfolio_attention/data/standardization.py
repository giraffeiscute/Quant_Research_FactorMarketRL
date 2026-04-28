"""Feature standardization and price-normalization helpers."""

from __future__ import annotations

import numpy as np

from .constants import (
    ANCHOR_PRICE_EPSILON,
    PRICE_FEATURE_INDEX,
    RELATIVE_TO_ANCHOR_PRICE_NORMALIZATION_MODE,
    STOCK_FEATURE_COLUMNS,
    VALID_PRICE_NORMALIZATION_MODES,
)


class Standardizer:
    """Simple ndarray standardizer fit only on train-scenario train-segment rows."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "Standardizer":
        if values.size == 0:
            raise ValueError("Cannot fit a scaler on empty values.")
        self.mean = values.mean(axis=0)
        std = values.std(axis=0)
        self.std = np.where(std < 1e-6, 1.0, std)
        return self

    def set_statistics(self, mean: np.ndarray, std: np.ndarray) -> "Standardizer":
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer must be fit before transform.")
        return (values - self.mean) / self.std


class RunningMoments:
    """Streaming moments helper used to avoid fitting scalers on validation/test rows."""

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim
        self.count = 0
        self.sum = np.zeros((feature_dim,), dtype=np.float64)
        self.sum_sq = np.zeros((feature_dim,), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected values with shape [*, {self.feature_dim}], received {values.shape}."
            )
        self.count += int(values.shape[0])
        self.sum += values.sum(axis=0, dtype=np.float64)
        self.sum_sq += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0:
            raise ValueError("Cannot finalize moments without any observations.")
        mean = self.sum / float(self.count)
        variance = (self.sum_sq / float(self.count)) - np.square(mean)
        variance = np.maximum(variance, 1e-12)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _slice_stock_features_for_context(
    stock_features_raw: np.ndarray,
    *,
    context_feature_start: int,
    context_feature_stop: int,
) -> np.ndarray:
    context_stock_features = np.asarray(
        stock_features_raw[context_feature_start:context_feature_stop],
        dtype=np.float32,
    )
    if context_stock_features.ndim != 3 or context_stock_features.shape[-1] != len(STOCK_FEATURE_COLUMNS):
        raise ValueError(
            "Expected stock feature slice with shape [time, stock, feature]. "
            f"Received {context_stock_features.shape}."
        )
    if context_stock_features.shape[0] <= 0:
        raise ValueError("Context stock feature slice must contain at least one time step.")
    return context_stock_features


def _compute_relative_price_feature(context_stock_features: np.ndarray) -> np.ndarray:
    anchor_prices = context_stock_features[0, :, PRICE_FEATURE_INDEX].astype(np.float64, copy=False)
    if np.any(np.abs(anchor_prices) < ANCHOR_PRICE_EPSILON):
        raise ValueError(
            "Anchor price must be non-zero before relative_to_anchor normalization."
        )
    relative_prices = (
        context_stock_features[..., PRICE_FEATURE_INDEX].astype(np.float64)
        / anchor_prices[np.newaxis, :]
    ) - 1.0
    return relative_prices.astype(np.float32)


def transform_stock_feature_context_array(
    context_stock_features: np.ndarray,
    *,
    price_normalization_mode: str,
) -> np.ndarray:
    if price_normalization_mode not in VALID_PRICE_NORMALIZATION_MODES:
        raise ValueError(
            "Unsupported price_normalization_mode for stock feature transformation: "
            f"{price_normalization_mode!r}."
        )
    context_stock_features = np.asarray(context_stock_features, dtype=np.float32)
    if context_stock_features.ndim != 3 or context_stock_features.shape[-1] != len(STOCK_FEATURE_COLUMNS):
        raise ValueError(
            "Expected context_stock_features with shape [time, stock, feature]. "
            f"Received {context_stock_features.shape}."
        )
    if context_stock_features.shape[0] <= 0:
        raise ValueError("context_stock_features must contain at least one time step.")

    transformed = context_stock_features.copy()
    if price_normalization_mode == RELATIVE_TO_ANCHOR_PRICE_NORMALIZATION_MODE:
        transformed[..., PRICE_FEATURE_INDEX] = _compute_relative_price_feature(context_stock_features)
    return transformed


def transform_stock_features_for_context(
    stock_features_raw: np.ndarray,
    *,
    context_feature_start: int,
    context_feature_stop: int,
    price_normalization_mode: str,
) -> np.ndarray:
    context_stock_features = _slice_stock_features_for_context(
        stock_features_raw,
        context_feature_start=context_feature_start,
        context_feature_stop=context_feature_stop,
    )
    return transform_stock_feature_context_array(
        context_stock_features,
        price_normalization_mode=price_normalization_mode,
    )


def scale_stock_feature_context_array(
    context_stock_features: np.ndarray,
    *,
    price_normalization_mode: str,
    stock_mean: np.ndarray,
    stock_std: np.ndarray,
) -> np.ndarray:
    transformed = transform_stock_feature_context_array(
        context_stock_features,
        price_normalization_mode=price_normalization_mode,
    )
    return ((transformed - stock_mean.reshape(1, 1, -1)) / stock_std.reshape(1, 1, -1)).astype(np.float32)


def scale_stock_features_for_context(
    stock_features_raw: np.ndarray,
    *,
    context_feature_start: int,
    context_feature_stop: int,
    price_normalization_mode: str,
    stock_mean: np.ndarray,
    stock_std: np.ndarray,
) -> np.ndarray:
    transformed = transform_stock_features_for_context(
        stock_features_raw,
        context_feature_start=context_feature_start,
        context_feature_stop=context_feature_stop,
        price_normalization_mode=price_normalization_mode,
    )
    return scale_stock_feature_context_array(
        transformed,
        price_normalization_mode="none",
        stock_mean=stock_mean,
        stock_std=stock_std,
    )
