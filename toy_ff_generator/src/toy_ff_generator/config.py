"""Default configuration for toy_ff_generator.

Edit the values in the "Tunable defaults" section when preparing a run.
``build_default_config()`` keeps returning the nested dict consumed by the
existing generation pipeline.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATE_ORDER = (-1, 0, 1)
STATE_NAME_MAP = {-1: "bear", 0: "neutral", 1: "bull"}

MU_CLASS_LABELS = ("low", "mid", "high")
MU_AXES = ("characteristic_1", "characteristic_2", "characteristic_3")
PROFILE_GROUP_LABELS = ("mid", "low", "high")


# ---------------------------------------------------------------------------
# Tunable defaults
# ---------------------------------------------------------------------------

# Run size and reproducibility.
DEFAULT_N = 4860
DEFAULT_T = 200
DEFAULT_DATASET_COUNT = 88
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_WORKERS = None

# Market regime. Change DEFAULT_INITIAL_STATE once; output_dir follows it.
DEFAULT_INITIAL_STATE = 1
DEFAULT_STATE_SEQUENCE = None
DEFAULT_TRANSITION_MATRIX = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]

# Output location.
DEFAULT_OUTPUT_VERSION = "data v3"

# Factor vector AR parameters.
DEFAULT_X0 = [0.0, 0.0, 0.0]
DEFAULT_PHI = [
    [0.6, 0.05, 0.02],
    [0.04, 0.55, 0.03],
    [0.02, 0.04, 0.25],
]
DEFAULT_MU_BEAR = [-0.003, -0.003, 0.005]
DEFAULT_MU_NEUTRAL = [0.0, 0.0, 0.0]
DEFAULT_MU_BULL = [0.002, 0.002, -0.0015]
DEFAULT_SIGMA_X_BEAR = [
    [0.000040, 0.0000035, 0.0000020],
    [0.0000035, 0.0000400, 0.0000018],
    [0.0000020, 0.0000018, 0.0000080],
]
DEFAULT_SIGMA_X_NEUTRAL = [
    [0.000012, 0.0000015, 0.0000008],
    [0.0000015, 0.0000065, 0.0000010],
    [0.0000008, 0.0000010, 0.0000055],
]
DEFAULT_SIGMA_X_BULL = [
    [0.000009, 0.0000018, 0.0000005],
    [0.0000018, 0.0000055, 0.0000006],
    [0.0000005, 0.0000006, 0.0000045],
]

# Latent characteristic parameters.
DEFAULT_USE_SHARED_LATENT_STATE_PARAMS = False
DEFAULT_SHARED_LATENT_PARAMS = {
    "Omega": [0.65, 0.65, 0.65],
    "mu_Z": [0.0, 0.0, 0.0],
    "lambda_Z": [0.05, 0.03, 0.03],
    "sigma_Z": [0.05, 0.05, 0.05],
    "Z0": [0.0, 0.0, 0.0],
}
DEFAULT_PER_STOCK_OMEGA = [0.65, 0.65, 0.65]
DEFAULT_PER_STOCK_LAMBDA = [0.08, 0.05, 0.05]
DEFAULT_PER_STOCK_SIGMA_Z = [0.06, 0.06, 0.06]

# Exposure mapping.
DEFAULT_EXPOSURE_A = [
    [0.05, 0.0, 0.0],
    [0.0, 0.4, 0.0],
    [0.0, 0.0, 0.4],
]
DEFAULT_EXPOSURE_B = [1, 0.4, 0.0]

# Alpha and epsilon groups.
DEFAULT_ALPHA_GROUP = "mid"
DEFAULT_EPSILON_GROUP = "mid"
DEFAULT_ALPHA_LEVELS = {
    "low": -0.0002,
    "mid": 0.0001,
    "high": 0.0004,
}
DEFAULT_EPSILON_LEVELS = {
    "low": 0.005,
    "mid": 0.01,
    "high": 0.02,
}

# Return clipping and initial prices.
DEFAULT_LIMIT_UP = 0.10
DEFAULT_LIMIT_DOWN = -0.10
DEFAULT_SHARED_INIT_PRICE = True
DEFAULT_INITIAL_PRICE = 100.0
DEFAULT_PER_STOCK_INITIAL_PRICE_START = 100.0
DEFAULT_PER_STOCK_INITIAL_PRICE_STEP = 2.5


def _copy_vector(values: list[Any]) -> list[Any]:
    return list(values)


def _copy_matrix(values: list[list[Any]]) -> list[list[Any]]:
    return [list(row) for row in values]


def _copy_vector_dict(values: dict[str, list[float]]) -> dict[str, list[float]]:
    return {key: list(value) for key, value in values.items()}


def _copy_scalar_dict(values: dict[str, float]) -> dict[str, float]:
    return dict(values)


def _default_mu_class_centers() -> dict[str, float]:
    """Return the fixed low/mid/high centers for each mu class."""

    return {
        "low": -0.5,
        "mid": 0.0,
        "high": 0.5,
    }


def _default_mu_class_triplets(n: int) -> list[tuple[str, str, str]]:
    """Build deterministic three-axis mu class assignments."""

    all_triplets = list(product(MU_CLASS_LABELS, repeat=len(MU_AXES)))
    return [all_triplets[idx % len(all_triplets)] for idx in range(n)]


def _default_stock_profiles(
    n: int,
) -> list[tuple[tuple[str, str, str], str, str]]:
    """Build deterministic stock profiles for mu, alpha, and epsilon groups."""

    mu_triplets = _default_mu_class_triplets(len(MU_CLASS_LABELS) ** len(MU_AXES))
    all_profiles = [
        (mu_triplet, alpha_group, epsilon_group)
        for alpha_group in PROFILE_GROUP_LABELS
        for epsilon_group in PROFILE_GROUP_LABELS
        for mu_triplet in mu_triplets
    ]

    base_profile_count = len(all_profiles)
    base_block_size = n // base_profile_count
    remainder = n % base_profile_count

    stock_profiles: list[tuple[tuple[str, str, str], str, str]] = []
    for idx, profile in enumerate(all_profiles):
        block_size = base_block_size + (1 if idx < remainder else 0)
        stock_profiles.extend([profile] * block_size)

    return stock_profiles


def _default_per_stock_alpha_epsilon_groups(n: int) -> dict[str, list[str]]:
    """Return per-stock alpha and epsilon group assignments."""

    stock_profiles = _default_stock_profiles(n)
    return {
        "per_stock_alpha_groups": [alpha_group for _, alpha_group, _ in stock_profiles],
        "per_stock_epsilon_groups": [epsilon_group for _, _, epsilon_group in stock_profiles],
    }


def _triplets_to_mu_vectors(
    triplets: list[tuple[str, str, str]],
    class_centers: dict[str, float],
) -> list[list[float]]:
    """Convert mu class triplets into numeric three-dimensional mu vectors."""

    return [
        [class_centers[class_label] for class_label in triplet]
        for triplet in triplets
    ]


def _default_per_stock_latent_state_params(n: int) -> dict[str, list[list[float]]]:
    """Return per-stock latent state parameters with deterministic mu_i."""

    class_centers = _default_mu_class_centers()
    mu_class_triplets = [mu_triplet for mu_triplet, _, _ in _default_stock_profiles(n)]
    mu_i = _triplets_to_mu_vectors(mu_class_triplets, class_centers)

    return {
        "Omega_i": [_copy_vector(DEFAULT_PER_STOCK_OMEGA) for _ in range(n)],
        "mu_i": mu_i,
        "lambda_i": [_copy_vector(DEFAULT_PER_STOCK_LAMBDA) for _ in range(n)],
        "sigma_Z_i": [_copy_vector(DEFAULT_PER_STOCK_SIGMA_Z) for _ in range(n)],
        "Z0_i": [list(mu_vector) for mu_vector in mu_i],
    }


def _default_per_stock_initial_prices(n: int) -> list[float]:
    """Return deterministic per-stock initial prices."""

    return [
        DEFAULT_PER_STOCK_INITIAL_PRICE_START
        + DEFAULT_PER_STOCK_INITIAL_PRICE_STEP * idx
        for idx in range(n)
    ]


def build_default_config() -> dict[str, Any]:
    """Build the default nested dict consumed by the simulation pipeline."""

    stock_count = DEFAULT_N
    initial_state = DEFAULT_INITIAL_STATE

    return {
        "simulation_setup": {
            "N": stock_count,
            "T": DEFAULT_T,
            "dataset_count": DEFAULT_DATASET_COUNT,
            "random_seed": DEFAULT_RANDOM_SEED,
        },
        "batch_setup": {
            "max_workers": DEFAULT_MAX_WORKERS,
        },
        "market_state_setup": {
            "state_sequence": DEFAULT_STATE_SEQUENCE,
            "initial_state": initial_state,
            "transition_matrix": _copy_matrix(DEFAULT_TRANSITION_MATRIX),
        },
        "factor_vector_ar_setup": {
            "X0": _copy_vector(DEFAULT_X0),
            "Phi": _copy_matrix(DEFAULT_PHI),
            "mu_bear": _copy_vector(DEFAULT_MU_BEAR),
            "mu_neutral": _copy_vector(DEFAULT_MU_NEUTRAL),
            "mu_bull": _copy_vector(DEFAULT_MU_BULL),
            "Sigma_X_bear": _copy_matrix(DEFAULT_SIGMA_X_BEAR),
            "Sigma_X_neutral": _copy_matrix(DEFAULT_SIGMA_X_NEUTRAL),
            "Sigma_X_bull": _copy_matrix(DEFAULT_SIGMA_X_BULL),
        },
        "mu_class_setup": {
            "class_centers": _default_mu_class_centers(),
        },
        "latent_characteristic_setup": {
            "use_shared_latent_state_params": DEFAULT_USE_SHARED_LATENT_STATE_PARAMS,
            "shared_params": _copy_vector_dict(DEFAULT_SHARED_LATENT_PARAMS),
            "per_stock_params": _default_per_stock_latent_state_params(stock_count),
        },
        "exposure_setup": {
            "A": _copy_matrix(DEFAULT_EXPOSURE_A),
            "b": _copy_vector(DEFAULT_EXPOSURE_B),
        },
        "alpha_epsilon_mode_setup": {
            "alpha_group": DEFAULT_ALPHA_GROUP,
            "epsilon_group": DEFAULT_EPSILON_GROUP,
            "alpha_levels": _copy_scalar_dict(DEFAULT_ALPHA_LEVELS),
            "epsilon_levels": _copy_scalar_dict(DEFAULT_EPSILON_LEVELS),
            **_default_per_stock_alpha_epsilon_groups(stock_count),
        },
        "clipping_price_setup": {
            "limit_up": DEFAULT_LIMIT_UP,
            "limit_down": DEFAULT_LIMIT_DOWN,
            "shared_init_price": DEFAULT_SHARED_INIT_PRICE,
            "initial_price": DEFAULT_INITIAL_PRICE,
            "per_stock_initial_price": _default_per_stock_initial_prices(stock_count),
        },
        "output_setup": {
            "output_dir": str(
                PROJECT_ROOT
                / "outputs"
                / DEFAULT_OUTPUT_VERSION
                / STATE_NAME_MAP[initial_state]
            ),
        },
    }
