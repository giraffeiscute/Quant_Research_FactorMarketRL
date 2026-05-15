"""Model domain facade."""

from .allocation_distribution import (
    AllocationDistribution,
    AllocationDistributionResult,
    dirichlet_mean_from_logits,
    logits_to_dirichlet_alpha,
    logits_to_rl_post_train_dirichlet_alpha,
)
from .allocation_path import AllocationResult, AllocationSmoother
from .cross_sectional import (
    AttentionCrossSectionalScorer,
    CrossSectionalScoreResult,
    MLPCrossSectionalScorer,
)
from .network import PortfolioAttentionModel
from .task_head import AttentionPortfolioHead, MLPPortfolioHead, TaskHeadResult
from .temp_encoders import MarketTemporalEncoder, StockTemporalEncoder

__all__ = [
    "AllocationResult",
    "AllocationDistribution",
    "AllocationDistributionResult",
    "AllocationSmoother",
    "AttentionCrossSectionalScorer",
    "AttentionPortfolioHead",
    "CrossSectionalScoreResult",
    "dirichlet_mean_from_logits",
    "logits_to_dirichlet_alpha",
    "logits_to_rl_post_train_dirichlet_alpha",
    "MarketTemporalEncoder",
    "MLPCrossSectionalScorer",
    "MLPPortfolioHead",
    "PortfolioAttentionModel",
    "StockTemporalEncoder",
    "TaskHeadResult",
]
