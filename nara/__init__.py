"""NARA: Noise-Adaptive Response Alignment for repeat-pass wireline logs."""

from .graph import MODELS, Graph, correlation, regularized_correlation, slope_dtw
from .solver import (
    align_nara,
    alternating_huber_dp,
    compute_base_costs,
    dp_with_precomputed,
    huber_loss,
    is_same_side_bound_sat,
    pilot_data_residual,
    solve_core,
)

__version__ = "1.0.0"

__all__ = [
    "MODELS",
    "Graph",
    "align_nara",
    "alternating_huber_dp",
    "compute_base_costs",
    "correlation",
    "dp_with_precomputed",
    "huber_loss",
    "is_same_side_bound_sat",
    "pilot_data_residual",
    "regularized_correlation",
    "slope_dtw",
    "solve_core",
]
