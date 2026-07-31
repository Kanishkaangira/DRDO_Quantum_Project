"""Zero-noise extrapolation from noise-scaled circuit executions."""

from collections.abc import Sequence

import numpy as np


def mitigate_zne(
    scale_factors: Sequence[float],
    scaled_expectations: Sequence[float],
    extrapolation_degree: int,
) -> float:
    """Extrapolate runner-provided noise-scaled expectations to zero noise."""
    if extrapolation_degree < 1:
        raise ValueError("extrapolation_degree must be at least 1.")
    if len(scale_factors) < extrapolation_degree + 1:
        raise ValueError("Not enough scale factors for the requested extrapolation degree.")
    if len(scale_factors) != len(scaled_expectations):
        raise ValueError("scale_factors and scaled_expectations must have equal lengths.")
    if len(set(scale_factors)) != len(scale_factors) or any(
        scale_factor <= 0 for scale_factor in scale_factors
    ):
        raise ValueError("scale_factors must be distinct positive values.")

    coefficients = np.polyfit(scale_factors, scaled_expectations, extrapolation_degree)
    return float(np.polyval(coefficients, 0.0))
