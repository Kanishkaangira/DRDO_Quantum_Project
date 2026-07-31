"""Clifford-data-regression expectation correction."""

from collections.abc import Sequence

import numpy as np


def mitigate_cdr(
    target_noisy_expectation: float,
    training_noisy_expectations: Sequence[float],
    training_ideal_expectations: Sequence[float],
) -> float:
    """Fit ideal expectations to noisy Clifford training data and correct the target."""
    if len(training_noisy_expectations) != len(training_ideal_expectations):
        raise ValueError("CDR training inputs must have equal lengths.")
    if len(training_noisy_expectations) < 2:
        raise ValueError("CDR requires at least two training circuits.")

    design_matrix = np.column_stack((training_noisy_expectations, np.ones(len(training_noisy_expectations))))
    slope, intercept = np.linalg.lstsq(
        design_matrix, training_ideal_expectations, rcond=None
    )[0]
    return float(slope * target_noisy_expectation + intercept)
