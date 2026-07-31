"""Probabilistic error-cancellation estimator."""

from collections.abc import Sequence

import numpy as np


def mitigate_pec(
    sampled_expectations: Sequence[float],
    sampling_signs: Sequence[int],
    sampling_overhead: float,
) -> float:
    """Combine signed quasi-probability samples into a PEC expectation estimate."""
    if not sampled_expectations:
        raise ValueError("sampled_expectations must not be empty.")
    if len(sampled_expectations) != len(sampling_signs):
        raise ValueError("sampled_expectations and sampling_signs must have equal lengths.")
    if sampling_overhead <= 0:
        raise ValueError("sampling_overhead must be positive.")
    if any(sign not in (-1, 1) for sign in sampling_signs):
        raise ValueError("sampling_signs must contain only -1 or 1.")

    return float(
        sampling_overhead
        * np.mean(np.asarray(sampled_expectations) * np.asarray(sampling_signs))
    )
