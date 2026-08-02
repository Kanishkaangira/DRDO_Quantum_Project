"""Distribution-error metrics for benchmark results."""

from collections.abc import Mapping
from dataclasses import dataclass


def normalize_distribution(distribution: Mapping[str, float]) -> dict[str, float]:
    """Convert counts or non-negative weights into normalized probabilities."""
    if not distribution:
        raise ValueError("distribution must not be empty.")
    if any(value < 0 for value in distribution.values()):
        raise ValueError("distribution values must be non-negative.")

    total = sum(distribution.values())
    if total <= 0:
        raise ValueError("distribution must contain positive total weight.")
    return {outcome: value / total for outcome, value in distribution.items()}


def calculate_total_variation_distance(
    observed: Mapping[str, float], ideal: Mapping[str, float]
) -> float:
    """Calculate total variation distance between two outcome distributions."""
    observed_probabilities = normalize_distribution(observed)
    ideal_probabilities = normalize_distribution(ideal)
    outcomes = observed_probabilities.keys() | ideal_probabilities.keys()
    return 0.5 * sum(
        abs(observed_probabilities.get(outcome, 0.0) - ideal_probabilities.get(outcome, 0.0))
        for outcome in outcomes
    )


@dataclass(frozen=True)
class ErrorMetrics:
    """Distribution error against an ideal reference result."""

    total_variation_distance: float


def calculate_error_metrics(
    observed: Mapping[str, float], ideal: Mapping[str, float]
) -> ErrorMetrics:
    """Calculate the standard distribution-error metrics for one result."""
    return ErrorMetrics(
        total_variation_distance=calculate_total_variation_distance(observed, ideal)
    )
