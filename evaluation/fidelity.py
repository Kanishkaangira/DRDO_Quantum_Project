"""Classical fidelity metrics for measured outcomes."""

from collections.abc import Mapping

from .error_metrics import normalize_distribution


def calculate_classical_fidelity(
    observed: Mapping[str, float], ideal: Mapping[str, float]
) -> float:
    """Calculate squared Bhattacharyya fidelity between outcome distributions."""
    observed_probabilities = normalize_distribution(observed)
    ideal_probabilities = normalize_distribution(ideal)
    outcomes = observed_probabilities.keys() | ideal_probabilities.keys()
    overlap = sum(
        (observed_probabilities.get(outcome, 0.0) * ideal_probabilities.get(outcome, 0.0))
        ** 0.5
        for outcome in outcomes
    )
    return overlap**2
