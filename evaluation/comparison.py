"""Comparison of a candidate result against its baseline and ideal reference."""

from collections.abc import Mapping
from dataclasses import dataclass

from .error_metrics import calculate_total_variation_distance
from .fidelity import calculate_classical_fidelity


@dataclass(frozen=True)
class ComparisonMetrics:
    """Quality change produced by one suppression or mitigation condition."""

    baseline_fidelity: float
    candidate_fidelity: float
    fidelity_improvement: float
    baseline_total_variation_distance: float
    candidate_total_variation_distance: float
    total_variation_distance_improvement: float


def compare_to_baseline(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    ideal: Mapping[str, float],
) -> ComparisonMetrics:
    """Compare candidate quality to an equivalent baseline against one reference."""
    baseline_fidelity = calculate_classical_fidelity(baseline, ideal)
    candidate_fidelity = calculate_classical_fidelity(candidate, ideal)
    baseline_distance = calculate_total_variation_distance(baseline, ideal)
    candidate_distance = calculate_total_variation_distance(candidate, ideal)
    return ComparisonMetrics(
        baseline_fidelity=baseline_fidelity,
        candidate_fidelity=candidate_fidelity,
        fidelity_improvement=candidate_fidelity - baseline_fidelity,
        baseline_total_variation_distance=baseline_distance,
        candidate_total_variation_distance=candidate_distance,
        total_variation_distance_improvement=baseline_distance - candidate_distance,
    )
