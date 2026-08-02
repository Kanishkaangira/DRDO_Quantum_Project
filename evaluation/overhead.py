"""Resource-overhead metrics for suppression and mitigation experiments."""

from dataclasses import dataclass

from qiskit import QuantumCircuit


@dataclass(frozen=True)
class OverheadMetrics:
    """Circuit, sampling, and optional wall-time overhead relative to baseline."""

    depth_ratio: float
    gate_count_ratio: float
    shot_ratio: float
    execution_time_ratio: float | None


def calculate_overhead(
    original_circuit: QuantumCircuit,
    evaluated_circuit: QuantumCircuit,
    baseline_shots: int,
    total_shots: int,
    *,
    baseline_execution_seconds: float | None = None,
    total_execution_seconds: float | None = None,
) -> OverheadMetrics:
    """Calculate resource ratios for one evaluated configuration."""
    if baseline_shots < 1 or total_shots < 1:
        raise ValueError("Shot counts must be at least 1.")
    if original_circuit.depth() < 1 or original_circuit.size() < 1:
        raise ValueError("original_circuit must contain operations.")
    if (baseline_execution_seconds is None) != (total_execution_seconds is None):
        raise ValueError("Both execution times must be provided together.")
    if baseline_execution_seconds is not None and baseline_execution_seconds <= 0:
        raise ValueError("baseline_execution_seconds must be positive.")

    execution_time_ratio = None
    if baseline_execution_seconds is not None and total_execution_seconds is not None:
        execution_time_ratio = total_execution_seconds / baseline_execution_seconds

    return OverheadMetrics(
        depth_ratio=evaluated_circuit.depth() / original_circuit.depth(),
        gate_count_ratio=evaluated_circuit.size() / original_circuit.size(),
        shot_ratio=total_shots / baseline_shots,
        execution_time_ratio=execution_time_ratio,
    )
