"""Execution of ordered benchmark cases."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from qiskit import QuantumCircuit

from circuits import build_circuit
from config.settings import SEED, SEED_STRIDE
from evaluation import (
    calculate_classical_fidelity,
    calculate_expectation_absolute_error,
    calculate_overhead,
    calculate_ring_maxcut_expectation,
    calculate_total_variation_distance,
    calculate_z_parity_expectation,
)
from execution import execute_circuit
from noise import build_noise_model
from suppression import apply_suppression

from .experiment import (
    ExperimentExecution,
    auxiliary_seed,
    build_m3_calibration_matrix,
    build_trex_calibration_expectation,
    mitigate_experiment,
)
from .parameter_sweep import BenchmarkCase, iter_active_benchmark_cases


class ConditionExecutor(Protocol):
    """Execute a condition circuit while recording its time and shot usage."""

    def __call__(
        self,
        execution_circuit: QuantumCircuit,
        execution_seed: int,
        execution_noise_model: Any = ...,
    ) -> Mapping[str, int]: ...


AuxiliaryDataProvider = Callable[
    [
        str,
        BenchmarkCase,
        QuantumCircuit,
        ExperimentExecution,
        ConditionExecutor,
    ],
    Mapping[str, Any],
]
SuppressionContextProvider = Callable[[BenchmarkCase], Mapping[str, Any]]


@dataclass(frozen=True)
class BenchmarkResult:
    """Evaluated output for one configured branch execution."""

    case: BenchmarkCase
    original_depth: int
    evaluated_depth: int
    raw_expectation: float
    ideal_expectation: float
    expectation_absolute_error: float
    fidelity: float
    total_variation_distance: float
    mitigated_values: Mapping[str, Mapping[str, float] | float]
    execution_time: float
    runtime_overhead: float
    sampling_overhead: float


def _expectation_for_circuit(circuit_name: str, counts: Mapping[str, int]) -> float:
    if circuit_name == "qaoa":
        return calculate_ring_maxcut_expectation(counts)
    if circuit_name == "vqe":
        raise NotImplementedError(
            "VQE evaluation requires a configured Hamiltonian; Z-parity is not valid."
        )
    return calculate_z_parity_expectation(counts)


def _run_seed(case: BenchmarkCase) -> int:
    return SEED + case.run_index * SEED_STRIDE


def _apply_mitigations(
    case: BenchmarkCase,
    evaluated_circuit: QuantumCircuit,
    raw_execution: ExperimentExecution,
    *,
    execute_condition: ConditionExecutor,
    auxiliary_data_provider: AuxiliaryDataProvider | None,
) -> dict[str, Mapping[str, float] | float]:
    mitigated_values: dict[str, Mapping[str, float] | float] = {}
    for technique in case.mitigation_techniques:
        if technique == "m3":
            calibration_index = 0

            def execute_calibration(calibration_circuit: QuantumCircuit) -> Mapping[str, int]:
                nonlocal calibration_index
                seed = auxiliary_seed(case, "m3", calibration_index)
                calibration_index += 1
                return execute_condition(calibration_circuit, seed)

            mitigated_values[technique] = mitigate_experiment(
                technique,
                raw_execution,
                calibration_matrix=build_m3_calibration_matrix(
                    case.num_qubits, execute_calibration
                ),
            )
        elif technique == "trex":
            mitigated_values[technique] = mitigate_experiment(
                technique,
                raw_execution,
                calibration_expectation=build_trex_calibration_expectation(
                    case.num_qubits,
                    lambda calibration_circuit: execute_condition(
                        calibration_circuit, auxiliary_seed(case, "trex", 0)
                    ),
                ),
            )
        else:
            if auxiliary_data_provider is None:
                raise ValueError(f"{technique} requires an auxiliary data provider.")
            mitigated_values[technique] = mitigate_experiment(
                technique,
                raw_execution,
                **auxiliary_data_provider(
                    technique,
                    case,
                    evaluated_circuit,
                    raw_execution,
                    execute_condition,
                ),
            )
    return mitigated_values


def run_benchmark_case(
    case: BenchmarkCase,
    *,
    ibm_backend: Any = None,
    auxiliary_data_provider: AuxiliaryDataProvider | None = None,
    suppression_context_provider: SuppressionContextProvider | None = None,
) -> BenchmarkResult:
    """Run one branch once, then apply every configured mitigation to its result."""
    original_circuit = build_circuit(
        case.circuit_name, case.num_qubits, **case.build_parameters
    )
    suppression_context: Mapping[str, Any] = {}
    if case.suppression_mode in {"dd", "both"}:
        if suppression_context_provider is None:
            raise ValueError("DD suppression requires a timing context provider.")
        suppression_context = suppression_context_provider(case)
    evaluated_circuit = apply_suppression(
        original_circuit,
        case.suppression_mode,
        seed=_run_seed(case),
        target=suppression_context.get("target"),
        instruction_durations=suppression_context.get("instruction_durations"),
        qubits=suppression_context.get("qubits"),
    )

    noise_model = None
    if case.noise_name is not None and case.noise_strength is not None:
        noise_model = build_noise_model(case.noise_name, case.noise_strength)

    execution_time = 0.0
    total_shots = 0
    default_noise_model = object()

    def execute_condition(
        execution_circuit: QuantumCircuit,
        execution_seed: int,
        execution_noise_model: Any = default_noise_model,
    ) -> Mapping[str, int]:
        nonlocal execution_time, total_shots
        started_at = perf_counter()
        counts = execute_circuit(
            execution_circuit,
            case.backend_name,
            case.shots,
            noise_model=(
                noise_model
                if execution_noise_model is default_noise_model
                else execution_noise_model
            ),
            seed=execution_seed,
            ibm_backend=ibm_backend,
        )
        execution_time += perf_counter() - started_at
        total_shots += case.shots
        return counts

    raw_counts = execute_condition(evaluated_circuit, _run_seed(case))
    baseline_started_at = perf_counter()
    ideal_counts = execute_circuit(
        original_circuit, "aer", case.shots, seed=_run_seed(case)
    )
    baseline_execution_time = perf_counter() - baseline_started_at
    raw_execution = ExperimentExecution(
        counts=raw_counts,
        expectation=_expectation_for_circuit(case.circuit_name, raw_counts),
    )
    ideal_expectation = _expectation_for_circuit(case.circuit_name, ideal_counts)
    mitigated_values = _apply_mitigations(
        case,
        evaluated_circuit,
        raw_execution,
        execute_condition=execute_condition,
        auxiliary_data_provider=auxiliary_data_provider,
    )
    overhead = calculate_overhead(
        original_circuit,
        evaluated_circuit,
        case.shots,
        total_shots,
        baseline_execution_seconds=baseline_execution_time,
        total_execution_seconds=execution_time,
    )
    if overhead.execution_time_ratio is None:
        raise RuntimeError("Execution-time overhead was not calculated.")

    return BenchmarkResult(
        case=case,
        original_depth=original_circuit.depth(),
        evaluated_depth=evaluated_circuit.depth(),
        raw_expectation=raw_execution.expectation,
        ideal_expectation=ideal_expectation,
        expectation_absolute_error=calculate_expectation_absolute_error(
            raw_execution.expectation, ideal_expectation
        ),
        fidelity=calculate_classical_fidelity(raw_counts, ideal_counts),
        total_variation_distance=calculate_total_variation_distance(
            raw_counts, ideal_counts
        ),
        mitigated_values=mitigated_values,
        execution_time=execution_time,
        runtime_overhead=overhead.execution_time_ratio,
        sampling_overhead=overhead.shot_ratio,
    )


def run_active_benchmark(
    *,
    ibm_backend: Any = None,
    auxiliary_data_provider: AuxiliaryDataProvider | None = None,
    suppression_context_provider: SuppressionContextProvider | None = None,
) -> list[BenchmarkResult]:
    """Run every branch case selected by the active development-phase configuration."""
    return [
        run_benchmark_case(
            case,
            ibm_backend=ibm_backend,
            auxiliary_data_provider=auxiliary_data_provider,
            suppression_context_provider=suppression_context_provider,
        )
        for case in iter_active_benchmark_cases()
    ]
