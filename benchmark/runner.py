"""Execution of ordered benchmark cases."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from qiskit import QuantumCircuit

from circuits import build_circuit
from config.benchmark_config import VQE_HAMILTONIAN
from config.settings import SEED, SEED_STRIDE
from evaluation import (
    calculate_classical_fidelity,
    calculate_expectation_absolute_error,
    calculate_overhead,
    calculate_ring_ising_energy,
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
ConditionExecutorFactory = Callable[
    [], tuple[ConditionExecutor, Callable[[], tuple[float, int]]]
]


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
    mitigation_contexts: Mapping[str, Mapping[str, Any]]
    mitigated_expectations: Mapping[str, float]
    mitigated_expectation_absolute_errors: Mapping[str, float]
    mitigated_fidelities: Mapping[str, float | None]
    mitigated_total_variation_distances: Mapping[str, float | None]
    mitigation_execution_times: Mapping[str, float]
    mitigation_total_shots: Mapping[str, int]
    mitigation_runtime_overheads: Mapping[str, float]
    mitigation_sampling_overheads: Mapping[str, float]
    raw_counts: Mapping[str, int]
    ideal_counts: Mapping[str, int]
    original_gate_count: int
    evaluated_gate_count: int
    depth_overhead: float
    gate_count_overhead: float
    total_shots: int
    baseline_execution_time: float
    execution_time: float
    runtime_overhead: float
    sampling_overhead: float


def _expectation_for_circuit(circuit_name: str, counts: Mapping[str, int]) -> float:
    if circuit_name == "qaoa":
        return calculate_ring_maxcut_expectation(counts)
    if circuit_name == "vqe":
        return calculate_ring_ising_energy(counts, **VQE_HAMILTONIAN)
    return calculate_z_parity_expectation(counts)


def _run_seed(case: BenchmarkCase) -> int:
    return SEED + case.run_index * SEED_STRIDE


def _apply_mitigations(
    case: BenchmarkCase,
    evaluated_circuit: QuantumCircuit,
    raw_execution: ExperimentExecution,
    *,
    execute_condition_factory: ConditionExecutorFactory,
    auxiliary_data_provider: AuxiliaryDataProvider | None,
) -> tuple[
    dict[str, Mapping[str, float] | float],
    dict[str, Mapping[str, Any]],
    dict[str, tuple[float, int]],
]:
    mitigated_values: dict[str, Mapping[str, float] | float] = {}
    mitigation_contexts: dict[str, Mapping[str, Any]] = {}
    mitigation_execution_metrics: dict[str, tuple[float, int]] = {}
    for technique in case.mitigation_techniques:
        execute_condition, execution_metrics = execute_condition_factory()
        if technique == "m3":
            calibration_index = 0

            def execute_calibration(calibration_circuit: QuantumCircuit) -> Mapping[str, int]:
                nonlocal calibration_index
                seed = auxiliary_seed(case, "m3", calibration_index)
                calibration_index += 1
                return execute_condition(calibration_circuit, seed)

            calibration_matrix = build_m3_calibration_matrix(
                case.num_qubits, execute_calibration
            )
            mitigated_values[technique] = mitigate_experiment(
                technique,
                raw_execution,
                calibration_matrix=calibration_matrix,
            )
            mitigation_contexts[technique] = {
                **case.mitigation_parameters[technique],
                "calibration_circuit_count": 2**case.num_qubits,
                "calibration_matrix": calibration_matrix.tolist(),
            }
        elif technique == "trex":
            calibration_expectation = build_trex_calibration_expectation(
                case.num_qubits,
                lambda calibration_circuit: execute_condition(
                    calibration_circuit, auxiliary_seed(case, "trex", 0)
                ),
            )
            mitigated_values[technique] = mitigate_experiment(
                technique,
                raw_execution,
                calibration_expectation=calibration_expectation,
            )
            mitigation_contexts[technique] = {
                **case.mitigation_parameters[technique],
                "calibration_expectation": calibration_expectation,
            }
        else:
            if auxiliary_data_provider is None:
                raise ValueError(f"{technique} requires an auxiliary data provider.")
            mitigation_data = auxiliary_data_provider(
                technique,
                case,
                evaluated_circuit,
                raw_execution,
                execute_condition,
            )
            mitigated_values[technique] = mitigate_experiment(
                technique,
                raw_execution,
                **mitigation_data,
            )
            mitigation_contexts[technique] = {
                **case.mitigation_parameters[technique],
                **mitigation_data,
            }
        mitigation_execution_metrics[technique] = execution_metrics()
    return mitigated_values, mitigation_contexts, mitigation_execution_metrics


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
        dd_sequence=case.suppression_parameters.get("dd_sequence"),
    )

    noise_model = None
    if case.noise_name is not None and case.noise_strength is not None:
        noise_model = build_noise_model(case.noise_name, case.noise_strength)

    default_noise_model = object()

    def make_condition_executor(
        initial_execution_time: float = 0.0, initial_total_shots: int = 0
    ) -> tuple[ConditionExecutor, Callable[[], tuple[float, int]]]:
        execution_time = initial_execution_time
        total_shots = initial_total_shots

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

        def get_execution_metrics() -> tuple[float, int]:
            return execution_time, total_shots

        return execute_condition, get_execution_metrics

    execute_raw_condition, raw_execution_metrics = make_condition_executor()

    raw_counts = execute_raw_condition(evaluated_circuit, _run_seed(case))
    raw_execution_time, raw_total_shots = raw_execution_metrics()

    def execute_condition_factory() -> tuple[
        ConditionExecutor, Callable[[], tuple[float, int]]
    ]:
        return make_condition_executor(raw_execution_time, raw_total_shots)

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
    mitigated_values, mitigation_contexts, mitigation_execution_metrics = (
        _apply_mitigations(
            case,
            evaluated_circuit,
            raw_execution,
            execute_condition_factory=execute_condition_factory,
            auxiliary_data_provider=auxiliary_data_provider,
        )
    )
    overhead = calculate_overhead(
        original_circuit,
        evaluated_circuit,
        case.shots,
        raw_total_shots,
        baseline_execution_seconds=baseline_execution_time,
        total_execution_seconds=raw_execution_time,
    )
    if overhead.execution_time_ratio is None:
        raise RuntimeError("Execution-time overhead was not calculated.")

    mitigation_execution_times: dict[str, float] = {}
    mitigation_total_shots: dict[str, int] = {}
    mitigation_runtime_overheads: dict[str, float] = {}
    mitigation_sampling_overheads: dict[str, float] = {}
    for technique, (technique_execution_time, technique_total_shots) in (
        mitigation_execution_metrics.items()
    ):
        technique_overhead = calculate_overhead(
            original_circuit,
            evaluated_circuit,
            case.shots,
            technique_total_shots,
            baseline_execution_seconds=baseline_execution_time,
            total_execution_seconds=technique_execution_time,
        )
        if technique_overhead.execution_time_ratio is None:
            raise RuntimeError("Technique execution-time overhead was not calculated.")
        mitigation_execution_times[technique] = technique_execution_time
        mitigation_total_shots[technique] = technique_total_shots
        mitigation_runtime_overheads[technique] = technique_overhead.execution_time_ratio
        mitigation_sampling_overheads[technique] = technique_overhead.shot_ratio

    mitigated_expectations: dict[str, float] = {}
    mitigated_expectation_absolute_errors: dict[str, float] = {}
    mitigated_fidelities: dict[str, float | None] = {}
    mitigated_total_variation_distances: dict[str, float | None] = {}
    for technique, mitigated_value in mitigated_values.items():
        if isinstance(mitigated_value, Mapping):
            mitigated_expectation = _expectation_for_circuit(
                case.circuit_name, mitigated_value
            )
            mitigated_fidelities[technique] = calculate_classical_fidelity(
                mitigated_value, ideal_counts
            )
            mitigated_total_variation_distances[technique] = (
                calculate_total_variation_distance(mitigated_value, ideal_counts)
            )
        else:
            mitigated_expectation = float(mitigated_value)
            mitigated_fidelities[technique] = None
            mitigated_total_variation_distances[technique] = None
        mitigated_expectations[technique] = mitigated_expectation
        mitigated_expectation_absolute_errors[technique] = (
            calculate_expectation_absolute_error(
                mitigated_expectation, ideal_expectation
            )
        )

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
        mitigation_contexts=mitigation_contexts,
        mitigated_expectations=mitigated_expectations,
        mitigated_expectation_absolute_errors=mitigated_expectation_absolute_errors,
        mitigated_fidelities=mitigated_fidelities,
        mitigated_total_variation_distances=mitigated_total_variation_distances,
        mitigation_execution_times=mitigation_execution_times,
        mitigation_total_shots=mitigation_total_shots,
        mitigation_runtime_overheads=mitigation_runtime_overheads,
        mitigation_sampling_overheads=mitigation_sampling_overheads,
        raw_counts=raw_counts,
        ideal_counts=ideal_counts,
        original_gate_count=original_circuit.size(),
        evaluated_gate_count=evaluated_circuit.size(),
        depth_overhead=overhead.depth_ratio,
        gate_count_overhead=overhead.gate_count_ratio,
        total_shots=raw_total_shots,
        baseline_execution_time=baseline_execution_time,
        execution_time=raw_execution_time,
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
