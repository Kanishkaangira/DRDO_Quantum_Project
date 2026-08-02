"""Single-experiment execution and mitigation integration."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil, floor, pi
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from config.settings import SEED, SEED_STRIDE
from evaluation import calculate_ring_maxcut_expectation, calculate_z_parity_expectation
from execution import execute_aer
from mitigation import apply_mitigation
from suppression import apply_pauli_twirling

if TYPE_CHECKING:
    from .parameter_sweep import BenchmarkCase


ZNE_SCALE_FACTORS = (1.0, 3.0, 5.0)
PEC_SAMPLE_COUNT = 8
CDR_TRAINING_CIRCUIT_COUNT = 4

MITIGATION_SEED_OFFSETS = {
    "zne": 0,
    "pec": 100,
    "cdr": 200,
    "m3": 300,
    "trex": 400,
}


def auxiliary_seed(case: "BenchmarkCase", technique: str, index: int) -> int:
    """Return a run-unique, technique-namespaced auxiliary execution seed."""
    try:
        offset = MITIGATION_SEED_OFFSETS[technique]
    except KeyError as error:
        raise ValueError(f"No seed offset is configured for: {technique}") from error
    return SEED + case.run_index * SEED_STRIDE + offset + index + 1


@dataclass(frozen=True)
class ExperimentExecution:
    """Raw result of one circuit execution before mitigation."""

    counts: Mapping[str, int]
    expectation: float


def run_aer_experiment(
    circuit: QuantumCircuit,
    shots: int,
    *,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
) -> ExperimentExecution:
    """Execute a circuit once and calculate its Z-parity expectation."""
    counts = execute_aer(circuit, shots, noise_model=noise_model, seed=seed)
    return ExperimentExecution(
        counts=counts,
        expectation=calculate_z_parity_expectation(counts),
    )


def build_m3_calibration_matrix(
    num_qubits: int, execute_counts: Callable[[QuantumCircuit], Mapping[str, int]]
) -> np.ndarray:
    """Build the measured-by-ideal assignment matrix required for M3 correction."""
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1.")

    dimension = 2**num_qubits
    calibration_matrix = np.zeros((dimension, dimension))
    for ideal_outcome in range(dimension):
        calibration_circuit = QuantumCircuit(num_qubits, num_qubits)
        for qubit in range(num_qubits):
            if ideal_outcome & (1 << qubit):
                calibration_circuit.x(qubit)
        calibration_circuit.measure(range(num_qubits), range(num_qubits))

        counts = execute_counts(calibration_circuit)
        total_shots = sum(counts.values())
        if total_shots <= 0:
            raise ValueError("Calibration execution returned no shots.")
        for measured_outcome, count in counts.items():
            calibration_matrix[int(measured_outcome.replace(" ", ""), 2), ideal_outcome] = (
                count / total_shots
            )
    return calibration_matrix


def build_trex_calibration_expectation(
    num_qubits: int, execute_counts: Callable[[QuantumCircuit], Mapping[str, int]]
) -> float:
    """Measure the all-zero calibration response used for TREX correction."""
    calibration_circuit = QuantumCircuit(num_qubits, num_qubits)
    calibration_circuit.measure(range(num_qubits), range(num_qubits))
    return calculate_z_parity_expectation(execute_counts(calibration_circuit))


def _calculate_case_expectation(
    case: "BenchmarkCase", counts: Mapping[str, int]
) -> float:
    if case.circuit_name == "qaoa":
        return calculate_ring_maxcut_expectation(counts)
    if case.circuit_name == "vqe":
        raise NotImplementedError("VQE requires a Hamiltonian-specific expectation.")
    return calculate_z_parity_expectation(counts)


def _fold_circuit(circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
    if scale_factor < 1 or int(scale_factor) != scale_factor or int(scale_factor) % 2 == 0:
        raise ValueError("ZNE scale factors must be positive odd integers.")

    body = circuit.remove_final_measurements(inplace=False)
    folded = QuantumCircuit(
        circuit.num_qubits, circuit.num_clbits, name=f"{circuit.name}_zne_{scale_factor}"
    )
    folded.compose(body, inplace=True)
    for _ in range((int(scale_factor) - 1) // 2):
        folded.compose(body.inverse(), inplace=True)
        folded.compose(body, inplace=True)
    folded.measure(range(circuit.num_qubits), range(circuit.num_clbits))
    return folded


def build_zne_scaled_expectations(
    case: "BenchmarkCase",
    circuit: QuantumCircuit,
    execute_condition: Callable[..., Mapping[str, int]],
    *,
    scale_factors: Sequence[float] = ZNE_SCALE_FACTORS,
    extrapolation_degree: int = 2,
) -> dict[str, Sequence[float] | int]:
    """Fold and execute noise-scaled variants for zero-noise extrapolation."""
    if len(scale_factors) < extrapolation_degree + 1:
        raise ValueError("Not enough ZNE scale factors for the extrapolation degree.")

    scaled_expectations = [
        _calculate_case_expectation(
            case,
            execute_condition(
                _fold_circuit(circuit, scale_factor), auxiliary_seed(case, "zne", index)
            ),
        )
        for index, scale_factor in enumerate(scale_factors)
    ]
    return {
        "scale_factors": scale_factors,
        "scaled_expectations": scaled_expectations,
        "extrapolation_degree": extrapolation_degree,
    }


def build_pec_samples(
    case: "BenchmarkCase",
    circuit: QuantumCircuit,
    execute_condition: Callable[..., Mapping[str, int]],
    *,
    sample_count: int = PEC_SAMPLE_COUNT,
) -> dict[str, Sequence[float] | Sequence[int] | float]:
    """Execute placeholder Pauli-twirled samples through the tracked executor.

    This is not full probabilistic error cancellation: it uses unit signs and
    unit sampling overhead until learned/quasi-probability decomposition data
    is available.
    """
    if sample_count < 1:
        raise ValueError("PEC sample_count must be at least 1.")

    sampled_expectations = []
    for sample_index in range(sample_count):
        seed = auxiliary_seed(case, "pec", sample_index)
        sampled_circuit = apply_pauli_twirling(
            circuit, seed=seed
        )
        sampled_expectations.append(
            _calculate_case_expectation(
                case,
                execute_condition(sampled_circuit, seed),
            )
        )
    return {
        "sampled_expectations": sampled_expectations,
        "sampling_signs": [1] * sample_count,
        "sampling_overhead": 1.0,
    }


def _near_clifford_circuit(circuit: QuantumCircuit, variant_index: int) -> QuantumCircuit:
    """Create an angle-snapped training circuit for variational ansatz circuits.

    Only RX, RY, and RZ gates are changed, so CDR training variants are
    meaningful for VQE/QAOA only; Bell, GHZ, and QFT circuits remain identical.
    """
    training_circuit = circuit.copy_empty_like(name=f"{circuit.name}_cdr_{variant_index}")
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in {"rx", "ry", "rz"}:
            angle = float(operation.params[0])
            quarter_turn = angle / (pi / 2)
            if variant_index % 3 == 0:
                snapped_turns = round(quarter_turn)
            elif variant_index % 3 == 1:
                snapped_turns = floor(quarter_turn)
            else:
                snapped_turns = ceil(quarter_turn)
            getattr(training_circuit, operation.name)(
                snapped_turns * pi / 2, instruction.qubits[0]
            )
        else:
            training_circuit.append(
                operation, instruction.qubits, instruction.clbits
            )
    return training_circuit


def build_cdr_training_data(
    case: "BenchmarkCase",
    circuit: QuantumCircuit,
    execute_condition: Callable[..., Mapping[str, int]],
    *,
    training_circuit_count: int = CDR_TRAINING_CIRCUIT_COUNT,
) -> dict[str, Sequence[float]]:
    """Execute noisy and ideal near-Clifford training circuits for CDR."""
    if case.backend_name != "aer":
        raise NotImplementedError("CDR ideal references currently require the Aer backend.")
    if training_circuit_count < 2:
        raise ValueError("CDR requires at least two training circuits.")

    noisy_expectations = []
    ideal_expectations = []
    for variant_index in range(training_circuit_count):
        training_circuit = _near_clifford_circuit(circuit, variant_index)
        noisy_seed = auxiliary_seed(case, "cdr", 2 * variant_index)
        ideal_seed = auxiliary_seed(case, "cdr", 2 * variant_index + 1)
        noisy_expectations.append(
            _calculate_case_expectation(
                case, execute_condition(training_circuit, noisy_seed)
            )
        )
        ideal_expectations.append(
            _calculate_case_expectation(
                case, execute_condition(training_circuit, ideal_seed, None)
            )
        )
    return {
        "training_noisy_expectations": noisy_expectations,
        "training_ideal_expectations": ideal_expectations,
    }


def provide_auxiliary_data(
    technique: str,
    case: "BenchmarkCase",
    circuit: QuantumCircuit,
    raw_execution: ExperimentExecution,
    execute_condition: Callable[..., Mapping[str, int]],
) -> dict[str, Any]:
    """Provide tracked additional-execution data for the selected mitigation technique."""
    del raw_execution
    if technique == "zne":
        return build_zne_scaled_expectations(case, circuit, execute_condition)
    if technique == "pec":
        return build_pec_samples(case, circuit, execute_condition)
    if technique == "cdr":
        return build_cdr_training_data(case, circuit, execute_condition)
    raise ValueError(f"Unsupported auxiliary mitigation technique: {technique}")


def mitigate_experiment(
    technique: str,
    execution: ExperimentExecution,
    **mitigation_data: Any,
) -> Any:
    """Dispatch one mitigation technique using a raw execution and auxiliary data."""
    if technique == "m3":
        return apply_mitigation(
            technique,
            counts=execution.counts,
            **mitigation_data,
        )
    if technique == "trex":
        return apply_mitigation(
            technique,
            raw_expectation=execution.expectation,
            **mitigation_data,
        )
    if technique == "cdr":
        return apply_mitigation(
            technique,
            target_noisy_expectation=execution.expectation,
            **mitigation_data,
        )
    return apply_mitigation(technique, **mitigation_data)
