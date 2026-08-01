"""Aer simulator execution used by benchmark experiments."""

from collections.abc import Mapping

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


def _is_timing_scheduled(circuit: QuantumCircuit) -> bool:
    """Return whether a circuit contains explicit scheduling delays."""
    return any(instruction.operation.name == "delay" for instruction in circuit.data)


def execute_aer(
    circuit: QuantumCircuit,
    shots: int,
    *,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
) -> Mapping[str, int]:
    """Execute one measured circuit on Aer and return its outcome counts."""
    if shots < 1:
        raise ValueError("shots must be at least 1.")
    simulator = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    executable_circuit = circuit
    if not _is_timing_scheduled(circuit):
        executable_circuit = transpile(
            circuit, simulator, optimization_level=0, seed_transpiler=seed
        )
    result = simulator.run(executable_circuit, shots=shots).result()
    return result.get_counts(executable_circuit)
