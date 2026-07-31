"""Suppression-mode dispatcher used by benchmark conditions."""

from qiskit import QuantumCircuit
from qiskit.transpiler import InstructionDurations, Target

from .dynamical_decoupling import apply_dynamical_decoupling
from .pauli_twirling import apply_pauli_twirling


def apply_suppression(
    circuit: QuantumCircuit,
    suppression_mode: str,
    *,
    seed: int | None = None,
    target: Target | None = None,
    instruction_durations: InstructionDurations | None = None,
    qubits: list[int] | None = None,
) -> QuantumCircuit:
    """Return a copy of a circuit with one configured suppression mode applied."""
    if suppression_mode == "none":
        return circuit.copy()
    if suppression_mode == "pauli_twirl":
        if seed is None:
            raise ValueError("Pauli twirling requires a run seed.")
        return apply_pauli_twirling(circuit, seed)
    if suppression_mode == "dd":
        return apply_dynamical_decoupling(
            circuit,
            target=target,
            instruction_durations=instruction_durations,
            qubits=qubits,
        )
    if suppression_mode == "both":
        if seed is None:
            raise ValueError("Pauli twirling requires a run seed.")
        return apply_dynamical_decoupling(
            apply_pauli_twirling(circuit, seed),
            target=target,
            instruction_durations=instruction_durations,
            qubits=qubits,
        )
    raise ValueError(f"Unsupported suppression mode: {suppression_mode}")
