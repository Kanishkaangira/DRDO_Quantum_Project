"""Quantum Fourier transform benchmark circuit."""

from math import pi

from qiskit import QuantumCircuit


def build_qft_circuit(
    num_qubits: int,
    approximation_degree: int,
    do_swaps: bool,
) -> QuantumCircuit:
    """Build an executable QFT circuit with optional small-angle omission."""
    if num_qubits < 2:
        raise ValueError("QFT circuits require at least 2 qubits.")
    if approximation_degree < 0:
        raise ValueError("approximation_degree must be non-negative.")

    circuit = QuantumCircuit(num_qubits, num_qubits, name=f"qft_{num_qubits}")
    for target_qubit in range(num_qubits - 1, -1, -1):
        circuit.h(target_qubit)
        for control_qubit in range(target_qubit - 1, -1, -1):
            distance = target_qubit - control_qubit
            if distance <= num_qubits - 1 - approximation_degree:
                circuit.cp(pi / (2**distance), control_qubit, target_qubit)

    if do_swaps:
        for left_qubit in range(num_qubits // 2):
            circuit.swap(left_qubit, num_qubits - left_qubit - 1)

    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit
