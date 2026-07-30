"""Ring-MaxCut QAOA benchmark circuit."""

from math import pi

from qiskit import QuantumCircuit


def build_qaoa_circuit(
    num_qubits: int,
    repetitions: int,
    gamma: float = pi / 4,
    beta: float = pi / 8,
) -> QuantumCircuit:
    """Build a deterministic QAOA circuit for MaxCut on a ring graph."""
    if num_qubits < 2:
        raise ValueError("QAOA circuits require at least 2 qubits.")
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1.")

    circuit = QuantumCircuit(num_qubits, num_qubits, name=f"qaoa_{num_qubits}")
    ring_edges = [(qubit, qubit + 1) for qubit in range(num_qubits - 1)]
    if num_qubits > 2:
        ring_edges.append((num_qubits - 1, 0))

    circuit.h(range(num_qubits))
    for _ in range(repetitions):
        for first_qubit, second_qubit in ring_edges:
            circuit.cx(first_qubit, second_qubit)
            circuit.rz(2 * gamma, second_qubit)
            circuit.cx(first_qubit, second_qubit)
        for qubit_index in range(num_qubits):
            circuit.rx(2 * beta, qubit_index)

    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit
