"""Hardware-efficient VQE ansatz benchmark circuit."""

from math import pi

from qiskit import QuantumCircuit


def build_vqe_circuit(num_qubits: int, ansatz_repetitions: int) -> QuantumCircuit:
    """Build a deterministic, hardware-efficient variational ansatz circuit."""
    if num_qubits < 2:
        raise ValueError("VQE circuits require at least 2 qubits.")
    if ansatz_repetitions < 1:
        raise ValueError("ansatz_repetitions must be at least 1.")

    circuit = QuantumCircuit(num_qubits, num_qubits, name=f"vqe_{num_qubits}")
    for layer_index in range(ansatz_repetitions):
        for qubit_index in range(num_qubits):
            angle = pi * (layer_index + qubit_index + 1) / (num_qubits + 1)
            circuit.ry(angle, qubit_index)
            circuit.rz(angle / 2, qubit_index)

        for control_qubit in range(num_qubits - 1):
            circuit.cx(control_qubit, control_qubit + 1)

    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit
