"""GHZ-state benchmark circuit."""

from qiskit import QuantumCircuit


def build_ghz_circuit(num_qubits: int) -> QuantumCircuit:
    """Build an ``n``-qubit GHZ circuit with final measurements."""
    if num_qubits < 2:
        raise ValueError("GHZ circuits require at least 2 qubits.")

    circuit = QuantumCircuit(num_qubits, num_qubits, name=f"ghz_{num_qubits}")
    circuit.h(0)
    for target_qubit in range(1, num_qubits):
        circuit.cx(0, target_qubit)
    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit
