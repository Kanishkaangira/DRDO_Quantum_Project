"""Bell-state benchmark circuit."""

from qiskit import QuantumCircuit


def build_bell_circuit(num_qubits: int) -> QuantumCircuit:
    """Build the fixed two-qubit Bell-state circuit with final measurements."""
    if num_qubits != 2:
        raise ValueError("Bell circuits require exactly 2 qubits.")

    circuit = QuantumCircuit(2, 2, name="bell")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(range(2), range(2))
    return circuit
