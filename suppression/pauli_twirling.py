"""Pauli twirling transformation for CX gates."""

from random import Random

from qiskit import QuantumCircuit


def _append_pauli(circuit: QuantumCircuit, pauli: str, qubit: object) -> None:
    if pauli == "X":
        circuit.x(qubit)
    elif pauli == "Y":
        circuit.y(qubit)
    elif pauli == "Z":
        circuit.z(qubit)


def _conjugate_cx_paulis(control: str, target: str) -> tuple[str, str]:
    pauli_bits = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
    pauli_labels = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}
    control_x, control_z = pauli_bits[control]
    target_x, target_z = pauli_bits[target]
    return (
        pauli_labels[(control_x, control_z ^ target_z)],
        pauli_labels[(control_x ^ target_x, target_z)],
    )


def apply_pauli_twirling(circuit: QuantumCircuit, seed: int) -> QuantumCircuit:
    """Return a functionally equivalent circuit with twirled CX gates."""
    generator = Random(seed)
    twirled = circuit.copy_empty_like(name=f"{circuit.name}_pauli_twirl")

    for instruction in circuit.data:
        if instruction.operation.name != "cx":
            twirled.append(instruction.operation, instruction.qubits, instruction.clbits)
            continue

        control_pauli = generator.choice(("I", "X", "Y", "Z"))
        target_pauli = generator.choice(("I", "X", "Y", "Z"))
        _append_pauli(twirled, control_pauli, instruction.qubits[0])
        _append_pauli(twirled, target_pauli, instruction.qubits[1])
        twirled.append(instruction.operation, instruction.qubits, instruction.clbits)
        output_control, output_target = _conjugate_cx_paulis(
            control_pauli, target_pauli
        )
        _append_pauli(twirled, output_control, instruction.qubits[0])
        _append_pauli(twirled, output_target, instruction.qubits[1])

    return twirled
