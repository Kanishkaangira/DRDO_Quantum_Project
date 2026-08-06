"""Backend-aware dynamical-decoupling transformation."""

from qiskit import QuantumCircuit
from collections.abc import Sequence

from qiskit.circuit import Gate
from qiskit.circuit.library import XGate
from qiskit.transpiler import InstructionDurations, PassManager, Target
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling


def apply_dynamical_decoupling(
    circuit: QuantumCircuit,
    *,
    target: Target | None = None,
    instruction_durations: InstructionDurations | None = None,
    qubits: list[int] | None = None,
    dd_sequence: Sequence[Gate] | None = None,
) -> QuantumCircuit:
    """Schedule a circuit and insert an X-X DD sequence into eligible idle windows."""
    if target is None and instruction_durations is None:
        raise ValueError(
            "Dynamical decoupling requires a backend target or instruction durations."
        )

    sequence = list(dd_sequence) if dd_sequence is not None else [XGate(), XGate()]
    if not sequence:
        raise ValueError("Dynamical decoupling requires a non-empty DD sequence.")

    if target is not None:
        scheduler = ALAPScheduleAnalysis(target=target)
        padding = PadDynamicalDecoupling(
            target=target, dd_sequence=sequence, qubits=qubits
        )
    else:
        scheduler = ALAPScheduleAnalysis(durations=instruction_durations)
        padding = PadDynamicalDecoupling(
            durations=instruction_durations,
            dd_sequence=sequence,
            qubits=qubits,
        )

    return PassManager([scheduler, padding]).run(circuit)
