"""Backend-aware dynamical-decoupling transformation."""

from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate
from qiskit.transpiler import InstructionDurations, PassManager, Target
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling


def apply_dynamical_decoupling(
    circuit: QuantumCircuit,
    *,
    target: Target | None = None,
    instruction_durations: InstructionDurations | None = None,
    qubits: list[int] | None = None,
) -> QuantumCircuit:
    """Schedule a circuit and insert an X-X DD sequence into eligible idle windows."""
    if target is None and instruction_durations is None:
        raise ValueError(
            "Dynamical decoupling requires a backend target or instruction durations."
        )

    if target is not None:
        scheduler = ALAPScheduleAnalysis(target=target)
        padding = PadDynamicalDecoupling(
            target=target, dd_sequence=[XGate(), XGate()], qubits=qubits
        )
    else:
        scheduler = ALAPScheduleAnalysis(durations=instruction_durations)
        padding = PadDynamicalDecoupling(
            durations=instruction_durations,
            dd_sequence=[XGate(), XGate()],
            qubits=qubits,
        )

    return PassManager([scheduler, padding]).run(circuit)
