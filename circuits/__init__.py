"""Original circuit builders used by the benchmark parameter sweep."""

from collections.abc import Callable

from qiskit import QuantumCircuit

from .bell import build_bell_circuit
from .ghz import build_ghz_circuit
from .qaoa import build_qaoa_circuit
from .qft import build_qft_circuit
from .vqe import build_vqe_circuit

CircuitBuilder = Callable[..., QuantumCircuit]

CIRCUIT_BUILDERS: dict[str, CircuitBuilder] = {
    "bell": build_bell_circuit,
    "ghz": build_ghz_circuit,
    "qft": build_qft_circuit,
    "vqe": build_vqe_circuit,
    "qaoa": build_qaoa_circuit,
}


def build_circuit(
    circuit_name: str, num_qubits: int, **build_parameters: object
) -> QuantumCircuit:
    """Build one measured original circuit for a benchmark loop iteration."""
    try:
        builder = CIRCUIT_BUILDERS[circuit_name]
    except KeyError as error:
        supported = ", ".join(CIRCUIT_BUILDERS)
        raise ValueError(
            f"Unsupported circuit: {circuit_name}. Supported circuits: {supported}."
        ) from error
    return builder(num_qubits=num_qubits, **build_parameters)


__all__ = [
    "CIRCUIT_BUILDERS",
    "build_bell_circuit",
    "build_circuit",
    "build_ghz_circuit",
    "build_qaoa_circuit",
    "build_qft_circuit",
    "build_vqe_circuit",
]
