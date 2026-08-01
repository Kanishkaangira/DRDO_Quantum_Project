"""Backend dispatcher for one benchmark circuit execution."""

from collections.abc import Mapping
from typing import Any

from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from .ibm_runtime import execute_ibm
from .simulator import execute_aer


def execute_circuit(
    circuit: QuantumCircuit,
    backend_name: str,
    shots: int,
    *,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
    ibm_backend: Any = None,
) -> Mapping[str, int]:
    """Execute one circuit on the selected configured backend."""
    if backend_name == "aer":
        return execute_aer(circuit, shots, noise_model=noise_model, seed=seed)
    if backend_name == "ibm":
        if noise_model is not None:
            raise ValueError("Injected noise models are supported only by the Aer backend.")
        return execute_ibm(circuit, shots, backend=ibm_backend)
    raise ValueError(f"Unsupported backend: {backend_name}")
