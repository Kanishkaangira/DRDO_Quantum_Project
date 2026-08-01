"""IBM Runtime execution adapter."""

from collections.abc import Mapping
from importlib import import_module
from typing import Any

from qiskit import QuantumCircuit, transpile


def execute_ibm(
    circuit: QuantumCircuit,
    shots: int,
    *,
    backend: Any,
) -> Mapping[str, int]:
    """Execute one measured circuit through IBM Runtime's sampler primitive."""
    if shots < 1:
        raise ValueError("shots must be at least 1.")
    if backend is None:
        raise ValueError("An IBM backend must be provided.")

    try:
        sampler_class = import_module("qiskit_ibm_runtime").SamplerV2
    except ImportError as error:
        raise RuntimeError(
            "IBM Runtime execution requires the qiskit-ibm-runtime package."
        ) from error

    compiled_circuit = transpile(circuit, backend, optimization_level=0)
    sampler = sampler_class(mode=backend)
    publication_result = sampler.run([compiled_circuit], shots=shots).result()[0]
    data = publication_result.data
    for field_name in ("meas", "c"):
        register_data = getattr(data, field_name, None)
        if hasattr(register_data, "get_counts"):
            return register_data.get_counts()
    for field_name in vars(data):
        register_data = getattr(data, field_name)
        if hasattr(register_data, "get_counts"):
            return register_data.get_counts()
    raise RuntimeError("IBM Runtime sampler result did not contain measurement counts.")
