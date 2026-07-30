"""Aer noise-model builders used by the benchmark parameter sweep."""

from collections.abc import Callable

from qiskit_aer.noise import NoiseModel, QuantumError

SINGLE_QUBIT_GATES = ("h", "rx", "ry", "rz")
TWO_QUBIT_GATES = ("cx", "cp", "swap")


def validate_noise_strength(noise_strength: float) -> None:
    """Validate a probability-like noise-strength value from configuration."""
    if not 0 <= noise_strength <= 1:
        raise ValueError("noise_strength must be between 0 and 1.")


def add_gate_noise(noise_model: NoiseModel, single_qubit_error: QuantumError) -> NoiseModel:
    """Attach a one-qubit error and its two-qubit counterpart to benchmark gates."""
    noise_model.add_all_qubit_quantum_error(single_qubit_error, SINGLE_QUBIT_GATES)
    noise_model.add_all_qubit_quantum_error(
        single_qubit_error.tensor(single_qubit_error), TWO_QUBIT_GATES
    )
    return noise_model

from .amplitude import build_amplitude_noise_model
from .coherent import build_coherent_noise_model
from .depolarizing import build_depolarizing_noise_model
from .phase import build_phase_noise_model
from .readout import build_readout_noise_model

NoiseBuilder = Callable[[float], NoiseModel]

NOISE_BUILDERS: dict[str, NoiseBuilder] = {
    "depolarizing": build_depolarizing_noise_model,
    "amplitude": build_amplitude_noise_model,
    "phase": build_phase_noise_model,
    "readout": build_readout_noise_model,
    "coherent": build_coherent_noise_model,
}


def build_noise_model(noise_name: str, noise_strength: float) -> NoiseModel:
    """Build one Aer noise model for a selected configured noise strength."""
    try:
        builder = NOISE_BUILDERS[noise_name]
    except KeyError as error:
        supported = ", ".join(NOISE_BUILDERS)
        raise ValueError(
            f"Unsupported noise model: {noise_name}. Supported noise models: {supported}."
        ) from error
    return builder(noise_strength)


__all__ = [
    "NOISE_BUILDERS",
    "build_amplitude_noise_model",
    "build_coherent_noise_model",
    "build_depolarizing_noise_model",
    "build_noise_model",
    "build_phase_noise_model",
    "build_readout_noise_model",
]
