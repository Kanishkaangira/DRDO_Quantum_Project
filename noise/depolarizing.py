"""Depolarizing gate-noise model."""

from qiskit_aer.noise import NoiseModel, depolarizing_error

from . import add_gate_noise, validate_noise_strength


def build_depolarizing_noise_model(noise_strength: float) -> NoiseModel:
    """Build an Aer noise model for symmetric gate depolarization."""
    validate_noise_strength(noise_strength)
    return add_gate_noise(NoiseModel(), depolarizing_error(noise_strength, 1))
