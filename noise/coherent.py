"""Coherent Z-rotation gate-noise model."""

from cmath import exp

from qiskit_aer.noise import NoiseModel, coherent_unitary_error

from . import add_gate_noise, validate_noise_strength


def build_coherent_noise_model(noise_strength: float) -> NoiseModel:
    """Build an Aer model with a consistent unintended Z rotation after gates."""
    validate_noise_strength(noise_strength)
    half_angle = noise_strength / 2
    unitary = [[exp(-1j * half_angle), 0], [0, exp(1j * half_angle)]]
    return add_gate_noise(NoiseModel(), coherent_unitary_error(unitary))
