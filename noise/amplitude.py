"""Amplitude-damping gate-noise model."""

from qiskit_aer.noise import NoiseModel, amplitude_damping_error

from . import add_gate_noise, validate_noise_strength


def build_amplitude_noise_model(noise_strength: float) -> NoiseModel:
    """Build an Aer noise model for energy-relaxation errors after gates."""
    validate_noise_strength(noise_strength)
    return add_gate_noise(NoiseModel(), amplitude_damping_error(noise_strength))
