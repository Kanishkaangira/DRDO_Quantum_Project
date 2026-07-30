"""Symmetric measurement readout-noise model."""

from qiskit_aer.noise import NoiseModel, ReadoutError

from . import validate_noise_strength


def build_readout_noise_model(noise_strength: float) -> NoiseModel:
    """Build an Aer model with independent symmetric bit-readout errors."""
    validate_noise_strength(noise_strength)
    error = ReadoutError(
        [[1 - noise_strength, noise_strength], [noise_strength, 1 - noise_strength]]
    )
    noise_model = NoiseModel()
    noise_model.add_all_qubit_readout_error(error)
    return noise_model
