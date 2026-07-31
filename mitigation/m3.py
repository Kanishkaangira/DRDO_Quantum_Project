"""Measurement-mitigation correction using a supplied assignment matrix."""

from collections.abc import Mapping

import numpy as np


def mitigate_m3(
    counts: Mapping[str, int], calibration_matrix: np.ndarray
) -> dict[str, float]:
    """Return corrected outcome probabilities for one executed circuit."""
    if not counts:
        raise ValueError("counts must not be empty.")

    normalized_counts = {bitstring.replace(" ", ""): count for bitstring, count in counts.items()}
    bitstring_length = len(next(iter(normalized_counts)))
    if any(len(bitstring) != bitstring_length for bitstring in normalized_counts):
        raise ValueError("All count bitstrings must have the same length.")

    dimension = 2**bitstring_length
    if calibration_matrix.shape != (dimension, dimension):
        raise ValueError(
            f"calibration_matrix must have shape ({dimension}, {dimension})."
        )

    total_shots = sum(normalized_counts.values())
    if total_shots <= 0:
        raise ValueError("counts must contain at least one shot.")

    observed = np.zeros(dimension)
    for bitstring, count in normalized_counts.items():
        observed[int(bitstring, 2)] = count / total_shots

    try:
        corrected = np.linalg.solve(calibration_matrix, observed)
    except np.linalg.LinAlgError as error:
        raise ValueError("calibration_matrix must be invertible.") from error

    corrected = np.clip(corrected, 0.0, None)
    corrected_total = corrected.sum()
    if corrected_total == 0:
        raise ValueError("Mitigation produced no valid probability mass.")
    corrected /= corrected_total
    return {
        format(outcome, f"0{bitstring_length}b"): float(probability)
        for outcome, probability in enumerate(corrected)
    }
