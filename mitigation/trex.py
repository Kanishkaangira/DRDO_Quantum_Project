"""Twirl-based expectation-value readout correction."""


def mitigate_trex(raw_expectation: float, calibration_expectation: float) -> float:
    """Correct one expectation value using its TREX calibration response."""
    if calibration_expectation == 0:
        raise ValueError("calibration_expectation must not be zero.")
    return raw_expectation / calibration_expectation
