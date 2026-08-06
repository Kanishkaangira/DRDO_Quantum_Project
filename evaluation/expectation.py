"""Expectation-value calculations for measured circuit outcomes."""

from collections.abc import Mapping

def calculate_z_parity_expectation(counts: Mapping[str, int]) -> float:
    """Calculate the all-qubit Z-parity expectation from measurement counts."""
    total_shots = sum(counts.values())
    if total_shots <= 0:
        raise ValueError("counts must contain at least one shot.")

    weighted_sum = 0
    for bitstring, count in counts.items():
        parity = bitstring.replace(" ", "").count("1") % 2
        weighted_sum += count if parity == 0 else -count
    return weighted_sum / total_shots


def calculate_expectation_absolute_error(
    observed_expectation: float, ideal_expectation: float
) -> float:
    """Calculate absolute error between measured and ideal expectations."""
    return abs(observed_expectation - ideal_expectation)


def calculate_ring_maxcut_expectation(counts: Mapping[str, int]) -> float:
    """Calculate the QAOA ring-MaxCut objective used by the QAOA circuit builder."""
    total_shots = sum(counts.values())
    if total_shots <= 0:
        raise ValueError("counts must contain at least one shot.")

    weighted_cost = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        if len(bits) < 2:
            raise ValueError("Ring-MaxCut evaluation requires at least two qubits.")
        cut_edges = sum(
            first_bit != second_bit for first_bit, second_bit in zip(bits, bits[1:])
        )
        if len(bits) > 2:
            cut_edges += bits[-1] != bits[0]
        weighted_cost += count * cut_edges
    return weighted_cost / total_shots


def calculate_ring_ising_energy(
    counts: Mapping[str, int], *, coupling: float, field: float
) -> float:
    """Calculate a diagonal ring-Ising VQE energy from Z-basis samples."""
    total_shots = sum(counts.values())
    if total_shots <= 0:
        raise ValueError("counts must contain at least one shot.")

    weighted_energy = 0.0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        if len(bits) < 2:
            raise ValueError("Ring-Ising evaluation requires at least two qubits.")
        spins = [1 if bit == "0" else -1 for bit in bits]
        interaction = sum(
            spins[index] * spins[(index + 1) % len(spins)]
            for index in range(len(spins))
        )
        weighted_energy += count * (coupling * interaction + field * sum(spins))
    return weighted_energy / total_shots
