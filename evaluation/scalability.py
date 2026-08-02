"""Scalability summaries across configured qubit counts."""

from collections.abc import Iterable, Mapping
from statistics import fmean


def summarize_by_qubit_count(
    records: Iterable[Mapping[str, float]], metric_name: str
) -> dict[int, float]:
    """Average one metric for every qubit count represented in result records."""
    grouped_values: dict[int, list[float]] = {}
    for record in records:
        try:
            qubit_count = int(record["num_qubits"])
            metric_value = float(record[metric_name])
        except KeyError as error:
            raise ValueError(
                f"Each record must include num_qubits and {metric_name}."
            ) from error
        grouped_values.setdefault(qubit_count, []).append(metric_value)
    if not grouped_values:
        raise ValueError("records must not be empty.")
    return {
        qubit_count: fmean(values)
        for qubit_count, values in sorted(grouped_values.items())
    }
