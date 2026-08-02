"""Append one ``BenchmarkResult`` from ``runner.py`` to ``results/csv/raw_results.csv``.

Each call writes one row per mitigation technique, or one row with a blank
``mitigation_technique`` for Baseline and Suppression-Only results. This module
does not build ``summary_results.csv``, ``best_configurations.csv``, or
``results/tables/`` comparison output; those are separate future steps built
from this file's output.
"""

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import BenchmarkResult


RAW_RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "csv" / "raw_results.csv"

RAW_RESULT_COLUMNS = (
    "experiment_id",
    "circuit",
    "qubits",
    "backend",
    "noise_model",
    "noise_strength",
    "experimental_condition",
    "suppression_mode",
    "mitigation_technique",
    "original_depth",
    "suppressed_depth",
    "shots",
    "run_number",
    "execution_time",
    "fidelity",
    "expectation",
    "runtime_overhead",
    "sampling_overhead",
)

_CONDITION_LABELS = {
    "baseline": "Baseline",
    "suppression_only": "Suppression Only",
    "mitigation_only": "Mitigation Only",
    "suppression_and_mitigation": "Suppression + Mitigation",
}


def _existing_experiment_numbers(path: Path) -> Iterable[int]:
    with path.open(newline="", encoding="utf-8") as file_handle:
        for row in csv.DictReader(file_handle):
            experiment_id = row.get("experiment_id", "")
            if experiment_id.startswith("EXP") and experiment_id[3:].isdigit():
                yield int(experiment_id[3:])


def _prepare_file(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return 1
    if not path.read_text(encoding="utf-8").strip():
        path.write_text("", encoding="utf-8")
        return 1

    with path.open(newline="", encoding="utf-8") as file_handle:
        columns = next(csv.reader(file_handle), [])
    if tuple(columns) != RAW_RESULT_COLUMNS:
        raise ValueError("raw_results.csv does not match the locked column order.")
    return max(_existing_experiment_numbers(path), default=0) + 1


def append_benchmark_result(
    result: "BenchmarkResult", path: Path = RAW_RESULTS_PATH
) -> list[str]:
    """Append one benchmark result as its locked-schema raw-result row or rows."""
    next_experiment_number = _prepare_file(path)
    mitigation_techniques = tuple(result.mitigated_values) or ("",)
    wrote_header = not path.exists() or not path.read_text(encoding="utf-8").strip()
    experiment_ids: list[str] = []

    with path.open("a", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=RAW_RESULT_COLUMNS)
        if wrote_header:
            writer.writeheader()
        for offset, mitigation_technique in enumerate(mitigation_techniques):
            experiment_id = f"EXP{next_experiment_number + offset:06d}"
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "circuit": result.case.circuit_name,
                    "qubits": result.case.num_qubits,
                    "backend": result.case.backend_name,
                    "noise_model": result.case.noise_name or "",
                    "noise_strength": (
                        result.case.noise_strength
                        if result.case.noise_strength is not None
                        else ""
                    ),
                    "experimental_condition": _CONDITION_LABELS[
                        result.case.condition_name
                    ],
                    "suppression_mode": result.case.suppression_mode,
                    "mitigation_technique": mitigation_technique,
                    "original_depth": result.original_depth,
                    "suppressed_depth": result.evaluated_depth,
                    "shots": result.case.shots,
                    "run_number": result.case.run_index + 1,
                    "execution_time": result.execution_time,
                    "fidelity": result.fidelity,
                    "expectation": result.raw_expectation,
                    "runtime_overhead": result.runtime_overhead,
                    "sampling_overhead": result.sampling_overhead,
                }
            )
            experiment_ids.append(experiment_id)
    return experiment_ids
