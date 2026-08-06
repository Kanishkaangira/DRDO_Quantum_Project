"""Append fully reproducible ``BenchmarkResult`` rows to ``raw_results.csv``.

Each mitigation technique receives its own row containing the raw execution,
its actual mitigation output, and the configuration and auxiliary data used to
produce it. Baseline and Suppression-Only results receive one row with blank
mitigation fields. This module writes only ``results/csv/raw_results.csv``;
summary CSVs, best-configuration selection, and comparison tables remain
separate analysis steps.
"""

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.settings import SEED, SEED_STRIDE

if TYPE_CHECKING:
    from .runner import BenchmarkResult


RAW_RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "csv" / "raw_results.csv"

RAW_RESULT_COLUMNS = (
    "experiment_id",
    "circuit",
    "qubits",
    "circuit_build_parameters",
    "backend",
    "noise_model",
    "noise_strength",
    "experimental_condition",
    "suppression_mode",
    "suppression_parameters",
    "mitigation_technique",
    "mitigation_parameters",
    "run_number",
    "run_seed",
    "shots_per_execution",
    "total_shots",
    "original_depth",
    "evaluated_depth",
    "original_gate_count",
    "evaluated_gate_count",
    "depth_overhead",
    "gate_count_overhead",
    "baseline_execution_time",
    "execution_time",
    "runtime_overhead",
    "sampling_overhead",
    "raw_expectation",
    "mitigated_expectation",
    "ideal_expectation",
    "raw_expectation_absolute_error",
    "mitigated_expectation_absolute_error",
    "raw_fidelity",
    "mitigated_fidelity",
    "raw_total_variation_distance",
    "mitigated_total_variation_distance",
    "raw_counts",
    "ideal_counts",
    "mitigation_result",
    "mitigation_context",
)

_LEGACY_RAW_RESULT_COLUMNS = (
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

_PREVIOUS_RAW_RESULT_COLUMNS = tuple(
    column
    for column in RAW_RESULT_COLUMNS
    if column not in {"suppression_parameters", "mitigation_parameters"}
)

_CONDITION_LABELS = {
    "baseline": "Baseline",
    "suppression_only": "Suppression Only",
    "mitigation_only": "Mitigation Only",
    "suppression_and_mitigation": "Suppression + Mitigation",
}


def _as_json(value: object) -> str:
    """Serialize structured per-experiment data deterministically for CSV."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _existing_experiment_numbers(path: Path) -> Iterable[int]:
    with path.open(newline="", encoding="utf-8") as file_handle:
        for row in csv.DictReader(file_handle):
            experiment_id = row.get("experiment_id", "")
            if experiment_id.startswith("EXP") and experiment_id[3:].isdigit():
                yield int(experiment_id[3:])


def _legacy_row(row: Mapping[str, str]) -> dict[str, str]:
    """Retain legacy rows while leaving unavailable historical fields blank."""
    shots = row["shots"]
    return {
        "experiment_id": row["experiment_id"],
        "circuit": row["circuit"],
        "qubits": row["qubits"],
        "circuit_build_parameters": "",
        "backend": row["backend"],
        "noise_model": row["noise_model"],
        "noise_strength": row["noise_strength"],
        "experimental_condition": row["experimental_condition"],
        "suppression_mode": row["suppression_mode"],
        "suppression_parameters": "",
        "mitigation_technique": row["mitigation_technique"],
        "mitigation_parameters": "",
        "run_number": row["run_number"],
        "run_seed": "",
        "shots_per_execution": shots,
        "total_shots": "",
        "original_depth": row["original_depth"],
        "evaluated_depth": row["suppressed_depth"],
        "original_gate_count": "",
        "evaluated_gate_count": "",
        "depth_overhead": "",
        "gate_count_overhead": "",
        "baseline_execution_time": "",
        "execution_time": row["execution_time"],
        "runtime_overhead": row["runtime_overhead"],
        "sampling_overhead": row["sampling_overhead"],
        "raw_expectation": row["expectation"],
        "mitigated_expectation": "",
        "ideal_expectation": "",
        "raw_expectation_absolute_error": "",
        "mitigated_expectation_absolute_error": "",
        "raw_fidelity": row["fidelity"],
        "mitigated_fidelity": "",
        "raw_total_variation_distance": "",
        "mitigated_total_variation_distance": "",
        "raw_counts": "",
        "ideal_counts": "",
        "mitigation_result": "",
        "mitigation_context": "",
    }


def _migrate_legacy_file(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        existing_columns = tuple(reader.fieldnames or ())
        if existing_columns == _LEGACY_RAW_RESULT_COLUMNS:
            migrated_rows = [_legacy_row(row) for row in reader]
        elif existing_columns == _PREVIOUS_RAW_RESULT_COLUMNS:
            migrated_rows = [
                {column: row.get(column, "") for column in RAW_RESULT_COLUMNS}
                for row in reader
            ]
        else:
            raise ValueError("raw_results.csv does not match a supported result schema.")

    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=RAW_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(migrated_rows)


def _prepare_file(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return 1

    with path.open(newline="", encoding="utf-8") as file_handle:
        columns = tuple(next(csv.reader(file_handle), []))
    if columns in {_LEGACY_RAW_RESULT_COLUMNS, _PREVIOUS_RAW_RESULT_COLUMNS}:
        _migrate_legacy_file(path)
    elif columns != RAW_RESULT_COLUMNS:
        raise ValueError("raw_results.csv does not match a supported result schema.")
    return max(_existing_experiment_numbers(path), default=0) + 1


def _row_for_result(
    result: "BenchmarkResult", experiment_id: str, mitigation_technique: str
) -> dict[str, object]:
    has_mitigation = bool(mitigation_technique)
    mitigated_value = (
        result.mitigated_values[mitigation_technique] if has_mitigation else ""
    )
    execution_time = (
        result.mitigation_execution_times[mitigation_technique]
        if has_mitigation
        else result.execution_time
    )
    total_shots = (
        result.mitigation_total_shots[mitigation_technique]
        if has_mitigation
        else result.total_shots
    )
    runtime_overhead = (
        result.mitigation_runtime_overheads[mitigation_technique]
        if has_mitigation
        else result.runtime_overhead
    )
    sampling_overhead = (
        result.mitigation_sampling_overheads[mitigation_technique]
        if has_mitigation
        else result.sampling_overhead
    )
    return {
        "experiment_id": experiment_id,
        "circuit": result.case.circuit_name,
        "qubits": result.case.num_qubits,
        "circuit_build_parameters": _as_json(result.case.build_parameters),
        "backend": result.case.backend_name,
        "noise_model": result.case.noise_name or "",
        "noise_strength": result.case.noise_strength if result.case.noise_strength is not None else "",
        "experimental_condition": _CONDITION_LABELS[result.case.condition_name],
        "suppression_mode": result.case.suppression_mode,
        "suppression_parameters": _as_json(result.case.suppression_parameters),
        "mitigation_technique": mitigation_technique,
        "mitigation_parameters": (
            _as_json(result.case.mitigation_parameters[mitigation_technique])
            if has_mitigation
            else ""
        ),
        "run_number": result.case.run_index + 1,
        "run_seed": SEED + result.case.run_index * SEED_STRIDE,
        "shots_per_execution": result.case.shots,
        "total_shots": total_shots,
        "original_depth": result.original_depth,
        "evaluated_depth": result.evaluated_depth,
        "original_gate_count": result.original_gate_count,
        "evaluated_gate_count": result.evaluated_gate_count,
        "depth_overhead": result.depth_overhead,
        "gate_count_overhead": result.gate_count_overhead,
        "baseline_execution_time": result.baseline_execution_time,
        "execution_time": execution_time,
        "runtime_overhead": runtime_overhead,
        "sampling_overhead": sampling_overhead,
        "raw_expectation": result.raw_expectation,
        "mitigated_expectation": (
            result.mitigated_expectations[mitigation_technique] if has_mitigation else ""
        ),
        "ideal_expectation": result.ideal_expectation,
        "raw_expectation_absolute_error": result.expectation_absolute_error,
        "mitigated_expectation_absolute_error": (
            result.mitigated_expectation_absolute_errors[mitigation_technique]
            if has_mitigation
            else ""
        ),
        "raw_fidelity": result.fidelity,
        "mitigated_fidelity": (
            result.mitigated_fidelities[mitigation_technique] if has_mitigation else ""
        ),
        "raw_total_variation_distance": result.total_variation_distance,
        "mitigated_total_variation_distance": (
            result.mitigated_total_variation_distances[mitigation_technique]
            if has_mitigation
            else ""
        ),
        "raw_counts": _as_json(result.raw_counts),
        "ideal_counts": _as_json(result.ideal_counts),
        "mitigation_result": _as_json(mitigated_value) if has_mitigation else "",
        "mitigation_context": (
            _as_json(result.mitigation_contexts[mitigation_technique])
            if has_mitigation
            else ""
        ),
    }


def append_benchmark_result(
    result: "BenchmarkResult", path: Path = RAW_RESULTS_PATH
) -> list[str]:
    """Append one result as complete raw rows, one for each mitigation output."""
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
            writer.writerow(_row_for_result(result, experiment_id, mitigation_technique))
            experiment_ids.append(experiment_id)
    return experiment_ids
