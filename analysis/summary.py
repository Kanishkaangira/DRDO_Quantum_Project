"""Build results/csv/summary_results.csv from results/csv/raw_results.csv.

Groups raw per-run rows into one row per unique experimental configuration,
computes mean/std across the repeated runs in each group, and writes the
result to summary_results.csv. This file is fully derived from
raw_results.csv -- it is overwritten each time this module runs, not
appended to incrementally like benchmark/results_writer.py.
"""

from pathlib import Path

import pandas as pd

RAW_RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "csv" / "raw_results.csv"
SUMMARY_RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "csv" / "summary_results.csv"

# Rows matching on ALL of these are treated as repeats of the same
# experiment and get averaged together. run_number is deliberately
# excluded -- it's the one thing allowed to vary within a group.
GROUP_COLUMNS = [
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
    "shots_per_execution",
]

# Structural columns -- should not vary within a group, so just copy the
# first row's value rather than averaging.
IDENTITY_COLUMNS = [
    "original_depth",
    "evaluated_depth",
    "original_gate_count",
    "evaluated_gate_count",
]

# Averaged with both mean and std -- these are the numbers whose
# run-to-run consistency actually matters for the research conclusions.
MEAN_STD_COLUMNS = {
    "execution_time": "execution_time",
    "runtime_overhead": "runtime_overhead",
    "sampling_overhead": "sampling_overhead",
    "fidelity": "fidelity",
    "expectation": "expectation",
}

# Averaged with mean only -- structural/low-variance, std wouldn't be
# informative.
MEAN_ONLY_COLUMNS = {
    "depth_overhead": "depth_overhead",
    "gate_count_overhead": "gate_count_overhead",
    "baseline_execution_time": "baseline_execution_time",
    "ideal_expectation": "ideal_expectation",
    "expectation_absolute_error": "expectation_absolute_error",
    "total_variation_distance": "total_variation_distance",
}

REQUIRED_RAW_COLUMNS = set(
    GROUP_COLUMNS
    + IDENTITY_COLUMNS
    + [
        "execution_time",
        "runtime_overhead",
        "sampling_overhead",
        "depth_overhead",
        "gate_count_overhead",
        "baseline_execution_time",
        "ideal_expectation",
    ]
    + [
        "raw_fidelity",
        "mitigated_fidelity",
        "raw_expectation",
        "mitigated_expectation",
        "raw_expectation_absolute_error",
        "mitigated_expectation_absolute_error",
        "raw_total_variation_distance",
        "mitigated_total_variation_distance",
    ]
)

FINAL_COLUMN_ORDER = (
    GROUP_COLUMNS
    + ["n_runs"]
    + IDENTITY_COLUMNS
    + [
        "depth_overhead_mean",
        "gate_count_overhead_mean",
        "baseline_execution_time_mean",
        "execution_time_mean",
        "execution_time_std",
        "runtime_overhead_mean",
        "runtime_overhead_std",
        "sampling_overhead_mean",
        "sampling_overhead_std",
        "expectation_mean",
        "expectation_std",
        "ideal_expectation_mean",
        "expectation_absolute_error_mean",
        "fidelity_mean",
        "fidelity_std",
        "total_variation_distance_mean",
        "error_reduction_percentage",
    ]
)


def _coalesce(row: pd.Series, mitigated_column: str, raw_column: str) -> float | None:
    """Prefer the mitigated value when present, otherwise fall back to raw."""
    mitigated_value = row.get(mitigated_column, "")
    if mitigated_value not in ("", None) and not pd.isna(mitigated_value):
        return float(mitigated_value)
    raw_value = row.get(raw_column, "")
    if raw_value in ("", None) or pd.isna(raw_value):
        return None
    return float(raw_value)


def _load_raw_results(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        raise FileNotFoundError(
            f"raw_results.csv is empty or missing at {path}; run some benchmark "
            "cases first."
        )
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_columns = REQUIRED_RAW_COLUMNS - set(raw.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"raw_results.csv is missing required columns: {missing}")

    # Collapse raw_* / mitigated_* pairs into one clean column per metric,
    # per row, before any grouping happens.
    raw["fidelity"] = raw.apply(
        lambda row: _coalesce(row, "mitigated_fidelity", "raw_fidelity"), axis=1
    )
    raw["expectation"] = raw.apply(
        lambda row: _coalesce(row, "mitigated_expectation", "raw_expectation"), axis=1
    )
    raw["expectation_absolute_error"] = raw.apply(
        lambda row: _coalesce(
            row, "mitigated_expectation_absolute_error", "raw_expectation_absolute_error"
        ),
        axis=1,
    )
    raw["total_variation_distance"] = raw.apply(
        lambda row: _coalesce(
            row, "mitigated_total_variation_distance", "raw_total_variation_distance"
        ),
        axis=1,
    )

    # Numeric columns needed for averaging / identity copying.
    numeric_columns = (
        IDENTITY_COLUMNS
        + list(MEAN_STD_COLUMNS)
        + list(MEAN_ONLY_COLUMNS)
    )
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    return raw


def _build_group_row(group_key: tuple, group: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = dict(zip(GROUP_COLUMNS, group_key))
    row["n_runs"] = len(group)

    for column in IDENTITY_COLUMNS:
        row[column] = group[column].iloc[0]

    for source_column in MEAN_STD_COLUMNS:
        row[f"{source_column}_mean"] = group[source_column].mean()
        row[f"{source_column}_std"] = group[source_column].std(ddof=1)

    for source_column in MEAN_ONLY_COLUMNS:
        row[f"{source_column}_mean"] = group[source_column].mean()

    return row


def _attach_error_reduction(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute error_reduction_percentage relative to each group's baseline."""
    baseline_lookup: dict[tuple, float] = {}
    for _, row in summary.iterrows():
        if (
            row["experimental_condition"] == "Baseline"
            and row["suppression_mode"] == "none"
            and row["mitigation_technique"] == ""
        ):
            key = (
                row["circuit"],
                row["qubits"],
                row["circuit_build_parameters"],
                row["backend"],
                row["noise_model"],
                row["noise_strength"],
                row["shots_per_execution"],
            )
            baseline_lookup[key] = 1.0 - row["fidelity_mean"]

    def error_reduction(row: pd.Series) -> float | None:
        if (
            row["experimental_condition"] == "Baseline"
            and row["suppression_mode"] == "none"
            and row["mitigation_technique"] == ""
        ):
            return 0.0
        key = (
            row["circuit"],
            row["qubits"],
            row["circuit_build_parameters"],
            row["backend"],
            row["noise_model"],
            row["noise_strength"],
            row["shots_per_execution"],
        )
        baseline_error = baseline_lookup.get(key)
        if baseline_error is None or baseline_error == 0:
            return None
        this_error = 1.0 - row["fidelity_mean"]
        return (baseline_error - this_error) / baseline_error * 100.0

    summary["error_reduction_percentage"] = summary.apply(error_reduction, axis=1)
    return summary


def build_summary_results(
    raw_path: Path = RAW_RESULTS_PATH, summary_path: Path = SUMMARY_RESULTS_PATH
) -> pd.DataFrame:
    """Read raw_results.csv, aggregate by configuration, write summary_results.csv."""
    raw = _load_raw_results(raw_path)

    grouped_rows = [
        _build_group_row(group_key, group)
        for group_key, group in raw.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    ]
    summary = pd.DataFrame(grouped_rows)
    summary = _attach_error_reduction(summary)
    summary = summary[FINAL_COLUMN_ORDER]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary


if __name__ == "__main__":
    result = build_summary_results()
    print(f"Wrote {len(result)} summary row(s) to {SUMMARY_RESULTS_PATH}")
