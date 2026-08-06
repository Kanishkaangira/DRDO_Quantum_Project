"""Build results/csv/best_configurations.csv from results/csv/summary_results.csv.

Ranks every suppression+mitigation combination within each scenario
(circuit, qubits, noise, shots) against the others tried for that same
scenario, and writes every ranked row -- not just the single winner --
so the full accuracy-vs-overhead trade-off picture survives into this
file. This file is fully derived from summary_results.csv -- it is
overwritten each time this module runs, not appended to incrementally.
"""

from pathlib import Path

import pandas as pd

SUMMARY_RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "csv" / "summary_results.csv"
BEST_CONFIGURATIONS_PATH = Path(__file__).resolve().parents[1] / "results" / "csv" / "best_configurations.csv"

# Rows matching on all of these belong to the same "scenario" -- every
# suppression+mitigation combination tried for this scenario gets ranked
# against every other one in the same group.
SCENARIO_COLUMNS = [
    "circuit",
    "qubits",
    "circuit_build_parameters",
    "backend",
    "noise_model",
    "noise_strength",
    "shots_per_execution",
]

# Carried straight through from summary_results.csv, unchanged.
PASSTHROUGH_COLUMNS = [
    "suppression_mode",
    "mitigation_technique",
    "n_runs",
    "original_depth",
    "evaluated_depth",
    "original_gate_count",
    "evaluated_gate_count",
    "depth_overhead_mean",
    "gate_count_overhead_mean",
    "execution_time_mean",
    "runtime_overhead_mean",
    "sampling_overhead_mean",
    "fidelity_mean",
    "fidelity_std",
    "error_reduction_percentage",
]

REQUIRED_SUMMARY_COLUMNS = set(SCENARIO_COLUMNS + PASSTHROUGH_COLUMNS + ["experimental_condition"])

FINAL_COLUMN_ORDER = (
    SCENARIO_COLUMNS
    + ["n_configurations_compared"]
    + ["suppression_mode", "mitigation_technique", "rank", "n_runs"]
    + [
        "original_depth",
        "evaluated_depth",
        "original_gate_count",
        "evaluated_gate_count",
        "depth_overhead_mean",
        "gate_count_overhead_mean",
        "execution_time_mean",
        "runtime_overhead_mean",
        "sampling_overhead_mean",
        "fidelity_mean",
        "fidelity_std",
        "error_reduction_percentage",
        "efficiency_score",
        "baseline_fidelity_mean",
    ]
)


def _load_summary_results(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        raise FileNotFoundError(
            f"summary_results.csv is empty or missing at {path}; run "
            "analysis/summary.py first."
        )
    summary = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_columns = REQUIRED_SUMMARY_COLUMNS - set(summary.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"summary_results.csv is missing required columns: {missing}")

    numeric_columns = [
        "n_runs",
        "original_depth",
        "evaluated_depth",
        "original_gate_count",
        "evaluated_gate_count",
        "depth_overhead_mean",
        "gate_count_overhead_mean",
        "execution_time_mean",
        "runtime_overhead_mean",
        "sampling_overhead_mean",
        "fidelity_mean",
        "fidelity_std",
        "error_reduction_percentage",
    ]
    for column in numeric_columns:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")

    return summary


def _is_baseline(row: pd.Series) -> bool:
    return (
        row["experimental_condition"] == "Baseline"
        and row["suppression_mode"] == "none"
        and row["mitigation_technique"] == ""
    )


def _efficiency_score(row: pd.Series) -> float | None:
    if _is_baseline(row):
        return None
    overhead = row["runtime_overhead_mean"]
    reduction = row["error_reduction_percentage"]
    if pd.isna(overhead) or overhead == 0 or pd.isna(reduction):
        return None
    return reduction / overhead


def _rank_scenario_group(group: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = group[group.apply(_is_baseline, axis=1)]
    baseline_fidelity_mean = (
        baseline_rows["fidelity_mean"].iloc[0] if len(baseline_rows) else None
    )

    ranked = group.sort_values(
        "error_reduction_percentage", ascending=False, na_position="last"
    ).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["n_configurations_compared"] = len(ranked)
    ranked["baseline_fidelity_mean"] = baseline_fidelity_mean
    ranked["efficiency_score"] = ranked.apply(_efficiency_score, axis=1)
    return ranked


def build_best_configurations(
    summary_path: Path = SUMMARY_RESULTS_PATH,
    best_configurations_path: Path = BEST_CONFIGURATIONS_PATH,
) -> pd.DataFrame:
    """Read summary_results.csv, rank per scenario, write best_configurations.csv."""
    summary = _load_summary_results(summary_path)

    ranked_groups = [
        _rank_scenario_group(group)
        for _, group in summary.groupby(SCENARIO_COLUMNS, dropna=False, sort=False)
    ]
    best_configurations = pd.concat(ranked_groups, ignore_index=True)
    best_configurations = best_configurations[FINAL_COLUMN_ORDER]

    best_configurations_path.parent.mkdir(parents=True, exist_ok=True)
    best_configurations.to_csv(best_configurations_path, index=False)
    return best_configurations


if __name__ == "__main__":
    result = build_best_configurations()
    print(f"Wrote {len(result)} ranked row(s) to {BEST_CONFIGURATIONS_PATH}")