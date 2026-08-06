"""Generate paper-ready plots from results/csv/best_configurations.csv.

This module reads only from results/csv/best_configurations.csv and never
recomputes any metric itself -- all fidelity/overhead/ranking numbers come
from analysis/best_configurations.py. Saved images go to results/plots/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; just save files
import matplotlib.pyplot as plt
import pandas as pd

BEST_CONFIGURATIONS_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "csv" / "best_configurations.csv"
)
PLOTS_DIR = Path(__file__).resolve().parents[1] / "results" / "plots"

TECHNIQUE_COLORS = {
    "m3": "#4C72B0",
    "trex": "#DD8452",
    "zne": "#55A868",
    "pec": "#C44E52",
    "cdr": "#8172B2",
}


def _load_best_configurations(path: Path = BEST_CONFIGURATIONS_PATH) -> pd.DataFrame:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        raise FileNotFoundError(
            f"best_configurations.csv is empty or missing at {path}; run "
            "analysis/best_configurations.py first."
        )
    data = pd.read_csv(path)
    return data


def _is_baseline(row: pd.Series) -> bool:
    return row["suppression_mode"] == "none" and row["mitigation_technique"] == ""


def _scenario_label(row: pd.Series) -> str:
    scenario = f"{row['noise_model']}={row['noise_strength']}, {int(row['qubits'])}q"
    return scenario


def _combination_label(row: pd.Series) -> str:
    if _is_baseline(row):
        return "Baseline"
    suppression = row["suppression_mode"] if row["suppression_mode"] != "none" else ""
    mitigation = row["mitigation_technique"]
    parts = [part for part in (suppression, mitigation) if part]
    return " + ".join(parts)


def plot_technique_comparison(
    circuit: str,
    data: pd.DataFrame | None = None,
    output_dir: Path = PLOTS_DIR,
) -> Path | None:
    """Bar chart of the rank-1 (winning) combination per scenario, for one circuit.

    X-axis: one bar per scenario this circuit was tested under (noise model,
    strength, qubit count), showing the winning suppression+mitigation label.
    Y-axis: error_reduction_percentage, with error bars from fidelity_std.
    Returns the saved file path, or None if there's no data to plot.
    """
    if data is None:
        data = _load_best_configurations()

    circuit_rows = data[(data["circuit"] == circuit) & (data["rank"] == 1)].copy()
    circuit_rows = circuit_rows[~circuit_rows.apply(_is_baseline, axis=1)]
    circuit_rows = circuit_rows.dropna(subset=["error_reduction_percentage"])

    if circuit_rows.empty:
        print(f"[plots] Skipping {circuit}: no rank-1 winning rows with data yet.")
        return None

    circuit_rows["scenario_label"] = circuit_rows.apply(_scenario_label, axis=1)
    circuit_rows["combination_label"] = circuit_rows.apply(_combination_label, axis=1)
    circuit_rows = circuit_rows.sort_values("error_reduction_percentage", ascending=False)

    bar_labels = [
        f"{scenario}\n({combo})"
        for scenario, combo in zip(
            circuit_rows["scenario_label"], circuit_rows["combination_label"]
        )
    ]
    bar_colors = [
        TECHNIQUE_COLORS.get(technique, "#999999")
        for technique in circuit_rows["mitigation_technique"]
    ]
    error_bars = circuit_rows["fidelity_std"].fillna(0) * 100  # rough visual scale

    fig, ax = plt.subplots(figsize=(max(6, len(circuit_rows) * 1.5), 5))
    ax.bar(
        bar_labels,
        circuit_rows["error_reduction_percentage"],
        yerr=error_bars,
        color=bar_colors,
        capsize=4,
    )
    ax.set_ylabel("Error Reduction (%)")
    ax.set_title(f"{circuit.upper()} -- Winning Technique per Scenario")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{circuit}_technique_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plots] Saved {output_path}")
    return output_path


def plot_tradeoff_scatter(
    data: pd.DataFrame | None = None,
    output_dir: Path = PLOTS_DIR,
) -> Path | None:
    """Scatter plot of runtime_overhead_mean vs error_reduction_percentage,
    across every circuit and technique, excluding Baseline rows.

    Returns the saved file path, or None if there's no data to plot.
    """
    if data is None:
        data = _load_best_configurations()

    rows = data[~data.apply(_is_baseline, axis=1)].copy()
    rows = rows.dropna(subset=["runtime_overhead_mean", "error_reduction_percentage"])

    if rows.empty:
        print("[plots] Skipping tradeoff scatter: no non-baseline rows with data yet.")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    for technique, technique_rows in rows.groupby("mitigation_technique"):
        if not technique:
            continue
        ax.scatter(
            technique_rows["runtime_overhead_mean"],
            technique_rows["error_reduction_percentage"],
            label=technique,
            color=TECHNIQUE_COLORS.get(technique, "#999999"),
            alpha=0.75,
            edgecolors="black",
            linewidths=0.5,
        )
        for _, row in technique_rows.iterrows():
            ax.annotate(
                row["circuit"],
                (row["runtime_overhead_mean"], row["error_reduction_percentage"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.set_xlabel("Runtime Overhead (ratio to baseline)")
    ax.set_ylabel("Error Reduction (%)")
    ax.set_title("Accuracy vs. Overhead Trade-off, All Circuits")
    ax.legend(title="Mitigation Technique")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tradeoff_scatter_all_circuits.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plots] Saved {output_path}")
    return output_path


def generate_all_plots(output_dir: Path = PLOTS_DIR) -> list[Path]:
    """Generate every configured plot from best_configurations.csv."""
    data = _load_best_configurations()
    saved_paths: list[Path] = []

    for circuit in sorted(data["circuit"].unique()):
        path = plot_technique_comparison(circuit, data=data, output_dir=output_dir)
        if path is not None:
            saved_paths.append(path)

    path = plot_tradeoff_scatter(data=data, output_dir=output_dir)
    if path is not None:
        saved_paths.append(path)

    return saved_paths


if __name__ == "__main__":
    saved = generate_all_plots()
    print(f"Generated {len(saved)} plot(s) in {PLOTS_DIR}")
