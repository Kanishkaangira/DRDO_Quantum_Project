"""Plot generation utilities -- reads only from results/csv/best_configurations.csv,
never recomputes anything itself. Re-plotting never requires re-running experiments.
"""

from .plots import (
    generate_all_plots,
    plot_technique_comparison,
    plot_tradeoff_scatter,
)

__all__ = [
    "generate_all_plots",
    "plot_technique_comparison",
    "plot_tradeoff_scatter",
]