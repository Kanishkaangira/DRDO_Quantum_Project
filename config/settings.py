"""
Fixed constants used across the entire framework.
Nothing else in the project should hardcode a seed, shot count, or run count —
that's what keeps every experiment reproducible and every comparison valid.
"""

SEED = 42
SEED_STRIDE = 10000
DEFAULT_SHOTS = 4000
DEFAULT_N_RUNS = 10
DEFAULT_BACKEND = "aer"  # "aer" or "ibm"
