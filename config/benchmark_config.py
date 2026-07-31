"""
Sweep ranges for the full four-condition experimental design.
Organized into three sections (per Section 7 of the locked architecture doc)
so vocabulary stays aligned with the problem statement's own wording:

  SECTION 1 - Study Components   : which circuits/noise/suppression/mitigation
                                    are being studied (NOT in the problem
                                    statement's own "Evaluation Parameters")
  SECTION 2 - Evaluation Parameters : Circuit Depth, Number of Qubits,
                                    Noise Strength, Number of Shots, Backend
                                    -- exactly the problem statement's Phase III/IV list
  SECTION 3 - General Settings   : development-mode active subset (Section 8.4)

This file does not decide loop ORDER -- that's benchmark/runner.py's job
(Backend outermost, per Section 8.1). This file only decides what values
exist to loop over.
"""

from math import pi

from .settings import DEFAULT_N_RUNS

# SECTION 1: STUDY COMPONENTS

CIRCUITS = ["bell", "ghz", "qft", "vqe", "qaoa"]

NOISE_MODELS = ["depolarizing", "amplitude", "phase", "readout", "coherent"]

# "none" is included so Baseline and Mitigation-Only branches can be expressed
# as suppression_mode == "none" rather than as special-cased branches
SUPPRESSION_MODES = ["none", "pauli_twirl", "dd", "both"]

MITIGATION_TECHNIQUES = ["m3", "trex", "zne", "pec", "cdr"]

EXPERIMENTAL_CONDITIONS = {
    "baseline": {"suppression_modes": ["none"], "mitigation_techniques": []},
    "suppression_only": {
        "suppression_modes": ["pauli_twirl", "dd", "both"],
        "mitigation_techniques": [],
    },
    "mitigation_only": {
        "suppression_modes": ["none"],
        "mitigation_techniques": MITIGATION_TECHNIQUES,
    },
    "suppression_and_mitigation": {
        "suppression_modes": ["pauli_twirl", "dd", "both"],
        "mitigation_techniques": MITIGATION_TECHNIQUES,
    },
}

# Techniques that correct a single execution's output directly (Section 8.1)
SINGLE_EXECUTION_MITIGATIONS = ["m3", "trex"]

# Techniques that require their own additional executions
# (noise-scaled circuits / sampled variants / near-Clifford training circuits)
ADDITIONAL_EXECUTION_MITIGATIONS = ["zne", "pec", "cdr"]

# SECTION 2: EVALUATION PARAMETERS

BACKENDS = ["aer"]  # add "ibm" later once hardware access is confirmed

# Per-circuit qubit counts -- NOT one flat list, since qubit count is
# fixed/scalable differently per circuit (Bell is always 2)
QUBIT_COUNTS = {
    "bell": [2],
    "ghz": [3, 4, 5],
    "qft": [2, 3, 4],
    "vqe": [2, 3],
    "qaoa": [2, 3],
}

CIRCUIT_BUILD_PARAMETERS = {
    "bell": [{}],
    "ghz": [{}],
    "qft": [{"approximation_degree": 0, "do_swaps": True}],
    "vqe": [
        {"ansatz_repetitions": 1},
        {"ansatz_repetitions": 2},
        {"ansatz_repetitions": 3},
    ],
    "qaoa": [
        {"repetitions": 1, "gamma": pi / 4, "beta": pi / 8},
        {"repetitions": 2, "gamma": pi / 4, "beta": pi / 8},
        {"repetitions": 3, "gamma": pi / 4, "beta": pi / 8},
    ],
}

# Only meaningful when Backend == "aer" -- real IBM hardware has its own
# physical noise and cannot take an injected NoiseModel (Section 8.1, step 5)
NOISE_STRENGTHS_BY_MODEL = {
    "depolarizing": [0.001, 0.01, 0.05],
    "amplitude": [0.001, 0.01, 0.05],
    "phase": [0.001, 0.01, 0.05],
    "readout": [0.01, 0.03, 0.05],
    "coherent": [0.01, 0.05, 0.10],
}

SHOT_VALUES = [1000, 4000, 8000]

# Run Number is an explicit loop step, not a trailing note (Section 8.1, step 7)
N_RUNS = DEFAULT_N_RUNS

# SECTION 3: GENERAL SETTINGS

DEVELOPMENT_PHASE = "A"

if DEVELOPMENT_PHASE not in {"A", "B", "C"}:
    raise ValueError("DEVELOPMENT_PHASE must be one of: A, B, C.")

# Development mode runs a small slice of everything above; toggling this one
# flag switches every parameter between its small and full versions at once.
if DEVELOPMENT_PHASE == "A":
    ACTIVE_CIRCUITS = ["bell"]
    ACTIVE_BACKENDS = ["aer"]
    ACTIVE_NOISE_MODELS = ["depolarizing"]
    ACTIVE_SHOT_VALUES = [1000]
    ACTIVE_N_RUNS = 3
    ACTIVE_EXPERIMENTAL_CONDITIONS = ["baseline"]
elif DEVELOPMENT_PHASE == "B":
    ACTIVE_CIRCUITS = ["bell"]
    ACTIVE_BACKENDS = ["aer"]
    ACTIVE_NOISE_MODELS = NOISE_MODELS
    ACTIVE_SHOT_VALUES = [1000]
    ACTIVE_N_RUNS = 3
    ACTIVE_EXPERIMENTAL_CONDITIONS = list(EXPERIMENTAL_CONDITIONS)
else:
    ACTIVE_CIRCUITS = CIRCUITS
    ACTIVE_BACKENDS = BACKENDS
    ACTIVE_NOISE_MODELS = NOISE_MODELS
    ACTIVE_SHOT_VALUES = SHOT_VALUES
    ACTIVE_N_RUNS = N_RUNS
    ACTIVE_EXPERIMENTAL_CONDITIONS = list(EXPERIMENTAL_CONDITIONS)

ACTIVE_NOISE_STRENGTHS_BY_MODEL = {
    noise_model: NOISE_STRENGTHS_BY_MODEL[noise_model][
        :1 if DEVELOPMENT_PHASE == "A" else None
    ]
    for noise_model in ACTIVE_NOISE_MODELS
}


def get_active_qubit_counts(circuit_name: str) -> list[int]:
    """Returns the valid qubit-count loop values for one active circuit."""
    if circuit_name not in ACTIVE_CIRCUITS:
        raise ValueError(f"Circuit is not active: {circuit_name}")
    return QUBIT_COUNTS[circuit_name]


def get_active_noise_strengths(noise_model: str) -> list[float]:
    """Returns the valid noise-strength loop values for one active model."""
    if noise_model not in ACTIVE_NOISE_MODELS:
        raise ValueError(f"Noise model is not active: {noise_model}")
    try:
        return ACTIVE_NOISE_STRENGTHS_BY_MODEL[noise_model]
    except KeyError as error:
        raise ValueError(f"No noise strengths are configured for: {noise_model}") from error


def get_active_circuit_build_parameters(circuit_name: str) -> list[dict[str, object]]:
    """Return independent concrete build requests for one active circuit."""
    if circuit_name not in ACTIVE_CIRCUITS:
        raise ValueError(f"Circuit is not active: {circuit_name}")
    return [parameters.copy() for parameters in CIRCUIT_BUILD_PARAMETERS[circuit_name]]
