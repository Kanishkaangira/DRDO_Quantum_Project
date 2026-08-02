"""Ordered benchmark-case generation from active configuration values."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from config.benchmark_config import (
    ACTIVE_BACKENDS,
    ACTIVE_CIRCUITS,
    ACTIVE_EXPERIMENTAL_CONDITIONS,
    ACTIVE_N_RUNS,
    ACTIVE_NOISE_MODELS,
    ACTIVE_SHOT_VALUES,
    EXPERIMENTAL_CONDITIONS,
    get_active_circuit_build_parameters,
    get_active_noise_strengths,
    get_active_qubit_counts,
)


@dataclass(frozen=True)
class BenchmarkCase:
    """One branch execution point with its post-execution mitigations."""

    backend_name: str
    circuit_name: str
    num_qubits: int
    build_parameters: Mapping[str, object]
    noise_name: str | None
    noise_strength: float | None
    shots: int
    run_index: int
    condition_name: str
    suppression_mode: str
    mitigation_techniques: tuple[str, ...]


def iter_active_benchmark_cases() -> Iterator[BenchmarkCase]:
    """Yield cases in backend, circuit, noise, shot, branch, then run order."""
    for backend_name in ACTIVE_BACKENDS:
        for circuit_name in ACTIVE_CIRCUITS:
            for num_qubits in get_active_qubit_counts(circuit_name):
                for build_parameters in get_active_circuit_build_parameters(circuit_name):
                    if backend_name == "aer":
                        noise_settings = (
                            (noise_name, noise_strength)
                            for noise_name in ACTIVE_NOISE_MODELS
                            for noise_strength in get_active_noise_strengths(noise_name)
                        )
                    else:
                        noise_settings = ((None, None),)

                    for noise_name, noise_strength in noise_settings:
                        for shots in ACTIVE_SHOT_VALUES:
                            for condition_name in ACTIVE_EXPERIMENTAL_CONDITIONS:
                                condition = EXPERIMENTAL_CONDITIONS[condition_name]
                                for suppression_mode in condition["suppression_modes"]:
                                    for run_index in range(ACTIVE_N_RUNS):
                                        yield BenchmarkCase(
                                            backend_name=backend_name,
                                            circuit_name=circuit_name,
                                            num_qubits=num_qubits,
                                            build_parameters=build_parameters,
                                            noise_name=noise_name,
                                            noise_strength=noise_strength,
                                            shots=shots,
                                            run_index=run_index,
                                            condition_name=condition_name,
                                            suppression_mode=suppression_mode,
                                            mitigation_techniques=tuple(
                                                condition["mitigation_techniques"]
                                            ),
                                        )
