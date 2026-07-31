"""Mitigation calculations applied after benchmark execution."""

# The benchmark runner executes all auxiliary circuits and passes their results here.

from collections.abc import Callable
from typing import Any

from .cdr import mitigate_cdr
from .m3 import mitigate_m3
from .pec import mitigate_pec
from .trex import mitigate_trex
from .zne import mitigate_zne

MITIGATION_METHODS: dict[str, Callable[..., Any]] = {
    "m3": mitigate_m3,
    "trex": mitigate_trex,
    "zne": mitigate_zne,
    "pec": mitigate_pec,
    "cdr": mitigate_cdr,
}


def apply_mitigation(technique: str, **mitigation_data: Any) -> Any:
    """Apply one configured mitigation technique to runner-supplied data."""
    try:
        method = MITIGATION_METHODS[technique]
    except KeyError as error:
        supported = ", ".join(MITIGATION_METHODS)
        raise ValueError(
            f"Unsupported mitigation technique: {technique}. Supported techniques: {supported}."
        ) from error
    return method(**mitigation_data)


__all__ = [
    "MITIGATION_METHODS",
    "apply_mitigation",
    "mitigate_cdr",
    "mitigate_m3",
    "mitigate_pec",
    "mitigate_trex",
    "mitigate_zne",
]
