"""Static quality SQ(c_i) — Eq. 13 with the Table 8 taxonomy."""
from __future__ import annotations

from . import config as C


def static_quality(defects: list[str]) -> float:
    total = sum(C.DEFECTS.get(d, (0.0, False))[0] for d in set(defects))
    return max(0.0, 1.0 - total)


def has_fatal(defects: list[str]) -> bool:
    return any(C.DEFECTS.get(d, (0.0, False))[1] for d in set(defects))
