"""Efficiency metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LatencyReport:
    p50: float
    p95: float
    mean: float
    n: int


def latency_percentiles(latencies_ms: Sequence[float]) -> LatencyReport:
    if not latencies_ms:
        return LatencyReport(p50=0.0, p95=0.0, mean=0.0, n=0)
    s = sorted(latencies_ms)
    n = len(s)

    def _pct(q: float) -> float:
        if n == 1:
            return s[0]
        # nearest-rank percentile
        k = max(0, min(n - 1, int(round(q * (n - 1)))))
        return s[k]

    return LatencyReport(
        p50=_pct(0.50),
        p95=_pct(0.95),
        mean=sum(s) / n,
        n=n,
    )


def secure_path_fraction(routes: Sequence[str]) -> float:
    if not routes:
        return 0.0
    return sum(1 for r in routes if r == "secure") / len(routes)
