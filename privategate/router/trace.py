"""Per-query routing trace.

Every query the gateway processes produces a JSON-serialisable record
that captures *what* was detected, *what* the policy decided, *what* the
probe said, and *why* the router chose its path. These traces are the
audit trail for the privacy claims in the report (plan §6, §M4).

The trace deliberately does **not** include the original sensitive
spans — only the category, risk, action, and source. Writing the
plaintext to a log file would defeat the threat model.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from privategate.types import (
    PolicyDecision,
    ProbeResult,
    RewriteResult,
    RoutingDecision,
)


@dataclass
class RoutingTrace:
    transformed_text: str
    routing_path: str
    routing_reason: str
    has_secure_slots: bool
    placeholder_count: int
    decisions: list[dict] = field(default_factory=list)
    probes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def build_trace(
    rewrite_result: RewriteResult,
    routing: RoutingDecision,
    probe_results: Sequence[ProbeResult] = (),
) -> RoutingTrace:
    decisions: list[dict] = []
    for d in rewrite_result.decisions:
        decisions.append({
            "category": d.span.category.value,
            "risk": d.span.risk.value,
            "source": d.span.source,
            "action": d.action.value,
        })
    probes: list[dict] = []
    for p in probe_results:
        probes.append({
            "category": p.span.category.value,
            "risk": p.span.risk.value,
            "divergence": round(p.divergence, 4),
            "semantic_critical": p.semantic_critical,
        })
    return RoutingTrace(
        transformed_text=rewrite_result.transformed_text,
        routing_path=routing.path,
        routing_reason=routing.reason,
        has_secure_slots=rewrite_result.has_secure_slots,
        placeholder_count=len(rewrite_result.placeholder_map),
        decisions=decisions,
        probes=probes,
    )


def append_trace(path: Path | str, trace: RoutingTrace) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(trace.to_json() + "\n")
