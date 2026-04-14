import json
from pathlib import Path

from privategate.router.router import Router
from privategate.router.trace import append_trace, build_trace
from privategate.types import (
    Action,
    Category,
    PolicyDecision,
    ProbeResult,
    RewriteResult,
    RiskLevel,
    Span,
)


def _rw_with_decision() -> RewriteResult:
    span = Span(0, 5, "xxxxx", Category.IDENTIFIER, RiskLevel.HIGH)
    return RewriteResult(
        original_text="xxxxx ssn",
        transformed_text="[IDENTIFIER] ssn",
        placeholder_map={"[SLOT_AB0]": "xxxxx"},
        decisions=[PolicyDecision(span=span, action=Action.PSEUDONYMIZE, reason="r")],
        has_secure_slots=False,
    )


def test_trace_records_decisions_without_plaintext():
    rw = _rw_with_decision()
    routing = Router().route(rw)
    trace = build_trace(rw, routing, probe_results=[])
    payload = trace.to_dict()

    # plaintext must not appear in the decision records
    assert all("xxxxx" not in str(v) for d in payload["decisions"] for v in d.values())
    assert payload["routing_path"] == "standard"
    assert payload["placeholder_count"] == 1
    assert payload["decisions"][0]["category"] == "IDENTIFIER"
    assert payload["decisions"][0]["action"] == "pseudonymize"


def test_trace_records_probe_results():
    rw = _rw_with_decision()
    routing = Router().route(rw)
    span = Span(0, 5, "xxxxx", Category.MEDICAL, RiskLevel.HIGH)
    probe = ProbeResult(span=span, divergence=0.42, semantic_critical=True)
    trace = build_trace(rw, routing, probe_results=[probe])
    payload = trace.to_dict()
    assert len(payload["probes"]) == 1
    assert payload["probes"][0]["semantic_critical"] is True
    assert abs(payload["probes"][0]["divergence"] - 0.42) < 1e-6


def test_append_trace_writes_jsonl(tmp_path):
    rw = _rw_with_decision()
    routing = Router().route(rw)
    trace = build_trace(rw, routing, probe_results=[])
    path = tmp_path / "traces.jsonl"
    append_trace(path, trace)
    append_trace(path, trace)
    lines = Path(path).read_text().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["routing_path"] == "standard"
