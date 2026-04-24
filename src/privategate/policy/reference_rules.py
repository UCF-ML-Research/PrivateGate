"""Reference routing policy (R1–R4) matching `tests/data/annotation_guide.md`.

Rules are evaluated by `PolicyEngine` with highest-priority-wins + stable
ordering. Priorities are chosen so the intended semantics hold even when
multiple rules match simultaneously:

  R1  injection detected               → ABSTAIN    (priority 100)
  R2  any category with severity = 3   → HE_SMPC    (priority 90)
  R3  any non-`none` category present  → PERTURB    (priority 50)
  R4  otherwise                        → PLAINTEXT  (implicit default)

Expected signal shape from detectors / oracle:

    Signal(type="category.<name>",
           score=<classifier_confidence>,
           evidence={"severity": 0..3},
           source=<detector_id>)

  where `<name>` ∈ {none, pii, phi, pci, secret, ip_confidential,
                    regulated_eu, regulated_us, injection}.
"""
from __future__ import annotations

from ..types import Mode, Request, Signal
from .rule import Rule

CATEGORY_PREFIX = "category."
INJECTION_TYPE = "category.injection"


def _present_categories(signals: list[Signal]) -> list[Signal]:
    return [
        s for s in signals
        if s.type.startswith(CATEGORY_PREFIX) and s.type != "category.none"
    ]


def R1_injection(_: Request, signals: list[Signal]) -> bool:
    return any(s.type == INJECTION_TYPE for s in signals)


def R2_severity_three(_: Request, signals: list[Signal]) -> bool:
    return any(
        int(s.evidence.get("severity", 0)) >= 3
        for s in _present_categories(signals)
    )


def R3_any_sensitive(_: Request, signals: list[Signal]) -> bool:
    return len(_present_categories(signals)) > 0


REFERENCE_RULES: list[Rule] = [
    Rule(
        id="R1_injection",
        when=R1_injection,
        route=Mode.ABSTAIN,
        priority=100,
        rationale="prompt-injection detected; block and ask user",
    ),
    Rule(
        id="R2_severity_three",
        when=R2_severity_three,
        route=Mode.HE_SMPC,
        priority=90,
        rationale="critical-severity sensitive content; must not leave client in plaintext",
    ),
    Rule(
        id="R3_any_sensitive",
        when=R3_any_sensitive,
        route=Mode.PERTURB,
        priority=50,
        rationale="sensitive category present; use local-DP",
    ),
]

# v1 strictness ordering (matches tests/harness/metrics.STRICTNESS).
REFERENCE_STRICTNESS: list[Mode] = [
    Mode.PLAINTEXT,
    Mode.PERTURB,
    Mode.HE_SMPC,
    Mode.ABSTAIN,
]
