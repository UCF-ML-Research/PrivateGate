from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Request, Signal


@runtime_checkable
class Detector(Protocol):
    """Extracts privacy-relevant signals from a request.

    Implementations must be deterministic given a fixed model version
    and must not mutate the request. Signals are purely additive — the
    policy engine is responsible for monotone escalation (signals only
    ever raise the sensitivity tier).

    Layers:
      L0 — deterministic: regex, secret scanners, keyword ontologies.
      L1 — small CPU ML: PII NER (GLiNER/Piiranha), sensitivity
           classifier (distilled ModernBERT), prompt-injection filter.
      L2 — remote TEE LLM-judge (optional, called only on low confidence).
    """

    id: str
    version: str

    def detect(self, request: Request) -> list[Signal]: ...
