from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Mode(str, Enum):
    PLAINTEXT = "plaintext"
    REDACT = "redact"
    PERTURB = "perturb"
    TEE = "tee"
    HE_SMPC = "he_smpc"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    field: str | None = None  # field path for structured (A2A) payloads


@dataclass(frozen=True)
class Signal:
    type: str                                    # e.g. "pii.email", "phi.diagnosis"
    score: float                                 # [0, 1]
    source: str                                  # detector id@version
    span: Span | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Context:
    user: str | None = None
    session: str | None = None
    app: str | None = None
    jurisdiction: str | None = None
    inherited_labels: tuple[str, ...] = ()       # A2A: labels from upstream hops


@dataclass(frozen=True)
class Request:
    payload: str | dict[str, Any]                # text (H2A) or structured (A2A)
    context: Context = field(default_factory=Context)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    mode: Mode
    matched_rule: str | None
    rationale: str
    signals: tuple[Signal, ...]
    confidence: float
    counterfactual: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformedRequest:
    mode: Mode
    payload: str | dict[str, Any] | bytes
    recovery_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseChunk:
    data: bytes | str
    is_final: bool = False
    attestation: dict[str, Any] | None = None
