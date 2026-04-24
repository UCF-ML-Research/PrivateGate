from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from privategate.types import Mode

# v1 uses three active backends (plus ABSTAIN) of the full Mode enum.
V1_MODES: frozenset[Mode] = frozenset(
    {Mode.PLAINTEXT, Mode.PERTURB, Mode.HE_SMPC, Mode.ABSTAIN}
)

# User-facing JSONL aliases ↔ canonical Mode.
# The author-level vocabulary is {plaintext, ldp, he, abstain}; canonical enum
# values are {plaintext, perturb, he_smpc, abstain}.
JSONL_TO_MODE: dict[str, Mode] = {
    "plaintext": Mode.PLAINTEXT,
    "ldp": Mode.PERTURB,
    "perturb": Mode.PERTURB,
    "he": Mode.HE_SMPC,
    "he_smpc": Mode.HE_SMPC,
    "abstain": Mode.ABSTAIN,
}

MODE_TO_JSONL: dict[Mode, str] = {
    Mode.PLAINTEXT: "plaintext",
    Mode.PERTURB: "ldp",
    Mode.HE_SMPC: "he",
    Mode.ABSTAIN: "abstain",
}


@dataclass(frozen=True)
class GoldItem:
    """One labeled example from privategate-gold."""

    id: str
    prompt: str
    source: str                            # sharegpt | synthetic | adversarial | a2a
    categories: dict[str, bool]            # multi-label sensitivity categories
    severity: dict[str, int]               # per-category 0..3
    spans: list[dict[str, Any]]            # [{start, end, type}]
    gold_mode: Mode
    notes: str = ""
    annotators: tuple[str, ...] = ()
    adjudicated: bool = False


@dataclass(frozen=True)
class RouterDecision:
    """What a router produced for a single item."""

    item_id: str
    mode: Mode
    latency_ms: float
    rationale: str = ""
    matched_rule: str | None = None
    confidence: float = 0.0


@dataclass
class EvalResult:
    """Aggregate evaluation over a dataset."""

    router: str
    n: int
    fdr_overall: float
    fdr_per_class: dict[str, float]
    oer: float
    abstain_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
