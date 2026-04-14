from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Category(str, Enum):
    IDENTIFIER = "IDENTIFIER"
    CREDENTIAL = "CREDENTIAL"
    MEDICAL = "MEDICAL"
    FINANCIAL = "FINANCIAL"
    PERSONAL_CONTEXT = "PERSONAL_CONTEXT"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]


class Action(str, Enum):
    KEEP = "keep"
    MASK = "mask"
    PSEUDONYMIZE = "pseudonymize"
    ABSTRACT = "abstract"
    SECURE_SLOT = "secure-slot"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    category: Category
    risk: RiskLevel
    source: str = "rule"  # "rule" | "ml" | "gazetteer"

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span offsets: start={self.start}, end={self.end}")
        if not self.text:
            raise ValueError("span text must be non-empty")

    def overlaps(self, other: "Span") -> bool:
        return not (self.end <= other.start or other.end <= self.start)


@dataclass
class DetectionResult:
    text: str
    spans: list[Span] = field(default_factory=list)


@dataclass
class PolicyDecision:
    span: Span
    action: Action
    reason: str = ""


@dataclass
class RewriteResult:
    original_text: str
    transformed_text: str
    placeholder_map: dict[str, str]
    decisions: list[PolicyDecision]
    has_secure_slots: bool = False


@dataclass
class RoutingDecision:
    path: str  # "standard" | "secure"
    reason: str


@dataclass
class ProbeResult:
    span: Span
    divergence: float
    semantic_critical: bool
