from __future__ import annotations

from privategate.policy.table import PolicyTable
from privategate.types import Action, PolicyDecision, Span


class PolicyEngine:
    def __init__(self, table: PolicyTable) -> None:
        self._table = table

    def decide(self, spans: list[Span]) -> list[PolicyDecision]:
        out: list[PolicyDecision] = []
        for span in spans:
            action = self._table.lookup(span.category, span.risk)
            out.append(
                PolicyDecision(
                    span=span,
                    action=action,
                    reason=f"{span.category.value}/{span.risk.value} -> {action.value}",
                )
            )
        return out

    @staticmethod
    def needs_secure_path(decisions: list[PolicyDecision]) -> bool:
        return any(d.action is Action.SECURE_SLOT for d in decisions)
