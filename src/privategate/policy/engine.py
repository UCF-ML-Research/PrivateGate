from __future__ import annotations

from ..types import Decision, Mode, Request, Signal
from .rule import Rule


def _rule_matches(rule: Rule, request: Request, signals: list[Signal]) -> bool:
    w = rule.when
    if callable(w):
        return bool(w(request, signals))
    raise NotImplementedError(
        f"Rule {rule.id!r} has a non-callable `when` ({type(w).__name__}); "
        "string/CEL rules require a CELPolicyEngine subclass."
    )


class PolicyEngine:
    """Evaluates rules over signals + context to pick a Mode.

    Resolution: highest-priority matched rule wins. Within the same priority,
    the first rule in the list wins (order-stable). Default if no rule
    matches: `DEFAULT_MODE` (PLAINTEXT).

    Fail-closed: any evaluation error returns `Mode.ABSTAIN` with the error in
    the trace. This means "never downgrade on error" is enforced at the engine
    level, not left to each rule author.
    """

    DEFAULT_MODE = Mode.PLAINTEXT

    def __init__(self, rules: list[Rule], strictness: list[Mode]):
        self.rules = list(rules)
        self.strictness = strictness

    def evaluate(self, request: Request, signals: list[Signal]) -> Decision:
        matched: list[Rule] = []
        trace: list[dict] = []
        try:
            for rule in self.rules:
                try:
                    if _rule_matches(rule, request, signals):
                        matched.append(rule)
                        trace.append({"rule": rule.id, "matched": True})
                    else:
                        trace.append({"rule": rule.id, "matched": False})
                except Exception as e:
                    trace.append({"rule": rule.id, "error": str(e)})
                    return self._fail_closed(signals, trace, f"rule {rule.id}: {e}")
        except Exception as e:
            return self._fail_closed(signals, trace, str(e))

        if not matched:
            return Decision(
                mode=self.DEFAULT_MODE,
                matched_rule=None,
                rationale="no rule matched; default",
                signals=tuple(signals),
                confidence=1.0,
                trace={"evaluated": trace},
            )

        best = max(matched, key=lambda r: (r.priority, -self.rules.index(r)))
        return Decision(
            mode=best.route,
            matched_rule=best.id,
            rationale=best.rationale,
            signals=tuple(signals),
            confidence=1.0,
            trace={"evaluated": trace, "matched": [r.id for r in matched]},
        )

    def _fail_closed(
        self, signals: list[Signal], trace: list[dict], reason: str
    ) -> Decision:
        return Decision(
            mode=Mode.ABSTAIN,
            matched_rule=None,
            rationale=f"fail-closed: {reason}",
            signals=tuple(signals),
            confidence=0.0,
            trace={"evaluated": trace},
        )
