from __future__ import annotations

from faker import Faker

from privategate.rewriter.actions import (
    apply_abstract,
    apply_mask,
    apply_pseudonymize,
    apply_secure_slot,
)
from privategate.rewriter.placeholder_map import PlaceholderMap
from privategate.types import Action, PolicyDecision, RewriteResult


class Rewriter:
    def __init__(
        self,
        faker_seed: int | None = None,
        abstraction_level: str = "medium",
    ) -> None:
        self._faker = Faker()
        self._abstraction_level = abstraction_level
        if faker_seed is not None:
            Faker.seed(faker_seed)

    def rewrite(self, text: str, decisions: list[PolicyDecision]) -> RewriteResult:
        pmap = PlaceholderMap()
        decisions_sorted = sorted(decisions, key=lambda d: d.span.start)

        out_parts: list[str] = []
        cursor = 0
        applied: list[PolicyDecision] = []
        has_secure = False

        for d in decisions_sorted:
            span = d.span
            if span.start < cursor:
                # overlap with an already-rewritten span; skip to keep output well-formed
                continue
            out_parts.append(text[cursor:span.start])

            if d.action is Action.KEEP:
                replacement = span.text
            elif d.action is Action.MASK:
                replacement = apply_mask(span)
            elif d.action is Action.PSEUDONYMIZE:
                replacement = apply_pseudonymize(span, self._faker, pmap)
            elif d.action is Action.ABSTRACT:
                replacement = apply_abstract(span, level=self._abstraction_level)
            elif d.action is Action.SECURE_SLOT:
                replacement = apply_secure_slot(span, pmap)
                has_secure = True
            else:
                raise ValueError(f"unknown action: {d.action}")

            out_parts.append(replacement)
            cursor = span.end
            applied.append(d)

        out_parts.append(text[cursor:])
        transformed = "".join(out_parts)

        return RewriteResult(
            original_text=text,
            transformed_text=transformed,
            placeholder_map=pmap.as_dict(),
            decisions=applied,
            has_secure_slots=has_secure,
        )
