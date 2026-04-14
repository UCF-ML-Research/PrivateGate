"""Gazetteer detector for closed-vocabulary terms (medical conditions, names).

Kept tiny on purpose: the ML detector (M3) takes over for open-vocabulary work.
"""
from __future__ import annotations

import re
from typing import Iterable

from privategate.types import Category, RiskLevel, Span


DEFAULT_MEDICAL_TERMS = {
    "diabetes",
    "type 1 diabetes",
    "type 2 diabetes",
    "t2 diabetes",
    "t2 diabetic",
    "hiv",
    "aids",
    "cancer",
    "pancreatic cancer",
    "metastatic pancreatic cancer",
    "depression",
    "schizophrenia",
    "hypertension",
    "metformin",
}

DEFAULT_NAME_PREFIXES = {"mr.", "mrs.", "ms.", "dr.", "prof."}


class GazetteerDetector:
    def __init__(
        self,
        medical_terms: Iterable[str] = DEFAULT_MEDICAL_TERMS,
        name_prefixes: Iterable[str] = DEFAULT_NAME_PREFIXES,
    ) -> None:
        # sort longest-first so multi-word terms beat their substrings
        self._medical = sorted({t.lower() for t in medical_terms}, key=len, reverse=True)
        self._name_prefixes = {p.lower() for p in name_prefixes}

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        lowered = text.lower()
        consumed = [False] * len(text)

        for term in self._medical:
            for m in re.finditer(rf"\b{re.escape(term)}\b", lowered):
                s, e = m.start(), m.end()
                if any(consumed[s:e]):
                    continue
                for i in range(s, e):
                    consumed[i] = True
                spans.append(
                    Span(
                        start=s,
                        end=e,
                        text=text[s:e],
                        category=Category.MEDICAL,
                        risk=RiskLevel.HIGH,
                        source="gazetteer",
                    )
                )

        for m in re.finditer(r"\b([A-Z][a-z]+)\.?\s+([A-Z][a-z]+)", text):
            prefix = m.group(1).lower() + "."
            if prefix in self._name_prefixes:
                s, e = m.start(), m.end()
                if any(consumed[s:e]):
                    continue
                for i in range(s, e):
                    consumed[i] = True
                spans.append(
                    Span(
                        start=s,
                        end=e,
                        text=text[s:e],
                        category=Category.IDENTIFIER,
                        risk=RiskLevel.HIGH,
                        source="gazetteer",
                    )
                )
        return spans
