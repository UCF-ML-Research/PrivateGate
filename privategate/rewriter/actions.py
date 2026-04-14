"""Per-action span transformation primitives.

Each function returns the replacement string and (optionally) records the
original value in the placeholder map. The map is the only place plaintext
of a sensitive span lives after rewriting.
"""
from __future__ import annotations

from faker import Faker

from privategate.rewriter.placeholder_map import PlaceholderMap
from privategate.types import Category, Span


_ABSTRACT_FALLBACK: dict[Category, str] = {
    Category.IDENTIFIER: "a personal identifier",
    Category.CREDENTIAL: "a credential",
    Category.MEDICAL: "a health condition",
    Category.FINANCIAL: "a financial detail",
    Category.PERSONAL_CONTEXT: "personal context",
}

_ABSTRACT_MEDICAL: dict[str, str] = {
    "metastatic pancreatic cancer": "a serious illness",
    "pancreatic cancer": "a serious illness",
    "type 2 diabetes": "a chronic condition",
    "t2 diabetes": "a chronic condition",
    "t2 diabetic": "a chronic condition",
    "type 1 diabetes": "a chronic condition",
    "diabetes": "a chronic condition",
    "hiv": "a chronic infection",
    "depression": "a mental-health condition",
    "schizophrenia": "a mental-health condition",
    "hypertension": "a chronic condition",
    "metformin": "a prescription medication",
}


def apply_mask(span: Span) -> str:
    return f"[{span.category.value}]"


def apply_pseudonymize(span: Span, faker: Faker, pmap: PlaceholderMap) -> str:
    if span.category is Category.IDENTIFIER:
        if "@" in span.text:
            fake = faker.email()
        elif any(c.isdigit() for c in span.text) and "-" in span.text and len(span.text) <= 11:
            fake = faker.ssn()
        elif any(c.isdigit() for c in span.text):
            fake = faker.phone_number()
        else:
            fake = faker.name()
    elif span.category is Category.FINANCIAL:
        fake = faker.credit_card_number()
    else:
        fake = f"[{span.category.value}]"
    pmap.add(span.category.value, span.text)
    return fake


def apply_abstract(span: Span, level: str = "medium") -> str:
    """Rewrite a span at one of three abstraction levels.

    - ``low``: return the span verbatim (no abstraction). Highest utility,
      no privacy gain. Useful as an ablation upper bound.
    - ``medium``: term-specific superclass (the default — see plan §4.3).
    - ``high``: collapse to the bare category mask. Lowest utility, best
      surface-form privacy.
    """
    if level not in {"low", "medium", "high"}:
        raise ValueError(f"unknown abstraction level: {level}")
    if level == "low":
        return span.text
    if level == "high":
        return f"[{span.category.value}]"
    if span.category is Category.MEDICAL:
        key = span.text.lower().strip()
        return _ABSTRACT_MEDICAL.get(key, _ABSTRACT_FALLBACK[Category.MEDICAL])
    return _ABSTRACT_FALLBACK.get(span.category, "[REDACTED]")


def apply_secure_slot(span: Span, pmap: PlaceholderMap) -> str:
    return pmap.add("SLOT", span.text)
