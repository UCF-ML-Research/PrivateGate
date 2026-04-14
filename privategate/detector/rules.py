"""Rule-based PII detectors. Each function returns a list of `Span`s.

Detectors are deliberately conservative: false positives are preferable to
false negatives, since a missed span is a privacy leak (see plan §8).
"""
from __future__ import annotations

import re

from privategate.types import Category, RiskLevel, Span

_SSN_RE = re.compile(r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)
_API_KEY_RE = re.compile(
    r"\b(?:sk|pk|api|key|token)[-_][A-Za-z0-9]{16,}\b", re.IGNORECASE
)
_DOB_RE = re.compile(
    r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b"
)
_MRN_RE = re.compile(r"\bMRN[:#\s-]*\d{5,10}\b", re.IGNORECASE)
_CC_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _scan(pattern: re.Pattern, text: str, category: Category, risk: RiskLevel) -> list[Span]:
    out: list[Span] = []
    for m in pattern.finditer(text):
        out.append(
            Span(
                start=m.start(),
                end=m.end(),
                text=m.group(0),
                category=category,
                risk=risk,
                source="rule",
            )
        )
    return out


def detect_ssn(text: str) -> list[Span]:
    return _scan(_SSN_RE, text, Category.IDENTIFIER, RiskLevel.HIGH)


def detect_email(text: str) -> list[Span]:
    return _scan(_EMAIL_RE, text, Category.IDENTIFIER, RiskLevel.MEDIUM)


def detect_phone(text: str) -> list[Span]:
    return _scan(_PHONE_RE, text, Category.IDENTIFIER, RiskLevel.MEDIUM)


def detect_api_key(text: str) -> list[Span]:
    return _scan(_API_KEY_RE, text, Category.CREDENTIAL, RiskLevel.CRITICAL)


def detect_dob(text: str) -> list[Span]:
    return _scan(_DOB_RE, text, Category.IDENTIFIER, RiskLevel.HIGH)


def detect_mrn(text: str) -> list[Span]:
    return _scan(_MRN_RE, text, Category.MEDICAL, RiskLevel.HIGH)


def detect_credit_card(text: str) -> list[Span]:
    out: list[Span] = []
    for m in _CC_RE.finditer(text):
        if _luhn(m.group(0)):
            out.append(
                Span(
                    start=m.start(),
                    end=m.end(),
                    text=m.group(0),
                    category=Category.FINANCIAL,
                    risk=RiskLevel.CRITICAL,
                    source="rule",
                )
            )
    return out


ALL_DETECTORS = [
    detect_api_key,  # run credentials first — they win on overlap
    detect_ssn,
    detect_credit_card,
    detect_email,
    detect_phone,
    detect_dob,
    detect_mrn,
]
