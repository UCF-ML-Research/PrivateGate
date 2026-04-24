"""Keyword / ontology-only router (B4).

HIPAA-PHI + PCI-DSS + export-control keyword lists → HE.
Other PII-adjacent keywords → LDP. Else plaintext.

Lists below are compact illustrative ontologies. A production policy layer
would load them from YAML and version them; keeping them inline here makes
the baseline self-contained.
"""
from __future__ import annotations

from time import perf_counter

from privategate.types import Mode

from ..harness.schemas import GoldItem, RouterDecision

_HE_KEYWORDS = {
    # HIPAA-PHI indicators
    "medical record", "mrn", "diagnosis", "icd-10", "icd10", "prescription",
    "patient id", "health plan", "hipaa", "phi",
    # PCI-DSS indicators
    "cardholder", "cvv", "cvv2", "pci-dss", "pci dss", "track data",
    "service code", "pan:",
    # Export-control indicators
    "eccn", "itar", "ear99", "dual-use", "dual use", "export controlled",
    "export-controlled",
}

_LDP_KEYWORDS = {
    "ssn", "social security", "date of birth", " dob ",
    "address", "zip code", "postal code", "driver's license",
    "passport", "bank account", "iban", "routing number",
}


def _hits(text_lower: str, vocab: set[str]) -> list[str]:
    return [k for k in vocab if k in text_lower]


class KeywordOnly:
    """B4 — keyword/ontology-only router."""

    name = "keyword_only"

    def route(self, item: GoldItem) -> RouterDecision:
        t0 = perf_counter()
        p = item.prompt.lower()
        he_hits = _hits(p, _HE_KEYWORDS)
        ldp_hits = _hits(p, _LDP_KEYWORDS) if not he_hits else []
        if he_hits:
            mode = Mode.HE_SMPC
            rationale = f"he_keywords={he_hits}"
        elif ldp_hits:
            mode = Mode.PERTURB
            rationale = f"ldp_keywords={ldp_hits}"
        else:
            mode = Mode.PLAINTEXT
            rationale = "no_ontology_match"
        return RouterDecision(
            item_id=item.id,
            mode=mode,
            latency_ms=(perf_counter() - t0) * 1000.0,
            rationale=rationale,
        )
