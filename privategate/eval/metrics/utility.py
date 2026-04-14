"""Utility metrics for QA and generative tasks."""
from __future__ import annotations

import re
from typing import Optional

from privategate.router.divergence import Embedder, answer_divergence


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def answer_similarity(
    prediction: str,
    reference: str,
    embedder: Optional[Embedder] = None,
) -> float:
    """Returns similarity in [0, 1] (1 = identical)."""
    return 1.0 - answer_divergence(prediction, reference, embedder=embedder)
