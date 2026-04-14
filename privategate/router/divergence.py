"""Answer-divergence metric for the semantic-dependency probe.

Plan §5.2 describes this as "answer-embedding divergence". A real
sentence-embedding model is the right tool for evaluation, but the
gateway runs in a hot path on the client and we want unit tests that
don't pull in `sentence-transformers`. We therefore default to a
deterministic token-Jaccard distance and let callers plug in a custom
embedder via the `embedder` argument.

Returns a value in [0, 1] where 0 = identical and 1 = no overlap.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Optional, Sequence

Embedder = Callable[[str], Sequence[float]]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _jaccard_distance(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return 1.0 - (inter / union)


def _cosine_distance(va: Sequence[float], vb: Sequence[float]) -> float:
    if len(va) != len(vb):
        raise ValueError(f"vector dim mismatch: {len(va)} vs {len(vb)}")
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na == 0 or nb == 0:
        return 1.0
    cos = dot / (na * nb)
    cos = max(-1.0, min(1.0, cos))
    # remap [-1, 1] -> [1, 0]
    return (1.0 - cos) / 2.0


def answer_divergence(
    a: str,
    b: str,
    embedder: Optional[Embedder] = None,
) -> float:
    if embedder is None:
        return _jaccard_distance(a, b)
    return _cosine_distance(embedder(a), embedder(b))
