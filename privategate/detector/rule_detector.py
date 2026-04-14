"""Composes regex + gazetteer detectors and resolves overlapping spans."""
from __future__ import annotations

from privategate.detector.gazetteer import GazetteerDetector
from privategate.detector.rules import ALL_DETECTORS
from privategate.types import Span


def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Keep the higher-risk span on overlap; on ties, keep the longer one."""
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Span] = []
    for s in spans:
        if not out:
            out.append(s)
            continue
        last = out[-1]
        if s.overlaps(last):
            keep = max(
                (s, last),
                key=lambda x: (x.risk.rank, x.end - x.start),
            )
            out[-1] = keep
        else:
            out.append(s)
    return out


class RuleDetector:
    def __init__(self) -> None:
        self._gazetteer = GazetteerDetector()

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for fn in ALL_DETECTORS:
            spans.extend(fn(text))
        spans.extend(self._gazetteer.detect(text))
        return _resolve_overlaps(spans)
