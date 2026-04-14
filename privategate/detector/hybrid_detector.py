"""Hybrid rule + ML detector.

The ML detector is optional. When it is `None` the hybrid degrades
gracefully to the rule layer alone — important so the rest of the
pipeline keeps working in environments without `transformers`.
"""
from __future__ import annotations

from typing import Optional

from privategate.detector.merge import merge_spans
from privategate.detector.ml_detector import MLDetector
from privategate.detector.rule_detector import RuleDetector
from privategate.types import Span


class HybridDetector:
    def __init__(
        self,
        rule_detector: Optional[RuleDetector] = None,
        ml_detector: Optional[MLDetector] = None,
    ) -> None:
        self._rule = rule_detector or RuleDetector()
        self._ml = ml_detector

    def detect(self, text: str) -> list[Span]:
        rule_spans = self._rule.detect(text)
        if self._ml is None:
            return rule_spans
        ml_spans = self._ml.detect(text)
        return merge_spans(rule_spans, ml_spans)

    @property
    def has_ml(self) -> bool:
        return self._ml is not None
