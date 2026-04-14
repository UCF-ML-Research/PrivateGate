"""Baseline #2 — mask every detected span, then send to the standard backend."""
from __future__ import annotations

from privategate.backends.base import Backend
from privategate.backends.mock_standard import MockStandardBackend
from privategate.detector.rule_detector import RuleDetector
from privategate.eval.baselines.base import Baseline, BaselineRun, _timed
from privategate.eval.example import EvalExample
from privategate.rewriter.actions import apply_mask


class FullMaskBaseline(Baseline):
    name = "full_mask"

    def __init__(self, backend: Backend | None = None) -> None:
        self._backend = backend or MockStandardBackend()
        self._detector = RuleDetector()

    def _mask_all(self, text: str) -> str:
        spans = sorted(self._detector.detect(text), key=lambda s: s.start)
        out: list[str] = []
        cursor = 0
        for s in spans:
            if s.start < cursor:
                continue
            out.append(text[cursor:s.start])
            out.append(apply_mask(s))
            cursor = s.end
        out.append(text[cursor:])
        return "".join(out)

    def run_one(self, example: EvalExample) -> BaselineRun:
        masked = self._mask_all(example.text)
        answer, ms = _timed(lambda: self._backend.complete(masked))
        return BaselineRun(
            name=self.name,
            outbound_text=masked,
            answer=answer,
            routing_path="standard",
            latency_ms=ms,
        )
