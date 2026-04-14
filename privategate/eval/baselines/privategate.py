"""Baseline #5 — PrivateGate (ours).

The full pipeline: detect, decide, rewrite, route, dispatch to the right
backend, reconstruct. Most knobs are exposed in __init__ so the M7
ablation runners can swap individual components without touching the
baseline implementation.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from privategate.backends.base import Backend
from privategate.backends.mock_secure import MockSecureBackend
from privategate.backends.mock_standard import MockStandardBackend
from privategate.detector.rule_detector import RuleDetector
from privategate.eval.baselines.base import Baseline, BaselineRun, _timed
from privategate.eval.example import EvalExample
from privategate.policy.engine import PolicyEngine
from privategate.policy.table import PolicyTable, load_default_policy
from privategate.reconstruct.reconstructor import Reconstructor
from privategate.rewriter.rewriter import Rewriter
from privategate.router.probe import three_variant_probe
from privategate.router.router import Router
from privategate.types import ProbeResult


class PrivateGateBaseline(Baseline):
    name = "privategate"

    def __init__(
        self,
        standard_backend: Backend | None = None,
        secure_backend: Backend | None = None,
        detector: Any | None = None,
        policy_table: PolicyTable | None = None,
        abstraction_level: str = "medium",
        probe_proxy: Optional[Callable[[str], str]] = None,
        probe_threshold: float = 0.3,
        name: str = "privategate",
    ) -> None:
        self.name = name
        self._standard = standard_backend or MockStandardBackend()
        self._secure = secure_backend or MockSecureBackend()
        self._detector = detector or RuleDetector()
        self._engine = PolicyEngine(policy_table or load_default_policy())
        self._rewriter = Rewriter(faker_seed=0, abstraction_level=abstraction_level)
        self._router = Router()
        self._reconstructor = Reconstructor()
        self._probe_proxy = probe_proxy
        self._probe_threshold = probe_threshold

    def _maybe_probe(self, example_text: str, decisions) -> list[ProbeResult]:
        if self._probe_proxy is None:
            return []
        targets = self._router.decide_probe_targets(decisions)
        return [
            three_variant_probe(
                example_text,
                span,
                self._probe_proxy,
                threshold=self._probe_threshold,
            )
            for span in targets
        ]

    def run_one(self, example: EvalExample) -> BaselineRun:
        def _pipeline() -> tuple[str, str, str]:
            spans = self._detector.detect(example.text)
            decisions = self._engine.decide(spans)
            rw = self._rewriter.rewrite(example.text, decisions)
            probes = self._maybe_probe(example.text, decisions)
            routing = self._router.route(rw, probe_results=probes)
            backend = self._secure if routing.path == "secure" else self._standard
            raw = backend.complete(rw.transformed_text)
            final = self._reconstructor.reconstruct(raw, rw.placeholder_map)
            return rw.transformed_text, final, routing.path

        result, ms = _timed(_pipeline)
        outbound, answer, path = result
        return BaselineRun(
            name=self.name,
            outbound_text=outbound,
            answer=answer,
            routing_path=path,
            latency_ms=ms,
        )
