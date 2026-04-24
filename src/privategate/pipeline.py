from __future__ import annotations

from typing import AsyncIterator

from .backends.base import Backend
from .detectors.base import Detector
from .policy.engine import PolicyEngine
from .transformers.base import Transformer
from .types import Mode, Request, ResponseChunk, Signal


class Gateway:
    """Seven-stage pipeline:

        ingress → context → detect → policy → transform → dispatch → respond
    """

    def __init__(
        self,
        detectors: list[Detector],
        policy: PolicyEngine,
        transformers: dict[Mode, Transformer],
        backends: dict[Mode, Backend],
    ):
        self.detectors = detectors
        self.policy = policy
        self.transformers = transformers
        self.backends = backends

    async def handle(self, request: Request) -> AsyncIterator[ResponseChunk]:
        signals: list[Signal] = []
        for d in self.detectors:
            signals.extend(d.detect(request))

        decision = self.policy.evaluate(request, signals)

        transformer = self.transformers[decision.mode]
        backend = self.backends[decision.mode]

        transformed = transformer.pre_send(request, decision)
        async for chunk in backend.dispatch(transformed):
            yield await transformer.post_receive(chunk, transformed.recovery_state)
