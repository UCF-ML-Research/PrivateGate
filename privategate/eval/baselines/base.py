"""Common interface every baseline implements.

A baseline takes an `EvalExample` and returns a `BaselineRun` with:
  - the **outbound payload** that left the client (used by privacy
    metrics — anything in here is exposed to the server),
  - the model answer (used by utility metrics),
  - the routing path label (used by efficiency metrics),
  - the wall-clock latency in milliseconds.

Holding all five of {plaintext, full_mask, full_abstract, full_secure,
PrivateGate} to the same interface is what lets the main runner build a
single comparison table.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from privategate.eval.example import EvalExample


@dataclass
class BaselineRun:
    name: str
    outbound_text: str
    answer: str
    routing_path: str
    latency_ms: float


class Baseline(ABC):
    name: str = "baseline"

    @abstractmethod
    def run_one(self, example: EvalExample) -> BaselineRun:
        ...

    def run_all(self, examples) -> list[BaselineRun]:
        return [self.run_one(ex) for ex in examples]


def _timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - start) * 1000.0
