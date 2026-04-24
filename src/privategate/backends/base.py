from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from ..types import Mode, ResponseChunk, TransformedRequest


@runtime_checkable
class Backend(Protocol):
    """Transport-level destination for a given mode.

    Backends are strictly streaming — HE/SMPC latency makes non-
    streaming dispatch unusable. For non-streaming providers, yield
    exactly one final chunk.
    """

    mode: Mode

    def dispatch(
        self, request: TransformedRequest
    ) -> AsyncIterator[ResponseChunk]: ...
