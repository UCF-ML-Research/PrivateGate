"""Abstract HE backend.

The interface is deliberately shaped around four phases so that a real
NEXUS / BumbleBee / CipherGPT integration drops in by implementing each
phase without touching any other part of the gateway pipeline. The mock
implementation simulates latency from published benchmark numbers; the
real-integration stub shares the exact same contract.

    setup()     — key-exchange + (optional) attestation (once per session)
    encrypt()   — encode the prompt into ciphertext / CKKS packing
    infer()     — remote HE inference, yields ciphertext chunks
    decrypt()   — recover plaintext client-side

A default `dispatch()` composes the four phases into the `Backend` protocol's
streaming contract so subclasses only implement primitives.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from ...types import Mode, ResponseChunk, TransformedRequest


@dataclass
class HEContext:
    """Per-session state carried across setup → encrypt → infer → decrypt."""

    session_id: str
    public_key: Any = None
    secret_key: Any = None
    params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class HEBackend(abc.ABC):
    """Abstract base for HE backends.

    Both the mock and any real implementation subclass this; swapping one
    for the other is a configuration change, not a pipeline change.
    """

    mode = Mode.HE_SMPC
    name: str = "abstract"

    @abc.abstractmethod
    async def setup(self) -> HEContext: ...

    @abc.abstractmethod
    async def encrypt(self, payload: str | bytes, ctx: HEContext) -> bytes: ...

    @abc.abstractmethod
    async def infer(
        self, ciphertext: bytes, ctx: HEContext
    ) -> AsyncIterator[bytes]: ...

    @abc.abstractmethod
    async def decrypt(self, ciphertext: bytes, ctx: HEContext) -> str: ...

    async def dispatch(
        self, request: TransformedRequest
    ) -> AsyncIterator[ResponseChunk]:
        ctx = await self.setup()
        ct_in = await self.encrypt(request.payload, ctx)  # type: ignore[arg-type]
        async for ct_chunk in self.infer(ct_in, ctx):
            plain = await self.decrypt(ct_chunk, ctx)
            yield ResponseChunk(
                data=plain,
                is_final=False,
                attestation=dict(ctx.metadata),
            )
        yield ResponseChunk(data="", is_final=True, attestation=dict(ctx.metadata))
