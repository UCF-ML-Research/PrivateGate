from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Decision, Mode, Request, ResponseChunk, TransformedRequest


@runtime_checkable
class Transformer(Protocol):
    """Per-mode pre-send and post-receive logic.

    A Transformer owns *both* directions so rehydration, decryption,
    and attestation verification stay co-located with the transform
    that made them necessary.

    Examples:
      REDACT   — mask PII on pre_send, rehydrate placeholders on post_receive.
      PERTURB  — inject calibrated token/embedding noise; identity on return.
      TEE      — handshake + encrypt under attested enclave key; verify
                 quote on return.
      HE_SMPC  — encrypt/secret-share query; decrypt/reconstruct response.
    """

    mode: Mode

    def pre_send(self, request: Request, decision: Decision) -> TransformedRequest: ...

    async def post_receive(
        self, chunk: ResponseChunk, recovery_state: dict
    ) -> ResponseChunk: ...
