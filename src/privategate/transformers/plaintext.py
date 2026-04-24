from __future__ import annotations

from ..types import Decision, Mode, Request, ResponseChunk, TransformedRequest


class PlaintextTransformer:
    """Identity transformer — used by the PLAINTEXT mode.

    Exists as a named class (rather than absent) so the pipeline can treat
    every mode symmetrically: each mode has exactly one Transformer +
    Backend pair.
    """

    mode = Mode.PLAINTEXT
    mechanism = "identity"

    def pre_send(self, request: Request, decision: Decision) -> TransformedRequest:
        return TransformedRequest(
            mode=Mode.PLAINTEXT,
            payload=request.payload,
            recovery_state={},
        )

    async def post_receive(
        self, chunk: ResponseChunk, recovery_state: dict
    ) -> ResponseChunk:
        return chunk
