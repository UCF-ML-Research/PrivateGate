from __future__ import annotations

from ...types import Decision, Mode, Request, ResponseChunk, TransformedRequest


class LDPTransformer:
    """Base for local-differential-privacy text transformers.

    Concrete mechanisms (InferDPT, Split-and-Denoise, SANTEXT, ...) implement
    `pre_send` to perturb the prompt and attach any state needed to interpret
    the server's reply.
    """

    mode = Mode.PERTURB
    mechanism: str = "abstract"

    def pre_send(self, request: Request, decision: Decision) -> TransformedRequest:
        raise NotImplementedError

    async def post_receive(
        self, chunk: ResponseChunk, recovery_state: dict
    ) -> ResponseChunk:
        # Default: no post-processing. Subclasses that return denoised or
        # re-assembled text should override.
        return chunk
