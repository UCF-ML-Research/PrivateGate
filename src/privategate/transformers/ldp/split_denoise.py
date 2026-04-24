"""Split-and-Denoise — client runs embedding layer + noise, server denoises.

Reference: Mai et al., ICML 2024 (arXiv:2310.09130).
Local PDF: related_works/DP_text/Split-and-Denoise.pdf.

STATUS: scaffold. The client-side embedding + Gaussian noise injection and
the server-side denoiser contract go here.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...types import Decision, Request, ResponseChunk, TransformedRequest
from .base import LDPTransformer


@dataclass
class SplitDenoiseConfig:
    epsilon: float = 3.0
    embedding_model: str = "bert-base-uncased"   # local embedding layer
    noise_sigma: float | None = None             # derived from ε if None
    denoiser_endpoint: str = "local-mock"


class SplitDenoiseTransformer(LDPTransformer):
    mechanism = "split_and_denoise"

    def __init__(self, config: SplitDenoiseConfig | None = None):
        self.config = config or SplitDenoiseConfig()

    def pre_send(self, request: Request, decision: Decision) -> TransformedRequest:
        # TODO(Phase 5 / D4b): embed locally, inject Gaussian noise, serialize
        # noisy embeddings + token-index map into `payload`.
        raise NotImplementedError(
            "Split-and-Denoise pre_send not implemented. Plug reference "
            "implementation here (arXiv:2310.09130)."
        )

    async def post_receive(
        self, chunk: ResponseChunk, recovery_state: dict
    ) -> ResponseChunk:
        # Server-side denoising is done server-side; client receives text.
        return chunk
