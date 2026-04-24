"""InferDPT — RANTEXT perturbation + distillation extraction.

Reference: Tong et al., 2023 (arXiv:2310.12214).
Local PDF: related_works/DP_text/InferDPT.pdf.

STATUS: scaffold. The RANTEXT perturbation mechanism and the extraction
module are large research implementations — plug in the paper's reference
code in `pre_send` / `post_receive` when available.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...types import Decision, Request, TransformedRequest
from .base import LDPTransformer


@dataclass
class InferDPTConfig:
    epsilon: float = 3.0
    mechanism: str = "rantext"            # rantext | santext_plus | custext
    top_k_adjacency: int = 50
    extraction_model: str = "llama-3-8b-instruct"


class InferDPTTransformer(LDPTransformer):
    mechanism = "inferdpt"

    def __init__(self, config: InferDPTConfig | None = None):
        self.config = config or InferDPTConfig()

    def pre_send(self, request: Request, decision: Decision) -> TransformedRequest:
        # TODO(Phase 5 / D4a): replace with RANTEXT perturbation.
        raise NotImplementedError(
            "InferDPT pre_send not implemented. Plug reference RANTEXT "
            "implementation here (arXiv:2310.12214)."
        )
