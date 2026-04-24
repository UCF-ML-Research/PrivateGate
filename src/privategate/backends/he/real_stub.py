"""Real HE backend integration — STUB.

Exists so the interface contract stays visible even before any real backend
is wired. Swapping `MockHEBackend` for a real implementation (e.g. a
NEXUS client) must be a one-line change in the backend registry — this
file's existence documents that contract.

Candidate real integrations:
  - NEXUS     (https://github.com/zju-abclab/NEXUS)
  - BumbleBee (paper reference impl, check authors' release)
  - CipherGPT (no public code as of latest eprint)

Subclass `HEBackend` and implement the four primitives; the default
`dispatch()` composes them into the streaming contract.
"""
from __future__ import annotations

from typing import AsyncIterator

from .base import HEBackend, HEContext


class NEXUSBackendStub(HEBackend):
    name = "nexus_real"

    def __init__(self, server_url: str, key_bundle_path: str):
        self.server_url = server_url
        self.key_bundle_path = key_bundle_path

    async def setup(self) -> HEContext:
        raise NotImplementedError(
            "Wire the real NEXUS client here. Return an HEContext carrying the "
            "CKKS parameters, client public/secret keys, and any attestation "
            "material returned by the server."
        )

    async def encrypt(self, payload, ctx: HEContext) -> bytes:
        raise NotImplementedError(
            "Encode `payload` into the CKKS ciphertext using ctx.public_key."
        )

    async def infer(
        self, ciphertext: bytes, ctx: HEContext
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError(
            "Stream ciphertext chunks from the remote NEXUS inference server."
        )
        yield b""  # unreachable; keeps typing happy for async-generator return

    async def decrypt(self, ciphertext: bytes, ctx: HEContext) -> str:
        raise NotImplementedError(
            "Decrypt a ciphertext chunk with ctx.secret_key and decode to text."
        )
