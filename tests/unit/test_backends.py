from __future__ import annotations

import asyncio

import pytest

from privategate.backends.he import MockHEBackend, MockHEConfig, NEXUSBackendStub
from privategate.transformers.he.mock import MockHETransformer
from privategate.transformers.plaintext import PlaintextTransformer
from privategate.types import Context, Decision, Mode, Request


def _req(text: str = "hello") -> Request:
    return Request(payload=text, context=Context())


def _decision(mode: Mode) -> Decision:
    return Decision(mode=mode, matched_rule=None, rationale="t", signals=(), confidence=1.0)


def test_plaintext_transformer_identity():
    t = PlaintextTransformer()
    tr = t.pre_send(_req("hi"), _decision(Mode.PLAINTEXT))
    assert tr.mode == Mode.PLAINTEXT
    assert tr.payload == "hi"
    assert tr.recovery_state == {}


def test_mock_he_transformer_preserves_prompt():
    t = MockHETransformer()
    tr = t.pre_send(_req("hello"), _decision(Mode.HE_SMPC))
    assert tr.mode == Mode.HE_SMPC
    assert "hello" in str(tr.payload)


def test_mock_he_backend_dispatch_streams_and_finalizes():
    cfg = MockHEConfig(profile="nexus", response_tokens=16, chunk_tokens=8, realtime=False)
    backend = MockHEBackend(cfg)

    async def collect():
        transformer = MockHETransformer()
        tr = transformer.pre_send(_req("demo"), _decision(Mode.HE_SMPC))
        out = []
        async for chunk in backend.dispatch(tr):
            out.append(chunk)
        return out

    chunks = asyncio.run(collect())
    assert len(chunks) >= 2
    assert chunks[-1].is_final is True
    # Non-final chunks should carry simulated attestation metadata.
    non_final = [c for c in chunks if not c.is_final]
    assert non_final and all(c.attestation and c.attestation.get("simulated") is True for c in non_final)
    # Simulated latency fields should be populated on the final metadata.
    assert "infer_ms_total" in chunks[-1].attestation
    assert chunks[-1].attestation["infer_ms_total"] > 0


def test_mock_he_backend_rejects_unknown_profile():
    with pytest.raises(ValueError):
        MockHEBackend(MockHEConfig(profile="totally-fake"))


def test_real_stub_implements_same_interface_as_mock():
    """Swapping MockHEBackend for NEXUSBackendStub must be a drop-in."""
    mock_methods = {"setup", "encrypt", "infer", "decrypt", "dispatch"}
    for method in mock_methods:
        assert callable(getattr(NEXUSBackendStub, method))
    stub = NEXUSBackendStub(server_url="https://example", key_bundle_path="/dev/null")
    assert stub.mode == Mode.HE_SMPC
