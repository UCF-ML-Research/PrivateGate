from privategate.eval.baselines.full_abstract import FullAbstractBaseline
from privategate.eval.baselines.full_mask import FullMaskBaseline
from privategate.eval.baselines.full_secure import FullSecureBaseline
from privategate.eval.baselines.plaintext import PlaintextBaseline
from privategate.eval.baselines.privategate import PrivateGateBaseline
from privategate.eval.datasets.synthetic_mixed import load_synthetic_mixed


def _credential_example():
    for ex in load_synthetic_mixed():
        if any(s.category.value == "CREDENTIAL" for s in ex.spans):
            return ex
    raise AssertionError("seed dataset must contain a credential example")


def _ssn_example():
    for ex in load_synthetic_mixed():
        if "123-45-6789" in ex.text:
            return ex
    raise AssertionError("seed dataset must contain the SSN example")


def test_plaintext_leaks_everything():
    ex = _ssn_example()
    run = PlaintextBaseline().run_one(ex)
    assert "123-45-6789" in run.outbound_text
    assert run.routing_path == "standard"


def test_full_mask_redacts_ssn():
    ex = _ssn_example()
    run = FullMaskBaseline().run_one(ex)
    assert "123-45-6789" not in run.outbound_text


def test_full_abstract_replaces_medical_terms():
    examples = load_synthetic_mixed()
    medical = next(ex for ex in examples if any(s.category.value == "MEDICAL" for s in ex.spans))
    run = FullAbstractBaseline().run_one(medical)
    assert "diabetes" not in run.outbound_text.lower()


def test_full_secure_leaks_nothing_outbound():
    ex = _ssn_example()
    run = FullSecureBaseline().run_one(ex)
    assert run.outbound_text == ""
    assert run.routing_path == "secure"


def test_privategate_routes_credential_to_secure_with_no_outbound_leak():
    ex = _credential_example()
    run = PrivateGateBaseline().run_one(ex)
    assert run.routing_path == "secure"
    # the credential plaintext must not appear in the outbound (transformed) text
    cred_value = next(s.text for s in ex.spans if s.category.value == "CREDENTIAL")
    assert cred_value not in run.outbound_text


def test_privategate_keeps_clean_query_unchanged():
    examples = load_synthetic_mixed()
    clean = next(ex for ex in examples if not ex.spans)
    run = PrivateGateBaseline().run_one(clean)
    assert run.routing_path == "standard"
    assert run.outbound_text == clean.text
