from privategate.eval.ablations.detector import run


def _by_name(arms):
    return {a["name"]: a for a in arms}


def test_detector_ablation_runs_both_arms():
    out = run()
    assert out["ablation"] == "detector"
    arms = _by_name(out["arms"])
    assert {"rule_only", "rule_plus_ml"} <= set(arms)


def test_hybrid_does_not_increase_leakage():
    out = run()
    arms = _by_name(out["arms"])
    rule = arms["rule_only"]["privacy"]["span_exposure_rate"]
    hybrid = arms["rule_plus_ml"]["privacy"]["span_exposure_rate"]
    assert hybrid <= rule
