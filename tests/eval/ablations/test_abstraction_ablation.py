from privategate.eval.ablations.abstraction import run


def _by_name(arms):
    return {a["name"]: a for a in arms}


def test_abstraction_ablation_three_arms():
    out = run()
    arms = _by_name(out["arms"])
    assert {"abstract_low", "abstract_medium", "abstract_high"} <= set(arms)


def test_low_abstraction_leaks_more_than_high():
    out = run()
    arms = _by_name(out["arms"])
    low_leak = arms["abstract_low"]["privacy"]["category_weighted_leakage"]
    high_leak = arms["abstract_high"]["privacy"]["category_weighted_leakage"]
    assert low_leak >= high_leak
