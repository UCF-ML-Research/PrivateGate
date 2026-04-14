from privategate.eval.ablations.policy import run


def _by_name(arms):
    return {a["name"]: a for a in arms}


def test_policy_ablation_three_arms():
    out = run()
    arms = _by_name(out["arms"])
    assert {"strict", "balanced", "permissive"} <= set(arms)


def test_strict_routes_everything_with_spans_to_secure():
    out = run()
    arms = _by_name(out["arms"])
    strict_secure = arms["strict"]["efficiency"]["secure_path_fraction"]
    permissive_secure = arms["permissive"]["efficiency"]["secure_path_fraction"]
    assert strict_secure >= permissive_secure


def test_permissive_leaks_more_than_strict():
    out = run()
    arms = _by_name(out["arms"])
    strict_leak = arms["strict"]["privacy"]["category_weighted_leakage"]
    permissive_leak = arms["permissive"]["privacy"]["category_weighted_leakage"]
    assert permissive_leak >= strict_leak
