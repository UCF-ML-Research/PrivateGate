from privategate.eval.ablations.probe import run


def _by_name(arms):
    return {a["name"]: a for a in arms}


def test_probe_ablation_runs_both_arms():
    out = run()
    assert out["ablation"] == "semantic_probe"
    arms = _by_name(out["arms"])
    assert {"no_probe", "with_probe"} <= set(arms)


def test_with_probe_routes_more_traffic_to_secure():
    out = run()
    arms = _by_name(out["arms"])
    no_probe = arms["no_probe"]["efficiency"]["secure_path_fraction"]
    with_probe = arms["with_probe"]["efficiency"]["secure_path_fraction"]
    assert with_probe >= no_probe
