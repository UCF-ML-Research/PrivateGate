from privategate.eval.metrics.efficiency import (
    latency_percentiles,
    secure_path_fraction,
)


def test_percentiles_single_sample():
    r = latency_percentiles([5.0])
    assert r.p50 == 5.0 and r.p95 == 5.0 and r.mean == 5.0 and r.n == 1


def test_percentiles_sorted_input():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    r = latency_percentiles(samples)
    # nearest-rank with banker's rounding lands on index 4 -> 5.0
    assert r.p50 == 5.0
    assert r.p95 >= 9.0
    assert abs(r.mean - 5.5) < 1e-9
    assert r.n == 10


def test_secure_fraction_all_secure():
    assert secure_path_fraction(["secure", "secure"]) == 1.0


def test_secure_fraction_mixed():
    assert secure_path_fraction(["secure", "standard", "standard"]) == 1 / 3


def test_empty_inputs_safe():
    r = latency_percentiles([])
    assert r.n == 0
    assert secure_path_fraction([]) == 0.0
