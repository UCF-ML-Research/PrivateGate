from privategate.eval.datasets.wikipii import load_wikipii


def test_loads_fixture():
    examples = load_wikipii()
    assert len(examples) >= 1
    assert "Einstein" in examples[0].text
