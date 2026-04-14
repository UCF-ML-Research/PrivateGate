from privategate.eval.datasets.medqa import load_medqa_subset


def test_loads_fixture_with_qa_fields():
    examples = load_medqa_subset()
    assert examples[0].task == "qa"
    assert examples[0].question is not None
    assert examples[0].reference_answer is not None
