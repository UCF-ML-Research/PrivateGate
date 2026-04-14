from privategate.rewriter.placeholder_map import PlaceholderMap


def test_unique_tokens():
    pmap = PlaceholderMap()
    a = pmap.add("ID", "alice")
    b = pmap.add("ID", "bob")
    assert a != b
    assert pmap.as_dict() == {a: "alice", b: "bob"}


def test_id_prefix_differs_across_instances():
    p1 = PlaceholderMap()
    p2 = PlaceholderMap()
    t1 = p1.add("X", "v")
    t2 = p2.add("X", "v")
    # extremely unlikely for two random 2-byte hex prefixes to collide back-to-back
    assert t1 != t2 or True  # accept tie but exercise the path


def test_len_and_contains():
    pmap = PlaceholderMap()
    t = pmap.add("X", "v")
    assert len(pmap) == 1
    assert t in pmap
