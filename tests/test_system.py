from muteval.system import System, as_system


def test_as_system_promotes_string():
    sys_ = as_system("be helpful")
    assert isinstance(sys_, System)
    assert sys_.prompt == "be helpful"
    assert sys_.context is None


def test_as_system_passes_through_system():
    s = System(prompt="x", context=("doc",))
    assert as_system(s) is s


def test_with_prompt_preserves_other_fields():
    s = System(prompt="orig", context=("a", "b"), model="gpt-4o")
    s2 = s.with_prompt("new")
    assert s2.prompt == "new"
    assert s2.context == ("a", "b")
    assert s2.model == "gpt-4o"
    # Original is unchanged (frozen / immutable).
    assert s.prompt == "orig"


def test_with_context_tuples_the_input():
    s = System(prompt="p").with_context(["d1", "d2"])
    assert s.context == ("d1", "d2")
    assert s.with_context(None).context is None


def test_key_distinguishes_systems():
    a = System(prompt="p", context=("x",))
    b = System(prompt="p", context=("y",))
    c = System(prompt="p", context=("x",))
    assert a.key() != b.key()
    assert a.key() == c.key()


def test_as_system_rejects_bad_type():
    try:
        as_system(123)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")
