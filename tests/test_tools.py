"""B4: tool-output operators (agents)."""

from muteval.mutators import (
    corrupt_tool_output,
    drop_tool_output,
    swap_tool_output,
)
from muteval.system import System


def _sys():
    return System(prompt="agent", tools=("search() -> 42 results", "lookup() -> ok"))


def test_drop_tool_output_one_at_a_time():
    ms = drop_tool_output(_sys())
    assert len(ms) == 2
    assert all(m.target == "tools" for m in ms)
    assert all(len(m.system.tools) == 1 for m in ms)


def test_corrupt_tool_output_changes_a_number():
    ms = corrupt_tool_output(_sys())
    assert len(ms) == 1                       # only the doc with a number
    assert "42" not in ms[0].system.tools[0]


def test_swap_tool_output_replaces_each():
    ms = swap_tool_output(_sys())
    assert len(ms) == 2
    for m in ms:
        assert any("unrelated" in str(t) for t in m.system.tools)


def test_tool_operators_noop_without_tools():
    for op in (drop_tool_output, corrupt_tool_output, swap_tool_output):
        assert op("just a prompt") == []
        assert op(System(prompt="p")) == []
