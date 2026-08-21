"""deny_tool_output — domain-failure-as-transport-success tool mutation.

Pure muteval-core operator test (no tracelint dependency).
"""

from muteval.mutators import OPERATORS, generate_mutants
from muteval.severity import HIGH, severity_of
from muteval.system import System

_SYS = System(prompt="p", tools=('{"status": "succeeded", "receipt": "R1"}',))


def test_registered():
    assert "deny_tool_output" in OPERATORS


def test_injects_declined_status():
    muts = generate_mutants(_SYS, operators=["deny_tool_output"])
    assert len(muts) == 1
    assert '"declined"' in muts[0].system.tools[0]
    assert muts[0].target == "tools"
    assert muts[0].operator == "deny_tool_output"


def test_one_mutant_per_tool_output():
    sys2 = System(prompt="p", tools=("a", "b", "c"))
    assert len(generate_mutants(sys2, operators=["deny_tool_output"])) == 3


def test_noop_without_tools():
    assert generate_mutants(System(prompt="p"), operators=["deny_tool_output"]) == []


def test_is_high_severity():
    m = generate_mutants(_SYS, operators=["deny_tool_output"])[0]
    assert severity_of(m) == HIGH
