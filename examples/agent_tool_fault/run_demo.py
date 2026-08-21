"""The whole muteval<->tracelint story in one script — offline, no API key.

    python examples/agent_tool_fault/run_demo.py     # needs: pip install muteval[tracelint]

It runs muteval TWICE against the same naive payments agent and the same
`deny_tool_output` mutant:

  1. semantic eval ALONE  -> the declined charge SURVIVES (a [HIGH] coverage gap)
  2. + checks.tracelint()  -> the declined charge is KILLED (0% -> 100%)

...and prints the side-by-side verdicts on the mutant so you can see WHY: the
user's eval passes (the answer still reads "successful") while tracelint fails
(the trace carries a declined charge its declared contract catches).
"""

import sys

from muteval import MutEvalConfig, checks
from muteval.mutators import generate_mutants
from muteval.runner import run_mutation_testing

sys.path.insert(0, __file__.rsplit("run_demo.py", 1)[0] or ".")
from muteval_config import CASES, SYSTEM, TOOL_REGISTRY, run, user_eval  # noqa: E402

try:
    tracelint_eval = checks.tracelint(registry=TOOL_REGISTRY)
except Exception as exc:  # pragma: no cover
    print(f"needs the extra: pip install muteval[tracelint]  ({exc})")
    raise SystemExit(1)


def _mk(evals, names):
    return MutEvalConfig(
        system=SYSTEM, cases=CASES, run=run, evals=evals, eval_names=names,
        operators=["deny_tool_output"],
    )


def _pct(score):
    return "n/a" if score is None else f"{round(score * 100)}%"


# --- the two runs -----------------------------------------------------------
before = run_mutation_testing(_mk([user_eval], ["user_eval:confirms_success"]))
after = run_mutation_testing(
    _mk([user_eval, tracelint_eval], ["user_eval:confirms_success", "tracelint"])
)

# --- the side-by-side verdicts on the mutant --------------------------------
mutant = generate_mutants(SYSTEM, operators=["deny_tool_output"])[0]
case = CASES[0]
mutant_out = run(mutant.system, case)
u_pass = user_eval(mutant_out, case).passed
t_pass = tracelint_eval(mutant_out, case).passed

print("\nmuteval + tracelint - a silent tool fault your evals miss\n")
print(f"  mutation: {mutant.description}")
print("  the agent's answer on the mutant: still reads clean (naive proceed)\n")
print("  verdicts on the DECLINED-charge mutant:")
print(f"    user_eval (confirms 'successful')  -> {'PASS  <- misses it' if u_pass else 'FAIL'}")
print(f"    checks.tracelint (declared contract) -> {'FAIL  <- kills it' if not t_pass else 'PASS'}\n")
print("  muteval score:")
print(f"    semantic eval alone      : {_pct(before.effective_score)}  "
      f"({len(before.real_survivors)} survivor)  <- the coverage gap")
print(f"    + checks.tracelint()     : {_pct(after.effective_score)}  "
      f"({len(after.real_survivors)} survivors) <- gap closed\n")
for s in before.real_survivors:
    print(f"  [{(s.severity or 'medium').upper()}] SURVIVED  [{s.mutant.operator}]  {s.mutant.description}")
print("\n  A declined charge the agent reported as success passed the semantic")
print("  suite; only the structural trace check with the tool's failure contract")
print("  caught it. That is the eval blind spot muteval surfaces.\n")
