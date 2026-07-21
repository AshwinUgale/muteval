# muteval — limitations & when to distrust the number

muteval is deliberately honest about what it does *not* do. Reading this should
make you trust the tool *more*, not less: a tool that names its own limits is
more reliable than one that pretends to measure everything.

## What muteval needs (and where it doesn't apply)

muteval mutates the **system under test** and reruns your **eval suite**, so it
only applies when both exist:

1. **A re-runnable system.** muteval degrades the prompt/context/tools/model and
   needs a *fresh* output for each mutant. If all you have is a cached CSV of
   outputs with no way to re-invoke the system, muteval can't help.
2. **A programmatic, output-grading eval.** Any `(output, case) -> pass/fail`
   works (hand-written, `checks`, deepeval/ragas metrics). It does **not** apply
   to:
   - **Model benchmarks** (MMLU, HumanEval) — input-driven, no system to mutate.
   - **Human / preference / A-B / Elo eval** — you can't re-run a human per mutant.
   - **Production / online monitoring** — muteval is offline/CI, not observability.

## When to distrust the number

- **Too few mutants/cases → wide CI.** The score is a proportion; a handful of
  mutants gives a near-useless interval (e.g. `50% [9-91%]`). **Trust the CI, not
  the point estimate.** Add cases/mutants for a tighter number.
- **A red/errored baseline.** If the suite errors on the original system (e.g. a
  flaky judge timeout), muteval flags the run as unreliable — the score is not
  trustworthy until the baseline is green.
- **The raw score, when there are inert mutants.** Read the **effective** score;
  the raw one counts equivalent (output-unchanged) mutants and understates good
  suites.
- **A noisy LLM judge with `runs_per_mutant=1`.** A single flaky verdict can
  flip a mutant. Use `runs_per_mutant > 1` (majority vote) for real judges; watch
  the `flaky` count.

## Known constraints

- **Third-party judge stability.** The deepeval/ragas adapters are only as
  reliable as those libraries. In testing, deepeval's async path hung on Windows
  and its heaviest calls timed out on Colab. muteval retries and reports
  honestly, but it cannot fix an upstream hang.
- **Rule-based mutations are approximations.** Current operators are synthetic
  string/context edits. They model real regressions but aren't identical to them;
  LLM-driven semantic mutations (roadmap) are more realistic.
- **Cost & time.** Real-judge runs cost API money and run sequentially; total
  work scales with mutants × cases × metrics × `runs_per_mutant`.
- **Non-prompt targets need System mode + a compatible `run()`.** Context/tool/
  model mutation only affects output if your `run(system, case)` actually consumes
  the mutated `System`.

## What the score does and does NOT mean

- **Does:** measure how many *injected regressions your eval suite caught* — i.e.
  eval **coverage** of degradations.
- **Does not:** measure the correctness/safety of your *system*, or whether your
  metric agrees with *humans* (that's validity — the optional human-agreement
  probe, which needs labels).

## The honest summary

Trust muteval's number when: the **baseline is green**, you have **enough
mutants for a tight CI**, you read the **effective** score, and (for real judges)
you used **`runs_per_mutant > 1`**. Outside that, treat the number as
directional and lean on the survivor list — a concrete, reproducible gap is
useful even when the exact percentage isn't.
