# muteval — limitations & when to distrust the number

muteval is deliberately honest about what it does *not* do. Reading this should
make you trust the tool *more*, not less: a tool that names its own limits is
more reliable than one that pretends to measure everything.

## The three things muteval cannot do for you (read this first)

These are the hard, structural limits — not bugs to be fixed later, but the
nature of the technique. Internalize them before you act on any output.

1. **muteval can't tell you whether a survivor *matters* — only that your eval
   didn't catch it.** A survivor is a *candidate* gap. It's a *real* gap only if
   the injected change (a) can actually happen in your system and (b) would
   actually be wrong if it did. Counter-example: corrupting a retrieved document
   is meaningless when the documents *are* your source of truth — the doc can't
   be "wrong," so a faithfulness metric passing the changed answer is correct
   behavior, not a gap. Every survivor needs a human to ask *"could this happen,
   and would it be bad?"* Many won't survive that question. **muteval surfaces
   candidates; you triage them.**

2. **muteval — and every label-free check — measures coverage/consistency, not
   validity.** Whether an eval is *correct* (agrees with ground truth or a human)
   cannot be measured without labels. The mutation score, judge-reliability,
   judge-bias, and redundancy all run label-free and catch *hygiene* problems
   (does the suite notice a change? is the judge stable / unbiased / redundant?).
   None of them tells you the eval is *right*. Validity requires the
   discrimination / human-agreement checks, which require labeled examples. This
   is the reliability-without-validity wall, and it's fundamental.

3. **muteval is a per-suite diagnostic, not a universal flaw-finder.** It tells
   you where *your specific* suite has a hole. It does not discover new universal
   truths about evaluation — the obvious gaps (faithfulness ≠ correctness,
   reference-free ≠ ground truth) are already well known, and competent teams
   mitigate them with other layers (offline labeled evals, retrieval-quality
   metrics, human review). Read muteval's output as *"here's a hole in this suite
   worth a look,"* never as *"we discovered that evals are broken."*

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
- **A red/errored baseline.** If the suite fails or errors on the *original*
  system, muteval **refuses to emit a score**: it reports the run as `INVALID`,
  writes no badge, and the CLI exits non-zero. It does **not** report a
  misleading 100%. Fix the baseline first.
- **No mutants / no evaluated mutants.** If nothing could be mutated, or every
  mutant errored, there is no evidence — muteval reports `N/A` (not a perfect
  score). Use `--allow-empty` only if a zero-mutant run should pass CI. If only
  meaning-preserving (robustness) operators ran, the status is
  `no_scored_mutants`: they're reported, never scored.
- **Partial mutant errors.** If *some* mutants error (timeouts/API blips), the
  score is computed over a shrunken denominator and is not trustworthy. By
  default muteval **fails closed**: any errored mutant makes the run
  `partial_errors` (CLI exits non-zero, badge `n/a`, terminal shows the partial
  score for diagnosis only). Set an error budget with `--max-error-rate` (or
  `config.max_error_rate`), or `--allow-mutant-errors`, to accept it explicitly.
- **The raw score, when there are observationally-unchanged mutants.** Read the
  **effective** score; the raw one counts mutants whose output didn't change and
  understates good suites.
- **A noisy LLM judge with `runs_per_mutant=1`.** A single flaky verdict can
  flip a mutant. Use `runs_per_mutant > 1` (majority vote) for real judges; watch
  the `flaky` count. With one baseline sample muteval **cannot see** that noise:
  a flaky kill looks like detection. With more samples it can — the baseline is
  graded `runs_per_mutant` times by the **same** majority rule as a mutant (if
  the original would itself be "killed", the run is `baseline_failed`, and the
  report shows the original's pass rate as a noise floor), and a kill whose
  output matches something the original produced is a **noise kill**, dropped
  from the effective score like an inert survivor. `--fail-under` gates on the
  lower of the raw and effective scores, so noise kills can't pass it.
- **A bad judge reply is an error, not a verdict.** The built-in judge asks for
  an integer 0-10 and reads the explicit "N/10" / "N out of 10", else the first
  number. An empty reply, no number, or a value outside 0-10 makes that mutant
  *errored* (counted against `max_error_rate`), never a silent kill. The same
  goes for a non-finite metric score (NaN), and for an eval that returns a
  number or a string instead of a bool/`EvalOutcome` (`0.2` and `"fail"` are
  truthy — they used to count as passes).
- **Judge drift is silent — pin the judge version.** A majority vote stabilizes
  run-to-run noise, but a model-version bump quietly replaces the judge, so a
  killed-rate from last month and this month can be measuring different things.
  muteval records the model under test and its own judge's model
  (`model_under_test` / `judge_models`), but it can't introspect an opaque
  user/deepeval/ragas judge — pin that version yourself when comparing over time.
- **Tied verdicts are `unresolved`, not survivors.** With an even
  `runs_per_mutant`, a mutant the judge caught exactly half the time hasn't
  earned a killed/survived verdict — it's reported as `unresolved` and left out
  of the score (numerator and denominator) rather than counted as a coverage gap.
  A high `unresolved` count means the judge is too noisy at this `runs_per_mutant`
  to decide. Note the number of repeats you need grows fast with the flip rate,
  so an unstable judge is expensive to resolve — the honest read is often "this
  judge can't decide here," not "add more runs." Ties are **gated** like errors:
  by default any tie makes the run `partial_unresolved` (a score over the few
  resolved mutants isn't a score — 1 resolved of 17 would read "100%"), and all
  ties is `no_confident_score`. Use an **odd** `runs_per_mutant` (ties can't
  happen), or accept a budget with `--max-unresolved-rate`.

## Known constraints

- **Third-party judge stability.** The deepeval/ragas adapters are only as
  reliable as those libraries. In testing, deepeval's async path hung on Windows
  and its heaviest calls timed out on Colab. muteval retries and reports
  honestly, but it cannot fix an upstream hang.
- **Rule-based mutations are approximations.** Current operators are synthetic
  string/context edits. They model real regressions but aren't identical to them;
  LLM-driven semantic mutations (roadmap) are more realistic.
- **"Observationally unchanged" ≠ provably equivalent.** A survivor whose output
  matched the baseline is dropped from the effective score. For a *deterministic*
  system that is a true equivalent mutant. For a *stochastic* one (an LLM at
  temperature > 0, a flaky judge), identical output on a few samples does not
  prove the mutant is harmless — it may differ on an unseen sample. muteval
  labels these "observationally unchanged," not "equivalent," on purpose.
- **The reverse also holds: with free-text output, nothing looks unchanged.** By
  default "changed" means *any* difference in the output text, so a mutant whose
  behavior is identical but whose wording drifted (or that just sampled
  differently) counts as a real survivor. On a classifier that emits a label plus
  an explanation, equivalent mutants then show up as coverage gaps. Raising
  `runs_per_mutant` makes this *worse* (more chances to see a wording change).
  Fix it by declaring what the behavior is: `output_key=lambda o: o.split()[0]`
  (compare only the label), and `baseline_runs=N` to sample the baseline's own
  variance: an output matching any baseline sample isn't a change, and on cases
  where the baseline itself varies, an unseen output is reported as
  *undetermined* rather than a confirmed gap. muteval can't derive `output_key`
  from your evals: a survivor passed every eval, so what the evals assert on is
  identical by definition, and comparing only that would mark every survivor
  inert.
- **Robustness operators are reported, not scored.** `paraphrase_instruction`
  and `swap_adjacent_instructions` make meaning-preserving edits, so surviving
  them is the healthy outcome. They never count toward the score; the report
  lists the ones that flipped a verdict ("an eval may key on wording/order").
  muteval can't tell whether that flip is a brittle eval or a system that really
  is sensitive to wording; read the listed mutant to decide. Swapping two steps
  of an ordered procedure *can* be a real regression; the operator assumes an
  unordered rule list.
- **Mutants never delete the input.** A mutant whose prompt lost an input
  placeholder (`{{var}}`, `{var}`, `${VAR}`) is dropped, since it's a guaranteed
  kill that measures nothing. The detection is syntactic: an input spliced in
  some other way (string concatenation in your `run`) isn't protected, and a
  literal `{word}` in the prompt is treated as a placeholder.
- **`downgrade_model` only knows a small model ladder.** It will not guess an
  ordering for models it doesn't recognize (it warns and emits nothing). Pass
  your own strong→weak ladder via `make_downgrade_model([...])`.
- **`downgrade_model` doesn't re-run inference by itself.** Like all System-mode
  operators, it only changes behavior if your `run(system, case)` actually reads
  `system.model` and calls that model.
- **Cost & time.** Real-judge runs cost API money; total work scales with
  mutants × cases × metrics × `runs_per_mutant`. `--concurrency` cuts wall-clock
  (not spend); `--cache` and `--max-calls` cut spend.
- **The cache is only as good as its fingerprint.** Outputs are keyed on the
  system, the case and a fingerprint of `run`; outcomes on the output, the case
  and a fingerprint of the eval (its code, closure values, thresholds, the simple
  globals it reads). An eval that depends on something muteval can't see — a
  file it reads, a remote rubric, an env var — can still serve a stale verdict:
  set a `cache_version` string on it, or don't use `--cache`. A `run()` that
  closes over changing state (a call counter) gets a new fingerprint each time,
  so it just misses. Outputs that can't round-trip through JSON aren't cached.
- **`--concurrency` needs thread-safe evals.** Each (mutant, case, run) gets a
  private copy of the case, and the deepeval adapter measures a private copy of
  its metric, so the built-ins are safe. Your own eval that keeps state on an
  object shared across calls must be thread-safe, or leave concurrency at 1.
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
