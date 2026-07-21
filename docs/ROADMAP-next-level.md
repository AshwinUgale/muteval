# muteval next-level roadmap: (1) best muteval → (2) eval evaluator → (3) adoption

Where we are: the A/B plan (`docs/PLAN-A-scope-B-system-mutation.md`) is COMPLETE —
18 operators (prompt/context/tool/model), deepeval + ragas adapters, severity +
`--fail-on-severity` gate, output-diffing (raw vs effective score), System mode,
112 tests. The eval-quality claim is proven AND CI-enforced: effective score
0→33→67→100% as suite coverage grows (`validation/eval_quality_experiment/`,
`tests/test_eval_quality.py`).

**Guiding rule: don't advance a phase until the previous one is genuinely
trustworthy.** There is a hard TRUST GATE between Phase 1 and Phase 2.

---

## Phase 1 — Best muteval (make the mutation core trustworthy + complete)

Bar: *you* would stake a claim on the number, and muteval covers the whole
system-under-test (prompt/context/tools/model + trajectory).

- [x] **1.1 Statistical stability  [P0]  — DONE**
  Majority-vote verdicts (fixes the any-run-kills bias) + Wilson CI on
  the score, reported in the terminal. `src/muteval/stats.py`,
  `config.kill_threshold`, `MutationResult.score_ci/effective_score_ci/flaky`.
  ORIGINAL NOTE:
  Aggregate `runs_per_mutant`: a mutant is killed if the suite fails in ≥X% of N
  runs; report a confidence interval on the score. A real-judge re-run must not
  swing. Files: `runner.py` (aggregate verdicts), `report.py` (show CI).
  Done = same suite, re-run, same score within a stated interval.
- [ ] **1.2 Second domain in the eval-quality experiment**
  Add a different system + graded suites (e.g. code-gen or medical QA) and extend
  `tests/test_eval_quality.py` to assert the 0→100 monotonic relationship there
  too. Kills the "n=1, you rigged one example" doubt.
- [ ] **1.3 One real LLM-judge validation, finished end to end**
  Run `validation/deepeval_rag_system/` (and ragas) to completion in Colab with a
  gpt-4o judge; record effective score + survivor list in NOTES. Bridges the
  controlled experiment to real metrics.
- [ ] **1.4 Agent evals — the Trace extension (completes the target)**
  `Trace` (final_output + steps), made str-compatible so existing string checks
  still work; trajectory-aware evals; new operators `mutate_tool_description` and
  `drop_tool`; trace-aware output-diffing. Prove it OFFLINE first:
  `examples/agent_offline/` where a mutated tool description survives a weak eval
  → [HIGH] survivor (mirror `rag_context_offline`). Files: new `trace.py`,
  `mutators.py`, `runner.py`, `config.py`.
- [ ] **1.5 LIMITATIONS doc / README section**
  Honest scope: the re-run requirement, offline/CI-only, and where muteval does
  NOT apply (model benchmarks, human/preference eval, production monitoring).
  Documenting limits *increases* trust.
- [ ] **1.6 LLM-driven semantic mutations (behind a `[llm]` extra)**
  More realistic regressions than rule-based edits — raises the credibility that
  "these mutations are failures people actually hit."

### TRUST GATE (must all hold before Phase 2)
stable score (1.1) · generalizes across ≥2 domains (1.2) · at least one real
LLM-judge result (1.3) · limits documented (1.5). If you wouldn't trust the
number yourself, do not broaden the product.

---

## Phase 2 — More than muteval: the Eval Evaluator (multi-technique report card)

Bar: muteval rates an eval suite across several honest dimensions — mutation
testing is the flagship, not the only lens. Broadens technique -> category (eval
quality), stays tool-agnostic. **NOT a full eval framework.**

Framing: **each probe detects a different way an eval can be bad.** Mutation
catches "it wouldn't notice a regression"; the others catch other defects.

- [ ] **2.1 Probe interface** — pluggable `Probe` registry mirroring `OPERATORS`;
  each probe returns a scored, named finding. New `probes/` module. Report
  aggregates them into a card (no single fake composite score).

- [ ] **2.2 Statistical-adequacy probe  [P0 — also powers Phase 1.1]**
  *Bad eval:* too few cases to trust the number (19/20 is not "95%"; the Wilson
  95% CI is ~[76%, 99%]).
  *Detect:* binomial confidence interval (Wilson; Clopper-Pearson exact) on the
  pass rate; compute required-n for a target claim.
  *Rate:* report `rate [CI, n]`; flag when the CI straddles the threshold, or
  show "need ~75 cases to defend >90%, you have 20". The mutation score and any
  judge pass-rate are proportions too, so this CI machinery is reused everywhere.

- [ ] **2.3 Judge-reliability probe  [P0 — cheap + strong]**
  *Bad eval:* the LLM-judge gives a different verdict on a re-run.
  *Detect:* run each metric N times on the SAME fixed outputs; measure verdict-flip
  rate / variance (a proportion -> CI applies).
  *Rate:* "faithfulness flips its verdict on 18% of cases between runs."

- [ ] **2.4 Discrimination / separability probe  [P0]**
  *Bad eval:* the metric can't tell a good answer from a bad one (useless at any
  threshold).
  *Detect:* feed known-good and known-bad outputs; measure the score gap /
  effect size / AUC.
  *Rate:* "good avg 0.82 vs bad avg 0.79 — barely separates; not measuring
  quality."

- [ ] **2.5 Threshold-calibration probe**
  *Bad eval:* pass/fail line in the wrong place.
  *Detect:* sweep the threshold; watch the pass rate and where good/bad actually
  separate.
  *Rate:* "real boundary ~0.85, your threshold 0.70 -> passing bad answers."

- [ ] **2.6 Redundancy probe**
  *Bad eval:* N metrics that all measure the same thing (false coverage, wasted
  cost).
  *Detect:* correlate metric scores across cases.
  *Rate:* "answer_relevancy vs custom_relevancy correlate 0.97 -> one redundant."

- [ ] **2.7 Human-agreement (validity) probe — OPTIONAL, needs a small labeled sample**
  *Bad eval:* the metric doesn't match what a human would say (the gold standard
  of validity).
  *Needs:* human labels — by definition you can't measure agreement-with-humans
  without humans. BUT: (a) only a *sample* (~30-50 labeled examples gives a usable
  Cohen's kappa + CI), not every case; (b) fully optional — skip the probe if no
  labels, and the card says "validity: not assessed".
  *Low-friction UX:* muteval emits a labeling worksheet
  (sampled case / output / metric_verdict); the user fills a `human_label`
  column; feed it back -> muteval computes kappa + CI. ~20 min of labeling, once.
  *Label-free proxy (separate, weaker claim):* agreement between the cheap judge
  and a STRONGER/ensemble judge ("judge-vs-reference agreement"). Cheap, but it is
  NOT human validity — label it honestly as a proxy.

- [ ] **2.8 Report-card output** — aggregate probes into a card of
  separately-interpretable signals. Mutation stays the headline section.

---

## Phase 3 — Adoption & UX

Bar: effortless to run, discoverable, sticky.

- [ ] **3.1 New-user path audit  [P0]** — fresh env → `pip install muteval` →
  offline example → first meaningful result in < 2 min. Fix every friction point;
  polish the README as the storefront.
- [ ] **3.2 GitHub Action + score badge** — one-line CI integration; a viral
  `eval coverage: X%` README badge.
- [ ] **3.3 HTML/Markdown report + JSON history** — the local "dashboard" (static,
  no SaaS).
- [ ] **3.4 Suggested-eval-per-survivor** — each survivor → a starter check that
  would kill it. Closes diagnostic → fix.
- [ ] **3.5 Docs site.**
- [ ] **3.6 promptfoo adapter** — cross-tool reach for the writeup.
- [ ] **3.7 Blog + outreach** — see `docs/OUTREACH-targets-and-plan.md`.

---

## Suggested execution order
1.1 (stability) → 1.4 (offline agent proof) → 1.2 + 1.3 (generalize + real judge)
→ **TRUST GATE** → 2.2 (reliability probe) → 2.3 (discrimination) → 2.6 (report
card). Phase 3 (esp. 3.1 usability + 3.2 badge) can ride alongside once Phase 1
is trusted — but the trust gate governs the *product* story.
