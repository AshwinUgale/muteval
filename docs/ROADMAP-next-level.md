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

Bar: *you* would stake a claim on the number. This phase is about **trust in
the EXISTING capability** (prompt/context/tool/model mutation) — reliability and
honesty — NOT adding new capabilities. New capabilities (agents, LLM-driven
mutations) are Phase 1.5, after the trust gate.

- [x] **1.1 Statistical stability  [P0]  — DONE**
  Majority-vote verdicts (fixes the any-run-kills bias) + Wilson CI on
  the score, reported in the terminal. `src/muteval/stats.py`,
  `config.kill_threshold`, `MutationResult.score_ci/effective_score_ci/flaky`.
  ORIGINAL NOTE:
  Aggregate `runs_per_mutant`: a mutant is killed if the suite fails in ≥X% of N
  runs; report a confidence interval on the score. A real-judge re-run must not
  swing. Files: `runner.py` (aggregate verdicts), `report.py` (show CI).
  Done = same suite, re-run, same score within a stated interval.
- [x] **1.2 Second domain in the eval-quality experiment — DONE**
  Code-review assistant (`run_experiment_codereview.py`): 0->35->71->100%,
  monotonic. `tests/test_eval_quality.py` now enforces the 0->100 relationship
  across BOTH domains (parametrized). Kills the "n=1" doubt.
- [~] **1.3 One real LLM-judge validation — MECHANISM CONFIRMED**
  Ran `validation/deepeval_rag_system/` on real deepeval metrics (Colab, gpt-4o-mini).
  The poisoned-retrieval survivor (drop/swap context) reproduced ~5x. A clean
  GREEN baseline was blocked by deepeval's timeout instability (the baseline is
  its heaviest call) — a deepeval issue, not muteval. muteval added baseline-retry
  and behaved correctly (retried, flagged unreliable). See its NOTES.md. A clean
  published number needs a stabler env; the *finding* is solid.
- [x] **1.4 LIMITATIONS doc / README section — DONE**
  `docs/LIMITATIONS.md` (+ README link): re-run requirement, when to distrust
  the number (CI/baseline/effective/noise), third-party stability, what the
  score does/doesn't mean.
### TRUST GATE (must all hold before Phase 2)
stable score (1.1) · generalizes across ≥2 domains (1.2) · at least one real
LLM-judge result (1.3) · limits documented (1.4). If you wouldn't trust the
number yourself, do not broaden the product.

---

## Phase 1.5 — Capability expansion (AFTER the trust gate)

New capabilities, only once the core is trusted. These add *scope*, not *trust*.

**Decision (recorded): agent evals is the STRATEGIC DESTINATION, not the next
build.** It's core to the grand vision — the thesis is strongest where evals are
hardest (agents), the market is heading there, it's the least-saturated corner,
and it's the deepest moat / a potential headline ("mutation testing for AGENT
evals"). But it's a big, hard-to-validate build and few teams have agent eval
suites *today*. So: build it when PULLED — by real user demand ("does this work
for my agent?") or as a deliberate mindshare bet once we can demonstrate the gap
convincingly (needs a believable agent + agent eval suite, harder than the RAG
case). Do NOT let it leapfrog getting the trusted core in front of real users.

- [ ] **Agent evals — the `Trace` extension.** `Trace` (final_output + steps),
  str-compatible so existing checks still work; trajectory-aware evals;
  trace-aware output-diffing; operators `mutate_tool_description` / `drop_tool`.
  Prove OFFLINE first (`examples/agent_offline/`): a mutation changes the
  tool-call trace, the final answer is unchanged, a weak eval misses it ->
  [HIGH] survivor. (Groundwork exists: `System.tools` + tool-output operators.)
- [ ] **LLM-driven semantic mutations (behind a `[llm]` extra).** More realistic
  regressions than rule-based edits.

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

- [~] **3.1 New-user path audit  [P0] — AUDITED + FIXED (1 user action left)**
  Empirical audit in a clean venv found: (a) [P0] `pip install muteval` returns
  the stale 0.0.1 placeholder (no System/checks/adapters) — the README's
  "pip install and run" is currently FALSE; (b) [P0] `muteval init` gave a
  confusing 0%/all-inert first result (placeholder run ignored the prompt);
  (c) [P1] bare `muteval` errored instead of showing help.
  FIXED (b)+(c): scaffold run() now reflects the prompt -> meaningful keyless
  first result (13%, 7 survivors); bare `muteval` prints help. Version bumped
  0.0.1 -> 0.1.0.
  REMAINING USER ACTION: republish to PyPI so `pip install muteval` works.
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
1.1 (stability, DONE) → 1.2 + 1.3 (generalize + one real judge) → 1.4 (limits)
→ **TRUST GATE** → Phase 1.5 (agent/Trace evals, LLM mutations) and/or Phase 2
probes (2.2 reliability → 2.3 discrimination → 2.6 report card). Phase 3
(usability + badge) can ride alongside once Phase 1 is trusted. **Agents come
AFTER the trust gate — they are expansion, not trust.**
