# Prior art & method citations

muteval builds on established statistics and a body of eval/meta-evaluation
research. This page names the lineage — both to give credit and to preempt the
fair questions "isn't this just mutation testing?" and "isn't judge-reliability
already solved?". Short answer: the *mutation-testing-for-eval-suites* framing and
its **absence detection** are novel as a product; the probes port well-known
statistics and, for judge reliability, reach parity with an already-served area.
(Full competitive audit: `docs/AUDIT-probe-prior-art.md`.)

## The core idea (mutation testing for evals)

- **MILE** — Wei et al., *Mutation Testing of In-Context-Learning Systems*
  (arXiv 2409.04831, SETTA 2024). The closest prior work and an independent
  confirmation of muteval's central claim that a mutation score tracks eval-suite
  quality. Scoped to ICL classification, a research prototype — muteval
  generalizes it (prompt/context/tools/model), packages it (CLI/adapters/survivor
  abstraction), and adds absence detection.
- **DeepMutation** (arXiv 1805.05206) and the broader ML-mutation-testing line
  mutate *models* to grade traditional-ML test data — adjacent, different axis.
- **`mutmut` / Stryker** — mutation testing for code; the "mutation score = test
  suite quality" heritage muteval transposes to evals.
- **Tangled up in BLEU** — Mathur et al., ACL 2020 (aclanthology 2020.acl-main.448).
  Metric-reliability critique; its Type-I/II framing maps onto muteval survivors
  as Type-II errors ("your suite would miss this").
- **CheckList** — Ribeiro et al., ACL 2020 (2020.acl-main.442). Software-testing
  discipline for NLP; tests models, not suites — a philosophical ancestor muteval
  can *consume* rather than compete with.

## Probe method lineage

**statistical_adequacy** — Wilson score interval (default) + Jeffreys/Beta-Binomial
for small n.
- Brown, Cai & DasGupta (2001), *Interval Estimation for a Binomial Proportion* —
  the binomial-CI authority (use Wilson/Jeffreys over Wald).
- Bowyer, Aitchison & Ivanova (2025), *Don't Use the CLT in LLM Evals With Fewer
  Than a Few Hundred Datapoints* (arXiv 2503.01747; lib `bayes_evals`) — argues
  muteval's exact thesis; the reason Jeffreys is offered for tiny n.
- Evan Miller (2024), *Adding Error Bars to Evals* (arXiv 2411.00640) — the
  power / "how many cases" framing.
- Card et al. (2020), *With Little Power Comes Great Responsibility* (EMNLP) — the
  canonical "your eval is too small" result.

**judge_reliability** — verdict-flip rate + Krippendorff's alpha over re-runs.
- *Reliability without Validity* (arXiv 2606.19544) — large-scale study using
  Krippendorff's alpha for judge test-retest; source of the reliability≠validity
  caveat muteval states.
- *The Coin Flip Judge* (arXiv 2606.13685) — ~11 runs for a stable majority;
  temperature sensitivity (why the probe recommends temp 0).
- Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench* (arXiv 2306.05685)
  and Wang et al., *LLMs are not Fair Evaluators* — the bias taxonomy (position/
  verbosity/self-preference) and the **CALM** framework (*Justice or Prejudice*,
  ICLR 2025) are the blueprint for the directional-**bias panel** (future work;
  needs a structured judge abstraction).
- Honest note: judge alignment/reliability is already served (LangSmith Align
  Evals, Ragas, Arize, standalone MetaEvaluator). muteval's probe is parity, not
  a differentiator.

**discrimination** — AUC (Mann–Whitney U) + Cohen's d + significance.
- Classical Test Theory *item-discrimination index* (point-biserial); recent LLM
  application with a ~0.15 threshold (arXiv 2606.18709).
- Kocmi et al., *To Ship or Not to Ship* (WMT21, arXiv 2107.10821) and *Ties
  Matter* (arXiv 2305.14324) — the NLG/MT field's move to rank/pairwise measures
  over raw score gaps, which is why muteval uses AUC not a mean gap.

**redundancy** — Spearman rank correlation + connected-component families.
- scikit-learn's multicollinearity recipe (Spearman + hierarchical clustering).
- *Agreement Metrics for LLM-as-Judge* (arXiv 2606.00093) — reporting several
  correlated metrics is an "illusion of corroborating evidence" (the probe's thesis).

## Neighbours in the eval-quality space (not the same axis)

- **MetaEvaluator** (GovTech) — rates *judges* (Cohen's kappa, alt-test), the
  nearest standalone package; overlaps the probe layer, not the mutation core.
- **FBI** (AI4Bharat, EMNLP 2024) — perturbs inputs to find evaluator-LLM blind
  spots; closest mechanism to muteval's, judge-scoped, academic.
- Incumbent eval-quality features (LangSmith Align Evals, Ragas Align, Arize
  meta-evals, DeepEval metric-alignment) are almost all *judge-vs-human alignment* —
  which by construction cannot surface "you have no eval for this at all."

## Positioning (evidence-backed)

The moat is the **mutation core + absence detection** and, secondarily,
**discrimination** and **redundancy** (essentially unserved by any tool).
**judge_reliability** is table-stakes parity — never lead with it. The
eval-evaluator superset stays demand-gated.
