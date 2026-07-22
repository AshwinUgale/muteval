# Prior-art audit: probes + the eval-evaluator (don't reinvent wheels)

Deep audit of existing research, libraries, and tools for each probe and for the
"eval-evaluator" concept, so we port proven methods instead of reinventing them —
and so we position honestly. Every probe's *current* implementation turns out to
be the weakest defensible version; the literature already has the stronger method,
and all of them are reimplementable dependency-free.

## Headline verdict

- **The mutation-testing core is genuinely novel** — no product does "mutate the
  system, rerun the user's real suite, score coverage." Only one academic twin
  exists (**MILE**, arXiv 2409.04831 — ICL-only research prototype), which
  *independently confirms our central claim* that a mutation score tracks
  eval-suite quality. Cite it, distinguish on generality + productization +
  absence detection. **Do not claim the technique is novel; claim the packaged,
  tool-agnostic superset is.**
- **The probe layer is mostly NOT novel**, and one dimension (judge reliability)
  is already crowded/commodity. This validates the standing decision: market
  muteval as the mutation tool; keep probes a one-way bonus layer; treat the
  eval-evaluator as demand-gated.

## Per-probe: current → recommended upgrade (all dependency-free-able)

### statistical_adequacy
- **Now:** Wilson CI on case count + min-sample-size. Solid, defensible default.
- **Upgrade:** add a dependency-free **Jeffreys / Beta-Binomial** interval for the
  tiny-n regime (degrades better than Wilson at n<30). Keep Wilson as default.
- **Reuse/cite:** Bowyer, Aitchison & Ivanova, *"Don't Use the CLT in LLM Evals
  With Fewer Than a Few Hundred Datapoints"* (ICML 2025) + lib **`bayes_evals`**
  (arxiv 2503.01747) — argues our exact thesis; Evan Miller, *"Adding Error Bars
  to Evals"* (arXiv 2411.00640) for power/"how many cases"; Card et al., *"With
  Little Power…"* (EMNLP 2020) as the canonical "your eval is too small" paper;
  Brown/Cai/DasGupta (2001) justifies Wilson/Jeffreys over Wald. Optional
  **`confseq`** (anytime-valid confidence sequences) for suites that *grow* over
  time — a real differentiator, low priority.
- **Gap we own:** adequacy as a first-class *warning* + required-N, not just a
  stderr number. lm-eval-harness/Inspect report error bars on scores; nobody
  flags "your suite is too small to defend this pass rate."

### judge_reliability  ⚠ crowded axis
- **Now:** verdict-flip rate over N re-runs = **stochastic noise only**. Misses
  directional bias entirely.
- **Upgrade:** report **Krippendorff's α** (chance-corrected, treat each re-run as
  a rater; dependency-free for the nominal case) + **ICC** for 0–10 judge scores;
  and — highest value — **add a small bias panel**: position bias (swap order),
  verbosity/length bias, self-preference. The **CALM** framework (*"Justice or
  Prejudice"*, ICLR 2025) is our own mutate-then-check loop applied to the judge —
  the best template to port.
- **Reuse/cite:** Zheng et al. MT-Bench (2306.05685, the bias taxonomy); Wang et
  al. *"LLMs are not Fair Evaluators"* (position bias); *"Reliability without
  Validity"* (2606.19544 — uses Krippendorff's α, real thresholds, and the
  crucial *reliability ≠ validity* caveat); *"Coin Flip Judge"* (2606.13685 — ~11
  runs needed for a stable majority; warn when judge temp>0). Libs behind a
  `[stats]` extra: `krippendorff`, `pingouin` (ICC), `irrCAC` (Gwet's AC1 for
  skewed pass-rates).
- **Remediations to recommend:** temperature=0, LLM-jury / panel voting (Verga et
  al. 2024), rubric+CoT.
- **Honesty:** already served by LangSmith Align Evals, Ragas, Arize, Galileo,
  Braintrust, Vertex + standalone **MetaEvaluator (GovTech)**. Position as
  table-stakes parity, **not** a differentiator.

### discrimination  (thinly served — we can own it)
- **Now:** raw mean-gap `mean(good) − mean(bad)`, flag if < 0.3. Unnormalized —
  0.3 is huge on SD 0.02, noise on SD 0.4.
- **Upgrade:** headline with **AUC** (= Mann–Whitney U / (n_good·n_bad),
  dependency-free, scale-free, "ranks a good answer above a bad one X% of the
  time"; 0.5 = coin flip → WARN), with a **significance test** (Mann–Whitney
  p-value — vital, exemplar sets are tiny) and **Cohen's d / point-biserial** as
  the magnitude signal. Handle ties as failure-to-discriminate.
- **Reuse/cite:** this is the **Classical Test Theory item-discrimination index**
  reinvented — point-biserial, with a published LLM threshold **~0.15** (arXiv
  2606.18709). NLG/MT meta-eval (Kocmi *"To Ship or Not to Ship"* WMT21; *"Ties
  Matter"* 2305.14324) converged on rank/pairwise measures over raw gaps for
  exactly our outlier/variance reasons. Libs: core = pure-Python AUC +
  `scipy.stats.mannwhitneyu`/`pointbiserialr`; optional `pingouin` for effect
  sizes+CIs. Avoid sklearn's weight for one function.
- **Gap we own:** no tool ships an automated "this metric doesn't separate good
  from bad" check — everyone leaves it as manual "correlate your scorer to labels."

### redundancy  (unserved — we own it)
- **Now:** Pearson r > 0.9. Weakest default — misses monotonic-but-nonlinear
  redundancy (common with saturated 0–1 judge scores).
- **Upgrade:** **Spearman** rank correlation (dependency-free: rank then Pearson;
  catches monotonic redundancy) + **hierarchical clustering** on the `1 − |ρ|`
  distance matrix to report redundant *families* ("these 3 are one construct; keep
  1") instead of loose pairs. Add **VIF** (numpy `lstsq`, no statsmodels) as an
  opt-in for ≥4 metrics — catches "redundant given the whole panel" that pairwise
  misses. This is literally scikit-learn's own multicollinearity recipe.
- **Reuse/cite:** *"Agreement Metrics for LLM-as-Judge"* (2606.00093 — reporting
  several correlated metrics is an "illusion of corroborating evidence"); measurement-
  theory framing (2305.14889). MI/distance-correlation/PCA = documented future
  power-modes, not defaults (weight + interpretability cost).
- **Gap we own:** no mainstream tool flags redundant metrics or recommends pruning.

## Meta prior-art: is the eval-evaluator novel?

- **Standalone "rate my eval suite" product: does not exist.** Closest is
  **MetaEvaluator (GovTech)** — but it rates *judges*, not suites (overlaps the
  probe layer, not the core). Academic cousins: **FBI** (AI4Bharat — inject
  perturbations, check if the evaluator catches them; closest to our mechanism,
  judge-scoped), **EvalSense** (NHS). All research, no product.
- **Incumbents' eval-quality features are almost entirely judge-vs-human
  alignment** (our probe #3): LangSmith **Align Evals** (Jul 2025), Ragas Align,
  Arize meta-evals, DeepEval/Galileo/Braintrust/Vertex. Statistical adequacy:
  only **Inspect** (`stderr`/epochs). Discrimination + redundancy: essentially
  nobody.
- **Namespaces clean:** PyPI + npm `muteval` uncontested.
- **Competition trend:** the space is heating (NeurIPS **Evaluations track** for
  2026; judge-meta-eval is the hottest adjacent area). Raises the bar for the
  superset → reinforces "validate demand before building it."

**Closest threats, ranked:** DeepEval/Confident AI (most likely to productize a
full superset) · MetaEvaluator (nearest standalone) · LangSmith Align Evals
(interop/parity reference) · MILE (academic, cite-and-distinguish).

## Action items

1. **Port the stronger method as part of each probe's validation work** — don't
   validate the weak version. Priority order matches the validation plan:
   - redundancy → **Spearman + clustering** (easy, big correctness win, unserved).
   - discrimination → **AUC + Mann–Whitney + Cohen's d** (unserved, strong story).
   - statistical_adequacy → **Jeffreys** option + citations (mostly done).
   - judge_reliability → **Krippendorff α + a position/verbosity/self-pref bias
     panel** (crowded, so build only enough for parity; lean on the bias panel as
     the one genuinely-useful add).
2. **Keep core dependency-free**; gate `bayes_evals`/`krippendorff`/`pingouin`/
   `confseq` behind a `[stats]` extra.
3. **Add a short prior-art / citations section** to the README or a `docs/PRIOR-ART.md`
   (MILE, DeepMutation, *Tangled up in BLEU*, Anthropic error-bars, EvalGen,
   MetaEvaluator) — preempts the HN "isn't this just mutation testing / judge
   alignment?" question and signals literature awareness.
4. **Positioning (unchanged, now evidence-backed):** the moat is the mutation core
   + absence detection. Judge reliability is commodity — never lead with it. The
   evaluator superset stays demand-gated.
