# Validation run #2 — context + model mutation (the stronger story)

**What's different from run #1:** run #1 mutated only the prompt. This mutates
the **retrieved context** (corrupt / swap / drop a doc) and **downgrades the
model**, in System mode — the failure modes teams actually fear in RAG/agents.

## Hypothesis (write before running)

1. **corrupt_context_doc survives.** Faithfulness grades the answer against the
   context the system used (`used_context`). A corrupted doc -> a confidently
   wrong-but-grounded answer -> Faithfulness PASSES it. AnswerRelevancy also
   passes (still on-topic). => survives, HIGH severity. This is the headline:
   *your evals don't catch a poisoned retrieval.*
2. **drop_context_doc / swap_context_doc** often survive too (answer changes,
   metrics don't object).
3. **downgrade_model** may survive (a weaker model still sounds relevant and
   grounded). HIGH severity.
4. Inert mutants (output identical) are excluded by output-diffing.

## What to capture
- effective_score, the HIGH-severity survivor list, and whether
  `--fail-on-severity high` exits non-zero (it should).
- Re-run final numbers with MUTEVAL_JUDGE_MODEL=gpt-4o for credibility.

## The pitch this run supports
"muteval fed deepeval's RAG suite a corrupted retrieved document and the evals
stayed green. Faithfulness can't catch a poisoned retrieval — it only checks
grounding, not correctness."

---

## RESULT — real LLM-judge run (Colab, gpt-4o-mini judge)

**Status: mechanism confirmed on real deepeval metrics; a clean-baseline run was
blocked by deepeval's timeout instability on this environment.**

Across ~5 runs, muteval consistently reproduced the predicted survivor:

    [HIGH] SURVIVED  drop_context_doc / swap_context_doc
           dropped/swapped the retrieved doc -> deepeval's AnswerRelevancy +
           Faithfulness did NOT catch it (near miss: passed Answer Relevancy
           by only +0.500)

i.e. **you can poison the retrieval and deepeval's RAG metrics stay green** —
the intended, non-gimmicky finding, on real LLM-judge metrics.

### The deepeval instability (not a muteval bug)
The BASELINE (original prompt + full context) is deepeval's heaviest call
(Faithfulness has the most claims to verify) and repeatedly timed out
(RetryError -> TimeoutError), even with `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=120`
and muteval's new baseline-retry (up to 3 attempts). Lighter mutant calls
(degraded context) mostly succeeded. muteval behaved correctly throughout:
retried, excluded errored mutants, reported a Wilson CI, and honestly flagged
the run as unreliable rather than emitting a fake-clean number.

### Takeaway for muteval
- The mechanism works end-to-end on real deepeval metrics.
- The deepeval adapter's reliability is bounded by deepeval's own stability;
  document this in LIMITATIONS.
- A clean published number should be produced on a stabler environment (or a
  lighter metric set / higher timeout budget). The *finding* does not depend on it.
