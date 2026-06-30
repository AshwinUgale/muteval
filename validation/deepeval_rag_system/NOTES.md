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
