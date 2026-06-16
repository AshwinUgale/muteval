# Validation run #1 — deepeval RAG suite

**Target:** the real RAG eval in `confident-ai/deepeval`
(`examples/rag_evaluation/rag_evaluation_with_qdrant.py`) — its actual system
prompt and its 5-metric suite: AnswerRelevancy, Faithfulness, ContextualPrecision,
ContextualRecall, ContextualRelevancy.

## Hypothesis (write this down BEFORE running)

1. **Three of the five metrics structurally cannot catch a prompt regression.**
   ContextualPrecision / Recall / Relevancy only score the *retrieval*
   (retrieved context vs. expected answer). They never inspect how the generated
   answer relates to the prompt. So every prompt mutation is invisible to them.
2. **Only AnswerRelevancy and Faithfulness depend on the generated answer** — so
   they're the only two that *could* catch a prompt mutation.
3. **Predicted survivors:** mutations that remove behaviours neither remaining
   metric checks —
   - dropping "respond with I don't know" / flipping "do not pretend to know"
     (Faithfulness may or may not catch a confident wrong answer; relevancy won't)
   - dropping "reference the source" (no metric checks citation)
   - dropping "friendly and professional tone" (no metric checks tone)

## What to capture

- The mutation score (one number).
- The survivor list (copy it verbatim — this is the writeup).
- Which mutations were killed, and by which metric.

## The takeaway to test

If the score is low and the survivors are exactly the "no metric covers this"
behaviours, that's the muteval value proposition demonstrated on a real,
third-party suite — not a toy. That's checkpoint #1.
