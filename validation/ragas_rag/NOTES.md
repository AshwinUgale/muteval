# Validation: RAGAS-backed RAG suite

This run reuses **RAGAS** metrics (Faithfulness, ResponseRelevancy) as muteval
evals via `muteval.adapters.ragas`, then mutates the prompt and reruns the suite.

## How to run

```bash
pip install "muteval[ragas]" langchain-openai
export OPENAI_API_KEY=sk-...
muteval run --config validation/ragas_rag/muteval_config.py --max-mutants 8
```

## Design notes

- **Threshold.** RAGAS metrics emit a raw score in `[0, 1]` and carry no
  pass/fail threshold of their own, so the adapter applies one (`threshold=0.7`).
  A mutant is *killed* when the mutated prompt drops a metric below the
  threshold. Because the adapter forwards score + threshold, survivors that pass
  by a hair are reported as **near misses** in the report.

- **Answer-dependent metrics only.** Like the deepeval run, we use metrics that
  depend on the *answer* (Faithfulness, ResponseRelevancy). Retrieval-only
  metrics (context precision/recall) can't be moved by a *prompt* mutation, so
  including them only adds baseline noise. (They *can* be moved by a **context**
  mutation — switch to `system=` mode with `System(context=...)` and the
  `drop_context_doc` / `clear_context` operators to exercise those.)

- **API drift.** RAGAS has changed its public API across releases. This config
  targets `SingleTurnSample` / `single_turn_score` (ragas >= 0.2) with an
  LLM-wrapped metric. If your version differs, pass `sample_factory=` and/or
  `score_fn=` to `metric_to_eval` to adapt the wiring without touching muteval.
