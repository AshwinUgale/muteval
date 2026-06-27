# Run muteval on REAL deepeval / RAGAS metrics (Google Colab)

Your local Windows Python can't build pydantic-core/jiter (no Rust), so deepeval
and ragas won't install there. Colab has a standard Linux Python where the wheels
install in seconds. This runs the LITERAL framework metrics via muteval's adapters
— not the stdlib replicas.

## Step 0 (once): push your local repo so Colab can pull your latest code

```bash
# in C:\Users\ugale\muteval\muteval
git add -A
git commit -m "scored evals, System target, adapters, validation configs"
git push
```

## Colab — deepeval (real AnswerRelevancyMetric + FaithfulnessMetric)

Paste into a Colab cell (Runtime can be CPU):

```python
!git clone https://github.com/AshwinUgale/muteval.git
%cd muteval
!pip install -q -e ".[deepeval]"

import os
os.environ["OPENAI_API_KEY"] = "sk-..."          # paste your key
os.environ["MUTEVAL_JUDGE_MODEL"] = "gpt-4o-mini"

!python -m muteval.cli run --config validation/deepeval_rag_qdrant/muteval_config.py --max-mutants 8
```

This is real deepeval: the config wraps `AnswerRelevancyMetric` + `FaithfulnessMetric`
through `muteval.adapters.deepeval` and reuses deepeval's own RAG example prompt.

## Colab — RAGAS (real Faithfulness + ResponseRelevancy)

```python
!pip install -q -e ".[ragas]" langchain-openai
!python -m muteval.cli run --config validation/ragas_rag/muteval_config.py --max-mutants 8
```

This is real ragas: the config wraps ragas metrics through `muteval.adapters.ragas`.

## Notes
- If the baseline shows FAILED, the metrics scored the original answer below
  threshold — lower the threshold in the config or tighten the cases, exactly like
  we did for the stdlib run.
- Bump `runs_per_mutant=3` in the config before quoting numbers (LLM judges are
  non-deterministic).
- Same thing works on any cloud notebook / Linux box / WSL — the point is just a
  Python that can install the wheels.

## Local alternative (if you'd rather not use Colab)
Install a standard 64-bit Python 3.12 from python.org, then:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev,deepeval,ragas]" langchain-openai
muteval run --config validation/deepeval_rag_qdrant/muteval_config.py --max-mutants 8
```
