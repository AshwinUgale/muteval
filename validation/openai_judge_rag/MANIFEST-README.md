# Reproducible real-LLM-judge run (manifest)

The v0.6 gate asks for **one reproducible real-LLM-judge run committed under
`validation/` with a manifest**. This config (`muteval_config.py`) is a real
OpenAI-backed run (stdlib, no heavy deps); the manifest captures its provenance
so the number is auditable and repeatable.

## Produce the committed manifest

```bash
pip install certifi
export OPENAI_API_KEY=sk-...
muteval run --config validation/openai_judge_rag/muteval_config.py \
  --seed 0 --manifest validation/openai_judge_rag/manifest.json
git add validation/openai_judge_rag/manifest.json
```

The manifest records: muteval version, Python/platform, timestamp, the model, the
operator set, the seed, the system fingerprint (sha256 of prompt+context+model),
and the full result (status, score, effective score with CI, and the survivors).
Secrets are redacted before writing.

## Why it isn't checked in already

Producing it requires a live API key (it makes real judge calls), so it must be
generated on a machine that has one — it is deliberately NOT fabricated here. The
harness (`--manifest`) and this recipe are committed; run the command above once
with your key to commit the actual `manifest.json`.

## Honesty note

A real-judge run is non-deterministic at the judge (temperature, provider
variance). Re-running should land within the reported 95% CI, not on the exact
same integer. The manifest's seed pins muteval's own RNG (mutant sampling), not
the remote model.
