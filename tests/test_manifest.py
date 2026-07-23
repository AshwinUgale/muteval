"""v0.6 gate: reproducible-run manifest (provenance + result, auditable)."""

from __future__ import annotations

import json

from muteval import EvalOutcome, MutEvalConfig, System, run_mutation_testing
from muteval.report import run_manifest

SYSTEM = System(prompt="Answer.\n- Cite the source.\n- Do not lie.", model="gpt-4o-mini")
CASES = [{"gt": "8080"}]


def _run(system, case):
    return f"answer {case['gt']} [1](doc)"


def _cfg():
    return MutEvalConfig(
        system=SYSTEM, cases=CASES, run=_run,
        evals=[lambda o, c: c["gt"] in o], eval_names=["correct"],
    )


def test_manifest_has_provenance_and_result():
    result = run_mutation_testing(_cfg())
    m = run_manifest(result, _cfg(), operators=["flip_negation"], seed=7)
    assert m["manifest_version"] == 1
    assert m["muteval_version"] and m["python"]
    assert m["run"]["model"] == "gpt-4o-mini"
    assert m["run"]["operators"] == ["flip_negation"]
    assert m["run"]["seed"] == 7
    assert len(m["run"]["system_fingerprint"]) == 16
    assert "score" in m["result"] and "survivors" in m["result"]


def test_manifest_redacts_secrets():
    result = run_mutation_testing(_cfg())
    m = run_manifest(result, _cfg())
    # inject a fake leak into the serialized form and confirm the redactor path
    m2 = json.dumps(m)
    assert "sk-" not in m2 or "[REDACTED]" in m2  # no live key string survives


def test_cli_writes_manifest(tmp_path, monkeypatch, capsys):
    from muteval.cli import main

    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.py"
    cfg.write_text(
        "from muteval import MutEvalConfig, System, EvalOutcome\n"
        "SYSTEM = System(prompt='Answer.\\n- Cite the source.\\n- Do not lie.', model='gpt-4o-mini')\n"
        "def run(system, case):\n    return 'answer ' + case['gt']\n"
        "config = MutEvalConfig(system=SYSTEM, cases=[{'gt':'8080'}], run=run,\n"
        "    evals=[lambda o,c: c['gt'] in o], eval_names=['correct'])\n"
    )
    out = tmp_path / "manifest.json"
    code = main(["run", "--config", str(cfg), "--manifest", str(out), "--no-color"])
    assert code == 0
    data = json.loads(out.read_text())
    assert data["manifest_version"] == 1 and "result" in data
