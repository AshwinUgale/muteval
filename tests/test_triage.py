"""v0.5 P0-7: triage UX — persist the last run; `muteval results` / `show <id>`.

After a run, the ranked survivors + a baseline-vs-mutant output sample are saved
to .muteval/last_run.json so you can inspect them without paying to re-run.
"""

from __future__ import annotations

from muteval.cli import _load_last_run, _save_last_run, main

# An offline config (no API key) whose weak eval lets an output change survive.
OFFLINE_CONFIG = '''
from muteval import MutEvalConfig, System, EvalOutcome

SYSTEM = System(
    prompt=(
        "Be a helpful support agent.\\n"
        "- Do not promise refunds; a manager must approve.\\n"
        "- Cite the source document.\\n"
        "- Greet the user politely."
    ),
    model="gpt-4o-mini",
)

def run(system, case):
    refuse = "do not promise refunds" in system.prompt.lower()
    return "I cannot promise a refund." if refuse else "Sure, refund coming right up!"

def weak(output, case):
    # WEAK on purpose: only checks the word 'refund' appears (passes either way).
    return EvalOutcome(passed="refund" in output.lower(), name="mentions_refund")

config = MutEvalConfig(
    system=SYSTEM, cases=[{"q": "can I get a refund?"}],
    run=run, evals=[weak], eval_names=["mentions_refund"],
)
'''


def _write_cfg(tmp_path):
    p = tmp_path / "cfg.py"
    p.write_text(OFFLINE_CONFIG)
    return p


def test_run_persists_and_results_lists_survivors(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = _write_cfg(tmp_path)

    assert main(["run", "--config", str(cfg), "--no-color"]) == 0
    assert (tmp_path / ".muteval" / "last_run.json").exists()
    capsys.readouterr()

    assert main(["results", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "survivors from the last run" in out
    assert "[weaken_modals]" in out or "[flip_negation]" in out or "[drop_instruction_lines]" in out


def test_show_renders_output_diff(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = _write_cfg(tmp_path)
    main(["run", "--config", str(cfg), "--no-color"])
    capsys.readouterr()

    assert main(["show", "0", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "survivor #0" in out
    assert "output diff" in out
    # The refund regression shows up in the diff.
    assert "refund" in out.lower()


def test_show_unknown_id_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = _write_cfg(tmp_path)
    main(["run", "--config", str(cfg), "--no-color"])
    capsys.readouterr()
    assert main(["show", "999", "--no-color"]) == 2
    assert "no survivor with id 999" in capsys.readouterr().err


def test_results_without_saved_run_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["results", "--no-color"]) == 2
    assert "no saved run" in capsys.readouterr().err


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from muteval import run_mutation_testing
    from muteval.config import load_config

    result = run_mutation_testing(load_config(_write_cfg(tmp_path)))
    _save_last_run(result)
    data = _load_last_run()
    assert data is not None
    assert "survivors" in data and data["survivors"]
    # Survivor carries the captured output sample.
    assert "baseline_output" in data["survivors"][0]
