"""v0.5 P0-1: zero-config ingestion — `muteval run --promptfoo FILE`.

A user with an existing promptfooconfig.yaml should be able to run muteval
against it with NO muteval config file. --dry-run exercises the whole ingestion
path (parse -> MutEvalConfig -> mutant selection) without an API key.
"""

from __future__ import annotations

import pytest

from muteval.cli import main

pytest.importorskip("yaml")  # the promptfoo adapter needs PyYAML

PROMPTFOO_YAML = """
prompts:
  - "Answer the question about {{topic}}. Always cite the source."
defaultTest:
  assert:
    - type: contains
      value: source
tests:
  - vars: {topic: ports}
    assert:
      - type: contains
        value: "8080"
  - vars: {topic: keys}
    assert:
      - type: not-contains
        value: password
"""


def _write(tmp_path):
    p = tmp_path / "promptfooconfig.yaml"
    p.write_text(PROMPTFOO_YAML, encoding="utf-8")
    return p


def test_promptfoo_dry_run_ingests_without_config(tmp_path, capsys):
    path = _write(tmp_path)
    code = main(["run", "--promptfoo", str(path), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "dry-run OK" in out
    assert "cases:   2" in out
    # The prompt is long enough that prompt mutants are generated.
    assert "mutants that would run:" in out


def test_load_run_config_from_promptfoo(tmp_path):
    from types import SimpleNamespace

    from muteval.cli import _load_run_config

    path = _write(tmp_path)
    args = SimpleNamespace(config=None, promptfoo=str(path), model="gpt-4o-mini",
                           prompt=None, prompt_file=None, cases=None)
    cfg = _load_run_config(args)
    assert len(cfg.cases) == 2
    assert "cite the source" in cfg.prompt
    # The promptfoo assertions became the (single) suite eval.
    assert cfg.eval_names == ["promptfoo_asserts"]


def test_promptfoo_missing_file_errors_cleanly(capsys):
    code = main(["run", "--promptfoo", "does_not_exist.yaml", "--dry-run"])
    assert code == 2
    assert "muteval:" in capsys.readouterr().err
