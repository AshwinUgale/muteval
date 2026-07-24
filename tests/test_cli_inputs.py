"""Adoption-audit fixes: auxiliary commands accept the same inputs as `run`,
config errors are clean (not raw tracebacks), config auto-pick, base_url
endpoint resolution, and `muteval list`."""

from __future__ import annotations

import pytest

from muteval.cli import main

_PF = """
prompts:
  - "Answer about {{topic}}. Cite the source."
tests:
  - vars: {topic: ports}
    assert:
      - type: contains
        value: source
"""


def _pf(tmp_path):
    p = tmp_path / "promptfooconfig.yaml"
    p.write_text(_PF, encoding="utf-8")
    return str(p)


def test_check_accepts_promptfoo(tmp_path):
    # A: the doctor now works on the promptfoo on-ramp (was --config only).
    pytest.importorskip("yaml")
    code = main(["check", "--promptfoo", _pf(tmp_path), "--no-model", "--no-color"])
    assert code == 0


def test_probe_accepts_promptfoo_arg(tmp_path):
    # A: `probe` accepts --promptfoo now (it used to reject with "required --config").
    pytest.importorskip("yaml")
    # --no-model isn't a probe flag; just assert it doesn't die on argparse/loading.
    # (Probes that call the model will no-op without a key; we only need the load path.)
    from muteval.cli import _build_parser

    args = _build_parser().parse_args(["probe", "--promptfoo", _pf(tmp_path)])
    assert args.promptfoo.endswith("promptfooconfig.yaml")


def test_run_broad_config_error_is_clean(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text("config = does_not_exist\n", encoding="utf-8")  # NameError at load
    code = main(["run", "--config", str(bad)])
    assert code == 2
    assert "raised NameError" in capsys.readouterr().err


def test_no_input_errors_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no ./muteval_config.py here
    code = main(["check", "--no-color"])
    assert code == 2
    assert "nothing to run" in capsys.readouterr().err


def test_run_auto_picks_local_config(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "muteval_config.py"
    cfg.write_text(
        "from muteval import MutEvalConfig\n"
        "config = MutEvalConfig(prompt='answer.\\n- cite the id.', cases=[{'x': 1}],\n"
        "    run=lambda p, c: 'ok', evals=[lambda o, c: True])\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = main(["run", "--dry-run", "--no-color"])
    assert code == 0
    assert "dry-run OK" in capsys.readouterr().out


def test_list_command(capsys):
    code = main(["list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "operators" in out and "probes" in out
    assert "weaken_modals" in out and "judge_reliability" in out


def test_base_url_endpoint_resolution(monkeypatch):
    from muteval.runners import _endpoint

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert _endpoint(None).endswith("api.openai.com/v1/chat/completions")
    # a bare base gets /chat/completions appended...
    assert _endpoint("https://api.groq.com/openai/v1") == \
        "https://api.groq.com/openai/v1/chat/completions"
    # ...a full endpoint is left alone (not doubled)
    assert _endpoint("https://x/v1/chat/completions") == "https://x/v1/chat/completions"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env/v1")
    assert _endpoint(None) == "https://env/v1/chat/completions"


def test_eval_names_auto_derived():
    from muteval import MutEvalConfig

    def cites_id(o, c):
        return True

    cfg = MutEvalConfig(prompt="p.\n- x.", cases=[{"x": 1}], run=lambda p, c: "ok",
                        evals=[cites_id])
    assert cfg.eval_names == ["cites_id"]  # derived from the function name, no dup needed
