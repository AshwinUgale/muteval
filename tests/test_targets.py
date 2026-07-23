"""v0.5 P0-2: callable / endpoint targets.

Let a user point muteval at their OWN function (--target pkg.mod:fn) or a
deployed HTTP endpoint (--endpoint URL) as the system under test, instead of the
built-in OpenAI runner — no run() wrapper, no config file.
"""

from __future__ import annotations

import io
import json

import pytest

from muteval import System
from muteval import runners


# --- callable_run ------------------------------------------------------------

def test_callable_run_imports_and_calls(tmp_path, monkeypatch):
    mod = tmp_path / "mypipe.py"
    mod.write_text("def answer(prompt, case):\n    return prompt + '|' + case['q']\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    run = runners.callable_run("mypipe:answer")
    # Legacy prompt target.
    assert run("PROMPT", {"q": "hi"}) == "PROMPT|hi"
    # System target -> the mutated system.prompt is passed through.
    assert run(System(prompt="SYS"), {"q": "yo"}) == "SYS|yo"


def test_callable_run_rejects_bad_spec():
    for bad in ("no_colon", "mod:", ":fn", ""):
        with pytest.raises(ValueError):
            runners.callable_run(bad)


def test_callable_run_rejects_non_callable(tmp_path, monkeypatch):
    mod = tmp_path / "notfn.py"
    mod.write_text("value = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError):
        runners.callable_run("notfn:value")


# --- http_run ----------------------------------------------------------------

class _FakeResp:
    def __init__(self, payload: str):
        self._b = payload.encode("utf-8")

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, payload, captured=None):
    def fake_urlopen(req, timeout=60, context=None):
        if captured is not None:
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(payload)

    monkeypatch.setattr(runners.urllib.request, "urlopen", fake_urlopen)


def test_http_run_parses_json_output_key(monkeypatch):
    captured = {}
    _patch_urlopen(monkeypatch, json.dumps({"output": "hello"}), captured)
    run = runners.http_run("http://x/answer")
    out = run(System(prompt="P"), {"q": "why"})
    assert out == "hello"
    # It POSTed the mutated prompt + case.
    assert captured["body"]["prompt"] == "P"
    assert captured["body"]["case"] == {"q": "why"}


def test_http_run_accepts_plain_text(monkeypatch):
    _patch_urlopen(monkeypatch, "just text, not json")
    assert runners.http_run("http://x")("P", {}) == "just text, not json"


def test_http_run_falls_back_to_raw_for_unknown_shape(monkeypatch):
    raw = json.dumps({"weird": "shape"})
    _patch_urlopen(monkeypatch, raw)
    assert runners.http_run("http://x")("P", {}) == raw


def test_http_run_alternate_output_keys(monkeypatch):
    for key in ("text", "content", "answer", "response", "completion"):
        _patch_urlopen(monkeypatch, json.dumps({key: "V"}))
        assert runners.http_run("http://x")("P", {}) == "V"


# --- CLI wiring (dry-run: no network, no key) --------------------------------

def test_cli_target_dry_run(tmp_path, capsys, monkeypatch):
    from muteval.cli import main

    (tmp_path / "mypipe.py").write_text("def answer(prompt, case):\n    return 'x'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"q": "hi"}\n')

    code = main([
        "run", "--target", "mypipe:answer",
        "--prompt", "You are a bot. Always cite the source. Do not lie.",
        "--cases", str(cases), "--check", "contains:x", "--dry-run",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "dry-run OK" in out
    assert "mutants that would run:" in out
