"""Secrets never reach any output, whichever format you pick.

From a full audit: only JSON and the manifest were redacted — the terminal
report, JUnit, `muteval check` and CLI error lines printed a provider error's
echoed key verbatim, and the pattern missed `Bearer <token>`, `Authorization:
Bearer`, GitHub/Hugging Face/xAI tokens and `"api_key": "..."`. And the
zero-config `judge:` check ignored --base-url, so OPENAI_API_KEY (holding e.g. a
Groq key) was sent to api.openai.com.
"""

import pytest

from muteval import MutEvalConfig, checks, run_mutation_testing
from muteval.cli import _build_parser, _check_from_spec, _config_from_flags, main
from muteval.redact import REDACTED, redact, redact_obj
from muteval.report import (
    format_report,
    format_report_html,
    format_report_junit,
    result_to_dict,
)


def _fake(prefix: str, n: int = 24) -> str:
    """A token-SHAPED placeholder, built at runtime with a repeated low-entropy
    body. This file must never contain a literal credential-looking string:
    secret scanners (GitGuardian) flag those even when they're fake fixtures."""
    return prefix + ("Fake0" * n)[:n]


LEAKS = {
    name: _fake(prefix)
    for name, prefix in {
        "openai": "sk-",
        "openai_project": "sk-proj-",
        "anthropic": "sk-ant-api03-",
        "groq": "gsk_",
        "google": "AIza",
        "github": "ghp_",
        "github_pat": "github_pat_",
        "huggingface": "hf_",
        "xai": "xai-",
    }.items()
}
JWT_LIKE = _fake("eyJ", 20) + ".payload.sig"
BEARER = _fake("tok", 16)


@pytest.mark.parametrize("secret", list(LEAKS.values()), ids=list(LEAKS))
def test_known_key_shapes_are_redacted(secret):
    assert secret not in redact(f"request failed with key {secret} (401)")


@pytest.mark.parametrize(
    "text, secret",
    [
        ("Authorization: Bearer " + JWT_LIKE, JWT_LIKE.split(".")[0]),
        ("401 Unauthorized: Bearer " + BEARER, BEARER),
        ('{"api_key": "zzz-some-secret-value", "model": "x"}', "zzz-some-secret-value"),
        (
            "...:generateContent?key=" + LEAKS["google"] + "&alt=json",
            LEAKS["google"],
        ),
        ("https://x.test/v1?access_token=tok_abcdef123456", "tok_abcdef123456"),
        ("password: hunter2hunter2", "hunter2hunter2"),
        ("OPENAI_API_KEY=plainsecret", "plainsecret"),  # prefixed key name
        ("token=plainsecretvalue", "plainsecretvalue"),
        ('client_secret: "abcdefg123"', "abcdefg123"),
    ],
)
def test_credential_assignments_are_redacted_and_labelled(text, secret):
    out = redact(text)
    assert secret not in out and REDACTED in out
    # The label survives, so the reader knows WHAT was hidden.
    assert out.split(REDACTED)[0].strip()


def test_ordinary_prompt_text_is_untouched():
    text = (
        "The model must always cite sources. skip-unchanged stays; task-1234; "
        "desk-lamp costs $5; see /usr/bin; key points: be brief. "
        "temperature: 0.2, max_tokens=256, tokens=5; secretary: Ann; "
        "password_hint: pet name"
    )
    assert redact(text) == text


def test_secret_env_values_are_redacted_even_in_unknown_formats(monkeypatch):
    # A key in a shape no pattern knows (e.g. Azure's bare 32-char key).
    azure = ("fake" * 8)[:32]
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", azure)
    monkeypatch.setenv("MY_PROXY_TOKEN", "proxy-token-value-42")
    monkeypatch.setenv("SHORT_KEY", "abc")  # too short to redact safely
    out = redact(f"azure {azure} / proxy-token-value-42 / abc")
    assert azure not in out
    assert "proxy-token-value-42" not in out
    assert out.endswith("/ abc")


def test_redact_obj_recurses():
    blob = str(redact_obj({"a": [LEAKS["groq"], {"b": (LEAKS["github"],)}], "n": 3}))
    assert LEAKS["groq"] not in blob and LEAKS["github"] not in blob


# --- every output format ---------------------------------------------------------

SECRET = LEAKS["google"]
PROMPT = "- You must cite the order ID.\n- Never promise refunds."


def _run_raising(baseline_raises):
    def run(prompt, case):
        if baseline_raises or prompt != PROMPT:
            raise RuntimeError(f"POST .../generateContent?key={SECRET} -> 403")
        return "ok"

    cfg = MutEvalConfig(
        prompt=PROMPT, cases=[{"x": 1}], run=run, evals=[lambda o, c: True]
    )
    return run_mutation_testing(cfg, operators=["flip_negation"])


@pytest.mark.parametrize("baseline_raises", [True, False], ids=["baseline", "mutants"])
def test_no_output_format_leaks_an_error_string(baseline_raises):
    r = _run_raising(baseline_raises)
    assert r.baseline_error or r.errored  # the error really carried the key
    outputs = {
        "terminal": format_report(r, use_color=False),
        "junit": format_report_junit(r),
        "json": str(result_to_dict(r)),
        "html": format_report_html(result_to_dict(r)),
    }
    for fmt, text in outputs.items():
        assert SECRET not in text, fmt


def test_html_redacts_an_unredacted_input_file():
    data = {"status": "valid", "survivors": [{"description": f"leak {SECRET}"}]}
    assert SECRET not in format_report_html(data)


def _write(tmp_path, body):
    p = tmp_path / "cfg.py"
    p.write_text(body, encoding="utf-8")
    return str(p)


_DOCTOR_CFG = f"""
from muteval import MutEvalConfig
def run(prompt, case):
    raise RuntimeError("401 Unauthorized: Bearer {SECRET}")
config = MutEvalConfig(prompt={PROMPT!r}, cases=[1], run=run, evals=[lambda o, c: True])
"""


def test_doctor_output_is_redacted(tmp_path, capsys):
    main(["check", "--config", _write(tmp_path, _DOCTOR_CFG), "--no-color"])
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err


def test_config_load_error_is_redacted(tmp_path, capsys):
    body = f'raise RuntimeError("bad key {SECRET}")\n'
    assert main(["run", "--config", _write(tmp_path, body), "--no-color"]) == 2
    assert SECRET not in capsys.readouterr().err


def test_cli_run_terminal_and_junit_are_redacted(tmp_path, capsys):
    junit = tmp_path / "r.xml"
    main(
        [
            "run",
            "--config",
            _write(tmp_path, _DOCTOR_CFG),
            "--no-color",
            "--junit",
            str(junit),
        ]
    )
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err
    assert SECRET not in junit.read_text(encoding="utf-8")


# --- the judge goes where --base-url points ---------------------------------------


def _cases(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text('{"input": "hi"}\n', encoding="utf-8")
    return str(p)


def _capture_judge_calls(monkeypatch):
    calls = []

    def fake(prompt, model, base_url=None):
        calls.append((model, base_url))
        return "9"

    monkeypatch.setattr(checks, "_openai_chat_stdlib", fake)
    return calls


def test_zero_config_judge_inherits_base_url(monkeypatch, tmp_path):
    calls = _capture_judge_calls(monkeypatch)
    args = _build_parser().parse_args(
        [
            "run",
            "--cases",
            _cases(tmp_path),
            "--prompt",
            "Be polite.",
            "--judge",
            "is it polite",
            "--model",
            "openai/gpt-oss-20b",
            "--base-url",
            "https://api.groq.com/openai/v1",
        ]
    )
    cfg = _config_from_flags(args)
    judge = next(ev for ev in cfg.evals if getattr(ev, "is_llm", False))
    judge("hello", {})
    assert calls == [("openai/gpt-oss-20b", "https://api.groq.com/openai/v1")]


def test_judge_endpoint_can_be_overridden(monkeypatch, tmp_path):
    calls = _capture_judge_calls(monkeypatch)
    args = _build_parser().parse_args(
        [
            "run",
            "--cases",
            _cases(tmp_path),
            "--prompt",
            "Be polite.",
            "--judge",
            "is it polite",
            "--base-url",
            "https://api.groq.com/openai/v1",
            "--judge-model",
            "gpt-4o",
            "--judge-base-url",
            "https://api.openai.com/v1",
        ]
    )
    judge = next(
        ev for ev in _config_from_flags(args).evals if getattr(ev, "is_llm", False)
    )
    judge("hello", {})
    assert calls == [("gpt-4o", "https://api.openai.com/v1")]


def test_check_from_spec_passes_base_url(monkeypatch):
    calls = _capture_judge_calls(monkeypatch)
    ev = _check_from_spec("judge:polite", 0.5, "m", "https://example.test/v1")
    ev("hi", {})
    assert calls == [("m", "https://example.test/v1")]
