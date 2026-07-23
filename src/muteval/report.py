"""Human-readable terminal reporting for a MutationResult."""

from __future__ import annotations

import html
import re

from muteval.runner import MutationResult


def _bar(score: float, width: int = 24) -> str:
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


def format_report(result: MutationResult, use_color: bool = True) -> str:
    def c(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    lines = []
    lines.append("")
    lines.append(c("muteval — mutation testing for your eval suite", "1"))
    lines.append("")

    # Invalid / empty runs are terminal — there is NO trustworthy score to show.
    if result.status == "budget_exceeded":
        lines.append(c("⚠  INCOMPLETE — call budget exceeded (--max-calls)", "1;31"))
        lines.append(
            "   Stopped before finishing, so there is no trustworthy score. "
            "Raise --max-calls (or narrow with --sample) and re-run."
        )
        return "\n".join(lines)
    if result.baseline_error:
        lines.append(c("⚠  INVALID RUN — baseline ERRORED", "1;31"))
        lines.append(f"   {result.baseline_error}")
        lines.append(
            "   The eval suite raised on the ORIGINAL system, so there is no "
            "trustworthy score. Fix the error and re-run."
        )
        return "\n".join(lines)
    if not result.baseline_passed:
        lines.append(c("⚠  INVALID RUN — baseline FAILED", "1;31"))
        lines.append(
            "   Your eval suite does not pass on the ORIGINAL system, so a "
            "mutation score would be meaningless (every mutant 'fails' too). "
            "Fix the baseline, then re-run."
        )
        return "\n".join(lines)
    if result.total == 0:
        lines.append(c("⚠  NO MUTANTS — nothing to test", "33"))
        lines.append(
            "   No mutants were generated (prompt too short, or operators/scope "
            "filtered them all out). No score."
        )
        return "\n".join(lines)
    if result.evaluated == 0:
        lines.append(c("⚠  INVALID RUN — no mutant produced a clean verdict", "1;31"))
        lines.append(
            f"   All {result.total} mutant(s) errored (e.g. API failures). No "
            "score — investigate the failures and re-run."
        )
        return "\n".join(lines)

    pct = result.score * 100
    score_color = "32" if pct >= 80 else "33" if pct >= 50 else "31"
    lo, hi = result.score_ci
    lines.append(
        f"Mutation score: {c(f'{pct:.0f}%', score_color)}  "
        f"[{_bar(result.score)}]  "
        f"({result.killed}/{result.evaluated} mutants killed, "
        f"95% CI {lo * 100:.0f}-{hi * 100:.0f}%)"
    )
    if result.errored:
        from muteval.runner import PARTIAL_ERRORS

        if result.status == PARTIAL_ERRORS:
            lines.append(
                c(
                    f"   ⚠  INVALID for CI — {result.errored}/{result.total} "
                    f"mutant(s) errored ({result.error_rate * 100:.0f}% > allowed "
                    "budget). Score above is over a SHRUNKEN denominator and is "
                    "shown for diagnosis only; the CLI exits non-zero and the "
                    "badge is n/a. Re-run, or raise --max-error-rate to accept it.",
                    "1;31",
                )
            )
        else:
            lines.append(
                c(
                    f"   {result.errored} mutant(s) errored and were excluded "
                    "(e.g. API timeouts). Re-run to retry them.",
                    "33",
                )
            )

    # Effective score: drop observationally-unchanged survivors from the
    # denominator (their output didn't change on the samples we ran).
    inert = result.inert_survivors
    if inert and result.effective_score is not None:
        eff = result.effective_score * 100
        eff_color = "32" if eff >= 80 else "33" if eff >= 50 else "31"
        elo, ehi = result.effective_score_ci
        lines.append(
            f"Effective score: {c(f'{eff:.0f}%', eff_color)}  "
            f"({result.killed}/{result.evaluated - len(inert)} — excludes "
            f"{len(inert)} inert mutant(s) whose output didn't change; "
            f"95% CI {elo * 100:.0f}-{ehi * 100:.0f}%)"
        )
    elif inert:
        # Every evaluated mutant was observationally unchanged -> no observed
        # degradation to score. Say so rather than crash on a None effective score.
        lines.append(
            c(
                f"Effective score: n/a  (all {len(inert)} evaluated mutant(s) "
                "left the output unchanged on this run — nothing to score)",
                "33",
            )
        )

    flaky = result.flaky
    if flaky:
        lines.append(
            c(
                f"   {len(flaky)} mutant(s) flipped verdict between runs (judge "
                "noise) — raise runs_per_mutant to stabilize.",
                "33",
            )
        )
    lines.append("")

    survivors = result.survivors
    if not survivors:
        lines.append(
            c("✓ No survivors — your evals caught every injected regression.", "32")
        )
        return "\n".join(lines)

    real = result.real_survivors
    if real:
        from muteval.severity import HIGH, LOW, MEDIUM, severity_rank

        real = sorted(real, key=lambda o: severity_rank(o.severity or MEDIUM))
        n_high = sum(1 for o in real if o.severity == HIGH)
        header = c(f"{len(real)} SURVIVED", "31") + (
            "  (output changed but evals didn't notice — real coverage gaps"
        )
        if n_high:
            header += "; " + c(f"{n_high} HIGH-severity", "1;31")
        lines.append(header + "):")
        lines.append(
            c("  ranked by severity: ", "2")
            + c("HIGH", "31") + c(" › ", "2") + c("MED", "33") + c(" › ", "2")
            + c("LOW", "2")
        )
        lines.append("")
        _sev_color = {HIGH: "31", MEDIUM: "33", LOW: "2"}
        _sev_label = {HIGH: "HIGH", MEDIUM: "MED ", LOW: "LOW "}
        from muteval.suggest import suggest_eval

        for o in real:
            sev = o.severity or MEDIUM
            tag = c(f"[{_sev_label[sev]}]", _sev_color[sev])
            lines.append(f"  {tag} {c('SURVIVED', '31')}  [{o.mutant.operator}]")
            lines.append(f"            {o.mutant.description}")
            lines.append(c(f"            fix: {suggest_eval(o)}", "36"))
            if o.min_margin is not None and o.closest_eval:
                lines.append(
                    c(
                        f"            ↳ near miss: passed {o.closest_eval} by only "
                        f"+{o.min_margin:.3f}",
                        "33",
                    )
                )

    if inert:
        lines.append("")
        lines.append(
            c(f"{len(inert)} observationally unchanged", "2")
            + "  (output identical to baseline on this run — NOT eval blind "
            "spots; excluded from the effective score. For a stochastic system "
            "this is not proof of equivalence — raise runs_per_mutant):"
        )
        for o in inert:
            lines.append(
                f"  {c('inert', '2')}     [{o.mutant.operator}] {o.mutant.description}"
            )

    lines.append("")
    lines.append(
        "Each real survivor is an output change your evals would NOT notice. "
        "Write an eval that fails on it, then re-run."
    )
    return "\n".join(lines)


def format_probe_card(results, use_color: bool = True) -> str:
    """Render probe results as an eval-quality report card (no composite score)."""

    def c(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    lines = ["", c("muteval — eval quality report card", "1"), ""]
    if not results:
        lines.append("No probes ran.")
        return "\n".join(lines)
    for r in results:
        tag = c("PASS", "32") if r.ok else c("WARN", "33")
        lines.append(f"  [{tag}] {c(r.name, '1')}")
        lines.append(f"         {r.summary}")
        if r.detail:
            lines.append(c(f"         {r.detail}", "2"))
        lines.append("")
    return "\n".join(lines)


def format_probe_card_html(results, title: str = "muteval — eval quality report card") -> str:
    """Render the probe panel as a standalone HTML page. Deliberately NO composite
    score — the panel is a set of separately-interpretable signals."""
    cards = []
    for r in results:
        state = "pass" if r.ok else "warn"
        badge = "PASS" if r.ok else "WARN"
        detail = f'<div class="pd">{html.escape(r.detail)}</div>' if r.detail else ""
        cards.append(
            f'''<div class="card {state}">
  <div class="chd"><span class="badge {state}">{badge}</span>
    <span class="pn">{html.escape(r.name)}</span></div>
  <div class="psum">{html.escape(r.summary)}</div>
  {detail}
</div>'''
        )
    body = "\n".join(cards) or "<p>No probes ran.</p>"
    n_warn = sum(1 for r in results if not r.ok)
    subtitle = (
        f"{n_warn} of {len(results)} probes flagged an issue"
        if results else "no probes ran"
    )
    return f"""<!doctype html>
<meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1f2328}}
 h1{{font-size:1.4rem;margin-bottom:.2rem}} .muted{{color:#656d76}}
 .card{{border:1px solid #d0d7de;border-left-width:5px;border-radius:8px;padding:.7rem 1rem;margin:.7rem 0}}
 .card.pass{{border-left-color:#2ea043}} .card.warn{{border-left-color:#d29922}}
 .chd{{display:flex;gap:.6rem;align-items:center}} .pn{{font-family:ui-monospace,monospace;font-weight:600}}
 .badge{{font-size:.72rem;font-weight:700;padding:.1rem .45rem;border-radius:4px;color:#fff}}
 .badge.pass{{background:#2ea043}} .badge.warn{{background:#d29922}}
 .psum{{margin:.4rem 0}} .pd{{color:#656d76;font-size:.9rem}}
</style>
<h1>{html.escape(title)}</h1>
<p class="muted">{subtitle} — no composite score (each signal stands on its own).</p>
{body}
<p class="muted" style="margin-top:2rem;font-size:.85rem">Generated by muteval.</p>
"""


# The JSON schema version. Bump on any breaking change to result_to_dict's shape;
# consumers can branch on it. Snapshotted in tests/test_output.py.
RESULT_SCHEMA_VERSION = 1

# Patterns that must never appear in emitted JSON/logs (defense in depth: a
# survivor description or error string could echo a prompt containing a key).
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}"          # OpenAI-style
    r"|gsk_[A-Za-z0-9_\-]{8,}"          # Groq-style
    r"|AIza[A-Za-z0-9_\-]{20,}"         # Google API keys
    r"|(?i:(?:api[_-]?key|authorization|bearer)\s*[:=]\s*)\S+)"
)


def _redact(obj):
    """Recursively replace secret-looking substrings in any string value."""
    if isinstance(obj, str):
        return _SECRET_RE.sub("[REDACTED]", obj)
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def _severity_sorted(survivors):
    """Survivors ordered HIGH severity first (stable), for triage."""
    from muteval.severity import MEDIUM, severity_rank

    return sorted(survivors, key=lambda o: severity_rank(o.severity or MEDIUM))


def result_to_dict(result) -> dict:
    """Machine-readable summary of a MutationResult (for --json / CI / reports).

    Secrets (API keys) are redacted from all string fields before returning.
    """
    from muteval.suggest import suggest_eval

    score = result.score
    eff = result.effective_score
    return _redact({
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": result.status,
        "baseline_passed": result.baseline_passed,
        "baseline_error": result.baseline_error,
        "score": round(score, 4) if score is not None else None,
        "effective_score": round(eff, 4) if eff is not None else None,
        "score_ci": [round(x, 4) for x in result.score_ci],
        "effective_score_ci": [round(x, 4) for x in result.effective_score_ci],
        "killed": result.killed,
        "evaluated": result.evaluated,
        "total": result.total,
        "errored": result.errored,
        "error_rate": round(result.error_rate, 4),
        "inert": len(result.inert_survivors),
        "high_severity_survivors": len(result.high_severity_survivors),
        "survivors": [
            {
                "id": i,
                "operator": o.mutant.operator,
                "description": o.mutant.description,
                "severity": o.severity,
                "fix": suggest_eval(o),
                "baseline_output": o.baseline_output,
                "mutant_output": o.mutant_output,
            }
            for i, o in enumerate(_severity_sorted(result.real_survivors))
        ],
    })


def run_manifest(result, config, operators=None, seed=None) -> dict:
    """A reproducible-run manifest: provenance (version, model, seed, operator
    set, config fingerprint, timestamp) + the machine-readable result. Committing
    this next to a real-LLM-judge run makes the number auditable and repeatable.
    Secrets are redacted."""
    import hashlib
    import platform
    import sys as _sys
    from datetime import datetime, timezone

    from muteval import __version__

    system = getattr(config, "system", None)
    key = repr(system.key()) if system is not None else repr(getattr(config, "prompt", ""))
    fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return _redact({
        "manifest_version": 1,
        "muteval_version": __version__,
        "python": _sys.version.split()[0],
        "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "model": getattr(system, "model", None) if system is not None else None,
            "operators": list(operators) if operators else "all",
            "seed": seed,
            "n_cases": len(config.cases) if config.cases else 0,
            "eval_names": list(config.eval_names),
            "runs_per_mutant": config.runs_per_mutant,
            "system_fingerprint": fingerprint,
        },
        "result": result_to_dict(result),
    })


def badge_dict(result, label: str = "eval coverage") -> dict:
    """A shields.io endpoint payload for the effective mutation score."""
    eff = result.effective_score
    if eff is None or result.status != "valid":
        return {"schemaVersion": 1, "label": label, "message": "n/a",
                "color": "lightgrey"}
    pct = round(eff * 100)
    color = "brightgreen" if pct >= 80 else "yellow" if pct >= 50 else "red"
    return {"schemaVersion": 1, "label": label, "message": f"{pct}%", "color": color}


def _diff_html(base: str, mutant: str) -> str:
    """A minimal line-diff of two outputs as escaped, colored HTML rows."""
    import difflib

    rows = []
    for ln in difflib.unified_diff(
        base.splitlines(), mutant.splitlines(),
        fromfile="baseline", tofile="mutant", lineterm="",
    ):
        cls = "add" if ln.startswith("+") else "del" if ln.startswith("-") else "ctx"
        rows.append(f'<div class="dl {cls}">{html.escape(ln)}</div>')
    return "".join(rows) or '<div class="dl ctx">(no textual diff)</div>'


def format_report_html(data: dict, title: str = "muteval — eval coverage report") -> str:
    """Render a result_to_dict() payload (or a saved last_run.json) as a
    self-contained HTML report: score, survivors, and baseline→mutant diffs."""
    def pct(x):
        return "n/a" if x is None else f"{round(x * 100)}%"

    status = data.get("status", "unknown")
    valid = status == "valid"
    eff = data.get("effective_score")
    bar_color = (
        "#2ea043" if (eff or 0) >= 0.8 else "#d29922" if (eff or 0) >= 0.5 else "#f85149"
    )
    ci = data.get("effective_score_ci") or [0, 0]
    survivors = data.get("survivors", [])

    cards = []
    for s in survivors:
        sev = (s.get("severity") or "medium").lower()
        base, mut = s.get("baseline_output"), s.get("mutant_output")
        diff = _diff_html(base, mut) if (base is not None and mut is not None) else \
            '<div class="dl ctx">(output unchanged / not captured)</div>'
        cards.append(
            f'''<div class="card {sev}">
  <div class="chd"><span class="sev {sev}">{sev.upper()}</span>
    <span class="op">{html.escape(str(s.get("operator", "")))}</span>
    <span class="cid">#{s.get("id", "")}</span></div>
  <div class="desc">{html.escape(str(s.get("description", "")))}</div>
  <div class="fix"><b>fix:</b> {html.escape(str(s.get("fix", "") or "—"))}</div>
  <div class="diff">{diff}</div>
</div>'''
        )
    cards_html = "\n".join(cards) or '<p class="ok">No survivors — your evals caught every injected regression.</p>'

    banner = "" if valid else (
        f'<div class="warn">⚠ INVALID / INCOMPLETE run (status: {html.escape(status)}) '
        "— the score below is not trustworthy.</div>"
    )

    return f"""<!doctype html>
<meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1f2328}}
 h1{{font-size:1.4rem}} .muted{{color:#656d76}}
 .warn{{background:#ffebe9;border:1px solid #ff818266;padding:.6rem .8rem;border-radius:6px;margin:1rem 0;color:#a40e26}}
 .score{{font-size:2.4rem;font-weight:700}}
 .track{{height:12px;background:#eaeef2;border-radius:6px;overflow:hidden;margin:.4rem 0 1rem}}
 .fill{{height:100%;background:{bar_color}}}
 .stats{{display:flex;gap:1.5rem;flex-wrap:wrap;margin:.5rem 0 1.5rem}} .stats div b{{display:block;font-size:1.2rem}}
 .card{{border:1px solid #d0d7de;border-left-width:5px;border-radius:8px;padding:.8rem 1rem;margin:.8rem 0}}
 .card.high{{border-left-color:#f85149}} .card.medium{{border-left-color:#d29922}} .card.low{{border-left-color:#9aa0a6}}
 .chd{{display:flex;gap:.6rem;align-items:center}} .op{{font-family:ui-monospace,monospace;font-weight:600}} .cid{{color:#8b949e;margin-left:auto}}
 .sev{{font-size:.72rem;font-weight:700;padding:.1rem .4rem;border-radius:4px;color:#fff}}
 .sev.high{{background:#f85149}} .sev.medium{{background:#d29922}} .sev.low{{background:#9aa0a6}}
 .desc{{margin:.4rem 0}} .fix{{color:#0969da;font-size:.9rem;margin:.3rem 0}}
 .diff{{background:#f6f8fa;border-radius:6px;padding:.4rem;font-family:ui-monospace,monospace;font-size:.82rem;overflow:auto;margin-top:.5rem}}
 .dl{{white-space:pre-wrap}} .dl.add{{background:#e6ffec;color:#116329}} .dl.del{{background:#ffebe9;color:#a40e26}} .dl.ctx{{color:#656d76}}
 .ok{{color:#116329;font-weight:600}}
</style>
<h1>{html.escape(title)}</h1>
{banner}
<div class="score">{pct(eff)} <span class="muted" style="font-size:1rem">effective coverage</span></div>
<div class="track"><div class="fill" style="width:{round((eff or 0)*100)}%"></div></div>
<div class="stats">
 <div><b>{pct(data.get("score"))}</b>raw score</div>
 <div><b>{ci[0]*100:.0f}–{ci[1]*100:.0f}%</b>95% CI</div>
 <div><b>{data.get("killed",0)}/{data.get("evaluated",0)}</b>killed / evaluated</div>
 <div><b>{data.get("inert",0)}</b>inert (excluded)</div>
 <div><b>{data.get("high_severity_survivors",0)}</b>high-severity survivors</div>
</div>
<h2>Survivors <span class="muted">({len(survivors)})</span></h2>
{cards_html}
<p class="muted" style="margin-top:2rem;font-size:.85rem">Generated by muteval. Each survivor is an output change your evals did not catch — write an eval that fails on it, then re-run.</p>
"""
