"""v0.6: human-agreement probe — does your eval agree with a human?

Every other probe measures internal properties of the suite. This one measures
the thing that ultimately matters: whether the machine verdict matches a human's
judgement. Workflow:

  1. ``muteval label --config ... --out worksheet.csv`` writes one row per
     (case, eval) with the machine verdict and a blank ``human_label`` column.
  2. A human fills ``human_label`` (pass/fail) for ~30-50 rows.
  3. This probe reads the filled sheet and reports **Cohen's kappa** (agreement
     beyond chance) with a bootstrap CI.

Without labels it reports "not assessed" — it never fabricates agreement. By
convention the probe looks for ``.muteval/labels.csv``; pass a path for anything
else.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

from muteval.probes.base import ProbeResult

_TRUE = {"pass", "true", "1", "yes", "y", "ok", "good"}
_FALSE = {"fail", "false", "0", "no", "n", "bad"}


def _to_bool(v) -> Optional[bool]:
    s = str(v).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def load_label_rows(path) -> List[Tuple[bool, bool]]:
    """Read (machine_verdict, human_label) pairs from a filled worksheet CSV.
    Rows whose human_label is blank/unparseable are skipped."""
    rows: List[Tuple[bool, bool]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            m = _to_bool(r.get("machine_verdict"))
            h = _to_bool(r.get("human_label"))
            if m is not None and h is not None:
                rows.append((m, h))
    return rows


def human_agreement_from_rows(pairs: List[Tuple[bool, bool]]) -> ProbeResult:
    from muteval.stats import cohens_kappa, kappa_ci

    if len(pairs) < 2:
        return ProbeResult(
            name="human_agreement", ok=True,
            summary="not assessed (need >= 2 human-labeled rows)",
            detail="run `muteval label`, fill the human_label column, then re-run.",
            metrics={"assessed": False, "n": len(pairs)},
        )
    machine = [m for m, _ in pairs]
    human = [h for _, h in pairs]
    kappa = cohens_kappa(machine, human)
    lo, hi = kappa_ci(machine, human)
    agree = sum(1 for m, h in pairs if m == h) / len(pairs)
    # Landis & Koch: >=0.6 substantial. Flag weak agreement.
    ok = kappa is not None and kappa >= 0.6
    ci = f" [95% CI {lo:.2f}–{hi:.2f}]" if lo is not None else ""
    return ProbeResult(
        name="human_agreement", ok=ok,
        summary=f"Cohen's kappa {kappa:.2f}{ci} over {len(pairs)} labeled rows "
        f"({agree * 100:.0f}% raw agreement)",
        detail=(
            "substantial agreement with the human (kappa >= 0.6)."
            if ok else
            "weak agreement — the eval and a human often disagree; revisit the "
            "check or its threshold."
        ),
        metrics={"assessed": True, "kappa": kappa, "ci": [lo, hi],
                 "n": len(pairs), "raw_agreement": agree},
    )


def human_agreement(config, labels_path=None) -> ProbeResult:
    """Registry entry point: load labels from ``labels_path`` (default
    ``.muteval/labels.csv``) and assess, or report 'not assessed' if absent."""
    path = Path(labels_path) if labels_path else Path(".muteval") / "labels.csv"
    if not path.exists():
        return ProbeResult(
            name="human_agreement", ok=True,
            summary="not assessed (no human labels)",
            detail=f"run `muteval label` to emit a worksheet, fill it, save as {path}.",
            metrics={"assessed": False},
        )
    return human_agreement_from_rows(load_label_rows(path))
