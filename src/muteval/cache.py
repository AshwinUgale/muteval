"""v0.5: a local result cache so re-runs skip unchanged work.

The cost of mutation testing is re-running the suite (the model call + the
judges) once per mutant. Across runs, most of that work is *identical* — the same
(system, case) yields the same output for a deterministic system, and the same
(output, eval) yields the same outcome. This sqlite-backed cache stores both,
keyed by a hash of the inputs, so a second identical run makes ZERO model/judge
calls.

Determinism: caching assumes the system + evals are deterministic. The runner
disables it when ``runs_per_mutant > 1`` or ``baseline_runs > 1`` (repeated runs
exist precisely to observe non-determinism, which a cache would erase). If
``run()`` writes into the case (``case["used_context"] = ...`` for an eval to
read), the post-run case is stored with the output and replayed on a hit.

What the keys include (v2) — the trust boundary:

* an OUTPUT is keyed by ``System.key()`` (prompt + context + tools + model +
  extra), the case, and a fingerprint of your ``run`` function — so editing
  ``run`` (e.g. the model it calls in prompt mode) doesn't serve stale outputs;
* an OUTCOME is keyed by the output itself, the case, and a fingerprint of the
  eval (its code, closure values, thresholds; see ``muteval.fingerprint``) — so
  an edited eval, or two different evals sharing a label, never share a result.

(v1 keyed outcomes on the eval's LABEL: an edited ``contains("X1")`` →
``contains("ZZZ")`` kept serving the old verdicts. v1 entries are simply never
read again.) The cache is still only as good as the fingerprint: an eval that
depends on something muteval can't see (a file, a remote rubric) should set a
``cache_version`` attribute — or skip ``--cache``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from typing import Any, Optional, Tuple

from muteval.evals import EvalOutcome
from muteval.system import System

_NAMESPACE = "v2"


def _case_repr(case: Any) -> str:
    try:
        return json.dumps(case, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(case)


def _output_repr(output: Any) -> Optional[str]:
    """A stable text form of an output, or None if it can't be stored faithfully
    (a dict holding objects): such outputs simply aren't cached."""
    try:
        return json.dumps(output, sort_keys=True)
    except (TypeError, ValueError):
        return None


def _hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        # parts may be strings or structured (System.key() is a tuple).
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class Cache:
    """A sqlite key/value store for run outputs and eval outcomes."""

    def __init__(self, path: str):
        self.path = path
        # check_same_thread=False + a lock so the cache is safe under --concurrency.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    # --- low level ----------------------------------------------------------
    def _get(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def _set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
            )
            self._conn.commit()

    def _count(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self.hits += 1
            else:
                self.misses += 1

    # --- run outputs --------------------------------------------------------
    def _output_key(self, system: System, case: Any, run_fp: str) -> str:
        return _hash(_NAMESPACE, "out", system.key(), _case_repr(case), run_fp)

    def lookup_output(
        self, system: System, case: Any, run_fp: str = ""
    ) -> Optional[Tuple[Any, Any]]:
        """``(output, post_run_case)`` or None. ``post_run_case`` is the case as
        ``run()`` left it (None if run() didn't write into it): a cache hit skips
        ``run()``, so its side effects on the case are REPLAYED from here —
        otherwise an eval reading ``case["used_context"]`` would grade stale
        state."""
        v = self._get(self._output_key(system, case, run_fp))
        self._count(v is not None)
        if v is None:
            return None
        d = json.loads(v)
        return d["o"], d.get("c")

    def store_output(
        self,
        system: System,
        case: Any,
        output: Any,
        run_fp: str = "",
        post_run_case: Any = None,
    ) -> None:
        """Store an output — unless it (or the case state run() left behind)
        can't be round-tripped exactly through JSON, in which case it's simply
        not cached (a miss costs a call; a lossy replay could change a verdict)."""
        entry = {"o": output}
        if post_run_case is not None:
            entry["c"] = post_run_case
        try:
            text = json.dumps(entry, sort_keys=True)
            if json.loads(text) != entry:
                return
        except (TypeError, ValueError):
            return
        self._set(self._output_key(system, case, run_fp), text)

    # Back-compat conveniences (no case-state replay).
    def get_output(self, system: System, case: Any, run_fp: str = "") -> Optional[Any]:
        hit = self.lookup_output(system, case, run_fp)
        return None if hit is None else hit[0]

    def set_output(
        self, system: System, case: Any, output: Any, run_fp: str = ""
    ) -> None:
        self.store_output(system, case, output, run_fp)

    # --- eval outcomes ------------------------------------------------------
    def _outcome_key(self, output: Any, case: Any, eval_fp: str) -> Optional[str]:
        text = _output_repr(output)
        if text is None:
            return None
        return _hash(_NAMESPACE, "eval", text, _case_repr(case), eval_fp)

    def get_outcome(
        self, output: Any, case: Any, eval_fp: str, label: str = ""
    ) -> Optional[EvalOutcome]:
        key = self._outcome_key(output, case, eval_fp)
        v = self._get(key) if key is not None else None
        self._count(v is not None)
        if v is None:
            return None
        d = json.loads(v)
        return EvalOutcome(
            passed=d["passed"],
            score=d["score"],
            threshold=d["threshold"],
            # Identical evals may share an entry; report under THIS eval's label.
            name=label or d["name"],
            detail=d["detail"],
            higher_is_better=d.get("higher_is_better", True),
        )

    def set_outcome(
        self, output: Any, case: Any, eval_fp: str, outcome: EvalOutcome
    ) -> None:
        key = self._outcome_key(output, case, eval_fp)
        if key is None:
            return
        d = {
            "passed": bool(outcome.passed),
            "score": outcome.score,
            "threshold": outcome.threshold,
            "name": outcome.name,
            "detail": outcome.detail,
            "higher_is_better": outcome.higher_is_better,
        }
        self._set(key, json.dumps(d))

    def close(self) -> None:
        self._conn.close()
