"""v0.5: a local result cache so re-runs skip unchanged work.

The cost of mutation testing is re-running the suite (the model call + the
judges) once per mutant. Across runs, most of that work is *identical* — the same
(system, case) yields the same output for a deterministic system, and the same
(output, eval) yields the same outcome. This sqlite-backed cache stores both,
keyed by a hash of the inputs, so a second identical run makes ZERO model/judge
calls.

Determinism: caching assumes the system + evals are deterministic. The runner
disables it when ``runs_per_mutant > 1`` (repeated runs exist precisely to
observe non-determinism, which a cache would erase).

The value is also the trust boundary: a cache is keyed by ``System.key()`` (which
includes the full prompt + context + model), the case, and the eval label — so a
changed prompt/context/model/case/eval never collides with a stale entry.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from typing import Any, Optional

from muteval.evals import EvalOutcome
from muteval.system import System


def _case_repr(case: Any) -> str:
    try:
        return json.dumps(case, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(case)


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

    # --- run outputs --------------------------------------------------------
    def _output_key(self, system: System, case: Any) -> str:
        return "out:" + _hash(system.key(), _case_repr(case))

    def get_output(self, system: System, case: Any) -> Optional[str]:
        v = self._get(self._output_key(system, case))
        self.hits += v is not None
        self.misses += v is None
        return v

    def set_output(self, system: System, case: Any, output: str) -> None:
        self._set(self._output_key(system, case), output)

    # --- eval outcomes ------------------------------------------------------
    def _outcome_key(self, system: System, case: Any, label: str) -> str:
        return "eval:" + _hash(system.key(), _case_repr(case), label or "")

    def get_outcome(self, system: System, case: Any, label: str) -> Optional[EvalOutcome]:
        v = self._get(self._outcome_key(system, case, label))
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        d = json.loads(v)
        return EvalOutcome(
            passed=d["passed"], score=d["score"], threshold=d["threshold"],
            name=d["name"], detail=d["detail"],
        )

    def set_outcome(self, system: System, case: Any, label: str, outcome: EvalOutcome) -> None:
        d = {
            "passed": bool(outcome.passed), "score": outcome.score,
            "threshold": outcome.threshold, "name": outcome.name,
            "detail": outcome.detail,
        }
        self._set(self._outcome_key(system, case, label), json.dumps(d))

    def close(self) -> None:
        self._conn.close()
