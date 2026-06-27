"""Configuration object that a user defines to describe their system + evals."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from muteval.evals import EvalFn  # noqa: F401  (re-exported for convenience)
from muteval.system import System

# A function that runs the system under test.
#   - legacy/prompt mode: run(prompt: str, case) -> output_text
#   - system mode:        run(system: System, case) -> output_text
RunFn = Callable[..., str]


@dataclass
class MutEvalConfig:
    """Everything muteval needs to grade an eval suite.

    Describe the system under test in one of two ways:

    * ``prompt=...`` (the common case): the mutation target is a prompt string,
      and your ``run`` is called as ``run(prompt, case)``. This is the original,
      unchanged API.
    * ``system=System(...)``: the target is a full ``System`` (prompt + retrieved
      context + tools + model), and your ``run`` is called as ``run(system, case)``
      so it can read the mutated context/tools. Use this for RAG/agent suites.

    Attributes:
        prompt: The system prompt (the thing under test). Mutually exclusive
            with ``system``.
        cases: Inputs fed to the system. Each element is passed to ``run`` and
            to every eval. Can be anything (str, dict, dataclass...).
        run: Calls your model/app with the (possibly mutated) prompt/system and
            returns the text output. See the two modes above.
        evals: List of checks. Each is ``eval(output, case) -> bool | EvalOutcome``
            where a truthy result means the check passed. These are the evals
            being graded.
        runs_per_mutant: How many times to evaluate each mutant. >1 helps with
            non-deterministic systems. A mutant is "killed" if the suite fails
            on *any* run (i.e. the evals managed to detect the degradation).
        eval_names: Optional human labels for evals, used in reports.
        system: The full mutation target. Mutually exclusive with ``prompt``.
    """

    prompt: Optional[str] = None
    cases: Optional[List[Any]] = None
    run: Optional[RunFn] = None
    evals: Optional[List[EvalFn]] = None
    runs_per_mutant: int = 1
    eval_names: List[str] = field(default_factory=list)
    system: Optional[System] = None

    def __post_init__(self) -> None:
        # Which calling convention does the user's run expect?
        self._system_mode = self.system is not None

        if self.prompt is not None and self.system is not None:
            raise ValueError("provide either `prompt` or `system`, not both")

        if self.system is None:
            if not isinstance(self.prompt, str) or not self.prompt.strip():
                raise ValueError(
                    "config.prompt must be a non-empty string (or pass `system=`)"
                )
            self.system = System(prompt=self.prompt)
        else:
            if not isinstance(self.system, System):
                raise ValueError("config.system must be a muteval.System instance")
            # Keep `prompt` populated for any back-compat consumer/report.
            self.prompt = self.system.prompt

        if not self.cases:
            raise ValueError("config.cases must contain at least one case")
        if self.run is None:
            raise ValueError("config.run must be provided")
        if not self.evals:
            raise ValueError("config.evals must contain at least one eval")
        if self.runs_per_mutant < 1:
            raise ValueError("config.runs_per_mutant must be >= 1")

    def invoke(self, system: System, case: Any) -> str:
        """Call the user's ``run`` with the right calling convention."""
        if self._system_mode:
            return self.run(system, case)
        return self.run(system.prompt, case)


def load_config(path: str | Path) -> MutEvalConfig:
    """Load a ``MutEvalConfig`` from a Python file that defines ``config``."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    spec = importlib.util.spec_from_file_location("muteval_user_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["muteval_user_config"] = module
    spec.loader.exec_module(module)

    config = getattr(module, "config", None)
    if not isinstance(config, MutEvalConfig):
        raise TypeError(
            f"{path} must define a module-level variable `config` of type "
            f"MutEvalConfig (found: {type(config).__name__})"
        )
    return config
