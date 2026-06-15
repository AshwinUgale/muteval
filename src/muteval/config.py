"""Configuration object that a user defines to describe their system + evals."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List

# A function that runs the system under test: (prompt, case) -> output_text
RunFn = Callable[[str, Any], str]

# A single eval/check: (output_text, case) -> True if it PASSES, False if it fails.
EvalFn = Callable[[str, Any], bool]


@dataclass
class MutEvalConfig:
    """Everything muteval needs to grade an eval suite.

    Attributes:
        prompt: The system prompt (the thing under test) as a string.
        cases: Inputs fed to the system. Each element is passed to ``run`` and
            to every eval. Can be anything (str, dict, dataclass...).
        run: ``run(prompt, case) -> output``. Calls your model/app with the
            (possibly mutated) prompt and returns the text output.
        evals: List of checks. Each is ``eval(output, case) -> bool`` where
            True means the check passed. These are the evals being graded.
        runs_per_mutant: How many times to evaluate each mutant. >1 helps with
            non-deterministic systems. A mutant is "killed" if the suite fails
            on *any* run (i.e. the evals managed to detect the degradation).
        eval_names: Optional human labels for evals, used in reports.
    """

    prompt: str
    cases: List[Any]
    run: RunFn
    evals: List[EvalFn]
    runs_per_mutant: int = 1
    eval_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("config.prompt must be a non-empty string")
        if not self.cases:
            raise ValueError("config.cases must contain at least one case")
        if not self.evals:
            raise ValueError("config.evals must contain at least one eval")
        if self.runs_per_mutant < 1:
            raise ValueError("config.runs_per_mutant must be >= 1")


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
