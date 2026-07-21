"""muteval — mutation testing for LLM eval suites.

The question muteval answers is not "do my evals pass?" but
"would my evals *fail* if my system silently got worse?"

It deliberately degrades the thing under test (the prompt, and — via the
``System`` target — retrieved context and tools), reruns your existing eval
suite against each mutant, and reports the **mutation score**: the fraction of
injected regressions your evals actually caught. Mutants your evals fail to
catch are "survivors" — concrete blind spots in your eval coverage.
"""

from muteval.config import MutEvalConfig
from muteval.evals import EvalOutcome, coerce_outcome
from muteval.mutators import (
    Mutant,
    generate_mutants,
    make_downgrade_model,
    make_weaken_modals,
    register_operator,
)
from muteval.runner import MutationResult, run_mutation_testing
from muteval.system import System, as_system

try:  # keep in sync with pyproject automatically
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("muteval")
except Exception:  # noqa: BLE001 - dev tree without installed metadata
    __version__ = "0.0.0+unknown"

__all__ = [
    "MutEvalConfig",
    "System",
    "as_system",
    "EvalOutcome",
    "coerce_outcome",
    "Mutant",
    "generate_mutants",
    "register_operator",
    "make_weaken_modals",
    "make_downgrade_model",
    "MutationResult",
    "run_mutation_testing",
    "__version__",
]
