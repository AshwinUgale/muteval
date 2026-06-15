"""muteval — mutation testing for LLM eval suites.

The question muteval answers is not "do my evals pass?" but
"would my evals *fail* if my system silently got worse?"

It deliberately degrades the thing under test (the prompt today; retrieved
context and tools tomorrow), reruns your existing eval suite against each
mutant, and reports the **mutation score**: the fraction of injected
regressions your evals actually caught. Mutants your evals fail to catch are
"survivors" — concrete blind spots in your eval coverage.
"""

from muteval.config import MutEvalConfig
from muteval.mutators import Mutant, generate_mutants
from muteval.runner import MutationResult, run_mutation_testing

__version__ = "0.0.1"

__all__ = [
    "MutEvalConfig",
    "Mutant",
    "generate_mutants",
    "MutationResult",
    "run_mutation_testing",
    "__version__",
]
