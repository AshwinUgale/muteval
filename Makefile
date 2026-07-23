.PHONY: test test-all cov verify mutmut mutmut-results clean

# Fast suite (skips slow Monte-Carlo simulations).
test:
	pytest -q -m "not slow"

# Everything, including slow simulations + reference cross-checks.
test-all:
	pytest -q

# Coverage gate (>=90% per ROADMAP §7.5).
cov:
	coverage run --source=src/muteval -m pytest -q -m "not slow"
	coverage report --fail-under=90

# Statistical rigor: hand-rolled math vs reference libs + MC coverage.
verify:
	pip install -e ".[dev,verify]"
	pytest -q tests/test_stats_reference.py tests/test_ci_coverage.py

# Dogfood: mutation-test our own math. See docs/MUTMUT.md.
mutmut:
	mutmut run

mutmut-results:
	mutmut results

clean:
	rm -rf mutants .mutmut-cache .coverage htmlcov build dist
