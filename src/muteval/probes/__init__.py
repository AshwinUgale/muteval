"""The eval evaluator — probes that rate an eval suite's quality."""

from muteval.probes.base import PROBES, ProbeResult, register_probe, run_probes
from muteval.probes.judge_reliability import judge_reliability
from muteval.probes.statistical_adequacy import statistical_adequacy

register_probe("statistical_adequacy", statistical_adequacy)
register_probe("judge_reliability", judge_reliability)

__all__ = ["PROBES", "ProbeResult", "register_probe", "run_probes"]
