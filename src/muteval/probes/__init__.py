"""The eval evaluator — probes that rate an eval suite's quality."""

from muteval.probes.base import PROBES, ProbeResult, register_probe, run_probes
from muteval.probes.discrimination import discrimination
from muteval.probes.judge_reliability import judge_reliability
from muteval.probes.redundancy import redundancy
from muteval.probes.statistical_adequacy import statistical_adequacy
from muteval.probes.threshold_calibration import threshold_calibration

register_probe("statistical_adequacy", statistical_adequacy)
register_probe("judge_reliability", judge_reliability)
register_probe("discrimination", discrimination)
register_probe("redundancy", redundancy)
register_probe("threshold_calibration", threshold_calibration)

__all__ = ["PROBES", "ProbeResult", "register_probe", "run_probes"]
