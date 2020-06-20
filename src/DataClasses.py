from dataclasses import dataclass

__author__ = "Sanja Stegerer"

@dataclass
class ConfidenceInterval:
    """Class for keeping track of confidence intervals"""
    lower_bound: float
    upper_bound: float


@dataclass
class MetricResult:
    """Class for keeping track of an item in inventory."""
    metric_confidence: ConfidenceInterval
    pop_metric_in_conf: bool