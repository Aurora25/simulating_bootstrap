from typing import List, Any, Iterable

import numpy as np

from src.DataClasses import ConfidenceInterval

__author__ = "Sanja Stegerer"


def confidence_interval(statistics: List[Any], coverage: float = 0.95):
    """
    Calculate the lower and upper bound for the confidence interval for the given alpha. By default alpha = 0.95
    Args:
        statistics: list of data for which we want to get confidence statistics. DataType should be compatible with
                    numpy.percentile()
        coverage: Floating point number of which percentile to calculate

    Returns:

    """
    sstat = sorted(statistics)

    alpha = 1 - coverage

    lower = np.percentile(sstat, (alpha / 2) * 100)
    upper = np.percentile(sstat, (coverage + (alpha / 2)) * 100)

    return ConfidenceInterval(lower, upper)


def analytic_confidence_interval(sample_size, avg, std=1, crtc_val=2):
    """For gaussian distributions only, critical value for a 90% confidence interval: 1.645
    Default for 95% confidence interval"""

    lower = avg - (crtc_val * std) / np.sqrt(sample_size)
    upper = avg + (crtc_val * std) / np.sqrt(sample_size)

    return ConfidenceInterval(lower, upper)
