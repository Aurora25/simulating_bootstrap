import pytest
import numpy as np

from src.confidence_intervals import confidence_interval

def test_easy_confidence_interval():
    test = [10] * 10
    conf_int = confidence_interval(test, 0.8)
    assert conf_int.lower_bound == 10
    assert conf_int.upper_bound == 10


def conf_interval_testing(test, coverage):
    conf_int = confidence_interval(test, coverage)
    assert round((conf_int.upper_bound - conf_int.lower_bound) / (max(test) - min(test)), 2) == coverage

def test_confidence_interval95():
    test = list(range(10))
    conf_interval_testing(test, 0.95)

def test_confidence_interval95_reverse():
    test = list(reversed(range(10)))
    conf_interval_testing(test, 0.95)

def test_confidence_interval95_more_complex():
    test = [0] + [5] * 7 + [10]
    conf_int = confidence_interval(test, 0.95)

    lower = np.percentile(test, 2.5) # expect 1.0
    upper = np.percentile(test, 97.5) # expect 9.0

    assert np.round(conf_int.lower_bound, 1) == lower
    assert np.round(conf_int.upper_bound, 1) == upper

def test_confidence_interval95_more_complex_shuffled():
    test = list(np.random.permutation([0] + [5] * 7 + [10]))
    conf_int = confidence_interval(test, 0.95)

    lower = np.percentile(test, 2.5) # expect 1.0
    upper = np.percentile(test, 97.5) # expect 9.0

    assert np.round(conf_int.lower_bound, 1) == lower
    assert np.round(conf_int.upper_bound, 1) == upper
