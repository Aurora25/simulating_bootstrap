"""
This file contains the functions that easily allow for plug and play with the simulation function without the need
of an args kwargs situation.
Each function returns a function that only needs the population size as input. This function can then be given to
the simulation as population_function.
"""
from typing import Callable, Iterable, Any

import numpy as np

__author__ = "Sanja Stegerer"


def get_population_gauss(mu: float, sigma: float) -> Callable[[int], Iterable[float]]:
    def get_pop(pop_size):
        pop = np.array(np.random.normal(loc=mu, scale=sigma, size=pop_size), ndmin=2)
        return pop

    return get_pop


def get_population_binomial(p: float, n: int = 1) -> Callable[[int], Iterable[int]]:
    def get_pop(pop_size):
        pop = np.array(np.random.binomial(n=n, p=p, size=pop_size), ndmin=2)
        return pop

    return get_pop


def get_population_binary_classification(p_pop: float, p_pred: float, n: int = 1) \
        -> Callable[[int], Iterable[Any]]:
    """
    Simulating a binary classification result.
    Args:
        p_pop: probability of the 1 class for the population
        p_pred:  probability of the 1 class for the classification
        n: number of consecutive experiments to run (should be the same for both classifier and truth)

    Returns:
        Function that takes pop_size as input and will return a matrix of size (2, pop_size)
    """
    def get_pop(pop_size):
        pop_truth = np.random.binomial(n=n, p=p_pop, size=pop_size)
        pop_pred = np.random.binomial(n=n, p=p_pred, size=pop_size)
        return np.array([pop_truth, pop_pred], ndmin=2)

    return get_pop
