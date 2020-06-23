# Get the population and the metrics we are looking at
from typing import Dict, Any, List, Tuple, Callable, Iterable

import numpy as np
from tqdm import tqdm

from src.DataClasses import MetricResult, ConfidenceInterval
from src.bootstrap import bootstrap
from src.confidence_intervals import confidence_interval


def get_agg(bs_res, agg_fun, metrics: List[str]):
    agg = {metric_name: agg_fun([int(sample[metric_name].pop_metric_in_conf) for sample in bs_res])
           for metric_name in metrics}
    return agg


def get_agg_dist(bs_res, agg_fun, metrics: List[str]):
    agg = {metric_name: agg_fun([sample[metric_name].metric_confidence.upper_bound
                                 - sample[metric_name].metric_confidence.lower_bound
                                 for sample in bs_res])
           for metric_name in metrics}
    return agg


def get_population_and_metrics(simulation_function, pop_size, metric_functions: Dict[str, Any]):
    """simulation_function is the return value of the above get_population* functions """
    pop = simulation_function(pop_size)
    metrics = {metric_name: metric(pop) for metric_name, metric in metric_functions.items()}

    return pop, metrics


def get_sample(pop, sample_size):
    """Drawing a random sample without replacement from the population"""
    pop_t = pop.transpose()
    return (pop_t[np.random.choice(pop_t.shape[0], sample_size, replace=False), :]
            .transpose()
            )


def simulation(population_function, metric_functions: Dict[str, Any], num_sample_draws: int,
               conf_interval: Callable[[Iterable[Any], float], ConfidenceInterval] = confidence_interval,
               pop_size: int = 100000, sample_size: int = 1000, resample_size: int = 1000, num_bootstraps: int = 1000,
               coverage: float = 0.95) -> Tuple[List[Dict[str, MetricResult]], Dict[str, Any]]:
    """
    Simulates several sample draws from the actual population to see if about 90% of the time the measured metrics
    actually lie within the calculated confidence interval given by the bootstrapping procedure.

    Args:
        population_function: the function that returns a population of size pop_size
        metric_functions: a list of functions that take an iterable as input and return a single value as output
        num_sample_draws: How often do we want to simulate sampling from the distribution? (without replacement)
        conf_interval: the function which calculates the confidence interval from the bootstrapping procedure. There are
                       several ways of calculating a confidence interval.
        pop_size: size of the true population
        sample_size: size of the sample for which we simulate
        resample_size: size of the bootstrap resamples (it is recommended to use the sample size or at least 50% of it)
        num_bootstraps: how often the bootstrap resamples should be drawn
        coverage: the coverage of the confidence interval

    Returns:
        A Tuple sim_res, metrics:
            `sim_res` is a list of dictionaries of MetricResults, which tells us the confidence
            interval for this bootstrap and if the population metric is within the confidence interval.
            `metrics` is a dictionary of the population metrics.
    """
    sim_res = []
    pop, metrics = get_population_and_metrics(population_function, pop_size=pop_size,
                                              metric_functions=metric_functions)

    for _ in tqdm(range(num_sample_draws)):
        sample = get_sample(pop, sample_size=sample_size)
        res = bootstrap(sample, num_iter=num_bootstraps, resample_size=resample_size, metrics=metric_functions)

        # TODO: This can be a function
        bs_res = {}
        for metric_name, _ in metric_functions.items():
            bs_metric_res = [single_res[metric_name] for single_res in res]
            conf = conf_interval(bs_metric_res, coverage)
            pop_metric_in_conf = True if conf.lower_bound <= metrics[metric_name] <= conf.upper_bound else False

            bs_res[metric_name] = MetricResult(metric_confidence=conf, pop_metric_in_conf=pop_metric_in_conf)

        sim_res.append(bs_res)

    return sim_res, metrics