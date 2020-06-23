from typing import Iterable, Any, Dict

import numpy as np

__author__ = "Sanja Stegerer"


def _assure_np_matrix(sample: Iterable[Any]):
    """
    Make sure the given sample is in numpy matrix format:

    Args:
        sample: list or numpy array

    Returns:
        numpy matrix format of the given sample
    """

    if not isinstance(sample, np.ndarray):
        return np.array(sample, ndmin=2)
    else:
        return sample


def _bootstrap_sample(sample: np.array, resample_size: int):
    """
    Create one bootstrap sample.
    For multiindex the expected input is a numpy matrix with row-wise pairs. e.g. [[1, 2], [3, 4], [5, 6]]

    Args:
        sample: numpy matrix with the examples to be sampled as rows
        resample_size: The size of the sample to create

    Returns:
        resample: A single sample of the given matrix
    """
    resample_idx = np.random.randint(sample.shape[0], size=resample_size)
    resample = sample[resample_idx, :]

    return resample


def bootstrap(sample: Iterable[Any], num_iter: int, resample_size: int, metrics: Dict[str, Any]):
    """
    For multiindex the expected input is a matrix with columnwise samples from the distribution.
    e.g. [[1,2,3,4], [5,6,7,8]] if the [1,2,3,4] was created by distribution 1 and [5,6,7,8] was created by distribution 2
    it will be handled such that the metrics will compare the rows. In above example the pairwise comparisons will be
    [1,5], [2,6], [3,7], [4,8]

    Args:
        sample: numpy matrix with the examples to be sampled as rows
        num_iter: number of iterations of bootstrapping
        resample_size: The size of the each bootstrap sample to create
        metrics: the metrics to compute

    Returns:
        res: A list of calculated metrics for each bootstrap sample, list length == num_iter
    """

    sample = _assure_np_matrix(sample).transpose()

    res = [{} for _ in range(num_iter)]

    for b_iter in range(num_iter):
        resample = (_bootstrap_sample(sample, resample_size)
                    .transpose()
                    )
        res[b_iter] = {metric_name: metric(resample) for metric_name, metric in metrics.items()}

    return res

