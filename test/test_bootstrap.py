
import numpy as np
from collections import defaultdict

import pytest

from src.bootstrap import _bootstrap_sample, bootstrap

@pytest.fixture
def test_matrix():
    return np.array([[1, 2], [3, 4], [5, 6]])

@pytest.fixture
def sample_list_type():
    return list(range(100))

@pytest.fixture
def sample_list_md_type():
    return [list(range(10)), [x + 10 for x in range(10)]]

@pytest.fixture
def sample_array_type():
    return np.array(range(100), ndmin=2)

@pytest.fixture
def sample_array_md_type():
    return np.array([list(range(10)), [x + 10 for x in range(10)]])


def create_count_of_sampled_items(res_bs):
    """
    Creates a dictionary with strings of the entries as keys, regular collection Counters won't work, because
    an ndarray is not hashable.
    """
    counter = defaultdict(int)
    for row in res_bs:
        counter[str(row)] += 1

    return dict(counter)


class TestSubFunctions:
    """Testing the bootstrap sub-functions for their accuracy"""

    def test__bootstrap_sample_correct_shape(self, test_matrix):
        res_bs_sample = _bootstrap_sample(test_matrix, 4)
        assert res_bs_sample.shape == (4, 2)

    def test__bootstrap_sample_samples_several(self, test_matrix):
        res_bs_sample = _bootstrap_sample(test_matrix, 4)
        counter = create_count_of_sampled_items(res_bs_sample)
        assert len(counter) > 1


    def test__bootstrap_sample_has_at_least_one_duplicate(self, test_matrix):
        res_bs_sample = _bootstrap_sample(test_matrix, 4)
        counter = create_count_of_sampled_items(res_bs_sample)
        assert any([True if count > 1 else False for count in counter.values()])


class TestOneDimensionalBootstrap:
    """Testing one-dimensional bootstrap"""

    def test_bootstrap_input_list_samples_several(self, sample_list_type):
        res = bootstrap(sample_list_type, num_iter=100, resample_size=100, metrics={"mean": np.mean})

        # if all means are the same there could be two issues: not actually calculating the mean from the sample
        # OR not actually sampling with replacement
        counter = create_count_of_sampled_items(res)
        assert len(dict(counter)) > 1

    def test_bootstrap_input_list_has_at_least_one_duplicate(self, sample_list_type):
        res = bootstrap(sample_list_type, num_iter=100, resample_size=100, metrics={"mean": np.mean})
        counter = create_count_of_sampled_items(res)
        assert any([True if count > 1 else False for count in counter.values()])

        # checking if the mean lies around the true mean of the given 'sample' - this cannot be tested without fixing
        # the seed, which I do not want to in this implementation
        print(len(np.array(res)[[abs(metric['mean'] - 50) >= 5 for metric in res]]) < 15)

    def test_bootstrap_input_array_samples_several(self, sample_array_type):
        res = bootstrap(sample_array_type, num_iter=100, resample_size=100, metrics={"mean": np.mean})

        # if all means are the same there could be two issues: not actually calculating the mean from the sample
        # OR not actually sampling with replacement
        counter = create_count_of_sampled_items(res)
        assert len(dict(counter)) > 1

    def test_bootstrap_input_array_has_at_least_one_duplicate(self, sample_array_type):
        res = bootstrap(sample_array_type, num_iter=100, resample_size=100, metrics={"mean": np.mean})
        counter = create_count_of_sampled_items(res)
        assert any([True if count > 1 else False for count in counter.values()])

        # checking if the mean lies around the true mean of the given 'sample' - this cannot be tested without fixing
        # the seed, which I do not want to in this implementation
        print(len(np.array(res)[[abs(metric['mean'] - 50) >= 5 for metric in res]]) < 15)

class TestTwoDimensionalBootstrap:
    """Testing one-dimensional bootstrap"""

    def test_bootstrap_input_list_samples_several(self, sample_list_md_type):
        res = bootstrap(sample_list_md_type, num_iter=100, resample_size=100, metrics={"mean": np.mean})

        # if all means are the same there could be two issues: not actually calculating the mean from the sample
        # OR not actually sampling with replacement
        counter = create_count_of_sampled_items(res)
        assert len(dict(counter)) > 1

    def test_bootstrap_input_list_has_at_least_one_duplicate(self, sample_list_md_type):
        res = bootstrap(sample_list_md_type, num_iter=100, resample_size=100, metrics={"mean": np.mean})
        counter = create_count_of_sampled_items(res)
        assert any([True if count > 1 else False for count in counter.values()])

        # checking if the mean lies around the true mean of the given 'sample' - this cannot be tested without fixing
        # the seed, which I do not want to in this implementation
        print(len(np.array(res)[[abs(metric['mean'] - 50) >= 5 for metric in res]]) < 15)

    def test_bootstrap_input_array_samples_several(self, sample_array_md_type):
        res = bootstrap(sample_array_md_type, num_iter=100, resample_size=100, metrics={"mean": np.mean})

        # if all means are the same there could be two issues: not actually calculating the mean from the sample
        # OR not actually sampling with replacement
        counter = create_count_of_sampled_items(res)
        assert len(dict(counter)) > 1

    def test_bootstrap_input_array_has_at_least_one_duplicate(self, sample_array_md_type):
        res = bootstrap(sample_array_md_type, num_iter=100, resample_size=100, metrics={"mean": np.mean})
        counter = create_count_of_sampled_items(res)
        assert any([True if count > 1 else False for count in counter.values()])

        # checking if the mean lies around the true mean of the given 'sample' - this cannot be tested without fixing
        # the seed, which I do not want to in this implementation
        print(len(np.array(res)[[abs(metric['mean'] - 50) >= 5 for metric in res]]) < 15)
