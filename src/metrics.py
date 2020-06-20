import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

__author__ = "Sanja Stegerer"


def precision_calc(resample: np.matrix):
    """
    Calculate precision score from matrix. Assume the first column to be y_true and the second column to be y_pred
    Args:
        resample: matrix with 2 columns [y_true, y_pred]

    Returns:
        precision score: float
    """
    return precision_score(resample.tolist()[0], resample.tolist()[1])


def f1_calc(resample: np.matrix):
    """
    Calculate f1 score from matrix. Assume the first column to be y_true and the second column to be y_pred
    Args:
        resample: matrix with 2 columns [y_true, y_pred]

    Returns:
        f1 score: float
    """
    return f1_score(resample.tolist()[0], resample.tolist()[1])


def recall_calc(resample: np.matrix):
    """
    Calculate recall score from matrix. Assume the first column to be y_true and the second column to be y_pred
    Args:
        resample: matrix with 2 columns [y_true, y_pred]

    Returns:
        recall score: float
    """
    return recall_score(resample.tolist()[0], resample.tolist()[1])
