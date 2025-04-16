from math import atan

import numpy as np
import pandas as pd
import scipy
from scipy.stats import kendalltau


def calculate_kendalls_c(first_series, second_series):
    return kendalltau(
        first_series, second_series, nan_policy="raise", variant="c"
    ).statistic


def add_remove_probs(size, max, min):
    mid = (max + min) / 2
    # add_prob = atan((1 - size) / (1 - mid)) / pi
    add_prob = (size / max) ** 2
    remove_prob = 1 - atan((-size + max) / mid) / atan(max / min)
    # print("Add/Remove Probs")
    # print(size, add_prob, remove_prob)
    return add_prob, remove_prob


def zwangzig(scores_list, test_score, lamb, temp, k=1):
    assert isinstance(scores_list, (list, tuple, set, pd.Series))
    best_score = np.max(scores_list)
    delta_E = (best_score - test_score) / best_score
    if k * temp * lamb <= 0.0 or delta_E < 0:
        metric = 0.5
    else:
        metric = np.exp((-delta_E / (lamb * k * temp)))
        print("Zwangzig metric: {}".format(metric))
        print(delta_E, k, lamb, temp)
    return metric


def softmax(X, axis=0):
    if type(X) is pd.Series or type(X) is pd.DataFrame:
        ser = X.copy()
        eser = ser.map(np.exp)
        if type(X) is pd.Series:
            sm = eser / eser.sum()
        else:
            sm = eser / eser.sum(axis=axis)
    sm = np.exp(X) / np.sum(np.exp(X), axis=axis)
    return sm


def norm_df_by_trace(df):
    frame = df.copy()
    df_trace = [
        frame.iloc[i, j]
        for i, j in zip(np.arange(frame.shape[0]), np.arange(frame.shape[1]))
    ]
    trace_root = np.sqrt(df_trace)
    for i, col in enumerate(frame.columns):
        if trace_root[i] < 0.0001:
            continue
        frame[col] = frame[col] / trace_root[i]
        frame.iloc[i] = frame.iloc[i] = trace_root[i]
    return frame


def get_max_arr_indices(arr, threshold, tol=0, num_elements=None):
    # Returns the indices of the greatest [num_elements] elements in the array/DataFrame that are within [tol] of the maximum value.
    if type(arr) is np.ndarray:
        arr = pd.DataFrame(arr)
    sorted_stack = arr.stack().sort_values(ascending=False)
    indices = sorted_stack[
        sorted_stack >= max(threshold, sorted_stack.max()) - tol
    ].index
    if num_elements is not None:
        indices = indices[:num_elements]
    return indices


def scaled_softmax(values, lam=0.1, center=0):
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    zeroed = values - values.min() + center * values.median() / lam
    # print("Softmax centering: {}, {}".format(center * values.median() / lam))
    # print(zeroed)
    return scipy.special.softmax(zeroed)


def size_factor(set_size, select_params):
    return ((select_params["max_features_out"] - set_size)) / select_params[
        "min_features_out"
    ]


def complexity_penalty(complexity, alpha=1e-4):
    """
    Adds penalty based on number of parameters in a model.
    Parameters
    ----------
    complexity : float | int
    alpha : float, scaling parameter for adjustment, larger values increase penalty

    Returns
    -------
    adjust : float, scaling coefficient for model scores
    """
    penalty = 1  #  - alpha * complexity * np.asinh(complexity)
    if np.isnan(penalty):
        print("Null complexity")
        print("alpha: {}, complexity: {}".format(alpha, complexity))
    elif penalty < 0.0:
        print("alpha: {}, complexity: {}".format(alpha, complexity))
    return penalty
