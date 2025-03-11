import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def calculate_kendalls_c(first_series, second_series):
    return kendalltau(
        first_series, second_series, nan_policy="raise", variant="c"
    ).statistic


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
