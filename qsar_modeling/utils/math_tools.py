import numpy as np
import pandas as pd


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


def get_max_arr_indices(arr, num_elements=None):
    # Returns the indices of the greatest [num_element] elements in the array/DataFrame.
    if type(arr) is np.ndarray:
        arr = pd.DataFrame(arr)
    if num_elements is None or num_elements <= 0:
        indices = arr.stack().sort_values(ascending=False).index
    else:
        indices = arr.stack().nlargest(num_elements, keep=int(num_elements)).index
    return indices
