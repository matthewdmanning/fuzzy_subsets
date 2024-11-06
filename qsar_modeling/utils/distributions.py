import numpy as np
import pandas as pd


def _ser_is_discrete(input_array, unique_thresh=5):
    if np.unique(input_array).size > unique_thresh:
        return False
    # if np.all([validation.required_int_like(n, "input_array") for n in input_array]):
    #    return True
    else:
        return True


def is_discrete(features, unique_thresh=5, diff_thresh=0.001):
    if type(features) is pd.Series or (type(features) is np.ndarray and len(features.shape)):
        return _ser_is_discrete(features, unique_thresh)
    elif pd.DataFrame:
        return pd.Series([_ser_is_discrete(features[f], unique_thresh) for f in features.columns],
                         index=features.columns)
    elif np.ndarray:
        return np.array([_ser_is_discrete(features[:, f], unique_thresh) for f in range(features.shape[1])])


def is_low_cardinal(features, single_thresh=None):
    if type(features) is pd.DataFrame:
        sparse_dict = dict()
        for f in features.columns:
            sparse_dict[f] = is_low_cardinal(features[f], single_thresh)
        return pd.Series(sparse_dict)
    ucounts = features.value_counts(normalize=True)
    if not single_thresh:
        single_thresh = [0.8 / (i + 1) for i in range(8)]
        # approx 1/n - 0.2/n = 0.8/n
    for _ in range(min(ucounts.size, len(single_thresh))):
        single = all(u > s for u, s in zip(ucounts, single_thresh))
