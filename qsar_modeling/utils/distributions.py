import itertools
from collections import OrderedDict

import numpy as np
import pandas as pd
from statsmodels.tools import validation


def _ser_is_discrete(input_array, unique_thresh=5):
    if np.unique(input_array).size > unique_thresh:
        return False
    # if np.all([validation.required_int_like(n, "input_array") for n in input_array]):
    #    return True
    else:
        return True


def is_discrete(features, unique_thresh=5, diff_thresh=0.001):
    if type(features) is pd.Series or (
        type(features) is np.ndarray and len(features.shape)
    ):
        return _ser_is_discrete(features, unique_thresh)
    elif pd.DataFrame:
        return pd.Series(
            [_ser_is_discrete(features[f], unique_thresh) for f in features.columns],
            index=features.columns,
        )
    elif np.ndarray:
        return np.array(
            [
                _ser_is_discrete(features[:, f], unique_thresh)
                for f in range(features.shape[1])
            ]
        )


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


def ks_feature_tests(feature_dfs, dist_stat="ks"):
    assert all(
        [
            (a.shape[1] == b.shape[1] and a.shape[1] > 0)
            for a, b in itertools.combinations(feature_dfs, r=2)
        ]
    )
    assert all([a.shape[0] > 10 for a in feature_dfs])
    print([a.shape for a in feature_dfs])
    if type(dist_stat) is str:
        if dist_stat == "ks":
            from scipy.stats import ks_2samp

            dist_stat = ks_2samp
        elif dist_stat == "js":
            from scipy.spatial.distance import jensenshannon

            dist_stat = jensenshannon
    ks_dict = OrderedDict([(c, list()) for c in feature_dfs[0].columns])
    mean_dict = dict()
    for c in feature_dfs[0].columns:
        # if dist_stat == "ks":
        ks_results = ks_2samp(
            feature_dfs[0][c].to_numpy(), feature_dfs[0][c].to_numpy()
        ).statistic
        print(ks_results)
        mean_dict[c] = ks_results
    stat_means = pd.Series(mean_dict).sort_values()
    return stat_means
