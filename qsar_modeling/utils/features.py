import copy
import logging
import math

import numpy as np
import pandas as pd


# from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
# from statsmodels.stats.outliers_influence import variance_inflation_factor, cache_readonly
# import utils.distributions


def sort_ordinals(feature_list, start=0, stop=None, step=1):
    # Use negative steps for descending.

    def is_single(s):
        return all(str(n) not in s for n in range(10, 20))

    iter, max_iter = 0, 10
    remaining = copy.deepcopy(feature_list)
    sorted_list = list()
    i = start
    while len(feature_list) > 0:
        if i < 10:
            t = [
                f for f in feature_list if str(i) in f.lower() and is_single(f.lower())
            ]
        elif i >= 10:
            t = [f for f in feature_list if str(i) in f.lower()]
            [remaining.pop(f) for f in t]
            if len(t) > 1 or (len(t) == 1 and type(t) is list):
                sorted_list.extend(t)
            if len(t) == 1:
                sorted_list.append(t)
        i += step
        iter += 1
        if (step > 0 and i > stop) or (step < 0 and i < stop) or (iter > max_iter):
            break
    return sorted_list


def iterate_feature_pca(
    feature_df,
    new_feat,
    previous_subset,
    previous_pca=None,
    evr_thresh=(0.9,),
    delta_thresh=None,
    **pca_kwargs
):
    from sklearn.decomposition import PCA

    if previous_pca is None:
        previous_pca = PCA(**pca_kwargs).fit(feature_df[previous_subset])
    new_pca = PCA(**pca_kwargs).fit(feature_df[previous_subset.append(new_feat)])
    if evr_thresh is not None and evr_thresh[0] is not None:
        t_len = min(len(new_pca.explained_variance_ratio_), len(evr_thresh))
        if not all(
            [new_pca.explained_variance_ratio_[i] < evr_thresh[i] for i in t_len]
        ):
            return False, new_pca
    if delta_thresh is not None and delta_thresh[0] is not None:
        t_len = min(
            len(new_pca.explained_variance_ratio_),
            len(delta_thresh),
            len(new_pca.explained_variance_ratio_),
        )
        if not all(
            [
                (
                    1
                    - new_pca.explained_variance_ratio_[i]
                    / previous_pca.explained_variance_ratio_[i]
                    > delta_thresh[i]
                )
                for i in t_len
            ]
        ):
            return False, new_pca
    return True, new_pca


def thresholded_group_pca(
    feature_df, subset_list, smallest_size=3, evr_thresh=0.925, **pca_kwargs
):
    previous_pca = None
    pca_check, new_pca = iterate_feature_pca(
        feature_df,
        subset_list[smallest_size],
        subset_list[: smallest_size - 1],
        evr_thresh=(evr_thresh,),
        **pca_kwargs
    )
    while pca_check and new_pca.n_features_in_ < len(subset_list):
        previous_pca = new_pca
        pca_check, new_pca = iterate_feature_pca(
            feature_df,
            subset_list[new_pca.n_features_in_],
            subset_list[: new_pca.n_features_in_ + 1],
            previous_pca=previous_pca,
            **pca_kwargs
        )
    return previous_pca


def set_cov_matrix(df, sample_wts=None, *args, **kwargs):
    logging.info("Calculating covariance matrix...")
    if sample_wts is not None:
        freq_wts = np.round(sample_wts)
        cov_arr = np.cov(df, rowvar=False, ddof=1, fweights=freq_wts)
    else:
        cov_arr = np.cov(df, ddof=1, rowvar=False)
    cov_mat = pd.DataFrame(data=cov_arr, index=df.columns, columns=df.columns)
    logging.info("Covariance matrix of shape {} calculated.".format(cov_mat.shape))
    # if cov_mat.isna().astype(int).sum().sum() > 0:
    #    print('Covariance matrix contains invalid values.')
    return cov_mat


# WARNING: pyitlib's conditional_mutual_info may be unstable!
# Code take from: https://stackoverflow.com/questions/55402338/finding-conditional-mutual-information-from-3-discrete-variable


def gen_dict(x):
    dict_z = {}
    for key in x:
        dict_z[key] = dict_z.get(key, 0) + 1
    return dict_z


# I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
def entropy(a, b, c):
    # Calculates entropy for binary data.
    x = np.array([a, b, c]).T
    x = x[x[:, -1].argsort()]  # sorted by the last column
    w = x[:, -3]
    y = x[:, -2]
    z = x[:, -1]

    # dict_w = gen_dict(w)
    # dict_y = gen_dict(y)
    dict_z = gen_dict(z)
    list_z = [dict_z[i] for i in set(z)]
    p_z = np.array(list_z) / sum(list_z)
    pos = 0
    ent = 0
    for i in range(len(list_z)):
        w = x[pos : pos + list_z[i], -3]
        y = x[pos : pos + list_z[i], -2]
        z = x[pos : pos + list_z[i], -1]
        pos += list_z[i]
        list_wy = np.zeros((len(set(w)), len(set(y))), dtype=float, order="C")
        list_w = list(set(w))
        list_y = list(set(y))

        for j in range(len(w)):
            pos_w = list_w.index(w[j])
            pos_y = list_y.index(y[j])
            list_wy[pos_w, pos_y] += 1
            # print(pos_w)
            # print(pos_y)
        list_p = list_wy.flatten()
        list_p = np.array([k for k in list_p if k > 0]) / sum(list_p)
        ent_t = 0
        for j in list_p:
            ent_t += -j * math.log2(j)
        # print(ent_t)
        ent += p_z[i] * ent_t
    return ent


"""
X = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
Y = [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0]
Z = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]

a = drv.entropy_conditional(X, Z)
##print(a)
b = drv.entropy_conditional(Y, Z)
c = entropy(X, Y, Z)
p = a + b - c
print(p)
0.15834454415751043
"""


def weighted_entrop(data):
    df = data.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))
    k = 1.0 / math.log(df.shape[0])

    p = df / df.sum(axis=0)
    lnf = -np.log(p, where=df != 0) * p * k

    d = 1 - lnf.sum(axis=0)
    weighted_entropy = d / d.sum()

    weighted_entropy = pd.DataFrame(
        weighted_entropy, index=df.columns, columns=["weight"]
    )
    return weighted_entropy


# def joint_mutual_information(X, y):


def compute_gram(X, y, sample_weights):
    normalized_weights = sample_weights * (X.shape[0] / (sample_weights.sum()))
    X_centered = X - np.average(X, axis=0, weights=normalized_weights)
    X_scaled = X_centered * np.sqrt(normalized_weights)[:, np.newaxis]
    gram = np.dot(X_scaled.T, X_scaled)
    return gram, normalized_weights
