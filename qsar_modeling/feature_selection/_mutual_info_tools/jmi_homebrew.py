# Joint MI Estimator = I(X1; y) + I(X2; y)
import itertools
import logging
import pprint
from functools import partial

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils import compute_class_weight

import qsar_modeling.utils.distributions

np.set_printoptions(formatter={"float": "{: 0.4f}".format})

logger = logging.getLogger(__name__)


def estimate_cardinality(
    feature_ser,
    labels,
    mss_list="auto",
    criterion="entropy",
    thresh=0.85,
    n_trees=100,
    iter_max=100,
):
    from sklearn.ensemble import ExtraTreesClassifier

    X = feature_ser.to_frame()
    if X.name is None:
        X.name = "target"
    clf_dict, output_dict = dict(), dict()
    thresh_met = False
    mss = 1 / np.log2(X.size)
    i = 0
    while not thresh_met and i < iter_max:
        # for mss in mss_list:
        x_clf = ExtraTreesClassifier(
            n_estimators=n_trees,
            criterion=criterion,
            bootstrap=False,
            class_weight="balanced",
            min_samples_split=mss,
            n_jobs=-1,
        ).fit(X, labels)
        scores = [c.score(X=X, y=labels) for c in x_clf.estimators_]
        n_leaves = [c.get_n_leaves() for c in x_clf.estimators_]
        x_df = pd.DataFrame(
            data=zip(scores, n_leaves), columns=["Score", "n_leaves"]
        ).sort_values(by="Score", ascending=False)
        # clf_dict[mss] = x_clf
        if x_df[x_df["Score"] >= thresh].shape[0] >= np.sqrt(n_trees):
            min_leaves_thresh = x_df[x_df["Score"] >= thresh]["n_leaves"].min()
            print(min_leaves_thresh)
            thresh_met = True
            return min_leaves_thresh
        i += 1
    print("Did not meet criteria within max number of iterations.")
    print(pprint.pformat(x_df))
    return None


def balanced_filter_univar(
    feature_df, labels, partial_func, cv=30, sampling_strategy="auto", n_jobs=-1
):
    filter_list = list()
    for _ in list(range(cv)):
        rus_X, rus_y = RandomUnderSampler(
            sampling_strategy=sampling_strategy
        ).fit_resample(feature_df, labels)
        filter_list.append(pd.Series(partial_func(rus_X, rus_y)).T)
    combined_ser = pd.concat(filter_list, axis=1).mean(axis=1).squeeze()
    combined_ser.index = feature_df.columns
    print(combined_ser.head(), flush=True)
    return combined_ser.sort_values(
        ascending=False
    )  # .rename_axis(index=feature_df.columns)


def rus_wrapper(
    func,
    feature_df,
    labels,
    cv=5,
    sampling_strategy="auto",
    random_state=0,
    *args,
    **kwargs
):
    rus_list = list()
    for _ in list(range(cv)):
        rus_X, rus_y = RandomUnderSampler(
            sampling_strategy=sampling_strategy, random_state=random_state
        ).fit_resample(feature_df, labels)
        rus_list.append(func(rus_X, rus_y, *args, **kwargs))
    return rus_list


def balanced_mi_y(
    feature_df,
    labels,
    cv=30,
    n_neighbors=7,
    estimator="classifier",
    sampling_strategy="auto",
    n_jobs=-1,
):
    # Yields k-fold randomly undersampled MI of shape (n_features, cv)
    if estimator == "classifier":
        mi_func = partial(
            mutual_info_classif,
            discrete_features="auto",
            n_neighbors=n_neighbors,
            random_state=0,
        )
    else:
        mi_func = partial(
            mutual_info_regression,
            discrete_features="auto",
            n_neighbors=n_neighbors,
            random_state=0,
        )
    return balanced_filter_univar(
        feature_df=feature_df, labels=labels, cv=cv, partial_func=mi_func
    )


def mi_mixed_types(
    feature_df, disc_dict=None, normalize_by_self=False, discrete_kws=None, **mi_kws
):
    mi_df = pd.DataFrame(
        data=np.zeros(shape=(feature_df.shape[1], feature_df.shape[1])),
        index=feature_df.columns,
        columns=feature_df.columns,
    )
    if disc_dict is None:
        disc_dict = dict()
        for f in feature_df.columns:
            disc_dict[f] = qsar_modeling.utils.distributions.is_discrete(feature_df[f])
    disc_ind = [i for i, (k, v) in enumerate(disc_dict.items()) if v]
    cont_ind = [i for i, (k, v) in enumerate(disc_dict.items()) if not v]
    print("Discrete: {}, Continuous: {}".format(len(disc_ind), len(cont_ind)))
    unique_pairs_dual = np.tril_indices(n=feature_df.shape[1], m=feature_df.shape[1])
    tril_ind_i, tril_ind_j = (
        unique_pairs_dual[0].tolist(),
        unique_pairs_dual[1].tolist(),
    )
    for i, r in enumerate(mi_df.index):
        mi_df.iloc[i, i:] = mutual_info_regression(
            feature_df.iloc[:, i:], feature_df.iloc[:, i], **mi_kws
        )
    """
    for i, j in zip(tril_ind_i, tril_ind_j):
        if i in cont_ind:
            func = mutual_info_regression
        elif i in disc_ind:
            func = mutual_info_classif
        else:
            logger.error(
                "Index {} was not found in either cont_ind or disc_ind lists!!!".format(
                    i
                )
            )
            raise IndexError
        if type(j) is np.ndarray:
            cont_tril_ind = [f for f in j if f in cont_ind]
            disc_tril_ind = [f for f in j if f not in cont_ind]
            print(cont_tril_ind, disc_tril_ind)
            if len(cont_tril_ind) > 0:
                cont_feats = feature_df[feature_df.columns[cont_tril_ind]]
                mi_vec = func(
                    cont_feats,
                    feature_df[feature_df.columns[i]],
                    discrete_features=False,
                    **mi_kws
                )
                mi_df.iloc[i, cont_tril_ind] = mi_vec
                print("MI FxF Vector: {}".format(mi_vec.shape))
            if len(disc_tril_ind) > 0:
                disc_feats = feature_df[feature_df.columns[disc_tril_ind]]
                mi_vec = func(
                    disc_feats,
                    feature_df[feature_df.columns[i]],
                    discrete_features=True,
                    **mi_kws
                )
                if all([m == 0 for m in mi_vec]):
                    print("All discrete vector returned as all zeros!!!")
                mi_df.iloc[i, disc_tril_ind] = mi_vec
                print("MI FxF Vector: {}".format(mi_vec.shape))
        elif j in cont_ind:
            cont_feats = feature_df[feature_df.columns[j]].to_frame()
            mi_df.iloc[i, j] = func(
                cont_feats,
                feature_df[feature_df.columns[i]],
                discrete_features=False,
                **mi_kws
            )
        elif j in disc_ind:
            disc_feats = feature_df[feature_df.columns[j]].to_frame()
            mi_df.iloc[i, j] = func(
                disc_feats,
                feature_df[feature_df.columns[i]],
                discrete_features=True,
                **mi_kws
            )
        else:
            print(
                "{} is neither ndarray nor in index list!".format(
                    unique_pairs_dual[0][i]
                )
            )
    """
    for i, j in zip(tril_ind_i, tril_ind_j):
        mi_df.iloc[[j, i]] = mi_df.iloc[[i, j]]
    if normalize_by_self:
        print("This method produces MI values that are greater than 1.")
        raise NotImplementedError
        mi_df.divide(
            np.sqrt(mi_df.iloc[np.diag_indices_from(mi_df.to_numpy())]), axis=0
        )
        mi_df.divide(
            np.sqrt(mi_df.iloc[np.diag_indices_from(mi_df.to_numpy())]), axis=1
        )
    return mi_df


def condition_by_label(feature_df, labels, disc_dict=None):
    # Class weights are the inverse of their probability, normalized to 1, for default 'balanced' setting.
    label_df_list = list()
    weights = compute_class_weight(
        class_weight="balanced", classes=labels.unique(), y=labels
    )
    p_y = weights / np.sum(weights)
    if disc_dict is None:
        disc_dict = dict()
        for f in feature_df.columns:
            disc_dict[f] = qsar_modeling.utils.distributions.is_discrete(feature_df[f])
    for label_wt, label_val in zip(p_y, labels.unique()):
        label_ind = labels[labels == label_val].index
        label_df = mi_mixed_types(feature_df.loc[label_ind], disc_dict=disc_dict)
        label_df_list.append(label_df)
    return label_df_list
    """
    neg_ind = labels[labels == 0].index
    pos_ind = labels[labels == 1].index
    # neg_df = pd.DataFrame(data=np.zeros(shape=(feature_df.shape[1], feature_df.shape[1]), dtype=np.float32), index=feature_df.columns, columns=feature_df.columns)
    # pos_df = pd.DataFrame(data=np.zeros(shape=(feature_df.shape[1], feature_df.shape[1]), dtype=np.float32), index=feature_df.columns, columns=feature_df.columns)
    # for i, j in itertools.combinations(feature_df.columns, r=2):
    neg_df = p_y[0] * mi_mixed_types(feature_df.loc[neg_ind], disc_dict=disc_dict)
    pos_df = p_y[1] * mi_mixed_types(feature_df.loc[pos_ind], disc_dict=disc_dict)
    """


def bivariate_conditional(
    feature_df,
    labels,
    x_y_mi=None,
    conditional_dfs=None,
    disc_dict=None,
    lower_triangle=True,
):
    bivariate_measure = pd.DataFrame(
        np.zeros(shape=(feature_df.shape[1], feature_df.shape[1])),
        index=feature_df.columns,
        columns=feature_df.columns,
    )
    if (x_y_mi is None or conditional_dfs is None) and disc_dict is None:
        disc_dict = dict()
        for f in feature_df.columns:
            disc_dict[f] = qsar_modeling.utils.distributions.is_discrete(feature_df[f])
    if conditional_dfs is None:
        conditional_dfs = condition_by_label(feature_df, labels, disc_dict=disc_dict)
    if x_y_mi is None:
        x_y_mi = mi_mixed_types(feature_df, disc_dict=disc_dict)
    for i, j in itertools.combinations(list(range(feature_df.shape[1])), r=2):
        print(i, j)
        biv_measure = (
            x_y_mi.iloc[i]
            + x_y_mi.iloc[j]
            - 2 * (conditional_dfs[0].iloc[i, j] - conditional_dfs[1].iloc[i, j])
        )
        print(x_y_mi.iloc[i])
        print(x_y_mi.iloc[j])
        print(conditional_dfs[0].iloc[i, j])
        print(biv_measure, flush=True)
        bivariate_measure.iloc[i, j] = biv_measure
        if not lower_triangle:
            bivariate_measure.iloc[j, i] = bivariate_measure.iloc[i, j]
    return bivariate_measure
