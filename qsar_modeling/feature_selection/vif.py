import copy
import pprint
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
import scipy
import sklearn.utils
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.preprocessing import RobustScaler

from feature_selection.importance import logger
from utils.features import compute_gram


# from features import compute_gram


def calculate_vif(
    feature_df,
    subset=None,
    model="sgd",
    sample_wts=None,
    generalized=False,
    scorer=None,
    n_jobs=-1,
    verbose=0,
    vif_dict=None,
    **fit_kwargs
):
    if vif_dict is None:
        vif_dict = OrderedDict()
    if type(model) is str:
        if "elastic" in model:
            vif_model = ElasticNetCV(
                l1_ratio=[0.15, 0.3, 0.5],
                tol=5e-3,
                n_alphas=33,
                n_jobs=n_jobs,
                random_state=0,
                selection="random",
            )
        elif "ridge" in model:
            vif_model = Ridge(solver="lsqr", max_iter=1000, tol=1e-03, random_state=0)
        elif "sgd" in model:
            vif_model = SGDRegressor(
                loss="huber",
                penalty="elasticnet",
                max_iter=5000,
                alpha=0.0005,
                epsilon=0.0025,
                learning_rate="adaptive",
                l1_ratio=0.75,
                fit_intercept=False,
                early_stopping=True,
            )
        elif "hubb" in model:
            from sklearn.linear_model import HuberRegressor

            vif_model = HuberRegressor(max_iter=1000, tol=1e-04, fit_intercept=False)
        else:
            vif_model = LinearRegression(n_jobs=n_jobs)
    else:
        vif_model = LinearRegression(n_jobs=n_jobs)

    if subset is None:
        for col in feature_df.columns:
            y = feature_df[col]
            training = feature_df.drop(columns=col)
            vif_dict[col] = single_vif(
                training,
                y,
                col,
                generalized,
                sample_wts,
                sklearn.clone(vif_model),
                scorer=scorer,
            )
    else:
        if all(c in feature_df.columns for c in subset.columns):
            for col, target in subset.items():
                vif_dict[col] = single_vif(
                    feature_df.drop(columns=subset.columns),
                    y=target,
                    col_name=col,
                    generalized=generalized,
                    sample_wts=sample_wts,
                    vif_model=sklearn.clone(vif_model),
                    scorer=scorer,
                )
        else:
            for col, target in subset.items():
                vif_dict[col] = single_vif(
                    feature_df.drop(columns=col),
                    target,
                    col,
                    generalized,
                    sample_wts,
                    sklearn.clone(vif_model),
                    scorer=scorer,
                )
    # print('\n\nVIF length: {}\n\n'.format(len(vif_dict.keys())))
    if generalized:
        vif_ser = pd.DataFrame.from_dict(
            vif_dict, orient="index", columns=["VIF", "GVIF"]
        )
    else:
        vif_ser = pd.Series(vif_dict)
    # if verbose > 0:
    logger.info(pprint.pformat(vif_ser.sort_values(ascending=False), compact=True))
    assert not vif_ser.empty
    return vif_ser


def single_vif(
    training, y, col_name, generalized, vif_model, sample_wts=None, scorer=None
):
    new_model = sklearn.clone(vif_model)
    new_model.fit(X=training, y=y, sample_weight=sample_wts)
    if scorer is None:
        r_squared = new_model.score(X=training, y=y, sample_weight=sample_wts)
    else:
        r_squared = scorer(new_model)
    if r_squared != 1.0:
        vif_i = 1.0 / (1.0 - r_squared)
    else:
        logger.error(
            "Feature had OOB OLS R2 score of 1.0!!!! Feature name: {}".format(col_name)
        )
        vif_i = 1000000
    # print('Original VIF: {}'.format(vif_i))
    if generalized:
        vif_score = (vif_i, np.log(vif_i) / np.sqrt(2 * (training.shape[1])))
        # print('Generalized VIF: {}'.format(vif_dict[col]))
    else:
        vif_score = vif_i
    return vif_score


def repeated_stochastic_vif(
    feature_df,
    importance_ser=None,
    cut=10.0,
    feat_wts="auto",
    scoring="sgd",
    model_size=25,
    rounds=1000,
    sample_wts=None,
    n_jobs=-1,
    verbose=1,
    step_size=1,
    **kwargs
):
    df = (
        RobustScaler(with_centering=False, quantile_range=(15, 85), unit_variance=True)
        .set_output(transform="pandas")
        .fit_transform(feature_df)
    )
    if importance_ser is None:
        importance_ser = (
            pd.Series(
                scipy.linalg.norm(feature_df.corr(method="pearson").to_numpy()),
                index=feature_df.columns,
            )
            .squeeze()
            .sort_values(ascending=False)
        )
    if feat_wts == "uniform":
        feat_wts = importance_ser.replace(value=1)
    elif feat_wts == "auto":
        feat_wts = (
            pd.Series(
                scipy.linalg.norm(feature_df.corr(method="pearson").to_numpy()),
                index=feature_df.columns,
            )
            .squeeze()
            .sort_values(ascending=False)
        )
    print(
        "Number of features for CV Stochastic VIF selections: {}".format(
            importance_ser.size
        )
    )
    print("Running {} batches of {}".format(rounds, model_size))
    vif_list = list()
    # Start with 0.5 to avoid division by 0 later. Chosen to agree with no-info Bayesian prior.
    votes = pd.DataFrame(
        np.zeros((importance_ser.index.size, 2), dtype=np.float64),
        index=importance_ser.index,
        columns=["Appearances", "Cuts"],
    ).add(0.5)
    for i in list(range(rounds)):
        # print('Selected: {}'.format(len(select_set)))
        feats = importance_ser.sample(
            weights=scipy.special.softmax(feat_wts), n=model_size
        ).index
        votes.loc[feats, "Appearances"].add(1)
        vif_ser = calculate_vif(
            feature_df=df,
            subset=df[feats],
            model=scoring,
            sample_wts=sample_wts,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs
        )
        vif_list.append(vif_ser)
        votes.loc[
            vif_ser[vif_ser > cut].sort_values(ascending=False).index[:step_size],
            "Cuts",
        ] += 1
    cut_score = importance_ser.multiply(
        1
        - votes["Cuts"][importance_ser.index].divide(
            votes["Appearances"][importance_ser.index]
        )
    ).sort_values(ascending=False)
    print(
        "Weighted Cut Score: {}".format(
            pprint.pformat(cut_score.head(n=10), compact=True)
        )
    )
    vif_score_df = pd.concat(
        vif_list, axis=1, keys=["VIF_{}".format(i) for i in list(range(rounds))]
    )
    vif_stats = pd.DataFrame(
        data=pd.concat((vif_score_df.mean(axis=1), vif_score_df.std(axis=1)), axis=1)
    )
    vif_stats.columns = ["Mean", "StdDev"]
    vif_stats.sort_values(by="Mean", ascending=False, inplace=True)
    print("VIF scores")
    pprint.pp(vif_stats.sort_values(by="StdDev", ascending=False))
    return cut_score, vif_score_df, vif_stats, votes


def sequential_vif(
    feature_df,
    vif_cut=10,
    n_keep=1,
    step_size=1,
    generalized=False,
    scorer=None,
    model=None,
    **kwargs
):
    # Inputs: Features, Target correlations: Series, Covariance, correlation with target, VIF/Condition Num.
    # https://www.sciencedirect.com/science/article/abs/pii/S1386142521012294
    if model is None:
        model = "ols"
    feats_df = (
        RobustScaler(unit_variance=True)
        .set_output(transform="pandas")
        .fit_transform(feature_df)
    )
    vif_call = partial(
        calculate_vif, model=model, generalized=generalized, scorer=scorer, **kwargs
    )
    cut_list = list()
    while feats_df.shape[1] > n_keep:
        vif_ser = vif_call(feats_df)
        if generalized:
            vif_ser = vif_ser.drop(columns="VIF").squeeze()
        cuts = pick_vifs(
            feats_df, vif_ser, vif_call, step_size=step_size, vif_cut=vif_cut, **kwargs
        )
        if cuts is not None and feats_df.shape[1] - cuts.size >= n_keep:
            print("Number of cuts in this VIF step: {}".format(cuts.size))
            pprint.pp(cuts)
            cut_list.append(cuts)
            feats_df.drop(columns=cuts.index, inplace=True)
        elif feats_df.shape[1] - cuts.size >= n_keep:
            cuts = cuts.sort_values(ascending=False).index[: n_keep - feats_df.shape[1]]
            print("Number of cuts in this VIF step: {}".format(cuts.size))
            pprint.pp(cuts)
            cut_list.append(cuts)
            feats_df.drop(columns=cuts.index, inplace=True)
        else:
            print("VIF selection terminated at {} features.".format(feats_df.shape[1]))
            pprint.pp(vif_ser)
            break
    if len(cut_list) > 0:
        cut_ser = pd.concat(cut_list)
    else:
        print("No cuts in the cut list!!!")
        cut_ser = feature_df.columns.symmetric_difference(feats_df.columns).to_series()
    return cut_ser


def pick_vifs(feature_df, vif_ser, vif_call, step_size, vif_cut=10, **kwargs):
    over_thresh = vif_ser[vif_ser > vif_cut]
    if over_thresh.size > 0:
        thresh_cut = over_thresh.sort_values(ascending=False).iloc[:step_size].min()
        pprint.pp(thresh_cut)
        cuts = over_thresh[over_thresh >= thresh_cut]
        min_vifs = over_thresh[over_thresh == thresh_cut]
        max_vifs = over_thresh[over_thresh > thresh_cut]
        if min_vifs.size > 1:
            print("VIF tie of size: {}".format(min_vifs.size))
            pprint.pp(min_vifs)
            over_cut = max(cuts.size - step_size, 0)
            over_vif = vif_call(
                feature_df.drop(columns=max_vifs.index),
                subset=min_vifs,
                step_size=over_cut,
                **kwargs
            )
            cuts.drop(min_vifs, inplace=True)
            new_vifs = pick_vifs(
                feature_df, over_vif, vif_call, step_size=over_cut, vif_cut=None
            )
            cuts = pd.concat([cuts, new_vifs])
        else:
            print("Cutting VIFs...")
            pprint.pp(over_thresh.sort_values(ascending=False).head())
            cuts = over_thresh
    else:
        print("No VIFs over the cut-off.")
        pprint.pp(vif_ser.sort_values(ascending=False).head())
        cuts = None
    return cuts


def vif_bad_apples(
    feature_df,
    target,
    model="ols",
    sample_wts=None,
    class_wts=None,
    generalized=False,
    n_jobs=1,
    verbose=0,
    kernel="laplacian",
):
    # In progress function to remove features causing multicollinearity with several other functions.
    coefs_list, r_two_list = list(), list()
    if model == "lasso":
        gram, normed_weights = compute_gram(
            feature_df, target, sample_weights=sample_wts
        )
        reg = LassoCV(
            n_jobs=n_jobs, tol=5e-2, max_iter=5000, random_state=0, selection="random"
        )
        print(reg.coef_)
    weak_list = list()
    y = feature_df[target]
    while True:
        training = feature_df.drop(columns=target)
        if model == "ols":
            reg = LinearRegression(n_jobs=n_jobs).fit(
                X=training, y=y, sample_weight=sample_wts
            )
            coefs = pd.Series(data=reg.coef_, index=training.index)
            weak_one = coefs.idxmin()
            r_squared = reg.score(X=training, y=y, sample_weight=sample_wts)
            if r_squared != 1.0:
                vif_i = 1.0 / (1.0 - r_squared)
            else:
                logger.error(
                    "Feature had OOB OLS R2 score of 0!!!! Feature name: {}".format(
                        target
                    )
                )
                vif_i = 1000000


def stochastic_vif(
    feature_df,
    importance_ser,
    num_select=100,
    vif_cut=10,
    scoring="logistic",
    sample_wts=None,
    feat_wts="auto",
    chunk_size=3,
    n_jobs=-1,
    scale_data=True,
    verbose=1,
):
    if scale_data:
        df = (
            RobustScaler(
                with_centering=False, quantile_range=(15, 85), unit_variance=True
            )
            .set_output(transform="pandas")
            .fit_transform(feature_df)
        )
    else:
        df = feature_df.copy()
    logger.info("Number of features for VIF selections: {}".format(importance_ser.size))
    alpha = 0.3
    corr_ser = importance_ser.copy()
    select_set = list()
    if feat_wts == "auto":
        feat_wts = corr_ser
    if num_select is None:
        num_select = 1
    chunk = chunk_size
    cut_list = list()
    while len(select_set) < num_select and corr_ser.shape[0] > chunk:
        # print('Selected: {}'.format(len(select_set)))
        feats = corr_ser.sample(
            weights=feat_wts[corr_ser.index], n=chunk_size
        ).index.tolist()
        corr_ser.drop(feats)
        test_feats = copy.deepcopy(feats)
        test_feats.extend(select_set)
        # print(test_feats)
        if test_feats is None or len(test_feats) == 0:
            logger.warning("WARNING!!! No features left!")
            break
        vif_ser = calculate_vif(
            df[select_set],
            subset=feats,
            sample_wts=sample_wts,
            scorer=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        # probs = max((alpha * vif / 100.), 1.0)
        # for np.random.(n=1, p=0.5)
        # cuts = vif_ser[(vif_ser.iloc[:, 1] > alpha) | (vif_ser.iloc[:, 0] > vif_cut)]
        cuts = vif_ser[(vif_ser.iloc[:, 1] * (vif_ser.iloc[:, 0]) > vif_cut)]
        logger.info(pprint.pformat(cuts, compact=True))
        if not cuts.empty:
            cut_list.append((tuple(cuts.index.tolist()), vif_ser))
            for c in cuts.index.to_list():
                if c not in test_feats:
                    select_set.remove(c)
        [select_set.append(f) for f in test_feats if f not in cuts.index]
        chunk = min((chunk_size, corr_ser.shape[0]))
        if verbose > 1:
            logger.info(
                pprint.pformat(
                    vif_ser.sort_values(by="VIF", ascending=False), compact=True
                )
            )
    if verbose > 0:
        # pprint.pp(vif_ser.sort_values(ascending=False), compact=True, width=80)
        # print('Features cut by VIF:')
        # [print('{:.50s}'.format(x[0])) for x in cut_list]
        logger.info("Top 25 selected features")
        logger.info(
            pprint.pformat(vif_ser.sort_values(by="VIF", ascending=False).iloc[:25])
        )
    return vif_ser, select_set, cut_list
