import copy
import copy
import os
import pprint
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
from sklearn.pipeline import clone as clone_model
from sklearn.preprocessing import RobustScaler

from feature_selection.importance import logger
from utils.features import compute_gram
from utils.parallel_subsets import train_model_subsets


# from features import compute_gram


def calculate_vif(
    feature_df,
    model,
    subset=None,
    parallelize=False,
    sample_wts=None,
    generalized=False,
    verbose=0,
    **fit_kwargs
):
    predictors = list()
    if subset is None:
        for col in feature_df.columns:
            predictors.append((feature_df.columns.drop(col), col))
    else:
        if all(c in feature_df.columns for c in subset.columns):
            for col, target in subset.items():
                predictors.append((feature_df.columns.drop(subset.columns), col))
        else:
            for col, target in subset.items():
                predictors.append((feature_df.columns.drop(col), col))

    # print('\n\nVIF length: {}\n\n'.format(len(vif_dict.keys())))
    if parallelize:
        vif_models = train_model_subsets(
            feature_df,
            predictor_list=predictors,
            model=model,
            mem_dir=os.environ.get("JOBLIB_TMP"),
            sample_weights=sample_wts,
        )
    else:
        vif_models = [
            clone_model(model).fit(
                X=feature_df[p[0]], y=feature_df[p[1]], sample_weight=sample_wts
            )
            for p in predictors
        ]
    vif_dict = dict(
        [
            (
                p[1],
                model.score(
                    X=feature_df[p[0]], y=feature_df[p[1]], sample_weight=sample_wts
                ),
            )
            for p, model in zip(predictors, vif_models)
        ]
    )
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


def get_vif_model(model, n_jobs, uncentered=True):
    parallelize = False
    if type(model) is str:
        if "elastic" in model:
            vif_model = ElasticNetCV(
                l1_ratio=[0.15, 0.3, 0.5],
                tol=5e-3,
                n_alphas=33,
                n_jobs=n_jobs,
                random_state=0,
                selection="random",
                fit_intercept=uncentered,
            )
        elif "ridge" in model:
            vif_model = Ridge(
                solver="lsqr",
                max_iter=1000,
                fit_intercept=uncentered,
                tol=1e-03,
                random_state=0,
            )
            parallelize = True
        elif "sgd" in model:
            vif_model = SGDRegressor(
                loss="huber",
                penalty="elasticnet",
                max_iter=5000,
                alpha=0.0005,
                epsilon=0.0025,
                learning_rate="adaptive",
                l1_ratio=0.75,
                fit_intercept=uncentered,
                early_stopping=True,
            )
            parallelize = True
        elif "ransac" in model:
            from sklearn.linear_model import RANSACRegressor

            vif_model = RANSACRegressor()
        elif "hubb" in model:
            from sklearn.linear_model import HuberRegressor

            vif_model = HuberRegressor(
                max_iter=5000, tol=1e-03, fit_intercept=uncentered
            )
            parallelize = True
        else:
            vif_model = LinearRegression(n_jobs=n_jobs, fit_intercept=uncentered)
    else:
        vif_model = LinearRegression(n_jobs=n_jobs, fit_intercept=uncentered)
    return parallelize, vif_model


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
    threshold=10.0,
    model_name="ransac",
    model_size="auto",
    feat_wts="auto",
    min_feat_out=1,
    step_size=1,
    rounds=1000,
    sample_wts=None,
    n_jobs=-1,
    verbose=1,
    save_dir=None,
    **kwargs
):
    """
    Repeatedly calculates feature Variance Inflation Factors (VIF = 1 / (1-r^2)) where r^2 is the score of the linear regressor for each feature using the other features in the subset. Feature inclusion in a subset is stochastically chosen from feat_wts. This process is run rounds times. A features total score is a weighted ratio of percentage of model appearances where a model's VIF is above the cut threshold.
    Parameters
    ----------
    feature_df: pd.DataFrame
    Training data for feature selection.
    importance_ser: pd.Series
    Multiplier for final score (more is less multicollinear). If feat_wts not provided, weighting for selecting features for appearance (lower is more likely)
    threshold: float
    VIF threshold. Greatest step_size features in a subset have "Cuts" incremented by 1.
    feat_wts: pd.Series
    Probabilities for including features in VIF calculation subset. Values >= 0.
    model_name: sklearn.linear_model._base
    Callable for model scoring to calculate VIF. If not given, model's default score method is used.
    model_size: int
    Number of features used in model.
    rounds
    sample_wts
    n_jobs
    verbose
    step_size
    kwargs

    Returns
    -------

    """
    df = (
        RobustScaler(with_centering=True, quantile_range=(15, 85), unit_variance=True)
        .set_output(transform="pandas")
        .fit_transform(feature_df)
    )
    if model_size == "auto":
        model_size = -1
        while model_size < 10:
            model_size += os.cpu_count()
    if importance_ser is None:
        importance_ser = (
            pd.Series(
                1,
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
    vif_list, cut_score_list, vif_score_df_list, vif_stats_list, votes_list = (
        list(),
        list(),
        list(),
        list(),
        list(),
    )
    feat_score = dict()
    if all([type(rounds) is not t for t in [list, set, tuple]]):
        rounds = (int(np.ceil((df.shape[1] - min_feat_out) // step_size)), int(rounds))
    # Start with 0.5 to avoid division by 0 later. Chosen to agree with no-info Bayesian prior.
    votes_list.append(
        pd.DataFrame(
            np.zeros((importance_ser.index.size, 2), dtype=np.float32),
            index=importance_ser.index,
            columns=["Appearances", "Cuts"],
        ).add(0.5)
    )
    parallelize, vif_model = get_vif_model(model_name, n_jobs, uncentered=False)
    print(
        "Running {}rounds of {} batches of {}".format(rounds[0], rounds[1], model_size)
    )

    for i in list(range(rounds[0])):
        if df.shape[1] <= min_feat_out:
            break
        importance_ser = importance_ser[df.columns]
        if feat_wts == "uniform":
            feat_wts = importance_ser.replace(value=1)
        elif feat_wts == "auto":
            feat_wts = 0.5 - pd.Series(
                scipy.linalg.norm(df.corr(method="pearson").to_numpy()),
                index=df.columns,
            ).squeeze().sort_values(ascending=False)
        for j in list(range(rounds[1])):
            # print('Selected: {}'.format(len(select_set)))
            if len(feat_wts.shape) == 2:
                feats = importance_ser.sample(
                    weights=scipy.special.softmax(feat_wts, axis=1).sum(axis=0),
                    n=model_size,
                ).index
            else:
                feats = importance_ser.sample(
                    weights=scipy.special.softmax(feat_wts), n=model_size
                ).index
            # iterative_feats = vif_subset_iterative(feat_wts, importance_ser, model_size)
            votes_list[-1].loc[feats, "Appearances"].add(1)
            vif_ser = calculate_vif(
                feature_df=df,
                model=vif_model,
                subset=df[feats],
                parallelize=parallelize,
                sample_wts=sample_wts,
                verbose=verbose,
                **kwargs
            )
            vif_list.append(vif_ser)
            votes_list[-1].loc[
                vif_ser[vif_ser > threshold]
                .sort_values(ascending=False)
                .index[:step_size],
                "Cuts",
            ] += 1
        cut_score_list.append(
            importance_ser.multiply(
                votes_list[-1]["Cuts"][importance_ser.index].divide(
                    votes_list[-1]["Appearances"][importance_ser.index]
                )
            )
        )
        vif_score_df_list.append(
            pd.concat(
                vif_list,
                axis=1,
                keys=["VIF-{}_{}".format(i, j) for j in list(range(rounds[1]))],
            )
        )

        vif_stats_list.append(
            pd.DataFrame(
                data=pd.concat(
                    (
                        vif_score_df_list[-1].mean(axis=1),
                        vif_score_df_list[-1].std(axis=1),
                    ),
                    axis=1,
                )
            )
        )
        vif_stats_list[-1].columns = ["Mean", "StdDev"]
        vif_stats_list[-1].sort_values(by="Mean", ascending=False, inplace=True)
        if save_dir is not None:
            vif_score_df_list[-1].to_csv("{}vif_scores_{}.csv".format(save_dir, i))
            vif_stats_list[-1].to_csv("{}vif_stats_{}.csv".format(save_dir, i))
        print("VIF Stats")
        pprint.pp(vif_stats_list[-1].sort_values(by="StdDev", ascending=False))
        dropped_cols = dict(
            [
                (c, rounds[0] - i)
                for c in cut_score_list[-1].iloc[
                    : min(step_size, df.shape[1]) - min_feat_out
                ]
            ]
        )
        feat_score.update(dropped_cols)
        df.drop(columns=dropped_cols.keys(), inplace=True)
    feat_score_df = pd.DataFrame.from_dict(feat_score)
    return feat_score_df, cut_score_list, vif_score_df_list, vif_stats_list, votes_list


def vif_subset_iterative(feat_wts, importance_ser, cross_corr, model_size):
    selection_trials = np.ceil(np.sqrt(importance_ser.size))
    feats = list()
    while len(feats) < model_size:
        next_feats = importance_ser.sample(
            weights=scipy.special.softmax(feat_wts, axis=1).sum(axis=0),
            n=selection_trials,
        ).index
        next_feats.sample()
    return feats


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


def corr_sampler(
    feature_df, labels, cross_corr, n_feats, seed_feats, feat_weights=None
):

    def _cc_norm(feat_set):
        return np.linalg.norm(x=cross_corr[feat_set], axis=0)

    def _ser_norm(feat_set):
        return feat_weights[feat_set]

    subset_list = list(seed_feats)
    if feat_weights is None and (
        type(cross_corr) is pd.DataFrame or type(len(cross_corr.shape) == 2)
    ):
        prob_func = _cc_norm
    else:
        prob_func = _ser_norm
        """
    while len(subset_list) < n_feats:
        feature_df.drop(columns=subset_list).sample(n=1, weights=)
        pd.DataFrame.sample(weights=prob_func())
"""


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
