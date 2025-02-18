import copy
import itertools
import os
import pprint
from functools import partial

import numpy as np
import pandas as pd
import scipy
import sklearn.utils
from scipy.stats import bayes_mvs
from sklearn.linear_model import (
    ElasticNetCV,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.pipeline import clone as clone_model
from sklearn.preprocessing import RobustScaler

from feature_selection.importance import logger
from utils.parallel_subsets import train_model_subsets


def calculate_vif(
    feature_df,
    model,
    subset=None,
    parallelize=True,
    sample_wts=None,
    generalized=False,
    verbose=0,
    **fit_kwargs
):
    """
    Returns a Series containing the Variance Inflation Factor with each of "subset" as the dependent variable.
    The index is the feature_df index of the dependent variable

    Parameters
    ----------
    feature_df
    model
    subset
    parallelize
    sample_wts
    generalized
    verbose
    fit_kwargs

    Returns
    -------

    """
    predictor_list = list()
    if subset is None:
        for col in feature_df.columns:
            predictor_list.append((feature_df.columns.drop(col).tolist(), col))
    else:
        for col, target in subset.items():
            predictor_list.append((feature_df.columns.drop(col).tolist(), col))
    if parallelize and len(predictor_list[0]) > 1:
        with sklearn.config_context(enable_metadata_routing=True):
            vif_models = train_model_subsets(
                feature_df,
                predictor_list=predictor_list,
                model=model,
                mem_dir=os.environ.get("JOBLIB_TMP"),
                sample_weights=sample_wts,
            )
    elif not parallelize and len(predictor_list[0]) > 1:
        vif_models = [
            clone_model(model).fit(
                X=feature_df[p[0]], y=feature_df[p[1]], sample_weight=sample_wts
            )
            for p in predictor_list
            if len(p[0]) > 0 and len(p[1]) > 0
        ]
    else:
        print(predictor_list)
        raise ValueError
    try:
        vif_dict = dict(
            [
                (
                    p[-1],
                    1 / (1 - model.score(X=feature_df[p[0]], y=feature_df[p[1]])),
                )
                for p, model in zip(predictor_list, vif_models)
            ]
        )
    except:
        vif_dict = dict([(tuple(p[-1]), 99999) for p in predictor_list])
    if generalized:
        vif_ser = pd.DataFrame.from_dict(
            vif_dict, orient="index", columns=["VIF", "GVIF"]
        )
    else:
        vif_ser = pd.Series(vif_dict)
    # if verbose > 0:
    # logger.info(pprint.pformat(vif_ser.sort_values(ascending=False), compact=True))
    assert not vif_ser.empty
    return vif_ser


def get_vif_model(model, n_jobs, uncentered=False):
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
    model_name="OLS",
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
    df = feature_df.copy()
    df = df[~df.index.duplicated(keep="first")]
    rsvif_results = dict(
        list(
            zip(
                [
                    "vif_list",
                    "cut_score_list",
                    "vif_score_df_list",
                    "vif_stats_list",
                    "votes_list",
                    "feat_score",
                ],
                ([list(), list(), list(), list(), list(), dict()]),
            )
        )
    )
    if model_size == "auto":
        model_size = 10
    model_size = min(model_size, feature_df.columns.size - 2)
    if importance_ser is None:
        importance_ser = pd.Series(
            1,
            index=feature_df.columns,
        ).squeeze()
    else:
        importance_ser = importance_ser[df.columns.intersection(importance_ser.index)]
        importance_ser = importance_ser[~importance_ser.index.duplicated(keep="first")]

    if all([type(rounds) is not t for t in [list, set, tuple]]):
        rounds = (int(np.ceil((df.shape[1] - min_feat_out) // step_size)), int(rounds))
    # Start with 0.5 to avoid division by 0 later. Chosen to agree with no-info Bayesian prior.
    rsvif_results["votes_list"].append(
        pd.DataFrame(
            np.zeros((importance_ser.index.size, 2), dtype=np.float32),
            index=importance_ser.index,
            columns=["Appearances", "Cuts"],
        ).add(0.5)
    )
    parallelize, vif_model = get_vif_model(model_name, n_jobs, uncentered=False)
    if verbose:
        print(
            "Running {} rounds of {} models with {} of {} features".format(
                rounds[0], rounds[1], model_size, feature_df.shape[1]
            )
        )
    if feat_wts is None or (type(feat_wts) is str and feat_wts == "auto"):
        corr = df.corr(method="pearson")
        corr.dropna(how="all", axis=0).dropna(how="all", axis=1, inplace=True)
        fweights = 0.5 - pd.Series(
            scipy.linalg.norm(corr, axis=1), index=corr.columns
        ).squeeze().sort_values(ascending=False)
    elif type(feat_wts) is pd.Series:
        fweights = feat_wts
    elif importance_ser is not None and type(importance_ser) is pd.Series:
        fweights = pd.Series(data=1, index=importance_ser.index)
    else:
        fweights = pd.Series(data=1, index=df.index)
    if df.shape[1] <= min_feat_out:
        i = rounds[0]
    df = df.T[~df.columns.duplicated()].T
    importance_ser = importance_ser[~importance_ser.index.duplicated()]
    assert not any(df.columns.duplicated())
    assert not any(importance_ser.index.duplicated())
    fweights = fweights[df.columns]
    prior_groups_ix = list()
    for i in list(range(1)):
        round_fwts, round_imp_wts = copy.deepcopy(fweights), copy.deepcopy(
            importance_ser
        )
        round_imp_wts = round_imp_wts[~round_imp_wts.index.duplicated()]
        for j in list(range(rounds[1])):
            feats = get_features_weighted(
                round_fwts.copy(),
                round_imp_wts.copy(),
                model_size,
                prior_groups=prior_groups_ix,
            )
            # if all([type(f) is str for f in feats]):
            prior_groups_ix.append(feats)
            rsvif_results["votes_list"][-1].loc[feats, "Appearances"].add(1)
            if len(feats) > 3:
                subset = df[feats].copy()
            else:
                subset = None
            vif_ser = calculate_vif(
                feature_df=df.copy(),
                model=vif_model,
                subset=subset,
                parallelize=parallelize,
                sample_wts=sample_wts,
                verbose=verbose,
                **kwargs
            )
            rsvif_results["vif_list"].append(vif_ser)
            overthresh = vif_ser[vif_ser > threshold].sort_values(ascending=False)
            if overthresh.size > 0:
                rsvif_results["votes_list"][-1].loc[
                    overthresh.index[0],
                    "Cuts",
                ] += 1
            else:
                continue
        vif_combined = dict(
            [
                (c, [d[c] for d in rsvif_results["vif_list"] if c in d.index])
                for c in feats
            ]
        )
        vif_means = pd.Series(dict([(k, np.max(v)) for k, v in vif_combined.items()]))
        vif_stds = pd.Series(dict([(k, np.std(v)) for k, v in vif_combined.items()]))
        rsvif_results["vif_stats_list"].append(pd.concat([vif_means, vif_stds], axis=1))
        if save_dir is not None:
            rsvif_results["vif_score_df_list"][-1].to_csv(
                "{}vif_scores_{}.csv".format(save_dir, i)
            )
            rsvif_results["vif_stats_list"][-1].to_csv(
                "{}vif_stats_{}.csv".format(save_dir, i)
            )
        """        
        rsvif_results["vif_score_df_list"].append(
            pd.concat(
                rsvif_results["vif_list"],
                axis=1,
                keys=["VIF-{}_{}".format(i, j) for j in list(range(rounds[1]))],
            )
        )
        rsvif_results["vif_stats_list"].append(
            pd.DataFrame(
                data=pd.concat(
                    (
                        rsvif_results["vif_score_df_list"][-1].mean(axis=1),
                        rsvif_results["vif_score_df_list"][-1].std(axis=1),
                    ),
                    axis=1,
                )
            )
        )
        """
        rsvif_results["vif_stats_list"][-1].columns = ["Max", "SD"]
        rsvif_results["vif_stats_list"][-1].sort_values(
            by="Max", ascending=False, inplace=True
        )

        # print("VIF Stats")
        # pprint.pp(rsvif_results["vif_stats_list"][-1].sort_values(by="StdDev", ascending=False))
        dropped_cols = dict(
            [
                (c, rounds[1] - j)
                for c in rsvif_results["votes_list"][-1].index[
                    : min(step_size, df.shape[1] - min_feat_out)
                ]
            ]
        )
        rsvif_results["feat_score"].update(dropped_cols)
        # Reactive for repeated VIF
        # df.drop(columns=dropped_cols.keys(), inplace=True)
    try:
        rsvif_results["feat_scores"] = pd.Series(rsvif_results["feat_score"])
    except ValueError:
        print("\nError with feature scores: {}".format(rsvif_results["feat_score"]))
        rsvif_results["feat_scores"] = pd.Series()
    rsvif_results.pop("feat_score")
    return rsvif_results


def posterior_vif_cut(results_dict, k_highest=1):
    raise NotImplementedError
    vifs_df = pd.concat(results_dict["vif_list"])
    print(vifs_df.head())
    bayes_ser = pd.DataFrame(columns=["Mean", "Lower", "Upper"]).sort_values(
        by="Mean", ascending=False
    )
    for col, data in vifs_df.iterrows():
        baye = bayes_mvs(data)[0]
    return None


def get_features_weighted(fweights, importance_ser, model_size, prior_groups=None):
    if len(fweights.shape) == 2:
        fweighting = pd.Series(
            np.sum(scipy.special.softmax(fweights, axis=1), axis=0),
            index=fweights.index,
        )
    elif type(fweights) is pd.Series:
        fweighting = fweights.apply(scipy.special.softmax)
    else:
        fweighting = pd.Series(
            scipy.special.softmax(fweights), index=importance_ser.index
        )
    fweighting = pd.Series(fweighting, index=fweights.index)
    if prior_groups is not None and len(prior_groups) > 0:
        counts = pd.Series(
            data=[c for c in itertools.chain.from_iterable(prior_groups)]
        ).value_counts(sort=False)
        for col in fweights.index:
            if col not in counts.index:
                counts[col] = 0
        cweighting = np.divide(1, (counts.add(0.25) ** 2))
        cols = fweights.index  # .intersection(cweighting.index)
        weighting = fweighting[cols] * cweighting[cols]
    else:
        weighting = pd.Series(1, index=importance_ser.index)
    # print("VIF model weightings: {}".format(weighting))
    try:
        feats = importance_ser.sample(weights=weighting, n=model_size).index.tolist()
    except ValueError:
        print("Duplicated indices!!")
        feats = importance_ser.sample(n=model_size).index.tolist()
    i = 0
    while (
        any([len([c for c in feats if c in fl]) > len(fl) - 2 for fl in prior_groups])
        and i < 50
    ):
        feats = importance_ser.sample(weights=weighting, n=model_size).index.tolist()
        i += 1
    return feats


def vif_subset_iterative(feat_wts, importance_ser, cross_corr, model_size):
    selection_trials = np.ceil(np.sqrt(importance_ser.size))
    feats = list()
    while len(feats) < model_size:
        next_feats = importance_ser.sample(
            weights=np.sum(scipy.special.softmax(feat_wts, axis=1), axis=0),
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
