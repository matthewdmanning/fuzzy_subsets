import copy
import logging
import os
import pickle
import pprint

import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    SelectFromModel,
    VarianceThreshold,
)
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import clone, FunctionTransformer, Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

import PadelChecker
import sample_clusters
import scoring
from archive.grove_feature_selection import get_search_features
from RingSimplifier import RingSimplifer
from vif import calculate_vif, repeated_stochastic_vif
from XCorrFilter import XCorrFilter

mcc = make_scorer(balanced_accuracy_score)
logger = logging.getLogger(name="selection")


def _permutation_removal(train_df, labels, estimator, select_params, selection_state):
    n_repeats = select_params["perm_n_repeats"]
    while len(selection_state["current_features"]) > select_params["features_min_perm"]:
        estimator = clone(estimator).fit(
            train_df[selection_state["current_features"]], labels
        )
        perm_results = permutation_importance(
            estimator,
            train_df[selection_state["current_features"]],
            labels,
            n_repeats=n_repeats,
            scoring=select_params["scoring"],
            n_jobs=-1,
        )
        import_mean = pd.Series(
            data=perm_results["importances_mean"],
            index=selection_state["current_features"],
            name="Mean",
        ).sort_values()
        import_std = pd.Series(
            data=perm_results["importances_std"],
            index=selection_state["current_features"],
            name="StD",
        )
        adj_importance = import_mean.iloc[0] + import_std[import_mean.index].iloc[0]
        if (
            selection_state["fails"] < 2
            or len(selection_state["current_features"]) < 15
        ):
            import_thresh = min(select_params["thresh_perm"] + 0.05, adj_importance)
        else:
            import_thresh = min(select_params["thresh_perm"], adj_importance)
        unimportant = import_mean[import_mean <= import_thresh].index.tolist()
        if len(unimportant) > 1 and n_repeats <= 50:
            n_repeats = n_repeats + 25
            continue
        elif len(unimportant) == 1 or (n_repeats >= 50 and len(unimportant) > 0):
            low_feats = unimportant[0]
        else:
            selection_state["fails"] = max(0, selection_state["fails"] - 1)
            break
            # [selection_state["current_features"].remove(c) for c in low_feats]
        selection_state["current_features"].remove(low_feats)
        selection_state[""].update([(low_feats, "Importance")])
        print(
            "Permutation drop: \n{}: {} {}".format(
                low_feats,
                import_mean[low_feats],
                import_std[low_feats],
            )
        )
        selection_state["fails"] = max(0, selection_state["fails"] - 1)
    return selection_state


def _vif_elimination(
    train_df, best_corrs, cross_corr, select_params, selection_state, verbose=False
):
    model_size = 10
    vif_rounds = int(
        len(selection_state["current_features"])
        * (len(selection_state["current_features"]) + 1)
        / model_size
    )
    # fweights = 0.5 - pd.Series(scipy.linalg.norm(cross_corr.loc[selection_state["current_features"], selection_state["current_features"]], axis=1), index=selection_state["current_features"]).squeeze().sort_values(ascending=False)
    vif_results = repeated_stochastic_vif(
        feature_df=train_df[selection_state["current_features"]],
        importance_ser=best_corrs[selection_state["current_features"]].abs(),
        threshold=select_params["thresh_vif"],
        model_size=model_size,
        feat_wts=cross_corr[selection_state["current_features"]].loc[
            selection_state["current_features"]
        ],
        min_feat_out=len(selection_state["current_features"]) - 1,
        rounds=vif_rounds,
    )
    if len(vif_results["vif_stats_list"]) > 1:
        all_vif_mean = (
            pd.concat([df["Max"] for df in vif_results["vif_stats_list"]], axis=1)
            .max()
            .sort_values(ascending=False)
        )
        all_vif_sd = pd.concat(
            [df["StdDev"] for df in vif_results["vif_stats_list"]], axis=1
        ).std(ddof=len(selection_state["current_features"]) - model_size)
    else:
        all_vif_mean = vif_results["vif_stats_list"][0]["Max"].sort_values(
            ascending=False
        )
        all_vif_sd = vif_results["vif_stats_list"][0]["SD"]
    vif_dropped = all_vif_mean.index[0]
    # [print(c, [d.loc[c] for d in vif_results["vif_stats_list"] if c in d.index]) for c in vif]
    if (
        vif_dropped in selection_state["current_features"]
        and all_vif_mean[vif_dropped] > select_params["thresh_vif"]
    ):
        selection_state["current_features"].remove(vif_dropped)
        # selection_state["best_score_adj"].update([(vif_dropped, "VIF")])
        selection_state["fails"] -= 1
        if False:
            print("VIF Scores")
            pprint.pp(
                pd.concat(
                    [
                        all_vif_mean,
                        all_vif_sd[all_vif_mean.index],
                        vif_results["votes_list"][-1].loc[all_vif_mean.index],
                    ],
                    axis=1,
                ).head(n=3),
                width=120,
            )
    else:
        selection_state["fails"] -= 1
        if verbose:
            print("Nothing above threshold.")

    return selection_state


def select_feature_subset(
    train_df,
    labels,
    target_corr,
    cross_corr,
    select_params,
    initial_subset=None,
    save_dir=None,
    hidden_test=None,
    selection_models=None,
):
    if (
        labels.nunique() == 2
        and (labels[labels == 1].size < 10 or labels[labels == 0].size < 10)
    ) or labels.nunique == 1:
        raise ValueError
    if select_params["cv"] is None:
        select_params["cv"] = RepeatedStratifiedKFold(random_state=0, n_repeats=3)
    if select_params["scoring"] is None:
        select_params["scoring"] = make_scorer(balanced_accuracy_score)
    with open("{}selection_params.txt".format(save_dir), "w") as f:
        for k, v in select_params.items():
            f.write("{}: {}\n".format(k, v))
    if initial_subset is None:
        initial_subset = [target_corr.abs().idxmax()]
    selection_state = {
        "current_features": initial_subset,
        "subset_scores": dict(),
        "best_subset": list(),
        "rejected_features": dict(),
        "best_score_adj": 0.0,
        "previous_best_score": 0.0,
        "fails": 0,
    }
    print(selection_state["current_features"])

    # PCA
    # calculate_pca(save_dir, train_df)

    # pprint.pp(selection_state["rejected_features"].items())
    sqcc_df = cross_corr * cross_corr
    print("Matrix shapes.")
    print(train_df.shape, sqcc_df.shape, target_corr.shape)
    # Start feature loop
    for i in np.arange(select_params["max_trials"]):
        if (
            len(selection_state["current_features"]) > select_params["max_features_out"]
            or train_df.shape[1]
            - len([selection_state["rejected_features"].keys()])
            - len(selection_state["current_features"])
            < select_params["n_vif_choices"]
        ):
            print("Out of features to pick.")
            print(selection_state["current_features"])
            print(len([selection_state["rejected_features"].keys()]))
            break
        clean_up = False
        try:
            sqcc_df_choices = sqcc_df.drop(
                index=selection_state["rejected_features"]
            ).drop(columns=selection_state["rejected_features"])
        except KeyError:
            print("Key Error in sqcc_df")
            raise KeyError
            sqcc_df_choices = (
                sqcc_df[train_df.columns.intersection(sqcc_df.index)]
                .loc[train_df.columns.intersection(sqcc_df.index)]
                .drop(index=selection_state["rejected_features"])
                .drop(columns=selection_state["rejected_features"])
            )
        target_corr_choices = target_corr[
            train_df.columns.intersection(target_corr.index)
        ].drop(selection_state["rejected_features"])
        new_feat, selection_state = choose_next_feature(
            feature_df=train_df,
            feature_list=selection_state["current_features"],
            target_corr=target_corr_choices,
            sq_xcorr=sqcc_df_choices,
            selection_models=selection_models,
            selection_state=selection_state,
        )
        if new_feat is None:
            break
        else:
            selection_state, subset_scores = score_subset(
                feature_df=train_df,
                labels=labels,
                selection_models=selection_models,
                selection_state=selection_state,
                select_params=select_params,
                save_dir=save_dir,
                record_results=True,
            )
        print(selection_state["current_features"], subset_scores)
        if score_drop_exceeded(
            subset_scores,
            selection_params=select_params,
            selection_state=selection_state,
        ):
            while (
                np.mean(subset_scores) + np.std(subset_scores)
                < selection_state["best_score_adj"]
            ):
                if (
                    len(selection_state["current_features"])
                    <= select_params["features_min_sfs"]
                ):
                    selection_state["current_features"] = selection_state["best_subset"]
                    break
                selection_state, subset_scores = sequential_elimination(
                    train_df,
                    labels,
                    select_params,
                    selection_state,
                    selection_models,
                    clean_up=False,
                    save_dir=save_dir,
                )
            continue
        # Variance Inflation Factor: VIF check implemented in "new feature" selection function.
        if (
            _get_fails(selection_state) >= select_params["fails_min_vif"]
            and len(selection_state["current_features"])
            >= select_params["features_min_vif"]
        ):
            n_fails = copy.deepcopy(_get_fails(selection_state))
            selection_state = _vif_elimination(
                train_df=train_df,
                best_corrs=target_corr,
                cross_corr=cross_corr,
                select_params=select_params,
                selection_state=selection_state,
            )
            if n_fails > _get_fails(selection_state):
                break
        # Feature Importance Elimination
        if _get_fails(selection_state) >= select_params[
            "fails_min_perm"
        ] and select_params["features_min_perm"] < len(
            selection_state["current_features"]
        ):
            if select_params["importance"] == "permutate":
                selection_state = _permutation_removal(
                    train_df=train_df,
                    labels=labels,
                    estimator=clone(selection_models["permutation"]),
                    select_params=select_params,
                    selection_state=selection_state,
                )
            elif (
                select_params["importance"] != "permutate"
                and select_params["importance"] is not False
            ):
                # rfe = RFECV(estimator=grove_model, min_features_to_select=len(selection_state["current_features"])-1, n_jobs=-1).set_output(transform="pandas")
                rfe = SelectFromModel(
                    estimator=clone(selection_models["importance"]),
                    max_features=len(selection_state["current_features"]) - 1,
                ).set_output(transform="pandas")
                rfe.fit(train_df[selection_state["current_features"]], labels)
                # TODO: Add dropped to "rejected_features"
                # TODO: Fix "fails" adjustment
                if True:  # any(~rfe.support_):
                    dropped = [
                        c
                        for c in selection_state["current_features"]
                        if c
                        not in rfe.get_feature_names_out(
                            selection_state["current_features"]
                        )
                    ][0]
                    # dropped = train_df[selection_state["current_features"]].columns[~rfe.support_][0]
                    print("Dropped: {}".format(dropped))
                    selection_state["rejected_features"].update((dropped, "importance"))
                    selection_state["current_features"].remove(dropped)
                    continue
                else:
                    # _get_fails(selection_state)
                    break
        if (
            len(selection_state["current_features"])
            >= select_params["max_features_out"]
        ):
            clean_up = True
        while (
            _get_fails(selection_state) >= select_params["fails_min_sfs"] or clean_up
        ) and (
            select_params["features_min_sfs"] < len(selection_state["current_features"])
        ):
            # DEBUG
            print(len(selection_state["current_features"]))
            print(select_params["min_features_out"])
            if (
                len(selection_state["current_features"])
                <= select_params["min_features_out"]
            ):
                clean_up = False
                break
            n_features_in = copy.deepcopy(len(selection_state["current_features"]))
            selection_state, subset_scores = sequential_elimination(
                train_df,
                labels,
                select_params=select_params,
                selection_state=selection_state,
                selection_models=selection_models,
                clean_up=clean_up,
                save_dir=save_dir,
            )
            # SFS fails to eliminate a feature.
            # DEBUG
            print(
                "Features in: {}. Features out: {}".format(
                    n_features_in, len(selection_state["current_features"])
                )
            )
            if n_features_in == len(selection_state["current_features"]):
                clean_up = False
                break
            too_much, selection_state = score_drop_exceeded(
                new_scores=subset_scores,
                selection_params=select_params,
                selection_state=selection_state,
            )
            if too_much:
                clean_up = False
                break

    print(
        "Best adjusted score of {} with feature set: {}".format(
            selection_state["best_score_adj"],
            selection_state["best_subset"],
        )
    )
    # TODO: Check if this is even possible.
    if len(selection_state["best_subset"]) > 0:
        best_fit_model = FrozenEstimator(
            selection_models["predict"].fit(
                train_df[selection_state["best_subset"]], labels
            )
        )
        with open("{}best_model.pkl".format(save_dir), "wb") as f:
            pickle.dump(best_fit_model, f)
    else:
        raise RuntimeError
    print("Rejects: {}".format(selection_state["rejected_features"]))
    pd.Series(selection_state["rejected_features"], name="Dropped Features").to_csv(
        "{}dropped_features.csv".format(save_dir)
    )
    return (
        selection_models["predict"],
        selection_state["subset_scores"],
        selection_state["rejected_features"],
        selection_state["best_subset"],
    )


def _get_fails(selection_state):
    return max(
        0,
        len(selection_state["current_features"]) - len(selection_state["best_subset"]),
    )


def sequential_elimination(
    train_df,
    labels,
    select_params,
    selection_state,
    selection_models,
    clean_up,
    save_dir,
    randomize=False,
    depth=1,
):
    if clean_up:
        if "thresh_sfs_cleanup" not in select_params.keys():
            sfs_tol = 2 * select_params["thresh_sfs"]
        else:
            sfs_tol = select_params["thresh_sfs_cleanup"]
    else:
        sfs_tol = select_params["thresh_sfs"]
    # TODO: Implement configurable predict function for boosting.
    # TODO: Eliminate duplicated scoring.
    sfs = (
        SequentialFeatureSelector(
            estimator=clone(selection_models["predict"]),
            direction="backward",
            tol=sfs_tol,
            n_features_to_select=len(selection_state["current_features"]) - 1,
            scoring=select_params["scoring"],
        )
        .set_output(transform="pandas")
        .fit(train_df[selection_state["current_features"]], y=labels)
    )
    new_features = sorted(
        sfs.get_feature_names_out(selection_state["current_features"]).tolist()
    )
    original_features = copy.deepcopy(selection_state["current_features"])
    selection_state["current_features"] = new_features
    selection_state, subset_scores = score_subset(train_df, labels=labels, selection_models=selection_models,
                                                  selection_state=selection_state, select_params=select_params,
                                                  save_dir=save_dir, hidden_test=None, record_results=True)
    if (
        np.mean(subset_scores) - np.std(subset_scores)
        > selection_state["best_score_adj"] + sfs_tol
    ):
        sfs_drops = []
        for c in original_features:
            if c not in new_features:
                sfs_drops.append(c)
        selection_state["rejected_features"].update([(c, "SFS") for c in sfs_drops])
    else:
        selection_state["current_features"] = original_features
    """
    return selection_state, subset_scores


def score_subset(
    feature_df,
    labels,
    selection_models,
    selection_state,
    select_params,
    save_dir,
    subset=None,
    record_results=False,
):
    print("Subset passed".format(subset))
    if subset is None or len(subset) == 0:
        subset = tuple(sorted(selection_state["current_features"]))
    elif len(selection_state["current_features"]):
        print(selection_state, flush=True)
        raise ValueError
    if isinstance(subset, str):
        # raise KeyError
        current_features = [copy.deepcopy(subset)]
    else:
        current_features = tuple(sorted(copy.deepcopy(subset)))
    score_tuple = [(select_params["score_name"], select_params["scoring"])]
    scores = None
    for prior_set in selection_state["subset_scores"].keys():
        if len(set(current_features).symmetric_difference(prior_set)) == 0:
            scores = selection_state["subset_scores"][current_features]
            print("Duplicate scoring found:\n{}\n".format(current_features, prior_set))
            break
    if scores is None:
        selection_state["subset_scores"][current_features] = list()
        # best_corrs, cross_corr = get_correlations(train_df, train_labels, path_dict["corr_path"], path_dict["xc_path"], select_params["corr_method"], select_params["xc_method"])
        results = scoring.cv_model_generalized(
            estimator=selection_models["predict"],
            feature_df=feature_df[list(current_features)],
            labels=labels,
            cv=select_params["cv"],
            scorer_tups=score_tuple,
        )
        scores = results[select_params["score_name"]]["test"]
        selection_state["subset_scores"][tuple(sorted(current_features))] = scores
        if record_results:
            selection_state = record_score(
                selection_state=selection_state,
                scores=scores,
                save_dir=save_dir,
            )
    # print(np.mean(scores))
    return selection_state, scores


def choose_next_feature(
    feature_df,
    feature_list,
    target_corr,
    sq_xcorr,
    selection_models,
    vif_choice=5,
    selection_state=None,
):
    feat_corrs = target_corr.drop(
        [
            c
            for c in target_corr.index
            if (c in selection_state["rejected_features"].keys() or c in feature_list)
        ]
    )
    if len(feature_list) == 0:
        sum_sqcc = pd.Series(1 / feat_corrs.index.size, index=feat_corrs.index)
    else:
        ones = pd.DataFrame(
            data=np.ones(
                shape=(feat_corrs.index.size, len(feature_list)), dtype=np.float32
            ),
            index=feat_corrs.index,
            columns=feature_list,
        )
        sum_sqcc = ones.subtract(sq_xcorr[feature_list].loc[feat_corrs.index]).sum(
            axis=1
        )
    x = sum_sqcc.multiply(other=np.abs(feat_corrs), fill_value=0.0)
    feat_probs = pd.Series(scipy.special.softmax(x), index=x.index).sort_values(
        ascending=False
    )
    # feature_list = list(set(feature_list))
    # noinspection PyTypeChecker
    # TODO: Clean up this logic. Purpose: Avoid string splitting. Need to get better type checking.
    if len(feature_list) <= 1:
        vif_choice = None
    new_feat = np.random.choice(
        a=feat_corrs.index.to_numpy(), size=vif_choice, replace=False, p=feat_probs
    )
    if new_feat is None:
        print("No new feature selected.")
        return None, selection_state
    elif isinstance(new_feat, (str, int)):
        if new_feat not in sq_xcorr.columns.tolist():
            print("{} not in cross_corr".format(new_feat))
            raise RuntimeError("New feature is not in cross-correlation matrix")
            new_feat = None
        else:
            new_feat = [new_feat]
    else:
        new_feat = [
            x for x in new_feat if x in sq_xcorr.columns and x in feature_df.columns
        ]
        assert len(new_feat) > 0
    if np.size(new_feat) > 1 or len(new_feat) > 1:
        vifs = dict()
        for nf in new_feat:
            if nf not in feature_df.columns:
                print("Feature {} not in feature_df".format(nf))
            predictors = copy.deepcopy(feature_list)
            predictors.append(nf)
            vifs_ser = calculate_vif(
                feature_df=feature_df[predictors],
                model=clone(selection_models["vif"]),
                subset=feature_df[[nf]],
            )
            vifs[nf] = vifs_ser.min()
        print(vifs)
        vif_selected = pd.Series(vifs).idxmin()
        print(vif_selected)
    else:
        vif_selected = new_feat[0]
    if len(selection_state["current_features"]) == 0:
        selection_state["current_features"] = [vif_selected]
    else:
        selection_state["current_features"].append(vif_selected)
        try:
            selection_state["current_features"] = sorted(
                selection_state["current_features"]
            )
        except TypeError:
            print("Couldn't sort current_features list.")
            print(vif_selected)
            print(selection_state["current_features"])
            raise ValueError
    if any([isinstance(f, tuple) for f in selection_state["current_features"]]):
        # DEBUG
        print("Current features list contains tuple.")
        raise TypeError
    return vif_selected, selection_state


def process_selection_data(
    feature_df=None,
    labels=None,
    dropped_dict=None,
    save_dir=None,
    select_params=None,
    scaler="standard",
    transform=None,
):
    preloaded = False
    drop_corr = False
    best_corrs, cross_corr = None, None
    if feature_df is None:
        feature_df, labels = sample_clusters.grab_enamine_data()
    if dropped_dict is None:
        dropped_dict = dict()
    combo_transform, scaler, best_corrs, cross_corr = get_standard_preprocessor(
        scaler, transform, select_params
    )
    combo_transform.fit(X=feature_df, y=labels)
    with open("{}transformer.pkl".format(save_dir), "wb") as f:
        pickle.dump(combo_transform, f)
    train_df = combo_transform.transform(feature_df)
    return train_df, labels, best_corrs, cross_corr, scaler


def get_standard_preprocessor(
    scaler=None, transform_func=None, corr_params=None, use_short_names=True
):
    if scaler is not None and scaler == "standard":
        scaler = StandardScaler().set_output(transform="pandas")
    elif scaler is not None and scaler == "robust":
        scaler = RobustScaler(unit_variance=True).set_output(transform="pandas")
    if transform_func is not None and transform_func == "asinh":
        transform_func = np.arcsinh  # get_transform_func(np.arcsinh)
        inv_transform = None
    else:
        transform_func, inv_transform = None, None
    # Define individual transformers.
    ring_tranform = (
        "rings",
        RingSimplifer(short=use_short_names).set_output(transform="pandas"),
    )
    padel_transform = (
        "padel",
        PadelChecker.PadelCheck().set_output(transform="pandas"),
    )
    var_thresh = ("var", VarianceThreshold().set_output(transform="pandas"))
    pipe_list = [padel_transform, ring_tranform, var_thresh]
    # pipe_list = [padel_transform, var_thresh]
    if transform_func is not None:
        smooth_transform = FunctionTransformer(
            func=transform_func,
            inverse_func=inv_transform,
            feature_names_out="one-to-one",
        ).set_output(transform="pandas")
        pipe_list.append(("smooth", smooth_transform))
    if scaler is not None:
        pipe_list.append(("scaler", scaler))
    if corr_params is not None:
        xcorr_filter = XCorrFilter(
            max_features=None,
            thresh_xc=corr_params["thresh_xc"],
            method_xc=corr_params["xc_method"],
        )
        xc_df = xcorr_filter.xcorr_
        pipe_list.append(("xcorr", xcorr_filter))
    else:
        xc_df = None
        corr_df = None
    # combo_transform = Pipeline(steps=[("rings", ring_tranform), ("padel", padel_transform), ("var", var_thresh), ("scale", smooth_transform), ("xcorr", xcorr_filter)]).set_output(transform="pandas")
    combo_transform = Pipeline(steps=pipe_list).set_output(transform="pandas")
    return combo_transform, scaler, xc_df


def record_score(selection_state, scores, save_dir, test_score=None):
    with open("{}feature_score_path.csv".format(save_dir), "a", encoding="utf-8") as f:
        f.write(
            "{}\t{}\n".format(
                "\t".join(["{:.5f}".format(sc) for sc in scores]),
                "\t".join(selection_state["current_features"]),
            )
        )
    if test_score is not None:
        with open("{}test_scores.csv".format(save_dir), "a", encoding="utf-8") as f:
            f.write(
                "{:.5f}\t{}\n".format(
                    test_score,
                    "\t".join(selection_state["current_features"]),
                )
            )
    selection_state["subset_scores"][
        tuple(sorted(selection_state["current_features"]))
    ] = scores
    best_yet = _compare_to_best(scores, selection_state)
    return selection_state


def _compare_to_best(scores, selection_state):
    if np.mean(scores) - np.std(scores) > selection_state["best_score_adj"]:
        print(
            "New top results for {} feature model: Mean: {:.4f}, Std {:.4f}".format(
                len(selection_state["current_features"]),
                np.mean(scores),
                np.std(scores),
            )
        )
        selection_state["previous_best_score"] = copy.deepcopy(
            selection_state["best_score_adj"]
        )
        selection_state["best_score_adj"] = np.mean(scores) - np.std(scores)
        selection_state["best_subset"] = copy.deepcopy(
            selection_state["current_features"]
        )
        best_yet = True
    else:
        best_yet = False
    return best_yet


def get_clf_model(model_name):
    if "log" in model_name:
        grove_model = LogisticRegressionCV(
            scoring=mcc,
            solver="newton-cholesky",
            tol=2e-4,
            cv=5,
            max_iter=10000,
            class_weight="balanced",
            n_jobs=-4,
        )
    elif "ridge" in model_name:
        from sklearn.linear_model import RidgeClassifierCV

        grove_model = RidgeClassifierCV(scoring=mcc, class_weight="balanced")
    elif "svc" in model_name:
        from sklearn.svm import SVC

        if "poly" in model_name:
            grove_model = SVC(class_weight="balanced", random_state=0, kernel="poly")
        elif "sigmoid" in model_name:
            grove_model = SVC(class_weight="balanced", random_state=0, kernel="sigmoid")
        elif "linear" in model_name:
            grove_model = SVC(class_weight="balanced", random_state=0, kernel="linear")
        else:
            grove_model = SVC(class_weight="balanced", random_state=0, kernel="rbf")
    elif "passive" in model_name:
        from sklearn.linear_model import PassiveAggressiveClassifier

        grove_model = PassiveAggressiveClassifier(
            C=5.0, class_weight="balanced", random_state=0
        )
    elif "xtra" in model_name:
        grove_model = ExtraTreesClassifier(
            max_leaf_nodes=200,
            min_impurity_decrease=0.005,
            max_depth=30,
            class_weight="balanced",
            bootstrap=False,
        )
    else:
        grove_model = RandomForestClassifier(
            max_leaf_nodes=200,
            min_impurity_decrease=0.005,
            max_depth=30,
            class_weight="balanced",
            bootstrap=False,
        )
    return grove_model


def get_regression_model(model_name):
    if "line" in model_name:
        from sklearn.linear_model import LinearRegression

        grove_model = LinearRegression()
    elif "elastic" in model_name:
        from sklearn.linear_model import ElasticNetCV

        grove_model = ElasticNetCV(
            l1_ratio=[0.25, 0.5, 0.75, 0.9],
            tol=1e-4,
            max_iter=10000,
            selection="random",
        )
    elif "hub" in model_name:
        from sklearn.linear_model import HuberRegressor

        grove_model = HuberRegressor(max_iter=1000, tol=1e-04)
    elif "sgd" in model_name:
        from sklearn.linear_model import SGDRegressor

        grove_model = SGDRegressor(max_iter=5000, random_state=0)
    elif "ridge" in model_name:
        from sklearn.linear_model import RidgeCV

        grove_model = RidgeCV()
    elif "lasso" in model_name:
        from sklearn.linear_model import LassoCV

        grove_model = LassoCV(random_state=0)
    elif "xtra" in model_name:
        grove_model = ExtraTreesRegressor(
            max_leaf_nodes=300,
            min_impurity_decrease=0.005,
        )
    elif "krr" in model_name:
        from sklearn.kernel_ridge import KernelRidge

        if "poly" in model_name:
            grove_model = KernelRidge(kernel="polynomial")
        elif "rbf" in model_name:
            grove_model = KernelRidge(kernel="rbf")
        elif "sigmoid" in model_name:
            grove_model = KernelRidge(kernel="sigmoid")
        else:
            grove_model = KernelRidge(kernal="linear")
    elif "gauss" in model_name:
        from sklearn.gaussian_process import GaussianProcessRegressor

        grove_model = GaussianProcessRegressor(n_restarts_optimizer=3, normalize_y=True)
    elif "pls" in model_name:
        from sklearn.cross_decomposition import PLSRegression

        grove_model = PLSRegression(
            n_components=1, max_iter=5000, tol=1e-05
        ).set_output(transform="pandas")
    else:
        grove_model = RandomForestRegressor(
            max_leaf_nodes=200,
            min_impurity_decrease=0.005,
            max_depth=30,
            bootstrap=False,
        )
    return grove_model


def score_drop_exceeded(new_scores, selection_params, selection_state):
    new_score = np.mean(new_scores) - np.std(new_scores)
    if new_score < selection_state["best_score_adj"] + selection_params["thresh_reset"]:
        print(
            "Score (adjusted) drop exceeded: {:.4f} {:.4f}".format(
                selection_state["best_score_adj"], new_score
            )
        )
        selection_state["current_features"] = copy.deepcopy(
            selection_state["best_subset"]
        )
        return True, selection_state
    else:
        return False, selection_state


def main(model_name, importance_name):
    # data_dir = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/test_data/Vapor pressure OPERA/Vapor pressure OPERA/"
    # opera_dir = "{}test_train_split/".format(data_dir)
    """
    train_dir = "{}Vapor pressure OPERA T.E.S.T. 5.1 training.tsv".format(opera_dir)
    train_dir = "{}Vapor pressure OPERA Padelpy webservice single training.tsv".format(
        data_dir
    )
    # train_data = pd.read_csv(train_dir, delimiter="\t")
    train_data.set_index(keys=train_data.columns[0], inplace=True)
    labels = train_data[train_data.columns[0]].copy()
    feature_df = train_data.drop(columns=train_data.columns[0])
    logger.info(train_data.head())
    logger.info(train_data.shape)
    """
    raise DeprecationWarning
    select_params = {
        "corr_method": "kendall",
        "xc_method": "kendall",
        "max_features_out": 30,
        "fails_min_vif": 5,
        "fails_min_perm": 6,
        "fails_min_sfs": 6,
        "features_min_vif": 15,
        "features_min_perm": 15,
        "features_min_sfs": 20,
        "thresh_reset": 0.05,
        "thresh_vif": 15,
        "thresh_perm": 0.0025,
        "thresh_sfs": -0.005,
        "thresh_xc": 0.95,
        "max_trials": 200,
        "cv": 5,
        "importance": True,
        "scoring": None,
    }
    data_transform = "asinh"
    data_dir = "{}enamine_transform_test/".format(os.environ.get("MODEL_DIR"))
    opera_dir = data_dir
    search_dir = "{}test_train_split/{}_1/".format(
        opera_dir, "_".join([model_name, data_transform])
    )
    os.makedirs(search_dir, exist_ok=True)
    train_data, labels, best_corrs, cross_corr, scaler = process_selection_data(
        save_dir=data_dir,
        select_params=select_params,
        transform=data_transform,
    )
    search_features = get_search_features(train_data)
    if labels.nunique() > 2:
        selection_models = {
            "predict": get_regression_model(model_name),
            "permutation": get_regression_model(model_name),
            "importance": get_regression_model(importance_name),
            "vif": LinearRegression(),
        }
    else:
        selection_models = {
            "predict": get_clf_model(model_name),
            "permutation": get_clf_model(model_name),
            "importance": get_clf_model(importance_name),
            "vif": LinearRegression(),
        }
    trained_model = get_clf_model(model_name).fit(train_data, labels)
    with open("{}best_model.pkl".format(search_dir), "wb") as f:
        pickle.dump(FrozenEstimator(trained_model), f)
    # members_dict = membership(feature_df, labels, grove_cols, search_dir)
    dev_idx, eval_idx = [
        (a, b)
        for a, b in StratifiedKFold(shuffle=True, random_state=0).split(
            train_data, labels
        )
    ][0]
    dev_data = train_data.iloc[dev_idx]
    dev_labels = labels[dev_data.index]
    eval_data = train_data.iloc[eval_idx]
    eval_labels = labels[eval_data.index]
    model_dict, score_dict, dropped_dict, best_features = select_feature_subset(
        train_df=dev_data,
        labels=dev_labels,
        target_corr=best_corrs,
        cross_corr=cross_corr,
        select_params=select_params,
        selection_models=selection_models,
        hidden_test=(eval_data, eval_labels),
        save_dir=search_dir,
    )
    if any([a is None for a in [model_dict, score_dict, dropped_dict, best_features]]):
        print(model_dict, score_dict, dropped_dict, best_features)
        raise ValueError
    trained_model = clone(selection_models["predict"]).fit(
        dev_data[best_features], dev_labels
    )
    train_score = trained_model.score(dev_data[best_features], dev_labels)
    test_score = trained_model.score(eval_data[best_features], labels[eval_data.index])
    print(
        "Best feature subset scores: Train: {:.5f}, Test: {:.5f}".format(
            train_score, test_score
        )
    )
    print("Using {} features".format(len(best_features)))
    print("Dropped features: ", dropped_dict.items())
    print(best_features)
    return None
    """
    for i, (col, members) in enumerate(list(members_dict.items())):
        if (
            labels[members][labels[members] == 0].size < 15
            or labels[members][labels[members] == 1].size < 15
        ):
            continue
        col_dir = "{}/{}/".format(search_dir, i)
        os.makedirs(col_dir, exist_ok=True)
        with sklearn.config_context(
            enable_metadata_routing=True, transform_output="pandas"
        ):
            model_dict[col], score_dict[col], dropped_dict, best_features = (
                select_feature_subset(
                    feature_df.loc[members][search_features],
                    labels[members],
                    None,
                    None,
                    select_params=None,
                    initial_subset=col,
                    save_dir=col_dir,
                )
            )
    """


if __name__ == "__main__":
    importance = "rfc"
    logger = logging.getLogger(name="selection")
    for md in ["svc_rbf", "xtra", "ridge", "rfc", "passive", "log"]:
        main(model_name=md, importance_name=importance)
