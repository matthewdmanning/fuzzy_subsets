import copy
import logging
import os
import pickle
import pprint

import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    SelectFromModel,
    SequentialFeatureSelector,
    VarianceThreshold,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import (
    cross_val_score,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import clone, FunctionTransformer, Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

import PadelChecker
import sample_clusters
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
    if select_params["cv"] is None:
        select_params["cv"] = RepeatedStratifiedKFold(random_state=0, n_repeats=3)
    if select_params["scoring"] is None:
        select_params["scoring"] = make_scorer(balanced_accuracy_score)
    with open("{}selection_params.txt".format(save_dir), "w") as f:
        for k, v in select_params.items():
            f.write("{}: {}\n".format(k, v))
    if initial_subset is None:
        initial_subset = [target_corr.abs().sort_values(ascending=False).index[0]]
    selection_state = {
        "current_features": initial_subset,
        "subset_scores": dict(),
        "best_subset": list(),
        "rejected_features": dict(),
        "best_score_adj": 0.0,
        "previous_best_score": 0.0,
        "fails": 0,
    }
    """    
    """
    with open("{}best_model.pkl".format(save_dir), "wb") as f:
        pickle.dump(selection_models["predict"], f)
    if labels.nunique() == 2 and (
        labels[labels == 1].size < 10 or labels[labels == 0].size < 10
    ):
        raise ValueError
    # PCA
    # calculate_pca(save_dir, train_df)

    pprint.pp(selection_state["rejected_features"].items())
    sqcc_df = cross_corr * cross_corr
    # Start feature loop
    for i in np.arange(select_params["max_trials"]):
        if (
            len(selection_state["current_features"]) > select_params["max_features_out"]
            or train_df.shape[1]
            - len([selection_state["rejected_features"].keys()])
            - len(selection_state["current_features"])
            > 2
        ):
            print("Out of features to pick")
            break
        clean_up = False
        try:
            sqcc_df_choices = sqcc_df.drop(
                index=selection_state["rejected_features"]
            ).drop(columns=selection_state["rejected_features"])
        except KeyError:
            print("Key Error in sqcc_df")
            sqcc_df_choices = (
                sqcc_df[train_df.columns.intersection(sqcc_df.index)]
                .loc[train_df.columns.intersection(sqcc_df.index)]
                .drop(index=selection_state["rejected_features"])
                .drop(columns=selection_state["rejected_features"])
            )
        target_corr_choices = target_corr[
            train_df.columns.intersection(target_corr.index)
        ].drop(selection_state["rejected_features"])
        new_feat = choose_next_feature(
            train_df,
            feature_list=selection_state["current_features"],
            target_corr=target_corr_choices,
            sq_xcorr=sqcc_df_choices,
            selection_models=selection_models,
            selection_state=selection_state,
        )
        if new_feat is None:
            break
        selection_state["current_features"].append(new_feat)
        selection_state, scores = score_subset(
            train_df,
            labels,
            selection_models=selection_models,
            selection_state=selection_state,
            select_params=select_params,
            save_dir=save_dir,
            hidden_test=hidden_test,
            record_results=True,
        )
        # Variance Inflation Factor: Now implemented in feature selection phase
        if False and (
            selection_state["fails"] >= select_params["fails_min_vif"]
            and len(selection_state["current_features"])
            >= select_params["features_min_vif"]
        ):
            n_fails = copy.deepcopy(selection_state["fails"])
            selection_state = _vif_elimination(
                train_df=train_df,
                best_corrs=target_corr,
                cross_corr=cross_corr,
                select_params=select_params,
                selection_state=selection_state,
            )
            if n_fails > selection_state["fails"]:
                break
        if selection_state["fails"] >= select_params[
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
                    selection_state["rejected_features"][dropped] = "importance"
                    selection_state["current_features"].remove(dropped)
                    selection_state["fails"] = max(0, selection_state["fails"] - 1.5)
                    continue
                else:
                    selection_state["fails"] = max(0, selection_state["fails"] - 1)
                    break
        if (
            len(selection_state["current_features"])
            == select_params["max_features_out"]
        ):
            clean_up = True
        if (
            selection_state["fails"] >= select_params["fails_min_sfs"]
            and select_params["features_min_sfs"]
            < len(selection_state["current_features"])
            or (clean_up and len(selection_state["current_features"]) > 3)
        ):
            selection_state, subset_scores = sequential_elimination(
                train_df,
                labels,
                select_params,
                selection_state,
                selection_models,
                clean_up,
                hidden_test,
                save_dir,
            )
            if score_drop_exceeded(subset_scores, select_params, selection_state):
                select_params["current_features"] = selection_state["best_subset"]
                continue
            else:
                selection_state["fails"] = max(0, selection_state["fails"] - 1)
                clean_up = False
                continue
    print(
        "Best adjusted score of {} for {} with feature set: {}".format(
            selection_state["best_score_adj"],
            initial_subset,
            selection_state["best_subset"],
        )
    )
    if len(selection_state["best_subset"]) > 0:
        with open("{}best_model.pkl".format(save_dir), "wb") as f:
            pickle.dump(
                selection_models["predict"].fit(
                    train_df[selection_state["best_subset"]], labels
                ),
                f,
            )
    pd.Series(selection_state["rejected_features"], name="Dropped Features").to_csv(
        "{}dropped_features.csv".format(save_dir)
    )
    return (
        selection_models["predict"],
        selection_state["subset_scores"],
        selection_state["rejected_features"],
        selection_state["best_subset"],
    )


def sequential_elimination(
    train_df,
    labels,
    select_params,
    selection_state,
    selection_models,
    clean_up,
    hidden_test,
    save_dir,
    randomize=True,
    depth=1,
):
    sfs_tol = select_params["thresh_sfs"]
    if clean_up:
        if "thresh_sfs_cleanup" not in select_params.keys():
            sfs_tol = 2 * select_params["thresh_sfs"]
        else:
            sfs_tol = select_params["thresh_sfs_cleanup"]
    # TODO: Manual SFS for duplicated fitting and over-subscribing.
    # TODO: Implement configurable predict function for boosting.
    # TODO: Eliminate duplicated scoring.
    sfs = (
        SequentialFeatureSelector(
            estimator=clone(selection_models["predict"]),
            direction="backward",
            tol=sfs_tol,
            n_features_to_select=len(selection_state["current_features"]) - 1,
            scoring=select_params["scoring"],
            n_jobs=-2,
        )
        .set_output(transform="pandas")
        .fit(train_df[selection_state["current_features"]], y=labels)
    )
    new_features = sorted(
        sfs.get_feature_names_out(selection_state["current_features"]).tolist()
    )
    selection_state["current_features"] = new_features
    selection_state, subset_scores = score_subset(
        train_df,
        labels=labels,
        selection_models=selection_models,
        selection_state=selection_state,
        select_params=select_params,
        save_dir=save_dir,
        hidden_test=hidden_test,
        record_results=True,
    )
    if (
        np.mean(subset_scores) - np.std(subset_scores)
        > selection_state["best_score_adj"] + select_params["thresh_sfs"]
    ):
        sfs_drops = []
        for c in selection_state["current_features"]:
            if c not in new_features:
                sfs_drops.append(c)
        [
            selection_state["current_features"].remove(d)
            for d in selection_state["current_features"]
            if d not in new_features
        ]
        selection_state["rejected_features"].update([(c, "SFS") for c in sfs_drops])
    return selection_state, subset_scores


def score_subset(
    train_df,
    labels,
    selection_models,
    selection_state,
    select_params,
    save_dir,
    hidden_test=None,
    record_results=False,
):
    if (
        isinstance(selection_state["current_features"], str)
        or len(selection_state["current_features"]) == 1
    ):
        current_features = [selection_state["current_features"]]
    else:
        current_features = selection_state["current_features"]
    for subset in selection_state["subset_scores"].keys():
        if len(set(current_features).symmetric_difference(subset)) == 0:
            return (
                selection_state,
                selection_state["subset_scores"][tuple(sorted(current_features))],
            )
    scores = list()
    for dev_df, dev_labels, eval_df, eval_labels in cv_tools.split_df(
        train_df[current_features], labels, splitter=select_params["cv"]
    ):
        fitted_est = clone(selection_models["predict"]).fit(
            X=dev_df[current_features], y=dev_labels
        )
        scores.append(
            select_params["scoring"](
                estimator=fitted_est, X=eval_df, y_true=eval_labels
            )
        )
    if hidden_test is not None:
        test_score = select_params["scoring"](
            estimator=fitted_est,
            X=hidden_test[0][current_features],
            y_true=hidden_test[1],
        )
        test_score = trained_model.score(
            hidden_test[0][current_features], hidden_test[1]
        )
        print("Test Score: {}".format(test_score))
    else:
        test_score = None
    if record_results:
        selection_state = record_score(
            selection_state=selection_state,
            scores=scores,
            save_dir=save_dir,
            test_score=test_score,
        )
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
    print(feature_list)
    feat_corrs = target_corr.drop(
        [
            c
            for c in target_corr.index
            if c in selection_state["rejected_features"].keys() or c in feature_list
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
    if len(feature_list) <= 1:
        vif_choice = None
    new_feat = np.random.choice(
        a=feat_corrs.index.to_numpy(), size=vif_choice, replace=False, p=feat_probs
    )

    if isinstance(new_feat, str):
        if new_feat in sq_xcorr.columns.tolist():
            new_feat = new_feat
        else:
            print("{} not in cross_corr".format(new_feat))
            raise RuntimeWarning("New feature is not in cross-correlation matrix")
            new_feat = None
    elif np.size(new_feat) > 1 and all(
        [x in sq_xcorr.columns.tolist() for x in new_feat]
    ):
        vifs = dict()
        for nf in new_feat:
            predictors = copy.deepcopy(feature_list)
            predictors.append(nf)
            vifs = calculate_vif(
                feature_df=feature_df[predictors],
                model=clone(selection_models["vif"]),
                subset=feature_df[[nf]],
            )
        vifs = dict(sorted(vifs.items(), key=lambda x: x[1]))
        new_feat = list(vifs.keys())[0]
    return new_feat


def calculate_pca(save_dir, train_df):
    pca_path = "{}feature_pca.pkl".format(save_dir)
    if os.path.isfile(pca_path) and False:
        pc = PCA(n_components="mle").set_output(transform="pandas")
        train_pca = pc.fit_transform(train_df)
        pd.DataFrame(
            pc.get_covariance(), index=train_df.columns, columns=train_df.columns
        ).to_pickle(pca_path)
        print("PCA Results: Noise Variance and Number of Components to Reach 0.9 EVR:")
        print(pc.noise_variance_)
        evr = np.cumsum(pc.explained_variance_ratio_)
        [print(n, v) for n, v in enumerate(evr) if v < 0.99 and n <= 10]


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
    combo_transform, scaler = get_standard_preprocessor(
        scaler, transform, select_params
    )
    combo_transform.fit(X=feature_df, y=labels)
    with open("{}transformer.pkl".format(save_dir), "wb") as f:
        pickle.dump(combo_transform, f)
    # best_corrs = xcorr_filter.target_corr_
    # cross_corr = xcorr_filter.xcorr_
    # dropped_dict.update([(c, "Xcorr") for c in xcorr_filter.dropped_features_])

    # Update other DataFrames
    """
    if transform is not None:
        train_df = train_df.map(transform, na_action="ignore")
    else:
        train_df = feature_df
    if scaler is not None:
        scaler = scaler.fit(train_df)
        train_df = scaler.transform(train_df)
        combo_scaler = scaler
        if os.path.isdir(save_dir):
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
    elif feature_df is not None and isinstance(feature_df, pd.DataFrame):
        preloaded = True
        feature_df = feature_df.copy()
        feature_df = feature_df[~feature_df.index.duplicated()]
        labels = labels[feature_df.index]
        feature_df.dropna(axis=1, inplace=True)
        drop_corr = True
    # FunctionTransformer cannot be used for fitted transformers, only stateless functions.
    # custom_scaler = partial(double_scaler, type_transformer_dict={"apply": transform, "scikit": scaler})
    # combo_scaler = FunctionTransformer(func=custom_scaler, feature_names_out="one-to-one")

    if select_params is not None:
        best_corrs, cross_corr = get_corrs(train_df, labels, select_params)
        logger.debug(labels.shape, train_df.shape)
        logger.debug(
            train_df.isna().astype(int).sum(axis=0).sort_values(ascending=False)
        )
        # sample_wts = compute_sample_weight(class_weight="balanced", y=labels)
        if drop_corr:
            del_ser = xcorr_filter(
                train_df, best_corrs, cross_corr, select_params, dropped_dict
            )
            train_df.drop(columns=del_ser.index, inplace=True)
        best_corrs.sort_values(ascending=False, inplace=True, key=lambda x: np.abs(x))
        if os.path.isdir(save_dir) and not preloaded:
            train_df.to_pickle(data_df_path)
            labels.to_csv(label_path)
            best_corrs.to_csv(corr_path)
            cross_corr.to_csv(xc_path)
        assert train_df.index.equals(labels.index)
        """
    train_df = combo_transform.transform(feature_df)
    return train_df, labels, best_corrs, cross_corr, scaler


def get_standard_preprocessor(scaler=None, transform_func=None, corr_params=None):
    if scaler is not None and scaler == "standard":
        scaler = StandardScaler()  # .set_output(transform="pandas")
    elif scaler is not None and scaler == "robust":
        scaler = RobustScaler(unit_variance=True)  # .set_output(transform="pandas")
    if transform_func is not None and transform_func == "asinh":
        transform_func = np.arcsinh  # get_transform_func(np.arcsinh)
        inv_transform = None
    else:
        transform_func, inv_transform = None, None
    # Define individual transformers.
    ring_tranform = ("rings", RingSimplifer())
    padel_transform = ("padel", PadelChecker.PadelCheck())
    var_thresh = ("var", VarianceThreshold())
    pipe_list = [ring_tranform, padel_transform, var_thresh]
    pipe_list = [padel_transform, var_thresh]
    if transform_func is not None:
        smooth_transform = FunctionTransformer(
            func=transform_func,
            inverse_func=inv_transform,
            feature_names_out="one-to-one",
        ).set_output(transform="pandas")
        pipe_list.append(("smooth", smooth_transform))
    if corr_params is not None:
        xcorr_filter = XCorrFilter(
            max_features=None,
            thresh_xc=corr_params["thresh_xc"],
            method_corr=corr_params["corr_method"],
            method_xc=corr_params["xc_method"],
        )
        pipe_list.append(("xcorr", xcorr_filter))
    # combo_transform = Pipeline(steps=[("rings", ring_tranform), ("padel", padel_transform), ("var", var_thresh), ("scale", smooth_transform), ("xcorr", xcorr_filter)]).set_output(transform="pandas")
    combo_transform = Pipeline(steps=pipe_list).set_output(transform="pandas")
    return combo_transform, scaler


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
    if np.mean(scores) - np.std(scores) > selection_state["best_score_adj"]:
        logger.info(
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
        selection_state["fails"] = max(0, selection_state["fails"] - 3)
    else:
        selection_state["fails"] += 1
    return selection_state


def get_clf_model(model_name):
    if "log" in model_name:
        grove_model = LogisticRegressionCV(
            scoring=mcc,
            solver="newton-cholesky",
            tol=2e-4,
            cv=5,
            max_iter=10000,
            class_weight="balanced",
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


def score_drop_exceeded(
    new_scores, selection_params, selection_state, sequential_drop=True
):
    new_score = np.mean(new_scores) - np.std(new_scores)
    if new_score < selection_state["best_score_adj"] - selection_params["thresh_reset"]:
        print(
            "Score (adjusted) drop exceeded: {:.4f} {:.4f}".format(
                selection_state["best_score_adj"], new_score
            )
        )
        if sequential_drop:
            selection_state = sequential_backward_selection(selection_state)
        else:
            selection_state["current_features"] = selection_state["best_subset"]
        return True
    else:
        return False


def sequential_backward_selection(selection_state):
    selection_state["current_features"] = selection_state["best_subset"]
    return selection_state


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
        pickle.dump(trained_model, f)
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
    print(dropped_dict.items())
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
