import copy
import copy
import logging
import os
import pickle
import pprint
from math import asinh

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import (
    cross_val_score,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import clone
from sklearn.preprocessing import RobustScaler, StandardScaler

import sample_clusters
from correlation_filter import find_correlation
from grove_feature_selection import get_search_features
from vif import calculate_vif, repeated_stochastic_vif

mcc = make_scorer(balanced_accuracy_score)


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
        selection_state["best_score_adj"].update([(low_feats, "Importance")])
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
        selection_state["best_score_adj"].update([(vif_dropped, "VIF")])
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


def grove_features_loop(
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
        select_params["cv"] = RepeatedStratifiedKFold(random_state=0, n_repeats=5)
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
    if labels.nunique() == 2 and (
        labels[labels == 1].size < 10 or labels[labels == 0].size < 10
    ):
        raise ValueError
    # PCA
    calculate_pca(save_dir, train_df)

    pprint.pp(selection_state["rejected_features"].items())
    sqcc_df = cross_corr * cross_corr
    # Start feature loop
    for i in np.arange(select_params["max_trials"]):
        if not (
            len(selection_state["current_features"]) < select_params["max_features_out"]
            and target_corr.shape[0]
            - len([selection_state["rejected_features"].keys()])
            - len(selection_state["current_features"])
            > 2
        ):
            print("Out of features to pick")
            break
        clean_up = False
        new_feat = choose_next_feature(
            train_df,
            selection_state["current_features"],
            target_corr,
            sqcc_df,
            selection_models,
            selection_state=selection_state,
        )
        selection_state["current_features"].append(new_feat)
        selection_state, scores = score_subset(
            train_df,
            labels,
            selection_models,
            selection_state,
            select_params,
            save_dir,
            hidden_test,
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
        while selection_state["fails"] >= select_params[
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
                # print("RFECV Mean Test Score: {}".format(rfe.cv_results_["mean_test_score"]))
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
        while (
            selection_state["fails"] >= select_params["fails_min_sfs"]
            and select_params["features_min_sfs"]
            < len(selection_state["current_features"])
            or (clean_up and len(selection_state["current_features"]) > 3)
        ):
            sfs_tol = select_params["thresh_sfs"]
            if clean_up:
                sfs_tol = 2 * select_params["thresh_sfs"]
            sfs = (
                SequentialFeatureSelector(
                    estimator=clone(selection_models["predict"]),
                    direction="backward",
                    tol=sfs_tol,
                    n_features_to_select=len(selection_state["current_features"]) - 1,
                    scoring=select_params["scoring"],
                    n_jobs=-1,
                )
                .set_output(transform="pandas")
                .fit(train_df[selection_state["current_features"]], y=labels)
            )
            new_features = list(
                sfs.get_feature_names_out(selection_state["current_features"])
            )
            selection_state["current_features"] = new_features
            selection_state, subset_scores = score_subset(
                train_df,
                labels,
                selection_models,
                selection_state,
                select_params,
                save_dir,
                hidden_test,
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
                selection_state["rejected_features"].update(
                    [(c, "SFS") for c in sfs_drops]
                )
            elif score_drop_exceeded(subset_scores, select_params, selection_state):
                if (
                    len(selection_state["current_features"])
                    == select_params["max_features_out"]
                ):
                    selection_state["fails"] = 0
                    break
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
    else:
        with open("{}best_model.pkl".format(save_dir), "wb") as f:
            pickle.dump(selection_models["predict"], f)
    pd.Series(selection_state["rejected_features"], name="Dropped Features").to_csv(
        "{}dropped_features.csv".format(save_dir)
    )
    return (
        selection_models["predict"],
        selection_state["subset_scores"],
        selection_state["rejected_features"],
        selection_state["best_subset"],
    )


def score_subset(
    train_df,
    labels,
    selection_models,
    selection_state,
    select_params,
    save_dir,
    hidden_test=None,
):
    print(selection_state["current_features"])
    scores = cross_val_score(
        estimator=clone(selection_models["predict"]),
        X=train_df[selection_state["current_features"]],
        y=labels,
        scoring=select_params["scoring"],
        cv=select_params["cv"],
        n_jobs=-1,
        error_score="raise",
    )
    if hidden_test is not None:
        trained_model = clone(selection_models["predict"]).fit(
            train_df[selection_state["current_features"]], labels
        )
        test_score = trained_model.score(
            hidden_test[0][selection_state["current_features"]], hidden_test[1]
        )
        print("Test Score: {}".format(test_score))
    else:
        test_score = None
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
    feat_corrs = target_corr.drop(
        [
            c
            for c in target_corr.index
            if c in selection_state["rejected_features"].keys()
            or c in selection_state["current_features"]
        ]
    )
    sum_sqcc = (1 - sq_xcorr[feature_list].loc[feat_corrs.index]).sum(axis=1)
    feat_probs = scipy.special.softmax(np.abs(feat_corrs) * sum_sqcc)
    # feature_list = list(set(feature_list))
    # noinspection PyTypeChecker
    new_feat = np.random.choice(
        a=feat_corrs.index.to_numpy(), size=vif_choice, replace=False, p=feat_probs
    )
    if np.size(new_feat) > 1:
        vifs = dict()
        for nf in new_feat:
            vifs = calculate_vif(
                feature_df=feature_df[feature_list + [nf]],
                model=clone(selection_models["vif"]),
                subset=feature_df[[nf]],
            )
        vifs = dict(sorted(vifs.items(), key=lambda x: x[1]))
        new_feat = list(vifs.keys())[0]
    if new_feat not in sq_xcorr.columns:
        print("{} not in cross_corr".format(new_feat))
        raise RuntimeWarning("New feature is not in cross-correlation matrix")
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
    if dropped_dict is None:
        dropped_dict = dict()
    if scaler is not None and scaler == "standard":
        scaler = StandardScaler().set_output(transform="pandas")
    elif scaler is not None and scaler == "robust":
        scaler = RobustScaler(unit_variance=True).set_output(transform="pandas")
    if transform is not None and transform == "asinh":
        transform = asinh
    if save_dir is not None and os.path.isdir(save_dir):
        print(save_dir)
        data_df_path = "{}preprocessed_feature_df.pkl".format(save_dir)
        label_path = "{}member_labels.csv".format(save_dir)
        xc_path = "{}cross_corr.csv".format(save_dir)
        corr_path = "{}target_corr.csv".format(save_dir)
        """        

        """
    if feature_df is None:
        if os.path.isfile(data_df_path):
            train_df = pd.read_pickle(data_df_path)
            drop_corr = False
            scaler = None
            transform = None
        if os.path.isfile(label_path):
            labels = pd.read_csv(label_path, index_col="ID").squeeze().ravel()
            # labels = labels[~labels.index.duplicated()]
        if os.path.isfile(corr_path):
            best_corrs = pd.read_csv(corr_path)
            best_corrs = (
                best_corrs.set_index(keys=best_corrs.columns[0]).squeeze().copy()
            )
        if os.path.isfile(xc_path):
            cross_corr = pd.read_csv(corr_path)
            cross_corr = (
                cross_corr.set_index(keys=cross_corr.columns[0]).squeeze().copy()
            )
        save_files = False
    elif feature_df is not None and type(feature_df) is pd.DataFrame:
        save_files = True
        if transform is not None:
            train_df = feature_df.map(transform, na_action="ignore")
        else:
            train_df = feature_df.copy()
        if scaler is not None:
            train_df = scaler.fit_transform(train_df)
        drop_corr = True
        train_df = train_df[~train_df.index.duplicated()]
        if select_params is not None and "corr_method" in select_params.keys():
            best_corrs = train_df.corrwith(
                labels, method=select_params["corr_method"]
            ).sort_values(ascending=False)
        else:
            best_corrs = train_df.corrwith(labels).sort_values(ascending=False)
        if select_params is not None and "xc_method" in select_params.keys():
            cross_corr = train_df.corr(method=select_params["xc_method"])
        else:
            cross_corr = train_df.corr()
    train_df.dropna(axis=1, inplace=True)
    labels = labels[train_df.index]
    logger.debug(labels.shape, train_df.shape)
    logger.debug(train_df.isna().astype(int).sum(axis=0).sort_values(ascending=False))
    """
    else:
        target_corr, cross_corr, train_df, scaler = process_selection_data(
            dropped_dict, feature_df, labels, select_params=select_params
        )
        with open("{}scaler.pkl".format(save_dir), "wb") as f:
            pickle.dump(scaler, f)
    """

    # sample_wts = compute_sample_weight(class_weight="balanced", y=labels)
    if drop_corr:
        del_ser = find_correlation(
            cross_corr,
            cutoff=select_params["thresh_xc"],
            n_drop=max(1, train_df.shape[1] - select_params["max_features_out"]),
        )
        logger.debug([c for c in del_ser.index if type(c) is not str])
        train_df.drop(columns=del_ser.index, inplace=True)
        best_corrs.drop(del_ser.index, inplace=True)
        cross_corr = cross_corr.drop(columns=del_ser.index).drop(index=del_ser.index)
        dropped_dict.update([(c, "Cross-correlation") for c in del_ser.index])

        na_corrs = best_corrs.index[best_corrs.isna()]
        [dropped_dict.update([(c, "NA Correlation")]) for c in na_corrs]
        best_corrs.drop(na_corrs, inplace=True)
        cross_corr = cross_corr.drop(index=na_corrs).drop(columns=na_corrs)
        train_df.drop(columns=na_corrs, inplace=True)
    best_corrs.sort_values(ascending=False, inplace=True, key=lambda x: np.abs(x))
    if os.path.isdir(save_dir) and save_files:
        best_corrs.to_csv(corr_path)
        cross_corr.to_csv(xc_path)
        train_df.to_pickle(data_df_path)
        labels.to_csv(label_path)
    assert train_df.index.equals(labels.index)
    return train_df, labels, best_corrs, cross_corr, scaler


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
        tuple(selection_state["current_features"])
    ] = scores
    if np.mean(scores) - np.std(scores) > selection_state["best_score_adj"]:
        logger.info(
            "New top results for {} feature model: {:.4f}, {:.4f}".format(
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
            n_jobs=-1,
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
        grove_model = PassiveAggressiveClassifier(C=5.0, class_weight="balanced", n_jobs=-1, random_state=0)
    elif "xtra" in model_name:
        grove_model = ExtraTreesClassifier(
            n_jobs=-1,
            max_leaf_nodes=200,
            min_impurity_decrease=0.005,
            max_depth=30,
            class_weight="balanced",
            bootstrap=False,
        )
    else:
        grove_model = RandomForestClassifier(
            n_jobs=-1,
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

        grove_model = LinearRegression(n_jobs=-1)
    elif "elastic" in model_name:
        from sklearn.linear_model import ElasticNetCV

        grove_model = ElasticNetCV(
            l1_ratio=[0.25, 0.5, 0.75, 0.9],
            tol=1e-4,
            max_iter=10000,
            n_jobs=-1,
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

        grove_model = LassoCV(n_jobs=-1, random_state=0)
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
            n_jobs=-1,
            max_leaf_nodes=200,
            min_impurity_decrease=0.005,
            max_depth=30,
            bootstrap=False,
        )
    return grove_model


def score_drop_exceeded(new_scores, selection_params, selection_state):
    new_score = np.mean(new_scores) - np.std(new_scores)
    print(new_score)
    if new_score < selection_state["best_score_adj"] - selection_params["thresh_reset"]:
        return True
    else:
        return False


# def drop_feature(selection_state):


def main(model_name):
    feature_df, labels, grove_cols, kurtosis_stats = sample_clusters.main()
    # data_dir = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/test_data/Vapor pressure OPERA/Vapor pressure OPERA/"
    # opera_dir = "{}test_train_split/".format(data_dir)
    data_dir = "{}enamine_feat_selection_12-17-24/".format(os.environ.get("MODEL_DIR"))
    opera_dir = data_dir
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
    search_dir = "{}test_train_split/{}_asinh_1/".format(opera_dir, model_name)
    print(search_dir)
    os.makedirs(search_dir, exist_ok=True)
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
    reg_models = {
        "predict": get_regression_model(model_name),
        "permutation": get_regression_model(model_name),
        "importance": get_regression_model("lasso"),
        "vif": LinearRegression(n_jobs=-1),
    }
    clf_models = {
        "predict": get_clf_model(model_name),
        "permutation": get_clf_model(model_name),
        "importance": get_clf_model("rfc"),
        "vif": LinearRegression(n_jobs=-1),
    }
    search_features = get_search_features(feature_df, included=grove_cols)
    train_data, labels, best_corrs, cross_corr, scaler = process_selection_data(
        feature_df=feature_df[search_features].copy(),
        labels=labels,
        save_dir=data_dir,
        select_params=select_params,
        transform="asinh",
    )
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
    model_dict, score_dict, dropped_dict, best_features = grove_features_loop(
        train_df=dev_data,
        labels=dev_labels,
        target_corr=best_corrs,
        cross_corr=cross_corr,
        select_params=select_params,
        selection_models=clf_models,
        hidden_test=(eval_data, eval_labels),
        save_dir=search_dir,
    )
    if any([a is None for a in [model_dict, score_dict, dropped_dict, best_features]]):
        print(model_dict, score_dict, dropped_dict, best_features)
        raise ValueError
    trained_model = clone(clf_models["predict"]).fit(
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
                grove_features_loop(
                    feature_df.loc[members][search_features],
                    labels[members],
                    None,
                    None,
                    select_params=None,
                    initial_subset=col,
                    save_dir=col_dir,
                )
            )


if __name__ == "__main__":
    global logger
    logger = logging.getLogger()
    for md in ["passive", "ridge"]:
        main(model_name=md)
