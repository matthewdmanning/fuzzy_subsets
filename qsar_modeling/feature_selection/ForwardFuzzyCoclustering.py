import copy
import logging
import os
import pickle
import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy
from sklearn import clone, linear_model
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

import padel_categorization
import sample_clusters
import samples
import scoring
from build_preprocessor import get_standard_preprocessor
from correlation_filter import get_weighted_correlations
from epa_enamine_visualizer import _plot_proba_pairs
from vif import calculate_vif, repeated_stochastic_vif


def main():
    pass


if __name__ == "__main__":
    main()


def safe_mapper(x, map):
    if x in map.keys():
        return map[x]
    else:
        return x


def select_subsets_from_model(
    feature_df,
    labels,
    n_subsets,
    name_model_dict,
    label_corr,
    cross_corr,
    exp_dir,
    select_params,
):
    """
    Selects best scoring feature subsets from specified scikit-learn estimators.
    Parameters for feature selection are specified in select_params.

    Parameters
    ----------
    feature_df: DataFrame, feature descriptors
    labels: Series, Target labels
    n_subsets: int, max number of feature subsets to return
    name_model_dict: dict[str, BaseEstimator], Dictionary with name and unfitted estimator
    label_corr: Series, correlation of each descriptor with label, method type in select_params
    cross_corr: DataFrame, feature cross-correlations, method type in select_params
    exp_dir: str, path of directory to save files
    select_params: dict, Describes parameters for feature selection function

    Returns
    -------
    model_scores_dict: dict[str, list[float]], CV scores of each estimator, refit using chosen subsets.
    model_subsets_dict: dict[str, list[str]], list of names of chosen features for each estimator.
    subset_predicts: dict[str, Iterable[pd.Series | pd.DataFrame]], outputs from each model using its best-scoring feature subset.
    name_weight_dict: dict[str, list[pd.Series]] Dictionary of lists of sample weights used for each subset selection and scoring.
    """
    model_subsets_dict, model_scores_dict, subset_predicts, name_weights_dict = (
        dict(),
        dict(),
        dict(),
        dict(),
    )
    search_features = feature_df.columns.tolist()
    for n, m in name_model_dict.items():
        model_dir = "{}{}/".format(exp_dir, n)
        os.makedirs(model_dir, exist_ok=True)
        remaining_features = deepcopy(
            [
                x
                for x in search_features
                if all(
                    [
                        x in a.index.tolist()
                        for a in [feature_df.T, label_corr, cross_corr]
                    ]
                )
            ]
        )
        name_weights_dict[n] = list()
        model_subsets_dict[n] = list()
        model_scores_dict[n] = list()
        subset_predicts[n] = list()
        # Set default weighting values.
        select_params["feature_appearances"] = pd.Series(
            np.ones(shape=feature_df.columns.size), index=feature_df.columns
        ).to_frame()
        select_params["sample_weight"] = pd.Series(
            np.ones_like(labels.to_numpy()),
            dtype=np.float32,
            index=labels.index,
            name="sample_weight",
        )

        # Start subset selection loop
        for i in np.arange(n_subsets):
            # TODO: Compare i to len(blah_blah_dict[n]) to see if a fit is needed.

            print("Selecting from {} features.".format(len(remaining_features)))
            subset_dir = "{}subset{}/".format(model_dir, i)
            os.makedirs(subset_dir, exist_ok=True)
            cv_predicts_path = "{}{}_{}_output.pkl".format(subset_dir, n, i)
            cv_predicts_csv_path_stub = "{}{}_{}".format(subset_dir, n, i)
            weight_path = "{}{}_{}_weights.csv".format(subset_dir, n, i)
            best_features_path = "{}best_features_{}_{}.csv".format(subset_dir, n, i)
            # Retrieve previous modeling runs.
            if os.path.isfile(best_features_path):
                top_score = 0
                # TODO: Add sample weighting from boosting algorithm for first round.
                with open("{}feature_score_path.csv".format(subset_dir), "r") as f:
                    for whole_line in f.readlines():
                        line = whole_line.split()
                        scores = [float(x) for x in line[:5]]
                        feats = [str(x) for x in line[5:]]
                        if np.mean(scores) - np.std(scores) > top_score:
                            top_score = np.mean(scores) - np.std(scores)
                            best_features = feats
                            new_appearances = pd.Series(
                                np.zeros_like(feature_df.columns.to_numpy()),
                                index=feature_df.columns,
                                name=i,
                            )
                            new_appearances = new_appearances[best_features].add(1)
                            select_params["feature_appearances"].merge(
                                new_appearances, left_index=True, right_index=True
                            )
                if i > 0:
                    if os.path.isfile(weight_path):
                        """
                        with open(weight_path, "rb") as f:
                            name_weights_dict[n].append(pickle.load(f))
                        """
                        name_weights_dict[n].append(pd.read_csv(weight_path))
                    if os.path.isfile(cv_predicts_path):
                        with open(cv_predicts_path, "rb") as f:
                            subset_predicts[n].append(pickle.load(f))

                    # Update sample weights based on prediction results.
                    if len(subset_predicts[n]) == 1:
                        test_scores = pd.concat(
                            subset_predicts[n][0][select_params["score_func"]]["test"]
                        )
                    elif len(subset_predicts[n]) > 1:
                        test_scores = [
                            pd.concat(a[select_params["score_func"]]["test"])
                            for a in subset_predicts[n]
                        ]
                    else:
                        raise ValueError
                    if is_classifier(m):
                        new_weights = samples.weight_by_proba(
                            y_true=labels,
                            probs=test_scores,
                        )
                    elif is_regressor(m):
                        new_weights = samples.weight_by_error(
                            labels, test_scores, loss=select_params["loss_func"]
                        )
                    else:
                        raise ValueError
                    # else:
                    name_weights_dict[n].append(new_weights)
                    new_weights.to_csv(weight_path, index_label="INCHI_KEY")
                # Correlation matrices are recalculated with new weights for accurate feature selection.
                cross_corr, label_corr = get_weighted_correlations(
                    feature_df,
                    labels,
                    select_params,
                    subset_dir,
                )

                # OPTIMIZE: Pass this to train_multilabel_models instead of tacking onto label_corr?
                if False:
                    label_corr = label_corr.divide(
                        select_params["feature_appearances"].sum(axis=1) ** 2
                    )
                print(label_corr)
                print(label_corr.shape)
                print(cross_corr.shape)
                assert np.shape(label_corr)[0] == cross_corr.shape[1]

            if not os.path.isfile(best_features_path):
                # Run iteration of feature selection on model architecture.
                cv_results, cv_predicts, best_features = train_multilabel_models(
                    feature_df,
                    labels,
                    model=m,
                    model_name=n,
                    best_corrs=label_corr,
                    cross_corr=cross_corr,
                    select_params=select_params,
                    save_dir=subset_dir,
                )
                new_appearances = pd.Series(
                    np.zeros_like(feature_df.columns.to_numpy()),
                    index=feature_df.columns,
                    name=i,
                )
                new_appearances = new_appearances[best_features].add(1.0)
                select_params["feature_appearances"].merge(
                    new_appearances, left_index=True, right_index=True
                )
                subset_predicts[n].append(cv_predicts)
                for k, v in cv_predicts.items():
                    print(k)
                    cv_predicts_csv_path = "{}_{}.csv".format(
                        cv_predicts_csv_path_stub, k
                    )
                    cv_predicts_pkl_path = "{}_{}.pkl".format(
                        cv_predicts_csv_path_stub, k
                    )
                    cvp = pd.DataFrame.from_records(v)
                    cvp.to_csv(cv_predicts_csv_path)
                    with open(cv_predicts_pkl_path, "wb") as f:
                        pickle.dump(cvp, f)
                """
                pd.concat(cv_predicts["predict"]["test"]).to_csv(
                    cv_predicts_path,
                    index_label="INCHI_KEY",
                )
                """
                model_scores_dict[n].append(cv_results)
                # Use selected subsets to fit meta-estimator (ex. VotingClassifier)
                """                
                meta_estimator, meta_results, test_idx_list, select_params = (
                    fit_predict_metaclassifier(feature_df=feature_df, labels=labels, model=m, subset_i=i,
                                              model_subsets_dict=model_subsets_dict, model_name=n,
                                              name_model_dict=name_model_dict, select_params=select_params))
                """
                model_subsets_dict[n].append(best_features)
            # print(voter_results.items(), flush=True)
            # print(voter_results["predict_proba"]["test"])

            # cv_scores = pd.DataFrame.from_records(model_scores_dict, orient="index", columns=np.arange(n_subsets))
            # print(model_subsets_dict)
            # Plot sample predictions (probabilities) for each pair of models.
        # name_distplot_dict = plot_proba_distances(feature_df, labels, model_subsets_dict, name_model_dict)
        pg = _plot_proba_pairs(labels, n, subset_predicts, select_params)
        pg.savefig("{}pair_trial.png".format(model_dir), dpi=300)
        """
        density = fig.subplots(
            nrows=len(subset_predicts[n]),
            ncols=len(subset_predicts[n]),
            #  sharex=True,
            #  sharey=True,
            gridspec_kw={"hspace": 0.5, "wspace": 0.5},
        )
        # density = gs.subplots(sharex=True, sharey=True)
        for i_j in itertools.pairwise(np.arange(len(subset_predicts[n]) - 1)):
            resids_full = resid_df[i_j]
            # resids_sample = resids_full.groupby(["Ground"]).sample(frac=sample_fraction, replace=True)
            resids_sample = resids_full.loc[pred_df.index].squeeze()
            # print(resids_sample)
            print(pred_df)
            density[i_j].scatter(
                x=pred_df.iloc[:, i_j[0]].to_numpy(),
                y=pred_df.iloc[:, i_j[1]].to_numpy(),
                s=size[pred_df.index].to_numpy(),
                c=labels[pred_df.index].to_numpy(),
                cmap="YlGn",
                marker=".",
                alpha=alpha[pred_df.index].to_numpy(),
            )
            density[i_j].autoscale()
            # pair_plots.xlim(left=-0.05, right=1.05)
            density[(i_j[1], i_j[0])].scatter(
                x=pred_df.iloc[:, i_j[1]].to_numpy(),
                y=pred_df.iloc[:, i_j[0]].to_numpy(),
                s=size[pred_df.index].to_numpy(),
                c=labels[pred_df.index].to_numpy(),
                cmap="YlGn",
                marker=".",
                alpha=alpha[pred_df.index].to_numpy(),
            )
            # pair_plots.xlim(left=-0.05, right=1.05)
            density[(i_j[1], i_j[0])].autoscale()
        fig.savefig("{}{}_prob_dist_pairplot_residualsonly.png".format(exp_dir, n))
        for i, (c_name, p_col) in enumerate(pred_df.drop(columns="Ground").items()):

            print(p_col)
            for group in labels.unique():
                group_idx = labels[pred_df.index][labels[pred_df.index] == group].index
                x = p_col[group_idx].to_numpy()
                eval_points = np.linspace(np.min(x), np.max(x))
                if full_kde:
                    kde_sk = KernelDensity(bandwidth=1, kernel="gaussian")
                    kde_sk.fit(x.reshape([-1, 1]))
                    y_sk = np.exp(kde_sk.score_samples(eval_points.reshape(-1, 1)))
                    density[i, i] = pair_plots.plot(eval_points, data=y_sk)
                else:
                    density[i, i].hist(
                        x=x, bins=12, density=True, histtype="stepfilled"
                    )
        fig.savefig("{}{}_prob_dist_pairplot.png".format(exp_dir, n))

        pg = sns.PairGrid(data=pred_df, hue="Ground", layout_pad=0.1, corner=True)
        pg.map_lower(
            sns.scatterplot,
            data=pred_df,
            hue="Ground",
            size="Ground",
            legend=False,
            sizes=(4.0, 1.0),
        )  # , kwargs={"alpha": 0.05})
        # pg.map_upper(sns.residplot, kwargs={"size": 0.5, "plot_kws": {"s": 0.5, "alpha": 0.05}})
        pg.map_offdiag(sns.kdeplot)
        pg.map_diag(sns.kdeplot, common_norm=False)
        pg.fig.set_size_inches(15, 15)
        pg.fig.set(dpi=300)

        pg.savefig("{}{}_prob_dist_pairplot.png".format(exp_dir, n))
        name_distplot = sns.pairplot(data=pred_df, hue="Ground", size=0.5, height=11, plot_kws={"s": 0.5, "alpha": 0.05})
        name_distplot.map_diag(
            sns.kdeplot, hue="Ground", common_norm=True, lw=3, legend=False
        )
        name_distplot.savefig("{}{}_prob_dist_pairplot.png".format(exp_dir, n))
        """
    return model_scores_dict, model_subsets_dict, subset_predicts, name_weights_dict


def _update_sample_weights(
    feature_df,
    labels,
    model_name,
    proba_model,
    model_subsets_dict,
    i,
    select_params,
    subset_dir,
):
    # Calculate sample-weighting based on latest features.

    if i >= len(model_subsets_dict[model_name]) > 0:
        if len(model_subsets_dict[model_name]) == 1:
            new_probs = scoring.calculate_proba_from_model(
                proba_model,
                feature_df[model_subsets_dict[model_name][0]],
                labels,
                cv=select_params["cv"],
                select_params=select_params,
            )[0]
        else:
            results_list = scoring.calculate_proba_from_model(
                proba_model,
                feature_df[model_subsets_dict[model_name][0]],
                labels,
                cv=select_params["cv"],
            )
            new_probs = pd.concat(results_list).mean(axis=1)
        if is_classifier(proba_model):
            select_params["sample_weight"] = samples.weight_by_proba(
                y_true=labels, probs=new_probs
            )
        elif is_regressor(proba_model):
            select_params["sample_weight"] = samples.weight_by_error(
                y_true=labels, predicts=new_probs
            )
        select_params["sample_weight"].to_csv("{}sample_weights.csv".format(subset_dir))
    else:
        select_params["sample_weight"] = pd.Series(
            data=np.ones_like(labels), index=labels.index
        )

    assert (
        select_params["sample_weight"][select_params["sample_weight"] >= 0.000001].size
        > 0
    )
    return select_params


def fit_predict_metaclassifier(
    feature_df,
    labels,
    model,
    subset_i,
    model_subsets_dict,
    model_name,
    name_model_dict,
    select_params,
    stack_train_frac=0.5,
    name_weights_dict=None,
):

    # TODO: Store sample weights
    raise NotImplementedError
    if len(model_subsets_dict[model_name]) > 1:
        model_train_y, stack_train_y = train_test_split(
            labels,
            train_size=stack_train_frac,
            random_state=0,
            shuffle=True,
            stratify=labels,
        )

        estimator_list = [
            (
                str(subset_i),
                clone(model).fit(X=feature_df[feats], y=labels, sample_weight=weights),
            )
            for subset_i, (feats, weights) in enumerate(
                zip(model_subsets_dict[model_name], name_weights_dict[model_name])
            )
        ]
        voter = StackingClassifier(
            estimators=estimator_list,
            stack_method=select_params["score_func"],
            n_jobs=-2,
            verbose=1,
        )
        voter.fit(feature_df)
        """
        voter_results, test_idx_list = scoring.cv_model_generalized(
            estimator=voter,
            feature_df=feature_df.loc[stack_train_y.index],
            score_list=[select_params["score_func"]],
            labels=stack_train_y,
            cv=select_params["cv"],
            return_train=True,
            clone_model=False
        )
        """
        predicts_arr = voter.predict_proba(
            voter,
            X=feature_df.loc[stack_train_y.index],
            y=stack_train_y,
            n_jobs=-2,
            method=select_params["score_func"],
        )
        votes = pd.DataFrame(
            predicts_arr,
            index=stack_train_y.index,
            columns=stack_train_y.unique().sort_values(),
        )
        print("\n\nVoter results:")
        print(type(votes))
        print(votes)
        # print(voter_results[select_params["score_func"]]["test"])
        select_params["sample_weight"] = samples.weight_by_proba(labels, votes)
    elif len(model_subsets_dict[model_name]) == 1:
        voter = name_model_dict[model_name]
        voter_results, test_idx_list = scoring.cv_model_generalized(
            estimator=voter,
            feature_df=feature_df[model_subsets_dict[model_name][0]],
            labels=labels,
            cv=select_params["cv"],
            return_train=True,
        )
        select_params["sample_weight"] = samples.weight_by_proba(
            labels, voter_results["predict_proba"]["test"][0]
        )
    else:
        voter = StackingClassifier(estimators=[name_model_dict[model_name]])
        voter_results = None
        test_idx_list = list(pd.Series())
        print("No subsets found. Setting all sample weights equal.")
        select_params["sample_weight"] = pd.Series(
            np.ones_like(feature_df.columns.to_numpy()),
            index=feature_df.columns,
            name="sample_weight",
        )
    return FrozenEstimator(voter), voter_results, test_idx_list, select_params


def train_multilabel_models(
    feature_df,
    labels,
    model,
    model_name,
    best_corrs,
    cross_corr,
    select_params,
    save_dir,
    **kwargs,
):
    """
    Returns a set of features that maximizes the score function of the estimator. Iteratively adds features, sampled based on label correlation and correlation with other features.
    Uses VIF score to reduce multicollinearity amongst current feature set and SFS to eliminate uninformative or confounding features.

    Parameters
    ----------
    feature_df: pd.DataFrame
    labels: pd.Series
    model: pd.BaseEstimator
    model_name: str
    best_corrs: pd.Series
    cross_corr: pd.DataFrame
    select_params: dict
    save_dir: str

    Returns
    -------
    score_dict : dict[dict[list[pd.Series]]]
        Contains sample-wise scores for each CV fold
        {score_name: {split_name: list[pd.Series] | pd.Series}}
    cv_results : dict[dict[list[pd.Series]]]
    best_features
    """
    # if "n_jobs" in model.get_params():
    #    model.set_params(**{"n_jobs": 1})
    selection_models = {
        "predict": model,
        "permutation": model,
        "importance": model,
        "vif": linear_model.LinearRegression(),
    }
    model_dict, score_dict, dropped_dict, best_features = select_feature_subset(
        feature_df,
        labels,
        target_corr=best_corrs,
        cross_corr=cross_corr,
        select_params=select_params,
        selection_models=selection_models,
        save_dir=save_dir,
    )
    print("Best features!")
    short_to_long = padel_categorization.padel_convert_length()
    best_features_long = short_to_long[
        [f for f in best_features if f in short_to_long.index]
    ]
    if len(best_features_long) > 0:
        missing_long = [f for f in best_features if f not in short_to_long.index]
        best_features_long = pd.concat(
            [best_features_long, pd.Series(missing_long, index=missing_long)]
        )
    else:
        best_features_long = pd.Series(best_features, index=best_features)
    print("\n".join(best_features_long.tolist()))
    pd.Series(best_features_long).to_csv(
        "{}best_features_{}.csv".format(save_dir, model_name)
    )
    eval_pred = list()
    cv_results, test_idx_splits = scoring.cv_model_generalized(
        estimator=model,
        feature_df=feature_df[best_features],
        labels=labels,
        cv=select_params["cv"],
        return_train=True,
        sample_weight=select_params["sample_weight"],
    )
    print(cv_results)
    score_dict = scoring.score_cv_results(
        results_dict=cv_results,
        score_func_dict={select_params["score_name"]: select_params["scoring"]},
        y_true=labels,
        sample_weight=select_params["sample_weight"],
    )
    """
    for dev_df, dev_labels, eval_df, eval_labels in cv_tools.split_df(
        feature_df[best_features],
        labels,
        splitter=select_params["cv"],
    ):
        fitted_est = clone(model).fit(X=dev_df, y=dev_labels)
        eval_pred.append(
            pd.Series(
                fitted_est.predict(X=eval_df[best_features]), index=eval_df.index
            )
        )
    # cv_results = select_params["scorer"](
    cv_results = select_params["scorer"](
        fitted_est, dev_df[best_features], dev_labels
    )
    """
    # model.fit(train, cmd_train_labels)
    print("These are return types!!!")
    print(type(score_dict))
    print(type(cv_results))
    print(type(best_features))
    return score_dict, cv_results, best_features


mcc = make_scorer(balanced_accuracy_score)
logger = logging.getLogger(name="selection")


def _permutation_removal(train_df, labels, estimator, select_params, selection_state):
    """

    Selects feature for removal based on score drop after permutation of value.

    Parameters
    ----------
    train_df : pd.DataFrame
    labels : pd.Series
    estimator : BaseEstimator
    select_params : dict
    selection_state : dict

    Returns
    -------

    """
    # TODO: Consider switching to partial permutation to avoid tree-based bias towards high cardinality features.
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
            scoring=select_params["scorer"],
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


def current_score(state):
    scores = state["subset_scores"][tuple(state["current_features"])]
    return np.mean(scores) - np.std(scores)


def select_feature_subset(
    train_df,
    labels,
    target_corr,
    cross_corr,
    select_params,
    initial_subset=None,
    save_dir=None,
    selection_models=None,
    prior_best_probs=None,
):
    """

    Parameters
    ----------
    train_df : pd.DataFrame
    labels : pd.Series
    target_corr : pd.Series
    cross_corr : pd.DataFrame
    select_params : dict
    initial_subset
    save_dir : str
    selection_models : dict
    prior_best_probs : pd.DataFrame

    Returns
    -------

    """
    if (
        labels.nunique() == 2
        and (labels[labels == 1].size < 10 or labels[labels == 0].size < 10)
    ) or labels.nunique == 1:
        raise ValueError
    if select_params["cv"] is None:
        select_params["cv"] = RepeatedStratifiedKFold(random_state=0, n_repeats=5)
    if select_params["scoring"] is None:
        select_params["scoring"] = balanced_accuracy_score
    if select_params["scorer"] is None:
        select_params["scorer"] = make_scorer(select_params["scoring"])
    with open("{}selection_params.txt".format(save_dir), "w") as f:
        for k, v in select_params.items():
            try:
                f.write("{}: {}\n".format(k, v))
            except AttributeError:
                pass
    print(target_corr)
    if initial_subset is None:
        initial_subset = (
            target_corr.index.to_series()
            .sample(weights=target_corr.abs(), n=1)
            .tolist()
        )
        print("\n\nInitial subset: {}".format(initial_subset))
    if prior_best_probs is None:
        prior_best_probs = pd.DataFrame(
            data=0.5, columns=labels.unique(), index=labels.index
        )
    selection_state = {
        "prior_best": prior_best_probs,
        "current_features": initial_subset,
        "subset_scores": dict(),
        "best_subset": list(),
        "rejected_features": dict(),
        "best_score_adj": -999,
        "previous_best_score": -999,
        "fails": 0,
        "attempts": 0,
    }
    selection_state, scores, improvement = score_subset(
        feature_df=train_df,
        labels=labels,
        selection_models=selection_models,
        selection_state=selection_state,
        select_params=select_params,
        save_dir=save_dir,
        subset=initial_subset,
        record_results=True,
    )
    sqcc_df = cross_corr * cross_corr
    clean_up = False
    print("Matrix shapes.")
    print(train_df.shape, sqcc_df.shape, target_corr.shape)
    # Start feature loop
    for i in np.arange(select_params["max_trials"]):
        # print("Feature_df shape: {}".format(train_df.shape))
        # print(len(selection_state["current_features"]))
        # TODO: Does this check need to be here? Especially with the new weighting for reused features
        maxed_out = (
            len(selection_state["current_features"])
            >= select_params["max_features_out"]
        )
        above_min = (
            len(selection_state["current_features"])
            >= select_params["features_min_sfs"]
        )
        if maxed_out or (above_min and clean_up):
            # or train_df.shape[1]
            # - len([selection_state["rejected_features"].keys()])
            # - len(selection_state["current_features"])
            # < select_params["n_vif_choices"]
            print(selection_state["current_features"])
            # Keep eliminating until sequential elimination is within range of going under score_exceeded limit.
            while _over_sfs_thresh(select_params, selection_state):
                print(current_score(selection_state), selection_state["best_score_adj"])
                original_size = len(selection_state["current_features"])
                selection_state, subset_scores = sequential_elimination(
                    train_df=train_df,
                    labels=labels,
                    select_params=select_params,
                    selection_state=selection_state,
                    selection_models=selection_models,
                    clean_up=True,
                    save_dir=save_dir,
                    randomize=True,
                )
                if original_size == len(selection_state["current_features"]):
                    clean_up = False
                    break
            continue
        else:
            clean_up = False
        """
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
        """
        new_feat, selection_state = choose_next_feature(
            feature_df=train_df,
            feature_list=selection_state["current_features"],
            target_corr=target_corr,
            sq_xcorr=sqcc_df,
            selection_models=selection_models,
            selection_state=selection_state,
            select_params=select_params,
        )
        if new_feat is None:
            break
        else:
            selection_state, subset_scores, score_improve = score_subset(
                feature_df=train_df,
                labels=labels,
                selection_models=selection_models,
                selection_state=selection_state,
                select_params=select_params,
                save_dir=save_dir,
                record_results=True,
            )
        if score_improve is not None:
            subset_metric = score_improve
        else:
            subset_metric = subset_scores
        print(score_improve, subset_scores, selection_state["current_features"])
        # Check if score has dropped too much.
        exceeded, selection_state = score_drop_exceeded(
            subset_metric,
            selection_params=select_params,
            selection_state=selection_state,
        )
        if exceeded:
            while (
                np.mean(subset_metric) + np.std(subset_metric)
                < selection_state["best_score_adj"]
            ):
                if (
                    len(selection_state["current_features"])
                    <= select_params["features_min_sfs"]
                ):
                    selection_state["current_features"] = copy.deepcopy(
                        selection_state["best_subset"]
                    )
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
        if False and (
            len(selection_state["current_features"])
            >= select_params["features_min_vif"]
        ):
            selection_state = _vif_elimination(
                train_df=train_df,
                best_corrs=target_corr,
                cross_corr=cross_corr,
                select_params=select_params,
                selection_state=selection_state,
            )
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
        while (
            len(selection_state["current_features"])
            >= select_params["features_min_sfs"]
        ) or clean_up:
            if (
                len(selection_state["current_features"])
                >= select_params["max_features_out"]
            ):
                clean_up = True
            # DEBUG
            if (
                len(selection_state["current_features"])
                <= select_params["min_features_out"]
            ):
                break
            n_features_in = copy.deepcopy(len(selection_state["current_features"]))
            selection_state, subset_scores, score_improve = sequential_elimination(
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
            too_much, selection_state = score_drop_exceeded(
                new_scores=score_improve,
                selection_params=select_params,
                selection_state=selection_state,
            )
            if too_much or n_features_in == len(selection_state["current_features"]):
                clean_up = False
                break
    print(
        "Best adjusted score of {} with feature set: {}".format(
            selection_state["best_score_adj"],
            selection_state["best_subset"],
        )
    )
    # TODO: Check if this is even possible to fail.
    if len(selection_state["best_subset"]) > 0:
        best_fit_model = FrozenEstimator(
            selection_models["predict"].fit(
                X=train_df[selection_state["best_subset"]], y=labels
            )
        )
        with open("{}best_model.pkl".format(save_dir), "wb") as f:
            pickle.dump(best_fit_model, f)
    else:
        print(selection_state["best_subset"])
        raise RuntimeError
    """
    print("Rejects: {}".format(selection_state["rejected_features"]))
    pd.Series(selection_state["rejected_features"], name="Dropped Features").to_csv(
        "{}dropped_features.csv".format(save_dir)
    )
    """
    return (
        selection_models["predict"],
        selection_state["subset_scores"],
        selection_state["rejected_features"],
        selection_state["best_subset"],
    )


def _over_sfs_thresh(
    select_params, selection_state, scores="current", set_size=None, factor="sfs"
):
    # Returns boolean of whether current score is within some amount of the best score.
    # Fewer features = Lower overmax = Greater adjust = Easier to pass
    if "sfs" in factor:
        factor = select_params["thresh_sfs"]
    elif "reset" in factor:
        factor = select_params["thresh_reset"]
    elif "cleanup" in factor:
        factor = select_params["thresh_sfs_cleanup"]
    reference = selection_state["best_score_adj"]
    if isinstance(scores, str) and "current" in scores:
        scores = current_score(selection_state)
        set_size = len(selection_state["current_features"])
    overmax = select_params["max_features_out"] / set_size
    adjust = 1 - factor * np.log(overmax)
    print(len(selection_state["current_features"]), adjust)
    return scores >= reference * adjust


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
        clean = "thresh_sfs_cleanup"
    else:
        clean = "thresh_sfs"
    # TODO: Implement configurable predict function for boosting.
    sfs_score_dict = dict().fromkeys(selection_state["current_features"], list())
    current_score_adj = copy.deepcopy(current_score(selection_state))
    for left_out, score_list in sfs_score_dict.items():
        new_subset = copy.deepcopy(selection_state["current_features"])
        new_subset.remove(left_out)
        selection_state, scores, score_improve = score_subset(
            train_df,
            labels,
            selection_models=selection_models,
            selection_state=selection_state,
            select_params=select_params,
            save_dir=save_dir,
            subset=new_subset,
            record_results=True,
        )
        if score_improve is not None:
            sfs_score_dict[left_out] = score_improve
        else:
            sfs_score_dict[left_out] = scores
    if len(sfs_score_dict.items()) == 0:
        print("No valid SFS scores returned.")
        raise RuntimeWarning
        subset_scores = selection_state["subset_scores"][
            tuple(selection_state["current_features"])
        ]
    else:
        worst_feature_tup = sorted(
            list(sfs_score_dict.items()), key=lambda x: (np.mean(x[1] - np.std(x[1])))
        )[0]
        drop_scores = np.mean(worst_feature_tup[1]) - np.std(worst_feature_tup[1])
        if _over_sfs_thresh(select_params, selection_state, drop_scores, factor=clean):
            subset_scores = worst_feature_tup[1]
            selection_state["current_features"].remove(worst_feature_tup[0])
        else:
            subset_scores = selection_state["subset_scores"][
                tuple(selection_state["current_features"])
            ]
    """
    sfs = (
        SequentialFeatureSelector(
            estimator=clone(selection_models["predict"]),
            direction="backward",
            tol=sfs_tol,
            n_features_to_select=len(selection_state["current_features"]) - 1,
            scoring=select_params["scoring"],
        )
        .set_output(transform="pandas")
        .fit(train_df[selection_state["current_features"]], y=labels)    )
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
    return selection_state, subset_scores, score_improve


def relative_brier_score(
    y_true,
    y_proba,
    pos_label=None,
    clips=(0, 1.0),
    y_prior=None,
    sample_weight=None,
    class_weight="balanced",
    decision_thresholds=None,
):
    onehot_labels, onehot_normed = samples.one_hot_conversion(
        y_true, threshold=decision_thresholds
    )
    y_proba.clip(lower=clips[0], upper=clips[1], inplace=True)
    y_proba_brier = (
        y_proba.sub(onehot_normed).abs().multiply(onehot_labels).sum(axis=1).squeeze()
    )
    if y_prior is not None:
        y_prior_brier = (
            y_prior.sub(onehot_normed)
            .abs()
            .multiply(onehot_labels)
            .sum(axis=1)
            .squeeze()
        )
        y_proba_brier = y_proba_brier - y_prior_brier
    y_proba_brier.clip(lower=0)
    brier_sq = y_proba_brier**2
    if sample_weight is None:
        rel_brier_score = brier_sq.sum() / brier_sq.shape[0]
    else:
        rel_brier_score = (sample_weight * brier_sq).sum() / sample_weight.sum()
    return rel_brier_score, y_proba_brier


def score_subset(
    feature_df,
    labels,
    selection_models,
    selection_state,
    select_params,
    save_dir,
    subset=None,
    record_results=False,
    sample_weight=None,
    class_weight="balanced",
):
    if subset is None or len(subset) == 0:
        subset = tuple(sorted(selection_state["current_features"]))
    elif len(selection_state["current_features"]) == 0:
        print("No features in current subset!")
        print(selection_state["current_features"], flush=True)
        raise ValueError
    if isinstance(subset, str):
        # raise KeyError
        current_features = [copy.deepcopy(subset)]
    else:
        current_features = tuple(sorted(copy.deepcopy(subset)))
    assert len(current_features) > 0
    score_tuple = [(select_params["score_name"], select_params["scoring"])]
    scores = None
    rel_brier_score = None
    for prior_set in selection_state["subset_scores"].keys():
        if len(set(current_features).symmetric_difference(prior_set)) == 0:
            scores = selection_state["subset_scores"][current_features]
            # print("Duplicate scoring found:\n{}\n".format(current_features, prior_set))
            break
    if scores is None:
        selection_state["subset_scores"][current_features] = list()
        # best_corrs, cross_corr = get_correlations(train_df, train_labels, path_dict["corr_path"], path_dict["xc_path"], select_params["corr_method"], select_params["xc_method"])
        if "sample_weight" in select_params.keys() or sample_weight is not None:
            if sample_weight is not None:
                weights = sample_weight
            else:
                weights = select_params["sample_weight"]
            results, test_idx_list = scoring.cv_model_generalized(
                estimator=selection_models["predict"],
                feature_df=feature_df[list(current_features)],
                labels=labels,
                cv=select_params["cv"],
                sample_weight=weights,
            )
            scores = scoring.score_cv_results(
                results,
                dict(score_tuple),
                y_true=labels,
                score_kwargs={"sample_weight": weights},
            )["Score"].tolist()
        else:
            results, test_idx_list = scoring.cv_model_generalized(
                estimator=selection_models["predict"],
                feature_df=feature_df[list(current_features)],
                labels=labels,
                cv=select_params["cv"],
            )
            scores = scoring.score_cv_results(
                results, dict(score_tuple), y_true=labels, **select_params
            )["Score"].tolist()
        if is_classifier(selection_models["predict"]):
            rel_brier_score, rel_brier = relative_brier_score(
                labels,
                pd.concat(results["predict_proba"]["test"]),
                pos_label=0,
                # clips=(0.2, 0.8),
                y_prior=selection_state["prior_best"],
            )
            print("Relative Brier Score: {}".format(rel_brier_score))
            selection_state["subset_scores"][
                tuple(sorted(current_features))
            ] = rel_brier_score
            _compare_to_best(rel_brier_score, selection_state)
        else:
            selection_state["subset_scores"][tuple(sorted(current_features))] = scores
            _compare_to_best(scores, selection_state)
        if record_results:
            record_score(
                selection_state=selection_state,
                scores=scores,
                save_dir=save_dir,
            )
    # print(np.mean(scores))
    return selection_state, scores, rel_brier_score


def choose_next_feature(
    feature_df,
    feature_list,
    target_corr,
    sq_xcorr,
    selection_models,
    selection_state,
    select_params,
    vif_choice=5,
):
    previous_features = set()
    selection_state["attempts"] += 1
    # OPTIMIZE: Use weighting from appearance in previous subsets (weighted by number of other additional features) to decrease repeat subsets.
    if len(selection_state["subset_scores"]) > 1:
        for subset in selection_state["subset_scores"].keys():
            sub_diff = set(subset).difference(selection_state["current_features"])
            if 0 < len(sub_diff) < 4:
                previous_features.update(sub_diff)
        feat_corrs = target_corr.drop(
            index=[
                c
                for c in target_corr.index
                if (c in feature_list or c in previous_features)
            ]
        )
    else:
        feat_corrs = target_corr
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
        sum_sqcc = (
            ones.subtract(sq_xcorr[feature_list].loc[feat_corrs.index])
            .sum(axis=1)
            .squeeze()
        )
    x = sum_sqcc.multiply(other=np.abs(feat_corrs), fill_value=0.0)
    feat_probs = pd.Series(scipy.special.softmax(x), index=x.index).sort_values(
        ascending=False
    )
    feat_probs.drop(index=tuple(previous_features), inplace=True, errors="ignore")
    # feature_list = list(set(feature_list))
    # noinspection PyTypeChecker
    # TODO: Clean up this logic. Purpose: Avoid string splitting. Need to get better type checking.
    if len(feature_list) <= select_params["add_n_feats"]:
        vif_choice = len(feature_list)
        new_feat = feature_list
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
        else:
            new_feat = [new_feat]
    else:
        new_feat = [
            x for x in new_feat if x in sq_xcorr.columns and x in feature_df.columns
        ]
    if (
        np.size(new_feat) > select_params["add_n_feats"]
        or len(new_feat) > select_params["add_n_feats"]
    ):
        vifs = pd.Series()
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
        under_thresh = vifs[vifs < select_params["thresh_vif"]]
        if under_thresh.shape[0] < select_params["add_n_feats"]:
            under_thresh = vifs.sort_values().iloc[
                : min(vifs.shape[0], select_params["add_n_feats"])
            ]
        vif_selected = under_thresh.sample(
            n=min(under_thresh.size, select_params["add_n_feats"]),
            weights=1 / under_thresh,
        ).index.tolist()
    else:
        vif_selected = new_feat[: select_params["add_n_feats"]]
    if isinstance(vif_selected, str):
        vif_selected = [vif_selected]
    if len(selection_state["current_features"]) == 0:
        selection_state["current_features"] = vif_selected
    else:
        selection_state["current_features"].extend(vif_selected)
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
    raise DeprecationWarning
    # TODO: Delete function.
    preloaded = False
    drop_corr = False
    best_corrs, cross_corr = None, None
    if feature_df is None:
        feature_df, labels = sample_clusters.grab_enamine_data()
    if dropped_dict is None:
        dropped_dict = dict()
    combo_transform, scaler, cross_corr = get_standard_preprocessor(
        scaler=scaler, transform_func=transform, corr_params=select_params
    )
    combo_transform.fit(X=feature_df, y=labels)
    with open("{}transformer.pkl".format(save_dir), "wb") as f:
        pickle.dump(combo_transform, f)
    train_df = combo_transform.transform(feature_df)
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
    return


# TODO: Add complexity penalty here?
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
    return selection_state, best_yet


def score_drop_exceeded(new_scores, selection_params, selection_state):
    new_score = np.mean(new_scores) - np.std(new_scores)
    if _over_sfs_thresh(
        selection_params, selection_state, scores=new_score, factor="reset"
    ):
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
