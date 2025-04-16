import copy
import itertools
import logging
import os
import pickle
import pprint
import random

import numpy as np
import pandas as pd
import scipy
import stats
from sklearn import clone, linear_model
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

import math_tools
import padel_categorization
import sample_clusters
import samples
import scoring
from build_preprocessor import get_standard_preprocessor
from correlation_filter import get_weighted_correlations
from epa_enamine_visualizer import _plot_proba_pairs
from FuzzyAnnealer import FuzzyAnnealer
from math_tools import complexity_penalty, scaled_softmax
from scoring import relative_brier_score
from vif import calculate_vif, repeated_stochastic_vif


def main():
    pass


if __name__ == "__main__":
    main()


def select_subsets_from_model(
    feature_df,
    labels,
    n_subsets,
    name_model_dict,
    label_corr,
    cross_corr,
    exp_dir,
    select_params,
    prior_probs,
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
        remaining_features = list()
        for x in search_features:
            for a in [feature_df.T, label_corr, cross_corr]:
                if all([x in a.index]):
                    remaining_features.append(x)
        print("Selecting from {} features.".format(len(remaining_features)))
        name_weights_dict[n] = list()
        model_subsets_dict[n] = list()
        model_scores_dict[n] = list()
        subset_predicts[n] = list()
        # Set default weighting values.
        if (
            select_params["sample_weight"] is None
            or select_params["sample_weight"].sum() == 0
        ):
            select_params["sample_weight"] = pd.Series(
                np.ones_like(labels.to_numpy()),
                dtype=np.float32,
                index=labels.index,
                name="sample_weight",
            )
        # Start subset selection loop
        for i in np.arange(n_subsets):
            # TODO: Compare i to len(blah_blah_dict[n]) to see if a fit is needed.
            subset_dir = "{}subset{}/".format(model_dir, i)
            os.makedirs(subset_dir, exist_ok=True)
            cv_predicts_path = "{}{}_{}_output.pkl".format(subset_dir, n, i)
            cv_predicts_csv_path_stub = "{}{}_{}".format(subset_dir, n, i)
            best_features_path = "{}best_features_{}_{}.csv".format(subset_dir, n, i)
            weighted_label_path = "{}weighted_label_corr.csv".format(subset_dir)
            weighted_cross_path = "{}weighted_cross_corr.csv".format(subset_dir)
            # Retrieve previous modeling runs.
            score_name = "predict_proba"
            r_name = "Original"
            cv_predicts_pkl_path = "{}_{}_{}.pkl".format(
                cv_predicts_csv_path_stub, score_name, r_name
            )
            if all(
                [
                    os.path.isfile(p)
                    for p in [
                        best_features_path,
                        cv_predicts_path,
                        cv_predicts_pkl_path,
                    ]
                ]
            ):
                # if i > 0:
                cv_predicts_csv_path = "{}_{}_{}.csv".format(
                    cv_predicts_csv_path_stub, score_name, r_name
                )

                with open(cv_predicts_path, "rb") as f:
                    cv_predicts = pickle.load(f)
                # Update sample weights based on prediction results.
                for k, v in cv_predicts.items():
                    for w, z in v.items():
                        if not isinstance(z, (pd.Series, pd.DataFrame)):
                            k[v][w] = pd.DataFrame(z).squeeze()
                subset_predicts[n].append(cv_predicts)
                new_weights = weights_from_predicts(
                    labels, subset_predicts[n], m, select_params
                )
                name_weights_dict[n].append(new_weights)
                if is_classifier(m):
                    prior_probs.append(pd.concat(cv_predicts["predict_proba"]["test"]))
                elif is_classifier(m):
                    prior_probs.append(
                        pd.concat(cv_predicts["predict"]["test"])
                    )  # Correlation matrices are recalculated with new weights for accurate feature selection.

            else:
                best_probs = get_best_probs(labels, prior_probs)
                if best_probs.sum() == 0:
                    print(best_probs.head())
                    print("\n\n\nBest probs sums to zero!\n\n\n")
                    best_probs = pd.Series(
                        1 / (1 + labels.nunique()), index=labels.index
                    )
                label_corr, cross_corr = get_weighted_correlations(
                    feature_df,
                    labels,
                    select_params,
                    subset_dir,
                    weights=1 / best_probs,
                )

                assert np.shape(label_corr)[0] == cross_corr.shape[1]
                label_corr.to_csv(weighted_label_path)
                cross_corr.to_csv(weighted_cross_path)
                # Run iteration of feature selection on model architecture.
                if isinstance(prior_probs, list) and len(prior_probs) > 1:

                    prob_list = list()
                    one_hot, one_hot_normed = samples.one_hot_conversion(labels)
                    if select_params["sample_weight"].shape[0] != one_hot.shape[0]:
                        print("OneHot")
                        print(one_hot)
                        print("Sample weight")
                        print(select_params["sample_weight"])
                        raise ValueError
                    for prob_out in prior_probs:
                        if len(prob_out.shape) == 2:
                            prob_list.append(
                                select_params["sample_weight"].mul(one_hot).sum(axis=1)
                            )
                    for a, b in itertools.combinations(np.arange(len(prob_list)), r=2):
                        sq_diff = (prob_list[a] - prob_list[b]) ** 2
                cv_predicts, cv_scores, best_features = train_multilabel_models(
                    feature_df,
                    labels,
                    model=m,
                    model_name="{}_{}".format(n, i),
                    best_corrs=label_corr,
                    cross_corr=cross_corr,
                    select_params=select_params,
                    save_dir=subset_dir,
                    prior_probs=[best_probs],
                )
                pd.Series(best_features).to_csv(best_features_path)
                subset_predicts[n].append(cv_predicts["Original"])

                df_list = list()
                for r_name, r_dict in cv_predicts.items():
                    for score_name, score_dict in r_dict.items():
                        cv_predicts_csv_path = "{}_{}_{}.csv".format(
                            cv_predicts_csv_path_stub, score_name, r_name
                        )
                        cv_predicts_pkl_path = "{}_{}_{}.pkl".format(
                            cv_predicts_csv_path_stub, score_name, r_name
                        )
                        """
                        id_list, score_list = list(), list()
                        for a, b in score_dict.items():
                            id_list.append(a)
                            score_list.append(b)
                        id_df = pd.DataFrame(id_list, columns=["CV_Fold", "Split", "Label"])
                        score_ser = pd.Series(score_list)
                        id_df.insert(loc=id_df.shape[1], column="score", value=score_ser)
                        id_df.insert(loc=id_df.shape[1], column="Metric", value=score_name)
                        df_list.append(id_df)
                        """
                        cvp = pd.concat(
                            [pd.DataFrame.from_records(d) for d in score_dict]
                        )
                        # cvp = pd.DataFrame(cv_predicts)
                        cvp.to_csv(cv_predicts_csv_path)
                        with open(cv_predicts_pkl_path, "wb") as f:
                            pickle.dump(cvp, f)
                # print(pd.concat(df_list))

                model_scores_dict[n].append(cv_scores)
                # Use selected subsets to fit meta-estimator (ex. VotingClassifier)
                model_subsets_dict[n].append(best_features)
            if is_classifier(m):
                prior_probs.append(
                    pd.concat(cv_predicts["Original"]["predict_proba"]["test"])
                )
            elif is_classifier(m):
                prior_probs.append(
                    pd.concat(cv_predicts["Original"]["predict"]["test"])
                )
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


def weights_from_predicts(
    y_true, y_predict, predict_model, select_params, method="max"
):
    if len(y_predict) == 1:
        test_scores = pd.concat(y_predict[0][select_params["score_func"]]["test"])
    elif len(y_predict) > 1:
        test_scores = [
            pd.concat(a[select_params["score_func"]]["test"]) for a in y_predict
        ]
    else:
        raise ValueError
    if is_classifier(predict_model):
        new_weights = samples.weight_by_proba(
            y_true=y_true,
            probs=test_scores,
            prob_thresholds=select_params["brier_clip"],
        )
    elif is_regressor(predict_model):
        new_weights = samples.weight_by_error(
            y_true, test_scores, loss=select_params["loss_func"]
        )
    else:
        raise ValueError
    return new_weights


def get_best_probs(labels, cv_predict_list):
    one_hot_label, _ = samples.one_hot_conversion(y_true=labels)
    c_predicts = list()
    for p in cv_predict_list:
        if isinstance(p, dict) and "predict_proba" in p.keys():
            if isinstance(p["predict_proba"]["test"], list):
                c_predicts.append(pd.concat(p["predict_proba"]["test"]))
            else:
                c_predicts.append(p["predict_proba"]["test"])
        elif isinstance(p, (pd.Series, pd.DataFrame)):
            c_predicts.append(p)
    correct_pred = [c.mul(one_hot_label).sum(axis=1) for c in c_predicts]
    best_preds = pd.concat(correct_pred, axis=1).max(axis=1)
    return best_preds


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
        # select_params["sample_weight"] = samples.weight_by_proba(labels, votes)
    elif len(model_subsets_dict[model_name]) == 1:
        voter = name_model_dict[model_name]
        voter_results, melted_results, test_idx_list = scoring.cv_model_generalized(
            estimator=voter,
            feature_df=feature_df[model_subsets_dict[model_name][0]],
            labels=labels,
            cv=select_params["cv"],
            return_train=True,
        )
        """
        select_params["sample_weight"] = samples.weight_by_proba(
            labels, voter_results["predict_proba"]["test"][0]
        )"""

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


def _get_subset_scores(selection_state, subset):
    feats = tuple(sorted(subset))
    if feats not in selection_state["subset_scores"].keys():
        score_subset()
    return selection_state["subset_scores"][feats]


def train_multilabel_models(
    feature_df,
    labels,
    model,
    model_name,
    best_corrs,
    cross_corr,
    select_params,
    save_dir,
    prior_probs,
    **kwargs,
):
    """
    Returns a set of features that maximizes the score function of the estimator. Iteratively adds features, sampled based on label correlation and correlation with other features.
    Uses VIF score to reduce multicollinearity amongst current feature set and SFS to eliminate uninformative or confounding features.

    Parameters
    ----------
    prior_probs
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
    results_dict : dict[str, dict[str, list[pd.Series]]]
        Contains sample-wise results for each CV fold
        {method_name: {split_name: list[pd.Series] | pd.Series}}
    cv_scores : dict[str, dict[str, list[pd.Series]]]
    best_features : list, Highest scoring set of features
    """
    # if "n_jobs" in model.get_params():
    #    model.set_params(**{"n_jobs": 1})
    selection_models = {
        "predict": model,
        "permutation": model,
        "importance": model,
        "vif": linear_model.LinearRegression(),
    }
    fuzz = FuzzyAnnealer(
        params=select_params,
        models=selection_models,
        save_dir=save_dir,
    )

    fuzz.fit(
        feature_df=feature_df,
        labels=labels,
        cross_corr=cross_corr,
        label_corr=best_corrs,
        other_probs=prior_probs,
    )
    model = fuzz.best_model
    if model is None:
        model = selection_models["predict"]
    subset_scores = fuzz.subset_scores
    best_features = fuzz.best_subset
    #    model, subset_scores, best_features = fuzz.delete_asap()
    """
    model_dict, score_dict, dropped_dict, best_features = select_feature_subset(
        feature_df,
        labels,
        target_corr=best_corrs,
        cross_corr=cross_corr,
        select_params=select_params,
        selection_models=selection_models,
        save_dir=save_dir,
        prior_best_probs=prior_probs,
    )
    """
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
    eval_pred = list()

    # Score Feature Selection Model.
    cv_results, long_form_results, test_idx_splits = scoring.cv_model_generalized(
        estimator=model,
        feature_df=feature_df[list(best_features)],
        labels=labels,
        cv=select_params["cv"],
        return_train=True,
        sample_weight=select_params["sample_weight"],
    )
    # for k, long_df in long_form_results.items():
    #    long_df.insert(loc=0, column="Subset", value=model_name)
    s_name, s_func = select_params["score_name"], select_params["scoring"]
    group_cols = ["CV_Fold", "Split", "Labels"]
    predict_df = long_form_results["predict"]
    predict_groups = predict_df.groupby(group_cols, as_index=False, group_keys=False)
    # score_dict = dict([(k, v) for k, v in (s_name, s_func)])
    # for score_name in score_dict.keys():
    score_dict = dict()
    score_name = s_name
    score_dict[score_name] = dict()
    score_dict[score_name] = list()
    for g in predict_groups:
        long_df = g[1].drop(columns=group_cols)
        long_score = scoring.score_long_form(
            s_func, x=long_df, true_col="True", remove_cols="INCHI_KEY"
        )
        score_tuple = g[1].iloc[0].tolist()
        score_tuple.append(long_score)

        score_dict[score_name].append(score_tuple)
    score_df = pd.DataFrame.from_records(score_dict[score_name])
    """
    score_dict = scoring.score_cv_results(
        results_dict=cv_results,
        score_func_dict={select_params["score_name"]: select_params["scoring"]},
        y_true=labels,
        sample_weight=select_params["sample_weight"],
    )
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
    print("Score dictionary:")
    # [print(k, "\n", d.values()) for k, d in score_dict.items()]
    # model.fit(train, cmd_train_labels)
    return cv_results, score_df, best_features


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
    scores = state["subset_scores"][tuple(sorted(state["current_features"]))]
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
    clean_up, selection_state, sqcc_df = selection_setup(
        train_df,
        labels,
        initial_subset,
        target_corr,
        cross_corr,
        select_params,
        selection_models,
        prior_best_probs,
        save_dir,
    )
    # Start feature loop
    for i in np.arange(select_params["max_trials"]):
        print("\n\nSelection step {} out of {}.".format(i, select_params["max_trials"]))
        if i > 0:
            selection_state["temp"] = math_tools.acf_temp(
                [
                    selection_state["subset_scores"][tuple(sorted(s))]
                    for s in selection_state["chosen_subsets"]
                ]
            )
        print("New temperature: {}".format(selection_state["temp"]))
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
        if False and (maxed_out or (above_min and clean_up)):
            # or train_df.shape[1]
            # - len([selection_state["rejected_features"].keys()])
            # - len(selection_state["current_features"])
            # < select_params["n_vif_choices"]
            print(selection_state["current_features"])
            # Keep eliminating until sequential elimination is within range of going under score_exceeded limit.
            while _over_sfs_thresh(select_params, selection_state):
                print(current_score(selection_state), selection_state["best_score_adj"])
                original_size = len(selection_state["current_features"])
                selection_state, subset_scores, score_improve = sequential_elimination(
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
        # Check if score has dropped too much.
        exceeded, selection_state = score_drop_exceeded(
            subset_metric,
            selection_params=select_params,
            selection_state=selection_state,
            set_size=len(selection_state["current_features"]),
            replace_current=False,
        )
        if (
            exceeded
            and len(selection_state["current_features"])
            > select_params["features_min_sfs"]
        ):
            while exceeded:
                if len(selection_state["current_features"]) <= select_params[
                    "features_min_sfs"
                ] or random.random() > math_tools.zwangzig(
                    selection_state["subset_scores"],
                    current_score(selection_state["current_features"]),
                    select_params["lang_lambda"],
                    math_tools.size_factor(
                        len(selection_state["current_features"]), select_params
                    ),
                ):
                    selection_state["current_features"] = copy.deepcopy(
                        selection_state["best_subset"]
                    )
                    break
                selection_state, subset_scores, score_improve = sequential_elimination(
                    train_df,
                    labels,
                    select_params,
                    selection_state,
                    selection_models,
                    clean_up=False,
                    save_dir=save_dir,
                )
                exceeded, selection_state = score_drop_exceeded(
                    score_improve,
                    selection_params=select_params,
                    selection_state=selection_state,
                    set_size=len(selection_state["current_features"]) - 1,
                )
            continue
        # Variance Inflation Factor: VIF check implemented in "new feature" selection function.
        if True and (
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
        if False and select_params["features_min_perm"] < len(
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
                set_size=len(selection_state["current_features"]),
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
                X=train_df[list(selection_state["best_subset"])], y=labels
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


def selection_setup(
    train_df,
    labels,
    initial_subset,
    target_corr,
    cross_corr,
    select_params,
    selection_models,
    prior_best_probs,
    save_dir,
):
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
    sqcc_df = cross_corr * cross_corr
    selection_state = {
        "prior_best": prior_best_probs,
        "current_features": initial_subset,
        "temp": 1,
        "chosen_subsets": list(),
        "subset_scores": dict(),
        "best_subset": list(),
        "rejected_features": dict(),
        "best_score_adj": -999,
        "previous_best_score": -999,
        "fails": 0,
        "attempts": 0,
    }
    if initial_subset is None:
        selection_state["current_features"] = (
            target_corr.index.to_series()
            .sample(weights=target_corr.abs(), n=1)
            .tolist()
        )
        for i in np.arange(3):
            new_feat, selection_state = choose_next_feature(
                train_df,
                feature_list=selection_state["current_features"],
                target_corr=target_corr,
                sq_xcorr=sqcc_df,
                selection_models=selection_models,
                selection_state=selection_state,
                select_params=select_params,
            )
    selection_state["chosen_subsets"].append(
        tuple(sorted(selection_state["current_features"]))
    )
    print("\n\nInitial subset: {}".format(selection_state["current_features"]))
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
    clean_up = False
    print("Matrix shapes.")
    print(train_df.shape, sqcc_df.shape, target_corr.shape)
    return clean_up, selection_state, sqcc_df


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
        if set_size is None:
            set_size = len(selection_state["current_features"])
    elif set_size is None:
        raise UserWarning
        print("Feature set size needed if not using current feature set.")
        set_size = select_params["max_features_out"]
    adjust = complexity_penalty(math_tools.size_factor(set_size, select_params), factor)
    # print(len(selection_state["current_features"]), adjust)
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
    randomize=True,
    depth=1,
):
    """

    Parameters
    ----------
    train_df : pd.DataFrame
    labels : pd.Series
    select_params : dict
    selection_state : dict
    selection_models : dict
    clean_up : bool
    save_dir : str
    randomize : bool, whether to use probabalistic method for elimination selection and accept-reject decision
    depth : int, (Not Implemented) number of features to eliminate

    Returns
    -------
    selection_state : dict
    subset_scores: list, scores for feature set after any elimination
    brier_list: list, brier improvement scores for feature set after any elimination
    """
    if clean_up:
        clean = "thresh_sfs_cleanup"
    else:
        clean = "thresh_sfs"
    # TODO: Implement configurable predict function for boosting.
    sfs_score_dict = dict().fromkeys(selection_state["current_features"], list())
    if tuple(sorted(selection_state["current_features"])) not in list(
        selection_state["subset_scores"].keys()
    ):
        selection_state, scores, score_improve = score_subset(
            train_df,
            labels,
            selection_models=selection_models,
            selection_state=selection_state,
            select_params=select_params,
            save_dir=save_dir,
            subset=selection_state["current_features"],
            record_results=True,
        )
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
            subset=sorted(new_subset),
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
    feats = list(sfs_score_dict.keys())
    if randomize and len(sfs_score_dict.items()) > 0:
        geoms = dict(
            (k, stats.geometric_mean(v))
            for k, v in sfs_score_dict.items()
            if len(v) > 0
        )
        soft_scores = scaled_softmax(geoms.values(), center=selection_state["temp"])
        scores_ser = pd.Series(np.max(soft_scores) - soft_scores, index=geoms.keys())
        if scores_ser.sum() == 0.0:
            raise ValueError
        chosen_feat = (
            pd.Series(data=geoms.keys(), index=geoms.keys())
            .sample(n=1, weights=scores_ser.squeeze())
            .iloc[0]
        )
        new_subset = copy.deepcopy(selection_state["current_features"])
        new_subset.remove(chosen_feat)
        selection_state, scores, brier_list = score_subset(
            train_df,
            labels,
            selection_models=selection_models,
            selection_state=selection_state,
            select_params=select_params,
            save_dir=save_dir,
            subset=new_subset,
            record_results=True,
        )
        langevin_metric = math_tools.zwangzig(
            selection_state["subset_scores"],
            scores,
            lamb=select_params["lang_lambda"],
            temp=selection_state["temp"],
            k=math_tools.size_factor(len(new_subset), select_params),
        )
        if random.random() > langevin_metric:
            subset_scores = sfs_score_dict[chosen_feat]
            selection_state["current_features"].remove(chosen_feat)
        else:
            subset_scores = selection_state["subset_scores"][
                tuple(selection_state["current_features"])
            ]
    else:
        worst_feature_tup = sorted(
            list(sfs_score_dict.items()), key=lambda x: (np.mean(x[1] - np.std(x[1])))
        )[0]
        drop_scores = np.mean(worst_feature_tup[1]) - np.std(worst_feature_tup[1])
        set_size = len(selection_state["current_features"]) - 1
        if _over_sfs_thresh(
            select_params, selection_state, drop_scores, set_size=set_size, factor=clean
        ):
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


def subset_relative_brier_score(
    y_true,
    y_proba,
    subset_probs,
    pos_label=None,
    clips=(0, 1.0),
    sample_weight=None,
    class_weight="balanced",
    decision_thresholds=None,
):
    """
    Calculates Brier Improvement for each previous submodel
    Parameters
    ----------
    y_true : pd.Series, ground truth
    y_proba : pd.Series, By-class probabilities for feature set in question
    subset_probs :
    pos_label
    clips
    sample_weight
    class_weight
    decision_thresholds

    Returns
    -------

    """
    score_list, rel_probs_list = list(), list()
    if subset_probs is None:
        subset_probs = [None]
    for prob_set in subset_probs:
        rel_score, rel_probs = relative_brier_score(
            y_true,
            y_proba,
            prob_set,
            pos_label,
            clips,
            sample_weight,
            class_weight,
            decision_thresholds,
        )
        score_list.append(rel_score)
        rel_probs_list.append(rel_probs)
        most_similar_ix = np.argmin(score_list)
        return score_list[most_similar_ix], rel_probs_list[most_similar_ix]


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
    """
    Cross-validated scoring using a subset of features.

    Parameters
    ----------
    feature_df
    labels
    selection_models
    selection_state
    select_params
    save_dir
    subset
    record_results
    sample_weight
    class_weight

    Returns
    -------
    selection_state : dict
    scores : list
    brier_list : list
    """
    if subset is None or len(subset) == 0:
        subset = tuple(sorted(selection_state["current_features"]))
    elif len(selection_state["current_features"]) == 0:
        print("No features in current subset!")
        print(selection_state["current_features"], flush=True)
        raise ValueError
    if isinstance(subset, str):
        # raise KeyError
        subset_feats = [copy.deepcopy(subset)]
    else:
        subset_feats = tuple(sorted(copy.deepcopy(subset)))
    assert len(subset_feats) > 0
    score_tuple = [(select_params["score_name"], select_params["scoring"])]
    scores = None
    brier_list = list()
    for prior_set in selection_state["subset_scores"].keys():
        if len(set(subset_feats).symmetric_difference(prior_set)) == 0:
            scores = selection_state["subset_scores"][prior_set]
            if is_classifier(selection_models["predict"]):
                brier_list = scores
            # print("Duplicate scoring found:\n{}\n".format(subset_feats, prior_set))
            break
    if scores is None:
        selection_state["subset_scores"][tuple(sorted(subset_feats))] = list()
        # best_corrs, cross_corr = get_correlations(train_df, train_labels, path_dict["corr_path"], path_dict["xc_path"], select_params["corr_method"], select_params["xc_method"])
        if "sample_weight" in select_params.keys() or sample_weight is not None:
            if sample_weight is not None:
                weights = sample_weight
            else:
                weights = select_params["sample_weight"]
            results, long_form_dict, test_idx_list = scoring.cv_model_generalized(
                estimator=selection_models["predict"],
                feature_df=feature_df[list(subset_feats)],
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
            results, long_form_dict, test_idx_list = scoring.cv_model_generalized(
                estimator=selection_models["predict"],
                feature_df=feature_df[list(subset_feats)],
                labels=labels,
                cv=select_params["cv"],
            )
            scores = scoring.score_cv_results(
                results, dict(score_tuple), y_true=labels, **select_params
            )["Score"].tolist()
        if is_classifier(selection_models["predict"]):
            # TODO: Add functionality for multiple predict_proba results from previous submodels.
            for proba in results["predict_proba"]["test"]:
                if isinstance(selection_state["prior_best"], (pd.Series, pd.DataFrame)):
                    prior = selection_state["prior_best"].loc[proba.index]
                else:
                    prior = None
                rel_brier_score, rel_brier = subset_relative_brier_score(
                    labels[proba.index],
                    proba,
                    subset_probs=prior,
                    pos_label=select_params["pos_label"],
                )
                brier_list.append(rel_brier_score)
            scores = brier_list
        selection_state["subset_scores"][tuple(sorted(subset_feats))] = scores
        _compare_to_best(scores, selection_state, features=subset_feats)
        if record_results:
            record_score(features=subset_feats, scores=scores, save_dir=save_dir)
    # print(np.mean(scores))
    return selection_state, scores, brier_list


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
            ones.subtract(sq_xcorr[list(feature_list)].loc[feat_corrs.index])
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
    selection_state["chosen_subsets"].append(tuple(selection_state["current_features"]))
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


def record_score(features, scores, save_dir, test_score=None):
    score_str = "{}\t{}\n".format(
        "\t".join(["{:.5f}".format(sc) for sc in scores]),
        "\t".join(features),
    )
    with open("{}feature_score_path.csv".format(save_dir), "a", encoding="utf-8") as f:
        f.write(score_str)
    # print(score_str)
    if test_score is not None:
        with open("{}test_scores.csv".format(save_dir), "a", encoding="utf-8") as f:
            f.write(
                "{:.5f}\t{}\n".format(
                    test_score,
                    "\t".join(features),
                )
            )

    return


# Complexity penalty should already be added during intial scoring.
def _compare_to_best(scores, selection_state, features=None):
    if features is None:
        features = selection_state["current_features"]
    if np.mean(scores) - np.std(scores) > selection_state["best_score_adj"]:
        print(
            "New top results for {} feature model: Mean: {:.4f}, Std {:.4f}".format(
                len(features),
                np.mean(scores),
                np.std(scores),
            )
        )
        selection_state["previous_best_score"] = copy.deepcopy(
            selection_state["best_score_adj"]
        )
        selection_state["best_score_adj"] = np.mean(scores) - np.std(scores)
        selection_state["best_subset"] = copy.deepcopy(features)
        best_yet = True
    else:
        best_yet = False
    return selection_state, best_yet


def score_drop_exceeded(
    new_scores, selection_params, selection_state, set_size=None, replace_current=True
):
    new_score = np.mean(new_scores) - np.std(new_scores)
    if (
        _over_sfs_thresh(
            selection_params,
            selection_state,
            scores=new_score,
            set_size=set_size,
            factor="reset",
        )
        and set_size >= selection_params["features_min_sfs"]
    ):
        print(
            "Score (adjusted) drop exceeded: {:.4f} {:.4f}".format(
                selection_state["best_score_adj"], new_score
            )
        )
        if replace_current:
            selection_state["current_features"] = copy.deepcopy(
                selection_state["best_subset"]
            )
        return True, selection_state
    else:
        return False, selection_state
