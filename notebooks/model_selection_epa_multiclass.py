import logging
import os
import pprint
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_log_error,
)
from sklearn.model_selection import (
    cross_val_predict,
    GridSearchCV,
    KFold,
    ParameterGrid,
    StratifiedKFold,
)
from sklearn.utils.validation import _check_y

import samples
import scoring
from correlation_filter import get_correlations, get_weighted_correlations
from dataset_creation import (
    _set_paths,
    assemble_dataset,
    assemble_dmso_dataset,
    preprocess_data,
)
from epa_enamine_visualizer import plot_clf_model_displays, plot_model_scores
from ForwardFuzzyCoclustering import (
    select_subsets_from_model,
)
from scoring_metrics import get_confusion_weights


def optimize_tree(feature_df, labels, model, scoring, cv, path_dict):
    optimum_path = "{}optimal_params.csv".format(path_dict["exp_dir"])
    all_param_path = "{}hyperparam_results.csv".format(path_dict["exp_dir"])
    hyper_score_path = "{}hyperparam_scores.csv".format(path_dict["exp_dir"])
    if True and os.path.isfile(optimum_path):
        optimal_params = (
            pd.read_csv(optimum_path, index_col=0, na_filter=False).squeeze().to_dict()
        )
        for k, v in optimal_params.items():
            print(v)
            if v == "True":
                optimal_params[k] = True
            elif v == "False":
                optimal_params[k] = False
            elif v is np.nan or v == "":
                optimal_params[k] = None
            elif isinstance(v, str) and "." in v:
                optimal_params[k] = float(v)
            elif isinstance(v, str) and v.isnumeric():
                optimal_params[k] = int(v)
            else:
                optimal_params[k] = v
    elif True:
        param_grid = {
            # "max_depth": [None, 25, 15],
            "n_estimators": [50, 250, 1000],
            "min_impurity_decrease": [0, 0.0025],
            "max_leaf_nodes": [150, 250],
            "class_weight": ["balanced", "balanced_subsample"],
            "bootstrap": [True, False],
        }
        tree_search = GridSearchCV(
            estimator=clone(model),
            param_grid=ParameterGrid(param_grid).param_grid,
            scoring=scoring,
            n_jobs=-1,
            cv=cv,
            return_train_score=True,
            error_score="raise",
        )
        tree_search.fit(X=feature_df, y=labels)
        optimal_params = tree_search.best_params_
        pd.Series(optimal_params).to_csv(optimum_path)
        hyper_rankings = pd.Series(tree_search.cv_results_["rank_test_score"]).tolist()
        pd.DataFrame.from_records(tree_search.cv_results_["params"]).to_csv(
            all_param_path
        )
        hyper_scores = (
            pd.Series(tree_search.cv_results_["mean_test_score"])
            .reindex(index=hyper_rankings)
            .sort_index()
        )
        hyper_scores.to_csv(hyper_score_path)
        try:
            pd.DataFrame.from_records(tree_search.cv_results_["params"]).sort_index(
                key=lambda x: pd.Series(hyper_rankings)
            ).to_csv(all_param_path, index_label="Ranking")
        except:
            pd.DataFrame.from_records(tree_search.cv_results_["params"]).sort_index(
                key=lambda x: pd.Series(hyper_rankings)
            ).to_csv(all_param_path, index_label="Ranking")
        finally:
            hyper_scores = (
                pd.Series(tree_search.cv_results_["mean_test_score"])
                .reindex(index=hyper_rankings)
                .sort_index()
            )
            hyper_scores.to_csv(hyper_score_path)
    estimator = RandomForestClassifier().set_params(**optimal_params)
    return estimator


def main():
    classification = True
    dataset = "rus_enamine-50"
    frac_enamine = 0.5
    rus = True
    plot_all_features = False
    estimator_name = "better_params"

    (
        data_tuple,
        select_params,
        path_dict,
        estimator,
        estimator_name,
        score_tups,
    ) = setup_params(dataset, frac_enamine, classification, rus, estimator_name)
    train_df, train_labels, test_df, test_labels = data_tuple

    (
        train_df,
        train_labels,
        test_df,
        test_labels,
        preprocessor,
        best_corrs,
        cross_corr,
    ) = preprocess_data(
        (train_df, train_labels), (test_df, test_labels), select_params, path_dict
    )
    print("Running model in directory: {}".format(path_dict["parent_dir"]))
    # Plot all features model.
    probs = None
    if True and classification:
        booster_path = "{}booster_weights.csv".format(path_dict["exp_dir"])
        booster, probs = hist_grad_boost(
            train_df,
            train_labels,
            select_params,
            booster_path,
            path_dict,
            plot_all_features,
            score_tups,
        )
    if select_params["sample_weight"] is None:
        label_corr, cross_corr = get_correlations(
            feature_df=train_df,
            labels=train_labels,
            corr_path=path_dict["corr_path"],
            xc_path=path_dict["xc_path"],
            corr_method=select_params["corr_method"],
            xc_method=select_params["xc_method"],
            use_disk=True,
        )
    else:
        label_corr, cross_corr = get_weighted_correlations(
            feature_df=train_df,
            labels=train_labels,
            select_params=select_params,
            subset_dir=path_dict["exp_dir"],
        )
    name_model_dict = {estimator_name: estimator}
    print(name_model_dict.items())
    print(
        "Discretized Labels: Value Counts:\n{}".format(
            pprint.pformat(train_labels.value_counts())
        )
    )
    # Get subsets from training loop
    prior_probs = [probs]
    if True:
        n_subsets = 6
        model_scores_dict, model_subsets_dict, subset_output_tups, name_weights_dict = (
            select_subsets_from_model(
                feature_df=train_df,
                labels=train_labels,
                n_subsets=n_subsets,
                name_model_dict=name_model_dict,
                label_corr=best_corrs,
                cross_corr=cross_corr,
                exp_dir=path_dict["parent_dir"],
                select_params=select_params,
                prior_probs=prior_probs,
            )
        )
        # print(subset_output_tups)
        cv_scores = pd.DataFrame.from_dict(model_scores_dict, orient="index")
        # print("CV Scores ({})".format([s for s in model_scores_dict.values()]))
        # print(cv_scores, flush=True)
        results_dict = dict()
        plot_dict = dict()
        for n, m in name_model_dict.items():
            if False and (
                len(model_subsets_dict[n]) < 2
                or isinstance(model_subsets_dict[n], str)
                or any([len(subset) < 2 for subset in model_subsets_dict[n]])
            ):
                print(
                    "Size of feature set to be plotted: {}".format(
                        len(model_subsets_dict[n])
                    )
                )
                continue
            plot_selections(
                train_df,
                train_labels,
                select_params,
                n,
                path_dict,
                score_tups,
                results_dict,
                model_subsets_dict,
                name_model_dict,
                plot_dict,
            )
    else:
        model_subsets_dict = dict.fromkeys(
            name_model_dict.keys(), train_df.columns.tolist()
        )
        model_scores_dict = dict.fromkeys(
            name_model_dict.keys(), train_df.columns.tolist()
        )
    # ("Confusion Matrix", ConfusionMatrixDisplay.from_estimator), ("Det", DetCurveDisplay.from_estimator), ("ROC", RocCurveDisplay.from_estimator))
    score_tup_dict = dict([(k[0], dict()) for k in score_tups])
    # results_dict = {"Feature Selection": list(), "Randomized Label": score_tup_dict}
    # results_dict = dict().fromkeys(name_model_dict.keys(), score_tup_dict) # {"Feature Selection": list(), "Randomized Label": score_tup_dict}

    return (
        (train_df, test_df),
        (train_labels, test_labels),
        model_subsets_dict,
        model_scores_dict,
    )


def hist_grad_boost(
    train_df,
    train_labels,
    select_params,
    booster_path,
    path_dict,
    plot_all_features,
    score_tups,
):
    """
    best_estimator = optimize_tree(
        train_df,
        train_labels,
        estimator,
        select_params["scoring"],
        select_params["cv"],
        path_dict,
    )
    """
    # Set initial sample weights to SAMME weights from boosting.
    custom = False
    booster_params = {
        "max_iter": 500,
        "max_bins": 100,
        "max_features": 0.5,
        "class_weight": "balanced",
        "n_iter_no_change": 10,
        "learning_rate": 0.025,
    }
    booster = HistGradientBoostingClassifier(
        # l2_regularization=0.025,
        # max_depth=7,
        #  verbose=1,
        # scoring=brier_score_loss,
    )
    booster.set_params(**booster_params)
    probs_list = list()
    if os.path.isfile(booster_path):
        probs = pd.read_csv(booster_path, index_col="INCHI_KEY").squeeze()
        select_params["sample_weight"] = samples.weight_by_proba(
            train_labels, probs, prob_thresholds=select_params["brier_clips"]
        )
    elif True:
        if custom:
            results, score_dict, long_form, test_idx_list, score_dict = (
                scoring.cv_model_generalized(
                    booster,
                    train_df,
                    train_labels,
                    v=select_params["cv"],
                )
            )
            # booster = FrozenEstimator(booster.fit(train_df, train_labels))
            # probs = pd.DataFrame(booster.predict_proba(train_df), index=train_df.index)
            probs = pd.concat(results["predict_proba"]["test"])
        else:
            probs = cross_val_predict(
                booster,
                X=train_df,
                y=train_labels,
                method="predict_proba",
                # cv=select_params["cv"],
                n_jobs=-1,
                params=None,
            )
            probs = pd.DataFrame(probs, index=train_labels.index)
        print("HGB Probabilities: {}".format(probs))
        select_params["sample_weight"] = samples.weight_by_proba(
            y_true=train_labels,
            probs=probs,
            prob_thresholds=select_params["brier_clips"],
        )
        probs.to_csv(booster_path, index_label="INCHI_KEY")
        # print(booster.n_iter_)
        # print(booster.validation_score_)
    if plot_all_features:
        display_dir = "{}all_features/".format(path_dict["exp_dir"])
        os.makedirs(display_dir, exist_ok=True)
        subsets = tuple(
            [
                train_df.columns.to_series().sample(n=select_params["max_features_out"])
                for a in range(4)
            ]
            + [train_df.columns.tolist()]
        )
        score_results, score_plot = plot_model_scores(
            feature_df=train_df,
            train_labels=train_labels,
            score_tups=score_tups,
            estimator=booster,
            subsets=subsets,
            cv=select_params["cv"],
            # sample_weight=select_params["sample_weight"],
        )
        score_plot.savefig("{}all_features_score.png".format(display_dir))
        all_feat_displays = plot_clf_model_displays(
            estimator=booster,
            estimator_name="HistGradBoost",
            train_df=train_df,
            train_labels=train_labels,
            select_params=select_params,
            subset_dir=display_dir,
            display_labels=["Insoluble", "Soluble"],
            sample_weight=select_params["sample_weight"],
            probs=probs,
        )
    return booster, probs


def setup_params(dataset_name, frac_enamine, classification, rus, estimator_name):
    if classification:
        # Keep score_func at predict_proba. This only affects how selection weights samples.
        # Plotting, etc is handled by scoring function signature.
        select_params = _set_params(
            score_func=balanced_accuracy_score,
            score_name="balanced_accuracy",
            response_method="predict_proba",
            greater_is_better=True,
        )
        # Model scoring metrics.
        select_params["cv"] = partial(
            StratifiedKFold,
            shuffle=True,
            # RepeatedStratifiedKFold,
            # n_splits=3,
            # n_repeats=3,
            random_state=0,
        )
        score_tups = (
            ("MCC", matthews_corrcoef),
            ("Balanced Acc", balanced_accuracy_score),
            # ("Brier Loss", brier_score_loss),
        )
        estimator = RandomForestClassifier(
            n_estimators=50,
            class_weight="balanced",
        ).set_params(
            **{
                "class_weight": "balanced",
                # "max_depth": 15,
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.001,
                "n_jobs": -3,
            }
        )
        print(estimator.__repr__())
    else:
        select_params = _set_params(
            score_func=partial(
                root_mean_squared_log_error,
                multioutput="raw_values",
            ),
            score_name="RMSLE",
            response_method="predict",
            greater_is_better=False,
        )
        select_params["cv"] = partial(KFold, shuffle=True, random_state=0)
        # estimator = HuberRegressor(warm_start=True, max_iter=2500, epsilon=1.2)
        # estimator_name = "Huber_1_2"
        estimator = LinearRegression()
        score_tups = (("r2", r2_score), ("mape", mean_absolute_percentage_error))
        # search_features = train_df.columns.tolist()
        # Labels are scikit-learn compatible.
    if "epa" in dataset_name or "enamine" in dataset_name or "dmso" in dataset_name:
        if rus:
            parent_dir = "{}{}_rus/".format(os.environ.get("MODEL_DIR"), dataset_name)
        else:
            parent_dir = "{}{}/".format(os.environ.get("MODEL_DIR"), dataset_name)
        data_dir = parent_dir
        path_dict = _set_paths(parent_dir, data_dir, select_params)
        train_df, train_labels, test_df, test_labels = assemble_dmso_dataset(
            dataset_name, select_params, path_dict, frac_enamine=frac_enamine, rus=True
        )
        path_dict["exp_dir"] = parent_dir
    else:
        train_data, test_data, preprocessor, data_dir, parent_dir = assemble_dataset(
            dataset_name
        )
        train_df, train_labels = train_data
        test_df, test_labels = test_data
        path_dict = _set_paths(parent_dir, data_dir, select_params)
    path_dict["exp_dir"] = "{}{}/".format(parent_dir, estimator_name)
    os.makedirs(path_dict["exp_dir"], exist_ok=True)
    train_labels = pd.Series(
        _check_y(train_labels.copy().squeeze(), estimator=estimator),
        index=train_labels.index,
        name="Labels",
    )
    test_labels = pd.Series(
        _check_y(test_labels.copy().squeeze(), estimator=estimator),
        index=test_labels.index,
        name="Labels",
    )
    train_df.dropna(axis="columns", inplace=True)
    train_df.dropna(axis="index", inplace=True)
    test_df = test_df[train_df.columns.copy()]
    output_data = (
        train_df,
        train_labels,
        test_df,
        test_labels,
    )
    return (
        output_data,
        select_params,
        path_dict,
        estimator,
        estimator_name,
        score_tups,
    )


def plot_selections(
    feature_df,
    labels,
    select_params,
    model_name,
    path_dict,
    score_tups,
    results_dict,
    subsets_dict,
    model_dict,
    plot_dict,
):

    for subset_i, subset in enumerate(subsets_dict[model_name]):
        subset_dir = "{}/subset{}/".format(path_dict["exp_dir"], subset_i)
        if not os.path.isdir(subset_dir):
            print("Subset dir not found: {}".format(subset_dir))
            os.makedirs(subset_dir, exist_ok=True)
        submodel_name = "{}_{}".format(model_name, subset_i)
        plots = plot_clf_model_displays(
            estimator=model_dict[model_name],
            estimator_name=submodel_name,
            train_df=feature_df[list(subset)],
            train_labels=labels,
            select_params=select_params,
            subset_dir=subset_dir,
        )
        plt.close()

    results_dict[model_name], plot_dict[model_name] = plot_model_scores(
        feature_df=feature_df,
        train_labels=labels,
        score_tups=score_tups,
        estimator=model_dict[model_name],
        subsets=subsets_dict[model_name],
        cv=select_params["cv"],
        # sample_weight=plot_weights,
    )
    # score_plot.figure.set(title="{}".format(model_name), ylabel="Score")
    print(results_dict[model_name])
    results_dict[model_name].to_csv(
        "{}{}results_long-form.csv".format(path_dict["exp_dir"], model_name)
    )
    plot_dict[model_name].savefig("{}{}.png".format(path_dict["exp_dir"], model_name))
    plt.close()
    # print(pd.DataFrame.from_dict(results_dict, orient="index"))


def _set_params(
    score_func=matthews_corrcoef,
    score_name=None,
    response_method="predict_proba",
    loss_func=mean_absolute_percentage_error,
    greater_is_better=True,
):
    """
    Set parameters for feature selection

    Parameters
    ----------
    score_func : callable
    score_name : str
    response_method : str
    loss_func : callab;e
    greater_is_better : bool

    Returns
    -------
    select_params : dict
    """
    select_params = {
        "thresh_xc": 0.95,
        "fails_min_vif": 100,
        "fails_min_perm": 0,
        "fails_min_sfs": 0,
        "W_confusion": get_confusion_weights(),
        "loss_func": loss_func,
        "lang_lambda": 0.1,
        # Features In Use
        "max_trials": 25,
        "max_features_out": 30,
        "min_features_out": 10,
        "tol": 0.01,
        "n_iter_no_change": 10,
        "corr_method": "spearman",
        "xc_method": "pearson",
        "thresh_reset": 0.05,
        "n_vif_choices": 5,
        "add_n_feats": 3,
        "features_min_vif": 8,
        "features_min_perm": 20,
        "features_min_sfs": 12,
        "thresh_vif": 30,
        "thresh_perm": 0.025,
        "thresh_sfs": 0,
        "thresh_sfs_cleanup": 0,
        "cv": partial(StratifiedKFold, shuffle=True, random_state=14),
        "importance": True,
        # "scoring": make_scorer(three_class_solubility),
        "scoring": score_func,
        "scorer": make_scorer(score_func, greater_is_better=greater_is_better),
        "score_func": response_method,
        "sample_weight": None,
        "initial_weights": None,
        "pos_label": 0,
        "brier_clips": (0, 1.0),
    }
    if score_name is None:
        select_params["score_name"] = str(score_func.__repr__())
    else:
        select_params["score_name"] = score_name
    return select_params


def _make_proba_residuals(data, labels=None, combine=True):
    resid = dict()
    for col_a in np.arange(data.shape[1]):
        for col_b in np.arange(col_a + 1, data.shape[1]):
            resid[(col_a, col_b)] = data.iloc[:, col_a] - data.iloc[:, col_b]
    if combine:
        resid = pd.DataFrame.from_dict(resid)
    return resid


if __name__ == "__main__":
    sklearn.set_config(transform_output="pandas")
    # balanced_accuracy_score
    logger = logging.getLogger(name="selection")
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        # warnings.simplefilter("error")
        main_dfs, main_labels, subset_dict, scores_dict = main()
    """
    name_model_dict = get_eval_models(scorer=make_scorer(matthews_corrcoef))
    forest_params = {
        "bootstrap": ["False"],
        "n_estimators": [10, 50, 100, 250],
        "min_impurity_decrease": [0, 0.0005, 0.001, 0.005, 0.01],
        "max_features": [3, 5, 6, 7, 9, None],
        "max_leaf_nodes": [100, 150, 200, 250, None],
        "random_state": [0],
    }
    for m, n in name_model_dict.items():
        gscv = GridSearchCV(
            m,
            param_grid=forest_params,
            scoring=make_scorer(three_class_solubility),
            n_jobs=-1,
            cv=RepeatedStratifiedKFold,
        )
        gscv.fit(main_dfs[0], main_labels[0])
        gs_results = gscv.cv_results_
        print(gs_results)
        """
