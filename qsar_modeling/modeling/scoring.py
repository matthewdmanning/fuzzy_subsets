import copy
import numbers
import pickle
from collections import defaultdict
from functools import partial
from inspect import signature

import numpy as np
import pandas as pd
import sklearn.pipeline
from matplotlib import pyplot as plt
from scipy.stats import gmean
from sklearn import clone, metrics
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    hamming_loss,
    jaccard_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, LearningCurveDisplay, StratifiedKFold
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.validation import _check_y

from qsar_modeling.utils import cv_tools

# TODO: Implement sample weight. Will probably need to be a separate function for cv_model_generalized that selects after the split.

learning_curve = ("Learning Curve", partial(LearningCurveDisplay.from_estimator))


def get_displays():
    displays = [
        ("Confusion Matrix", partial(metrics.ConfusionMatrixDisplay.from_estimator)),
        ("ROC", partial(metrics.RocCurveDisplay.from_estimator)),
        ("DET", partial(metrics.DetCurveDisplay.from_estimator)),
        ("Precision-Recall", partial(metrics.PrecisionRecallDisplay.from_estimator)),
        ("Prediction Error", metrics.PredictionErrorDisplay.from_estimator),
    ]
    return displays


def get_pred_score_funcs():
    # Returns tuples of scoring metric names and their callable functions. Option for one that only return scalar values.
    pred_scores = [
        ("Balanced Accuracy", balanced_accuracy_score),
        ("Matthews Correlation Coefficient", matthews_corrcoef),
        # ('Hinge Loss', hinge_loss),
        ("Recall Micro", partial(recall_score, average="micro")),
        ("Hamming Loss", hamming_loss),
        ("Negative Log Loss", partial(log_loss)),
        ("Negative Log Loss Normalized", partial(log_loss, normalize=True)),
        ("Jaccard score", partial(jaccard_score, pos_label=0)),
    ]
    # ('Classification Report', partial(classification_report, target_names=['Insoluble', 'Soluble'])),
    # ('Classification Report - Weighted', partial(classification_report, target_names=['Insoluble', 'Soluble']))
    score_list = [
        (name, make_scorer(f, response_method="predict")) for name, f in pred_scores
    ]
    return score_list


def get_score_bounds():
    # Returns bounds to check reasonableness of calculated scores for model performance.
    bounds = dict(
        [
            ("Balanced Accuracy", (0.5, 1.0)),
            ("Matthews Correlation Coefficient", (-1.0, 1.0)),
            # ('Hinge Loss', hinge_loss),
            ("Recall Micro", (0.0, 1.0)),
            ("Hamming Loss", (0.0, 1.0)),
            ("Negative Log Loss", (0, 999999)),
            ("Negative Log Loss Normalized", (0, 999999)),
            ("Jaccard score", (0.0, 1.0)),
        ]
    )
    return bounds


def get_prob_scores_dict(pos_label=0):
    # Returns classification scores that require probability or non-prediction based outputs.
    prob_scores = dict(
        [
            (
                "Average Precision Macro",
                partial(average_precision_score, pos_label=pos_label, average="macro"),
            ),
            (
                "Average Precision Micro",
                partial(average_precision_score, average="micro"),
            ),
            (
                "Average Precision Weighted",
                partial(
                    average_precision_score, pos_label=pos_label, average="weighted"
                ),
            ),
            ("Negative Brier Score", partial(brier_score_loss, pos_label=pos_label)),
            ("Precision Micro", partial(precision_score, average="micro")),
            # ('ROC Score', partial(roc_curve, pos_label=pos_label))
        ]
    )
    score_list = [
        (name, make_scorer(f, response_method="predict_proba"))
        for name, f in prob_scores
    ]
    return score_list


def score_cv_results(
    results_dict, score_func_dict, y_true=None, by_fold=True, **score_kws
):
    """
    Applies scoring function to nested dictionary of estimator outputs.
    results_dict:

    Parameters
    ----------
    results_dict: dict[str[str[list]]], ex. {"predict": {"test": [pd.Series]}}
    score_func_dict: dict[str, callable], Accepts estimator output from results_dict
    y_true: pd.Series | Iterable[pd.Series], arg for score_func, if by_fold == False, must be pd.Series
    by_fold: bool, whether to apply score_func to individual folds or score after concatenating folds
    score_kws: Applied to each score_func call.

    Returns
    -------
    score_dict: Same structure as results_dict, but contains output from score_funcs
    {score_name: {split_name: list[pd.Series] | pd.Series}}
    """
    score_dict = dict.fromkeys(score_func_dict.keys(), dict())
    score_df_dict = dict.fromkeys(score_func_dict.keys())
    for score_name, score_func in score_func_dict.items():
        score_keys = signature(score_func).parameters.keys()
        if "y_proba" in score_keys:
            output_name = "predict_proba"
        # elif "y_pred" in score_keys:
        elif "y_pred" in score_keys:
            output_name = "predict"
        else:
            output_name = "predict"
        for split_name, result in results_dict[output_name].items():
            score = list()
            if by_fold:
                for res in result:
                    if (
                        len(res.shape) > 1
                        and res.shape[1] > 1
                        and "pos_label" in score_kws
                    ):
                        r = res.iloc[:, score_kws["pos_label"]]
                    elif len(res.shape) > 1 and res.shape[1] > 1:
                        r = res.iloc[:, 1]
                    else:
                        r = res
                    new_kws = dict()
                    for k, v in score_kws.items():
                        if k in score_keys:
                            if isinstance(v, (pd.Series, pd.DataFrame)):
                                new_kws[k] = v[r.index]
                            else:
                                new_kws[k] = v
                    # print(score_func(y_true[r.index], r))
                    if y_true is not None:
                        if output_name == "predict":
                            # print(r.value_counts())
                            if y_true[r.index].value_counts().size < 2:
                                print(y_true[r.index].value_counts())
                            score.append(
                                score_func(y_true=y_true[r.index], y_pred=r, **new_kws)
                            )
                        elif output_name == "predict_proba":
                            score.append(
                                score_func(y_true=y_true[r.index], y_proba=r, **new_kws)
                            )
                        else:
                            score.append(
                                score_func(y_true=y_true[r.index], y_proba=r, **new_kws)
                            )
                    else:
                        score.append(score_func(r, **new_kws))
            else:
                if y_true is not None:
                    score = score_func(y_true, pd.concat(result), **score_kws)
                else:
                    score = score_func(pd.concat(result), **score_kws)
            score_dict[score_name][split_name] = score
        # print(score_dict[score_name])
        score_df_dict[score_name] = (
            pd.DataFrame.from_records(score_dict[score_name])
            .reset_index(names="CV Fold")
            .melt(id_vars=["CV Fold"], var_name="Split", value_name="Score")
        )
        # print(score_df_dict[score_name])
        score_df_dict[score_name].insert(loc=0, column="Metric", value=score_name)
    score_df = pd.concat(score_df_dict.values(), ignore_index=True)
    return score_df


def proba_from_meta_estimator(
    base_estimator,
    meta_est,
    feature_df,
    labels,
    subsets,
    cv=StratifiedKFold,
    sample_weights=None,
):
    """
    for model, weights in itertools.zip_longest(base_estimator, sample_weights):
        meta_fit = StackingClassifier(estimators=estimator_tups, cv=cv, n_jobs=-1).fit(
            X=feature_df, y=feature_df, sample_weight=weights
        )
    """
    return None


def calculate_proba_from_model(
    estimator,
    feature_df,
    labels,
    select_params,
    cv=StratifiedKFold,
    preprocessor=None,
    sample_weight=None,
):
    """
    Convenience function to calculate model predict_proba for one or more sample_weightings.

    Parameters
    ----------
    estimator: BaseEstimator
    feature_df: pd.DataFrame
    labels: pd.Series
    cv: default=StratifiedKFold
    preprocessor: BaseEstimator, fold-wise preprocessing before fitting
    sample_weight: None | pd.Series | Iterable[pd.Series], weights for model fitting

    Returns
    -------
    results_list: list[pd.Series], shape[n_samples], List of sample weights for each input sample weighting. Intended for use in feature selection.
    """
    results_list = list()
    if cv is None and is_classifier(estimator):
        cv = partial(StratifiedKFold, shuffle=True, random_state=0)
    elif cv is None and is_regressor(estimator):
        cv = partial(KFold, shuffle=True, random_state=0)
    if sample_weight is None:
        sample_weight = [
            pd.Series(
                data=np.ones(shape=feature_df.shape[0], dtype=np.float32),
                index=feature_df.index,
            )
        ]
    elif len(sample_weight) == 1 or isinstance(
        sample_weight, (pd.Series, pd.DataFrame)
    ):
        sample_weight = [sample_weight]
    for weights in sample_weight:
        results, idx_fold_list = cv_model_generalized(
            estimator,
            feature_df,
            labels,
            cv=cv,
            preprocessor=preprocessor,
            sample_weight=weights,
        )
        results_list.append(results)
    return [pd.concat(r[select_params["score_func"]]["test"]) for r in results_list]


def score_randomized_classes(
    estimator,
    feature_df,
    labels,
    cv=StratifiedKFold,
    preprocessor=None,
    method_list=None,
    label_seed=None,
    return_train=True,
    sample_weight=None,
    **splitter_kws,
):
    """
    Fits and scores original and randomized labelled data.
    Parameters
    ----------
    estimator: BaseEstimator, Unfit estimator or pipeline
    feature_df: pd.DataFrame, features
    labels: pd.Series, labels for supervised training
    cv: "Cross-Validator", default=StratifiedKFold
    preprocessor: BaseEstimator or implements "fit" and "transform", passed to cross-validator function
    return_train: bool, whether to also calculate and return results for training data
    label_seed: int, seed for randomizing labels
    score_list: Iterable[str], values must be methods from estimator (eg. "predict", "predict_proba", "decision_function")
    sample_weight: pd.Series, passed to cross-validator
    splitter_kws

    Returns
    -------
    results: dict[str, dict[str, list(pd.Series)]], nested dictionary of {callable name: {"test" | "train: list[pd.Series]}}, where the list contain results from callable over all CV folds.
    """
    scrambled_labels = pd.Series(
        data=labels.copy().sample(frac=1.0, random_state=label_seed).to_list(),
        index=labels.index.copy(name="INCHI_KEY"),
    )
    randomized_labels = pd.Series(
        _check_y(scrambled_labels, estimator=estimator), index=scrambled_labels.index
    )
    results, test_idx_list = cv_model_generalized(
        estimator=estimator,
        feature_df=feature_df,
        labels=randomized_labels,
        cv=cv,
        preprocessor=preprocessor,
        score_list=method_list,
        return_train=return_train,
        sample_weight=sample_weight,
    )
    return results


# noinspection PyUnresolvedReferences
def cv_model_generalized(
    estimator,
    feature_df,
    labels,
    cv=StratifiedKFold,
    preprocessor=None,
    methods=None,
    return_train=False,
    clone_model=True,
    sample_weight=None,
    warn_no_weights=False,
    **kwargs,
):
    """
    Generalized, all-in-one wrapper for cross-validated models. Desired outputs are given as nested tuples of name, callable.
    Results are returned as a dictionary of dictionaries of lists.
    Schema: results = {fname: {split_set: [pd.Series] } }
    Example: results = {"predict": {"test": [pd.Series, pd.Series, ...] } }

    Parameters
    ----------

    estimator: BaseEstimator, Unfit estimator or pipeline
    feature_df: pd.DataFrame, features
    labels: pd.Series, labels for supervised training
    cv: "Cross-Validator", default=StratifiedKFold
    preprocessor: TransformerMixin or implements "fit" and "transform"
    return_train: bool, whether to also calculate and return results for training data
    methods: Iterable[str], values must be methods from estimator (eg. "predict", "predict_proba", "decision_function")

    Returns
    -------
    results: dict[str, dict[str, list(pd.Series)]], nested dictionary of {callable name: {"test" | "train: list[pd.Series]}}, where the list contain results from callable over all CV folds.
    test_idx_list: list[pd.Index], list of indices in the order in which they appear during cross-validation
    """
    from sklearn.utils.estimator_checks import is_regressor, is_classifier

    assert isinstance(feature_df, pd.DataFrame) and not feature_df.empty
    sklearn.set_config(transform_output="pandas")
    # Set up dictionaries and check estimator calls and keywords.
    test_idx_list = list()
    if methods is None:
        if is_classifier(estimator):
            methods = ("predict", "predict_proba")
        elif is_regressor(estimator):
            methods = tuple(
                "predict",
            )
    elif isinstance(methods, str):
        methods = [methods]
    scorer_list = [
        s for s in methods if HasMethods(methods=s).is_satisfied_by(estimator)
    ]
    if return_train:
        splits_dict = {"train": list(), "test": list()}
        results = dict([(k, {"train": list(), "test": list()}) for k in scorer_list])
    else:
        results = dict([(k, {"test": list()}) for k in scorer_list])
    func_kwargs_dict = dict()
    for func in scorer_list:
        func_kwargs_dict[func] = [
            k
            for k in signature(getattr(estimator, func)).parameters.keys()
            if k != "self" and k != "X"
        ]
    # results = dict([(k[0], splits_dictcopy()) for k in return_list])
    i = 0
    assert isinstance(feature_df, pd.DataFrame)

    for train_X, train_y, test_X, test_y in cv_tools.split_df(
        feature_df, labels, splitter=cv, **kwargs
    ):
        split_X, split_y = {"train": train_X, "test": test_X}, {
            "train": train_y,
            "test": test_y,
        }
        if preprocessor is not None:
            preprocessor.fit(train_X)
            split_X["train"] = preprocessor.transform(train_X).to_frame()
            split_X["test"] = preprocessor.transform(test_X).to_frame()
        else:
            split_X["train"] = train_X
            split_X["test"] = test_X
        test_idx_list.append(test_y.index)
        assert not split_X["train"].empty
        assert isinstance(split_X["train"], pd.DataFrame)
        if clone_model:
            model_to_fit = clone(estimator)
        else:
            model_to_fit = copy.deepcopy(estimator)
        if (
            # HasMethods("sample_weight").is_satisfied_by(estimator) and
            "sample_weight" in kwargs.keys()
            and sample_weight is not None
        ):
            if "sample_weight" in kwargs.keys():
                sample_weight = kwargs["sample_weight"]
            if isinstance(sample_weight, pd.Series):
                weights = sample_weight
            # if isinstance(sample_weight, dict):
            elif isinstance(sample_weight, pd.DataFrame):
                weights = sample_weight[-1].squeeze()
            else:
                weights = pd.Series(sample_weight)
            if split_y["train"].index.difference(weights.index).size > 0:
                print(
                    "\n\nWeights index differs from training index!!!\n\n{}".format(
                        split_y["train"].index.difference(weights.index)
                    )
                )
                print(weights.index)
                weights.index = split_y["train"].index
            fit_est = model_to_fit.fit(
                split_X["train"],
                split_y["train"],
                sample_weight=weights[split_y["train"].index],
            )
        else:
            if warn_no_weights:
                print(
                    "Sample weight not used to fit model: {}.".format(
                        estimator.__repr__(20)
                    )
                )
                print(**kwargs)
                # traceback.print_stack()
                # raise ValueError

            fit_est = model_to_fit.fit(X=split_X["train"], y=split_y["train"].squeeze())
        for fname in results.keys():
            for split_set in results[fname].keys():
                # print(getattr(fit_est, fname)(X=split_X[split_set]))
                result_df = pd.DataFrame(
                    getattr(fit_est, fname)(X=split_X[split_set]),
                    index=split_X[split_set].index,
                ).squeeze()
                if isinstance(result_df, pd.Series):
                    result_df.name = split_set
                results[fname][split_set].append(result_df)

        i += 1
    return results, test_idx_list

    """    
        dev_score_df, eva_score_df = combine_scores(
        cv_dev_score_dict,
        cv_eval_score_dict,
        dev_score_summary,
        eva_score_summary,
        score_name_list,
    )
    if score_dir is not None:
        dev_score_path = "{}dev_score_summary.csv".format(score_dir)
        eva_score_path = "{}eval_score_summary.csv".format(score_dir)
        dev_score_df.to_csv(dev_score_path)
        eva_score_df.to_csv(eva_score_path)
    """


def combine_scores(
    cv_dev_score_dict,
    cv_eval_score_dict,
    dev_score_summary,
    eva_score_summary,
    score_name_list,
):
    for score_name in score_name_list:
        dev_score_summary[score_name] = list()
        eva_score_summary[score_name] = list()
        for stat_name, stat in zip(
            ["Mean", "StDev", "Median", "Min", "Max"],
            [np.mean, np.std, np.median, np.min, np.max],
        ):
            dev_score_summary[stat_name].append(
                "{:.5f}".format(stat([[s] for s in cv_dev_score_dict[score_name]]))
            )
            eva_score_summary[stat_name].append(
                "{:.5f}".format(stat([[s] for s in cv_eval_score_dict[score_name]]))
            )
    dev_df = pd.DataFrame.from_records(
        data=cv_dev_score_dict,
        index=[cv_dev_score_dict.keys()],
        columns=["Mean", "StDev", "Median", "Min", "Max"],
    )
    eva_df = pd.DataFrame.from_records(
        data=cv_eval_score_dict,
        index=[cv_eval_score_dict.keys()],
        columns=["Mean", "StDev", "Median", "Min", "Max"],
    )
    return dev_df, eva_df


def score_model(
    estimator,
    feature_df,
    labels,
    scorer_tuple=None,
    scalar_only=True,
    check_bounds=False,
):
    # Returns a dictionary of scores for a model, given the ground truth.
    cv_score_dict = defaultdict()
    preds = estimator.predict(feature_df)
    # select_dev_prob = fitted.predict_proba(X_dev)
    if scorer_tuple is None:
        # score_dict = get_pred_score_funcs()
        scorer_tuple = ("Balanced Accuracy", balanced_accuracy_score)
        # scorer_tuple = ("MCC", matthews_corrcoef)
        # for score_name, score_obj in scorer_tuple:
    score_name, score_obj = scorer_tuple
    score = score_obj(labels, preds)
    if (not scalar_only or np.isscalar(score)) and np.real(score):
        cv_score_dict[score_name] = score
    if check_bounds:
        check_model_bounds(cv_score_dict)
    return cv_score_dict


def collate_score_dicts(score_dict_list):
    # Rearrange a list of model results into a dictionary whose keys are score names and values are lists of model scores.
    assert len(score_dict_list) > 0
    collated = dict.fromkeys(score_dict_list[1].keys(), list())
    # print("Score dict list: {}".format(score_dict_list))
    for score_dict in score_dict_list:
        for k, v in score_dict.items():
            collated[k].append(v)
    # [print("Collated {}: {}".format(k, c)) for k, c in collated.items()]
    check_model_bounds(collated)
    return collated


def check_model_bounds(collated_scores):
    # Checks that metric scores fall within their range of possible value. For debugging erroneous score results.
    bounds_dict = get_score_bounds()
    for name, scores in collated_scores.items():
        bounds = bounds_dict[name]
        if np.isscalar(scores):
            values = [scores]
        else:
            values = scores
        for v in values:
            if bounds[0] > v > bounds[1]:
                print("{} score  of {} is out of bounds!".format(name, v))
                raise ValueError
    return True


def summarize_scores(score_dict_list, check_scores=True):
    # Calculates summary statistics of CV runs.
    assert len(score_dict_list) > 0
    collated_scores = collate_score_dicts(score_dict_list)
    score_series, score_series_dict = list(), dict()
    summary_zip = dict(
        zip(
            ["Mean", "GeoMean", "StDev", "Median", "Min", "Max"],
            [np.mean, gmean, np.std, np.median, np.min, np.max],
        )
    )
    # print(collated_scores)
    for k, v in collated_scores.items():
        if not np.iterable(v) or not all(
            [isinstance(val, numbers.Number) for val in v]
        ):
            print("Collated values for {} are not scalars: {}".format(k, v))
            continue
        # print(v)
        # print("Dictionary values: {}".format([(name, s(v)) for name, s in summary_zip.items()]))
        score_summary = defaultdict(
            None, [(name, s(v)) for name, s in summary_zip.items()]
        )
        score_ser = pd.Series(score_summary, name=k)
        # print("Calculating {} summary: \n{}".format(k, score_ser))
        if not score_ser.empty:
            score_series.append(score_ser)
            score_series_dict[k] = score_ser
        else:
            print("No scores found for {}: {}".format(k, v))
            raise RuntimeError
    assert len(score_series) > 0 and not any([a.empty for a in score_series])
    score_series_dict = pd.DataFrame.from_dict(score_series_dict, orient="columns")
    score_df = pd.concat(score_series, axis=1)
    if check_scores:
        check_model_bounds(score_df.to_dict(orient="list"))
    # pprint.pp(score_series)
    # pprint.pp(score_series_dict)
    # pprint.pp(score_df)
    assert not score_df.empty
    return score_df


def learn_curve(estimator, name, feature_df, labels, fname_stub=None):
    # Calculates and plots the learning curve (Training set size  vs. model performance).
    fig, ax = plt.subplots(figsize=(5, 6), dpi=600)
    common_params = {
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "scoring": make_scorer(matthews_corrcoef),
        #  "cv":                StratifiedKFold(),
        "score_type": "both",
        "n_jobs": -1,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Matthews Correlation Coeff",
    }
    lcd = LearningCurveDisplay.from_estimator(
        estimator, X=feature_df, y=labels, **common_params, ax=ax
    )
    print(lcd.train_scores, lcd.test_scores)
    ax.legend(["Training Score", "Test Score"])
    ax.set_title("Learning Curve for {}".format(name))
    ax.set_ylim(bottom=0.0, top=1.0)
    plt.savefig(fname="{}.svg".format(fname_stub), transparent=True)
    with open("{}.pkl".format(fname_stub), "wb") as f:
        pickle.dump(lcd, f)
    return lcd, fig, ax
