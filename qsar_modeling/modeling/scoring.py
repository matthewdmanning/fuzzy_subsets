import numbers
import pickle
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import sklearn.model_selection
from matplotlib import pyplot as plt
from scipy.stats import gmean
from sklearn import clone, metrics
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
from sklearn.model_selection import LearningCurveDisplay, StratifiedKFold

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


def score_randomized_classes(
    estimator,
    feature_df,
    labels,
    cv=StratifiedKFold,
    scorer_tups=None,
    label_seed=0,
    return_train=True,
    **splitter_kws
):
    random_labels = labels.copy()
    scrambled_labels = pd.Series(
        data=random_labels.sample(frac=1.0, random_state=label_seed).to_list(),
        index=labels.index,
    )
    results = cv_model_generalized(
        estimator,
        feature_df,
        labels=scrambled_labels,
        cv=cv,
        scorer_tups=scorer_tups,
        return_train=return_train,
    )
    return results


def cv_model_generalized(
    estimator,
    feature_df,
    labels,
    cv=StratifiedKFold,
    preprocessor=None,
    scorer_tups=None,
    return_train=False,
    **splitter_kws
):
    """
    Generalized, all-in-one wrapper for cross-validated models. Desired outputs are given as nested tuples of name, callable.
    Results are returned as a dictionary of dictionaries of lists.
    Commonly used functions are given as

    Parameters
    ----------

    estimator: BaseEstimator, Unfit estimator or pipeline
    feature_df: DataFrame, features
    labels: Series, labels for supervised training
    cv: Cross-Validator, default=StratifiedKFolds
    preprocessor: TransformerMixin or implements "fit" and "transform"
    return_train: bool, whether to also calculate and return results for training data
    scorer_tups: Iterable[Iterable[str, callable]], nested iterables of function name and callable with signature (fitted estimator, X, y_train)
    splitter_kws

    Returns
    -------

    results: dict[str, dict[str, list()]], nested dictionary of callable name["test"[, "train"], list[Any]], where the list contain results from callable over all CV folds.

    """
    # Fits, predicts, and scores a model using cross-validation.
    assert not feature_df.empty
    if return_train:
        splits_dict = {"train": list(), "test": list()}
        results = dict([(k[0], {"train": list(), "test": list()}) for k in scorer_tups])
    else:
        results = dict([(k[0], {"test": list()}) for k in scorer_tups])
    # results = dict([(k[0], splits_dictcopy()) for k in scorer_tups])
    i = 0
    for train_X, train_y, test_X, test_y in cv_tools.split_df(
        feature_df, labels, splitter=cv, **splitter_kws
    ):
        split_X, split_y = {"train": train_X, "test": test_X}, {
            "train": train_y,
            "test": test_y,
        }
        if preprocessor is not None:
            preprocessor.fit(train_X)
            split_X["train"] = preprocessor.transform(train_X)
            split_X["test"] = preprocessor.transform(test_X)
        else:
            split_X["train"] = train_X
            split_X["test"] = test_X
        assert not split_X["train"].empty
        fit_est = clone(estimator).fit(split_X["train"], split_y["train"])
        for fname, func in scorer_tups:
            for split_set, score_list in results[fname].items():
                score_list.append(func(fit_est, split_X[split_set], split_y[split_set]))
        i += 1
    # print("Results for model: {}".format(estimator))
    # print(pd.json_normalize(results).T.explode(column=[0]).T)
    # [print(splits, fun, results[fun][splits]) for fun, splits in zip([f for f, fn in scorer_tups], splits_dict.keys())]
    return results

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
