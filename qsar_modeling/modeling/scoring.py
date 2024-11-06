import numbers
import pickle
import pprint
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import stats
from matplotlib import pyplot as plt
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


def get_pred_score_funcs(scalar_only=True):
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
    return pred_scores


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
    return prob_scores


def fit_score_model_cv(
    estimator,
    feature_df,
    labels,
    cv=StratifiedKFold,
    score_dir=None,
    save_model=True,
    scaler=None,
):
    # Fits, predicts, and scores a model using cross-validation.
    if score_dir is not None:
        dev_score_path = "{}dev_score_summary.csv"
        eva_score_path = "{}eva_score_summary.csv"
    else:
        dev_score_path, eva_score_path = None, None
    cv_dev_score_dict, cv_eval_score_dict = dict(), dict()
    dev_score_summary, eva_score_summary = dict(), dict()

    for score_name, score_obj in get_pred_score_funcs():
        cv_dev_score_dict[score_name] = list()
        cv_eval_score_dict[score_name] = list()

        for dev_X, dev_y, eva_X, eva_y in cv_tools.split_df(
            feature_df, labels, splitter=cv
        ):
            if scaler is not None:
                scaler.fit(dev_X)
                dev_X = scaler.transform(dev_X)
                eva_X = scaler.transform(eva_X)
            fit_est = estimator.fit(dev_X, dev_y)
            dev_predict = fit_est.predict(dev_X)
            eva_predict = fit_est.predict(eva_X)
            cv_dev_score_dict[score_name].append(score_obj(dev_y, dev_predict))
            cv_eval_score_dict[score_name].append(score_obj(eva_y, eva_predict))
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
    dev_df = pd.DataFrame.from_dict(
        data=dev_score_summary,
        orient="index",
        columns=["Mean", "StDev", "Median", "Min", "Max"],
    )
    eva_df = pd.DataFrame.from_dict(
        data=eva_score_summary,
        orient="index",
        columns=["Mean", "StDev", "Median", "Min", "Max"],
    )
    dev_df.to_csv(dev_score_path)
    eva_df.to_csv(eva_score_path)
    return dev_score_summary, eva_score_summary, cv_dev_score_dict, cv_eval_score_dict


def score_model(
    estimator,
    feature_df,
    labels,
    scorer_tuple=None,
    scalar_only=True,
    check_bounds=True,
):
    # Returns a dictionary of scores for a model, given the ground truth.
    cv_score_dict = defaultdict()
    preds = estimator.predict(feature_df)
    # select_dev_prob = fitted.predict_proba(X_dev)
    if scorer_tuple is None:
        # score_dict = get_pred_score_funcs()
        scorer_tuple = ("Balanced Accuracy", balanced_accuracy_score)
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
    collated = dict.fromkeys(score_dict_list[1].keys(), list())
    print("Score dict list: {}".format(score_dict_list))
    for score_dict in score_dict_list:
        for k, v in score_dict.items():
            collated[k].append(v)
    [print("Collated {}: {}".format(k, c)) for k, c in collated.items()]
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
    collated_scores = collate_score_dicts(score_dict_list)
    score_series, score_series_dict = list(), dict()
    summary_zip = dict(
        zip(
            ["Mean", "GeoMean", "StDev", "Median", "Min", "Max"],
            [np.mean, stats.geometric_mean, np.std, np.median, np.min, np.max],
        )
    )
    print(collated_scores)
    for k, v in collated_scores.items():
        if not np.iterable(v) or not all(
            [isinstance(val, numbers.Number) for val in v]
        ):
            print("Collated values for {} are not scalars: {}".format(k, v))
            continue
        print(v)
        print(
            "Dictionary values: {}".format(
                [(name, s(v)) for name, s in summary_zip.items()]
            )
        )
        score_summary = defaultdict(
            None, [(name, s(v)) for name, s in summary_zip.items()]
        )

        score_ser = pd.Series(score_summary, name=k)

        print("Calculating {} summary: \n{}".format(k, score_ser))
        if not score_ser.empty:
            score_series.append(score_ser)
            score_series_dict[k] = score_ser
        else:
            print("No scores found for {}: {}".format(k, v))
            raise RuntimeError
    assert len(score_series) > 0 and not any([a.empty for a in score_series])
    score_series_dict = pd.DataFrame.from_dict(score_series_dict)
    score_df = pd.concat(score_series, axis=1)
    if check_scores:
        check_model_bounds(score_df.to_dict(orient="list"))
    pprint.pp(score_series)
    pprint.pp(score_series_dict)
    pprint.pp(score_df)
    assert not score_df.empty
    return score_series_dict


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
