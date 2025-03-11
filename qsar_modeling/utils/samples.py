import copy
import itertools
import pprint
import typing

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import clone
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import compute_sample_weight
from sklearn.utils._param_validation import HasMethods

import cv_tools
from data_handling import data_tools


def model_prediction_distance(
    predicts_list, metric=mean_squared_log_error, sample_weight=None
):
    n_predicts = len(predicts_list)
    distances = np.zeros(shape=(n_predicts, n_predicts), dtype=np.float32)
    for i, j in itertools.combinations(np.arange(n_predicts)):
        distances[i, j] = distances[j, i] = metric(
            predicts_list[i], predicts_list[j], sample_weight=sample_weight
        )
    return distances


def cv_model_prediction_distance(
    feature_df,
    labels,
    name_model_subset_tup,
    dist_metric=cosine_distances,
    cv=StratifiedKFold,
    response="predict_proba",
    **kwargs
):
    test_idx_list = list()
    distance_dict = {"test": list(), "train": list()}
    predict_dict = dict()
    for name, model, subset in name_model_subset_tup:
        predict_dict[name] = {"test": list(), "train": list()}
    for train_X, train_y, test_X, test_y in cv_tools.split_df(
        feature_df, labels, splitter=cv, **kwargs
    ):
        split_X, split_y = {"train": train_X, "test": test_X}, {
            "train": train_y,
            "test": test_y,
        }
        split_X["train"] = train_X
        split_X["test"] = test_X
        test_idx_list.append(test_y.index)
        assert not split_X["train"].empty
        estimator_list = list()
        for name, estimator, subset in name_model_subset_tup:
            if (
                HasMethods("sample_weight").is_satisfied_by(estimator)
                and "sample_weight" in kwargs.keys()
            ):
                print("Using sample weights to score subsets...")
                fit_est = clone(estimator).fit(
                    split_X["train"][subset],
                    split_y["train"],
                    sample_weight=kwargs["sample_weight"].loc[train_y.index],
                )
            else:
                fit_est = clone(estimator).fit(
                    split_X["train"][subset], split_y["train"]
                )
            estimator_list.append((fit_est, subset))
            for split_set in predict_dict[name].keys():
                result_df = pd.DataFrame(
                    getattr(fit_est, response)(X=split_X[split_set][subset]),
                    index=split_X[split_set].index,
                ).squeeze()
                if isinstance(result_df, pd.Series):
                    result_df.name = split_set
                predict_dict[name][split_set].append(result_df)
    split_distplot_dict = dict()
    for split_set in distance_dict.keys():
        dist_list = list()
        for r in zip(
            [predict_dict[name][split_set] for name, _, _ in name_model_subset_tup]
        ):
            zip_list = [pd.concat(df) for df in r]
            print(zip_list)
            # dist_list.append(r)
            # dist_list.append(model_prediction_distance(r, dist_metric))
            dist_list.append(np.vstack(zip_list))
        distance_dict[split_set] = pd.DataFrame(np.hstack(dist_list)).merge(
            labels, left_index=True, right_index=True
        )
        split_distplot_dict[split_set] = sns.pairplot(data=distance_dict[split_set])
    return split_distplot_dict


def weigh_single_proba(onehot_true, probs, prob_thresholds=None):
    if prob_thresholds is not None:
        probs.clip(lower=prob_thresholds[0], upper=prob_thresholds[1], inplace=True)
    unweighted = probs.add(onehot_true, axis=0).abs()
    return unweighted


def weight_by_proba(
    y_true, probs, prob_thresholds=(0, 0.75), label_type="binary", combo_type="max"
):
    probs = copy.deepcopy(probs)
    y_true = y_true.copy()
    if isinstance(y_true.squeeze(), pd.Series) and label_type == "binary":
        onehot_labels = pd.concat([y_true, 1 - y_true], axis=1)
    else:
        onehot_labels = LabelBinarizer().fit_transform(y_true.to_frame())
    assert onehot_labels.shape == (y_true.size, y_true.nunique())
    onehot_labels.columns = ["Soluble", "Insoluble"]
    # print("Onehot:\n{}".format(pprint.pformat(onehot_labels.shape)))
    onehot_normed = onehot_labels.sub(np.ones_like(onehot_labels) / y_true.nunique())
    onehot_normed.columns = onehot_labels.columns
    if isinstance(probs, typing.Iterable) and len(probs) == 1:
        probs = probs[0]
    if isinstance(probs, pd.DataFrame):
        sample_weights_raw = weigh_single_proba(onehot_true=onehot_normed, probs=probs)
        # pd.DataFrame(data=np.zeros_like(probs.to_numpy()), index=y_true, columns=probs.columns)
        # [ (p - (1-T)/n_classes) ]
        # where: p = probs, T = one-hot encoded classes
        # onehot_labels = OneHotEncoder(categories=y_true.tolist(), sparse_output=False, dtype=np.int16).fit_transform(y_true.to_frame())

        # print("Normed:\n{}".format(pprint.pformat(onehot_normed.shape)))
    elif isinstance(probs, list) and len(probs) > 1:
        for p in probs:
            p.columns = ["Soluble", "Insoluble"]
        unweighted = pd.concat(
            [
                p.multiply(onehot_labels)
                .sum(axis=1)
                .clip(lower=prob_thresholds[0], upper=prob_thresholds[1])
                for p in probs
            ],
            axis=1,
        )
        # print("Unweighted len: {}\n".format(len(unweighted)))
        # unweighted_sols = pd.concat([p["Soluble"] for p in unweighted], axis=1)
        # unweighted_ins = pd.concat([p["Insoluble"] for p in unweighted], axis=1)
        # print("Unweighted Insolubles")
        # pprint.pp(unweighted_ins)
        print("Unweighted:")
        pprint.pp(unweighted)
        # unweighted_min = unweighted.min(axis=1)
        # unweighted_avg = unweighted.mean(axis=1)
        # print("Unweighted Minimum")
        # pprint.pp(unweighted_min)
        # print("Unweighted Average")
        # pprint.pp(unweighted_avg)
        # unweighted = unweighted.max(axis=1)
        if combo_type == "sum":
            sample_weights_raw = unweighted.sum(axis=1)
            # TODO: Rethink / allow parameterization of sample_weight normalization
            sample_weights = (
                sample_weights_raw - sample_weights_raw.min()
            ) / sample_weights_raw.max()
        elif combo_type == "max":
            sample_weights_raw = unweighted.max(axis=1)
        # if "sq" in activate:
        #     pd.Series.apply()
    else:
        print("Unknown type for probabilities...")
        print(probs)
        raise ValueError

    sample_weights = sample_weights_raw**2
    print("Sample weights:")
    pprint.pprint(sample_weights.describe())
    return sample_weights


def compare_models_to_predictions(
    feature_df, model_subsets, metric, predictions=None, response="predict"
):
    new_predicts = list()
    for m, s in model_subsets:
        if response == "predict":
            new_predicts.append(m.predict(feature_df[s]))
        else:
            new_predicts.append(m.predict_proba(feature_df[s]))
    distances = pd.concat(
        [np.corrwith(other=predictions, axis=1, method=metric) for np in new_predicts],
        keys=[model_subsets.keys()],
    )
    return distances


def get_sample_info(inchi_keys, source=None, labels=None, drop_dupes=False):
    # Returns QSAR-ready SMILES and INCHI strings for list of INCHI keys.
    prop_list = ["SMILES_QSAR", "INCHI"]
    meta_loaded = data_tools.load_metadata()
    if source is not None:
        if type(source) is not str:
            source = dict((k, v) for k, v in meta_loaded.items() if k in source)
        meta_loaded = dict([(source, meta_loaded[source])])
    if labels is not None:
        if type(labels) is not str:
            source = dict((k, v) for k, v in meta_loaded.items() if k in labels)
        meta_loaded = dict([(labels, meta_loaded[labels])])
    metadata = pd.concat([m[1][prop_list] for m in meta_loaded.values()])
    try:
        info = metadata.loc[inchi_keys]
    except:
        info = metadata[inchi_keys]
        print("Except worked!")
    return info


def get_dmso_source_label_df(inchi_keys=None, include_only=None, exclude=None):
    # Returns DataFrame listing DMSO solubility and data source for each INCHI key.
    meta_loaded = data_tools.load_metadata()
    data_dfs = list()
    for k, v in meta_loaded.items():
        if include_only is not None:
            if not any([s == k for s in include_only]):
                continue
        if exclude is not None:
            if any([s == k for s in exclude]):
                continue
        if inchi_keys is not None:
            idx = v[0].index.intersection(inchi_keys)
            if idx.empty:
                continue
        else:
            idx = v[0].index
        source, label = k.split("_")
        data_df = pd.DataFrame(index=idx)
        data_df["Source"] = source
        data_df["Solubility"] = label
        data_dfs.append(data_df)
    return pd.concat(data_dfs)


def weight_dmso_samples(
    inchi_idx, by, group_by=None, include_only=None, exclude=None, class_wt="balanced"
):
    # Returns sample weights based on solubility and data source.
    meta_df = get_dmso_source_label_df(
        inchi_idx, include_only=include_only, exclude=exclude
    )
    if group_by.lower() == "solubility" or (by.lower() == "source" and group_by):
        by_source = [
            meta_df[meta_df["Solubility" == u]] for u in meta_df["Source"].unique()
        ]
        weights = pd.concat([compute_sample_weight(class_wt, s) for s in by_source])
    elif group_by.lower() == "source" or (by.lower() == "solubility" and group_by):
        by_source = [
            meta_df[meta_df["Source" == u]] for u in meta_df["Source"].unique()
        ]
        weights = pd.concat([compute_sample_weight(class_wt, s) for s in by_source])
    elif by.lower() == "source":
        weights = compute_sample_weight(class_wt, meta_df["Source"])
    elif by.lower() == "solubility":
        weights = compute_sample_weight(class_wt, meta_df["Solubility"])
    return weights


def get_confusion_samples(true_predict_tuple, labels=(1, 0)):
    # Returns INCHI keys corresponding to True/False Positive/Negatives.
    # tn, fp, fn, tp == sklearn confusion matrix convention
    tn, fp, fn, tp = list(), list(), list(), list()
    if all([(type(t) is list or type(t) is tuple) for t in true_predict_tuple]):
        for tup in true_predict_tuple:
            if all([(type(t) is list or type(t) is tuple) for t in tup]):
                false_list = [get_confusion_samples(a) for a in true_predict_tuple]
                return false_list
            else:
                print("Something is very wrong")
                print(tup)
                raise RuntimeError
    else:
        tup = true_predict_tuple
        assert len(tup) == 2
        if labels is not None:
            bin_tuple = [t.map(dict(zip(labels, [0, 1]))) for t in tup]
        else:
            bin_tuple = tup
        diff = bin_tuple[1].subtract(bin_tuple[0])
        truth = diff[diff == 0].index
        pos = bin_tuple[0][bin_tuple[0] == 1].index
        neg = bin_tuple[0][bin_tuple[0] == 0].index
        fp = diff[diff == 1].index
        fn = diff[diff == -1].index
        tp = truth.intersection(pos)
        tn = truth.intersection(neg)
    return tn, fp, fn, tp
