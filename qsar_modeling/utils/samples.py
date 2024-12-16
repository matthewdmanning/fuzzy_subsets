import pandas as pd
from sklearn.utils import compute_sample_weight

from data_handling import data_tools


def model_prediction_distance(feature_df, model_subsets, metric, response="predict"):
    predictions = list()
    for m, s in model_subsets:
        if response == "predict":
            predictions.append(m.predict(feature_df[s]))
        else:
            predictions.append(m.predict_proba(feature_df[s]))
    distance_df = pd.DataFrame(predictions).corr(method=metric)
    return distance_df


def compare_models_to_predictions(
    feature_df, model_subsets, metric, predictions=None, response="predict"
):
    new_predicts = list()
    for m, s in model_subsets:
        if response == "predict":
            new_predicts.append(m.predict(feature_df[s]))
        else:
            new_predicts.append(m.predict_proba(feature_df[s]))
    distances = pd.concat([np.corrwith(other=predictions, axis=1, method=metric) for np in new_predicts], keys=[model_subsets.keys()])
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
