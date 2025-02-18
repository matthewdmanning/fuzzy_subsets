import logging
import os
import pickle
import pprint
from copy import deepcopy
from numbers import Number

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import clone, linear_model
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    make_scorer,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

import correlation_filter
import cv_tools
import padel_categorization
import scoring
import vapor_pressure_selection
from correlation_filter import get_correlations
from data_tools import get_query_data
from descriptor_processing import epa_chem_lookup_api, get_standardizer


def get_confusion_weights():
    return np.array([[1.0, -0.5, -1.0], [0.1, 1.0, 0.0], [0.0, 0.25, 1.0]])


def three_class_solubility(y_true, y_pred, sample_weight=None, **kwargs):
    # For balanced accuracy, with W = I: np.diag(C) = np.sum(C * W)
    # In MCC, W = 2 * I - 1 (ie. off diagonals are -1 instead of 0)
    W = get_confusion_weights()
    try:
        C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    except UserWarning:
        print("True, Predicted, and Confusion Weighting")
        print(
            "\ny_pred contains classes not in y_true:\n{}\n".format(
                np.argwhere(np.astype(np.isnan(C), np.int16))
            )
        )
        C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    # with np.errstate(divide="ignore", invalid="ignore"):
    with np.errstate(divide="print", invalid="print"):
        per_class = np.sum(C * W, axis=1) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        raise UserWarning.add_note(
            "\ny_pred contains classes not in y_true:\n{}\n".format(
                np.argwhere(np.astype(np.isnan(per_class), np.int16))
            )
        )
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    return score


def optimize_tree(feature_df, labels, model, scoring, cv):
    param_grid = {
        "max_depth": [None, 25, 20, 15],
        "min_impurity_decrease": [0, 0.005, 0.01],
        "max_leaf_nodes": [100, 150, 200, 250],
    }
    tree_optimized = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=-1,
        cv=cv,
        return_train_score=True,
        error_score="raise",
    )
    tree_optimized.fit(feature_df, labels)
    print(tree_optimized.best_params_)
    return tree_optimized


def safe_mapper(x, map):
    if x in map.keys():
        return map[x]
    else:
        return x


def plot_model_scores(feature_df, train_labels, score_tups, model, subsets, path_dict):
    results_list = list()
    for i, best_features in enumerate(subsets):
        results_correct = scoring.cv_model_generalized(
            model,
            feature_df[best_features],
            train_labels,
            scorer_tups=score_tups,
            return_train=True,
        )
        exploded = _format_cv_generalized_score(
            results_correct, i, hue_category="Correct"
        )
        # exploded.reset_index(level=0, names="Test/Train", inplace=True)
        print("Exploded:\n{}".format(pprint.pformat(exploded)))
        results_list.append(exploded)
        rand_results = scoring.score_randomized_classes(
            estimator=model,
            feature_df=feature_df[best_features],
            labels=train_labels,
            scorer_tups=score_tups,
            return_train=True,
        )
        rexploded = _format_cv_generalized_score(
            rand_results, i, hue_category="Randomized"
        )

        """
        randresults_dict_df = pd.DataFrame.from_dict(rand_results, orient="index")
        rexploded = pd.concat(
            [
                ser.explode()
                .reset_index()
                .melt(id_vars="index", value_name="Score", ignore_index=False)
                .reset_index(drop=True)
                for col, ser in randresults_dict_df.items()
            ],
            keys=randresults_dict_df.columns,
        )
        rexploded.insert(loc=0, column="Subset", value=i)
        rexploded.insert(loc=0, column="Labels", value="Randomized")
        """
        # rexploded.reset_index(level=0, names="Test/Train", inplace=True)
        print("Exploded:\n{}".format(pprint.pformat(rexploded)))
        results_list.append(rexploded)
    all_results = pd.concat(results_list)
    pprint.pprint(all_results)
    plot = sns.catplot(
        all_results,
        x="Subset",
        y="Score",
        hue="Labels",
        col="index",
        row="variable",
        errorbar="se",
    )
    return all_results, plot


def _format_cv_generalized_score(results_correct, i, hue_category=None):
    results_dict_df = pd.DataFrame.from_dict(results_correct, orient="index")
    exploded = pd.concat(
        [
            ser.explode()
            .reset_index()
            .melt(id_vars="index", value_name="Score", ignore_index=False)
            .reset_index(drop=True)
            for col, ser in results_dict_df.items()
        ],
        keys=results_dict_df.columns,
    )
    exploded.insert(loc=0, column="Subset", value=i)
    if hue_category is not None:
        exploded.insert(loc=0, column="Labels", value=hue_category)
    return exploded


def main():
    select_params = _set_params()
    path_dict = _set_paths()
    path_dict["xc_path"].replace(".pkl", "_{}.pkl".format(select_params["xc_method"]))
    path_dict["corr_path"].replace(
        ".pkl", "_{}.pkl".format(select_params["corr_method"])
    )
    train_df, train_labels, test_df, test_labels, best_corrs, cross_corr = (
        _get_solubility_data(path_dict, select_params, conc_splits=(9.9))
    )
    print(
        "Discretized Labels: Value Counts:\n{}".format(
            pprint.pformat(train_labels.value_counts())
        )
    )
    print("Ring descriptors: \n{}".format(c for c in train_df.columns if "Ring" in c))
    # Get Candidate Features and Models.
    # if os.path.isfile(path_dict["search_features_path"]):
    #    search_features = pd.read_csv(path_dict["search_features_path"]).index.tolist()
    candidate_features = vapor_pressure_selection.get_search_features(train_df)
    candidate_features = candidate_features + ["nG8Ring", "nG8HeteroRing"]
    search_features = train_df.columns[train_df.columns.isin(candidate_features)]
    search_features = search_features.drop(["SsSH"], errors="ignore")
    print("Ring features: {}".format([f for f in search_features if "Ring" in f]))
    # Phosphorus is over-represented in EPA data and may bias results if organophosphates tested were more/less soluble.
    print("Dropping phosphorus and flourine features")
    for f in search_features:
        if "sP" in f or "nP" == f:
            print(f)
            search_features.drop(f)
        elif "nf" == f or "sf" in f:
            print(f)
            search_features.drop(f)
    search_features = train_df.columns
    print("{} features to select from.".format(len(search_features)))
    best_corrs, cross_corr = get_correlations(
        train_df[search_features],
        train_labels,
        path_dict["corr_path"],
        path_dict["xc_path"],
    )
    # search_features.to_series().to_csv(path_dict["search_features_path"])
    name_model_dict = get_multilabel_models(select_params["scoring"])
    print(name_model_dict)
    # Get subsetes from training loop
    print("Print Cross_corrs")
    print(cross_corr)
    n_subsets = 5
    model_scores_dict, model_subsets_dict = select_subsets_from_model(
        train_df[search_features],
        train_labels,
        n_subsets,
        name_model_dict,
        best_corrs,
        cross_corr,
        path_dict["exp_dir"],
        select_params,
    )
    cv_scores = pd.DataFrame.from_dict(model_scores_dict, orient="index")
    print(cv_scores, flush=True)
    # Get randomized label scores.
    # random_dev_scores = dict().fromkeys(name_model_dict.keys(), list())
    # random_eval_scores = dict().fromkeys(name_model_dict.keys(), list())
    score_tups = (
        ("MCC", make_scorer(matthews_corrcoef)),
        ("Balanced Acc", make_scorer(balanced_accuracy_score)),
    )
    # ("Confusion Matrix", ConfusionMatrixDisplay.from_estimator), ("Det", DetCurveDisplay.from_estimator), ("ROC", RocCurveDisplay.from_estimator))
    score_tup_dict = dict([(k[0], dict()) for k in score_tups])
    # results_dict = {"Feature Selection": list(), "Randomized Label": score_tup_dict}
    # results_dict = dict().fromkeys(name_model_dict.keys(), score_tup_dict) # {"Feature Selection": list(), "Randomized Label": score_tup_dict}
    results_dict = dict()
    plot_dict = dict()
    for n, m in name_model_dict.items():
        results_dict[n], plot_dict[n] = plot_model_scores(
            train_df,
            train_labels,
            score_tups,
            name_model_dict[n],
            model_subsets_dict[n],
            path_dict,
        )
        # score_plot.figure.set(title="{}".format(model_name), ylabel="Score")
        plot_dict[n].savefig("{}{}_score_plots.svg".format(path_dict["exp_dir"], n))
    # print(pd.DataFrame.from_dict(results_dict, orient="index"))
    exit()
    return (
        (train_df, test_df),
        (train_labels, test_labels),
        preprocessor,
        model_subsets_dict,
        model_scores_dict,
    )


def _get_solubility_data(path_dict, select_params, conc_splits, from_disk=True):
    if from_disk and all(
        [
            os.path.isfile(f)
            for f in [
                path_dict["prepped_train_df_path"],
                path_dict["prepped_test_df_path"],
                path_dict["prepped_train_labels"],
                path_dict["prepped_test_labels"],
            ]
        ]
    ):
        train_df = pd.read_pickle(path_dict["prepped_train_df_path"])
        test_df = pd.read_pickle(path_dict["prepped_test_df_path"])
        train_labels = pd.read_pickle(path_dict["prepped_train_labels"])
        test_labels = pd.read_pickle(path_dict["prepped_test_labels"])
        best_corrs, cross_corr = get_correlations(
            train_df, train_labels, path_dict["corr_path"], path_dict["xc_path"]
        )
        # epa_labels = pd.read_pickle(train_label_path)
        # with open(path_dict["transformer_path"], "rb") as f:
        #    preprocessor = pickle.load(f)
    else:
        # Get data
        insol_labels, maxed_sol_labels, sol100_labels = get_query_data()
        combo_labels, convert_df = get_conversions(
            maxed_sol_labels, path_dict["lookup_path"]
        )
        smiles_df, sid_to_key = standardize_smiles(
            convert_df, combo_labels, path_dict["std_pkl"]
        )
        max_conc = combo_labels.set_index(keys="inchiKey", drop=True)[
            "sample_concentration"
        ].astype(np.float32)
        max_conc.index = max_conc.index.map(
            lambda x: safe_mapper(x, sid_to_key), na_action="ignore"
        )
        # transformed_df = pd.read_pickle(path_dict["transformed_df_path"])
        # print(transformed_df)
        raw_df = pd.read_pickle(path_dict["desc_path"]).dropna()
        # TODO: Output list of compounds that failed to convert (INCHI, DTXSIDs, and SMILES)
        logger.debug("Missing labels")
        logger.debug(max_conc[~max_conc.index.isin(raw_df.index)])
        logger.debug("Missing raw descriptors:")
        logger.debug(raw_df[~raw_df.index.isin(combo_labels["inchiKey"])])
        epa_labels = label_solubility_clusters(
            max_conc[raw_df.index], path_dict["exp_dir"], conc_splits=conc_splits
        )
        train_raw_df, test_raw_df = train_test_split(
            raw_df, test_size=0.2, stratify=epa_labels.squeeze(), random_state=0
        )
        raw_df.index = raw_df.index.map(
            lambda x: safe_mapper(x, sid_to_key), na_action="ignore"
        )
        train_labels = epa_labels[train_raw_df.index]
        test_labels = epa_labels[test_raw_df.index]
        preprocessor, scaler, cross_corr = (
            vapor_pressure_selection.get_standard_preprocessor(
                transform_func="asinh", corr_params=select_params
            )
        )
        train_df = preprocessor.fit_transform(train_raw_df, train_labels)
        test_df = preprocessor.transform(test_raw_df)
        logger.debug("After transformer: {}".format(train_df))
        best_corrs = correlation_filter.calculate_correlation(
            train_df, train_labels, method=select_params["corr_method"]
        )
        with open(path_dict["transformer_path"], "wb") as f:
            pickle.dump(preprocessor, f)
        train_labels.to_pickle(path_dict["prepped_train_labels"])
        test_labels.to_pickle(path_dict["prepped_test_labels"])
        train_df.to_pickle(path_dict["prepped_train_df_path"])
        test_df.to_pickle(path_dict["prepped_test_df_path"])
    # print(train_df.drop(index=train_df.dropna().index))
    return train_df, train_labels, test_df, test_labels, best_corrs, cross_corr


def _set_params():
    select_params = {
        "corr_method": "kendall",
        "xc_method": "pearson",
        "max_features_out": 40,
        "min_features_out": 10,
        "n_vif_choices": 10,
        "fails_min_vif": 100,
        "fails_min_perm": 6,
        "fails_min_sfs": 4,
        "features_min_vif": 100,
        "features_min_perm": 8,
        "features_min_sfs": 15,
        "thresh_reset": -0.05,
        "thresh_vif": 20,
        "thresh_perm": 0.0025,
        "thresh_sfs": -0.005,
        "thresh_sfs_cleanup": 0.005,
        "thresh_xc": 0.95,
        "max_trials": 100,
        "cv": StratifiedKFold,
        "importance": False,
        # "scoring": make_scorer(three_class_solubility),
        "scoring": make_scorer(matthews_corrcoef),
        "score_name": "mcc",
        "W_confusion": get_confusion_weights(),
    }
    return select_params


def _set_paths():
    # Paths.
    parent_dir = "{}binary_new_epa_data/all_features/".format(
        os.environ.get("MODEL_DIR")
    )
    data_dir = "{}multiclass_weighted_all_preprocess/".format(
        os.environ.get("MODEL_DIR")
    )
    os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    path_dict = dict(
        [
            ("parent_dir", parent_dir),
            ("std_pkl", "{}stdizer_epa_query.pkl".format(data_dir)),
            ("lookup_path", "{}lookup.csv".format(data_dir)),
            ("desc_path", "{}padel_features_output_max_sol.pkl".format(data_dir)),
            ("transformer_path", "{}transformer.pkl".format(data_dir)),
            (
                "transformed_df_path",
                "{}transformed_epa_exact_sol_data.pkl".format(data_dir),
            ),
            ("prepped_train_df_path", "{}prepped_train_df.pkl".format(data_dir)),
            ("prepped_test_df_path", "{}prepped_test_df.pkl".format(data_dir)),
            ("xc_path", "{}xc_df.pkl".format(data_dir)),
            ("prepped_train_labels", "{}prepped_train_labels.pkl".format(parent_dir)),
            ("prepped_test_labels", "{}prepped_test_labels.pkl".format(parent_dir)),
            ("corr_path", "{}corr_df.pkl".format(parent_dir)),
            # Warning: This file may contain asinh transformed data for clustering algorithm input.
            # ("epa_labels", "{}epa_transformed.csv".format(parent_dir)),
            ("test_label_path", "{}test_labels.csv".format(parent_dir)),
            ("search_features_path", "{}search_features.csv".format(parent_dir)),
            ("exp_dir", "{}mcc/".format(parent_dir)),
        ]
    )
    os.makedirs(path_dict["exp_dir"], exist_ok=True)
    return path_dict


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
    """
    model_subsets_dict, model_scores_dict, subset_predicts = dict(), dict(), dict()

    dev_labels, eval_labels = train_test_split(
        train_labels,
        test_size=0.2,
        random_state=0,
        shuffle=True,
        stratify=train_labels,
    )
    dev_df = train_df.loc[dev_labels.index]
    eval_df = train_df.loc[eval_labels.index]
    best_corrs, cross_corr = get_correlations(
        dev_df, dev_labels, corr_path, xc_path, select_params
    )
    n_subsets = 10
    for m, n in zip(model_list, name_list):
        model_dir = "{}/{}/".format(exp_dir, n)
    search_features = feature_df.columns.tolist()
    for n, m in name_model_dict.items():
        model_dir = "{}{}/".format(exp_dir, n)
        os.makedirs(model_dir, exist_ok=True)
        remaining_features = deepcopy(
            [
                x
                for x in search_features
                if all(
                    [x in a.index.tolist() for a in [dev_df.T, best_corrs, cross_corr]]
                    [
                        x in a.index.tolist()
                        for a in [feature_df.T, label_corr, cross_corr]
                    ]
                )
            ]
        )
        print("Remaining features: {}".format(len(remaining_features)))
        model_subsets_dict[n], model_scores_dict[n], subset_predicts[n] = (
            list(),
            list(),
            list(),
        )
        for i in np.arange(n_subsets):
            print("Selecting from {} features.".format(len(remaining_features)))
            subset_dir = "{}subset{}/".format(model_dir, i)
            os.makedirs(subset_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            if os.path.isfile("{}best_features_{}.csv".format(subset_dir, n)):
                top_score = 0
                with open("{}feature_score_path.csv".format(subset_dir), "r") as f:
                    for whole_line in f.readlines():
                        line = whole_line.split()
                        cv_results = [float(x) for x in line[:5]]
                        feats = [str(x) for x in line[5:]]
                        if np.mean(cv_results) - np.std(cv_results) > top_score:
                            top_score = np.mean(cv_results) - np.std(cv_results)
                            best_features = feats
            else:
                print(remaining_features)
                cv_results, cv_predicts, best_features = train_multilabel_models(
                    feature_df[remaining_features],
                    labels,
                    model=m,
                    model_name=n,
                    best_corrs=label_corr[remaining_features],
                    cross_corr=cross_corr[remaining_features].loc[remaining_features],
                    select_params=select_params,
                    save_dir=subset_dir,
                )
                subset_predicts[n].append(cv_predicts)
            model_subsets_dict[n].append(best_features)
            model_scores_dict[n].append(cv_results)
    # cv_scores.mean().sort_values(ascending=False)
    print(cv_scores, flush=True)
    ax = sns.catplot(cv_scores)
    ax.figure.set_title(n)
    ax.set_axis_labels(x_var="Subset No.", y_var="Model-Subset Score (Adjusted BAC)")
    ax.savefig("{}test_scores.png".format(exp_dir))
    plt.show()

    # cmd = ConfusionMatrixDisplay.from_predictions(y_true=train_labels, y_pred=subset_predicts[n])
    # cmd.plot()
    # cmd.figure_.savefig("{}confusion_matrix_cv.png".format(exp_dir))

    for n in name_list:
        pprint.pprint(list(zip(model_subsets_dict[n], model_scores_dict[n])))
    # test_df = preprocessor.transform(test_raw_df, test_labels)
    return (
        (train_df, test_df),
        (train_labels, test_labels),
        preprocessor,
        model_subsets_dict,
        model_scores_dict,
    )
            [
                remaining_features.remove(f)
                for f in best_features
                if f in remaining_features
            ]
    # cv_scores = pd.DataFrame.from_records(model_scores_dict, orient="index", columns=np.arange(n_subsets))
    return model_scores_dict, model_subsets_dict


def get_conversions(maxed_sol_labels, lookup_path):
    if os.path.isfile(lookup_path):
        convert_df = pd.read_csv(lookup_path)
        if True:
            convert_df.drop(
                columns=[x for x in convert_df.columns if "Unnamed" in str(x)],
                inplace=True,
            )
            convert_df.drop(columns="mol", inplace=True, errors="ignore")
            convert_df.to_csv(lookup_path)
    else:
        convert_df = lookup_chem(comlist=maxed_sol_labels.index.tolist())
        convert_df.drop(columns="mol", inplace=True, errors="ignore")
        convert_df.to_csv(lookup_path)
        convert_df.drop(
            columns=["bingoId", "name", "casrn", "formula", "cid"],
            inplace=True,
            errors="ignore",
        )
    combo_labels = pd.concat(
        [convert_df, maxed_sol_labels.reset_index(names="label_id")], axis=1
    )
    failed_lookup = combo_labels[combo_labels["id"] != combo_labels["label_id"]]
    combo_labels.drop(index=failed_lookup.index, inplace=True)
    nosmiles = combo_labels[
        (combo_labels["smiles"] == "nan") | combo_labels["smiles"].isna()
    ]
    combo_labels.drop(index=nosmiles.index, inplace=True)
    return combo_labels, convert_df


def train_multilabel_models(
    dev_data,
    eval_data,
    model,
    model_name,
    best_corrs,
    cross_corr,
    select_params,
    save_dir,
):
    dev_df, dev_labels = dev_data
    eval_df, eval_labels = eval_data
    selection_models = {
        "predict": model,
        "permutation": model,
        "importance": model,
        "vif": linear_model.LinearRegression(),
    }
    model_dict, score_dict, dropped_dict, best_features = (
        vapor_pressure_selection.select_feature_subset(
            dev_df,
            dev_labels,
            target_corr=best_corrs,
            cross_corr=cross_corr,
            select_params=select_params,
            selection_models=selection_models,
            hidden_test=(eval_df, eval_labels),
            save_dir=save_dir,
        )
    )
    print("Best features!")
    short_to_long = padel_categorization.padel_short_to_long()
    best_features_long = short_to_long[best_features].tolist()
    print("\n".join(best_features_long))
    pd.Series(best_features_long).to_csv(
        "{}best_features_{}.csv".format(save_dir, model_name)
    )
    from sklearn.model_selection import cross_val_predict

    with sklearn.config_context(
        enable_metadata_routing=False, transform_output="pandas"
    ):
        cv_predicts = cross_val_predict(model, dev_df[best_features], dev_labels)
    cv_results = balanced_accuracy_score(
        y_true=dev_labels, y_pred=cv_predicts, adjusted=True
    )
    # model.fit(train, cmd_train_labels)
    return cv_results, cv_predicts, best_features




def get_multilabel_models(scorer, meta_est=False):
    best_tree = RandomForestClassifier(
        bootstrap=False,
        max_leaf_nodes=200,
        min_impurity_decrease=0.005,
        class_weight="balanced",
        random_state=0,
    )
    xtra_tree = ExtraTreesClassifier(
        max_leaf_nodes=200,
        min_impurity_decrease=0.005,
        class_weight="balanced_subsample",
        random_state=0,
    )
    lrcv = LogisticRegressionCV(
        scoring=scorer, class_weight="balanced", max_iter=5000, random_state=0
    )
    if not meta_est:
        model_list = [best_tree, lrcv, xtra_tree]
        name_list = ["RandomForest", "Logistic", "ExtraTrees"]
    else:
        ovo_tree = OneVsOneClassifier(estimator=best_tree)
        ovo_lr = OneVsOneClassifier(estimator=lrcv)
        ovr_tree = OneVsRestClassifier(estimator=best_tree)
        ovr_lr = OneVsRestClassifier(estimator=lrcv)
        model_list = [ovo_tree, ovo_lr, ovr_tree, ovr_lr]
        name_list = ["ovo_tree", "ovo_logit", "ovr_tree", "ovr_logit"]
    return model_list, name_list


def label_solubility_clusters(labels, exp_dir, algo=False):
    if algo:
        from sklearn.cluster import BisectingKMeans

        trimmed_labels = (
            labels.clip(upper=110.0, lower=4.5).multiply(100).apply(np.asinh)
        )
        print(
            "Asinh: {:.4f} {:.4f} {:.4f} {:.4f}".format(
                np.asinh(5), np.asinh(10), np.asinh(50), np.asinh(100)
            )
        )
        print(trimmed_labels.describe())
        # agg_clusters = AgglomerativeClustering(linkage="ward", metric="euclidean", n_clusters=3).fit_predict(trimmed_labels.to_frame())
        agg_clusters = BisectingKMeans(
            n_clusters=4, max_iter=5000, bisecting_strategy="largest_cluster"
        ).fit_predict(trimmed_labels.to_frame())
        agg_clusters = pd.Series(agg_clusters, index=trimmed_labels.index)
        for n in agg_clusters.sort_values().unique():
            cluster = labels.sort_values()[agg_clusters == n]
            print(labels[cluster.index].min(), labels[cluster.index].max())
        """
        label_clusterer = (
            KMeans(n_clusters=6, random_state=0)
            .set_output(transform="pandas")
            .fit(trimmed_labels.to_frame())
        )
        epa_labels = pd.Series(label_clusterer.predict(trimmed_labels.to_frame()), index=trimmed_labels.index)
        """
        epa_labels = agg_clusters
    else:
        epa_labels = labels.astype(np.float32).sort_values()
        print(epa_labels.describe())
        splits = (9.5, 95.0)
        epa_labels[epa_labels < splits[0]] = 0
        epa_labels[(splits[0] <= epa_labels) & (epa_labels < splits[1])] = 1
        epa_labels[splits[1] <= epa_labels] = 2
        epa_labels.astype(dtype=np.int8)
        print(splits)
    # from sklearn.preprocessing import MultiLabelBinarizer
    # epa_labels = MultiLabelBinarizer(classes=epa_labels).fit_transform(epa_labels)

    print("Cluster-assigned labels")
    print(epa_labels.value_counts(sort=False), flush=True)
    return epa_labels


def standardize_smiles(feature_df, combo_labels, maxed_sol_labels, std_pkl):
    if os.path.isfile(std_pkl):
        smiles_df = pd.read_pickle(std_pkl)
        if False:
            clean_list = [x for x in smiles_df.tolist() if "sid" in x.keys()]
            unregistered_list = [
                x
                for x in smiles_df.to_dict()
                if "sid" not in x.keys() and "smiles" in x.keys()
            ]
            failed_list = [x for x in smiles_df.tolist() if "smiles" not in x.keys()]
            clean_df = pd.json_normalize(clean_list)
            unregistered_df = pd.json_normalize(unregistered_list)
            smiles_df = pd.concat([clean_df, unregistered_df], ignore_index=True)
            failed_df = pd.json_normalize(failed_list)
            if not failed_df.empty:
                failed_df.to_csv("{}failed_standardizer.csv".format(failed_df))
            assert not smiles_df.empty
            smiles_df.to_pickle(std_pkl)
            smiles_df.to_csv(std_path)
            if not os.path.isfile("{}failed_standardizer.csv".format(failed_df)):
                # TODO: Extract failed compounds by difference of inputs and outputs.
                pass

    else:
        smiles_df, failed_df = get_standardizer(comlist=feature_df["smiles"].tolist())
        if not failed_df.empty:
            failed_df.to_csv("{}failed_standardizer.csv".format(failed_df))
        assert not smiles_df.empty
        smiles_df.to_pickle(std_pkl)
    smiles_df.drop(
        columns=[
            "cid",
            "casrn",
            "name",
            "canonicalSmiles",
            "inchi",
            "mol",
            "formula",
            "id",
        ],
        inplace=True,
        errors="ignore",
    )
    assert not smiles_df.empty
    smiles_df = smiles_df[smiles_df["chemId"].isin(combo_labels["id"])]
    """
        if "smiles" in smiles_df.columns:
            smiles_df = smiles_df[~smiles_df.duplicated() & ~smiles_df["smiles"].isna()]
            # smiles_df.set_index(keys="sid", drop=True, inplace=True)
            print("Standardizer output:\n{}".format(smiles_df))
            print(smiles_df.columns)
            sid_to_smiles = smiles_df["smiles"].squeeze().to_dict()
            sid_to_key = smiles_df["inchiKey"].squeeze().to_dict()
            smiles_ind_df = smiles_df.reset_index().set_index(keys="smiles")
            smiles_to_sid = smiles_ind_df["sid"].squeeze().to_dict()
            smiles_fails = pd.Index(
                x for x in maxed_sol_labels.index if x not in sid_to_key.values()
            )
        else:
            smiles_fails = maxed_sol_labels.index
        # Get Descriptors.
        print("Raw label to InchiKey failures:\n{}".format(smiles_fails))
        raw_retry_df, retry_failures_df = get_standardizer(smiles_fails.tolist())
        if not raw_retry_df.empty and "smiles" in raw_retry_df.columns:
            smiles_df = pd.concat([smiles_df, raw_retry_df[~raw_retry_df["smiles"].isna()]])
            smiles_df.drop(index=smiles_df[~smiles_df["error"].isna()].index, inplace=True)
            smiles_df = smiles_df[~smiles_df.index.duplicated()]
            smiles_df.to_csv("{}updated_std_output.csv".format(exp_dir))
            sid_to_key = smiles_df["inchiKey"].squeeze().to_dict()
            smiles_fails = pd.Index(
                x for x in maxed_sol_labels.index if x not in sid_to_key.keys()
            )
            print("Retry label to InchiKey failures:\n{}".format(smiles_fails))
            # print(smiles_df.loc[smiles_fails])
        else:
            print("No return from retrying Standardizer on failed labels.")
            if "smiles" in smiles_df.columns:
                smiles_df.to_csv("{}updated_std_output.csv".format(exp_dir))
        smiles_fails.difference(smiles_df.index).to_series().to_csv(
            "{}failed_standardization.csv".format(exp_dir)
        )
        maxed_sol_labels.index = maxed_sol_labels.index.map(sid_to_key, na_action="ignore")
        """
    sid_to_key = (
        smiles_df[["inchiKey", "chemId"]]
        .set_index(keys="chemId", drop=True)
        .squeeze()
        .to_dict()
    )
    id_to_orig_smi = (
        smiles_df[["chemId", "smiles"]]
        .set_index(keys="chemId", drop=True)
        .squeeze()
        .to_dict()
    )
    smiles_df.set_index(keys="inchiKey", drop=True, inplace=True)
    return smiles_df, sid_to_key




def get_standardizer(comlist, batch_size=100, input_type="smiles"):
    api_url = "https://ccte-cced-cheminformatics.epa.gov/api/stdizer"
    response_list, failed_list = list(), list()
    auth_header = {"x-api-key": os.environ.get("INTERNAL_KEY")}
    with requests.session() as r:
        for c in comlist:
            params = {"workflow": "qsar-ready", input_type: c}
            response = r.request(
                method="GET", url=api_url, params=params, headers=auth_header
            )
            if isinstance(response.json(), list) and len(response.json()) > 0:
                response_list.append(response.json()[0])
            else:  # isinstance(response, (list, str)):
                failed_list.append(response.content)
    clean_list = [x for x in response_list if "sid" in x.keys()]
    unregistered_list = [
        x for x in response_list if "sid" not in x.keys() and "smiles" in x.keys()
    ]
    # failed_list = [x for x in response_list if "smiles" not in x.keys()]
    clean_df = pd.json_normalize(clean_list)
    unregistered_df = pd.json_normalize(unregistered_list)
    response_df = pd.concat([clean_df, unregistered_df], ignore_index=True)
    failed_df = pd.json_normalize(failed_list)
    return response_df.drop(columns="mol"), failed_df


def get_descriptors(smi_list, batch_size=100, desc_type="padel", input_type="smiles"):
    # TODO: Add original SMILES/identifier to info_df to link original data and descriptor data through info_df.
    # stdizer = DescriptorRequestor.QsarStdizer(input_type="dtxsid")
    api_url = "https://ccte-cced-cheminformatics.epa.gov/api/padel"
    """
    {
        "chemicals": ["string"],
        "options": {
            "headers": true,
            "timeout": 0,
            "compute2D": true,
            "compute3D": true,
            "computeFingerprints": true,
            "descriptorTypes": ["string"],
            "removeSalt": true,
            "standardizeNitro": true,
            "standardizeTautomers": true,
        },
    }
   """
    info_list, desc_dict, failed_list = list(), dict(), list()
    with requests.session() as r:
        auth_header = {"x-api-key": os.environ.get("INTERNAL_KEY")}
        req = requests.Request(method="GET", url=api_url, headers=auth_header).prepare()
        for c in smi_list:
            params = {"smiles": c, "type": "padel"}
            req.prepare_url(url=api_url, params=params)
            response = r.send(req)
            if response.status_code == 200 and len(response.json()) > 0:
                single_info_dict = dict(
                    [
                        (k, v)
                        for k, v in response.json()["chemicals"][0].items()
                        if k != "descriptors"
                    ]
                )
                single_info_dict.update([("SMILES_QSAR", c)])
                info_list.append(single_info_dict)
                desc_dict[response.json()["chemicals"][0]["inchiKey"]] = (
                    response.json()["chemicals"][0]["descriptors"]
                )
            else:
                failed_list.append(response.content)
    padel_names = padel_categorization.get_short_padel_names()
    info_df = pd.json_normalize(info_list)
    info_df.set_index(keys="inchiKey", inplace=True)
    info_df.drop(columns=padel_names, inplace=True, errors="ignore")
    desc_df = pd.DataFrame.from_dict(
        data=desc_dict, orient="index", columns=padel_names
    )
    return desc_df, info_df, failed_list


if __name__ == "__main__":
    logger = logging.getLogger(name="selection")
    main()
