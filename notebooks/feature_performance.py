import os
import pickle
import pprint
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import balancing
import data_tools
from archive.dmso_hyperparameter_part_two import (
    combined_rus_cv_results,
    cv_model_documented,
)


def hyperparam(
    feature_df,
    labels,
    out_dir,
    model_tup,
    cv=None,
    samp_wts=None,
    splitter_kws=None,
):
    if cv is None:
        cv = partial(StratifiedKFold, shuffle=True, random_state=0)
    if splitter_kws is None:
        splitter_kws = {"shuffle": True, "random_state": 0}
    os.makedirs(out_dir, exist_ok=True)
    model_id, base_model, search_params = model_tup
    param_ser_list = list()
    model_path = "{}{}/".format(out_dir, model_id)

    summary_dict, params_dict = dict(), dict()
    for p_id, (param_set) in enumerate(search_params):
        params_dict[p_id] = param_set
    param_path = "{}param_list.csv".format(out_dir)
    params_df = pd.DataFrame.from_dict(params_dict, orient="index")
    params_df.to_csv(param_path, index_label="set_index")
    param_score_dict = OrderedDict()
    for p_id, (param_set) in enumerate(search_params):
        print(param_set)
        param_dir = "{}param_{}/".format(model_path, p_id)
        os.makedirs(param_dir, exist_ok=True)
        cv_model = base_model.set_params(**param_set)
        # feature_df, labels = feature_df.iloc[:500], labels.iloc[:500]
        dev_scores, eva_scores = cv_model_documented(
            feature_df,
            labels,
            cv_model,
            model_name=model_id,
            save_dir=param_dir,
            cv_splitter=cv,
            sweight=samp_wts,
            **splitter_kws
        )
        print(([s["Balanced Accuracy"] for s in dev_scores]))
        print(np.mean([s["Balanced Accuracy"] for s in dev_scores]))
        print(([s["Balanced Accuracy"] for s in eva_scores]))
        print(np.mean([s["Balanced Accuracy"] for s in eva_scores]))
        param_score_dict[(param_set.values())] = np.mean(
            [s["Balanced Accuracy"] for s in eva_scores]
        )
        continue
    score_ser = pd.Series(param_score_dict)
    score_ser.sort_values(ascending=False, inplace=True)
    pprint.pp(score_ser.head(n=10))
    score_ser.to_csv("{}param_scores.csv".format(model_path))
    return score_ser
    if False:
        hyper_dev_df, hyper_eva_df, all_dev_scores, all_eva_scores = (
            combined_rus_cv_results(
                feature_df,
                labels,
                base_model,
                param_set,
                model_name=model_id,
                save_dir=param_dir,
                sweight=samp_wts,
            )
        )
        param_ser_list.append(pd.Series(param_set, name=p_id))
        hyper_dev_df = hyper_dev_df.add_suffix(suffix="_dev", axis="index")
        hyper_eva_df = hyper_eva_df.add_suffix(suffix="_eval", axis="index")
        pprint.pp(hyper_eva_df)
        combo_scores = pd.concat([hyper_dev_df, hyper_eva_df], axis="index")
        combo_scores.sort_index(axis="columns", kind="stable", inplace=True)
        combo_scores.to_csv("{}score_summary.csv".format(param_dir))
        summary_dict[p_id] = hyper_eva_df["Balanced Accuracy"].squeeze()
        print(summary_dict.items())
    crit_df = pd.DataFrame.from_dict(summary_dict, orient="index")
    if "GeoMean_eval" in crit_df.columns:
        crit_df.sort_values(by="GeoMean_eval", ascending=False, inplace=True)
    params_df = pd.concat(param_ser_list)
    crit_path = "{}criteria_scores.csv".format(model_path)
    param_path = "{}param_list.csv".format(out_dir)
    crit_df.to_csv(crit_path, index_label="set_index")
    params_df.to_csv(param_path, index_label="set_index")
    return crit_df


def get_data(selection_dir):
    X_path = "{}selection_features_df.pkl".format(selection_dir)
    y_path = "{}selection_labels_df.csv".format(selection_dir)
    features_path = "{}final_features_selected.csv".format(selection_dir)
    if not (os.path.isfile(X_path) and os.path.isfile(y_path)):
        idx_dict = data_tools.load_idx_selector()
        train_df, train_labels = data_tools.load_training_data()
        train_dict = dict(
            [
                (k, train_df.loc[v.intersection(train_labels.index)])
                for k, v in idx_dict.items()
            ]
        )
        min_sers = [train_dict[k] for k in ["epa_in", "en_in"]]
        maj_sers = [train_dict[k] for k in ["epa_sol", "en_sol"]]
        sampled_ins, sampled_sols = balancing.mixed_undersampling(
            min_sers, maj_sers, maj_ratio=(0.8, 0.2)
        )
        sampled_ins.extend(sampled_sols)
        print("Undersampled training sizes:")
        print([s.shape for s in sampled_ins])
        all_X = pd.concat(sampled_ins)
        all_y = train_labels[all_X.index]

        all_X.to_pickle(X_path)
        all_y.to_csv(y_path, index=True, index_label="INCHI_KEY")
    else:
        all_X = pd.read_pickle(X_path)
        all_y = pd.read_csv(y_path, index_col="INCHI_KEY").squeeze()
    selected_features = pd.read_csv(features_path, index_col="Features").index
    return all_X[selected_features], all_y


def svm(feature_df, labels, save_dir):
    from sklearn.svm import SVC

    out_dir = "{}svm/".format(save_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    rs = RobustScaler(unit_variance=True).set_output(transform="pandas")
    model_id = "svm"
    mcc = make_scorer(matthews_corrcoef)
    cv = StratifiedKFold(shuffle=True, random_state=0)
    search_params = ParameterGrid(
        {
            "svc__C": [0.1, 1.0, 5.0, 25.0, 1000],
            "svc__kernel": ["rbf"],
            "svc__gamma": [
                0.1 / feature_df.shape[1],
                "scale",
                5.0 / feature_df.shape[1],
                25.0 / feature_df.shape[1],
            ],
            "svc__cache_size": [1000],
            "svc__class_weight": ["balanced"],
        }
    )
    svc = SVC()
    svm_pipe = Pipeline(steps=[("rs", rs), ("svc", svc)]).set_output(transform="pandas")
    svm_search = GridSearchCV(
        estimator=svm_pipe,
        param_grid=search_params.param_grid,
        scoring=mcc,
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        error_score="raise",
    )
    svm_search.fit(feature_df, labels)
    print(svm_search.best_params_, svm_search.best_score_)
    print(svm_search.best_estimator_.steps[1][1].n_support_)
    print(svm_search.best_estimator_.steps[1][1].dual_coef_)
    print(svm_search.best_estimator_.steps[1][1].support_vectors_)
    # print(pd.Series(zip(svm_search.feature_names_in_, svm_search.best_estimator_.steps[1][1].coef_)).sort_values(ascending=False))
    grid_path = "{}best_svm.pkl".format(out_dir)
    results_path = "{}svm_cv_results.csv".format(out_dir)
    with open(grid_path, "wb") as f:
        pickle.dump(svm_search.best_estimator_, f)
    results = pd.DataFrame.from_dict(svm_search.cv_results_)
    results.to_csv(results_path, index=True)
    return svm_search


def rf_hyperparam(feature_df, labels, out_dir, extra=False):
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

    rf_kwargs = {"bootstrap": False, "n_jobs": -1, "random_state": 0}
    if extra:
        base_model = ExtraTreesClassifier(**rf_kwargs)
        model_id = "extra_trees"
    else:
        base_model = RandomForestClassifier(**rf_kwargs)
        model_id = "random_forest"
    search_params = ParameterGrid(
        {
            "n_estimators": [10, 100, 1000],
            "min_impurity_decrease": [0, 0.0001, 0.001],
            "max_features": [3, 6, 9],
            "max_leaf_nodes": [100, 200, None],
        }
    )
    model_info = (model_id, base_model, search_params)
    crit_df = hyperparam(feature_df, labels, out_dir, model_info)
    return crit_df


def main():
    data_dir = "{}correlation_corrected_feat_selection/".format(
        os.environ.get("MODEL_DIR")
    )
    selection_dir = "{}20_en_sol_100_undersampled_rf_3/".format(data_dir)
    out_dir = "{}hyperparam/".format(selection_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    feature_df, labels = get_data(selection_dir)
    # search_obj = svm(feature_df, labels, out_dir)
    scores = rf_hyperparam(feature_df, labels, out_dir)


if __name__ == "__main__":
    main()
