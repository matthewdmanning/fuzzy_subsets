import itertools
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import RobustScaler

import data_tools
from correlation_filter import find_correlation
from data_cleaning import clean_and_check
from quick_models import combined_rus_cv_results


def rus_train(all_df, all_labels, sfs_model, model_params, save_dir=None):
    dev_score_df, eva_score_df, total_dev_scores, total_eva_scores = (
        combined_rus_cv_results(
            all_df,
            all_labels,
            sfs_model,
            model_params,
            n_rus=3,
            model_name="rf",
            save_dir=save_dir,
        )
    )
    total_dev_scores.to_csv("{}dev_scores.csv".format(save_dir))
    total_eva_scores.to_csv("{}eva_scores.csv".format(save_dir))
    return dev_score_df, eva_score_df


def feature_loop(
    train_df, train_y, sfs_model, model_params, max_feats=50, tol=0.001, save_dir=None
):
    score_name = "Balanced Accuracy"
    feature_list = train_df.columns.tolist()
    random.shuffle(feature_list)
    size_dict = defaultdict()
    size_dir = "{}{}_features/".format(save_dir, 3)
    os.makedirs(size_dir)
    print(len(feature_list))
    three_list = deepcopy(feature_list)
    sorted_firsts = (
        pd.Series(f_classif(train_df[feature_list], train_y)[0], index=feature_list)
        .sort_values(ascending=False)
        .iloc[:100]
        .index.tolist()
    )
    i = 0
    with open("{}{}-feature_scores_tmp.csv".format(sel_dir, 3), "a") as f:
        f.write(
            "Index, Feature Set, Eval Mean {}, Eval StDev {}, Dev Mean {}, Dev StDev {}\n".format(
                score_name, score_name, score_name, score_name
            )
        )
    for first in sorted_firsts:
        print(first)
        three_list.remove(first)
        for seconds in itertools.combinations(feature_list, r=2):
            feat_set = list([first])
            feat_set.extend(seconds)
            i += 1
            feat_dir = "{}set_{}/".format(size_dir, i)
            os.makedirs(feat_dir)
            model_params = {
                "min_impurity_decrease": 0.01,
                "n_estimators": int(np.ceil(np.sqrt(len(feat_set)))),
            }
            dev_scores, eval_scores = rus_train(
                train_df[feat_set],
                train_y,
                sfs_model,
                model_params,
                save_dir=feat_dir,
            )
            with open("{}{}-feature_scores_tmp.csv".format(sel_dir, 3), "a") as f:
                f.write(
                    "{}, {}, {}, {}, {}\n".format(
                        i,
                        str(sorted(feat_set)),
                        eval_scores[score_name]["Mean"],
                        eval_scores[score_name]["StDev"],
                        dev_scores[score_name]["Mean"],
                        dev_scores[score_name]["StDev"],
                    )
                )
            size_dict[i] = (
                eval_scores[score_name]["Mean"] - eval_scores[score_name]["StDev"]
            )
    feat_evals = pd.Series(size_dict).sort_values(ascending=False)
    feat_evals.to_csv(
        "{}{}-feature_scores.csv".format(save_dir, 3),
        index=True,
        index_label="Feature Set",
        columns="BAC-StD",
    )
    feat_set = feat_evals.iloc[0].index.tolist()
    [feature_list.remove(f) for f in feat_set]
    for i in list(np.arange(4, max_feats)):
        size_dir = "{}{}_features/".format(save_dir, i)
        os.makedirs(size_dir)
        model_params = {"n_estimators": int(np.ceil(2 * np.sqrt(i)))}
        size_dict = defaultdict()
        for f, feat in enumerate(feature_list):
            feat_dir = "{}set_{}/".format(size_dir, f)
            dev_scores, eval_scores = rus_train(
                train_df[feat_set + [feat]],
                train_y,
                sfs_model,
                model_params,
                save_dir=feat_dir,
            )
            size_dict[feat] = (
                eval_scores.loc[score_name]["Mean"] - eval_scores[score_name]["StdDev"]
            )
        feat_evals = pd.Series(size_dict).sort_values(ascending=False)
        feat_evals.to_csv(
            "{}{}-feature_scores.csv".format(size_dir, i),
            index=True,
            index_label="Feature Set",
            columns="BAC-StD",
        )
        feat_set.append(feat_evals.iloc[0])
        feature_list.remove(feat_evals.iloc[0])
        pprint(feat_evals.haed(n=10))
    pd.Series(feat_set).to_csv("{}selected_features.csv".format(save_dir))
    return feat_set


def main(sel_dir):
    data_path = "{}scaled_data_df.pkl".format(sel_dir)
    label_path = "{}train_labels.pkl".format(sel_dir)
    if os.path.isfile(data_path) and os.path.isfile(label_path):
        train_df = pd.read_pickle(data_path)
        train_y = pd.read_pickle(label_path)
    else:
        idx_dict = data_tools.load_idx_selector()
        train_dfs, train_labels = data_tools.load_training_data()
        all_df = data_tools.load_all_descriptors()
        print("All DF data: {}".format(all_df.shape))
        print("Zero-var features: {}".format(all_df[all_df.var(axis=1) == 0]))
        unscaled_df = all_df.loc[train_labels.index]
        unscaled_df, train_y = clean_and_check(unscaled_df, train_labels)
        print(unscaled_df.shape)

        unique_dict = [
            c
            for c, s in unscaled_df.items()
            if s.value_counts(normalize=True).iloc[0] < 0.95
        ]
        high_var_df = unscaled_df[unique_dict]
        low_var_df = unscaled_df.drop(columns=high_var_df.columns)
        train_df = (
            RobustScaler(quantile_range=(15.9, 84.1), unit_variance=True)
            .set_output(transform="pandas")
            .fit_transform(high_var_df)
        )
        print(
            "{} features have the most commone value is less than 95%. {} removed".format(
                high_var_df.shape[1], unscaled_df.shape[1] - high_var_df.shape[1]
            )
        )
        with open(data_path, "wb") as f:
            pickle.dump(train_df, f)
        with open(label_path, "wb") as f:
            pickle.dump(train_y, f)
    mi_path = "{}low_card_mi.csv".format(sel_dir)
    if not os.path.isfile(mi_path):
        low_var_mi = pd.Series(
            mutual_info_classif(X=low_var_df, y=train_y, random_state=0, n_jobs=-1),
            index=low_var_df.columns,
        ).sort_values(ascending=False)
        print(low_var_mi.head())
        low_var_mi.to_csv(mi_path)

    else:
        low_var_mi = pd.read_csv(mi_path)
    corr_path = "{}pearson_cross_corr_95.csv".format(sel_dir)
    if os.path.isfile(corr_path):
        cross_corr = pd.read_csv(corr_path)
    else:
        cross_corr = train_df.corr()
    overthresh_path = "{}overthresh.csv".format(sel_dir)
    if os.path.isfile(overthresh_path):
        overthresh_corr = pd.read_csv(overthresh_path)
    else:
        overthresh_corr = find_correlation(cross_corr)
        overthresh_corr.to_csv(overthresh_path)
    print(
        "Removed {} features for correlation over 0.95.".format(
            overthresh_corr.shape[0]
        )
    )
    # TDDO: Create function to account for low cardinality. Examplo 10-membered rings.
    train_df.drop(columns=train_df.columns[overthresh_corr.index], inplace=True)
    feature_loop(
        train_df,
        train_y,
        sfs_model=RandomForestClassifier,
        model_params=None,
        save_dir=sel_dir,
    )


if __name__ == "__main__":
    sel_dir = "{}enamine_sfs/".format(os.environ.get("MODEL_DIR"))
    if not os.path.isdir(sel_dir):
        os.makedirs(sel_dir)
    run_name = "{}".format("enamine")
    dir_i = 0
    elimination_dir = "{}{}_{}/".format(sel_dir, run_name, dir_i)
    """    
    while os.path.isdir(elimination_dir):
        dir_i += 1
        elimination_dir = "{}{}_{}/".format(sel_dir, run_name, dir_i)
    os.makedirs(elimination_dir)
    """
    main(sel_dir)
