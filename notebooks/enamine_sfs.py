import os
import pickle
from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
from imblearn.under_sampling import NearMiss
from scipy.stats import gmean, hmean
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import RobustScaler

import data_tools
from correlation_filter import cross_corr_filter
from data_cleaning import clean_and_check
from quick_models import combined_rus_cv_results


def rus_train(
    all_df, all_labels, sfs_model, model_params, cv=None, n_rus=3, save_dir=None
):
    dev_score_df, eva_score_df, total_dev_scores, total_eva_scores = (
        combined_rus_cv_results(
            all_df,
            all_labels,
            sfs_model,
            model_params,
            n_rus=n_rus,
            cv_splitter=cv,
            save_dir=save_dir,
        )
    )
    if len(total_eva_scores) > 0 and len(total_dev_scores) > 0:
        total_dev_scores.to_csv("{}dev_scores.csv".format(save_dir))
        total_eva_scores.to_csv("{}eva_scores.csv".format(save_dir))
    return dev_score_df, eva_score_df


def feature_loop(
    train_df,
    train_y,
    sfs_model,
    model_params,
    max_feats=30,
    tol=0.001,
    save_dir=None,
    trial_method="auto",
    n_trials=15,
    corr_df=None,
):
    score_name = "Balanced Accuracy"
    size_dir = "{}{}_features/".format(save_dir, 3)
    os.makedirs(size_dir, exist_ok=True)
    if model_params is None:
        model_params = {
            "min_impurity_decrease": 0.005,
            "min_samples_split": 4,
            "n_estimators": 10,
            "max_depth": 10,
            "bootstrap": False,
            "class_weight": "balanced",
            "n_jobs": -1,
        }
    undersampler = NearMiss(version=2)
    # train_df, train_y = undersampler.fit_resample(feature_df, labels)
    feature_list = train_df.columns.tolist()

    if trial_method == "auto":
        f_original = pd.Series(
            f_classif(train_df[feature_list], train_y)[0], index=feature_list
        )
        f_switched = pd.Series(
            f_classif(train_df[feature_list], 1 - train_y)[0], index=feature_list
        )
        predict_ser = f_original.multiply(f_switched).sort_values(ascending=False)
        predict_ser = predict_ser / predict_ser.max()
        sorted_firsts = score_predict_crosscorr(train_df, predict_ser, corr_df)
    if trial_method == "mi":
        predict_ser = pd.Series(
            mutual_info_classif(
                X=train_df, y=train_y, n_neighbors=7, random_state=0, n_jobs=-1
            ),
            index=train_df.columns,
        )
        sorted_firsts = score_predict_crosscorr(train_df, predict_ser, corr_df)
    feat_set = [sorted_firsts[0]]
    feature_list = score_predict_crosscorr(
        train_df, predict_ser, corr_df, selected=feat_set
    )
    for n in list(np.arange(len(feat_set), max_feats - 1)):
        i = n + 1
        size_dir = "{}{}_features/".format(save_dir, i)
        os.makedirs(size_dir, exist_ok=True)
        with open("{}base_features.csv".format(size_dir), "w") as f:
            f.write("\t".join(feat_set))
        size_dict = defaultdict()
        for f_n, feat in enumerate(feature_list):
            score_feat_set(
                feat_set + [feat],
                f_n,
                model_params,
                score_name,
                sfs_model,
                size_dict,
                size_dir,
                train_df,
                train_y,
            )
        feat_evals = pd.Series(size_dict).sort_values(ascending=False)
        feat_set.append(feat_evals.index[0])
        pprint(feat_evals.head(n=10))
        feat_evals.to_csv(
            "{}{}-feature_scores.csv".format(size_dir, i),
            index=True,
            index_label="Feature Set",
            columns="BAC-StD",
        )
        sorted_firsts = score_predict_crosscorr(
            train_df, predict_ser, corr_df, selected=feat_set
        )
        feature_list = sorted_firsts[:n_trials]
    pd.Series(feat_set).to_csv("{}selected_features.csv".format(save_dir))
    return feat_set


def score_predict_crosscorr(
    feature_df, predict_ser, corr_df, eval_fn="harmonic", selected=None, initial="corr"
):
    # Scores candidate features based on univariate predictive power and cross correlation.
    predict_ser.sort_values(ascending=False)
    comp_corr_df = 1 - corr_df
    if eval_fn == "harmonic":
        eval_fn = hmean
    feature_list = corr_df.columns.tolist()
    if selected is not None:
        [feature_list.remove(f) for f in selected if f in feature_list]
        feats = selected
    else:
        feats = []
    # Use geometric mean of scores and 1 / correlations
    score_dict = dict()
    if len(feats) == 0 and initial == "corr":
        for i in list(range(corr_df.shape[0])):
            corr_df.iloc[i, i] = 0
        quart_corr = 1 - (corr_df * corr_df * corr_df * corr_df)
        scores = (
            quart_corr.multiply(predict_ser)
            .apply(gmean)
            .sort_values(ascending=False)
            .dropna()
        )
        assert not scores.hasnans
    elif len(feats) == 0 and initial == "huber":
        inv_rms = dict()
        feature_list = predict_ser.iloc[:15].index
        from sklearn.linear_model import HuberRegressor

        for feat in feature_list:
            model = HuberRegressor(fit_intercept=False, max_iter=1000, tol=1e-03).fit(
                X=feature_df[feature_list].drop(columns=feat),
                y=feature_df[feat].squeeze(),
            )
            inv_rms[feat] = 1 / model.score(
                X=feature_df[feature_list].drop(columns=feat),
                y=feature_df[feat].squeeze(),
            )
        scores = (
            pd.concat([pd.Series(inv_rms), predict_ser])
            .apply(eval_fn)
            .sort_values(ascending=False)
        )
    else:
        print(comp_corr_df.shape)
        for feat in feature_list:
            # Grab all scores and cross-corrs for each (selected, candidate).
            comp_corr_dict = dict([(f, comp_corr_df.loc[f, feat]) for f in feats])
            score_dict[feat] = eval_fn(
                pd.concat([predict_ser[feats], pd.Series(comp_corr_dict)], axis=1)
                .stack()
                .values
            )
        scores = pd.Series(score_dict).sort_values(ascending=False)
    print("Scores for candidate features:")
    pprint(scores.head())
    return scores.index.tolist()


def seq_predict_corr_select(
    feature_df, predict_ser, corr_df, eval_fn="harmonic", initial=None
):
    # Feature selection using univariate prediction and cross-correlation.
    # Features should be scaled and centered and highly correlated (r > 0.95) should be removed prior to use.
    return


def score_feat_set(
    feat_set,
    i,
    model_params,
    score_name,
    sfs_model,
    size_dict,
    size_dir,
    train_df,
    train_y,
):
    feat_dir = "{}set_{}/".format(size_dir, i)
    os.makedirs(feat_dir, exist_ok=True)
    dev_scores, eval_scores = rus_train(
        train_df[feat_set],
        train_y,
        sfs_model,
        model_params,
        n_rus=2,
        save_dir=feat_dir,
    )
    if dev_scores.empty or eval_scores.empty:
        return
    with open("{}{}-feature_scores_tmp.csv".format(sel_dir, len(feat_set)), "a") as f:
        f.write(
            "{}\t{}\t{}\t{}\t{}\t{}\n".format(
                i,
                str(feat_set),
                eval_scores[score_name]["Mean"],
                eval_scores[score_name]["StDev"],
                dev_scores[score_name]["Mean"],
                dev_scores[score_name]["StDev"],
            )
        )
    size_dict[i] = eval_scores[score_name]["Mean"] - eval_scores[score_name]["StDev"]
    return


def main(selection_dir):
    data_path = "{}scaled_data_df.pkl".format(selection_dir)
    label_path = "{}train_labels.pkl".format(selection_dir)
    train_df, train_y, cross_corr_df = preprocess_data(
        data_path, label_path, selection_dir
    )
    feature_loop(
        train_df,
        train_y,
        sfs_model=RandomForestClassifier,
        model_params=None,
        save_dir=selection_dir,
        corr_df=cross_corr_df,
        n_trials=10,
    )


def preprocess_data(data_path, label_path, selection_dir):
    if os.path.isfile(data_path) and os.path.isfile(label_path):
        train_df = pd.read_pickle(data_path)
        train_y = pd.read_pickle(label_path)
        unique_dict = [
            c
            for c, s in train_df.items()
            if s.value_counts(normalize=True).iloc[0] < 0.95
        ]
        high_var_df = train_df[unique_dict]
        low_var_df = train_df.drop(columns=high_var_df.columns)
    else:
        # idx_dict = data_tools.load_idx_selector()
        train_dfs, train_labels = data_tools.load_training_data()
        combo_dict = data_tools.load_combo_data(["en_in", "en_sol"])
        enamine_idx = list()
        [enamine_idx.extend(f.index.tolist()) for f in combo_dict.values()]
        all_df = data_tools.load_all_descriptors()

        print("All DF data: {}".format(all_df.shape))
        # print("Zero-var features: {}".format(all_df[all_df.var(axis=1) <= 0.0001]))
        unscaled_df = all_df.loc[train_labels.index.intersection(pd.Index(enamine_idx))]
        enamine_path = "{}data/enamine_all_padel.pkl".format(
            os.environ.get("PROJECT_DIR")
        )
        unscaled_df.to_pickle(enamine_path)
        exit()
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
            "{} features have the most common value is less than 95%. {} removed".format(
                high_var_df.shape[1], unscaled_df.shape[1] - high_var_df.shape[1]
            )
        )
        with open(data_path, "wb") as f:
            pickle.dump(train_df, f)
        with open(label_path, "wb") as f:
            pickle.dump(train_y, f)
    mi_path = "{}low_card_mi.csv".format(selection_dir)
    if not os.path.isfile(mi_path):
        low_var_mi = pd.Series(
            mutual_info_classif(X=low_var_df, y=train_y, random_state=0, n_jobs=-1),
            index=low_var_df.columns,
        ).sort_values(ascending=False)
        print(low_var_mi.head())
        low_var_mi.to_csv(mi_path)

    else:
        low_var_mi = pd.read_csv(mi_path)
    corr_path = "{}pearson_cross_corr_95.csv".format(selection_dir)
    if os.path.isfile(corr_path):
        cross_corr = pd.read_csv(corr_path, index_col="Features")
    else:
        cross_corr = train_df.corr()
        cross_corr.to_csv(corr_path, index=True, index_label="Features")
    overthresh_path = "{}overthresh.csv".format(selection_dir)
    if os.path.isfile(overthresh_path):
        overthresh_corr = pd.read_csv(overthresh_path, index_col="Features")
    else:
        overthresh_corr = cross_corr_filter(cross_corr)
        overthresh_corr.to_csv(overthresh_path, index=True, index_label="Features")
    train_df.drop(columns=train_df[overthresh_corr.index], inplace=True)
    print(
        "Removed {} features for correlation over 0.95.".format(
            overthresh_corr.shape[0]
        )
    )
    return train_df, train_y, cross_corr


if __name__ == "__main__":
    sel_dir = "{}enamine_sfs_selective/".format(os.environ.get("MODEL_DIR"))
    os.makedirs(sel_dir, exist_ok=True)
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
