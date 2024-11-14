import os.path
import pprint
from functools import partial

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (
    make_scorer,
    matthews_corrcoef,
)

# from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_X_y

from data_handling.data_cleaning import remove_duplicate_idx
from feature_selection import vif
from feature_selection.correlation_filter import findCorrelation
from feature_selection.importance import (
    brute_force_importance_rf_clf,
    dummy_score_elimination,
)
from qsar_modeling.data_handling.data_tools import (
    get_interpretable_features,
    load_metadata,
)


def get_epa_sol_all_insol(feature_df, labels, tups):
    en_in = tups["en_in"][0][
        [c for c in tups["en_in"][0].index if c not in tups["epa_in"][0].index]
    ]
    all_samples_ind = pd.concat(
        [tups["epa_in"][0], en_in, tups["epa_sol"][0]]
    ).index.intersection(feature_df.index)
    all_samples = labels[all_samples_ind]
    all_samples[all_samples == "Insoluble"] = 0
    all_samples[all_samples == "Soluble"] = 1
    select_y = labels[all_samples.index]
    select_X = feature_df.loc[all_samples.index]
    # assert not select_X.isna().any()
    return select_X, select_y


def data_by_groups(labels, group_dict):
    ind_dict = dict()
    for k, v in group_dict.items():
        if type(v) is list or type(v) is list:
            ind_dict[k] = v[0].index.intersection(labels.index)
        elif type(v) is pd.DataFrame or type(v) is pd.Series:
            ind_dict[k] = v.index.intersection(labels.index)
    return ind_dict


def load_data(epa_sol=False):
    # Select EPA soluble/combined insoluble dataset.
    meta = load_metadata(desc=False)
    # meta_dict = dict(zip(['epa_sol', "epa_in", "en_in", "en_sol"], meta))
    interp_X, interp_y = get_interpretable_features()
    unique_idx = remove_duplicate_idx(interp_X)
    unique_X, unique_y = interp_X.loc[unique_idx.index], interp_y[unique_idx.index]
    if epa_sol:
        select_X, select_y = get_epa_sol_all_insol(unique_X, unique_y, meta)
    else:
        select_X, select_y = unique_X, unique_y
    check_X_y(select_X, select_y)
    return select_X, select_y


def brute_force_with_collinear(
    feature_df, labels, clf, clf_name, feat_step_tups, select_dir, **fit_kwargs
):
    scorer = make_scorer(matthews_corrcoef)
    X_selected = feature_df.copy()

    if not os.path.isdir(select_dir):
        os.makedirs(select_dir)
    for i, (corr_method, n_feats, step_size, corr_cut) in enumerate(feat_step_tups):
        print(X_selected.shape)
        print(step_size)
        n_drop = X_selected.shape[1] - n_feats
        if corr_method == "rfe":
            feat_ranks, elim = brute_force_importance_rf_clf(
                X_selected,
                labels,
                clf=clf,
                n_features_out=n_feats,
                step_size=step_size,
                **fit_kwargs
            )
            feats = feat_ranks[feat_ranks == 1]

            feature_path = "{}n{}_step{}_rfe_{}.csv".format(
                select_dir, n_feats, step_size, i
            )
        elif "rfe" in corr_method and "dummy" in corr_method:
            dummy_list = list()
            X_selected = dummy_score_elimination(
                X_selected,
                labels,
                estimator=clf,
                min_feats=n_feats,
                cv=StratifiedKFold(shuffle=True),
                score_func=scorer,
                **fit_kwargs
            )
            feats = X_selected.columns.to_series()

            feature_path = "{}n{}_step{}_dummy_elimination_{}.csv".format(
                select_dir, n_feats, step_size, i
            )
        elif "mi" in corr_method:
            cross_mi_path = "{}rus5_cross_mi.csv".format(select_dir)
            if os.path.isfile(cross_mi_path):
                feat_corr = pd.read_csv(cross_mi_path)
            else:
                from feature_selection.mutual_info_tools import jmi_homebrew
                from sklearn.feature_selection import mutual_info_regression

                mi_opt = {
                    "n_neighbors": 5,
                    "random_state": 0,
                    "n_jobs": -1,
                }
                mi_part = partial(
                    mutual_info_regression,
                    discrete_features="auto",
                    n_neighbors=5,
                    random_state=0,
                    n_jobs=-1,
                )
                rus_list = list()
                for _ in np.arange(5):
                    rus_X, rus_y = RandomUnderSampler(random_state=0).fit_resample(
                        X_selected, labels
                    )
                    rus_list.append(jmi_homebrew.mi_mixed_types(rus_X, **mi_opt))
                feat_corr = (
                    pd.concat(rus_list, axis=1)
                    .stack()
                    .groupby(level=[0, 1])
                    .mean()
                    .unstack()
                )
                feat_corr.index = X_selected.columns
                feat_corr.to_csv(cross_mi_path, index_label=True)
                corr_path = "{}{}}_matrix_{}_{}.csv".format(
                    select_dir, corr_method, n_feats, i
                )
                feat_corr.to_csv(cross_mi_path, index_label="Features", index=True)
            drop_str = corr_method
            drop_list = findCorrelation(corr=feat_corr, cutoff=corr_cut, n_drop=n_drop)
            X_selected.drop(columns=drop_list, inplace=True)
            feature_path = "{}collinear_{}_rfe_{}.csv".format(select_dir, drop_str, i)
            feats = X_selected.columns.to_series()
        elif any([corr_method == a for a in ["pearson", "kendall", "spearman"]]):
            corr_path = "{}{}_matrix.csv".format(select_dir, corr_method, drop_str, i)
            if os.path.isfile(corr_path):
                feat_corr = pd.read_csv(corr_path, index_col=True)
            else:
                feat_corr = X_selected.corr(method=corr_method)
                feat_corr.to_csv(corr_path, index_label="Features", index=True)
            drop_str = "{}-{}".format(n_feats, str(corr_cut).replace(".", "-"))
            print("Cross-correlation scores:")
            sorted_corr = feat_corr.stack().sort_values(ascending=False, kind="stable")
            pprint.pp(
                sorted_corr[[(a, b) for a, b in sorted_corr.index if a != b]],
                compact=True,
            )
            drop_list = findCorrelation(corr=feat_corr, cutoff=corr_cut, n_drop=n_drop)
            # print("Correlation Drops")
            # print(drop_list[:10])
            # print(len([a for a in drop_list]))
            X_selected.drop(columns=drop_list, inplace=True)
            feature_path = "{}{}_{}_rfe_{}.csv".format(
                select_dir, corr_method, drop_str, i
            )
            feats = X_selected.columns.to_series()
        elif "seq" in corr_method:
            if n_feats < 0:
                direction = "backwards"
            else:
                direction = "forwards"
            if n_feats >= 1 or n_feats <= -1:
                print(
                    "Starting Sequential Feature Selection of {} features".format(
                        n_feats
                    )
                )
                sfs = SequentialFeatureSelector(
                    clf,
                    direction=direction,
                    scoring=scorer,
                    n_jobs=-2,
                    n_features_to_select=n_feats,
                ).set_output(transform="pandas")
            elif -1 < n_feats < 1 and n_feats != 0:
                print(
                    "Starting Sequential Feature Selection with tolerance of {}.".format(
                        n_feats
                    )
                )
                sfs = SequentialFeatureSelector(
                    clf, scoring=scorer, n_jobs=-2, tol=n_feats, direction=direction
                ).set_output(transform="pandas")
            else:
                raise ValueError
            sfs.fit(X_selected, labels)
            sfs_out = sfs.transform(X_selected)
            if type(X_selected) is not pd.DataFrame:
                X_selected = pd.DataFrame(
                    sfs_out, columns=sfs.feature_names_in_[sfs.support_]
                )
            else:
                X_selected = sfs_out
            feats = X_selected.columns.to_series()
            feature_path = "{}{}_fwrd_seq_{}_{}.csv".format(
                select_dir, clf_name, n_feats, i
            )
        elif "vif" in corr_method:
            drop_ser = vif.sequential_vif(
                X_selected,
                vif_cut=corr_cut,
                n_keep=n_feats,
                step_size=step_size,
                model="elasticnet",
                **fit_kwargs
            )
            drop_list = drop_ser.index
            X_selected.drop(columns=drop_list, inplace=True)
            feats = X_selected.columns.to_series()
            feature_path = "{}vif_{}_{}_{}.csv".format(
                select_dir, clf_name, n_feats, corr_cut, i
            )
        else:
            raise ValueError
        feats.to_csv(feature_path, index=True, index_label="Features")
    return X_selected.columns


def main(
    data_set="epa_sol",
    model_name="rf",
    correlation_type="spearman",
    run_name=None,
    project_dir=None,
):
    pd.set_option("display.precision", 4)
    pd.set_option("format.precision", 4)
    # np.set_printoptions(formatter={"float_kind": "{: 0.4f}".format})
    matthews_scorer = make_scorer(matthews_corrcoef)
    if project_dir is None:
        project_dir = "{}correlation_corrected_feat_selection/".format(
            os.environ.get("MODEL_DIR")
        )
    if not os.path.isdir(project_dir):
        os.makedirs(project_dir)
    if (
        type(data_set) is str
        and "epa" in data_set
        and "sol" in data_set
        and "insol" not in data_set
    ):
        all_X, all_y = load_data(epa_sol=True)
    elif data_set is None:
        all_X, all_y = load_data()
    else:
        all_X, all_y = data_set

    if model_name == "brf":
        importance_model = BalancedRandomForestClassifier(
            bootstrap=False,
            n_jobs=-2,
            random_state=0,
            replacement=False,
            sampling_strategy="all",
        )
    else:
        importance_model = RandomForestClassifier(
            random_state=0, bootstrap=False, n_jobs=-2, class_weight="balanced"
        )
        model_name = "rf"
    feat_step_list = (
        ("spearman", 500, 0, 0.95),
        ("rfe", 500, 25, 0),
        ("rfe", 250, 10, 0),
        ("vif", 200, 1, 50),
        ("rfe", 150, 5, 0),
        ("vif", 125, 1, 15),
        ("dummy_rfe", 50, 1, 0),
        # ("seq", 40, -0.005, 0),
    )
    if run_name is None:
        run_name = "{}_{}".format(data_set, model_name)
    dir_i = 0
    elimination_dir = "{}{}_{}/".format(project_dir, run_name, dir_i)
    while os.path.isdir(elimination_dir):
        dir_i += 1
        elimination_dir = "{}{}_{}/".format(project_dir, run_name, dir_i)
    os.makedirs(elimination_dir, exist_ok=True)
    print(elimination_dir)
    rfe_features = brute_force_with_collinear(
        all_X,
        all_y,
        importance_model,
        model_name,
        feat_step_list,
        elimination_dir,
    ).to_series()
    rfe_path = "{}final_features_selected.csv".format(elimination_dir, data_set)
    rfe_features.to_csv(rfe_path, index=True, index_label="Features")
    return rfe_features


if __name__ == "__main__":
    main()
