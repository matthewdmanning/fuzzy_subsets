import os.path
import pickle
import pprint
from collections import defaultdict

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest,
    SequentialFeatureSelector,
    VarianceThreshold,
)
from sklearn.metrics import (
    make_scorer,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold

import balancing
import data_tools
import feature_combination
from feature_selection import vif
from feature_selection.correlation_filter import find_correlation
from feature_selection.importance import (
    brute_force_importance_rf_clf,
    dummy_score_elimination,
)


def get_epa_sol_all_insol(feature_df, labels, tups):
    raise DeprecationWarning
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
    raise DeprecationWarning
    ind_dict = dict()
    for k, v in group_dict.items():
        if type(v) is list or type(v) is list:
            ind_dict[k] = v[0].index.intersection(labels.index)
        elif type(v) is pd.DataFrame or type(v) is pd.Series:
            ind_dict[k] = v.index.intersection(labels.index)
    return ind_dict


def brute_force_with_collinear(
    feature_df, labels, filter_kws_list, importance, select_dir, s_weights=None
):
    X_selected = feature_df.copy()
    results_list = list()
    if not os.path.isdir(select_dir):
        os.makedirs(select_dir)
    for i, filter_kws in enumerate(filter_kws_list):
        filter_kws["fit_kwargs"] = None
        print(X_selected.shape)
        assert type(X_selected) is pd.DataFrame
        results = defaultdict()
        results["inputs"] = filter_kws
        if filter_kws["method_name"] == "rfe":
            results = rfe(X_selected, labels, filter_kws)
            feature_path = "{}{}_n{}_rfe_{}.csv".format(
                select_dir, i, filter_kws["n_feats"], filter_kws["step_size"]
            )
        elif (
            "rfe" in filter_kws["method_name"] and "dummy" in filter_kws["method_name"]
        ):
            results = rfe_dummy(X_selected, labels, filter_kws, save_dir=select_dir)
            feature_path = "{}{}_n{}_dummy_step{}.csv".format(
                select_dir, i, filter_kws["n_feats"], filter_kws["step_size"]
            )
        elif "mi" in filter_kws["method_name"]:
            continue
            """
            cross_mi_path = "{}rus5_cross_mi.csv".format(select_dir)
            results = feature_mi(
                X_selected,
                labels,
                filter_kws,
                select_dir,
            )
            feature_path = "{}{}_n{}_{}-{}.csv".format(
                select_dir, i, filter_kws["n_feats"], filter_kws["corr_method"], filter_kws["corr_cut"]
            )
            """
        elif any(
            [filter_kws["method_name"] == a for a in ["pearson", "kendall", "spearman"]]
        ):
            if "corr_method" not in filter_kws.keys():
                filter_kws["corr_method"] = [
                    a
                    for a in ["pearson", "kendall", "spearman"]
                    if a in filter_kws["method_name"]
                ][0]
            if "cross_corr" not in filter_kws:
                if "corr_path" not in filter_kws.keys():
                    filter_kws["corr_path"] = "{}{}_matrix.csv".format(
                        select_dir, filter_kws["method_name"]
                    )
                if os.path.isfile(filter_kws["corr_path"]):
                    feat_corr_more = pd.read_csv(
                        filter_kws["corr_path"], index_col=True
                    )
                    filter_kws["cross_corr"] = feat_corr_more[X_selected.columns].loc[
                        X_selected.columns
                    ]
                else:
                    filter_kws["cross_corr"] = X_selected.corr(
                        method=filter_kws["corr_method"]
                    )
                    filter_kws["cross_corr"].to_csv(
                        filter_kws["corr_path"], index_label="Features", index=True
                    )
            results["scores_"] = correlation_filter(
                X_selected, labels, filter_kws, select_dir
            )
            feature_path = "{}{}_n{}_{}-{}.csv".format(
                select_dir,
                i,
                filter_kws["n_feats"],
                filter_kws["corr_method"],
                filter_kws["threshold"],
            )
        elif "sfs" in filter_kws["method_name"]:
            if filter_kws["direction"] is None:
                if filter_kws["n_feats"] > X_selected.shape[1] / 2:
                    filter_kws["direction"] = "backwards"
                else:
                    filter_kws["direction"] = "forwards"
            elif "back" in filter_kws["method_name"]:
                filter_kws["direction"] = "backwards"
            else:
                filter_kws["direction"] = "forwards"
            results["scores_"] = sequential(X_selected, labels, filter_kws, select_dir)
            feature_path = "{}{}_n{}_{}-sfs_{}.csv".format(
                select_dir,
                i,
                results["scores_"].size,
                filter_kws["direction"],
                filter_kws["model_name"],
            )

        elif "stoch-vif" in filter_kws["method_name"]:
            vif_survival, vif_score_df, vif_stats, votes = vif.repeated_stochastic_vif(
                X_selected,
                importance[X_selected.columns],
                cut=filter_kws["threshold"],
                sample_wts=s_weights,
                step_size=filter_kws["step_size"],
                # filter_kws["fit_kwargs"]
            )
            results["scores_"] = vif_score_df
            feature_path = "{}{}_n{}_stoch-vif-{}_{}.csv".format(
                select_dir,
                i,
                filter_kws["n_feats"],
                filter_kws["threshold"],
                filter_kws["step_size"],
            )
        elif "vif" in filter_kws["method_name"]:
            drop_ser = vif.sequential_vif(
                X_selected,
                vif_cut=filter_kws["threshold"],
                n_keep=filter_kws["n_feats"],
                step_size=filter_kws["step_size"],
                model=filter_kws["model_name"],
                **filter_kws["fit_kwargs"]
            )
            results["scores_"] = drop_ser
            feature_path = "{}{}_n{}_vif-{}_{}.csv".format(
                select_dir,
                i,
                filter_kws["n_feats"],
                filter_kws["threshold"],
                filter_kws["model_name"],
            )
        else:
            raise ValueError
        results["n_drop_"] = min(
            results["scores_"].index.size, (X_selected.shape[1] - filter_kws["n_feats"])
        )
        results["dropped_"] = results["scores_"].iloc[: results["n_drop_"]].index
        results["feats_out_"] = X_selected.columns[
            ~X_selected.columns.isin(results["dropped_"])
        ]
        results["n_feats_"] = results["feats_out_"].size
        results["feats_out_"].to_series().to_csv(feature_path, index=False)
        results_list.append(results)
        X_selected.drop(columns=results["dropped_"], inplace=True)
    return results_list


def sequential(X_selected, labels, filter_kws, save_dir=None):
    results = defaultdict()
    print(
        "Starting Sequential Feature Selection of {} features".format(
            filter_kws["n_feats"]
        )
    )
    sfs = (
        SequentialFeatureSelector(
            estimator=filter_kws["model"],
            n_features_to_select=filter_kws["n_feats"],
            direction=filter_kws["direction"],
            n_jobs=filter_kws["n_jobs"],
            scoring=filter_kws["scorer"],
        )
        .set_output(transform="pandas")
        .fit(X_selected, labels)
    )
    return pd.Series(sfs.feature_names_in_[sfs.support_])


def correlation_filter(
    X_selected,
    labels,
    filter_kws,
    save_dir=None,
):
    results = dict()
    n_drop = X_selected.shape[1] - filter_kws["n_feats"]
    assert n_drop > 0
    results["scores_"] = find_correlation(
        corr=filter_kws["cross_corr"],
        cutoff=filter_kws["threshold"],
        n_drop=filter_kws["n_feats"],
    )
    results["n_drop_"] = min(results["scores_"].shape[0], n_drop)
    results["dropped_"] = results["scores_"].index[: results["n_drop_"]]
    results["n_feats_"] = X_selected.shape[1] - results["n_drop_"]
    results["threshold_"] = (
        filter_kws["cross_corr"].max(axis=1)[results["dropped_"]].min()
    )
    results["feats_out_"] = X_selected.columns[
        ~X_selected.columns.isin(results["dropped_"])
    ]
    # print("Correlation Drops")
    # print(drop_list[:10])
    # print(len([a for a in drop_list]))
    if save_dir is not None and os.path.isdir(save_dir):
        results["scores_"].to_csv(
            "{}{}_{}.csv".format(
                save_dir, filter_kws["corr_method"], filter_kws["cross_corr"].shape[0]
            )
        )
    return results["feats_out_"]


def rfe_dummy(X_selected, labels, filter_kws, fit_kwargs=None, save_dir=None):
    scores = dummy_score_elimination(
        X_selected,
        labels,
        estimator=filter_kws["model"],
        min_feats=filter_kws["n_feats"],
        cv=StratifiedKFold(shuffle=True),
        step_size=filter_kws["step_size"],
        score_func=filter_kws["scorer"],
        **fit_kwargs
    )
    return scores


def rfe(X_selected, labels, filter_kws, fit_kwargs=None, save_dir=None):
    scores, elim = brute_force_importance_rf_clf(
        X_selected,
        labels,
        clf=filter_kws["model"],
        n_features_out=filter_kws["n_feats"],
        step_size=filter_kws["step_size"],
        **fit_kwargs
    )
    return scores.sort_values()


def main(
    data_set_name="enamine_only",
    data_set=None,
    model_name="brf",
    correlation_type="spearman",
    run_name=None,
    rfe_dir=None,
):
    pd.set_option("display.precision", 4)
    pd.set_option("format.precision", 4)
    # np.set_printoptions(formatter={"float_kind": "{: 0.4f}".format})
    mcc = make_scorer(matthews_corrcoef)
    if rfe_dir is None:
        rfe_dir = "{}data_gaps_feat_selection/".format(os.environ.get("MODEL_DIR"))
    if not os.path.isdir(rfe_dir):
        os.makedirs(rfe_dir)
    if run_name is None:
        run_name = "{}".format(data_set_name)
    dir_i = 0
    elimination_dir = "{}{}_{}/".format(rfe_dir, run_name, dir_i)
    while os.path.isdir(elimination_dir):
        dir_i += 1
        elimination_dir = "{}{}_{}/".format(rfe_dir, run_name, dir_i)
    os.makedirs(elimination_dir, exist_ok=True)
    print(elimination_dir)
    if data_set is None:
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
    else:
        all_X, all_y = data_set
    X_path = "{}{}_df.pkl".format(rfe_dir, data_set_name)
    y_path = "{}selection_labels_df.csv".format(rfe_dir)
    all_X.to_pickle(X_path)
    all_y.to_csv(y_path, index=True, index_label="INCHI_KEY")
    importance_model, model_name = get_importance_model(model_name)
    # Start Feature Selection
    all_X = VarianceThreshold().set_output(transform="pandas").fit_transform(all_X)
    all_X = pca_combos(all_X, elimination_dir)
    nbest = 500
    f_kbest = SelectKBest(k=nbest).set_output(transform="pandas")
    f_kbest.fit(all_X, all_y)
    f_best = pd.Series(index=all_X.columns, data=f_kbest.scores_)
    f_best.to_csv("{}{}-fscore.csv".format(elimination_dir, nbest))
    pprint.pp(f_best.sort_values(ascending=False), compact=True)
    kbest_cols = f_kbest.transform(all_X).columns
    corr_path = "{}{}_{}.csv".format(rfe_dir, correlation_type, nbest)
    if os.path.isfile(corr_path):
        cross_corr = pd.read_csv(corr_path, index_col="Features")
    else:
        cross_corr = all_X[kbest_cols].corr(method=correlation_type)
        cross_corr.to_csv(corr_path, index=True, index_label="Features")
    feat_step_list = (
        (correlation_type, 250, 0, 0.95),
        ("stoch-vif", 200, 1, 25),
        ("rfe", 150, 5, 0),
        ("stoch-vif", 125, 1, 15),
        ("dummy_rfe", 50, 1, 0),
    )
    feat_step_list = (
        {
            "threshold": 0.95,
            "n_feats": 250,
            "method_name": correlation_type,
            "cross_corr": cross_corr,
        },
        {"method_name": "stoch-vif", "n_feats": 200, "step_size": 1, "threshold": 25},
        {
            "method_name": "rfe",
            "n_feats": 150,
            "step_size": 5,
            "model": get_importance_model("brf")[0],
        },
        {"method_name": "stoch-vif", "n_feats": 125, "threshold": 15},
        {
            "method_name": "dummy_rfe",
            "n_feats": 50,
            "model": get_importance_model("brf")[0],
            "model_name": "brf",
            "scorer": mcc,
        },
    )
    importance = pd.Series(
        data=f_kbest.scores_ / max(f_kbest.scores_), index=f_kbest.feature_names_in_
    )
    rfe_features = brute_force_with_collinear(
        all_X[kbest_cols], all_y, feat_step_list, importance, elimination_dir
    )[-1]["feats_out_"]
    rfe_path = "{}final_features_selected.csv".format(elimination_dir, data_set)
    rfe_features.to_csv(rfe_path)  # , index=True, index_label="Features")
    return rfe_features


def pca_combos(all_X, elimination_dir):
    transformed_feats, transformers = feature_combination.combine_feature_groups(all_X)
    with open("{}pca_trans_tup.pkl".format(elimination_dir), "wb") as f:
        pickle.dump(transformers, f)
    with open("{}new_pca_data.pkl".format(elimination_dir), "wb") as f:
        pickle.dump(transformed_feats, f)
    transformed_features = list()
    [
        transformed_features.extend(t.feature_names_in_)
        for t in transformers
        if t is not None
    ]
    pd.Series(transformed_features).to_csv(
        "{}pca_original_features.csv".format(elimination_dir)
    )
    print(transformed_feats.head())
    new_X = pd.concat(
        [
            all_X,
            *[
                t
                for t in transformed_feats
                if t is not None and type(t) is pd.DataFrame and not t.empty
            ],
        ]
    ).drop(columns=transformed_features)
    return new_X


def get_importance_model(model_name):
    if model_name == "brf":
        importance_model = BalancedRandomForestClassifier(
            bootstrap=False,
            n_jobs=-2,
            random_state=0,
            replacement=False,
            sampling_strategy="all",
            class_weight="balanced",
        )
    elif model_name == "xtra":
        importance_model = ExtraTreesClassifier(
            n_jobs=-2, random_state=0, class_weight="balanced"
        )
    else:
        importance_model = RandomForestClassifier(
            random_state=0, bootstrap=False, n_jobs=-2, class_weight="balanced"
        )
        model_name = "rf"
    return importance_model, model_name


if __name__ == "__main__":
    idx_dict = data_tools.load_idx_selector()
    train_dfs, train_labels = data_tools.load_training_data()
    train_df = pd.concat(
        [
            train_dfs.loc[v.intersection(train_labels.index)]
            for k, v in idx_dict.items()
            if "en" in k
        ]
    )
    train_y = train_labels[train_df.index]
    selected_feats = main(data_set=(train_df, train_y))
    print(selected_feats)

"""
def feature_mi(
    X_selected,
    labels,
    corr_cut,
    filter_kws["corr_method"],
    i,
    cross_mi_path,
    n_drop,
    n_feats,
    select_dir,
):
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
            pd.concat(rus_list, axis=1).stack().groupby(level=[0, 1]).mean().unstack()
        )
        feat_corr.index = X_selected.columns
        feat_corr.to_csv(cross_mi_path, index_label=True)
        feat_corr.to_csv(cross_mi_path, index_label="Features", index=True)
    drop_list = find_correlation(corr=feat_corr, cutoff=corr_cut, n_drop=n_drop)
    X_selected.drop(columns=drop_list, inplace=True)
    feats = X_selected.columns.to_series()
    return feats, feature_path
"""
