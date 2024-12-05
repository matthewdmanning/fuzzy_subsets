import os
import pickle
import pprint

import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import clone
from sklearn.preprocessing import RobustScaler
from sklearn.utils import compute_sample_weight

import sample_clusters
from correlation_filter import find_correlation
from vif import repeated_stochastic_vif

mcc = make_scorer(balanced_accuracy_score)


def _permutation_removal(
    train_df, labels, feature_list, grove_model, dropped_dict, fails
):
    n_repeats = 25
    while fails > 2 and len(feature_list) > 20:
        perm_model = clone(grove_model).fit(train_df[feature_list], labels)
        perm_results = permutation_importance(
            perm_model,
            train_df[feature_list],
            labels,
            n_repeats=n_repeats,
            scoring=mcc,
            n_jobs=-1,
        )
        import_mean = pd.Series(
            data=perm_results["importances_mean"],
            index=feature_list,
            name="Mean",
        ).sort_values()
        import_std = pd.Series(
            data=perm_results["importances_std"],
            index=feature_list,
            name="StD",
        )
        adj_importance = import_mean.iloc[0] + import_std[import_mean.index].iloc[0]
        if fails < 2 or len(feature_list) < 15:
            import_thresh = min(0.05, adj_importance)
        else:
            import_thresh = adj_importance
        unimportant = import_mean[import_mean <= import_thresh].index.tolist()
        if len(unimportant) > 1 and n_repeats <= 50:
            n_repeats = n_repeats + 25
            continue
        elif len(unimportant) == 1 or (n_repeats >= 50 and len(unimportant) > 0):
            low_feats = unimportant[0]
        else:
            fails -= 1
            break
            # [feature_list.remove(c) for c in low_feats]
        feature_list.remove(low_feats)
        dropped_dict.update([(low_feats, "Importance")])
        print(
            "Permutation drop: \n{}: {} {}".format(
                low_feats,
                import_mean[low_feats],
                import_std[low_feats],
            )
        )
        fails -= 2
    return feature_list, dropped_dict, fails


def _vif_elimination(
    train_df, feature_list, best_corrs, cross_corr, vif_thresh, dropped_dict, fails
):
    model_size = 10
    vif_rounds = int(len(feature_list) * (len(feature_list) + 1) / model_size)
    # fweights = 0.5 - pd.Series(scipy.linalg.norm(cross_corr.loc[feature_list, feature_list], axis=1), index=feature_list).squeeze().sort_values(ascending=False)
    vif_results = repeated_stochastic_vif(
        feature_df=train_df[feature_list],
        importance_ser=best_corrs[feature_list],
        threshold=vif_thresh,
        model_size=model_size,
        feat_wts=cross_corr[feature_list][feature_list],
        min_feat_out=len(feature_list) - 1,
        rounds=vif_rounds,
    )
    if len(vif_results["vif_stats_list"]) > 1:
        all_vif_mean = (
            pd.concat([df["Max"] for df in vif_results["vif_stats_list"]], axis=1)
            .max()
            .sort_values(ascending=False)
        )
        all_vif_sd = pd.concat(
            [df["StdDev"] for df in vif_results["vif_stats_list"]], axis=1
        ).std(ddof=len(feature_list) - model_size)
    else:
        all_vif_mean = vif_results["vif_stats_list"][0]["Max"].sort_values(
            ascending=False
        )
        all_vif_sd = vif_results["vif_stats_list"][0]["SD"]
    vif_dropped = all_vif_mean.index[0]
    # [print(c, [d.loc[c] for d in vif_results["vif_stats_list"] if c in d.index]) for c in vif]
    if vif_dropped in feature_list and all_vif_mean[vif_dropped] > vif_thresh:
        feature_list.remove(vif_dropped)
        dropped_dict.update([(vif_dropped, "VIF")])
        fails -= 2
        if False:
            print("VIF Scores")
            pprint.pp(
                pd.concat(
                    [
                        all_vif_mean,
                        all_vif_sd[all_vif_mean.index],
                        vif_results["votes_list"][-1].loc[all_vif_mean.index],
                    ],
                    axis=1,
                ).head(n=3),
                width=120,
            )
    else:
        fails -= 1
        print("Nothing above threshold.")
    return feature_list, dropped_dict, fails


def geary_trials(
    feature_df,
    labels,
    grove_cols,
    member_col,
    grove_model,
    removal="perm",
    vif_thresh=10,
    save_dir=None,
):
    score_dict, best_features, dropped_dict = dict(), list(), dict()
    top_score, last_best, fails = 0.5, 0.5, 0
    tol = 0.01
    print("\nFeature selection for category: {}".format(member_col))
    geary_cols = list(
        set(
            [
                c
                for c in feature_df
                if "eta" in c.lower()
                or "shape" in c.lower()
                or "geary" in c.lower()
                or "chain" in c.lower()
                or "donor" in c.lower()
                or "hydrogen bond" in c.lower()
                or c in grove_cols
            ]
        )
    )
    best_corrs, cross_corr, train_df = process_selection_data(
        dropped_dict, feature_df, geary_cols, labels, save_dir
    )
    print("Training Size: {}".format(train_df.shape))
    pc = PCA(n_components="mle").fit(train_df)
    print("PCA Results: Noise Variance and Number of Components to Reach 0.9 EVR:")
    print(
        pc.noise_variance_,
        len([v for v in np.cumsum(pc.explained_variance_ratio_) if v < 0.9]),
    )
    if member_col in train_df.index:
        feature_list = [member_col]
    else:
        feature_list = [best_corrs.abs().sort_values(ascending=False).index[0]]
    pprint.pp(dropped_dict.items())
    # Start feature loop
    while (
        len(feature_list) < 30 and best_corrs.shape[0] - len([dropped_dict.keys()]) > 0
    ):
        feat_corrs = best_corrs.drop(
            [
                c
                for c in best_corrs.index
                if c in dropped_dict.keys() or c in feature_list
            ]
        )
        feat_probs = scipy.special.softmax(np.abs(feat_corrs))
        feature_list = list(set(feature_list))
        # noinspection PyTypeChecker
        new_feat = np.random.choice(
            a=feat_corrs.index.to_numpy(), replace=False, p=feat_probs
        )
        if new_feat not in cross_corr.columns:
            print("{} not in cross_corr".format(new_feat))
            continue
        feature_list.append(new_feat)
        scores = cross_val_score(
            grove_model,
            X=train_df[feature_list],
            y=labels,
            scoring=mcc,
            n_jobs=-1,
            error_score="raise",
        )
        best_features, score_dict, top_score, last_best, fails = record_score(
            best_features,
            fails,
            feature_list,
            save_dir,
            score_dict,
            scores,
            top_score,
            last_best,
        )
        if fails > 4 and len(feature_list) > 15:
            feature_list, dropped_dict, fails = _vif_elimination(
                train_df,
                feature_list,
                best_corrs,
                cross_corr,
                vif_thresh,
                dropped_dict,
                fails,
            )
            while fails > 2 and 20 < len(feature_list) < 30:

                if "perm" in removal:
                    print("Starting permutation importance")
                    feature_list, dropped_dict, fails = _permutation_removal(
                        train_df, labels, feature_list, grove_model, dropped_dict, fails
                    )
        while fails > 10 and 15 < len(feature_list) < 20 or len(feature_list) == 30:
            sfs = SequentialFeatureSelector(
                grove_model,
                direction="backward",
                tol=0,
                n_features_to_select=len(feature_list) - 1,
                scoring=mcc,
                n_jobs=-1,
            ).fit(train_df[feature_list], y=labels)
            new_features = list(sfs.get_feature_names_out(feature_list))
            sfs_drops = [c for c in feature_list if c not in new_features]
            sfs_score = cross_val_score(
                grove_model,
                train_df[new_features],
                labels,
                scoring=mcc,
                n_jobs=-1,
                error_score="raise",
            )
            if np.mean(sfs_score) - np.std(sfs_score) > top_score:
                best_features, score_dict, top_score, last_best, fails = record_score(
                    best_features,
                    fails,
                    feature_list,
                    save_dir,
                    score_dict,
                    sfs_score,
                    top_score,
                    last_best,
                )
                dropped_dict.update([(c, "SFS") for c in sfs_drops])
                fails -= len(sfs_drops) * 2
            elif len(feature_list) == 30:
                fails = 0
                break
            else:
                fails -= 1
                break
    print(
        "Best score of {} for {} with feature set: {}".format(
            top_score, member_col, feature_list
        )
    )
    with open("{}best_model.pkl".format(save_dir), "wb") as f:
        pickle.dump(grove_model.fit(train_df[best_features], labels), f)
    pd.Series(best_features).to_csv(
        "{}best_features.csv".format(save_dir),
        index_label="Index",
        encoding="utf-8",
        header=member_col,
    )
    pd.Series(dropped_dict, name="Dropped Features").to_csv(
        "{}dropped_features.csv".format(save_dir)
    )
    return grove_model, score_dict, dropped_dict, best_features


def process_selection_data(dropped_dict, feature_df, geary_cols, labels, save_dir):
    sample_wts = compute_sample_weight(class_weight="balanced", y=labels)
    scaler = (
        RobustScaler(unit_variance=True)
        .set_output(transform="pandas")
        .fit(feature_df[geary_cols])
    )
    if save_dir is not None:
        with open("{}scaler.pkl".format(save_dir), "wb") as f:
            pickle.dump(scaler, f)
    train_df = scaler.transform(feature_df[geary_cols])
    cross_corr = train_df.corr(method="kendall")
    del_ser = find_correlation(
        cross_corr, cutoff=0.95, n_drop=max(1, train_df.shape[1] - 30)
    )
    print([c for c in del_ser.index if type(c) is not str])
    train_df.drop(columns=del_ser.index, inplace=True)
    cross_corr = cross_corr.drop(columns=del_ser.index).drop(index=del_ser.index)
    dropped_dict.update([(c, "Cross-correlation") for c in del_ser.index])
    best_corrs = train_df.corrwith(labels, method="kendall").sort_values(
        ascending=False
    )
    na_corrs = best_corrs.index[best_corrs.isna()]
    [dropped_dict.update([(c, "NA Correlation")]) for c in na_corrs]
    best_corrs.drop(na_corrs, inplace=True)
    cross_corr = cross_corr.drop(index=na_corrs).drop(columns=na_corrs)
    train_df = train_df[cross_corr.index]
    best_corrs.sort_values(ascending=False, inplace=True, key=lambda x: np.abs(x))
    train_df.to_pickle("{}feature_df.pkl".format(save_dir))
    return best_corrs, cross_corr, train_df


def record_score(
    best_features,
    fails,
    feature_list,
    save_dir,
    score_dict,
    scores,
    top_score,
    last_best,
):
    with open("{}feature_score_path.csv".format(save_dir), "a", encoding="utf-8") as f:
        f.write(
            "{}\t{}\n".format(
                "\t".join([str(sc) for sc in scores]), "\t".join(feature_list)
            )
        )
    score_dict[tuple(feature_list)] = scores
    if np.mean(scores) - np.std(scores) > top_score:
        print(
            "New top results for {} feature model: {:.4f}, {:.4f}".format(
                len(feature_list), np.mean(scores), np.std(scores)
            )
        )
        top_score = np.mean(scores) - np.std(scores)
        last_best = best_features
        best_features = feature_list
        fails = max(0, fails - 3)
    else:
        fails += 1
    return best_features, score_dict, top_score, last_best, fails


def get_model(model_name):
    if "log" in model_name:
        grove_model = LogisticRegressionCV(
            scoring=mcc,
            solver="newton-cholesky",
            tol=9e-5,
            cv=5,
            max_iter=10000,
            class_weight="balanced",
        )
    else:
        grove_model = RandomForestClassifier(
            n_jobs=-1,
            max_leaf_nodes=50,
            min_impurity_decrease=0.05,
            max_depth=6,
            class_weight="balanced",
            bootstrap=False,
        )
    return grove_model


def membership(feature_df, labels, grove_cols, kurtosis_counts, search_dir):
    grove_features = feature_df[grove_cols]
    member_dict, accepted_features = dict(), dict()
    i = 0
    for col, kurtcol in kurtosis_counts.iterrows():
        members = grove_features[col][
            grove_features[col] != grove_features[col].mode().iloc[0]
        ].index
        insol = labels[members][labels[members] == 0]
        sol = labels[members][labels[members] == 1]
        if kurtcol["Kurtosis"] > 5 and insol.shape[0] > 15 and sol.shape[0] > 15:
            print("Feature Selection for {}".format(col))
            print("Dataset size: {} {}".format(insol.shape[0], sol.shape[0]))
            member_dict[col] = members
            os.makedirs("{}{}/".format(search_dir, i), exist_ok=True)
            labels[members].to_csv("{}{}/feature_members.csv".format(search_dir, i))
            accepted_features[col] = (insol.shape[0], sol.shape[0])
            i += 1
        else:
            print(col)
            print("Dataset size: {} {}".format(insol.shape[0], sol.shape[0]))
        pd.DataFrame.from_dict(
            accepted_features, orient="index", columns=["Insoluble", "Soluble"]
        ).to_csv("{}feature_names.csv".format(search_dir), sep="\t")
    return member_dict


def main():
    feature_df, labels, grove_cols, kurtosis_stats = sample_clusters.main()
    # kurtosis_stats.columns = ["Kurtosis", "Counts"]
    pos_samples, neg_samples = labels[labels == 0].index, labels[labels == 1].index
    model_name = "logit"
    search_dir = "{}{}_grove_features_2/".format(
        os.environ.get("MODEL_DIR"), model_name
    )
    os.makedirs(search_dir, exist_ok=True)
    members_lists = membership(
        feature_df, labels, grove_cols, kurtosis_stats, search_dir
    )
    model_dict, score_dict = dict(), dict()
    for i, (col, members) in enumerate(members_lists.items)():
        col_dir = "{}/{}/".format(search_dir, i)
        os.makedirs(col_dir, exist_ok=True)
        model_dict[col], score_dict[col], dropped_dict, best_features = geary_trials(
            feature_df.loc[members],
            labels[members],
            grove_cols,
            member_col=col,
            grove_model=get_model(model_name),
            save_dir=col_dir,
        )


if __name__ == "__main__":
    main()
