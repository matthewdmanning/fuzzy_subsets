import itertools
import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import balanced_accuracy_score, make_scorer

from archive.feature_selection_plots import get_selection_scores


def violin_plots(feature_df, labels, subset, batch_size=4, title=None):
    ax_list = list()
    sns.set_theme(style="white")
    # vfig = plt.figure(num="violin", dpi=600)
    for batch in itertools.batched(np.arange(len(subset)), n=batch_size):
        feature_df = feature_df[
            [subset[b] for b in batch if subset[b] in feature_df.columns]
        ]
        feature_df.loc[:, "Solubility"] = labels.replace(
            to_replace=0, value="Insoluble"
        ).replace(to_replace=1, value="Soluble")
        print(feature_df)
        melted_feat = feature_df.melt(
            id_vars="Solubility", var_name="Features", value_name="Value"
        )
        # vax = vfig.add_subplot()
        print(melted_feat)
        vplot = sns.catplot(
            data=melted_feat,
            y="Value",
            x="Features",
            hue="Solubility",
            aspect=2,
            split=True,
            gap=0.05,
            width=0.95,
            bw_adjust=0.6,
            inner="quartile",
            scale="area",
            errorbar=("pi", 0.68),
            margin_titles=True,
            kind="violin",
        )
        # plt.tight_layout()
        plt.show()
        ax_list.append(vplot)
    return ax_list


def importance_plots(
    feature_df,
    labels,
    model,
    subset=None,
    filter=False,
    scoring=balanced_accuracy_score,
    title=None,
):
    scorer = make_scorer(scoring)
    if subset is not None:
        feature_df = feature_df[subset]
    else:
        subset = feature_df.columns.tolist()
    fit_model = model.fit(feature_df, labels)
    sns.set_theme(style="whitegrid", palette="pastel")
    r = permutation_importance(
        fit_model,
        feature_df,
        labels,
        n_repeats=100,
        n_jobs=-1,
        scoring=scorer,
        random_state=0,
    )
    imp_df = pd.DataFrame(r.importances, index=feature_df.columns)
    print("Importances")
    print(imp_df)
    imp_df.reset_index(drop=False, names="Features", inplace=True)
    ordered_features_ix = r.importances_mean.argsort()[::-1]
    if filter:
        unimportants = list()
        for i in ordered_features_ix:
            if r.importances_mean[i] - 2 * r.importances_std[i] <= 0:
                unimportants.append(i)
        imp_df.drop(index=unimportants, inplace=True)
    imp_df = imp_df[imp_df.columns[ordered_features_ix]]
    print(imp_df)
    melt_imp = imp_df.melt(
        id_vars="Features", value_name="Importance", var_name="Repeat"
    ).drop(columns="Repeat")
    print(melt_imp)
    imp_plot = sns.boxplot(
        data=imp_df,
        order=imp_df.columns.tolist(),
        # saturation=0.25,
    )
    # imp_plot = sns.barplot(data=imp_df, order=imp_df.columns.tolist(), width=0.2, fill=False, capsize=0.5, errorbar=("pi", 0.68), saturation=0.25)
    # imp_plot.set(suptitle="Feature Importances: {}".format(title))
    return imp_plot, imp_df


def partial_importance_plots(feature_df, labels, model, subset):
    pdd = PartialDependenceDisplay().from_estimator(
        model, feature_df, subset, target=labels, n_jobs=-1, random_state=0, kind="both"
    )
    return pdd


def main():
    data_dir = "{}enamine_feat_selection_12-17-24/".format(os.environ.get("MODEL_DIR"))
    # data_dir = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/test_data/Vapor pressure OPERA/Vapor pressure OPERA/"
    exp_dir = "{}test_train_split/".format(data_dir)
    classify = True
    with open("{}preprocessed_feature_df.pkl".format(data_dir), "rb") as f:
        data_df = pickle.load(f)
    labels = pd.read_csv("{}member_labels.csv".format(data_dir))
    labels = labels.set_index(keys=labels.columns[0]).squeeze()
    run_dirs = list(os.walk(exp_dir))[0][1]
    for run in run_dirs:
        score_path = "{}/".format(os.path.join(exp_dir, run))
        print(score_path)
        try:
            with open("{}best_model.pkl".format(score_path), "rb") as f:
                run_model = pickle.load(f)
        except FileNotFoundError:
            continue
        _, _, test_score_subsets = get_selection_scores(
            "{}test_scores.csv".format(score_path)
        )
        _, _, cv_score_path = get_selection_scores(
            "{}feature_score_path.csv".format(score_path)
        )
        scores_col = test_score_subsets.columns[0]
        subsets_col = test_score_subsets.columns[1]
        test_score_subsets["Mean"] = test_score_subsets[scores_col].apply(
            lambda x: x[0]
        )
        test_score_subsets["Std"] = cv_score_path.iloc[:, 0].apply(np.std)
        test_score_subsets["n_features"] = test_score_subsets[subsets_col].apply(len)
        test_score_subsets["AdjScore"] = (
            test_score_subsets["Mean"] - 0.5 * test_score_subsets["Std"]
        )
        top_scorers = test_score_subsets[
            test_score_subsets["Mean"] >= test_score_subsets["AdjScore"].max()
        ].copy()
        top_scorers.sort_values(by="n_features", inplace=True)
        top_subset = top_scorers[subsets_col].iloc[0]
        print("Top Scorers")
        print(top_scorers)
        print(data_df)
        if classify:
            imp_plot, importances = importance_plots(
                feature_df=data_df,
                labels=labels,  # .replace(to_replace=0, value="Insoluble").replace(to_replace=1, value="Soluble"),
                model=run_model,
                subset=top_subset,
                title=run,
            )
            imp_plot.set_xlabel("")
            plt.show()
            imp_plot.figure.savefig("{}{}_importances.png".format(score_path, run))
            violin = violin_plots(
                feature_df=data_df,
                labels=labels,
                subset=top_subset,
                title="asinh-Standard Scaled Features",
            )
            violin.savefig("{}{}_violins.png".format(score_path, run))
            pip_subset = importances["Features"].unique()
        else:
            pip_subset = top_subset
        if len(pip_subset) < 5:
            pdp = partial_importance_plots(
                feature_df=data_df,
                labels=labels,
                model=run_model,
                subset=pip_subset,
            )
            pdp.figure_.savefig("{}{}_pdp.png".format(score_path, run))


if __name__ == "__main__":
    main()
