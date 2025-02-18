import os
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from feature_selection_plots import get_selection_scores, grab_selection_results


def size_score_swarm(data, orient="listlike", id_vars=None, x_name=None, hue_name=None):
    if orient == "separate":
        long_data = data.melt(id_vars=id_vars, var_name="Fold", value_name="Score")
    elif orient == "long":
        long_data = data
    cat = sns.catplot(
        long_data,
        x=x_name,
        y="Score",
        kind="swarm",
        alpha=1,
        size=4,
        hue=hue_name,
        height=8,
        aspect=1.8,
    )
    return cat


def plot_scores_process(long_data, feature_data):
    # feature_data = data[["run", "Step", "n_features"]].copy()
    #  .melt(id_vars=["run", "Step"], value_vars=["n_features"], var_name="measure")
    # print(feature_data)
    # data.drop(columns="n_features", inplace=True)
    # long_data = data.melt(id_vars=["run", "Step"], value_vars=["Test", "Train-CV"], var_name="measure")
    pprint.pp(long_data)
    rp = sns.relplot(
        long_data,
        x="Step",
        y="value",
        col="run",
        hue="measure",
        col_wrap=3,
        kind="line",
        facet_kws={"sharey": True, "sharex": False, "despine": False, "margin_titles": False},
    )
    # rp.set(title="OPERA Vapor Pressure: 5.1 (Balanced Accuracy)", ylabel="CV Fold Score")
    # rp.set(title="OPERA Vapor Pressure: 5.1 (Balanced Accuracy)", ylabel="CV Fold Score")
    rp.set(ylim=(0.5, 1), ylabel="Balanced Accuracy")
    """
    fp = sns.relplot(
        data=feature_data,
        x="Step",
        y="n_features",
        col="run",
        col_wrap=3,
        facet_kws={"sharey": False, "sharex": False},
    ).map(
        sns.scatterplot,
        data=feature_data,
        x="Step",
        y="n_features",
        hue="run",
        style="run",
    )
    fp.set(ylim=(0, 50))
    plt.show()
    """
    for sid, rel_ax in rp.axes_dict.items():
        facet_feat = feature_data[feature_data["run"] == sid]
        print(sid)
        print()
        feat_ax = rel_ax.twinx()
        feat_ax.plot(
         facet_feat["Step"], facet_feat["n_features"], linestyle="solid", alpha=0.5, color="black"
    )
    # feat_ax.set_ylim(0, feature_data["value"].max()+5)
    # rel_ax.set_ylim(0.5, 1.0)
    # rp.figure.set(title="OPERA Vapor Pressure: 5.1 Training")
    rp.tight_layout()
    return rp


def main():
    # data_dir = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/test_data/Vapor pressure OPERA/Vapor pressure OPERA/"
    data_dir = "{}enamine_feat_selection_12-17-24/".format(os.environ.get("MODEL_DIR"))
    opera_dir = "{}test_train_split/".format(data_dir)
    # opera_dir = data_dir
    separate_scores_dict = dict()
    score_size_dict = dict()
    feature_subsets_dict = dict()
    best_adj_dict = dict()
    raw_score_subset_dict = dict()
    run_dirs = list(os.walk(opera_dir))[0][1]
    print(run_dirs)
    scores_list, feature_list = list(), list()
    for run in run_dirs:
        if "padel" in run.lower() or "sigmoid" in run.lower():
            continue
        score_path = "{}/".format(os.path.join(opera_dir, run))
        print(score_path)
        _, _, test_score_subsets = get_selection_scores(
            "{}test_scores.csv".format(score_path)
        )
        _, _, cv_score_subsets = get_selection_scores(
            "{}feature_score_path.csv".format(score_path)
        )
        test_score_subsets["Test"] = test_score_subsets["Scores"].map(lambda x: x[0])
        test_score_subsets.drop(columns="Scores", inplace=True)
        test_score_subsets["n_features"] = test_score_subsets["Subsets"].map(len)
        test_score_subsets["Features"] = test_score_subsets["Subsets"].map(lambda x: ','.join(x))
        print(test_score_subsets.columns)
        print(test_score_subsets.head())
        test_score_subsets.drop(columns=test_score_subsets.columns[0], inplace=True)
        test_score_subsets.reset_index(names="Step", inplace=True)
        cv_score_subsets["Train-CV"] = cv_score_subsets["Scores"].map(np.mean)
        cv_score_subsets.drop(columns=cv_score_subsets.columns[0], inplace=True)
        cv_score_subsets["n_features"] = cv_score_subsets["Subsets"].map(len)
        cv_score_subsets["Features"] = cv_score_subsets["Subsets"].map(lambda x: ','.join(x))
        cv_score_subsets.drop(columns="Subsets", inplace=True)
        cv_score_subsets.reset_index(names="Step", inplace=True)
        print("heads")
        print(cv_score_subsets.head())
        print(test_score_subsets.head())
        print(test_score_subsets.columns)
        combo_scores = pd.merge(
            test_score_subsets, cv_score_subsets, on=["Features", "Step", "n_features"]
        ).copy()
        print("Clipping all values at 0")
        # combo_scores.clip(lower=(0, 0, 0, 0, 0, 0), inplace=True)
        combo_scores["run"] = run
        just_scores = combo_scores.drop(columns=["Features", "n_features"]).copy()
        just_scores = just_scores.melt(id_vars=["run", "Step"], value_vars=["Test", "Train-CV"], var_name="measure", ignore_index=False)
        scores_list.append(just_scores)
        feature_df = combo_scores.drop(columns=["Test", "Train-CV", "Features"]).copy()
        feature_list.append(feature_df)
    process_fig = plot_scores_process(pd.concat(scores_list), pd.concat(feature_list))
    # process_fig.set(ylim=(0, 50))
    process_fig.savefig("{}scores_features_plots.png".format(opera_dir))
    plt.show()
    return
    for run in run_dirs:
        if os.path.isdir(score_path):
            (
                sep_scores,
                score_size_df,
                fsubs,
                best_adj,
                score_sub_df,
            ) = grab_selection_results(score_path)
            if sep_scores is None:
                continue
            sep_scores["run"] = run
            score_size_df["run"] = run
            score_sub_df["run"] = run
            separate_scores_dict[run] = sep_scores
            score_size_dict[run] = score_size_df
            raw_score_subset_dict[run] = score_sub_df
    print(separate_scores_dict.items())
    swarm_fig = size_score_swarm(
        pd.concat(separate_scores_dict.values()),
        orient="separate",
        id_vars=["n_features", "run"],
        x_name="run",
        hue_name="n_features",
    )
    swarm_fig.set(title="OPERA Vapor Pressure: 5.1 Training", ylabel="CV Fold Score")
    swarm_fig.savefig("{}cv_subset_scores.png".format(opera_dir))


if __name__ == "__main__":
    main()
