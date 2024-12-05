import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import clone
from sklearn.model_selection import StratifiedKFold


# import clustering_viz


def get_score_features(feature_dir, score_path):
    if os.path.isdir(feature_dir) and os.path.isfile(score_path):
        with open(score_path, encoding="utf-8") as f:
            best_list = f.readlines()
    else:
        print("Score file not found: {}".format(score_path))
        return None, None
    scores, feature_lists = list(), list()
    for line in best_list:
        score_feats = line.split("\t")
        scores.append([float(x) for x in score_feats[:5]])
        feature_lists.append(score_feats[5:])
    score_feat_df = pd.DataFrame(
        data=(list(zip(scores, [len(f) for f in feature_lists]))),
        index=list(range(len(scores))),
        columns=["Scores", "n_features"],
    )
    return score_feat_df, feature_lists


def heirarchical_predictions(
    feature_df, labels, model, features_dicts, cv=(3, 5), save_dir=None
):
    # cv_members = pd.DataFrame(index=labels.index, columns=["Split", "Repeat"])
    print(feature_df.head(), labels.head())
    # Feature-set dictionary of lists of CV split predictions
    feature_repeats_predicts = dict().fromkeys(features_dicts.keys(), list())
    interrater_feats = dict().fromkeys(features_dicts.keys())
    set_sers_list, all_repeats_feat_sers = list(), dict()
    # Fit, predict and organize results.
    for repeat_i in list(range(cv[1])):
        feat_cv_preds_dict = dict().fromkeys(features_dicts.keys(), dict())
        rand_state = 4 * repeat_i**2
        for split_i, (train, test) in enumerate(
            StratifiedKFold(
                n_splits=cv[0], shuffle=True, random_state=rand_state
            ).split(feature_df, labels)
        ):
            train_df, train_y = feature_df.iloc[train], labels.iloc[train]
            test_df, test_y = feature_df.iloc[test], labels.iloc[test]
            # cv_members.iloc[test] = (split, repeat)
            for set_ix, feature_set in features_dicts.items():
                new_model = clone(model).fit(X=train_df[feature_set], y=train_y)
                feat_cv_preds_dict[set_ix][split_i] = new_model.predict(
                    X=test_df[feature_set]
                )
        for set_ix, feature_set in features_dicts.items():
            feature_repeats_predicts[set_ix].append(
                pd.concat([feat_cv_preds_dict[set_ix][i] for i in list(range(cv[0]))])
            )
            ser = feature_repeats_predicts[set_ix][repeat_i]
            ser.rename(
                [
                    "{}_{}".format(str(ix), str(repeat_i))
                    for ix in feature_repeats_predicts[set_ix][repeat_i].index
                ]
            )
            set_sers_list[set_ix].append(ser)
    for set_ix in features_dicts.keys():
        all_repeats_feat_sers[set_ix] = pd.concat(set_sers_list[set_ix])
        interrater_feats[set_ix] = pd.concat(
            feature_repeats_predicts[set_ix], columns=list(range(cv[0])), axis=1
        )
        interrater_feats[set_ix]["InterRepeat"] = interrater_feats.apply(
            lambda x: np.abs(2 * np.mean(x) - 1)
        )
        all_repeats_feat_sers[set_ix].to_pickle(
            "{}{}_set_cv_{}x{}_predict.pkl".format(save_dir, set_ix, cv[0], cv[1])
        )
        all_repeats_feat_sers[set_ix].to_csv(
            "{}{}_set_cv_{}x{}_predict.csv".format(save_dir, set_ix, cv[0], cv[1])
        )
    return pd.DataFrame.from_dict(all_repeats_feat_sers)


def grab_selection_results(grove_names, model_dir, model_name, best_only=True):
    grove_df_dict, features_dict, best_adj_dict = dict(), dict(), dict()
    for i in list(range(0, 100)):
        feature_dir = "{}{}/".format(model_dir, i + 7)
        score_path = "{}feature_score_path.csv".format(feature_dir)
        if not os.path.isdir(feature_dir) or not os.path.isfile(score_path):
            continue
        score_feat_df, feature_lists = get_score_features(feature_dir, score_path)
        if score_feat_df is None:
            continue
        score_feat_df["Mean"] = score_feat_df["Scores"].apply(np.mean)
        score_feat_df["Std"] = score_feat_df["Scores"].apply(np.std)
        score_feat_df["AdjMean"] = score_feat_df["Mean"] - score_feat_df["Std"]
        score_feat_df.sort_values(by="AdjMean").drop_duplicates(
            subset="n_features", inplace=True
        )
        best_adj_dict[i] = score_feat_df["AdjMean"].max()
        score_feat_df.drop(columns=["Mean", "Std", "AdjMean"], inplace=True)
        score_feat_df = score_feat_df.explode(column=["Scores"], ignore_index=True)
        # score_feat_df["Grove"] = grove_names[i].split("\t")[0]
        grove_df_dict[i] = score_feat_df
        features_dict[i] = feature_lists
    return grove_df_dict, features_dict, best_adj_dict


def main():
    model_nick, full_model_name = "logit", "Logistic Regression"
    model_dir = "{}{}_grove_features_2/".format(os.environ.get("MODEL_DIR"), model_nick)
    grove_names = pd.read_csv(
        "{}feature_names.csv".format(model_dir), index_col=0, sep="\t"
    ).squeeze()
    grove_names.name = "Grove"
    grove_nick = grove_names.index.map(mapper=lambda x: x[: min(len(x), 40)]).tolist()
    grove_df_dict, features_dict, best_adj_dict = grab_selection_results(
        grove_nick, model_dir, full_model_name
    )
    grove_df_dict = dict(
        sorted(grove_df_dict.items(), key=lambda x: best_adj_dict[x[0]], reverse=True)
    )
    # sns.set_context("talk")
    grove_nick = [
        "{:.50s}, {}, {}".format(x[:50], y[0], y[1]) for x, y in grove_names.iterrows()
    ]
    grove_list = list(grove_df_dict.values())
    sns.set_style("whitegrid")
    for i, (grove_dfs) in enumerate(itertools.batched(grove_list, n=4)):
        for ix, df in enumerate(grove_dfs):
            df["Grove"] = grove_nick[ix]
        grove_df = pd.concat(grove_dfs)
        score_grid = sns.FacetGrid(
            grove_df, col="Grove", col_wrap=2, height=5, xlim=(0, 31), ylim=(0.5, 1.0)
        )
        score_grid.map(sns.boxplot, "n_features", "Scores", native_scale=True)
        score_grid.set_axis_labels("Num Features", "CV Scores")
        score_grid.set(xticks=[0, 10, 20, 30], yticks=[0.5, 0.6, 0.7, 0.8, 0.9])
        score_grid.figure.subplots_adjust(wspace=0.02, hspace=0.2)
        score_grid.savefig("{}feature_selection_best_{}.png".format(model_dir, i))
        score_grid.fig.subplots_adjust(top=0.9)
        score_grid.fig.suptitle(full_model_name)
        plt.show()


if __name__ == "__main__":
    main()
