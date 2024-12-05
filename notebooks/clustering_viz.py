import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from feature_selection_plots import grab_selection_results, heirarchical_predictions


def plot_dendrogram(score_df, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    model = AgglomerativeClustering(
        distance_threshold=0, n_clusters=None, metric="jaccard"
    )
    model = model.fit(score_df)
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def main(distance_thresh=None):
    model_nick, full_model_name = "logit", "Logistic Regression"
    model_dir = "{}{}_grove_features_2/".format(os.environ.get("MODEL_DIR"), model_nick)
    grove_names = pd.read_csv(
        "{}feature_names.csv".format(model_dir), index_col=0, sep="\t"
    ).squeeze()
    grove_df_dict, features_dict, best_adj_dict = grab_selection_results(
        grove_names, model_dir, full_model_name
    )
    for i in list(range(30)):
        if os.path.isdir("{}{}/".format(model_dir, i)):
            feature_df = pd.read_pickle("{}{}/feature_df.pkl".format(model_dir, i))
            labels = pd.read_csv("{}{}/feature_members.csv".format(model_dir, i), index_col=0)
            model = pd.read_pickle("{}{}/best_model.pkl".format(model_dir, i))
            score_df = heirarchical_predictions(
                feature_df, labels, model, features_dict, cv=(3, 5), save_dir=model_dir
            )
            plot_dendrogram(score_df)
    # setting distance_threshold=0 ensures we compute the full tree.


if __name__ == "__main__":
    main()
