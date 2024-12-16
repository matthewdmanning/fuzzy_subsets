import copy
import os
import pickle
import pprint

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    SpectralBiclustering,
)
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import distance
from sklearn.neighbors import radius_neighbors_graph

import feature_selection_plots
from feature_selection_plots import get_subset_predictions, grab_selection_results


def create_distance_matrix(score_df, metric="auto"):
    """

    Parameters
    ----------
    score_df : pd.DataFrame[bool] (n_samples, n_estimator instances)
    Contains binary predictions for multiple classifiers/feature sets.
    """
    if np.unique(score_df).size == 2 and metric == "auto":
        metric = distance.jaccard
    elif metric == "auto":
        metric = distance.sqeuclidean
    sim_data = pd.DataFrame(
        index=score_df.columns, columns=score_df.columns, dtype=np.float64
    )
    for x in np.arange(score_df.shape[1]):
        for y in np.arange(x, score_df.shape[1]):
            d = metric(score_df.iloc[:, x], score_df.iloc[:, y])
            if np.isnan(d):
                break
            sim_data.iloc[x, y] = d
            sim_data.iloc[y, x] = d
    for i in list(range(sim_data.shape[1])):
        sim_data.iloc[i, i] = 1
    return sim_data


def feature_subset_connectivity(subsets_list):
    all_features = set()
    [all_features.difference_update(s) for s in subsets_list]
    subsets_df = pd.DataFrame(
        data=0, columns=all_features, index=list(range(len(subsets_list)))
    )
    for s_i, subset in enumerate(subsets_list):
        for f in all_features:
            if f in subset:
                subsets_df[f].iloc[s_i] = 1
        subsets_df.iloc[s_i].transform(lambda x: x / subsets_df.iloc[s_i].sum())
    subset_connectivity = radius_neighbors_graph(
        X=subsets_df.to_numpy(), mode="connectivity", n_jobs=-1
    )
    return subset_connectivity


def weighted_subset_similarity(
    feature_df, subsets_list, weight="pearson", greedy=False
):
    af = set()
    for c in subsets_list:
        af.update(c)
    all_features = list(af)
    print(len(all_features))
    print(feature_df.head())
    xcorr_df = (
        feature_df[all_features].corr(method=weight).abs().clip(lower=0.0, upper=1.0)
    )
    similarity_df = pd.DataFrame(
        0.0, columns=np.arange(len(subsets_list)), index=np.arange(len(subsets_list))
    )
    for i, first in enumerate(subsets_list):
        for k, second in enumerate(subsets_list[i + 1 :]):
            j = k + i + 1
            f = copy.deepcopy(first)
            s = copy.deepcopy(second)
            while True:
                r, c = linear_sum_assignment(
                    xcorr_df[s].loc[f].to_numpy(), maximize=True
                )
                a = [f[a] for a in r]
                b = [s[a] for a in c]
                for row, col in list(zip(r, c)):
                    score = xcorr_df[s].loc[f].iat[row, col]
                    similarity_df.iloc[i, j] += score
                if len(f) > len(s):
                    [f.remove(t) for t in a]
                elif len(s) > len(f):
                    [s.remove(t) for t in b]
                else:
                    break
                if len(s) == 0 and len(f) == 0:
                    break
            # score = xcorr_df[assignments].sum() / max(len(first), len(second))
            similarity_df.iloc[i, j] = similarity_df.iloc[i, j] / (
                len(first) + len(second)
            )
            similarity_df.iloc[j, i] = similarity_df.iloc[i, j]
    similarity_df.clip(upper=0.999, inplace=True)
    for i in np.arange(len(subsets_list)):
        similarity_df.iloc[i, i] = 1.0
    print(similarity_df)
    return similarity_df


def similarity_heatmaps(
    score_df, feature_subsets_list, feature_df, labels, save_dir=None
):
    # dissim_mat = create_distance_matrix(score_df)
    n_subsets = [len(s) for s in feature_subsets_list]
    normed_n_subsets = [n / max(n_subsets) for n in n_subsets]
    print("Plotting Similarity Clustermap")
    print(score_df.shape)
    # if score_df.shape[1] < score_df.shape[0]:
    #    score_df = score_df.T
    subset_sim_df = weighted_subset_similarity(feature_df, feature_subsets_list)
    print("subset similarity: \n{}".format(subset_sim_df))
    subset_sim_df.to_pickle("{}subset_similarity_df.pkl".format(save_dir))
    subset_links = linkage(squareform(1 - subset_sim_df), optimal_ordering=True)
    with open("{}feature_subsets_linkage.pkl".format(save_dir), "wb") as f:
        pickle.dump(subset_links, f)
    score_mds = MDS(
        n_components=10, n_init=10, max_iter=2500, n_jobs=-1, random_state=0
    )
    mds_df = score_mds.fit_transform(feature_df.T)
    dissim = score_mds.dissimilarity_matrix_
    # score_dist = pd.DataFrame(score_arr)
    # score_dist = squareform()
    score_dist = squareform(pdist(X=score_df.T, metric="sqeuclidean"))
    # pprint.pp(pd.DataFrame(score_dist), compact=True, width=120)
    score_links = linkage(dissim, method="ward", optimal_ordering=True)
    print("Score links")
    pprint.pp(score_links)
    pprint.pp(np.shape(squareform(score_links)))
    with open("{}score_ward_linkage.pkl".format(save_dir), "wb") as f:
        pickle.dump(score_links, f)
    # sns.scatterplot(x=subset_sim_df.unstack(), y=score_dist.unstack(), palette="crest", hue=normed_n_subsets, hue_norm=[0, 30], alpha=0.25)
    # plt.savefig("{}score_subset_sim_scatter.png".format(save_dir))
    fig = sns.clustermap(
        score_df,
        col_cluster=True,
        row_cluster=True,
        row_linkage=subset_links,
        col_linkage=score_links,
        row_colors=labels,
        dendrogram_ratio=(0.4, 0.2),
        colors_ratio=0.01,
        linewidth=0,
        antialiased=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.show()
    fig.ax_heatmap.set(ylabel="Subset Linkages", xlabel="Prediction Linkages")
    # sns.heatmap(dissim_mat, square=True, vmin=0, vmax=1)
    if save_dir is not None:
        fig.savefig("{}score_subsets_heatmap.png".format(save_dir))


def plot_affinity_prop(score_df, labels):
    print("Data shape: {}".format(score_df.shape))
    pprint.pp(score_df)
    jmem = joblib.Memory(os.environ.get("JOBLIB_TMP"))

    two_d_array = PCA(n_components=2, whiten=True).fit_transform(score_df)
    two_d_data = pd.DataFrame(
        two_d_array, index=score_df.index, columns=["pca0", "pca1"]
    )
    af = AffinityPropagation(damping=0.95, max_iter=2500, random_state=0).fit(
        two_d_data.to_numpy()
    )
    print("Iterations required: {}".format(af.n_iter_))
    print("Clusters: {}".format(af.labels_))
    print("Samples: {}".format(len(af.labels_)))
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    print("Number of clusters: {}".format(n_clusters_))
    """
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, af.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, af.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, af.labels_))
    print(
        "Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels, af.labels_)
    )
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels, af.labels_)
    )
    print(
        "Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(score_df, af.labels_, metric="sqeuclidean")
    )
    """
    fig, ax = plt.subplots(layout="constrained", clear=True)

    colors = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, n_clusters_)))
    for k, col in zip(list(range(n_clusters_)), colors):
        class_members = af.labels_ == k
        print("Class members: {}".format(class_members))
        cluster_center = two_d_data.iloc[cluster_centers_indices[k]]
        ax.scatter(
            two_d_data.iloc[class_members, 0],
            two_d_data.iloc[class_members, 1],
            color=col["color"],
            marker=".",
            s=10,
            alpha=0.5,
        )
        ax.scatter(
            cluster_center[0],
            cluster_center[1],
            s=16,
            color=col["color"],
            marker="*",
            alpha=1,
        )
        print("Cluster center: \n{}".format(cluster_center))
        for idx, x in two_d_data[class_members].items():
            print(cluster_center, x)
            ax.plot(
                data=[[cluster_center[0], x], [cluster_center[1], x]],
                s=7,
                color=col["color"],
                alpha=0.75,
            )
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(ticks=None)
    plt.yticks(ticks=None)
    plt.show()


def plot_dendrogram(feature_df, score_df, subsets, save_dir, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    jmem = joblib.Memory(os.environ.get("JOBLIB_TMP"))
    print(score_df.shape)
    # feature_connectivity = feature_subset_connectivity(subsets)
    subset_sim_df = weighted_subset_similarity(feature_df, subsets)
    print(subset_sim_df)
    print(subset_sim_df.shape)
    sample_connect = linkage(subset_sim_df.to_numpy(), method="ward")
    # print(np.shape(sample_connect))
    model = AgglomerativeClustering(
        n_clusters=4,
        memory=jmem,
        compute_distances=True,
        # connectivity=sample_connect,
    )
    model.fit(score_df.to_numpy())
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    print("Number of clustering samples: {}".format(n_samples))
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
    # plot the top three levels of the dendrogram
    dendrogram(
        linkage_matrix, truncate_mode="level", p=7, no_labels=True, distance_sort=True
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig("{}agg_cluster_trial.png".format(save_dir))
    plt.show()


def main(distance_thresh=None):
    model_nick, full_model_name = "rfc", "Logistic Regression"
    model_dir = "{}{}_all_samples_2/".format(os.environ.get("MODEL_DIR"), model_nick)
    """
    grove_names = pd.read_csv(
        "{}feature_names.csv".format(model_dir), index_col=0, sep="\t"
    ).squeeze()
    """
    feature_dir = model_dir
    feature_df, feature_subsets, labels, mean_scores_df, separate_scores_df = (
        get_selection_results(feature_dir)
    )
    similarity_heatmaps(
        separate_scores_df, feature_subsets, feature_df, labels, feature_dir
    )
    plot_dendrogram(feature_df, mean_scores_df, feature_subsets, feature_dir)

    pass
    for i in list(range(0, 50)):
        feature_dir = "{}{}/".format(model_dir, i)
        feature_dir = model_dir
        if os.path.isdir(feature_dir):
            if os.path.isfile("{}score_subsets_heatmap.png".format(feature_dir)):
                continue
            feature_df, feature_subsets, labels, mean_scores_df, separate_scores_df = (
                get_selection_results(feature_dir)
            )
            similarity_heatmaps(
                separate_scores_df, feature_subsets, feature_df, labels, feature_dir
            )
            # print(predict_df.head())
            # print(predict_df.shape)
            # repeat_sorted_nested = [["{}_{}".format(idx, r) for r in np.arange(5)] for idx in labels.sort_values().index]
            # repeat_sorted = [c for c in itertools.chain(*repeat_sorted_nested)]
            model = SpectralBiclustering((3, 10)).fit(mean_scores_df.T)
            reordered_rows = mean_scores_df.T.to_numpy()[np.argsort(model.row_labels_)]
            reordered_data = reordered_rows[:, np.argsort(model.column_labels_)]

            sns.heatmap(reordered_data)
            # plt.matshow(reordered_data, cmap=plt.cm.Blues)
            plt.title("After biclustering; rearranged to show biclusters")
            plt.show()
            plot_dendrogram(feature_df, mean_scores_df, feature_subsets, feature_dir)
            similarity_heatmaps(
                mean_scores_df.loc[labels.sort_values().index].T,
                feature_subsets,
                feature_df,
                labels,
                feature_dir,
            )

            # for scores in score_df:
    # setting distance_threshold=0 ensures we compute the full tree.


def get_selection_results(feature_dir):
    separate_scores_df, grove_df, feature_subsets, best_adj, raw_score_subset_df = (
        grab_selection_results(feature_dir)
    )
    feature_df = pd.read_pickle("{}preprocessed_feature_df.pkl".format(feature_dir))
    labels = pd.read_csv(
        "{}member_labels.csv".format(feature_dir), index_col="Unnamed: 0"
    ).squeeze()
    labels.name = "Solubility"
    model = pd.read_pickle("{}best_model.pkl".format(feature_dir))
    predict_df, interrater_df = get_subset_predictions(
        feature_df,
        labels,
        model,
        features_lists=feature_subsets,
        save_dir=feature_dir,
    )
    # interrater_df.drop(index=interrater_df.iloc[:2], inplace=True, ignore_index=True)
    print(interrater_df)
    # print(interrater_df.columns)
    # [df.drop(index=df.index[:2], inplace=True) for df in interrater_df.values()]
    mean_scores_df = feature_selection_plots.average_subset_repeats(interrater_df)
    return feature_df, feature_subsets, labels, mean_scores_df, separate_scores_df


if __name__ == "__main__":
    main()
