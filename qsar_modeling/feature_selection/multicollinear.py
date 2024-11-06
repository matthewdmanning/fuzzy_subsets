from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


# This function uses the Spearman rank correlation to construct a hierarchical clustering of features in a dataset, calculated using the distance matrix.
# graph_hierarchy plots the results of the dendrogram.


def spearman_rank_multicollinear(
    feature_df, labels=None, corr=None, nan_policy="propagate", cluster_feats=False
):
    def graph_hierarchy():
        fig2, ax2 = plt.subplots(figsize=(60, 40), dpi=1600)
        ax1.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
        ax1.set_xticks(dendro_idx)
        ax1.set_yticks(dendro_idx)
        ax1.set_xticklabels(dendro["ivl"], rotation="vertical")
        ax1.set_yticklabels(dendro["ivl"])
        return fig2, ax2

    fig1, ax1 = plt.subplots(1, 1, figsize=(60, 40), dpi=1600)

    if nan_policy == "raise":
        assert feature_df.isna().count().count() == 0
    if corr is None:
        print(feature_df.var().shape)
        print(feature_df.var(axis=1).shape)
        print(feature_df.std(numeric_only=True) > 1e-3)
        feature_df = feature_df[
            feature_df.columns[feature_df.std(numeric_only=True) > 1e-2]
        ]
        print(feature_df.shape)
        corr = feature_df.corr(method="spearman").to_numpy()
    else:
        corr = corr.to_numpy()
    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    print(np.sum(corr - corr.T))
    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1.0 - np.abs(corr)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    print(np.sum(distance_matrix - distance_matrix.T))
    np.fill_diagonal(corr, 0)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage,
        labels=feature_df.columns.to_list(),
        ax=ax1,
        leaf_rotation=90,
        leaf_font_size=8,
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    corr_df = pd.DataFrame(corr, index=feature_df.columns, columns=feature_df.columns)
    selected_features_names = list()
    if cluster_feats:
        cluster_ids = hierarchy.fcluster(dist_linkage, 1, criterion="distance")
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
            selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
            selected_features_names = feature_df.columns[selected_features]
    fig2, ax2 = graph_hierarchy()
    return selected_features_names, corr_df, dendro, fig1, fig2
