import itertools
import os
from math import atanh

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.preprocessing import StandardScaler

import data_tools
import feature_name_lists
from data_cleaning import clean_and_check


def find_sparse(feature_df, labels, sparse_cut=0.9):
    sparsity_dict = dict()
    for col in feature_df.columns:
        counts = feature_df[col].value_counts()
        if counts.iloc[0] >= sparse_cut * feature_df.shape[0]:
            sparsity_dict[col] = np.sum(counts.iloc[1:])
    [print(k, v) for k, v in sparsity_dict.items()]
    return sparsity_dict


def simplify_rings(feature_df, all_names, large_start=8):
    cdf = feature_df.copy()
    ring_num_dict, large_rings = dict(), dict()
    ring_num_dict[13] = [n for n in all_names if " >12" in n and "includes" not in n]
    large_name = "Number of >{}-membered rings".format(large_start)
    large_het_name = "Number of >{}-membered rings containing heteratoms (N, O, P, S, or halogens)".format(
        large_start
    )
    cdf[large_name] = 0
    cdf[large_het_name] = 0
    for k in np.arange(3, 13):
        # plain_rings[k] = [r for in t if "heteroatoms" not in r]
        if k < 13:
            s = [n for n in all_names if str(k) in n and ">" not in n]
            s = pd.Index(s).intersection(feature_df.columns).tolist()
            t = [r for r in s if "includes" not in r]
            ring_num_dict[k] = t
        else:
            t = ring_num_dict[k]
        if 13 > k >= large_start:
            cdf[large_name] += cdf["Number of {}-membered rings".format(k)]
            cdf.drop(columns="Number of {}-membered rings".format(k), inplace=True)
            cdf[large_het_name] += cdf[
                "Number of {}-membered rings containing heteroatoms (N, O, P, S, or halogens)".format(
                    k
                )
            ]
            cdf.drop(
                columns="Number of {}-membered rings containing heteroatoms (N, O, P, S, or halogens)".format(
                    k
                ),
                inplace=True,
            )
        """   
        if len(t) == 0:
            print("Error with {}-sized rings".format(k))
        elif k > 3 and len(t) > 0:
            if (len(t) >= 3 and
                cdf[cdf[t[0]] > 0].shape[0] < 0.05 * cdf[cdf[t[2]] > 0].shape[0]
                or cdf[cdf[t[1]] > 0].shape[0] < 0.05 * cdf[cdf[t[2]] > 0].shape[0]
            ):
                cdf.drop(columns=t[0:2], inplace=True)
            if (len(t) >= 6 and
                cdf[cdf[t[3]] > 0].shape[0] < 0.05 * cdf[cdf[t[5]] > 0].shape[0]
                or cdf[cdf[t[4]] > 0].shape[0] < 0.05 * cdf[cdf[t[5]] > 0].shape[0]
            ):
                cdf.drop(columns=t[3:5], inplace=True)
        elif cdf[cdf[t[0]] > 0].shape[0] < 0.05 * cdf[cdf[t[1]] > 0].shape[0]:
            cdf.drop(columns=t[0], inplace=True)
            """
        # print(k, [cdf[cdf[n] > 0][n].value_counts(sort=False) for n in t if n in cdf.columns])
    cdf[large_name] += cdf["Number of >12-membered rings"]
    cdf.drop(columns="Number of >12-membered rings", inplace=True)
    cdf[large_het_name] += cdf[
        "Number of >12-membered rings containing heteroatoms (N, O, P, S, or halogens)"
    ]
    cdf.drop(
        columns="Number of >12-membered rings containing heteroatoms (N, O, P, S, or halogens)",
        inplace=True,
    )
    [
        [
            cdf.drop(columns=r, inplace=True)
            for r in all_names
            if r not in k and r in cdf.columns
        ]
        for k in ring_num_dict.values()
    ]
    # [print(k, [feature_df[feature_df[n] > 0].shape[0] for n in ring_num_dict[k]]) for k in np.arange(3, 14)]
    print("Rings shape: {}".format(cdf.shape))

    return cdf


def cluster_samples(feature_df, labels, method="tsne"):
    feature_df = (
        StandardScaler()
        .set_output(transform="pandas")
        .fit_transform(feature_df.map(atanh))
    )
    if method == "spectral":

        embedded = SpectralEmbedding(
            eigen_solver="amg", n_neighbors=25, n_jobs=-1
        ).fit_transform(feature_df, labels)
        embedded = pd.DataFrame(embedded, index=feature_df.index)
    elif method == "tsne":
        n_clusters = 10
        cl = TSNE(
            verbose=1, metric="cosine", early_exaggeration=15, perplexity=50, n_jobs=-1
        ).set_output(transform="pandas")
        embedded = cl.fit_transform(feature_df)
        print(cl.kl_divergence_)
    plt.figure(figsize=(16, 10))
    embedded.columns = ["embed-2d-one", "embed-2d-two"]
    sns.scatterplot(
        x="embed-2d-one",
        y="embed-2d-two",
        hue=labels,
        palette=sns.color_palette("hls", 2),
        data=embedded,
        legend="full",
        alpha=0.025,
    )
    plt.show()


def main():
    # padel_two_d = padel_categorization.get_two_dim_only()
    count_names_dict = feature_name_lists.get_count_descriptors()
    all_count_descriptors = list()
    [
        all_count_descriptors.extend(d["Description"].to_list())
        for d in count_names_dict.values()
    ]
    all_count_descriptors = list(set(all_count_descriptors))
    feature_path = "{}data/enamine_all_padel.pkl".format(os.environ.get("PROJECT_DIR"))
    all_df = pd.read_pickle(feature_path)
    train_dfs, train_labels = data_tools.load_training_data(clean=False)
    train_labels = train_labels[all_df.index]
    # print("Zero-var features: {}".format(all_df[all_df.var(axis=1) <= 0.0001]))
    counts_df = all_df[all_count_descriptors].copy()
    remaining_df = all_df.drop(columns=all_count_descriptors)
    counts_df.drop(
        columns=counts_df.columns[counts_df.nunique(axis=0) == 1], inplace=True
    )
    remaining_df.drop(
        columns=remaining_df.columns[remaining_df.nunique(axis=0) == 1], inplace=True
    )
    # print(len(all_count_descriptors))
    remaining_df, train_labels = clean_and_check(remaining_df, train_labels)
    counts_df, train_labels = clean_and_check(counts_df, train_labels)
    counts_df = counts_df.T[~counts_df.columns.duplicated()].T
    ring_df = simplify_rings(counts_df, [c for c in counts_df.columns if "rings" in c])
    print(ring_df.shape)
    print("All clustering data: {}".format(counts_df.shape))
    # print("Zero-var shape: {}".format(ring_df.shape))
    corr_cleaned_df = exact_corr(ring_df, count_names_dict)
    # cluster_samples(ring_df, labels=train_labels, method='tsne')
    # sparse_list = find_sparse(corr_cleaned_df, train_labels)
    nonmodal_counts = pd.Series(
        [
            corr_cleaned_df[
                corr_cleaned_df[c] != corr_cleaned_df[c].mode().iloc[0]
            ].shape[0]
            for c in corr_cleaned_df.columns
        ],
        index=corr_cleaned_df.columns,
        name="Counts",
    )
    kurtosis = corr_cleaned_df.kurtosis()
    kurtosis.name = "Kurtosis"
    kc_df = pd.concat([kurtosis, nonmodal_counts], axis=1)
    kc_df.sort_values(by="Kurtosis", ascending=False, inplace=True)
    # ks_means = ks_feature_tests(feature_dfs=[corr_cleaned_df.loc[pos_samples], corr_cleaned_df.loc[neg_samples]])
    # pprint.pp(ks_means)
    # print(remaining_df.columns.intersection(corr_cleaned_df.columns))
    new_df = pd.concat([remaining_df, corr_cleaned_df.dropna(axis=1)], axis=1)
    new_df = new_df.T[~new_df.columns.duplicated()].T
    return new_df, train_labels, kc_df.index.tolist(), kc_df


def exact_corr(feature_df, count_names_dict):
    corr_df = feature_df.corr(method="kendall", numeric_only=True)
    group_names = count_names_dict["estate"]["Description"]
    for c in corr_df.columns.intersection(pd.Index((group_names.to_list()))):
        for d in corr_df.index.difference(pd.Index(group_names.to_list())):
            if c not in corr_df.columns or d not in corr_df.index:
                # print("{} or {} not found".format(c, d))
                continue
            if corr_df.loc[c, d] == 1:
                # print("Dropping columns {}".format(d))
                feature_df.drop(columns=d, inplace=True)
    # Chosen from features with correlations >= 0.95
    high_corr_cols = [
        "Total number of double bonds (excluding bonds to aromatic bonds",
        "Number of triple bonds" "Total number of bonds (including bonds to hydrogens)",
        "Triply bound carbon bound to one other carbon",
        "Number of aromatic atoms",
        "Number of atoms",
        "Number of bonds (excluding bonds with hydrogen)",
        "Total number of bonds (including bonds to hydrogens)",
        "Number of triple bonds",
        "Total number of double bonds (excluding bonds to aromatic bonds)",
        "Count of atom-type E-State: -S-ii",
    ]
    [
        feature_df.drop(columns=c, inplace=True)
        for c in high_corr_cols
        if c in feature_df.columns
    ]
    for c, d in itertools.combinations(feature_df.columns, r=2):
        if False and c != d and corr_df.loc[c, d] >= 0.95:
            print("WARNING: High correlation columns {}".format(corr_df.loc[c, d]))
            print(c)
            print(d)
    return feature_df


if __name__ == "__main__":
    main()
