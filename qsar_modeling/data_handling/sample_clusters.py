import itertools
import os
from math import atanh

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import clone
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import data_tools
import feature_name_lists
from data_cleaning import clean_and_check
from RingSimplifier import RingSimplifer


def find_sparse(feature_df, labels, sparse_cut=0.8):
    sparsity_dict = dict()
    for col in feature_df.columns:
        counts = feature_df[col].value_counts()
        if counts.iloc[0] >= sparse_cut * feature_df.shape[0]:
            sparsity_dict[col] = np.sum(counts.iloc[1:])
    [print(k, v) for k, v in sparsity_dict.items()]
    return sparsity_dict


def find_enriching_splits(feature_df, labels, depth=2):
    pos_members = labels[labels == 1].index
    neg_members = labels[labels == 0].index
    dt = DecisionTreeClassifier(
        max_depth=depth, class_weight="balanced", criterion="entropy"
    )
    split_dict = dict()
    for col in feature_df.columns:
        trained_tree = clone(dt).fit(feature_df[col].to_frame(), labels)
        minpurity = np.argmin(trained_tree.tree_.impurity)
        thresh = trained_tree.tree_.threshold[0]
        leaves = pd.Series(trained_tree.apply(feature_df[col].to_frame()), index=feature_df.index)
        # Need to get (weighted?) n_samples / impurity measure.
        smallest_members = leaves[leaves == minpurity].index
        pos_small = smallest_members.intersection(pos_members).size
        neg_small = smallest_members.intersection(neg_members).size
        # smallest = trained_tree.tree_.weighted_n_node_samples[minpurity]
        split_dict[col[:15]] = np.array([thresh, trained_tree.tree_.impurity[minpurity], pos_small, neg_small])
    split_df = pd.DataFrame.from_dict(split_dict, orient="index", columns=["Threshold", "Impurity", "Solubles", "Insolubles"]).sort_values(by="Impurity")
    return split_df


def cluster_split():
    KMeans(n_clusters=2, )


def simplify_rings(feature_df, all_names, large_start=8):
    print(all_names)
    cdf = feature_df.copy()
    ring_num_dict, large_rings = dict(), dict()
    ring_num_dict[13] = [n for n in all_names if " >12" in n and "includes" not in n]
    large_name = "Number of >{}-membered rings".format(str(large_start))
    large_het_name = "Number of >{}-membered rings containing heteratoms (N, O, P, S, or halogens)".format(
        str(large_start)
    )
    cdf[large_name] = 0
    cdf[large_het_name] = 0
    for k in np.arange(3, 13):
        # plain_rings[k] = [r for in t if "heteroatoms" not in r]
        if k < 13:
            m = [n for n in all_names if str(k) in n and ">" not in n]
            s = cdf.columns.intersection(pd.Index(m)).tolist()
            t = [r for r in s if "includes" not in r]
            print(s)
            ring_num_dict[k] = t
        if 13 > k >= large_start:
            ring_list = [c for c in t if str(k) in c and "hetero" not in c]
            if len(ring_list) > 0:
                ring_name = ring_list[0]
                if ring_name in cdf.columns:
                    cdf[large_name] = cdf[large_name].add(cdf[ring_name], fill_valus=0)
                    print(cdf[ring_name].value_counts)
                    cdf.drop(columns=ring_name, inplace=True)
            else:
                print(k, t)

            # het_name = "Number of {}-membered rings containing heteroatoms (N, O, P, S, or halogens)".format(k)
            het_list = [c for c in t if k in c and "hetero" in c]
            if len(het_list) > 0:
                het_name = het_list[0]
                if het_name in cdf.columns:
                    print(cdf[het_name].value_counts)
                    cdf[large_het_name] = cdf[large_het_name].add(cdf[het_name], fill_valus=0)
                    cdf.drop(
                        columns=het_name,
                        inplace=True,
                    )

            else:
                print(k, t)
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
        if "Number of >12-membered rings" in cdf.columns:
            cdf[large_name] = cdf[large_name].add(cdf["Number of >12-membered rings"], fill_valus=0)
            cdf.drop(columns="Number of >12-membered rings", inplace=True)
        if (
            "Number of >12-membered rings containing heteroatoms (N, O, P, S, or halogens)"
            in cdf.columns
        ):
            cdf[large_het_name] = cdf[large_het_name].add(cdf[
                "Number of >12-membered rings containing heteroatoms (N, O, P, S, or halogens)"
            ], fill_valus=0)
            cdf.drop(
                columns="Number of >12-membered rings containing heteroatoms (N, O, P, S, or halogens)",
                inplace=True,
            )
        [
            cdf.drop(columns=r, inplace=True)
            for r in all_names
            if r not in ring_num_dict.values() and r in cdf.columns
        ]
    # [print(k, [feature_df[feature_df[n] > 0].shape[0] for n in ring_num_dict[k]]) for k in np.arange(3, 14)]
    print(ring_num_dict.items())
    print("Rings shape: {}".format(cdf.shape))
    print(cdf[[large_name, large_het_name]].head())
    assert not cdf.isna().any().any()
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
    # ToDo: Organize Rings/Counts/Remaining split.
    # ToDo: Abstract Rings/Counts/Remaining
    # ToDo: Refactor value counts/kurtosis lines.
    # padel_two_d = padel_categorization.get_two_dim_only()
    count_names_dict = feature_name_lists.get_count_descriptors()
    all_count_descriptors = list()
    [
        all_count_descriptors.extend(d["Description"].to_list())
        for d in count_names_dict.values()
    ]
    all_count_descriptors = list(set(all_count_descriptors))
    all_df, train_labels = grab_enamine_data()
    # Combine ring count features.
    ringer = RingSimplifer().fit(all_df)
    ring_df = ringer.transform(all_df)
    # print("Zero-var features: {}".format(all_df[all_df.var(axis=1) <= 0.0001]))
    zero_var_counts = ring_df.columns[ring_df.nunique() == 1]
    ring_df.drop(columns=zero_var_counts, inplace=True)
    all_df = clean_and_check(ring_df, train_labels)
    counts_df = all_df[all_df.columns[all_df.columns.isin(all_count_descriptors)]].copy()
    remaining_df = all_df.drop(columns=counts_df.columns)
    print(remaining_df.shape)
    # print("Zero var: {}".format(zero_var_counts))
    # print(len(all_count_descriptors))
    print("All clustering data: {}".format(counts_df.shape))
    sparse_dict = find_sparse(counts_df, train_labels)
    sparse_grove_df = counts_df[list(sparse_dict.keys())]
    dense_df = counts_df.drop(sparse_grove_df.columns)
    dense_corr_df = exact_corr(dense_df, count_names_dict)
    corr_cleaned_df = exact_corr(sparse_grove_df, count_names_dict)
    exact_drop, corr_cleaned_df = exact_corr(sparse_grove_df, count_names_dict)
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
    print(new_df.shape)
    return new_df, train_labels, corr_cleaned_df.columns.tolist(), kc_df


def grab_enamine_data():
    feature_path = "{}data/enamine_all_padel.pkl".format(os.environ.get("PROJECT_DIR"))
    all_df = pd.read_pickle(feature_path)
    train_dfs, train_labels = data_tools.load_training_data(clean=False)
    train_labels = train_labels[all_df.index]
    return all_df, train_labels


def exact_corr(feature_df, count_names_dict, corr_meth="kendall"):
    feature_df = feature_df.copy()
    corr_df = feature_df.corr(method=corr_meth, numeric_only=True)
    group_names = count_names_dict["estate"]["Description"]
    drop_list = list()
    for c in corr_df.columns.intersection(pd.Index((group_names.to_list()))):
        for d in corr_df.index.difference(pd.Index(group_names.to_list())):
            if c not in corr_df.columns or d not in corr_df.index:
                # print("{} or {} not found".format(c, d))
                continue
            if corr_df.loc[c, d] == 1:
                # print("Dropping columns {}".format(d))
                drop_list.append(d)
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
    drop_list.extend(high_corr_cols)
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
    return feature_df, drop_list


if __name__ == "__main__":
    main()
