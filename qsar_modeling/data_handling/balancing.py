import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def balance_df(df, sample_wts, rounding="auto"):
    raise NotImplementedError
    if type(sample_wts) is dict:
        sample_wts = pd.Series(sample_wts)
    if rounding == "auto":
        sample_wts = sample_wts.round(0)
    elif rounding == "ceiling" or rounding == "ceil":
        sample_wts = pd.Series(np.ceil(sample_wts), index=sample_wts.index)
    elif rounding == "floor":
        sample_wts = pd.Series(np.floor(sample_wts), index=sample_wts.index)
    divisor = np.gcd(sample_wts)
    if divisor > 1:
        sample_wts = sample_wts.divide(divisor)
    if sample_wts.le(1).all():
        print(
            "N will increase by {} when DataFrame of N={}.".format(
                sample_wts.subtract(1).sum(), df.shape[0]
            )
        )
        balancing_df = None


# from statsmodels.stats.moment_helpers import corr2cov
def get_epa_sol_all_insol(feature_df, labels, tups):
    # Retrieves all insoluble and EPA soluble compounds and returns as a combined feature DataFrame and solubility Series.
    # insol_samples = pd.concat([tups['epa_in'][0], tups['en_in'][0]]).index.intersection(train_insols)
    # train_sols = labels[labels == 1].index
    # sol_samples = tups['epa_sol'][0].index.intersection(train_sols)
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


def mixed_undersampling(
    minority_group: pd.DataFrame,
    majority_group,
    maj_ratio,
    min_sizes=None,
    random_state=0,
):
    # Combines minority groups and undersamples majority groups.
    # maj_ratio is an iterable of each majority group's sampled size relative to total minority size.
    # Undersamples majority class, like Imbalanced-Learns Random Undersampler, but allows different ratios of
    # categories within the majority class (given by maj_ratio). The size of the minimum class(es) can be specified in
    # min_sizes.
    if np.iterable(minority_group):
        if min_sizes is not None and not np.iterable(min_sizes):
            all_min = [minority_group.sample(min_sizes, random_state=random_state)]
        elif min_sizes is not None and np.iterable(min_sizes):
            all_min = [
                m.sample(s, random_state=random_state)
                for m, s in zip(minority_group, min_sizes)
            ]
        else:
            all_min = minority_group
    else:
        all_min = [minority_group]
    if not np.iterable(majority_group):
        majority_group = [majority_group]
    maj_sizes = [int(r * np.sum([a.shape[0] for a in all_min])) for r in maj_ratio]
    all_maj = [
        m.sample(s, random_state=random_state)
        for m, s in zip(majority_group, maj_sizes)
    ]
    # sampler = RandomUnderSampler(sampling_strategy=maj_ratio, random_state=random_state)
    return all_min, all_maj


def data_by_groups(labels, group_dict):
    # Makes a dictionary of pandas Indexes, each containing the members of a group.
    # Group membership is identified by group_dict. Labels is a pandas Series/DF that contains all members, labelled by group membership
    ind_dict = dict()
    for k, v in group_dict.items():
        if type(v) is list or type(v) is list:
            ind_dict[k] = v[0].index.intersection(labels.index)
        elif type(v) is pd.DataFrame or type(v) is pd.Series:
            ind_dict[k] = v.index.intersection(labels.index)
    return ind_dict


def random_test_train_by_group(combo_data, split_ratio=0.2, random_state=0):
    # This is the function used to generate the new train/test split. Nov 11, 2024
    train_test_dict = dict()
    test_dict, train_dict = dict(), dict()
    for k, df in combo_data.items():
        train_idx, test_idx = train_test_split(
            df.index.tolist(), test_size=split_ratio, random_state=random_state
        )
        train_test_dict[k] = train_idx, test_idx
        test_dict[k] = test_idx
        train_dict[k] = train_idx
    # all_idx_dict = dict([(k, df.index) for k, df in combo_data.items()])
    return train_dict, test_dict
