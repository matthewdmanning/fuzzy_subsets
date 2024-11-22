import itertools
import os
import pprint
from functools import partial

import numpy as np
import pandas as pd
from sklearn._config import set_config as sk_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.utils import check_X_y

import data_handling.data_cleaning
from data import feature_name_lists
from data_handling.balancing import data_by_groups
from qsar_modeling.data_handling.data_cleaning import remove_duplicate_idx
from qsar_modeling.data_handling.data_tools import (
    get_interpretable_features,
    load_metadata,
)
from quick_models import combined_rus_cv_results

# from utils import cv_tools

"""
This script runs an unsampled, cross-validated model training, with optional hyperparameterization, to predict DMSO
solubility using EPA Tox21 data and Enamine data, stored on OChem.eu and published by Tetko, et al.
The primary logic is contained in main(). Other functions are used to implement this logic and handle complex tasks,
such as implementing undersampling/cross-validation loops.
"""

pd.set_option("display.precision", 4)
pd.set_option("format.precision", 4)
np.set_printoptions(formatter={"float": "{: 0.4f}".format})
sk_config(transform_output="pandas", display="text", print_changed_only=True)


def load_sample_keys(filepath):
    # Loads INCHIKeys for data in filepath and verifies that they end in '-N' as QSAR-ready INCHIKeys do.
    sample_list = list()
    with open(filepath, "r", encoding="utf-8") as f:
        [sample_list.extend(s.split(",")) for s in f.readlines()]
        sample_list = [s.partition("\\")[0] for s in sample_list if s.endswith("-N")]
        # csv.reader(f)
    return sample_list


def load_feature_rankings(filepath, threshold=None):
    # Gives features/descriptors selected for training as a pandas Index.
    # Threshold can passed to specify how many (groups) of selected features are returned. Includes ties in selection process.
    ranking_ser = (
        pd.read_csv(filepath).set_index(keys="Features", inplace=False).squeeze()
    )  # , index_col="Features")
    if threshold is not None:
        ranking_ser.drop(ranking_ser[ranking_ser > threshold].index, inplace=True)
    return ranking_ser.index


def quadratic_splits(grouped_sers, n_splits=5):
    # Takes separate groups (each in separate Series), gets n_splits splits, and yields indices of test set.
    # Used in the case that data contains more than one type of categories, with each combination in a different subset.
    # Example: EPA and soluble, EPA and insoluble, Enamine and insoluble, Enamine and soluble.
    test_list, train_list = list(), list()
    indices = [s.copy().index.tolist() for s in grouped_sers]
    [np.random.shuffle(idx) for idx in indices]
    nested_splits = list()
    for ind in indices:
        spaces = np.linspace(0, len(ind) - 1, num=n_splits + 1, dtype=int)
        nested_splits.append(
            [ind[int(a) : int(b)] for a, b in itertools.pairwise(spaces)]
        )
    return nested_splits


def get_quadratic_test_folds(grouped_sers, n_splits=5):
    # Wrapper function for quadratic_splitter
    fold_list = list()
    quad_splits = quadratic_splits(grouped_sers, n_splits)
    # print(quad_splits)
    for qx in quad_splits:
        for fold_id, idxs in enumerate(qx):
            fold_list.append(pd.Series(data=[fold_id for _ in idxs], index=idxs))
    return pd.concat(fold_list)


def get_notebook_data(epa_soluble=True):
    def get_features(feature_path, feature_df, ecounts=False):
        if os.path.isfile(ranked_features_path):
            selected_feats = data_handling.data_tools.load_feature_rankings(
                ranked_features_path, threshold=1
            )
            print("Ranked features loaded: \n{}".format(len(selected_feats)))
        else:
            if ecounts:
                print("Using E-count descriptors")
                selected_feats = feature_name_lists.get_estate_counts(
                    feature_df.columns.tolist()
                )
            else:
                selected_feats = feature_df.index
        return selected_feats

    # Select EPA soluble/combined insoluble dataset.
    # meta_dict = dict(zip(['epa_sol', "epa_in", "en_in", "en_sol"], meta))
    meta = load_metadata(desc=False)
    interp_X, interp_y = get_interpretable_features()
    valid_inchi, invalid_inchi = data_handling.data_cleaning.check_inchi_only(
        interp_y.index
    )
    if len(invalid_inchi) > 0:
        print("Invalid INCHI keys:")
        print([i for i in invalid_inchi])

    # Drop duplicates
    print("Original size: {}".format(interp_X.shape))
    g_dict = data_by_groups(interp_y, meta)
    full_groups_list = dict(
        [
            (k, pd.Series(data=k, index=v[0].index, name=k))
            for k, v in g_dict.items()
            if k in ["epa_sol", "epa_in", "en_in"]
        ]
    )
    meta["en_in"][0].drop(
        index=meta["en_in"][0].index.intersection(meta["epa_in"][0].index), inplace=True
    )
    full_groups_list = dict(
        [(k, v[0]) for k, v in meta.items() if k in ["epa_sol", "epa_in", "en_in"]]
    )
    new_idx = (
        remove_duplicate_idx(pd.concat(full_groups_list.values(), sort=True))
        .squeeze()
        .index.intersection(interp_y.index)
    )
    valid_idx = new_idx.intersection(valid_inchi)
    unique_y = interp_y[valid_idx]
    unique_X = interp_X.loc[valid_idx]
    check_X_y(unique_X, unique_y)

    return unique_X, unique_y


model_passing = True
feature_method = "file"
n_features_out = 40
project_dir = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/qsar-modeling-workflow/"
# exp_dir = '{}models/hyperopted-training_curve/'.format(rfe_dir)
feature_dir = "{}models/epa_solubles_hyperparam/".format(project_dir)
exp_dir = "{}models/epa_solubles_final_test/".format(project_dir)
train_data_dir = "{}epa_solubles_atom_type_count/".format(project_dir)
key_path = "{}EPA-only_keys.csv".format(project_dir)
ranked_features_path = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/qsar-modeling-workflow/models/epa_solubles_hyperparam/feature_rankings.csv"


def train_model():
    combined_X, combined_y = get_notebook_data()
    if not os.path.isfile(key_path):
        combined_y.to_csv(key_path)
    selected_train_X, selected_train_y = get_interpretable_features(
        combined_X, combined_y
    )
    # full_groups = full_groups[full_groups.index.drop_duplicates()]
    # Determine feature set.
    # ranked_features_path = '{}feature_rankings.csv'.format(feature_dir)
    ranked_features = pd.DataFrame([])
    check_X_y(selected_train_X, selected_train_y)

    # Get split indices.
    """
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
        
    for i in np.arange(5):
         i = i + 1
        rus_dir = '{}rus{}/'.format(exp_dir, i)
        os.makedirs(rus_dir, exist_ok=True)
        if not os.path.isdir(rus_dir) and not os.path.exists(rus_dir):
            print('Not here.')
        else:
            print('{} already exists.'.format(rus_dir))
      """
    """
    with open('{}X_train.pkl'.format(exp_dir), 'wb') as f:
        pickle.dump(selected_train_X, f)
    with open('{}y_train.pkl'.format(exp_dir), 'wb') as f:
        pickle.dump(combined_y, f)
    with open('{}grouped_by_source.pkl'.format(exp_dir), 'wb') as f:
        pickle.dump(full_groups_list, f)
        """
    # rand_idx = interp_y.index.tolist()
    # random.shuffle(rand_idx)
    # shuffle_X = interp_X[ranked_features].loc[rand_idx]
    # shuffle_y = interp_y[rand_idx]
    epa_rand = selected_train_y.sample(frac=1.0, random_state=0).index
    shuffle_epa_X = selected_train_X.loc[epa_rand]
    shuffle_epa_y = selected_train_y[epa_rand]
    # shuffle_X, shuffle_y = RandomUnderSampler(random_state=rus_random_state).fit_resample(shuffle_epa_X, shuffle_epa_y)
    # epa_rus, epa_labels = RandomUnderSampler(random_state=rus_random_state).fit_resample(selected_train_X, selected_train_y)
    # insol_ser = pd.concat([full_groups_list['epa_in'], full_groups_list['en_in']], verify_integrity=True).squeeze()
    # full_groups_list['epa_sol'].drop(index=full_groups_list['epa_sol'].index.intersection(insol_ser.index), inplace=True)
    # shared_ind = rus_y.index.drop_duplicates().intersection(full_groups.index.drop_duplicates())
    # shared_ind = [c for c in rus_y.index.drop_duplicates() if c in full_groups.index.drop_duplicates()]
    # Even group sampling
    # rus_groups = [g[g.index.intersection(rus_y.index)] for g in full_groups_list.values()]
    # group_folds = remove_duplicate_idx(get_quadratic_test_folds(rus_groups))
    # cv_method = PredefinedSplit(group_folds)
    # print('Input size: {}'.format(shuffle_y.size))
    # print('RUS size: {}'.format(rus_y.size))
    # rus_y = remove_duplicate_idx(rus_y)
    # rus_X = remove_duplicate_idx(rus_X)
    # scramble_y = pd.Series(data=np.random.randint(0, 2, size=rus_y.size), index=rus_y.index)
    #
    # The final inputs into the CV cycle
    #
    # final_X, final_y = shuffle_epa_X, shuffle_epa_y
    final_X, final_y = selected_train_X, selected_train_y
    cv_method = partial(StratifiedKFold, shuffle=True, random_state=0)
    samp_wts = None
    splitter_kw = {"shuffle": True, "random_state": 0}
    sampling_name = "epa_sol_shuffled-rus_shuffled-strat-kfold"
    hyper_path = "{}models/rehyperparam/{}/".format(project_dir, sampling_name)
    os.makedirs(hyper_path, exist_ok=True)
    model_id = "random_forest"
    model_path = "{}{}/".format(hyper_path, model_id)
    rf_kwargs = {"bootstrap": False, "n_jobs": -1, "random_state": 0}
    base_model = partial(RandomForestClassifier, **rf_kwargs)
    # replica_model = model(n_estimators=100, max_features=5, max_leaf_nodes=200, min_impurity_decrease=5e-5, random_state=0, n_jobs=-1)
    param_ser_list = list()
    forest_params = ParameterGrid(
        {
            "n_estimators": [10, 50, 100, 250],
            "min_impurity_decrease": [0, 0.00005, 0.0001, 0.0005, 0.001],
            "max_features": [3, 5, 6, 7, 9],
            "max_leaf_nodes": [100, 150, 200, 250, None],
        }
    )
    summary_dict, params_dict = dict(), dict()
    for p_id, (param_set) in enumerate(forest_params):
        params_dict[p_id] = param_set
    param_path = "{}param_list.csv".format(hyper_path)
    params_df = pd.DataFrame.from_dict(params_dict, orient="index")
    params_df.to_csv(param_path, index_label="set_index")
    for p_id, (param_set) in enumerate(forest_params):
        print(param_set)
        param_dir = "{}param_{}/".format(model_path, p_id)
        os.makedirs(param_dir, exist_ok=True)
        # final_X, final_y = final_X.iloc[:500], final_y.iloc[:500]
        hyper_dev_df, hyper_eva_df, all_dev_scores, all_eva_scores = (
            combined_rus_cv_results(
                final_X,
                final_y,
                base_model,
                param_set,
                model_name=model_id,
                save_dir=param_dir,
                sweight=samp_wts,
            )
        )
        param_ser_list.append(pd.Series(param_set, name=p_id))
        hyper_dev_df = hyper_dev_df.add_suffix(suffix="_dev", axis="index")
        hyper_eva_df = hyper_eva_df.add_suffix(suffix="_eval", axis="index")
        pprint.pp(hyper_eva_df)
        combo_scores = pd.concat([hyper_dev_df, hyper_eva_df], axis="index")
        combo_scores.sort_index(axis="columns", kind="stable", inplace=True)
        combo_scores.to_csv("{}score_summary.csv".format(param_dir))
        summary_dict[p_id] = hyper_eva_df["Balanced Accuracy"].squeeze()
        print(summary_dict.items())
    crit_df = pd.DataFrame.from_dict(summary_dict, orient="index")
    if "GeoMean_eval" in crit_df.columns:
        crit_df.sort_values(by="GeoMean_eval", ascending=False, inplace=True)
    params_df = pd.concat(param_ser_list)
    crit_path = "{}criteria_scores.csv".format(model_path)
    param_path = "{}param_list.csv".format(hyper_path)
    crit_df.to_csv(crit_path, index_label="set_index")
    params_df.to_csv(param_path, index_label="set_index")


def main():
    train_model()


if __name__ == "__main__":
    main()
