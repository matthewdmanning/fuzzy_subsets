import os

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split

import feature_selection_script
from data import feature_name_lists
from data_handling.balancing import data_by_groups
from data_handling.data_cleaning import check_inchi_only, clean_and_check
from data_handling.data_tools import (
    get_interpretable_features,
    load_all_descriptors,
    load_combo_data,
    load_idx_selector,
    load_metadata,
    load_training_data,
)
from notebooks.dmso_hyperparameter_part_two import cv_model_documented


def get_training_data(epa_sol=True, interpret=True):
    # Select EPA soluble/combined insoluble dataset.
    interp_X, interp_y = get_interpretable_features()
    valid_inchi, invalid_inchi = check_inchi_only(interp_y.index)
    if len(invalid_inchi) > 0:
        print("\n\nInvalid INCHI keys:")
        print([i for i in invalid_inchi])
        interp_X.drop(index=invalid_inchi, inplace=True)
        interp_y.drop(invalid_inchi, inplace=True)
    assert interp_X is not None and not interp_X.empty
    assert interp_y is not None and not interp_y.empty
    meta = load_metadata()
    g_dict = data_by_groups(interp_y, meta)
    # print(g_dict.items())
    if epa_sol:
        meta_keys = ["epa_sol", "epa_in", "en_in"]
    else:
        meta_keys = ["epa_sol", "epa_in", "en_in", "en_sol"]
    full_groups_dict = dict(
        [
            (k, pd.Series(data=k, index=v[2].index, name=k))
            for k, v in meta.items()
            if k in meta_keys
        ]
    )
    meta_y = pd.concat(full_groups_dict.values())
    print(interp_y.index.difference(meta_y.index))
    train_y = interp_y[interp_y.index.intersection(meta_y.index)]
    print(train_y.shape)
    train_X = interp_X.loc[train_y.index.intersection(interp_X.index)]
    print(train_X.shape)
    train_y = interp_y[train_X.index]
    # meta['en_in'][0].drop(index=meta['en_in'][0].index.intersection(meta['epa_in'][0].index), inplace=True)
    # full_groups_list = dict([(k, v[0]) for k, v in meta.items() if k in ['epa_sol', 'epa_in', 'en_in']])
    unique_X, unique_y = clean_and_check(train_X, train_y, y_dtype=int)

    return unique_X, unique_y


def grab_test_data():
    group_dict = load_idx_selector()
    all_X = load_all_descriptors()
    all_y = load_metadata()[0]
    print(all_y.head())
    # all_y = pd.concat(all_y_dict.values())
    # all_X = pd.concat(all_X_dict.values())
    print("Shapes of all descriptor DFs")
    print(all_X.shape)
    train_X, train_y = load_training_data()
    test_idx = all_X.index.difference(train_X.index)
    print("Test data sizes:")
    print(test_idx.size)
    print("By group test:train sizes")
    for k, idx in group_dict.items():
        print(k, all_y[test_idx].shape, train_y[idx].shape)


def get_grouped_data():
    combo_data = load_combo_data()
    all_idx_dict = dict([(k, df.index) for k, df in combo_data.items()])
    return all_idx_dict


def random_grouped_sample(combo_data, split_ratio=0.2, random_state=0):
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


def data_check():
    all_X = load_all_descriptors()
    print("Shapes of all descriptor DFs")
    print(all_X.shape)
    train_X, train_y = load_training_data()
    combo_data = load_combo_data()
    all_idx_dict = dict([(k, df.index) for k, df in combo_data.items()])
    # Verification
    print("Shape from combo data dictionary: ")
    print([(k, idx.size) for k, idx in all_idx_dict.items()])
    all_y_dict = dict([(k, df["DMSO_SOLUBILITY"]) for k, df in combo_data.items()])
    all_y = pd.concat(all_y_dict.values())
    # all_y = pd.concat(all_y_dict.values())
    # all_X = pd.concat(all_X_dict.values())
    test_idx = all_X.index.difference(train_X.index)
    test_y = all_y[test_idx]
    test_X = all_X.loc[test_idx]
    print("Test data sizes:")
    print(test_idx.size)
    for k, idx in all_idx_dict.items():
        print(k, test_idx.intersection(idx).size, train_y.index.intersection(idx).size)


def get_new_test_train(random_state=0, subgroups="all"):
    combo_data = load_combo_data(subgroups)
    train_meta_dict, test_meta_dict = random_grouped_sample(
        combo_data, random_state=random_state
    )
    train_idx = pd.concat([pd.Series(df) for df in train_meta_dict.values()])
    test_idx = pd.concat([pd.Series(df) for df in test_meta_dict.values()])
    all_padel_df = load_all_descriptors()
    train_X = all_padel_df.loc[train_idx]
    test_X = all_padel_df.loc[test_idx]
    all_labels = pd.concat(
        [df["DMSO_SOLUBILITY"].squeeze() for df in combo_data.values()]
    )
    train_y = all_labels[train_idx]
    test_y = all_labels[test_idx]
    return (train_X, train_y), (test_X, test_y)


def quick_brf_train():
    train_data, test_data = get_new_test_train()
    feature_df, labels = clean_and_check(train_data[0], train_data[1], y_dtype=int)
    quick_model_dir = "{}quick_resplit/run_1/".format(os.environ.get("MODEL_DIR"))
    if not os.path.isdir(quick_model_dir):
        os.makedirs(quick_model_dir)
    brf_model = BalancedRandomForestClassifier(random_state=0, n_jobs=-1)
    estates = feature_name_lists.get_estate_counts(feature_df.columns)
    dev_scores, eval_scores = cv_model_documented(
        feature_df[estates],
        labels,
        brf_model,
        model_name="default_brf",
        save_dir=quick_model_dir,
    )
    print(dev_scores)
    print(eval_scores)


def save_train_test_data():
    train_data, test_data = get_new_test_train()
    feature_df, labels = get_interpretable_features(
        train_data[0], train_data[1], clean=False
    )
    feats = feature_df.columns
    clean_train_df, clean_train_y = clean_and_check(
        train_data[0][feats], train_data[1], y_dtype=int
    )
    print(clean_train_df.shape)
    clean_train_df.to_csv(
        "{}filtered/padel_random_split_train.csv".format(os.environ.get("FINAL_DIR")),
        index=True,
        index_label="INCHI_KEY",
    )
    clean_train_y.to_csv(
        "{}filtered/solubility_random_split_train.csv".format(
            os.environ.get("FINAL_DIR")
        ),
        index=True,
        index_label="INCHI_KEY",
    )
    clean_test_df, clean_test_y = clean_and_check(
        test_data[0][feats], test_data[1], y_dtype=int
    )
    print(clean_test_df.shape)
    clean_test_df.to_csv(
        "{}filtered/padel_random_split_test.csv".format(os.environ.get("FINAL_DIR")),
        index=True,
        index_label="INCHI_KEY",
    )
    clean_test_y.to_csv(
        "{}filtered/solubility_random_split_test.csv".format(
            os.environ.get("FINAL_DIR")
        ),
        index=True,
        index_label="INCHI_KEY",
    )


def main():
    feature_df, labels = load_training_data()
    # clean_df, clean_y = clean_and_check(train_data[0], train_data[1], y_dtype=int)
    feature_selection_script.main(
        (feature_df, labels),
        model_name="brf",
        run_name="brf_epa_sol",
        project_dir=os.environ.get("PROJECT_DIR"),
    )


if __name__ == "__main__":
    main()
