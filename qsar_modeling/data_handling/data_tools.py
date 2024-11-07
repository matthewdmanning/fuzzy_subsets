import os
import pickle
from functools import partial
import pandas as pd
from sklearn.utils import check_X_y
from data_handling.data_cleaning import clean_and_check, check_inchi_only
from padel_categorization import get_two_dim_only

checker = partial(
    check_X_y, accept_sparse=False, ensure_min_features=10, ensure_min_samples=1000
)


def load_all_descriptors():
    with open("{}filtered/PADEL_EPA_ENAMINE_5mM.pkl".format(os.environ.get("FINAL_DIR")), "rb") as f:
        feature_df = pickle.load(f)
    return feature_df


def load_idx_selector(source_label_combos):
    # with open("{}filtered/PADEL_EPA_ENAMINE_5mM.pkl".format(os.environ.get("FINAL_DIR")), "rb") as f:
    with open("{}filtered/PADEL_CFP_COMBO_5mM.pkl".format(os.environ.get("FINAL_DIR")), "rb") as f:
        data_df = pickle.load(f)
    print(data_df.columns)
    print(data_df["DATA_SOURCE"].squeeze().unique())
    print(data_df["DMSO_SOLUBILITY"].squeeze().value_counts())
    idx_dict = dict()
    en_idx = data_df[data_df["DATA_SOURCE"] == 'ENAMINE']
    epa_idx = data_df[data_df["DATA_SOURCE"] == 'EPA']
    sol_idx = data_df[data_df["DATA_SOURCE"] == 1]
    insol_idx = data_df[data_df["DATA_SOURCE"] == 0]
    for combo in source_label_combos:
        if "epa" in combo.lower():
            if "insol" in combo.lower():
                idx_dict[combo] = epa_idx.intersection(insol_idx)
            elif "sol" in combo.lower():
                idx_dict[combo] = epa_idx.intersection(sol_idx)
            else:
                idx_dict[combo] = epa_idx
        elif "en" in combo.lower():
            if "insol" in combo.lower():
                idx_dict[combo] = en_idx.intersection(insol_idx)
            elif "sol" in combo.lower():
                idx_dict[combo] = en_idx.intersection(sol_idx)
            else:
                idx_dict[combo] = en_idx
    return idx_dict


def load_metadata(desc=False):
    # Returns dictionary {SOURCE_SOLUBILITY: List of Series/DataFrames}
    # Keys: 'epa_sol', 'epa_in', 'en_in', 'en_sol'
    # List values:
    #   -INCHI_KEY: Solubility
    #   -INCHI_KEY: Metadata from descriptors and fingerprints, plus one "DESCRIPTORS" column of lists.
    #   -INCHI_KEY: Descriptors (DataFrame w/1444 columns)
    # Metadata contains INCHI key, QSAR SMILES, INCHI strings, Circular Fingerprints, and PaDeL descriptors.
    # with open("{}filtered/PADEL_CFP_COMBO_5mM.pkl".format(os.environ.get("FINAL_DIR")), "rb") as f:
    with open("{}padel/PADEL_EPA_ENAMINE_5mM_TUPLES.pkl".format(os.environ.get("FINAL_DIR")), "rb") as f:
        tups = pickle.load(f)
    if type(tups) is not dict or not all(
        [t in tups.keys() for t in ["en_sol", "epa_sol", "en_in", "epa_in"]]
    ):
        print("Metadata pkl file does not contain dictionary with all keys.")
        raise ValueError
    elif not desc:
        [
            t.drop(columns=["DESCRIPTORS"], inplace=True)
            for t in tups.values()
            if type(t) is pd.DataFrame and "DESCRIPTORS" in t.columns
        ]
    else:
        if not any([[(df is None or df.empty) for df in val] for val in tups.values()]):
            print(
                [
                    [df for df in val if (df is None or df.empty)]
                    for val in tups.values()
                ]
            )
            raise ValueError
    return tups


def load_training_idx(clean=True):
    # Loads data containing combined PaDeL descriptors and soluble/insoluble labels from all sources.
    meta_idx = ""
    # from data import constants
    # print(constants.paths.combo_path)
    # with open(constants.paths.combo_path, 'rb') as f:
    # with open(os.environ.get("FINAL_DIR").format(padel/PADEL_EPA_ENAMINE_5mM_TUPLES.pkl", 'rb') as f:
    # meta_idx = pickle.load(f)
    # with open(constants.paths.train, 'rb') as f:
    with open(
        "{}filtered/MAXMIN_PADEL_TRAIN.pkl".format(
            os.environ.get("FINAL_DIR")
        ),
        "rb",
    ) as f:
        train_tup = pickle.load(f)
    tupX, tupy = train_tup
    if clean:
        feature_df, labels = clean_and_check(tupX, tupy)
    else:
        feature_df, labels = tupX, tupy
    return feature_df, labels, meta_idx


def get_interpretable_features(feature_df=None, labels=None, clean=True):
    # Removes less intepretable features from PaDeL data.
    from data.constants import names

    if feature_df is None:
        feature_df, labels, meta_idx = load_training_idx(clean=clean)
    padel_df = get_two_dim_only()
    short_names, long_names = (
        padel_df["Descriptor name"].tolist(),
        padel_df["Description"].tolist(),
    )
    for c in feature_df.columns:
        if c not in short_names and c not in long_names or type(c) is not str:
            print(c)
    bad_feats = pd.Index(
        [
            c
            for c in feature_df.columns
            if any([u.lower() in c.lower() for u in names.uninterpretable])
        ]
    )
    feature_df.drop(columns=bad_feats, inplace=True)
    # feature_df, labels = clean_and_check(feature_df, labels)
    print("Training data size: {}".format(feature_df.shape))
    return feature_df, labels


def get_all_idx(epa_sol=True, interpret=True, meta_desc=False):
    # Tuples: Solubility Series, Metadata DF, Descriptor DF
    meta_dict = load_metadata(desc=True)
    if epa_sol:
        meta_keys = ["epa_sol", "epa_in", "en_in"]
    else:
        meta_keys = ["epa_sol", "epa_in", "en_in", "en_sol"]
    sol_dict, desc_dict = dict(), dict()
    for k, v in meta_dict.items():
        if k in meta_keys:
            valid_inchi, invalid_inchi = check_inchi_only(v[0].squeeze().index)
            if len(invalid_inchi) > 0:
                print(v[0].head())
                print("Invalid INCHI keys in {}: {}".format(k, invalid_inchi[:10]))
                raise ValueError
            sol_dict[k] = v[0].squeeze()[valid_inchi]
            if meta_desc:
                padel_df = get_two_dim_only()
                short_names, long_names = (
                    padel_df["Descriptor name"].tolist(),
                    padel_df["Description"].tolist(),
                )
                desc_df = pd.concat([pd.Series(data=r, index=long_names, name=n) for n, r in v[1]['DESCRIPTORS'].items()], axis=1).T
                print(desc_df.head())
            else:
                desc_df = v[2]
            print("Duplicated...")
            print(desc_df.T.loc[desc_df.T.duplicated(keep=False)].T)
            if interpret:
                desc_dict[k] = get_interpretable_features(
                    feature_df=desc_df, labels=v[0]
                )[0]
            else:
                desc_dict[k] = desc_df
    assert all([type(s) is pd.Series for s in sol_dict.values()])
    assert all([type(s) is pd.DataFrame for s in desc_dict.values()])
    return sol_dict, desc_dict


"""
mi_df = pd.read_csv('{}balanced_mi_all_train_cv30.csv'.format(os.environ.get('MODELS_DIR')))
mi_df.set_index(keys=mi_df.columns[0], drop=True, inplace=True)
mi_avg = mi_df.mean(axis=1).sort_values(ascending=False)
"""


def load_feature_rankings(filepath, threshold=None):
    # Loads features ranked by feature importance as determined by "brute-force" RFE with Random Forest estimator.
    ranking_ser = (
        pd.read_csv(filepath).set_index(keys="Features", inplace=False).squeeze()
    )  # , index_col="Features")
    if threshold is not None:
        ranking_ser.drop(ranking_ser[ranking_ser > threshold].index, inplace=True)
    return ranking_ser.index
