import os
import pickle
import pandas as pd
from data_handling.data_cleaning import clean_and_check

checker = partial(check_X_y, accept_sparse=False, ensure_min_features=10, ensure_min_samples=1000)
final_dir = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/dmso_zip/final_datasets/"


def load_metadata(desc=False):
    # Returns dictionary {SOURCE_SOLUBILITY: List of Series/DataFrames}
    # Keys: 'epa_sol', 'epa_in', 'en_in', 'en_sol'
    # List values:
    #   -INCHI_KEY: Solubility
    #   -INCHI_KEY: Metadata from descriptors and fingerprints, plus one "DESCRIPTORS" column of lists.
    #   -INCHI_KEY: Descriptors (DataFrame w/1444 columns)
    # Metadata contains INCHI key, QSAR SMILES, INCHI strings, Circular Fingerprints, and PaDeL descriptors.
    with open("{}padel/PADEL_EPA_ENAMINE_5mM_TUPLES.pkl".format(final_dir), 'rb') as f:
        tups = pickle.load(f)
    if type(tups) is not dict or not all([t in tups.keys() for t in ['en_sol', 'epa_sol', 'en_in', 'epa_in']]):
        print("Metadata pkl file does not contain dictionary with all keys.")
        raise ValueError
    elif not desc:
        [t.drop(columns=['DESCRIPTORS'], inplace=True) for t in tups.values() if
         type(t) is pd.DataFrame and 'DESCRIPTORS' in t.columns]
    else:
        if not any([[(df is None or df.empty) for df in val] for val in tups.values()]):
            print([[df for df in val if (df is None or df.empty)] for val in tups.values()])
            raise ValueError
    return tups


def load_training_data(clean=True):
    # Loads data containing combined PaDeL descriptors and soluble/insoluble labels from all sources.
    meta_data = ""
    # from data import constants
    # print(constants.paths.combo_path)
    # with open(constants.paths.combo_path, 'rb') as f:
    # with open(os.environ.get("FINAL_DIR").format(padel/PADEL_EPA_ENAMINE_5mM_TUPLES.pkl", 'rb') as f:
    # meta_data = pickle.load(f)
    # with open(constants.paths.train, 'rb') as f:
    with open('{}MAXMIN_PADEL_TRAIN.pkl'.format(
            "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/dmso_zip/final_datasets/filtered/"),
              'rb') as f:
        train_tup = pickle.load(f)
    tupX, tupy = train_tup
    if clean:
        feature_df, labels = clean_and_check(tupX, tupy)
    else:
        feature_df, labels = tupX, tupy
    return feature_df, labels, meta_data


def get_interpretable_features(feature_df=None, labels=None, clean=True):
    # Removes less intepretable features from PaDeL data.
    from data.constants import names
    if feature_df is None or labels is None:
        feature_df, labels, meta_data = load_training_data(clean=clean)
    # ecounts = data.feature_name_lists.get_estate_counts(feature_df.columns.tolist())
    bad_feats = pd.Index([c for c in feature_df if
                          any([u.lower() in c.lower() for u in names.uninterpretable])])
    feature_df.drop(columns=bad_feats, inplace=True)
    feature_df, labels = clean_and_check(feature_df, labels)
    print("Training data size: {}".format(feature_df.shape))
    return feature_df, labels


def get_all_data(epa_sol=True, interpret=True):
    # Tuples: Solubility Series, Metadata DF, Descriptor DF
    meta_dict = load_metadata(desc=True)
    if epa_sol:
        meta_keys = ['epa_sol', 'epa_in', 'en_in']
    else:
        meta_keys = ['epa_sol', 'epa_in', 'en_in', 'en_sol']
    sol_dict = dict([(k, v[0].squeeze()) for k, v in meta_dict.items() if k in meta_keys])
    if interpret:
        desc_dict = dict(
            [(k, get_interpretable_features(v[2], v[0])[0]) for k, v in meta_dict.items() if k in meta_keys])
    else:
        desc_dict = dict([(k, v[2]) for k, v in meta_dict.items() if k in meta_keys])
    assert all([type(s) is pd.Series for s in sol_dict.values()])
    assert all([type(s) is pd.DataFrame for s in desc_dict.values()])
    return sol_dict, desc_dict


'''
mi_df = pd.read_csv('{}balanced_mi_all_train_cv30.csv'.format(os.environ.get('MODELS_DIR')))
mi_df.set_index(keys=mi_df.columns[0], drop=True, inplace=True)
mi_avg = mi_df.mean(axis=1).sort_values(ascending=False)
'''


def load_feature_rankings(filepath, threshold=None):
    # Loads features ranked by feature importance as determined by "brute-force" RFE with Random Forest estimator.
    ranking_ser = pd.read_csv(filepath).set_index(keys='Features', inplace=False).squeeze()  # , index_col="Features")
    if threshold is not None:
        ranking_ser.drop(ranking_ser[ranking_ser > threshold].index, inplace=True)
    return ranking_ser.index
