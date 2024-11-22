import pandas as pd
from sklearn.utils import check_X_y as checker


def remove_duplicate_idx(df):
    # Reliably removes duplicated rows in a DataFrame.
    new_dict = dict()
    for k, v in df.items():
        if k in new_dict.keys():
            continue
        else:
            new_dict[k] = v
    if type(df) is pd.DataFrame:
        new_df = pd.DataFrame.from_dict(new_dict)
    elif type(df) is pd.Series:
        new_df = pd.Series(data=new_dict, name=df.name)
    return new_df


def rename_duplicate_features(feature_df):
    # Renames unique-valued, identically-named features in a DataFrame.
    # feature_df = feature_df.T.drop_duplicates().T
    while feature_df.columns[feature_df.columns.duplicated()].size > 0:
        dup_col = feature_df.columns[feature_df.columns.duplicated(keep="first")]
        feature_df.columns = feature_df.columns.where(
            ~feature_df.columns.duplicated(keep="first"),
            feature_df.columns[feature_df.columns.duplicated()] + "i",
        )
    assert feature_df.columns[feature_df.columns.duplicated()].size == 0
    """
    dups = feature_df.columns[feature_df.columns.duplicated(keep=False)].unique()
    for col in dups:
        inds = feature_df.columns
        for i, ind in enumerate(inds):
            feature_df.columns[ind] = '{}_i'.format(feature_df.columns[ind])
    """
    return feature_df


def check_inchi_only(inchi_keys):
    # Check if a list of strings contains any non-INCHI key entries.
    correct_keys, bad_keys = list(), list()
    for k in inchi_keys:
        ksplit = k.split("-")
        if (
            len(ksplit) == 3
            and len(ksplit[0]) == 14
            and len(ksplit[1]) == 10
            and len(ksplit[2]) == 1
        ):
            correct_keys.append(k)
        else:
            bad_keys.append(k)
    return correct_keys, bad_keys


def clean_and_check(feature_df, labels, y_dtype=None, var_thresh=0, verbose=False):
    X_df = feature_df.copy()
    X_df = rename_duplicate_features(X_df)
    valid_idx, invalid_idx = check_inchi_only(X_df.index)
    if len(invalid_idx) > 0:
        raise UserWarning
        print("INCHI keys are not valid: {}".format(invalid_idx))
    y_ser = remove_duplicate_idx(labels)
    if type(y_ser.index) is pd.RangeIndex:
        X_df = X_df.loc[y_ser]
    else:
        X_df = X_df.loc[y_ser.index]
    if y_dtype is not None:
        y_ser = y_ser.astype(y_dtype)
    zero_var_cols = X_df[X_df.var(axis=1) <= var_thresh].columns
    if verbose:
        print(feature_df.shape)
        print(
            "{} features with less than {} variance removed.".format(
                feature_df.shape[1] - zero_var_cols.size, var_thresh
            )
        )
    if zero_var_cols.size == 0:
        X_df.drop(columns=zero_var_cols, inplace=True)
    feature_arr, labels_arr = checker(X_df, y=y_ser)
    X_out = pd.DataFrame(data=feature_arr, index=X_df.index, columns=X_df.columns)
    y_out = pd.Series(data=labels_arr, index=y_ser.index)
    assert not X_out.empty
    assert not y_out.empty
    return X_out, y_out


def data_check():
    from data_tools import load_all_descriptors, load_combo_data, load_training_data

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
