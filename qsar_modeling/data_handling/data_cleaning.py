from functools import partial
import pandas as pd


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
    while feature_df.columns[feature_df.columns.duplicated()].size > 0:
        feature_df.columns = feature_df.columns.where(~feature_df.columns.duplicated(), feature_df.columns + 'i')
    assert feature_df.columns[feature_df.columns.duplicated()].size == 0

    dups = feature_df.columns[feature_df.columns.duplicated(keep=False)].unique()
    for col in dups:
        inds = feature_df.columns
        for i, ind in enumerate(inds):
            feature_df.columns[ind] = '{}_i'.format(feature_df.columns[ind])
    return feature_df


def check_inchi_only(inchi_keys):
    # Check if a list of strings contains any non-INCHI key entries.
    correct_keys, bad_keys = list(), list()
    for k in inchi_keys:
        ksplit = k.split("-")
        if len(ksplit) == 3 and len(ksplit[0]) == 14 and len(ksplit[1]) == 10 and len(ksplit[2]) == 1:
            correct_keys.append(k)
        else:
            bad_keys.append(k)
    return correct_keys, bad_keys


def clean_and_check(feature_df, labels, y_dtype=None):
    tup_X = rename_duplicate_features(feature_df)
    valid_idx, invalid_idx = check_inchi_only(tup_X.index)
    if len(invalid_idx) > 0:
        raise UserWarning
        print("INCHI keys are not valid: {}".format(invalid_idx))
    tup_X = tup_X[tup_X.var(axis=1) > 0]
    tup_y = remove_duplicate_idx(labels)
    tup_X = tup_X.loc[tup_y.index]
    tup_y = tup_y.squeeze()
    if y_dtype is not None:
        tup_y = tup_y.astype(y_dtype)
    feature_arr, labels_arr = checker(tup_X, y=tup_y)
    X_out = pd.DataFrame(data=feature_arr, index=tup_X.index, columns=tup_X.columns)
    y_out = pd.Series(data=labels_arr, index=tup_y.index)
    assert not X_out.empty
    assert not y_out.empty
    return X_out, y_out
