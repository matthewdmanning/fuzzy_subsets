import numpy as np
import pandas as pd
from data_handling.data_tools import load_training_data, load_metadata, get_interpretable_features, data_by_groups, \
    get_all_data
from data_handling.data_cleaning import check_inchi_only, clean_and_check


def get_training_data(epa_sol=True, interpret=True):
    # Select EPA soluble/combined insoluble dataset.
    interp_X, interp_y = get_interpretable_features()
    valid_inchi, invalid_inchi = check_inchi_only(interp_y.index)
    if len(invalid_inchi) > 0:
        print('\n\nInvalid INCHI keys:')
        print([i for i in invalid_inchi])
        interp_X.drop(index=invalid_inchi, inplace=True)
        interp_y.drop(invalid_inchi, inplace=True)
    assert interp_X is not None and not interp_X.empty
    assert interp_y is not None and not interp_y.empty
    meta = load_metadata()
    g_dict = data_by_groups(interp_y, meta)
    # print(g_dict.items())
    if epa_sol:
        meta_keys = ['epa_sol', 'epa_in', 'en_in']
    else:
        meta_keys = ['epa_sol', 'epa_in', 'en_in', 'en_sol']
    full_groups_dict = dict(
        [(k, pd.Series(data=k, index=v[2].index, name=k)) for k, v in meta.items() if k in meta_keys])
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


# def combined_data():


def main():
    train_X, train_y = get_training_data()
    all_y, all_X = get_all_data()
    print("Shapes of all descriptor DFs")
    print([df.shape for df in all_X.values()])


if __name__ == "__main__":
    main()
