import itertools
import os

import pandas as pd
import qsar_readiness
from qsar_readiness import QSAR_COLUMNS

import DescriptorRequestor
import padel_categorization


# TODO: Calculate and plot descriptive stats
# TODO: Filter compounds by MW, peptides, etc.
# TODO: Eliminate compounds/features with higher NaN values
# TODO: Feature agglomeration
# TODO: Feature selection


def get_api_descriptors(qsar_df, desc_path, desc_set="padel"):
    # Wrapper function that gets descriptors for many compounds.
    desc_grabber = DescriptorRequestor.DescriptorGrabber(
        desc_set=desc_set, timeout=(30, 600)
    )
    desc_list = list()
    inchi_dict = pd.Series(qsar_df.index.values, index=qsar_df["SMILES_QSAR"]).to_dict()
    with open(desc_path, "a"):
        for desc, smile in desc_grabber.bulk_epa_call(qsar_df["SMILES_QSAR"].tolist()):
            if not isinstance(desc, dict):
                exit()
            else:
                desc.update(("FOREIGN_KEY", inchi_dict[smile]))
                desc_list.append(desc)


def map_cols(df, original, new):
    qsar_map = dict(zip(df[original], df[new]))
    return qsar_map


def extract_cached_desc(qsar_ready_df, cached_df):
    # Find SMILES_QSAR column in both dfs, check for union (recheck for different smiles column), return descriptor df
    upper_cache_cols = [
        x
        for x in cached_df.columns.tolist()
        if ("QSAR" in x.upper() or "SMILE" in x.upper())
    ]
    last_num = 0
    best_match, best_col, qsar_col = None, None, None
    for col, qsar_col in itertools.product(
        upper_cache_cols, ["SMILES_QSAR", "SMILES_CANONICAL"]
    ):
        cache_smiles = pd.Index(cached_df[col].squeeze().tolist())
        qsar_smiles = pd.Index(qsar_ready_df[qsar_col].squeeze().tolist())

        # Find QSAR column, if different from SMILES_QSAR, convert column to SMILES_QSAR

        matches = qsar_smiles.intersection(cache_smiles)
        if len(matches.tolist()) > last_num:
            last_num = len(matches.tolist())
            best_col = (col, qsar_col)
            best_match = matches
    smiles_matches = pd.Index(
        qsar_ready_df["SMILES_QSAR"].squeeze().tolist()
    ).intersection(cached_df.index)
    canon_matches = pd.Index(
        qsar_ready_df["SMILES_CANONICAL"].squeeze().tolist()
    ).intersection(cached_df.index)
    if last_num == 0:
        return pd.DataFrame([]), None
    elif len(smiles_matches) > last_num and len(canon_matches):
        return (
            cached_df.loc[smiles_matches].rename(
                index=map_cols(qsar_ready_df, "SMILES_QSAR", "INCHI_KEY"), inplace=True
            ),
            "SMILES_QSAR",
        )
    elif len(canon_matches) > last_num:
        return (
            cached_df.loc[canon_matches].rename(
                index=map_cols(qsar_ready_df, "SMILES_CANONICAL", "INCHI_KEY"),
                inplace=True,
            ),
            "SMILES_CANONICAL",
        )
    else:
        return (
            cached_df.loc[cached_df[best_col[0]].isin(best_match.tolist())].rename(
                index=map_cols(qsar_ready_df, best_col[1], "FOREIGN_KEY"), inplace=True
            ),
            best_col[1],
        )


def get_api_desc(desc_path, qsar_df, d_set):
    # Hits API to obtain descriptor (d_set).
    grabber, desc_names = None, None
    # TODO: Sketch descriptor name getter.
    desc_names = padel_categorization.padel_names
    last_ind = 0
    print(desc_path)
    inchi_dict = pd.Series(qsar_df.index.values, index=qsar_df["SMILES_QSAR"]).to_dict()
    desc_dict = dict()
    ser_list = list()
    grabber = DescriptorRequestor.DescriptorGrabber(d_set)
    for response, smile in grabber.bulk_epa_call(
        qsar_df["SMILES_QSAR"].tolist()[last_ind:]
    ):
        if type(response) is dict:
            desc_dict[inchi_dict[smile]] = [response["descriptors"]]
            if os.path.isfile(desc_path):
                desc_mode = "a"
            else:
                desc_mode = "w"
                print(desc_mode)
            with open(desc_path, desc_mode) as fo:
                fo.write("{}\n".format(response.values()))
            temp_srs = pd.Series(data=response["descriptors"], name=inchi_dict[smile])
            # temp_srs = pd.Series(data=response['descriptors'], index=desc_names, name=inchi_dict[smile])
            ser_list.append(temp_srs)
        else:
            print(smile, response)
            with open(desc_path, "a") as fo:
                fo.write("{}\n".format(response.items()))
    desc_df = pd.concat(ser_list, axis=1).T.rename(QSAR_COLUMNS, inplace=True)
    return desc_df


def main(
    data_path=None,
    qsar_df=None,
    desc_cache=None,
    qsar_ready_name=None,
    cache_path=None,
    desc_set=None,
    intermetals=False,
    salt_filter=False,
):
    """
    Runs compounds from data_path through EPA internal descriptor calculator.
    :param data_path:
    :param cache_path:
    :param qsar_ready_name:
    :param desc_cache: pd.DataFrame,
    :param qsar_df: pd.DataFrame, Contains QSAR Standardizer output
    """
    if type(desc_cache) is not pd.DataFrame and cache_path:
        # print(desc_cache)
        desc_cache = pd.read_pickle("{}{}.pkl".format(cache_path, qsar_ready_name))
    # if not type(qsar_df) is not pd.DataFrame and qsar_ready_name:
    #    print(qsar_df)
    #    qsar_df = pd.read_pickle('{}{}.pkl'.format(data_path, qsar_ready_name))
    qsar_df.rename(mapper=QSAR_COLUMNS, inplace=True)
    salts, intermetallics = qsar_readiness.salts_and_intermetallics(
        qsar_df["SMILES_QSAR"].tolist(), intermetallics=intermetals
    )
    if salt_filter and len(salts) > 0:
        salt_df = qsar_df[qsar_df["SMILES_QSAR"].isin(salts)]
        salt_df.to_pickle("{}SALTS_{}.pkl".format(data_path, qsar_ready_name))
        qsar_df.drop(index=salt_df.index, inplace=True)
    if intermetallics and len(intermetallics) > 0:
        intermetallics_df = qsar_df[qsar_df["SMILES_QSAR"].isin(intermetallics)]
        intermetallics_df.to_pickle(
            "{}INTERMETALLICS_{}.pkl".format(data_path, qsar_ready_name)
        )
        qsar_df.drop(index=intermetallics_df.index, inplace=True)
    if type(desc_cache) is pd.DataFrame and not desc_cache.empty:
        cached_values, qsar_col_name = extract_cached_desc(qsar_df, desc_cache)
        missing_desc = qsar_df.loc[qsar_df.index.difference(cached_values.index)]
    else:
        missing_desc = qsar_df
    print(missing_desc.shape)
    api_desc = get_api_desc(
        "{}{}/{}_{}.csv".format(data_path, desc_set, desc_set.upper(), qsar_ready_name),
        missing_desc,
        desc_set,
    )
    if desc_cache and type(desc_cache) is pd.DataFrame and desc_cache.empty:
        all_desc = pd.concat(
            [cached_values, api_desc], axis=1, verify_integrity=True, sort=True
        )
    else:
        all_desc = api_desc
    return all_desc
