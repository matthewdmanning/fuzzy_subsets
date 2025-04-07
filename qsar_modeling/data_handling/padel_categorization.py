from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache

import pandas as pd


@dataclass
class Descriptor:
    index: int
    name: str
    type: str
    description: str
    dimension: int
    ex_class: str


@cache
def get_two_dim_only(long_names=False, short_names=False):
    raise DeprecationWarning
    # Returns only one and two-dimensional descriptors from PaDeL.
    padel_names = get_full_padel_df()
    two_d = padel_names[padel_names["Dimension"] <= 2]
    if long_names and short_names:
        two_d = two_d[["Description", "Descriptor name"]]
    elif long_names:
        two_d = two_d["Description"]
    elif short_names:
        two_d = two_d["Descriptor name"]
    return two_d


@cache
def get_full_padel_names(two_d=True):
    raise DeprecationWarning
    # Returns long padel names.
    if two_d:
        long_padel = get_two_dim_only()["Description"]
    else:
        long_padel = get_full_padel_df()["Description"]

    return long_padel


@cache
def get_short_padel_names(two_d=True):
    raise DeprecationWarning
    # Return short PaDeL codes.
    full_padel = get_full_padel_df()
    if two_d:
        full_padel = get_two_dim_only()
    else:
        full_padel = get_full_padel_df()
    short_padel = full_padel["Descriptor name"]
    return short_padel


def get_full_padel_df():
    """

    Returns all PaDeL short and long names, type, dimensionality, and extended class.

    """
    #
    return pd.read_excel(
        "{}data/padel_all_descriptor_names.xlsx".format(os.environ.get("PROJECT_DIR")),
        index_col=0,
    )


def get_padel_names(length, dimension=(1, 2), types=None, classes=None):
    full_padel = get_full_padel_df()
    print(full_padel.columns)
    if (
        "long" not in length and "all" not in length
    ) and "Description" in full_padel.columns:
        full_padel.drop(columns="Description", inplace=True)
    elif "short" not in length and "all" not in length:
        full_padel.drop(columns="Descriptor name", inplace=True)
    for d in [1, 2, 3]:
        if d not in dimension:
            full_padel.drop(
                index=full_padel[full_padel["Dimension"] == d].index, inplace=True
            )
    if types is not None:
        full_padel = full_padel[full_padel["Type"].isin(types)]
    if classes is not None:
        full_padel = full_padel[full_padel["Extended class"].isin(classes)]
    assert full_padel is not None
    return full_padel


@cache
def padel_short_to_long():
    padel_df = get_full_padel_df()
    convert_df = padel_df[["Descriptor name", "Description"]]
    convert_df.set_index(keys="Descriptor name", inplace=True)
    return convert_df.squeeze()


@cache
def padel_convert_length(short_to_long=True, three_d=False):
    """
    Returns Series for conversion between short and long PaDeL names.
    Parameters
    ----------
    short_to_long : bool
    Index is short names if True
    three_d: bool
    Whether to also return 3-D descriptors

    Returns
    -------
    convert_df : pd.Series
    """
    padel_df = get_full_padel_df()
    if not three_d:
        padel_df = padel_df[padel_df["Dimension"] < 3]
    convert_df = padel_df[["Description", "Descriptor name"]]
    if short_to_long:
        convert_df.set_index(keys="Descriptor name", inplace=True)
    else:
        convert_df.set_index(keys="Description", inplace=True)
    return convert_df.squeeze()
