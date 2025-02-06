from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from functools import cache

import pandas as pd

"""
Tuples = (Standard, Average, Centered, Average-Centered)
bm == Broto-Moreau
bm_c == centered bm
m == mass
vdw == van der Waals
sen == Sanderson electronegativity
pol == polarizability
fip == first ionization potential
ist == I state
bar == average of
vdwv == van der waals volume
ch == charge

bm_ac_m = (range(21, 31), range(76, 85), range(76, 85), range(202, 211))
bm_ac_vdw = (range(31, 40), range(85, 94), range(148, 157), range(211, 220))
bm_ac_sen = (range(40, 49), range(94, 103), range(157, 166), range(220, 229))
bm_ac_pol = (range(49, 58), range(103, 112), range(166, 175), range(229, 238))
bm_ac_fip = (range(58, 67), range(112, 121), range(175, 184), range(238, 247))
bm_ac_ist = (range(67, 76), range(121, 130), range(184, 193), range(247, 256))
bm_c_ch = (range(130, 139), range(193, 202))
"""


@dataclass
class Descriptor:
    index: int
    name: str
    type: str
    description: str
    dimension: int
    ex_class: str


def classify_padel_columns(feature_df: pd.DataFrame):
    classified_dict = dict()
    padel_df = get_full_padel_df()
    for col in feature_df.columns:
        if col in padel_df["Description"].tolist():
            classified_dict[col] = "Description"
        elif col in padel_df["Descriptor name"].tolist():
            classified_dict[col] = "Descriptor name"
        elif col in padel_df["Index"].tolist():
            classified_dict[col] = "Index"
    return classified_dict


def padel_descriptors_to_types(full=True, short=False, type_only=False):
    # Returns dictionary relating descriptor name to its type (ex. Constitutional descriptor).
    desc_val_df = get_full_padel_df()
    if full:
        return desc_val_df["Description", "Type"].to_dict(orient="records")
    elif short:
        return desc_val_df["Descriptor Name", "Type"].to_dict(orient="records")
    elif type_only:
        return desc_val_df["Type"]


def get_feat_names():
    return get_full_padel_names().tolist()


def autogroup_descriptors(desc_names, vectorizer, clusterer):
    # In-progress function thta groups feature names using an ML classifier.
    train = vectorizer.fit_transform(raw_documents=desc_names)
    # print(vectorizer.get_feature_names_out())
    labelled = clusterer.fit_predict(train)
    label_dict = dict(zip(desc_names, labelled))
    nested = list()
    for a in clusterer.labels_:
        nested.append([l[0] for l in label_dict.items() if l[1] == a])
    # logging.info(('{}\n'.format(*nested[0])))
    return nested


def group_padel_descriptors_manual(desc_names):
    # Groups PaDeL descriptors based on keywords present in their names.
    lags = ["lag {}".format(a) for a in range(0, 12)]
    ac_names = ["Broto-Moreau", "Geary", "Moran"]
    weights = [
        "charges",
        "mass",
        "van der Waals volumes",
        "Sanderson electronegativities",
        "polarizabilities",
        "first ionization potential",
        "I-state",
    ]
    lag_pairs = list(itertools.product(ac_names, lags))
    weight_pairs = list(itertools.product(ac_names, weights))
    name_lags = ["{}_{}".format(a, b) for a, b in lag_pairs]
    name_weights = ["{}_{}".format(a, b) for a, b in weight_pairs]
    names = ["Barysz", "Randic"]
    measures = [
        "E-state",
        "walk count",
        "distance edge",
        "information content",
        "acceptor",
        "donor",
    ]
    halogens = ["chlorine", "bromine", "iodine", "Cl", "Br ", "I"]
    # ('halogens', halogens),
    lrings = [
        "Number of {} rings".format(*["{}-membered".format(a) for a in range(7, 13)])
    ]
    frings = [
        "Number of {} fused rings".format(
            *["{}-membered".format(a) for a in range(3, 13)]
        )
    ]
    stop_words = []  # ['average', 'logarithm', 'centered']
    kwarg_dict = dict(
        [
            ("lrings", lrings),
            ("names", names),
            ("frings", frings),
            ("autocorr", name_lags),
            ("lagac", name_weights),
            ("measures", measures),
        ]
    )
    desc_dict = dict()
    # ('names', names),
    # desc_dict['halogen'] = list()
    # desc_dict['halogen'] = [desc_dict['halogen'].extend([desc_dict[a] for a in desc_dict.keys()])]
    temp_list = list()
    for key in kwarg_dict.keys():
        for subst in kwarg_dict[key]:
            desc_dict[subst] = list()
            for name in desc_names:
                if key == "autocorr" or key == "lagac":
                    if all(
                        [
                            all([(a in name), (a not in stop_words)])
                            for a in subst.split()
                        ]
                    ):
                        temp_list.append(name)
                else:
                    if subst in name:
                        temp_list.append(name)
            # print('{}\n'.format(desc_dict.items()))
            if len(temp_list) > 0:
                desc_dict[subst] = sorted(temp_list)
                temp_list = list()
    return desc_dict


@cache
def get_two_dim_only(long_names=False, short_names=False):
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
    # Returns long padel names.
    if two_d:
        long_padel = get_two_dim_only()["Description"]
    else:
        long_padel = get_full_padel_df()["Description"]

    return long_padel


@cache
def get_short_padel_names(two_d=True):
    # Return short PaDeL codes.
    full_padel = get_full_padel_df()
    if two_d:
        full_padel = get_two_dim_only()
    else:
        full_padel = get_full_padel_df()
    short_padel = full_padel["Descriptor name"]
    return short_padel


@cache
def get_full_padel_df():
    # Returns full CSV files of PaDeL descriptors in DataFrame format.
    return pd.read_excel(
        "{}data/padel_all_descriptor_names.xlsx".format(os.environ.get("PROJECT_DIR"))
    )


@cache
def padel_short_to_long():
    padel_df = get_full_padel_df()
    convert_df = padel_df[["Descriptor name", "Description"]]
    convert_df.set_index(keys="Descriptor name", inplace=True)
    return convert_df.squeeze()


@cache
def padel_short_to_long():
    padel_df = get_full_padel_df()
    convert_df = padel_df[["Description", "Descriptor name"]]
    convert_df.set_index(keys="Descriptor name", inplace=True)
    return convert_df.squeeze()
