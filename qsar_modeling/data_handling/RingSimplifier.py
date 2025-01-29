from functools import cache

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

import padel_categorization


@cache
def get_padel_rings(length="long"):
    padel_df = padel_categorization.get_full_padel_df()
    padel_rings = padel_df[padel_df["Type"] == "RingCountDescriptor"]
    if length == "short":
        return padel_rings["Descriptor name"].tolist()
    else:
        return padel_rings["Description"].tolist()


class RingSimplifer(BaseEstimator, TransformerMixin):

    def __init__(self, large_start=8, short=False):
        self.short = short
        self.feature_names_out_ = list()
        self.feature_names_in = list()
        self.feature_transform_dict = dict()
        self.plain_number_dict = dict()
        self.het_number_dict = dict()
        self.all_names_dict = dict()
        self.large_start = large_start
        self.large_name = None
        self.large_het_name = None
        self._greater_name = None
        self._het_greater_name = None
        self._ring_descriptors = None

    def set_names(self):
        if self.short:
            self.large_name = "nG{}Ring".format(self.large_start)
            self.large_het_name = "nG{}HeteroRing".format(self.large_start)
            self._greater_name = "nG12Ring"
            self._het_greater_name = "nG12HeteroRing"
        elif not self.short:
            self.large_name = "Number of >{}-membered rings".format(self.large_start)
            self.large_het_name = "Number of >{}-membered rings containing heteroatoms (N, O, P, S, or halogens)".format(
                self.large_start
            )
            self._greater_name = "Number of >12-membered rings"
            self._het_greater_name = "Number of >12-membered rings containing heteroatoms (N, O, P, S, or halogens)"

    def fit(self, X, y=None, **kwargs):
        self.feature_names_in = X.columns.tolist()
        self._get_features(X)
        for cl in [self.plain_number_dict.values(), self.het_number_dict.values()]:
            [self.feature_names_out_.extend(c) for c in cl]
        print(self.feature_names_out_)
        return self

    def _get_features(self, X):
        if self.short:
            self._ring_descriptors = [str(s) for s in get_padel_rings(length="short")]
            self._get_short_dicts(X)
        else:
            self._ring_descriptors = [str(s) for s in get_padel_rings()]
            self._get_long_dicts(X)

    def _get_short_dicts(self, X):
        for k in np.arange(3, 13):
            self.all_names_dict[k] = [
                n
                for n in self._ring_descriptors
                if (
                    n.startswith("n{}".format(k))
                    and n.endswith("Ring")
                    and n in X.columns
                )
            ]
            self.plain_number_dict[k] = ["n{}Ring".format(k)]
            self.het_number_dict[k] = ["n{}HeteroRing".format(k)]
        self.plain_number_dict[13] = [self._greater_name]
        self.het_number_dict[13] = [self._het_greater_name]
        self.all_names_dict[13] = [self._greater_name, self._het_greater_name]

    def _get_long_dicts(self, X):
        for k in np.arange(3, 13):
            self.all_names_dict[k] = [
                n
                for n in self._ring_descriptors
                if str(k) in n and ">" not in n and n in X.columns
            ]
            self.plain_number_dict[k] = [
                r
                for r in self.all_names_dict[k]
                if all([s not in r.lower() for s in ["includes", "hetero", "fused"]])
            ]
            self.het_number_dict[k] = [
                r
                for r in self.all_names_dict[k]
                if "includes" not in r.lower()
                and "hetero" in r.lower()
                and "fused" not in r.lower()
            ]
        self.plain_number_dict[13] = [self._greater_name]
        self.het_number_dict[13] = [self._het_greater_name]
        self.all_names_dict[13] = [self._greater_name, self._het_greater_name]

    # Fit: Define X for Sum(X) -> Xt
    # Transform: Sum(X) -> Xt ==> Requires X be present
    # If < large_start -> Ensure required features. Delete extra features.
    # If >= large_start -> Add to combined feature. Delete all features.
    def transform(self, X, y=None, **kwargs):
        Xt = X.copy()
        self._get_features(X)
        for k in np.arange(3, self.large_start):
            if any([c not in Xt.columns for c in self.plain_number_dict[k]]):
                raise KeyError(
                    "Missing feature name: {}".format(self.plain_number_dict[k])
                )
            elif any([c not in Xt.columns for c in self.het_number_dict[k]]):
                raise KeyError(
                    "Missing feature name: {}".format(self.het_number_dict[k])
                )
            else:
                [
                    Xt.drop(columns=c, inplace=True)
                    for c in self.all_names_dict[k]
                    if c not in self.plain_number_dict[k]
                    and c not in self.het_number_dict[k]
                ]
        sum_df = pd.DataFrame(
            np.zeros(shape=(Xt.shape[0], 2)),
            columns=[self.large_name, self.large_het_name],
            index=Xt.index,
        )
        for k in np.arange(self.large_start, 13):
            combo_cols = list(self.plain_number_dict[k])
            combo_cols.extend(list(self.het_number_dict[k]))
            if combo_cols is not None and len(combo_cols) > 0:
                sum_df = sum_df.copy().add(Xt[combo_cols].to_numpy())
            else:
                print("Ring size {} has no members".format(k), flush=True)
                print(combo_cols)
                # raise RuntimeWarning
            Xt.drop(columns=self.all_names_dict[k], inplace=True)
        if self._greater_name in Xt.columns and self._het_greater_name in Xt.columns:
            sum_df = sum_df.copy().add(Xt[[self._greater_name, self._het_greater_name]])
        Xt = pd.concat([Xt.copy(), sum_df], axis=1)
        """   
        if len(self.ring_number_dict[k]) == 0:
            print("Error with {}-sized rings".format(k))
        elif k > 3 and len(self.ring_number_dict[k]) > 0:
            if (len(self.ring_number_dict[k]) >= 3 and
                X[X[self.ring_number_dict[k][0]] > 0].shape[0] < 0.05 * X[X[self.ring_number_dict[k][2]] > 0].shape[0]
                or X[X[self.ring_number_dict[k][1]] > 0].shape[0] < 0.05 * X[X[self.ring_number_dict[k][2]] > 0].shape[0]
            ):
                X.drop(columns=self.ring_number_dict[k][0:2], inplace=True)
            if (len(self.ring_number_dict[k]) >= 6 and
                X[X[self.ring_number_dict[k][3]] > 0].shape[0] < 0.05 * X[X[self.ring_number_dict[k][5]] > 0].shape[0]
                or X[X[self.ring_number_dict[k][4]] > 0].shape[0] < 0.05 * X[X[self.ring_number_dict[k][5]] > 0].shape[0]
            ):
                X.drop(columns=self.ring_number_dict[k][3:5], inplace=True)
        elif X[X[self.ring_number_dict[k][0]] > 0].shape[0] < 0.05 * X[X[self.ring_number_dict[k][1]] > 0].shape[0]:
            X.drop(columns=self.ring_number_dict[k][0], inplace=True)
            """
        # print("Rings shape: {}".format(Xt.shape))
        # print(sum_df[[self.large_name, self.large_het_name]].head())
        Xt.dropna(axis=0, inplace=True)
        if Xt.isna().any().any():
            print("NA values found in ring descriptors after transformation.")
            na_columns = Xt.columns[Xt.columns.isna()]
            print(na_columns)
            print(Xt.dropna(), flush=True)
            print(Xt.dropna(axis=1))
            raise ValueError
        self.feature_names_out_ = Xt.columns.astype(str).tolist()
        return Xt.astype(dtype=np.float32)

    def get_feature_names_out(self, input_features=None, *args, **params):
        if input_features is not None:
            converted_feats = [
                self.feature_transform_dict[c]
                for c in input_features
                if c in self.feature_transform_dict.keys()
            ]
            return [
                c for c in input_features if c not in self.feature_transform_dict.keys()
            ].extend(converted_feats)
        else:
            return self.feature_names_out_

    @staticmethod
    def _sum_features(X, r_name, ring_name):
        X[r_name] = X[r_name].add(X[ring_name])
        X.drop(columns=ring_name, inplace=True)
        return X
