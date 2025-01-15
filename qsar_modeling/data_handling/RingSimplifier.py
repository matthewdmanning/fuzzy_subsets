from functools import cache

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

import padel_categorization


@cache
def get_padel_rings(length="long"):
    padel_df = padel_categorization.get_full_padel_df()
    padel_rings = padel_df[padel_df["Type"] == "RingCountDescriptor"]
    if length == "short":
        return padel_rings["Descriptor name"].tolist()
    else:
        return padel_rings["Description"].tolist()


class RingSimplifer(TransformerMixin):

    def __init__(self, large_start=8):
        self.feature_names_out_ = list()
        self.feature_names_in = list()
        self.large_start = large_start
        self.plain_number_dict = dict()
        self.het_number_dict = dict()
        self.large_name = "Number of >{}-membered rings".format(large_start)
        self.large_het_name = "Number of >{}-membered rings containing heteroatoms (N, O, P, S, or halogens)".format(large_start)
        self.all_names_dict = dict()
        self.feature_transform_dict = dict()
        self._greater_name = "Number of >12-membered rings"
        self._het_greater_name = (
            "Number of >12-membered rings containing heteroatoms (N, O, P, S, or halogens)"
        )
        self._ring_descriptors = None

    def fit(self, X, y=None, **kwargs):
        self.feature_names_in = X.columns.tolist()
        self._ring_descriptors = get_padel_rings()
        self.all_names_dict, self.plain_number_dict, self.het_number_dict = (
            self._get_features(X)
        )
        [[self.feature_names_out_.extend(c) for c in cl] for cl in [self.plain_number_dict.values(), self.het_number_dict.values()]]
        return self

    def _get_features(self, X):
        all_names_dict, plain_number_dict, het_number_dict = dict(), dict(), dict()
        if self._ring_descriptors is None:
            self._ring_descriptors = get_padel_rings()
        for k in np.arange(3, 13):
            all_names_dict[k] = [n for n in self._ring_descriptors if str(k) in n and ">" not in n and n in X.columns]
            plain_number_dict[k] = [
                r
                for r in all_names_dict[k]
                if all([s not in r.lower() for s in ["includes", "hetero", "fused"]])
            ]
            het_number_dict[k] = [
                r
                for r in all_names_dict[k]
                if "includes" not in r.lower() and "hetero" in r.lower() and "fused" not in r.lower()
            ]
        plain_number_dict[13] = [self._greater_name]
        het_number_dict[13] = [self._het_greater_name]
        all_names_dict[13] = [self._greater_name, self._het_greater_name]
        return all_names_dict, plain_number_dict, het_number_dict

    # Fit: Define X for Sum(X) -> Xt
    # Transform: Sum(X) -> Xt ==> Requires X be present
    # If < large_start -> Ensure required features. Delete extra features.
    # If >= large_start -> Add to combined feature. Delete all features.
    def transform(self, X, y=None, **kwargs):
        Xt = X.copy()
        all_names_dict, plain_number_dict, het_number_dict = self._get_features(X)
        for k in np.arange(3, self.large_start):
            if any([c not in Xt.columns for c in plain_number_dict[k]]):
                raise KeyError("Missing feature name: {}".format(plain_number_dict[k]))
            elif any([c not in Xt.columns for c in het_number_dict[k]]):
                raise KeyError("Missing feature name: {}".format(het_number_dict[k]))
            else:
                [Xt.drop(columns=c, inplace=True) for c in all_names_dict[k] if c not in plain_number_dict[k] and c not in het_number_dict[k]]
        sum_df = pd.DataFrame(np.zeros(shape=(Xt.shape[0], 2)), columns=[self.large_name, self.large_het_name], index=Xt.index)
        for k in np.arange(self.large_start, 13):
            combo_cols = list(self.plain_number_dict[k])
            combo_cols.extend(list(self.het_number_dict[k]))
            if combo_cols is not None and len(combo_cols) > 0:
                sum_df = sum_df.copy().add(Xt[combo_cols].to_numpy())
            else:
                print("Ring size {} has no members".format(k), flush=True)
                # raise RuntimeWarning
            Xt.drop(columns=all_names_dict[k], inplace=True)
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
        assert not Xt.isna().any().any()
        self.feature_names_out_ = Xt.columns.tolist()
        return Xt.astype(dtype=np.float32)

    def get_feature_names_out(self, input_features=None, *args, **params):
        if input_features is not None:
            converted_feats = [self.feature_transform_dict[c] for c in input_features if c in self.feature_transform_dict.keys()]
            return [c for c in input_features if c not in self.feature_transform_dict.keys()].extend(converted_feats)
        else:
            return self.feature_names_out_

    @staticmethod
    def _sum_features(X, r_name, ring_name):
        X[r_name] = X[r_name].add(X[ring_name])
        X.drop(columns=ring_name, inplace=True)
        return X
