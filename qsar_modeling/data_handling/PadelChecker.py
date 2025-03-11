import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

from data_cleaning import rename_duplicate_features


class PadelCheck(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):

    def __init__(self):
        self.feature_names_in = list()
        self.feature_names_out_ = pd.Index([])
        self.rename_dict = dict()

    def rename_duplicates(self, X):
        Xt = rename_duplicate_features(X.copy())
        self.rename_dict = dict(zip(X.columns, Xt.columns))
        return Xt

    def fit(self, X, y=None, **fit_params):
        self.feature_names_in = X.columns.tolist()
        Xt = self.rename_duplicates(X)
        self.feature_names_out_ = Xt.columns.tolist()
        return self

    def transform(self, X, y=None, **kwargs):
        Xt = X.copy()
        Xt.columns = Xt.columns.map(self.rename_dict)
        return Xt

    def get_feature_names_out(self, input_features=None):
        converted = [
            self.rename_dict[c] for c in input_features if c in self.rename_dict.keys()
        ]
        remain = [c for c in input_features if c not in self.rename_dict.keys()]
        return converted.extend(remain)
