import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import correlation_filter
from correlation_filter import cross_corr_filter


class XCorrFilter(BaseEstimator, TransformerMixin):

    def __init__(self, max_features, thresh_xc, method_corr=None, method_xc=None):
        self.target_corr_ = pd.Series([])
        self.xcorr_ = pd.DataFrame([])
        self.feature_names_in = None
        self.dropped_features_ = dict()
        self.feature_names_out_ = None
        self.max_features = max_features
        self.thresh_xc = thresh_xc
        self.method_corr = method_corr
        self.method_xc = method_xc

    def fit(self, X, y, **fit_params):
        self.feature_names_in = X.columns.tolist()
        self.get_corrs(X, y)
        xcorr_dropped = self.xcorr_filter(X).index
        self.feature_names_out_ = X.drop(columns=xcorr_dropped).columns.tolist()
        return self

    def transform(self, X, y=None, **kwargs):
        Xt = X.drop(
            columns=[
                c for c in pd.Index(self.dropped_features_.keys()) if c in X.columns
            ]
        )
        return Xt

    def xcorr_filter(self, X):
        if self.max_features is None:
            self.max_features = X.shape[1]
        if self.xcorr_.empty:
            self.xcorr_ = correlation_filter.calculate_correlation(
                X, method=self.method_xc
            )
        xcorr_deleted = cross_corr_filter(
            self.xcorr_,
            cutoff=self.thresh_xc,
            n_drop=max(1, X.shape[1] - self.max_features),
        )
        self.xcorr_ = self.xcorr_.drop(columns=xcorr_deleted.index).drop(
            index=xcorr_deleted.index
        )
        self.dropped_features_.update(
            [(c, "Cross-correlation") for c in xcorr_deleted.index]
        )
        return xcorr_deleted

    def get_corrs(self, X, y=None):
        X = self.convert_inputs(X, y)
        self.xcorr_ = correlation_filter.calculate_correlation(
            X, X, method=self.method_xc
        )

    def get_feature_names_out(self, *args, **params):
        return self.feature_names_out_

    def convert_inputs(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            Xt = pd.DataFrame(X)
        else:
            Xt = X
        return Xt
