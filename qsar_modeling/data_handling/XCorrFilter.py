import pandas as pd
from sklearn.base import TransformerMixin

from correlation_filter import find_correlation


class XCorrFilter(TransformerMixin):

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
        xcorr_dropped = self.xcorr_filter(X)
        self.feature_names_out_ = X.drop(columns=xcorr_dropped).columns.tolist()
        return self

    def transform(self, X, y=None, **kwargs):
        Xt = X.drop(columns=self.dropped_features_.keys())
        return Xt.to_numpy()

    def xcorr_filter(self, X):
        if self.max_features is None:
            self.max_features = X.shape[1]
        xcorr_deleted = find_correlation(
            self.xcorr_,
            cutoff=self.thresh_xc,
            n_drop=max(1, X.shape[1] - self.max_features),
        )
        self.target_corr_.drop(xcorr_deleted.index, inplace=True)
        self.xcorr_ = self.xcorr_.drop(columns=xcorr_deleted.index).drop(index=xcorr_deleted.index)
        self.dropped_features_.update([(c, "Cross-correlation") for c in xcorr_deleted.index])
        # na_corrs = best_corrs.index[best_corrs.isna()]
        # [dropped_dict.update([(c, "NA Correlation")]) for c in na_corrs]
        # best_corrs.drop(na_corrs, inplace=True)
        # cross_corr = cross_corr.drop(index=na_corrs).drop(columns=na_corrs)
        # train_df.drop(columns=na_corrs, inplace=True)
        return xcorr_deleted

    def get_corrs(self, X, y):
        if self.method_corr is not None:
            self.target_corr_ = X.corrwith(
                y, method=self.method_corr
            ).sort_values(ascending=False)
        else:
            self.target_corr_ = X.corrwith(y).sort_values(ascending=False)
        if self.method_xc is not None:
            self.xcorr_ = X.corr(method=self.method_xc)
        else:
            self.xcorr_ = X.corr()

    def get_feature_names_out(self, *args, **params):
        return self.feature_names_out_
