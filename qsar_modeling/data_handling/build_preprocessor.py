import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from sklearn.utils.estimator_checks import check_transformer_general

import padel_categorization
import PadelChecker
from RingSimplifier import RingSimplifer
from XCorrFilter import XCorrFilter


class Preprocessor:

    def __init__(self):
        self.transformers = list()

    def add_transformer(self, name_transformer):
        check_transformer_general(*name_transformer)
        self.transformers.append((name_transformer))

    def add_center_scaler(self, scaler="standard", **scaler_kwargs):
        if scaler is not None and scaler == "standard":
            scaler = StandardScaler(**scaler_kwargs).set_output(transform="pandas")
        elif scaler is not None and scaler == "robust":
            if len(scaler_kwargs.keys()) == 0:
                scaler_kwargs = {"unit_variance": True}
            scaler = RobustScaler(**scaler_kwargs).set_output(transform="pandas")
        self.scaler = scaler
        self.transformers.append(scaler)

    def add_func_transform(self, func, inv_func=None):
        if func is not None and func == "asinh":
            func = np.arcsinh  # get_transform_func(np.arcsinh)
            inv_transform = None
        transform_func = FunctionTransformer(
            func=func,
            inverse_func=inv_transform,
            feature_names_out="one-to-one",
        ).set_output(transform="pandas")
        self.transformers.append(transform_func)

    def add_feature_correlation_filter(
        self, corr_method="pearson", threshold=0.95, max_features_out=None
    ):
        pair_corr_filter = XCorrFilter(
            max_features=max_features_out,
            thresh_xc=threshold,
            method_xc=corr_method,
        )
        xc_df = pair_corr_filter.xcorr_
        self.add_transformer(("pair_corr", pair_corr_filter))


def padel_rings(feature_df, large_start=8):
    padel_names = padel_categorization.get_padel_names(length="short")
    if len([f for f in feature_df.columns if f in padel_names]) > 0:
        use_short_names = True
    else:
        use_short_names = False
    return (
        "rings",
        RingSimplifer(short=use_short_names, large_start=large_start).set_output(
            transform="pandas"
        ),
    )


def get_standard_preprocessor(
    feature_df,
    scaler=None,
    transform_func=None,
    corr_params=None,
    var_thresh=0.0,
    use_short_names=True,
):

    # Define individual transformers.
    ring_tranform = padel_rings(feature_df)
    padel_transform = ("padel", PadelChecker.PadelCheck())
    var_thresh = ("var", VarianceThreshold(threshold=var_thresh))
    pipe_list = [padel_transform, var_thresh]
    # pipe_list = [padel_transform, var_thresh]
    if transform_func is not None:
        transformer = FunctionTransformer(
            func=transform_func, feature_names_out="one-to-one"
        )
        pipe_list.append(("smooth", transformer))
    if scaler is not None:
        pipe_list.append(("scaler", scaler))
    # combo_transform = Pipeline(steps=[("rings", ring_tranform), ("padel", padel_transform), ("var", var_thresh), ("scale", transform_func), ("xcorr", xcorr_filter)]).set_output(transform="pandas")
    combo_transform = Pipeline(steps=pipe_list).set_output(transform="pandas")
    return combo_transform, scaler
