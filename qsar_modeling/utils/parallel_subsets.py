import os

from joblib import delayed, Parallel
from sklearn.pipeline import clone as clone_model


def train_model_subsets(
    feature_df, predictor_list, model, mem_dir, sample_weights=None
):
    fit_models = Parallel(
        n_jobs=int(os.cpu_count()), prefer="processes", temp_folder=mem_dir
    )(
        delayed(clone_model(model).fit)(
            X=feature_df[predictors[0]],
            y=feature_df[predictors[1]],
            sample_weight=sample_weights,
        )
        for predictors in predictor_list
    )
    return fit_models
