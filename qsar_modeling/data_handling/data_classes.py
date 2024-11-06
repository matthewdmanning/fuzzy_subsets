from dataclasses import dataclass, field
from typing import Callable, Any
import pandas as pd
import sklearn.model_selection

import cv_tools
import scoring


# Dataclasses for use when running cross-validation and hyperparameterizations.

@dataclass
class ParamedModel:
    name: str
    model_type: Callable
    model_params: dict
    inst_model: Any = field(init=False)

    def __post_init__(self):
        self.inst_model = self.model_type(**self.model_params)


@dataclass
class FittedModel:
    paramed_model: ParamedModel
    train_data: pd.DataFrame
    train_labels: pd.Series
    fit_kwargs: dict
    fit_model: Any = field(init=False)

    def __post_init__(self):
        self.fit_model = sklearn.clone(self.paramed_model.inst_model)(self.train_data, self.train_labels,
                                                                      **self.fit_kwargs)


@dataclass
class ModelPredictions:
    model: FittedModel
    input_data: pd.DataFrame
    predict_kwargs: dict
    predicted: pd.Series | pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.predicted = self.model.fit_model.predict(self.input_data, self.predict_kwargs)


@dataclass
class ModelScoring:
    predictions: ModelPredictions
    truth: pd.Series | pd.DataFrame
    scorer_dict: dict[str, Callable] = field(init=False)
    score_dict: dict[str, Callable] = field(init=False)

    def score_predictions(self, scorers=None):
        if self.scorer_dict is None:
            self.scorer_dict = scoring.get_pred_score_funcs()
        else:
            self.scorer_dict = self.scorer_dict
        self.score_dict = dict([(n, s(self.truth, self.predictions.predicted)) for n, s in self.scorer_dict.items()])


@dataclass
class CrossValSplits:
    name: str
    n_splits: int
    splitter: Callable
    splitter_kwargs: dict
    train_df: pd.DataFrame
    train_labels: pd.Series | pd.DataFrame
    dev_eval_list: tuple[tuple[pd.Index, pd.Index]] = field(init=False)

    def __post_init__(self):
        self.dev_eval_list = cv_tools.get_split_ind(self.train_df, self.train_labels, self.n_splits,
                                                    splitter=self.splitter, **self.splitter_kwargs)

    def get_dev_eval_data(self):
        return cv_tools.split_df(self.train_df, self.train_labels, indices_list=self.dev_eval_list)


@dataclass
class Sampling:
    name: str
    sampler: Callable
    sampler_kwargs: dict
    data_df: pd.DataFrame
    labels: pd.Series | pd.DataFrame
    sampling_kwargs: dict
    sample_idx: pd.Index = field(init=False)

    def __post_init__(self):
        sdf, slabels = self.sampler(**self.sampler_kwargs).fit_resample(self.data_df, self.labels,
                                                                        **self.sampling_kwargs)
        self.sample_idx = slabels.index

    def get_X_y(self):
        return self.data_df.loc[self.sample_idx], self.labels[self.sample_idx]


@dataclass
class ValidationLoop:
    name: str


@dataclass
class HyperSearch:
    pgrid: sklearn.model_selection.ParameterGrid
    search_method: Any
    base_models: list[ParamedModel] = field(init=False)
    cross_val: CrossValSplits = field(init=False)
    fitted_models: list[FittedModel] = field(init=False)
    predictions: list[ModelPredictions] = field(init=False)
    scores: list[ModelScoring] = field(init=False)
