from dataclasses import dataclass

import pandas as pd
from sklearn.base import (BaseEstimator)


# Not ready for use.

@dataclass
class ModelIteration:
    model_type: BaseEstimator
    model_name: str
    model_params: dict
    fitted: bool
    dev_indices: pd.Index
    eval_indices: pd.Index
    scores: dict

    def make_model(self):
        pass
