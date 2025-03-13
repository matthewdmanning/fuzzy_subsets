import sys

import pandas as pd
from sklearn.conftest import set_config

sys.path.append("C:/Users/mmanning/PycharmProjects/CCMI/")
set_config(**{"transform_output": pd.DataFrame, "enable_cython_pairwise_dist": 1})


class MutualInformationStation:

    def __init__(self, df):
        self.whole_data = df
        self.current_data = None
        self.distArray = None
        # encoded_target = LabelEncoder().fit_transform(y=namedata[y[0]])

    def joint_mi(self, X, Z, k=5):
        pass
