from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from sklearn._config import set_config as sk_config


def set_options():
    # Sets options for ouputs from pandas and scikit-learn
    pd.set_option('display.precision', 4)
    pd.set_option('format.precision', 4)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    sk_config(transform_output='pandas', display='text', print_changed_only=True)
    pass


# Sets options for logging.
log_opts = {'filename': "dmso_model.log",
            'encoding': 'utf-8',
            'mode':     'a'}

logger_opts = {'filename': "{}dmso_model.log".format(os.environ.get('PROJECT_DIR')),
               'encoding': 'utf-8',
               'filemode': 'a',
               'fmt':      '%(message)s', 'style': '%'}

logger = logging.getLogger(__name__)
set_options()
