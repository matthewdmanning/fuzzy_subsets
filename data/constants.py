from __future__ import annotations

import os
from collections import namedtuple

from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

FINAL_DIR = os.environ.get("FINAL_DIR")

MODELS_DIR = "{}models/".format(FINAL_DIR)
# FINAL_DIR = os.environ.setdefault('FINAL_DIR', FINAL_DIR)
# MODELS_DIR = os.environ.setdefault('MODELS_DIR', MODELS_DIR)
#
#

# Class -> Data Structure Depedendent
Paths = namedtuple(
    "Paths",
    (
        "feature_groups",
        "combo_path",
        "feat_label_tup",
        "train",
        "feature_frame",
        "mutual_info",
        "cov_mat",
    ),
    module=__name__,
)

# Instance -> System/Model Dependent
paths = Paths(
    "{}padel/padel_desc_groups.csv".format(FINAL_DIR),
    "{}filtered/PADEL_CFP_COMBO_5mM.pkl".format(FINAL_DIR),
    "{}padel/PADEL_EPA_ENAMINE_5mM_TUPLES.pkl".format(FINAL_DIR),
    "{}filtered/MAXMIN_PADEL_TRAIN.pkl".format(FINAL_DIR),
    "{}feature_frame.pkl".format(FINAL_DIR),
    "{}mutual_info.pkl".format(MODELS_DIR),
    "{}cov.pkl".format(MODELS_DIR),
)
Names = namedtuple(
    "Names",
    (
        "feat_cols",
        "auto_corr_weights",
        "index_groups",
        "uninterpretable",
        "forbidden",
        "info",
    ),
    module=__name__,
)
names = Names(
    ("Type", "Number", "Descriptor", "Class"),
    (
        "mass",
        "charges",
        "van der Waals",
        "Sanderson",
        "electronegativites",
        "polarizabilities",
        "ionization",
        "I-state",
    ),
    ("zero_var", "nonzero_var", "sparse", "dense", "discrete", "cont", "nonneg", "neg"),
    (
        "information",
        "eigen",
        "e-state",
        "spectral",
        "barysz",
        "randic",
        "bcut",
        "estrada",
        "detour",
        "burden",
    ),
    (
        "BurdenModifiedEigenvaluesDescriptor",
        "BaryszMatrixDescriptor",
        "ExtendedTopochemicalAtomDescriptor",
        "DetourMatrixDescriptor",
        "ChiClusterDescriptor",
        "ChiPathDescriptor",
        "ChiPathClusterDescriptor",
        "ChiChainDescriptor",
    ),
    (
        "InformationContentDescriptor",
        "TotalInformationContentDescriptor",
        "StructuralInformationContentDescriptor",
        "ComplementaryInformationContentDescriptor",
        "BondInformationContentDescriptor",
        "ModifiedInformationContentDescriptor",
        "Z-modifiedInformationContentDescriptor",
    ),
)
min_feats_dict = {
    "min_sparse": 10,
    "min_dense": 10,
    "min_continuous": 20,
    "min_discrete": 10,
}
thresholds_dict = {
    "cov_thresh": 0.85,
    "sparse_thresh": 0.025,
    "discrete_thresh": 0.0025,
}
tolerances_dict = {"tol_discrete": 0.001, "tol_sparse": 0.01}
methods_dict = {"cov_meth": "spearman"}
SelectorParams = namedtuple(
    "SelectorParams",
    (
        *min_feats_dict.keys(),
        *thresholds_dict.keys(),
        *tolerances_dict.keys(),
        *methods_dict.keys(),
    ),
    module=__name__,
)
selector_params = SelectorParams(
    *min_feats_dict.values(),
    *thresholds_dict.values(),
    *tolerances_dict.values(),
    *methods_dict.values(),
)
run_params_dict = {"pjobs": -1, "debug": False}
# min_feats = namedtuple('min_feats', field_names=min_feats_dict.keys(), defaults=min_feats_dict.values())
# thresholds = namedtuple('thresholds', field_names=thresholds_dict.keys(), defaults=thresholds_dict.values())
# tolerances = namedtuple('tolerances', field_names=tolerances_dict.keys(), defaults=tolerances_dict.values())
# methods = namedtuple('methods', field_names=methods_dict.keys(), defaults=methods_dict.values())
RunParams = namedtuple("RunParams", (run_params_dict.keys()))
run_params = RunParams(*run_params_dict.values())

"""
Options = namedtuple('Options', ['paths', 'names', 'log_opts', 'min_feats', 'threshholds', 'tolerances', 'methods',
                                 'run_params'], module=__name__)
options = Options(paths, names, log_opts, min_feats, thresholds, tolerances, methods, run_params)
# model_opts = options_tuple(__fields='model_opts', __defaults= [paths, names, log_opts, min_feats, thresholds, tolerances,methods, run_params])
LogOpts = namedtuple('LogOpts', field_names=log_opts.keys(), 
log_opts = LogOpts(log_opts.values(), module=__name__)
"""
score_dict = {
    "Balanced Accuracy": balanced_accuracy_score,
    "Matthews Correlation": matthews_corrcoef,
}
names_dict = {
    "mlp": "Multilayer Perceptron",
    "rf": "Random Forest",
    "lr": "Logistic Regression",
    "eval": "Eval",
    "dev": "Dev",
}
# 'Jaccard': jaccard_score,
