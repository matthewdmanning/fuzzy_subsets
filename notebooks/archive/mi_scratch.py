# from chefboost import Chefboost
import logging
import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import jaccard_score
from sklearn.utils.validation import check_X_y

import constants
import cv_tools
import distributions
import feature_name_lists
import features
from data_handling.data_tools import load_training_data
from modeling.quick_models import balanced_forest, logistic_clf

logger = logging.getLogger("dmso_solubility.mi_scratch")
indices_list = list()
results_dev_dict = {}
results_eval_dict = {}
i = 0


def get_feat_mi(input_df, input_labels):
    disc_dict, mi_list = dict(), list()
    for f in input_df.columns:
        disc_dict[f] = [
            np.all([a, b])
            for a, b in zip(
                distributions.is_discrete(input_df).tolist(),
                distributions.is_low_cardinal(input_df),
                strict=False,
            )
        ]
    disc_ser = pd.Series(data=disc_dict)[input_df.columns]
    for _ in list(range(30)):
        rus_X, rus_y = RandomUnderSampler().fit_resample(input_df, input_labels)
        mi_list.append(
            pd.Series(
                mutual_info_classif(
                    rus_X, rus_y, discrete_features=disc_ser, random_state=0, n_jobs=-1
                ),
                index=rus_X.columns,
            ).T
        )
    comb_mi_df = pd.concat(mi_list, axis=1)
    print(comb_mi_df.shape, flush=True)
    comb_mi_df.index = input_df.columns
    return comb_mi_df


def get_pca_vector(
    train_data,
    test_data,
    train_labels,
    feature_list,
    group_name,
    mi_ser=None,
    evr_thresh=0.925,
    mi_tol_pct=5,
):
    # mi_dict[remain_name] = list()
    over_X, over_y = RandomOverSampler(random_state=0).fit_resample(
        train_data, train_labels
    )
    ordered_feats = features.sort_ordinals(feature_list, start=10, step=1)
    subset = ordered_feats[:2]
    for feat in ordered_feats[2:]:
        subset.append()
        this_PCA = PCA(n_components=1, whiten=False, random_state=0).fit(
            X=over_X[subset], y=over_y
        )
        logger.info("Features in {}".format(remain_name))
        [logger.info("Mean MI: {} for {}".format(top_mi_ser[f], f)) for f in feats]
        logger.info(
            "Explained Variance Ratios: {}".format(
                pd.Series(this_PCA.explained_variance_ratio_).to_string()
            )
        )
        logger.info("Principal Components: {}".format(this_PCA.components_))
        pca_dev_X = pd.DataFrame(
            this_PCA.transform(train_data[feats].copy()),
            columns=[remain_name],
            index=train_data.index,
        )
        pca_eval_X = pd.DataFrame(
            this_PCA.transform(test_data[feats].copy()),
            columns=[remain_name],
            index=test_data.index,
        )
        if this_PCA.explained_variance_ratio_[0] < evr_thresh:
            subset.pop()
    return (pca_dev_X, pca_eval_X), subset


# Load data
loaded_X, loaded_y, total_meta_df = load_training_data()
X_arr, y_arr = check_X_y(X=loaded_X, y=loaded_y)
train_X = pd.DataFrame(data=X_arr, index=loaded_X.index, columns=loaded_X.columns)
train_X = train_X[train_X.columns[train_X.std(axis=0) > 0.01]]
bad_feats = pd.Index(
    [
        c
        for c in train_X
        if any([u.lower() in c.lower() for u in constants.names.uninterpretable])
    ]
)
train_X.drop(columns=bad_feats, inplace=True)
print(train_X.shape)
train_y = pd.Series(data=y_arr, index=loaded_y.index)
train_y.name = "Solubility"
assert not train_y.empty and not train_X.empty
split_data = cv_tools.split_df(train_X, train_y)

# Sample Clustering
#
# print(mol_wt_col, logp, rotate)
# train_y = LabelBinarizer().fit_transform(y=train_y.astype(int))
listed_top_feats = [
    "Molecular path count of order 6",
    "Total path count (up to order 10)",
    "Molecular path count of order 9",
    "Molecular path count of order 10",
    "Molecular path count of order 5",
    "Molecular path count of order 7",
    "Molecular path count of order 8",
    "Number of rings (includes counts from fused rings)",
    "Molecular path count of order 4",
    "Molecular walk count of order 9 (ln(1+x)",
    "Molecular path count of order 3",
    "Molecular walk count of order 10 (ln(1+x)",
    "Molecular walk count of order 8 (ln(1+x)",
    "Molecular walk count of order 5 (ln(1+x)",
    "Molecular walk count of order 6 (ln(1+x)",
    "Molecular walk count of order 7 (ln(1+x)",
    "Self-returning walk count of order 10 (ln(1+x)",
    "Conventional bond order ID number of order 3 (ln(1+x)",
    "Molecular walk count of order 4 (ln(1+x)",
    "Self-returning walk count of order 8 (ln(1+x)",
    "Conventional bond order ID number of order 2 (ln(1+x)",
    "Total self-return walk count (up to order 10) (ln(1+x))",
    "A measure of electronegative atom countii",
    "Conventional bond order ID number of order 6 (ln(1+x)",
    "Number of rings containing heteroatoms (N, O, P, S, or halogens)",
    "Conventional bond order ID number of order 7 (ln(1+x)",
    "Weiner polarity number",
    "Self-returning walk count of order 6 (ln(1+x)",
    "Number of rings",
    "Conventional bond order ID number of order 8 (ln(1+x)",
    "Self-returning walk count of order 4 (ln(1+x)",
    "Conventional bond order ID number of order 10 (ln(1+x)",
    "Conventional bond order ID number of order 9 (ln(1+x)",
    "Molecular walk count of order 3 (ln(1+x)",
    "Molecular walk count of order 2 (ln(1+x)",
    "Conventional bond order ID number of order 5 (ln(1+x)",
    "Total walk count (up to order 10)",
    "Conventional bond order ID number of order 4 (ln(1+x)",
    "Molecular path count of order 2",
    "Number of fused rings",
    "Self-returning walk count of order 2 (ln(1+x)",
    "A measure of electronegative atom count of the molecule relative to "
    "molecular size",
    "Number of bonds (excluding bonds with hydrogen)",
    "A measure of electronegative atom count of the molecule",
]
misc_cols = [
    "Number of hydrogen bond donors (using CDK HBondDonorCountDescriptor algorithm)",
    "Molecular weight",
    "Number of rotatable bonds, " "excluding terminal " "bonds",
    "Weiner polarity number",
    "Molecular path count of order 9",
]
lipinki_col = [
    c for c in train_X.columns if "five" in c.lower() and "rule" in c.lower()
]
# hbond = [c for c in train_X.columns if ('e-state' in c.lower() and 'hydrogen bond' in c.lower() and not ('minimum' in c.lower() or 'maximum' in c.lower() or 'sum' in c.lower()))]
# names = {'Fused rings': fused, 'Hetero plain': plain_hetero, 'Hetero fused': all_rings, 'Path_Counts_PCA': molpath, 'Bond_Orders_PCA': border}
train_y.name = "Solubility"
mi_df = pd.DataFrame(index=list(range(5, 11)), columns=list(range(5, 11)))
# for i in (list(range(6))):
names = feature_name_lists.get_features_dict(train_X.columns.tolist())

# Make MI dictionary.
all_replacers = list()
[all_replacers.extend(f) for f in names.values()]
# [all_replacers.extend(l) for l in replacers_list]
parent_mi_dict = dict()
for k in all_replacers:
    if k not in train_X.columns:
        match = train_X.columns[train_X.columns.isin(k)]
        if match.empty:
            logger.warning("Feature in replacers not found in Dataframe: {}".format(k))
            continue
        elif match.size > 1:
            logger.warning("Multiple partial feature matches: {}".format(match))
            continue
        else:
            k = match
    parent_mi_dict[k] = list()

n_total_feats = 30
pca_split_data, mi_df_list = list(), list()
for cv, (split_dev_X, original_dev_y, split_eval_X, original_eval_y) in enumerate(
    split_data
):
    original_dev_X, original_eval_X = split_dev_X.copy(), split_eval_X.copy()
    if (
        False
    ):  # os.path.isfile('{}rus30_mi_cv{}.csv'.format(os.environ.get('MODELS_DIR'), cv)):
        # mi_df = pd.read_csv('{}rus30_mi_cv{}.csv'.format(os.environ.get('MODELS_DIR'), cv))
        mi_df = pd.read_csv(
            "balanced_mi_all_train_cv30.csv".format(os.environ.get("MODELS_DIR"), cv)
        )
    else:
        mi_df = get_feat_mi(original_dev_X, original_dev_y)
        mi_df.to_csv("{}rus30_mi_cv{}.csv".format(os.environ.get("MODELS_DIR"), cv))
    # mi_df_list.append(mi_df)
    replacers_list, top_feats = list(), list()
    top_mi_ser = mi_df.mean(axis=1).sort_values(ascending=False)
    print(top_mi_ser.head())
    for n_feat in list(range(top_mi_ser.shape[0])):
        print(
            "Feature selected with MI of {:.5f}: {}".format(
                top_mi_ser[n_feat], top_mi_ser.index[n_feat]
            )
        )
        if top_mi_ser.index[n_feat] in all_replacers:
            replacers_list.append(top_mi_ser)
        else:
            top_feats.append(top_feats)
        if len(top_feats) + len(pca_replacers) >= n_total_feats:
            break
    # mi_dict = copy.deepcopy(parent_mi_dict)
    # mi_dict[f].append(mutual_info_classif(rus_X[f], rus_y, discrete_features=disc, random_state=0, n_jobs=-1))
    pca_dict = dict()  # {'dev': list(), 'eval': list()}
    drop_list = list()
    for remain_name in pca_replacers:
        feats = [
            f
            for f in names[remain_name]
            if f in replacers_list[0].index or replacers_list[0].columns
        ]
        print(feats)
        pca_vecs, dropped_feats = get_pca_vector(
            original_dev_X,
            original_eval_X,
            original_dev_y,
            group_name=remain_name,
            feature_list=feats,
            mi_ser=mi_df,
        )
        pca_dict[remain_name] = pca_vecs
        drop_list.append(dropped_feats)
    dev_pca = pd.concat([d for d, e in pca_dict.values()], axis=1)
    eval_pca = pd.concat([e for d, e in pca_dict.values()], axis=1)
    dev_remain = pd.concat([original_dev_X[top_feats].copy(), dev_pca], axis=1)
    eval_remain = pd.concat(
        [original_eval_X[top_feats].copy(), *pca_dict["eval"]], axis=1
    )
    logger.info(
        "Mutual Information for original and PCA transformed features. Discrete used for original path counts."
    )
    # [logger.info('{}: {}'.format(k, v)) for k, v in mi_dict.items()]
    # [dev_remain.drop(names[n], inplace=True) for n in pca_replacers if n in dev_remain.columns]
    # [eval_remain.drop(names[n], inplace=True) for n in pca_replacers if n in eval_remain.columns]
    pca_split_data.append((dev_remain, original_dev_y, eval_remain, original_eval_y))
    logger.info("Using the following features for modeling:")
    [logger.info(c) for c in dev_remain.columns]
    rf_dev_path = "{}rf_{}_dev_cv{}.csv".format(
        os.environ.get("MODELS_DIR"), remain_name, cv
    )
    lr_dev_path = "{}lr_{}_dev_cv{}.csv".format(
        os.environ.get("MODELS_DIR"), remain_name, cv
    )
    rf_eval_path = "{}rf_{}_eval_cv{}.csv".format(
        os.environ.get("MODELS_DIR"), remain_name, cv
    )
    lr_eval_path = "{}lr_{}_eval_cv{}.csv".format(
        os.environ.get("MODELS_DIR"), remain_name, cv
    )
rbf_results_list, probs_results_list, lr_results_list = list(), list(), list()
for dev_X, dev_y, eval_X, eval_y in pca_split_data:
    # brfc, dev_predict, eval_predict, dev_probs, eval_probs
    lr_results_list.append(logistic_clf(dev_X, dev_y, eval_X))
    rbf_results_list.append(balanced_forest(dev_X, dev_y, eval_X))
    dev_y_tups = np.vstack([dev_y.to_numpy(), np.ones_like(dev_y) - dev_y.to_numpy()]).T
    assert np.sum(dev_y_tups[:, 1]) < dev_y_tups.shape[0]
    dev_wts = np.matmul(np.array(rbf_results_list[4]), dev_y_tups.T)[:, 0]
    # follow_forest, dev_probs_predict, eval_probs_predict, dev_probs_proba, eval_probs_proba
    probs_results_list.append(balanced_forest(dev_X, dev_y, eval_X, sample_wts=dev_wts))
    """    
    brfc = BalancedRandomForestClassifier(n_estimators=n_trees, max_depth=15, n_jobs=-1, random_state=0,
                                      sampling_strategy='auto', min_weight_fraction_leaf=0.05,
                                      replacement=False, verbose=1, class_weight='balanced_subsample',
                                      bootstrap=True)
    brfc.fit(X=dev_remain, y=dev_y)
    eval_predict = brfc.predict(X=eval_remain)
    dev_predict = brfc.predict(X=dev_remain)
    logger.info('BRFC Importances: {}'.format(brfc.feature_importances_))
    logger.info('Balanced Accuracy: {:.5f} (Dev) {:.5f} (Eval)    MCC: {:.5f} (Dev) {:.5f} (Eval)'.format(
    balanced_accuracy_score(y_true=dev_y, y_pred=dev_predict),
    balanced_accuracy_score(y_true=eval_y, y_pred=eval_predict),
    matthews_corrcoef(y_true=dev_y, y_pred=dev_predict),
    matthews_corrcoef(y_true=eval_y, y_pred=eval_predict)))eval_probs = brfc.predict(X=eval_remain)
    dev_probs = brfc.predict_proba(X=np.array(dev_remain))
    print(dev_probs)
    """
# PCA BRF

lr_predict_tups = [
    (a[1], b[1], b[2]) for (a, b) in zip(pca_split_data, lr_results_list)
]
rbf_predict_tups = [
    (a[1], b[1], b[2]) for (a, b) in zip(pca_split_data, rbf_results_list)
]
lr_scores = cv_tools.score_cv_results(lr_predict_tups)
rbf_scores = cv_tools.score_cv_results(rbf_predict_tups)
logger.info("Logistic Regression Scores")
cv_tools.log_score_summary(lr_scores, score_logger=logger)
logger.info("Balanced Random Forest Scores")
cv_tools.log_score_summary(rbf_scores, score_logger=logger)
# Probability-weighted BRF
probs_predict_tups = [
    (a[1], b[1], b[2]) for (a, b) in zip(pca_split_data, probs_results_list)
]
proba_scores = cv_tools.score_cv_results(probs_predict_tups)
cv_tools.log_score_summary(proba_scores, score_logger=logger)
logger.info("(Weighted) Jaccard scores between first and second BRF:")
[
    logger.info(
        jaccard_score(y_true=a[1], y_pred=b[1], pos_label=0, average="weighted")
    )
    for a, b in zip(rbf_results_list, probs_results_list)
]
# rbf_importance = dict([(zip(a[0].feature_names_in_, a[0].feature_importances_)) for a in rbf_results_list])
# probs_importance = [dict(zip(a[0].feature_names_in_, a[0].feature_importances_)) for a in probs_results_list]
# comb_importance = [(a, rbf_importance[a], probs_importance[a]) for a in rbf_importance.keys()]
# logger.info('BRF Feature Importances (Gini Impurity-based)')
# logger.info(comb_importance)
# for (dev_X, dev_y, eval_X, eval_y), (brfc, dev_predict, eval_predict, dev_probs, eval_probs) in pca_split_data, rbf_results_list:


"""    
logger.info('Balanced Random Forest: Weighted by Opposite-Class Prediction Probilities of First RF')
follow_forest = BalancedRandomForestClassifier(n_estimators=n_trees, max_depth=15, n_jobs=-1, random_state=0,
                                                   sampling_strategy='auto', min_weight_fraction_leaf=0.05,
                                                   replacement=False, verbose=1, class_weight='balanced_subsample',
                                                   bootstrap=True)
    follow_forest.fit(X=dev_remain, y=dev_y, sample_weight=dev_wts)
    eval_probs_predict = follow_forest.predict(X=eval_remain)
    dev_probs_predict = follow_forest.predict(X=dev_remain)
    
    logger.info('BRFC Prob-Predict Feature Importances: {}'.format(follow_forest.feature_importances_))
        logger.info('Prob-Predict Balanced Accuracy: {:.5f} (Dev) {:.5f} (Eval)    MCC: {:.5f} (Dev) {:.5f} (Eval)'.format(
        balanced_accuracy_score(y_true=dev_y, y_pred=dev_probs_predict),
        balanced_accuracy_score(y_true=eval_y, y_pred=eval_probs_predict),
        matthews_corrcoef(y_true=dev_y, y_pred=dev_probs_predict),
        matthews_corrcoef(y_true=eval_y, y_pred=eval_probs_predict)))
for p in range(10):
        new_X, y = RandomUnderSampler().fit_resample(train_X, y=train_y)
        mi = mutual_info_regression(X=new_X[cols[i]].to_frame(), y=new_X[cols[j]],
                                    discrete_features=True, random_state=0)
        mi_list.append(mi)
    pprint.pp('{:.5f} ({:.5f})    {}    {}'.format(np.mean(mi_list), np.std(mi_list), i, j))
    mi_df.iloc[i, j] = np.mean(mi_list)
print(mi_df.to_string(max_colwidth=10))
"""
