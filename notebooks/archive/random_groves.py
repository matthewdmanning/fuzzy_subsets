import copy
import csv
import datetime
import os
from functools import partial

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import cv_tools
import univariate_filters
from data import feature_name_lists
from feature_combination import weight_by_feature
from qsar_modeling.data_handling.data_tools import (
    get_interpretable_features,
    load_metadata,
    load_training_data,
)
from qsar_modeling.feature_selection import correlation_filter

# from utils import cv_tools

start = datetime.datetime.now()


def get_score_partials(pos_label=None, label_names=None):
    scores = dict(
        [
            ("Balanced Accuracy", balanced_accuracy_score),
            ("Matthews Correlation Coefficient", matthews_corrcoef),
            # ('Hinge Loss', hinge_loss),
            (
                "Classification Report",
                partial(classification_report, target_names=label_names),
            ),
            (
                "Classification Report - Weighted",
                partial(classification_report, target_names=label_names),
            ),
            (
                "Recall Micro",
                partial(recall_score, pos_label=pos_label, average="micro"),
            ),
            ("Hamming Loss", hamming_loss),
            ("Negative Log Loss", partial(log_loss)),
            ("Negative Log Loss Normalized", partial(log_loss, normalize=True)),
            ("Jaccard score", partial(jaccard_score, pos_label=pos_label)),
        ]
    )
    return scores


prob_scores = dict(
    [
        (
            "Average Precision Macro",
            partial(average_precision_score, pos_label=0, average="macro"),
        ),
        (
            "Average Precision Micro",
            partial(average_precision_score, pos_label=0, average="micro"),
        ),
        (
            "Average Precision Weighted",
            partial(average_precision_score, pos_label=0, average="weighted"),
        ),
        ("Negative Brier Score", partial(brier_score_loss, pos_label=0)),
        ("Precision Micro", partial(precision_score, pos_label=0, average="micro")),
        # ('ROC Score', partial(roc_curve, pos_label=0))
    ]
)


def brute_force_importance_rf_clf(
    feature_df, labels, n_features_out, save_dir, n_jobs=-2, **model_kwargs
):
    clf = RandomForestClassifier(
        random_state=0, oob_score=True, n_jobs=n_jobs, **model_kwargs
    )
    eliminator = RFE(estimator=clf, n_features_to_select=int(n_features_out)).fit(
        feature_df, y=labels
    )
    brute_features = pd.Series(
        eliminator.ranking_, index=feature_df.columns.tolist()
    ).sort_values()
    brute_features.to_csv(
        "{}feature_rankings.csv".format(save_dir),
        index_label="Features",
        float_format="%.4f",
    )
    return eliminator.support_, eliminator


def get_subsample_from_meta(
    field, include=None, exclude=None, feature_df=None, meta_df=None
):
    """
    Retrieves a single selection from metadata as INCHI keys.
    Parameters
    ----------
    meta_df
    feature_df
    field
    include
    exclude
    """
    subsample = list()
    if feature_df is None or meta_df is None:
        x, y, m = load_training_data()
        if feature_df is None:
            feature_df = x
        if meta_df is None:
            meta_df = load_metadata()
    for ind, val in meta_df[field].items:
        if all([i.lower() in val.lower() for i in include]) and not any(
            [i.lower() in val.lower() for i in exclude]
        ):
            subsample.append(ind)
    return subsample


n_feats = 30

"""
# Select EPA soluble/combined insoluble dataset.
X, y, _ = load_training_data()
meta = load_metadata()
interp_X, interp_y = get_interpretable_features(X, y)
print(meta['en_in'][1].columns)
train_insols = y[y == 0].index
insol_samples = pd.concat([meta['epa_in'][0], meta['en_in'][0]]).index.intersection(train_insols)
train_sols = y[y == 1].index
sol_samples = meta['epa_sol'][0].index.intersection(train_sols)
all_samples = insol_samples.append(sol_samples)
train_y = y[all_samples]
train_X = X.loc[all_samples][interp_X.columns]
"""
train_X, train_y = get_interpretable_features()
ind_list = cv_tools.get_split_ind(train_X, train_y, n_splits=5)
feature_groups = feature_name_lists.get_features_dict(train_X.columns)
element_feats = feature_name_lists.get_atom_numbers(train_X.columns)
estate_counts = feature_name_lists.get_estate_counts(train_X.columns)
eplus_features = copy.deepcopy(estate_counts)
# [estate_counts.extend(feature_groups[n]) for n in ['HBond', 'LogP']]
weighting = False
if weighting:
    weighting_desc = "Number of heavy atoms (i.e. not hydrogen)"
    estate_weighted = weight_by_feature(
        feature_df=train_X.copy(), to_norm=estate_counts, norm_by=weighting_desc
    )

# models_dir = os.environ.get('MODELS_DIR')
# pearson_path = '{}pearson_cov_feature_lists.pkl'.format(os.environ.get('MODELS_DIR'))
# mi_path = '{}mi_bivariate_feature_lists.pkl'.format(os.environ.get('MODELS_DIR'))
# pearson_list, mi_list = get_feature_set(train_X.copy(), train_y.copy(), split_list=ind_list, n_features_out=n_feats)

print("Check feature lists")
print([c for c in train_X.columns if "e-state" in c.lower()])
# Get split indices.
exp_dir = "{}grove_trials/".format(
    "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/qsar-modeling-workflow/models/"
)
if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    indices_list = list()


def brute_force_importance_rf_clf(
    feature_df, labels, n_features_out, save_dir, n_jobs=-2, **model_kwargs
):
    clf = RandomForestClassifier(
        n_estimators=500, random_state=0, n_jobs=n_jobs, **model_kwargs
    )
    eliminator = RFE(estimator=clf, n_features_to_select=n_features_out).fit(
        feature_df, y=labels
    )
    brute_features = pd.Series(
        eliminator.ranking_, index=feature_df.columns.tolist()
    ).sort_values()
    brute_features.to_csv(
        "{}feature_rankings.csv".format(save_dir),
        index_label="Features",
        float_format="%.4f",
    )
    return brute_features, eliminator


def undersampler(feature_df, labels, feature_name, feature_dir):
    if labels.size < 50:
        print("Less than 50 compounds in list. Not running {}".format(feature_name))
        return None
    elif any([v < 25 for v in labels.value_counts().values]):
        print("Insufficient values for {}".format(feature_name))
        return None
    if not os.path.isdir(feature_dir):
        os.mkdir(feature_dir)
    for i in np.arange(1):
        i = i + 1
        run_dir = "{}run_{}/".format(feature_dir, i)
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)
        else:
            print("Path already exists")
            print(run_dir)
        cv = 0

        score_dict = dict([(n, []) for n, m in model_tup])
        for dev, eva in StratifiedKFold().split(feature_df, labels):
            cv = cv + 1
            cv_dir = "{}cv{}/".format(run_dir, cv)
            if not os.path.isdir(cv_dir):
                os.mkdir(cv_dir)
            elif os.path.isdir("{}random_forest/".format(cv_dir)):
                continue
            print("Starting CV run at {}".format(datetime.datetime.now()))
            rus_dev_X, rus_dev_y = feature_df.copy().iloc[dev], labels.copy().iloc[dev]
            rus_eva_X, rus_eva_y = feature_df.copy().iloc[eva], labels.copy().iloc[eva]
            print("Feature subsampledin CV: {}".format(feature_name))
            print(
                "Number of insoluble compounds selected: {}".format(
                    rus_dev_y[rus_dev_y == 0].shape
                )
            )

            with open("{}dev_keys.csv".format(cv_dir), "w") as f:
                csv.writer(f).writerow(rus_dev_y.index)
            with open("{}eval_keys.csv".format(cv_dir), "w") as f:
                csv.writer(f).writerow(rus_eva_y.index)
            rus_dev_X = rus_dev_X.copy()[rus_dev_X.var(axis=1) > 0]
            rus_eva_X = rus_eva_X.copy()[rus_dev_X.columns]
            label_mi = univariate_filters.balanced_mi_y(
                rus_dev_X, rus_dev_y, n_neighbors=9
            ).iloc[: int(5 * n_feats)]
            rus_dev_X = rus_dev_X[label_mi.index]
            rus_eva_X = rus_eva_X[label_mi.index]
            # Eliminate collinear features.
            dev_corr = rus_dev_X.corr(method="spearman")
            # dev_corr_T = rus_dev_X[label_mi.index].T.corr(method='spearman')
            # dev_corr_symm = (dev_corr + dev_corr_T) / 2
            drop_feats = []
            cut_off = 0.85
            max_cut = 0.99
            min_cut = 0.85
            step = 0.005
            smaller_size = 0
            n_drop_feats = rus_dev_X.shape[1] - 3 * n_feats
            while True:
                hc_feats = correlation_filter.find_correlation(
                    dev_corr, n_drop=n_drop_feats, cutoff=cut_off, exact=True
                )
                high_corr_feats = [c for c in hc_feats if c not in estate_counts]
                size = len(high_corr_feats)
                print("Cutoff: {}, Size: {}".format(cut_off, size))
                if size > (rus_dev_X.shape[1] - n_feats * 2):
                    if smaller_size == 1:
                        break
                    smaller_size = -1
                    cut_off = cut_off + step
                    drop_feats = high_corr_feats
                elif size <= (rus_dev_X.shape[1] - 2 * n_feats):
                    if smaller_size == -1:
                        break
                    smaller_size = 1
                    drop_feats = high_corr_feats
                    cut_off = cut_off - step
                else:
                    break
                if cut_off > max_cut or cut_off < min_cut:
                    drop_feats = high_corr_feats
                    break
            print("Cutoff: {}, Size: {}".format(cut_off, size))
            print("Dropping highly correlated features: {}".format(drop_feats))
            rus_dev_X.drop(columns=drop_feats, inplace=True)
            rus_eva_X.drop(columns=drop_feats, inplace=True)
            # Brute Force Importance
            final_feats, imp_model = brute_force_importance_rf_clf(
                rus_dev_X, rus_dev_y, n_feats, cv_dir
            )
            brute_drops = [c for c in rus_dev_X.columns if c not in final_feats]
            print("Dropping unimportant features: {}".format(brute_drops))
            rus_dev_X.drop(columns=brute_drops, inplace=True)
            rus_eva_X.drop(columns=brute_drops, inplace=True)
            label_mi[rus_dev_X.columns].to_csv(
                "{}solubility_features_mutual_info.csv".format(cv_dir),
                index_label="Features",
                float_format="%.5f",
            )
            print("Using features: {}".format(rus_dev_X.columns.tolist()))
            std_scaler = StandardScaler().fit(rus_dev_X)
            fit_dev_X = std_scaler.transform(rus_dev_X)
            fit_eva_X = std_scaler.transform(rus_eva_X)
            # fit_dev_X, fit_eva_X = rus_dev_X, rus_eva_X
            # Run model fit
            positive_lab = rus_dev_y.unique()[0]
            logistic_cv = partial(
                LogisticRegressionCV, max_iter=1000, solver="liblinear"
            )
            forest_params = {
                "n_estimators": [10, 100, 1000],
                "max_depth": [5, 10],
                "bootstrap": [False],
                "ccp_alpha": [0.0025, 0.01],
            }
            svc_params = {"kernel": ["poly", "rbf", "sigmoid"]}
            gbc_params = {"n_estimators": [50, 250, 1000]}
            model_tup = (
                ("logistic-cv", logistic_cv, []),
                ("random-forest", RandomForestClassifier, ParameterGrid(forest_params)),
                ("svm", SVC, ParameterGrid(svc_params)),
                ("gradboost", GradientBoostingClassifier, ParameterGrid(gbc_params)),
            )
            for model_name, model, hypers in model_tup:
                if "logistic" not in model_name:
                    search_party = GridSearchCV(model(), hypers, n_jobs=-2, cv=3).fit(
                        X=rus_dev_X, y=rus_dev_y
                    )
                    fitted = search_party.best_estimator_
                    print(
                        "MCC score of {} for parameters: {}".format(
                            search_party.best_score_, search_party.best_params_
                        )
                    )
                    score_dict[model_name].append(search_party.best_score_)
                if model_name == "logistic-cv":
                    fitted = model(n_jobs=-2).fit(fit_dev_X, rus_dev_y)
                    score_dict[model_name].append(fitted.score)

                select_dev_pred = fitted.predict(fit_dev_X)
                select_eva_pred = fitted.predict(fit_eva_X)
                select_dev_prob = fitted.predict_proba(fit_dev_X)
                select_eva_prob = fitted.predict_proba(fit_eva_X)
                model_path = "{}{}/".format(cv_dir, model_name)
                if not os.path.isdir(model_path):
                    os.mkdir(model_path)
                cv_dev_score_dict = dict()
                cv_eval_score_dict = dict()
                pred_scores = get_score_partials(pos_label=positive_lab)
                for score_name, score_obj in pred_scores.items():
                    cv_dev_score_dict[score_name] = score_obj(
                        rus_dev_y, select_dev_pred
                    )
                    cv_eval_score_dict[score_name] = score_obj(
                        rus_eva_y, select_eva_pred
                    )
                """
                for score_name, score_obj in prob_scores.items():
                    cv_dev_score_dict[score_name] = score_obj(rus_dev_y, select_dev_pred)
                    cv_eval_score_dict[score_name] = score_obj(rus_eva_y, select_eva_pred)
                    # cv_dev_score_dict[score_name] = score_obj(rus_dev_y, select_dev_prob)
                    # cv_eval_score_dict[score_name] = score_obj(rus_eva_y, select_eva_prob)
                """
                with open("{}dev_scores.csv".format(model_path), "w") as f:
                    csv.writer(f).writerow(cv_dev_score_dict.items())
                with open("{}eval_scores.csv".format(model_path), "w") as f:
                    csv.writer(f).writerow(cv_eval_score_dict.items())
                """
                with open('{}dev_proba.csv'.format(model_path), 'w') as f:
                    csv.writer(f).writerow(model.predict_proba(rus_dev_y))
                with open('{}eval_proba.csv'.format(model_path), 'w') as f:
                    csv.writer(f).writerow(model.predict_proba(rus_eva_y))
                """
        end = datetime.datetime.now()
        exec_time = end - start
        print("One CV run took {}".format(exec_time))
        return score_dict


for i, ecount in enumerate(estate_counts):
    feat_dir = "{}{}/".format(exp_dir, i)
    if os.path.isdir(feat_dir) and os.path.isdir("{}cv5/".format(feat_dir)):
        continue
    # subsample = train_X[ecount].replace(lambda x: x == 0, value=0)
    e_label = train_X[ecount].squeeze().astype(int)
    e_label[e_label > 0] = 1
    print(e_label.value_counts())
    if e_label.value_counts().size != 2 or e_label.size / train_y.size < 0.75:
        continue
    # null_sample = train_X[ecount][train_X[ecount] == 0].index
    under_X, under_sample = RandomUnderSampler(random_state=1000 * i).fit_resample(
        train_X, e_label
    )
    cv_scores = undersampler(under_X, train_y[under_X.index], ecount, feat_dir)
    if cv_scores is not None:
        for k, v in cv_scores:
            print("{} {}: {}".format(ecount, k, np.mean(v)))
