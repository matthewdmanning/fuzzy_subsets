import copy
import logging
import os
import pickle
import pprint

import pandas as pd

import cv_tools
import samples
from data import feature_name_lists
from data_handling.data_tools import get_interpretable_features
from feature_combination import combine_feature_groups
from feature_selection.feature_combination import weight_by_feature
from feature_selection.univariate_filters import get_mi_features
from modeling.quick_models import balanced_forest, logistic_clf

logger_opts = {
    "filename": "dmso_model.log",
    "filemode": "w",
    "format": "%(message)s",
    "style": "%",
}
logging.basicConfig(**logger_opts)
grove_log = logging.getLogger()
# Load data
train_X, train_y = get_interpretable_features()
ind_list = cv_tools.get_split_ind(train_X, train_y, n_splits=5)

feature_groups = feature_name_lists.get_features_dict(train_X.columns)
element_feats = feature_name_lists.get_atom_numbers(train_X.columns)
estate_counts = feature_name_lists.get_estate_counts(train_X.columns)
models_dir = os.environ.get("MODELS_DIR")

n_feats = len(estate_counts)


#
# TOOD: Import and loop through feature lists.
# Get MI and feature importance (compared to all other members of the list or candidates within cutoff) on all members.
# Run PCA on list of features. Drop old features and replace with PCA vector.
# Update MI/candidate list. Recalculate importance as needed.
# Run models and compare with previous methods.


def get_feature_set(feature_df, labels, split_list, n_features_out):
    pearson_feat_list, mi_feat_list = list(), list()
    for dev_X, dev_y, eva_X, eva_y in cv_tools.split_df(
        feature_df.copy(), labels, indices_list=split_list
    ):
        # dev_X = dev_X.iloc[:1000, :]
        # dev_y = dev_y.iloc[:1000]
        mi_feat_list.append(
            get_mi_features(
                feature_df=dev_X.copy(),
                labels=dev_y,
                test_feature_df=eva_X,
                n_features_out=n_features_out,
                n_neighbors=7,
            )
        )
        if (
            False
        ):  # os.path.isfile('{}pearson_cov_feature_lists.pkl'.format(os.environ.get('MODELS_DIR'))):
            with open(
                "{}pearson_cov_feature_lists.pkl".format(os.environ.get("MODELS_DIR")),
                "rb",
            ) as f:
                pearson_feat_list = pickle.load(f)
        # else:
        #    pearson_feat_list.append(pearson_filtered(feature_df=dev_X.copy(), labels=dev_y.copy(), n_features_out=n_features_out, covar_thresh=0.9))
    return pearson_feat_list, mi_feat_list


def model_feat_loop(
    model,
    model_name,
    feature_df,
    labels,
    feature_set_list,
    indices_list,
    pca_features=None,
):
    model_list, cv_results, wrong_insol, wrong_sol = list(), list(), list(), list()
    print(indices_list)
    print(feature_set_list)
    i = 0
    from imblearn.under_sampling import RandomUnderSampler

    for indices, feature_set in zip(
        cv_tools.split_df(feature_df, labels, indices_list=indices_list),
        feature_set_list,
    ):
        dev_X, dev_y, eva_X, eva_y = indices
        print(feature_set)
        if type(feature_set) is pd.DataFrame:
            feature_set = feature_set.columns.tolist()
        elif type(feature_set) is pd.Series:
            feature_set = feature_set.index.tolist()
        try:
            feature_set = feature_set.index.tolist()
            rus, rus_y = RandomUnderSampler().fit_resample(dev_X, dev_y)
            erus, erus_y = RandomUnderSampler().fit_resample(eva_X, eva_y)
            dx, dy, ex, ey, former = combine_feature_groups(feature_df=rus[feature_set])
            dev_X = former(dev_X)
            eva_X = former(eva_X)
            feature_set = [c for c in dev_X if c in feature_set]
        except:
            print("PCA failed.")
        dev_X = dev_X[feature_set]
        eva_X = eva_X[feature_set]
        fit_model = model(dev_X, train_labels=dev_y, test_data=eva_X)
        model_list.append(fit_model)
        cv_results.append(cv_tools.package_output(dev_y, eva_y, fit_model))
        wrongs = samples.get_confusion_samples((eva_y, fit_model[2]))
        wrong_insol.extend(wrongs[0])
        wrong_sol.extend(wrongs[1])
        print("Features used in {} model on {}".format(model_name, set_name))
        # feature_mi = balanced_mi_y(feature_df=dev_X[feature_set], labels=dev_y).mean(axis=1).squeeze().sort_values(ascending=False)
        pd.Series(dev_X.columns).to_csv(
            "{}{}{}_pca_feats_mi.csv".format(models_dir, set_name, model_name)
        )
        # pprint.pp(feature_mi)
    print("Score for {} model on {} features".format(model_name, set_name))
    cv_tools.log_score_summary(
        cv_tools.score_cv_results(cv_results), grove_log, level=20
    )
    return model_list, cv_results, wrong_insol, wrong_sol


# if pca_features is not None:
#     dev_X, dev_X_pca, eva_X, eva_X_pca = combine_feature_groups(train_data=all_dev_X[feature_set], test_data=all_eva_X[feature_set], train_labels=dev_y, sort_key=mi_avg,
#                                                   oversampler=False)
#    features_in = dev_X.columns[:pca_features - dev_X_pca.shape[1]].append(dev_X_pca.columns)
# for group_name, feat_list in
# [print(i, g) for i, g in list(mi_avg[estate_counts].items())]
"""
from sklearn.manifold import TSNE
ts = TSNE(perplexity=25, metric='manhattan', angle=0.25, n_jobs=-1, verbose=1)
ts_data = ts.fit_transform(train_X)
with open('{}tsne_data'.format(os.environ.get('MODELS_DIR')), 'wb') as p:
    pickle.dump(ts_data, p)
print(ts.kl_divergence_)
plt.scatter(x=ts_data[:, 0], y=ts_data[:, 1], c=train_y, s=0.1, alpha=0.025)
plt.show()
exit()
"""


def get_feature_stats(feature_X, labels):
    estate_list, count_list, positive_list = dict(), dict(), dict()
    for feat in estate_counts:
        positive_list[feat] = labels[feature_X[feat] > 0].value_counts(
            sort=False, normalize=True
        )[0]
        feat_name = feat.replace("Count of atom-type E-State: ", "")
        estate_list[feat_name] = (labels[feature_X[feat] > 0] == 0).astype(
            int
        ).sum() / labels.size
        insol_count = feature_X.loc[labels == 0][feat].value_counts(
            sort=False
        )  # .rename(feat.replace('Count of atom-type E-State: ', ''))
        sol_count = feature_X.loc[labels == 1][feat].value_counts(
            sort=False
        )  # .rename(feat.replace('Count of atom-type E-State: ', ''))
        # pprint.pp(pd.concat([insol_count, sol_count], axis=1).sort_index())
        count_list[feat_name] = pd.concat([insol_count, sol_count], axis=1).sort_index()
    count_keys = sorted(count_list.keys(), key=lambda x: estate_list[x], reverse=True)
    [pprint.pp(count_list[f]) for f in count_keys[:9]]


# results_list.append(e_count.rename())


# Estate Descriptors
eplus_features = copy.deepcopy(estate_counts)
# [estate_counts.extend(feature_groups[n]) for n in ['HBond', 'LogP']]
weighting_desc = "Number of heavy atoms (i.e. not hydrogen)"
estate_weighted = weight_by_feature(
    feature_df=train_X.copy(), to_norm=estate_counts, norm_by=weighting_desc
)

pearson_path = "{}pearson_cov_feature_lists.pkl".format(os.environ.get("MODELS_DIR"))
mi_path = "{}mi_bivariate_feature_lists.pkl".format(os.environ.get("MODELS_DIR"))

pearson_list, mi_list = get_feature_set(
    train_X.copy(), train_y.copy(), split_list=ind_list, n_features_out=n_feats
)
print("Check feature lists")
# print(len(pearson_list), len(pearson_list[0]))
# ('Pearson_filtered_list', pearson_list,),
# with open(pearson_path, 'wb') as f:
#     pickle.dump(pearson_list, f)
# with open(mi_path, 'wb') as f:
#     pickle.dump(mi_list, f)
# [print(n, feature_groups[n]) for n in ['HBond', 'Mol_Wt', 'LogP']]
# eplus_weighted = pd.concat([weight_by_feature(feature_df=train_X, to_norm=estate_counts, norm_by=weighting_desc), train_X[weighting_desc]])
# , ('Weighted_Estate', estate_weighted.columns)
# for set_name, feat_set in zip(['E-State', 'E-Plus'], [estate_counts, copy.deepcopy(estate_counts).extend(eplus_features)]):
# for set_name, feat_set in zip(['E-State', 'E-Plus'], [estate_counts, copy.deepcopy(estate_counts).extend(eplus_features)]):
# ('Pearson_filtered_list', pearson_list,),
for set_name, feat_sets in [("MI_interaction_list", mi_list)]:
    print("Training on: ", set_name)
    grove_log.info("Training models on {}".format(set_name))
    pca = False
    if set_name == "Top_MI":
        pca = True
    for mod_name, mod in zip(
        ["Logistic_Regression", "Balanced_Random_Forest"],
        [logistic_clf, balanced_forest],
    ):
        model_out, cv_out, miss_insol, miss_sol = model_feat_loop(
            mod,
            mod_name,
            train_X,
            train_y,
            feature_set_list=mi_list,
            indices_list=ind_list,
        )
        wrong_insol_df = pd.concat(
            [utils.samples.get_sample_info(w) for w in miss_insol], axis=1
        )
        wrong_sol_df = pd.concat(
            [utils.samples.get_sample_info(w) for w in miss_sol], axis=1
        )
        wrong_insol_df.to_csv(
            "{}{}{}_incorrect_insol.csv".format(models_dir, set_name, mod_name)
        )
        wrong_sol_df.to_csv(
            "{}{}{}_incorrect_sol.csv".format(models_dir, set_name, mod_name)
        )
exit()
"""
        log_list.append(logistic_clf(dev_X[estate_counts], train_labels=dev_y, test_data=eva_X[estate_counts]))
        forest_list.append(balanced_forest(dev_X[estate_counts], train_labels=dev_y, test_data=eva_X[estate_counts]))
    log_true_pred = [utils.cv_tools.package_output(train_y.iloc[a[0]], train_y.iloc[a[1]], b) for a, b in zip(ind_list, log_list)]
    bf_true_pred = [utils.cv_tools.package_output(train_y.iloc[a[0]], train_y.iloc[a[1]], b) for a, b in zip(ind_list, forest_list)]
    # Get info from incorrect predictions and save to file.
    for model_tups, model_name in zip([log_true_pred, bf_true_pred], ['logistic', 'balanced_forest']):
        #
        incorrect_df = pd.concat([pd.Series(utils.cv_tools.get_confusion_samples(r)) for r in model_tups], axis=1).T
        print('Incorrect list: {}'.format(incorrect_df))
        sample_info = [pd.concat([pd.DataFrame(d) for d in samples.get_sample_info(incorrect_df)])]
        for i, s in sample_info:
            i.to_csv('{}{}{}_incorrect_insol.csv'.format(os.environ.get('MODELS_DIR'), set_name, model_name))
            s.to_csv('{}{}{}_incorrect_sol.csv'.format(os.environ.get('MODELS_DIR'), set_name, model_name))

    log_score_dict = utils.cv_tools.score_cv_results(log_true_pred)
    bf_score_dict = utils.cv_tools.score_cv_results(bf_true_pred)
    [utils.cv_tools.log_score_summary(sd, grove_log) for sd in [log_score_dict, bf_score_dict]]

for feat_list in [feature_groups, element_feats, estate_counts]:
    feats = [f for f in feat_list if f in mi_avg.index]
    feat_mis = mi_avg[feats]
    print(pprint.pformat(feat_mis))
"""
"""
def prep_info_nce(features_df, labels, **kwargs):
    info_loss = info_nce.InfoNCE()
    batch_size, num_negative, embedding_size = 32, 48, 128 Examples:
    >>> loss = InfoNCE()
    >>> batch_size, num_negative, embedding_size = 32, 48, 128
    >>> query = torch.randn(batch_size, embedding_size)
    >>> positive_key = torch.randn(batch_size, embedding_size)
    >>> negative_keys = torch.randn(num_negative, embedding_size)
    >>> output = loss(query, positive_key, negative_keys)
"""
