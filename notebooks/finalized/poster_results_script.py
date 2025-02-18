import datetime
import itertools
import os
import pickle

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn._config import set_config as sk_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import *
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_X_y

import cv_tools
import data_handling.data_cleaning
from data import feature_name_lists
from qsar_modeling.data_handling.data_tools import (
    get_interpretable_features,
    load_metadata,
    load_training_data,
)
from qsar_modeling.utils import samples

# from utils import cv_tools
pd.set_option("display.precision", 4)
pd.set_option("format.precision", 4)
np.set_printoptions(formatter={"float": "{: 0.4f}".format})
sk_config(transform_output="pandas", display="text", print_changed_only=True)

start = datetime.datetime.now()


def load_sample_keys(filepath):
    sample_list = list()
    with open(filepath, "r", encoding="utf-8") as f:
        [sample_list.extend(s.split(",")) for s in f.readlines()]
        sample_list = [s.partition("\\")[0] for s in sample_list if s.endswith("-N")]
        # csv.reader(f)
    return sample_list


def load_feature_rankings(filepath, threshold=None):
    ranking_ser = (
        pd.read_csv(filepath).set_index(keys="Features", inplace=False).squeeze()
    )  # , index_col="Features")
    if threshold is not None:
        ranking_ser.drop(ranking_ser[ranking_ser > threshold].index, inplace=True)
    return ranking_ser.index


def brute_force_importance_rf_clf(
    feature_df, labels, clf, n_features_out, n_jobs=-2, step_size=1, **model_kwargs
):
    eliminator = RFE(
        estimator=clf, n_features_to_select=n_features_out, step=step_size
    ).fit(feature_df, y=labels)
    brute_features = pd.Series(
        eliminator.ranking_, index=feature_df.columns.tolist()
    ).sort_values()
    return brute_features, eliminator


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


def get_epa_sol_all_insol(feature_df, labels, tups):
    # insol_samples = pd.concat([tups['epa_in'][0], tups['en_in'][0]]).index.intersection(train_insols)
    # train_sols = labels[labels == 1].index
    # sol_samples = tups['epa_sol'][0].index.intersection(train_sols)
    en_in = tups["en_in"][0][
        [c for c in tups["en_in"][0].index if c not in tups["epa_in"][0].index]
    ]
    all_samples_ind = pd.concat(
        [tups["epa_in"][0], en_in, tups["epa_sol"][0]]
    ).index.intersection(feature_df.index)
    all_samples = labels[all_samples_ind]
    all_samples[all_samples == "Insoluble"] = 0
    all_samples[all_samples == "Soluble"] = 1
    select_y = labels[all_samples.index]
    select_X = feature_df.loc[all_samples.index]
    # assert not select_X.isna().any()
    return select_X, select_y


def sample_observations_by_source(
    groups_observations, freq_list, replace=False, random_state=None
):
    if all([0.0 <= p <= 1.0 for p in freq_list]):
        freq_list = [int(p * len(n)) for n, p in zip(groups_observations, freq_list)]
    yield itertools.repeat(
        [
            pd.concat(
                [
                    obs.sample(n=freq, random_state=random_state, replace=replace)
                    for obs, freq in zip(groups_observations, freq_list)
                ]
            )
        ]
    )


def mix_insol_all_insol(feature_df, labels, tups, epa_ratio=0.75):
    train_insols = labels[labels == 0].index
    insol_samples = pd.concat([tups["epa_in"][0], tups["en_in"][0]]).index.intersection(
        train_insols
    )
    train_sols = labels[labels == 1].index
    epa_sol_samples = tups["epa_sol"][0].index.intersection(train_sols)
    en_sol_samples = tups["en_sol"][0]
    select_y = labels[all_samples]
    select_X = feature_df.loc[all_samples][feature_df.columns]
    return select_X, select_y


def data_by_groups(labels, group_dict):
    ind_dict = dict()
    for k, v in group_dict.items():
        if type(v) is list or type(v) is list:
            ind_dict[k] = v[0].index.intersection(labels.index)
        elif type(v) is pd.DataFrame or type(v) is pd.Series:
            ind_dict[k] = v.index.intersection(labels.index)
    return ind_dict


def quadratic_splits(grouped_sers, n_splits=5):
    # Takes separate groups (each in separate Series), gets n_splits splits, and yields indices of test set.
    test_list, train_list = list(), list()
    indices = [s.copy().index.tolist() for s in grouped_sers]
    [np.random.shuffle(idx) for idx in indices]
    nested_splits = list()
    for ind in indices:
        spaces = np.linspace(0, len(ind) - 1, num=n_splits + 1, dtype=int)
        nested_splits.append(
            [ind[int(a) : int(b)] for a, b in itertools.pairwise(spaces)]
        )
    return nested_splits


def get_quadratic_test_folds(grouped_sers, n_splits=5):
    fold_list = list()
    quad_splits = quadratic_splits(grouped_sers, n_splits)
    # print(quad_splits)
    for qx in quad_splits:
        for fold_id, idxs in enumerate(qx):
            fold_list.append(pd.Series(data=[fold_id for _ in idxs], index=idxs))
    return pd.concat(fold_list)


def drop_duplicate_indices(df):
    if df.index.has_duplicates:
        # pd.Index.duplicated()
        print("Index duplicates found. Starting with {}".format(df.index.size))
        drop_list = [i for i, x in enumerate(df.index.duplicated()) if x is True]
        new_df = df.drop(index=drop_list)
        print("Ending with {}".format(new_df.index.size))
    else:
        new_df = df
    return new_df


def stepwise_duplicate(df):
    new_dict = dict()
    for k, v in df.items():
        if k in new_dict.keys():
            continue
        else:
            new_dict[k] = v
    if type(df) is pd.DataFrame:
        new_df = pd.DataFrame.from_dict(new_dict)
    elif type(df) is pd.Series:
        new_df = pd.Series(data=new_dict, name=df.name)
    return new_df


def get_notebook_data():
    # Select EPA soluble/combined insoluble dataset.
    meta = load_metadata(desc=False)
    # meta_dict = dict(zip(['epa_sol', "epa_in", "en_in", "en_sol"], meta))
    interp_X, interp_y = get_interpretable_features()
    valid_inchi, invalid_inchi = data_handling.data_cleaning.check_inchi_only(
        interp_y.index
    )
    if len(invalid_inchi) > 0:
        print("Invalid INCHI keys:")
        print([i for i in invalid_inchi])

    # Drop duplicates
    print("Original size: {}".format(interp_X.shape))
    g_dict = data_by_groups(interp_y, meta)
    full_groups_list = dict(
        [
            (k, pd.Series(data=k, index=v[0].index, name=k))
            for k, v in g_dict.items()
            if k in ["epa_sol", "epa_in", "en_in"]
        ]
    )
    meta["en_in"][0].drop(
        index=meta["en_in"][0].index.intersection(meta["epa_in"][0].index), inplace=True
    )
    full_groups_list = dict(
        [(k, v[0]) for k, v in meta.items() if k in ["epa_sol", "epa_in", "en_in"]]
    )
    new_idx = (
        stepwise_duplicate(pd.concat(full_groups_list.values(), sort=True))
        .squeeze()
        .index.intersection(interp_y.index)
    )
    valid_idx = new_idx.intersection(valid_inchi)
    unique_y = interp_y[valid_idx]
    unique_X = interp_X.loc[valid_idx]
    check_X_y(unique_X, unique_y)
    return unique_X, unique_y


n_features_out = 40
project_dir = "//"
# exp_dir = '{}models/hyperopted-training_curve/'.format(project_dir)
feature_dir = "{}models/epa_solubles_hyperparam/".format(project_dir)
exp_dir = "{}models/epa_solubles_final_test/".format(project_dir)
train_data_dir = "{}epa_solubles_atom_type_count/".format(project_dir)
key_path = "{}EPA-only_keys.csv".format(project_dir)
ranked_features_path = "/models/epa_solubles_hyperparam/feature_rankings.csv"

combined_X, combined_y = get_notebook_data()
if not os.path.isfile(key_path):
    combined_y.to_csv(key_path)
# full_groups = full_groups[full_groups.index.drop_duplicates()]
# Determine feature set.
# ranked_features_path = '{}feature_rankings.csv'.format(feature_dir)
ranked_features = pd.DataFrame([])
if os.path.isfile(ranked_features_path):
    ranked_features = load_feature_rankings(ranked_features_path, threshold=1)
    print("Ranked features loaded: \n{}".format(len(ranked_features)))
    selected_train_X = combined_X[ranked_features]
elif not os.path.isfile(
    ranked_features_path
):  # or (ranked_features.empty and selected_train_X.isna().astype(int).sum().sum() > 0):
    feature_method = "brute"
    if feature_method == "brute":
        print("Brute force feature selection started.")
        # train_X, train_y = get_epa_sol_all_insol(combined_X, combined_y, meta_dict)
        rfc = RandomForestClassifier(random_state=0, oob_score=True, n_jobs=-2)
        lr = LogisticRegressionCV(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            scoring=make_scorer(matthews_corrcoef),
            n_jobs=-2,
            random_state=0,
            cv=5,
            l1_ratios=[0.25],
        )
        ranked_features, rfe_model = brute_force_importance_rf_clf(
            combined_X, combined_y, clf=rfc, n_features_out=n_features_out
        )
        ranked_features.to_csv(
            ranked_features_path, index_label="Features"
        )  # , float_format='%.4f')
        selected_train_X = rfe_model.transform(combined_X)
        # print([c for c in selected_train_X.columns if 'e-state' in c.lower()])
    elif feature_method == "list":
        print("Using E-count descriptors")
        ecounts = feature_name_lists.get_estate_counts(combined_X.columns.tolist())
        selected_train_X = combined_X[ecounts]
# Get split indices.

if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
log_mcc_list, log_bac_list, log_auc_list = list(), list(), list()
for i in np.arange(5):
    i = i + 1
    rus_dir = "{}rus{}/".format(exp_dir, i)
    os.makedirs(rus_dir, exist_ok=True)
    if not os.path.isdir(rus_dir) and not os.path.exists(rus_dir):
        print("Not here.")
    else:
        print("{} already exists.".format(rus_dir))
    check_X_y(selected_train_X, combined_y)
    """
    with open('{}X_train.pkl'.format(exp_dir), 'wb') as f:
        pickle.dump(selected_train_X, f)
    with open('{}y_train.pkl'.format(exp_dir), 'wb') as f:
        pickle.dump(combined_y, f)
    with open('{}grouped_by_source.pkl'.format(exp_dir), 'wb') as f:
        pickle.dump(full_groups_list, f)
        """
    # rand_idx = interp_y.index.tolist()
    # random.shuffle(rand_idx)
    # shuffle_X = interp_X[ranked_features].loc[rand_idx]
    # shuffle_y = interp_y[rand_idx]
    epa_rand = combined_y.sample(frac=1.0, random_state=0).index
    shuffle_epa_X = selected_train_X.loc[epa_rand]
    shuffle_epa_y = combined_y[epa_rand]
    rus_random_state = 1000 * i
    shuffle_X, shuffle_y = RandomUnderSampler(
        random_state=rus_random_state
    ).fit_resample(shuffle_epa_X, shuffle_epa_y)
    epa_rus, epa_labels = RandomUnderSampler(
        random_state=rus_random_state
    ).fit_resample(selected_train_X, combined_y)

    # insol_ser = pd.concat([full_groups_list['epa_in'], full_groups_list['en_in']], verify_integrity=True).squeeze()
    # full_groups_list['epa_sol'].drop(index=full_groups_list['epa_sol'].index.intersection(insol_ser.index), inplace=True)
    # shared_ind = rus_y.index.drop_duplicates().intersection(full_groups.index.drop_duplicates())
    # shared_ind = [c for c in rus_y.index.drop_duplicates() if c in full_groups.index.drop_duplicates()]
    # Even group sampling
    # rus_groups = [g[g.index.intersection(rus_y.index)] for g in full_groups_list.values()]
    # group_folds = remove_duplicate_idx(get_quadratic_test_folds(rus_groups))
    # cv_method = PredefinedSplit(group_folds)
    rus_X = shuffle_X
    rus_y = shuffle_y
    print("Input size: {}".format(shuffle_y.size))
    print("RUS size: {}".format(rus_y.size))
    rus_y = stepwise_duplicate(rus_y)
    rus_X = stepwise_duplicate(rus_X)
    scramble_y = pd.Series(
        data=np.random.randint(0, 2, size=rus_y.size), index=rus_y.index
    )
    #
    # The final inputs into the CV cycle
    #
    input_X, input_y = epa_rus, epa_labels
    cv_method = StratifiedKFold
    sweight = None
    splitter_kw = {"shuffle": True, "random_state": 0}
    model_name = "rf_epa_sol_only_rus-{}_strat_trained".format(rus_random_state)
    hyper_path = "{}grid_search/".format(rus_dir)
    model = RandomForestClassifier
    cv = 0
    from sklearn.pipeline import clone

    replica_model = model(
        n_estimators=100,
        max_features=5,
        max_leaf_nodes=200,
        min_impurity_decrease=5e-5,
        random_state=0,
        n_jobs=-1,
    )
    for dev_X, dev_y, eva_X, eva_y in cv_tools.split_df(
        input_X, input_y, splitter=cv_method
    ):
        cv += 1
        replica_dir = "{}replica_cv{}/".format(rus_dir, i)
        model_dir = "{}model_name/".format(replica_dir)
        if not os.path.isdir(replica_dir):
            os.mkdir(replica_dir)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_path = "{}model_obj.pkl".format(model_dir)
        cv_dev_path = "{}dev_true.csv".format(replica_dir)
        cv_eva_path = "{}eval_true.csv".format(replica_dir)
        cv_dev_pred_path = "{}dev_pred.csv".format(model_dir)
        cv_eva_pred_path = "{}eval_pred.csv".format(model_dir)
        dev_y.to_csv(cv_dev_path)
        eva_y.to_csv(cv_eva_path)
        cv_model = clone(replica_model).fit(dev_X, dev_y, sample_weight=sweight)
        with open(model_path, "wb") as f:
            pickle.dump(replica_model, f)
        dev_pred = pd.Series(data=cv_model.predict(dev_X), index=dev_y.index)
        eva_pred = pd.Series(data=cv_model.predict(eva_X), index=eva_y.index)
        dev_pred.to_csv(cv_dev_pred_path, index_label="INCHI_KEY")
        eva_pred.to_csv(cv_eva_pred_path, index_label="INCHI_KEY")
        dev_conf_mat = confusion_matrix(
            dev_y, dev_pred, normalize="true", sample_weight=sweight
        )
        eva_conf_mat = confusion_matrix(
            eva_y, eva_pred, normalize="true", sample_weight=sweight
        )
        print("Dev Confusion Matrix:\n{}".format(dev_conf_mat))
        print("Eval Confusion Matrix:\n {}".format(eva_conf_mat))
        print("MCC: Dev score, Eval score")
        print(
            matthews_corrcoef(dev_y, dev_pred, sample_weight=sweight),
            matthews_corrcoef(eva_y, eva_pred, sample_weight=sweight),
        )
        print("BAC: Dev score, Eval score")
        print(
            balanced_accuracy_score(dev_y, dev_pred, sample_weight=sweight),
            balanced_accuracy_score(eva_y, eva_pred, sample_weight=sweight),
        )
        print("AUC: Dev score, Eval score")
        print(
            roc_auc_score(dev_y, dev_pred, sample_weight=sweight),
            roc_auc_score(eva_y, eva_pred, sample_weight=sweight),
        )
        # tn, fp, fn, tp = samples.get_confusion_samples((dev_y, dev_pred))
        tn, fp, fn, tp = samples.get_confusion_samples((eva_y, eva_pred))
        # conf_path_list = ['{}_eval_{}'.format(model_dir)]
        [
            samples.get_sample_info(iks).to_csv(
                "{}{}_eval_samples.csv".format(model_dir, s), index_label="INCHI_KEY"
            )
            for iks, s in zip([tn, fp, fn, tp], ["tp", "fp", "fn", "tp"])
        ]
    """
    Grid search used to get poster results
    forest_params = ParameterGrid(
        {'n_estimators': [10, 50, 100], 'min_impurity_decrease': [0, 0.00005, 0.0001, 0.00025],
         'max_features': [3, 5], 'max_leaf_nodes': [100, 150, 200, 250], 'bootstrap': [False]}).param_grid
    search_party = GridSearchCV(model(), forest_params, scoring=make_scorer(matthews_corrcoef),
                                cv=cv_method, n_jobs=-1, error_score='raise',
                                return_train_score=True,
                                verbose=0).fit(X=rus_X, y=rus_y)
    pprint.pp(search_party.best_params_)
    pprint.pp(search_party.best_score_)
    clf = search_party.best_estimator_

    try:
        with open('{}gridsearch_obj.pkl'.format(hyper_path), 'wb') as f:
            pickle.dump(search_party, f)
    except:
        with open('{}best_model.pkl'.format(hyper_path), 'wb') as f:
            pickle.dump(search_party.best_estimator_, f)
    """
    """
    print("CV Optimized Random Forest Scoring")
    for train_X, train_y, test_X, test_y in cv_tools.split_df(epa_rus, epa_labels, **splitter_kw):
        print('Train value counts: \n{}'.format(train_y.value_counts()))
        scale = RobustScaler().fit(train_X)
        train_X = scale.transform(train_X)
        test_X = scale.transform(test_X)
        clf = LogisticRegressionCV(max_iter=5000, n_jobs=-1, class_weight='balanced',
                                   scoring=make_scorer(matthews_corrcoef),
                                   l1_ratios=[0.5, 0.6, 0.7], solver='saga', penalty='elasticnet')
        print(clf.l1_ratio_)
        print(clf.C_)        
        # alldata_X = interp_X[ranked_features].drop(index=train_X.index)
        alldata_X = selected_train_X.drop(index=train_X.index)
        alldata_y = combined_y.drop(index=train_X.index)
        clf.fit(train_X, train_y)
        train_pred = clf.predict(train_X)
        test_pred = clf.predict(test_X)
        interp_pred = clf.predict(alldata_X)
        log_mcc_list.append((matthews_corrcoef(train_y, train_pred), matthews_corrcoef(test_y, test_pred),
                             matthews_corrcoef(alldata_y, interp_pred)))
        log_bac_list.append((balanced_accuracy_score(train_y, train_pred), balanced_accuracy_score(test_y, test_pred),
                             balanced_accuracy_score(alldata_y, interp_pred)))
        log_auc_list.append((roc_auc_score(train_y, train_pred),
                             roc_auc_score(test_y, test_pred), roc_auc_score(alldata_y, interp_pred)))
summary = dict()
for scorename, scorelist in zip(["MCC", "BAC", "ROC-AUC"], [log_mcc_list, log_bac_list, log_auc_list]):
    trains = [a[0] for a in scorelist]
    tests = [a[1] for a in scorelist]
    alldata = [a[2] for a in scorelist]
    # print(trains)
    # print(tests)
    for setname, scoreset in zip(["Train", "Test", "AllData"], [trains, tests, alldata]):
        sumname = '{}({})'.format(scorename, setname)
        summary[sumname] = (
            np.mean(scoreset), np.std(scoreset), np.min(scoreset), np.median(scoreset), np.max(scoreset))
[print('{}: {}'.format(k, v)) for k, v in summary.items()]
summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Mean', 'StDev', 'Min', 'Median', 'Max'])
summary_df.to_csv('{}{}_alldata_score_summary.csv'.format(exp_dir, model_name))
exit()
# dev_summary, eva_summary, dev_dict, eva_dict = scoring.cv_model_generalized(clf, rus_X, rus_y, score_dir="{}logit/".format(rus_dir), scaler=RobustScaler())
# print(dev_summary)
# print(eva_summary)
# check_consistent_length(rus_y, rus_X)
# check_consistent_length(rus_y, group_folds)
# check_consistent_length(rus_X, group_folds)
# Hyperparameterize
for i in [0]:

    print(rus_X.head())
    print(rus_y.head())

    #  {'n_estimators': [100, 500], 'max_leaf_nodes': [250, 300], 'max_features': [3, 5, 10]}])
    # ,{'n_estimators': [10000], 'bootstrap': [False], 'ccp_alpha': [0.0001, 0.001, 0.005]}])

    for train_X, train_y, test_X, test_y in cv_tools.split_df(rus_X, rus_y):
        clf = model().set_params(**search_party.best_params_)
        clf.fit(train_X, train_y)
        train_pred = clf.predict(train_X)
        test_pred = clf.predict(test_X)
        print(matthews_corrcoef(train_y, train_pred), matthews_corrcoef(test_y, test_pred))
        print(balanced_accuracy_score(train_y, train_pred), balanced_accuracy_score(test_y, test_pred))
        print(roc_auc_score(train_y, train_pred, average='weighted'),
              roc_auc_score(test_y, test_pred, average='weighted'))
    fitted = search_party.best_estimator_
    cv_dev_score_dict = dict()
    cv_eval_score_dict = dict()
    depth_list, node_list = list(), list()
    for t in search_party.best_estimator_.estimators_:
        depth_list.append(t.get_depth())
        node_list.append(t.get_n_leaves())
        # print(search_party.best_estimator_.tree_.max_depth())
    with open('{}cv_results_headline.txt', 'w') as f:
        f.write('Best Parameters: {}\nResults: {}'.format(search_party.best_params_, search_party.best_index_))
        f.write('Tree depths: \nMean: {} \nStd Dev: {} \nMin: {} \nMedian: {} \nMax: {}'.format(
            np.mean(depth_list), np.std(depth_list), np.min(depth_list), np.median(depth_list), np.max(depth_list)))
        f.write('Number of leaves:: \nMean {}: \nStd Dev: {} \nMin: {} \nMedian: {} \nMax: {}'.format(
            np.mean(node_list), np.std(node_list), np.min(node_list), np.median(node_list), np.max(node_list)))
    # Needs lots more coding for full score summary.
    # dev_score_dict, eva_score_dict = score_model(fitted, shuffle_X, fit_eva_X, rus_dev_y, rus_eva_y)
    """
    """
    with open('{}hyper-dev_scores.csv'.format(hyper_path), 'w') as f:
        csv.writer(f).writerow(cv_dev_score_dict.items())
    with open('{}hyper-eval_scores.csv'.format(hyper_path), 'w') as f:
        csv.writer(f).writerow(cv_eval_score_dict.items())    
    lcurve = learning_curve(RandomForestClassifier(), rus_X, rus_y,
                            n_jobs=-1, #  scoring=make_scorer(matthews_corrcoef),
                            fit_params=best.get_params())[0]
    print(lcurve)    
    best = RandomForestClassifier(max_features=5, max_leaf_nodes=250, min_impurity_decrease=5e-05, n_estimators=100, bootstrap=False, n_jobs=-1)
    fitted = best.fit(rus_X, rus_y)
    """
    """
    from sklearn.metrics import DetCurveDisplay
    import matplotlib.pyplot as plt

    det_path = 'det_curve_rus'.format(rus_dir)
    fig, ax = plt.subplots(figsize=(5, 6), dpi=600)
    det = DetCurveDisplay.from_estimator(model(), X=rus_X, y=rus_y, pos_label=0,
                                         name='Detection Error Curve: Random Forest',
                                         ax=ax, fit_params=search_party.best_params_)
    det.plot(name='Detection Error Curve: RF (Undersampled)', ax=ax)
    print(det.fnr, det.fpr)
    plt.savefig(fname='{}.svg'.format(det_path), transparent=True)
    whole_stub = '{}learning_curve_not_rus'.format(rus_dir)
    # whole_lcd, _, __ = scoringlearn_curve(best, 'Random Forest (All Data)', shuffle_X, shuffle_y, fname_stub='{}all_data_lc'.format(rus_dir))
    fig, ax = plt.subplots(figsize=(5, 6), dpi=600)
    rus_lcd, _, __ = scoring.learn_curve(best, 'Random Forest (Undersampled)', rus_X, rus_y,
                                         fname_stub='{}rus_lc'.format(rus_dir))

    continue
    # cv_model_generalized(best, X=rus_X, y=rus_y, cv_inst=PredefinedSplit(group_folds), score_dir=rus_dir)
    std_scaler = StandardScaler().fit(rus_X)
    fit_dev_X = std_scaler.fit_transform(rus_X)
    fit_eva_X = std_scaler.transform(rus_eva_X)
    fig, ax = plt.subplots(figsize=(5, 6), dpi=600)
    # train_curve_ser = pd.Series(lcurve[1], index=lcurve[0], name='Test Scores')
    # test_curve_ser = pd.Series(lcurve[2], index=lcurve[0], name='Train Scores')
    # test_curve_ser = pd.Series(lcurve.test_scores, index=train_sizes_abs, name='Train Scores')
    continue

    import seaborn as sns

    fig, ax = plt.subplots()
    train_plot = sns.lineplot(train_curve_ser)
    test_plot = sns.lineplot(test_curve_ser)
    fig.savefig(lc_path)
    [print(c) for c in lcurve]
    continue

    dev_keys, eva_keys = list(), list()
    for j in np.arange(5):
        cv_dir = '{}cv{}/'.format(rus_dir, j + 1)
        dev_key_path = '{}dev_keys.csv'.format(cv_dir, i)
        eval_key_path = '{}dev_keys.csv'.format(cv_dir, i)
        if not os.path.isdir(cv_dir):
            os.mkdir(cv_dir)
        if os.path.isfile(dev_key_path) and os.path.isfile(eval_key_path):
            dev_keys.append(load_sample_keys(dev_key_path))
            eva_keys.append(load_sample_keys(eval_key_path))
        else:
            dev_ind, eva_ind = [(d, e) for d, e in StratifiedKFold().split(rus_X, rus_y)][j]
            dev_keys.append(rus_y.iloc[dev_ind].index)
            eva_keys.append(rus_y.iloc[eva_ind].index)
            with open('{}dev_keys.csv'.format(cv_dir), 'w') as f:
                csv.writer(f).writerow(rus_y.iloc[dev_ind].index)
            with open('{}eval_keys.csv'.format(cv_dir), 'w') as f:
                csv.writer(f).writerow(rus_y.iloc[eva_ind].index)
    continue
    # Manual Cross-val
    for i, (dev, eva) in enumerate(zip(dev_keys, eva_keys)):
        cv = i + 1
        cv_dir = '{}cv{}/'.format(rus_dir, cv)
        print('Starting CV run at {}'.format(datetime.datetime.now()))
        rus_dev_X, rus_dev_y = selected_train_X.loc[dev], train_y[dev]
        rus_eva_X, rus_eva_y = selected_train_X.loc[eva], train_y[eva]
        print('Number of insoluble:soluble training compounds selected: {}:{}'.format(rus_dev_y[rus_dev_y == 0].shape,
                                                                                      rus_dev_y[rus_dev_y == 1].shape))

        std_scaler = StandardScaler().fit(rus_dev_X)
        continue
        fit_dev_X = std_scaler.transform(rus_dev_X)
        fit_eva_X = std_scaler.transform(rus_eva_X)
        logistic_cv = partial(LogisticRegressionCV, max_iter=1000, solver='newton-cg')
        positive_lab = rus_dev_y.unique()[0]
        # ('logistic-cv', logistic_cv),
        for model_name, model in [('optimized_forest', fitted.params)]:
            model_path = '{}{}/'.format(cv_dir, model_name)
            if not os.path.isdir(model_path):
                os.mkdir(model_path)
            if model_name == 'logistic-cv':
                fitted = model(n_jobs=-1).fit(fit_dev_X, rus_dev_y)
            # if model_name != 'logistic-cv':
            # fitted = model(bootstrap=False, max_leaf_nodes=200, n_estimators=10000, n_jobs=-1).fit(fit_dev_X, rus_dev_y)
            dev_scores, eva_scores = scoring.score_model(fitted, fit_dev_X, fit_eva_X, rus_dev_y, rus_eva_y)

            for score_name, score_obj in prob_scores.items():
                cv_dev_score_dict[score_name] = score_obj(rus_dev_y, select_dev_pred)
                cv_eval_score_dict[score_name] = score_obj(rus_eva_y, select_eva_pred)
                # cv_dev_score_dict[score_name] = score_obj(rus_dev_y, select_dev_prob)
                # cv_eval_score_dict[score_name] = score_obj(rus_eva_y, select_eva_prob)

            with open('{}dev_proba.csv'.format(model_path), 'w') as f:
                csv.writer(f).writerow(eliminator.estimator_.predict_proba(rus_dev_y))
            with open('{}eval_proba.csv'.format(model_path), 'w') as f:
                csv.writer(f).writerow(eliminator.estimator_.predict_proba(rus_eva_y))
    end = datetime.datetime.now()
    exec_time = end - start
    print('One CV run took {}'.format(exec_time))
    exit()
"""
