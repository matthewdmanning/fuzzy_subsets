import datetime
import itertools
import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn._config import set_config as sk_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import clone
from sklearn.utils import check_X_y

import data_handling.data_cleaning
from data import feature_name_lists
from data_handling.balancing import data_by_groups
from dmso_utils.data_tools import (
    get_interpretable_features,
    load_metadata,
    load_training_data,
)
from qsar_modeling.modeling import scoring
from qsar_modeling.utils import cv_tools, samples

# from utils import cv_tools
pd.set_option("display.precision", 4)
pd.set_option("format.precision", 4)
# np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
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


def mix_sol_all_insol(feature_df, labels, tups, epa_ratio=0.75):
    raise NotImplementedError
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
    interp_X, interp_y = get_interpretable_features()
    print("Original size: {}".format(interp_X.shape))
    valid_inchi, invalid_inchi = data_handling.data_cleaning.check_inchi_only(
        interp_y.index
    )
    if len(invalid_inchi) > 0:
        print("Invalid INCHI keys:")
        print([i for i in invalid_inchi])
    meta = load_metadata()
    g_dict = data_by_groups(interp_y, meta)
    # full_groups_list = dict([(k, pd.Series(data=k, index=v[0].index, name=k)) for k, v in g_dict.items() if k in ['epa_sol', 'epa_in', 'en_in']])
    # meta['en_in'][0].drop(index=meta['en_in'][0].index.intersection(meta['epa_in'][0].index), inplace=True)
    # full_groups_list = dict([(k, v[0]) for k, v in meta.items() if k in ['epa_sol', 'epa_in', 'en_in']])
    unique_X, unique_y = data_handling.data_cleaning.clean_and_check(
        interp_X, interp_y, y_dtype=int
    )
    return unique_X, unique_y


def get_dir(parent_dir, middle=None, suffix=None):
    if middle is None and suffix is None:
        return parent_dir
    if middle is None:
        return "{}{}".format(parent_dir, suffix)
    elif suffix is None:
        return "{}{}".format(parent_dir, middle)


def cv_model_documented(
    input_X,
    input_y,
    cv_model,
    model_name,
    save_dir,
    cv_splitter=None,
    sweight=None,
    **splitter_kw
):
    cv = 0
    dev_score_list, eva_score_list = list(), list()
    for dev_X, dev_y, eva_X, eva_y in cv_tools.split_df(
        input_X, input_y, splitter=cv_splitter, **splitter_kw
    ):
        cv += 1
        cv_dir = "{}cv_{}/".format(save_dir, cv)
        model_dir = "{}{}/".format(cv_dir, model_name)
        if not os.path.isdir(cv_dir):
            os.makedirs(cv_dir, exist_ok=True)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        model_path = "{}model_obj.pkl".format(model_dir)
        cv_dev_path = "{}dev_true.csv".format(cv_dir)
        cv_eva_path = "{}eval_true.csv".format(cv_dir)
        cv_dev_pred_path = "{}dev_pred.csv".format(model_dir)
        cv_eva_pred_path = "{}eval_pred.csv".format(model_dir)
        dev_y.to_csv(cv_dev_path)
        eva_y.to_csv(cv_eva_path)
        cv_model = clone(cv_model).fit(dev_X, dev_y, sample_weight=sweight)
        with open(model_path, "wb") as f:
            pickle.dump(cv_model, f)
        dev_pred = pd.Series(data=cv_model.predict(dev_X), index=dev_y.index)
        eva_pred = pd.Series(data=cv_model.predict(eva_X), index=eva_y.index)
        dev_pred.to_csv(cv_dev_pred_path, index_label="INCHI_KEY")
        eva_pred.to_csv(cv_eva_pred_path, index_label="INCHI_KEY")
        dev_score_dict, eva_score_dict = scoring.score_model(
            cv_model, dev_X, eva_X, dev_y, eva_y
        )
        dev_score_list.append(dev_score_dict)
        eva_score_list.append(eva_score_dict)
        tn, fp, fn, tp = samples.get_confusion_samples((eva_y, eva_pred))
        for iks, s in zip([tn, fp, fn, tp], ["tn", "fp", "fn", "tp"]):
            samples.get_sample_info(iks).to_csv(
                "{}{}_eval_samples.csv".format(model_dir, s), index_label="INCHI_KEY"
            )
    return dev_score_list, eva_score_list


def combined_rus_cv_results(
    feature_df,
    labels,
    model,
    model_params,
    model_name,
    save_dir,
    n_rus=5,
    cv_splitter=None,
    sweight=None,
    **splitter_kw
):
    total_dev_scores, total_eva_scores = list(), list()
    model_inst = model().set_params(**model_params)

    for r in np.arange(n_rus):
        if callable(save_dir):
            get_rus_dir = partial()
        rus_dir = "{}rus_{}/".format(save_dir, r)
        if not os.path.isdir(rus_dir):
            os.makedirs(rus_dir, exist_ok=True)
        rus_state = 1000 * r
        X_under, y_under = RandomUnderSampler(random_state=rus_state).fit_resample(
            feature_df, labels
        )
        rus_dev_scores, rus_eva_scores = cv_model_documented(
            X_under,
            y_under,
            model_inst,
            model_name,
            get_rus_dir,
            cv_splitter,
            sweight,
            **splitter_kw
        )
        assert len(rus_dev_scores) > 0 and len(rus_eva_scores) > 0
        for dl in [rus_dev_scores, rus_eva_scores]:
            for cv_num, d in enumerate(dl):
                d["RUS"] = r
                d["CV"] = cv_num
        total_dev_scores.extend(rus_dev_scores)
        total_eva_scores.extend(rus_eva_scores)
    dev_score_df = scoring.summarize_scores(total_dev_scores)
    eva_score_df = scoring.summarize_scores(total_eva_scores)
    dev_score_path = "{}dev_score_summary.csv".format(save_dir)
    eva_score_path = "{}eval_score_summary.csv".format(save_dir)
    dev_score_df.to_csv(dev_score_path, index_label="Metric")
    eva_score_df.to_csv(eva_score_path, index_label="Metric")
    assert not dev_score_df.empty and not eva_score_df.empty
    return dev_score_df, eva_score_df, total_dev_scores, total_eva_scores


n_features_out = 40
project_dir = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/qsar-modeling-workflow/"
# exp_dir = '{}models/hyperopted-training_curve/'.format(rfe_dir)
feature_dir = "{}models/epa_solubles_hyperparam/".format(project_dir)
exp_dir = "{}models/epa_solubles_final_test/".format(project_dir)
train_data_dir = "{}epa_solubles_atom_type_count/".format(project_dir)
key_path = "{}EPA-only_keys.csv".format(project_dir)
ranked_features_path = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/qsar-modeling-workflow/models/epa_solubles_hyperparam/feature_rankings.csv"

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
"""
if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    
for i in np.arange(5):
     i = i + 1
    rus_dir = '{}rus{}/'.format(exp_dir, i)
    os.makedirs(rus_dir, exist_ok=True)
    if not os.path.isdir(rus_dir) and not os.path.exists(rus_dir):
        print('Not here.')
    else:
        print('{} already exists.'.format(rus_dir))
  """
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
# shuffle_X, shuffle_y = RandomUnderSampler(random_state=rus_random_state).fit_resample(shuffle_epa_X, shuffle_epa_y)
# epa_rus, epa_labels = RandomUnderSampler(random_state=rus_random_state).fit_resample(selected_train_X, combined_y)
# insol_ser = pd.concat([full_groups_list['epa_in'], full_groups_list['en_in']], verify_integrity=True).squeeze()
# full_groups_list['epa_sol'].drop(index=full_groups_list['epa_sol'].index.intersection(insol_ser.index), inplace=True)
# shared_ind = rus_y.index.drop_duplicates().intersection(full_groups.index.drop_duplicates())
# shared_ind = [c for c in rus_y.index.drop_duplicates() if c in full_groups.index.drop_duplicates()]
# Even group sampling
# rus_groups = [g[g.index.intersection(rus_y.index)] for g in full_groups_list.values()]
# group_folds = remove_duplicate_idx(get_quadratic_test_folds(rus_groups))
# cv_method = PredefinedSplit(group_folds)
# print('Input size: {}'.format(shuffle_y.size))
# print('RUS size: {}'.format(rus_y.size))
# rus_y = remove_duplicate_idx(rus_y)
# rus_X = remove_duplicate_idx(rus_X)
# scramble_y = pd.Series(data=np.random.randint(0, 2, size=rus_y.size), index=rus_y.index)
#
# The final inputs into the CV cycle
#
final_X, final_y = shuffle_epa_X, shuffle_epa_y
cv_method = StratifiedKFold
samp_wts = None
splitter_kw = {"shuffle": True, "random_state": 0}
sampling_name = "epa_sol_shuffled-rus_strat-kfold"
hyper_path = "{}models/rehyperparam/{}/".format(project_dir, sampling_name)
os.makedirs(hyper_path, exist_ok=True)
model_id = "random_forest"
rf_kwargs = {"bootstrap": False, "n_jobs": -1, "random_state": 0}
base_model = partial(RandomForestClassifier, **rf_kwargs)
# replica_model = model(n_estimators=100, max_features=5, max_leaf_nodes=200, min_impurity_decrease=5e-5, random_state=0, n_jobs=-1)
param_ser_list = list()
forest_params = ParameterGrid(
    {
        "n_estimators": [10, 50, 100, 250],
        "min_impurity_decrease": [0, 0.00005, 0.0001, 0.0005, 0.001],
        "max_features": [3, 5, 6, 7, 9],
        "max_leaf_nodes": [100, 150, 200, 250, None],
    }
)
summary_dict, params_dict = dict(), dict()
for p_id, (param_set) in enumerate(forest_params):
    params_dict[p_id] = param_set
param_path = "{}param_list.csv".format(hyper_path)
params_df = pd.DataFrame.from_dict(params_dict, orient="index")
params_df.to_csv(param_path, index_label="set_index")
for p_id, (param_set) in enumerate(forest_params):
    print(param_set)
    param_dir = "{}{}_param_{}/".format(hyper_path, model_id, p_id)
    get_dir = partial(
        get_dir(parent_dir=hyper_path, suffix="{}_param_{}/".format(model_id, p_id))
    )
    os.makedirs(param_dir, exist_ok=True)
    hyper_dev_df, hyper_eva_df, all_dev_scores, all_eva_scores = (
        combined_rus_cv_results(
            final_X,
            final_y,
            base_model,
            param_set,
            model_name=model_id,
            save_dir=get_dir,
            sweight=samp_wts,
        )
    )
    param_ser_list.append(pd.Series(param_set, name=p_id))
    hyper_dev_df.add_suffix(suffix="_dev")
    hyper_eva_df.add_suffix(suffix="_eval")
    combo_scores = pd.concat([hyper_dev_df, hyper_eva_df], axis=1)
    combo_scores.to_csv("{}score_summary.csv".format(param_dir))
    summary_dict[p_id] = combo_scores["Balanced Accuracy"].squeeze()
crit_df = pd.DataFrame.from_dict(summary_dict, orient="index").sort_values(
    by="GeoMean_eval", ascending=False
)
params_df = pd.concat(param_ser_list)
crit_path = "{}{}_criteria_scores.csv".format(hyper_path, model_id)
param_path = "{}{}_param_list.csv".format(hyper_path, model_id)
crit_df.to_csv(crit_path, index_label="set_index")
params_df.to_csv(param_path, index_label="set_index")
