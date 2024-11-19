import logging
import os.path
import os.path
import pickle
import pprint
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
# from chefboost import Chefboost
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from scipy.special import softmax
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils import check_X_y, compute_class_weight, compute_sample_weight

from constants import names, paths, run_params
from feature_selection.importance import extra_trees_gini_importances
from feature_selection.vif import calculate_vif, repeated_stochastic_vif
from modeling.score_report import plot_results
from qsar_modeling.feature_selection import FeatureSelector

# Load Data & Options
logger = logging.getLogger("dmso_logger.working_nb")
plt.style.use("ggplot")
checker = partial(
    check_X_y,
    accept_sparse=True,
    ensure_min_features=10,
    ensure_min_samples=30000,
    y_numeric=True,
)
reuse_fs = False
fs_loaded = False
ci, use_vif, isolate = False, False, False
fs, loaded_cov = None, None


def train_models(train_X, train_y, total_meta_df):
    solubility_vec = total_meta_df["DMSO_SOLUBILITY"].squeeze()[train_X.index]
    classes = np.unique(train_y)
    class_wts_arr = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_y
    )
    # class_wts_normed = class_wts_arr / np.min(class_wts_arr)
    sol_weighting = dict(zip(classes, class_wts_arr))
    # cweights = solubility_vec.copy(deep=True).astype(float).map(sol_weighting)
    sample_wts = pd.Series(
        data=compute_sample_weight(class_weight="balanced", y=train_y),
        index=train_y.index,
    )
    # pprint.pp(solubility_vec)
    dataset_vec = total_meta_df["DATA_SOURCE"].squeeze()[train_X.index]
    source_converter = dict(
        zip(dataset_vec.unique(), list(range(dataset_vec.unique().size)))
    )
    samp_wts = compute_sample_weight(class_weight=sol_weighting, y=train_y)
    # weights = {'class_wts': sol_weighting, 'sample_wts': sample_wts}

    if os.path.isfile(paths.cov_mat) and os.path.getsize(paths.cov_mat) > 10000:
        with open(paths.cov_mat, "r+b") as f:
            loaded_cov = pickle.load(f, encoding="utf-8")
            if loaded_cov is not None and not loaded_cov.empty:
                pprint.pp(
                    "Covariance matrix has been loaded with shape: {}".format(
                        loaded_cov.shape
                    )
                )
            else:
                pprint.pp("Covariance matrix failed to load properly!")
                loaded_cov = None
    if reuse_fs and not run_params.debug and os.path.isfile(feature_frame_path):
        with open(feature_frame_path, "r+b") as f:
            try:
                fs = pickle.load(f, encoding="utf-8")
            except EOFError:
                pprint.pp("FeatureFrame file is invalid or corrupt.")
                # os.remove(path=feature_frame_path)
                fs_loaded = False
        if fs is not None and type(fs) is FeatureSelector.FeatureSelectorTrainer:
            fs_loaded = True
        else:
            pprint.pp("Content from {} is not valid!!!".format(feature_frame_path))
    if not fs_loaded:
        fs = FeatureSelector.FeatureSelectorTrainer()
    if fs is None or type(fs) is not FeatureSelector.FeatureSelectorTrainer:
        raise ValueError("FeatureSelectionTrainer was not instantiated!!!")
    fs.fit(X=train_X, y=train_y, cov_mat=loaded_cov)
    # print('THESE ARE THE COUNTS FOR UNIQUE LABELS!!!')
    insol_label, sol_label = fs.y.unique()[0], fs.y.unique()[1]
    # from modeling.preprocessor import subsample
    # subsample(fs=fs, sparse_feats=fs.X.sparse)
    # fs.X.check_duplicate_feats()
    # Cardinality
    std_cut_feats = fs.X.feat_frame[fs.X.feat_frame.std(axis=1, ddof=1) > 0.1]
    assert not fs.X.feat_frame.empty and not std_cut_feats.empty
    if (
        reuse_fs
        and not run_params.debug
        and not fs_loaded
        and not fs.X.feat_frame.empty
    ):
        with open(feature_frame_path, "w+b") as f:
            try:
                pickle.dump(fs, f)
            except pickle.PicklingError:
                pprint.pp("Pickling failed!")
    # Isolation Forest found no outliers for the command below.
    # isofor, pr = fs.isolate_observations(n_samples=0.1, n_est=0.05 * fs.X.feat_frame.shape[1])
    #  predicted = pd.DataFrame(pr, index=fs.X.feat_frame.index)
    # outliers = predicted[predicted == -1].index
    # fs.X.feat_frame = fs.X.feat_frame.drop(index=outliers, inplace=True)
    # pprint.pp('Outliers based on Isolation Forest method:\n'.format(outliers))
    if fs.X.cov_mat is None or (
        (type(fs.X.cov_mat) is not pd.DataFrame or fs.X.cov_mat.empty)
        or type(fs.X.cov_mat) is not np.ndarray
    ):
        # print('Not using Cov?')
        pass
        # logger.info('Covariance matrix could not be calculated!!!')
    elif (
        not os.path.isfile(paths.cov_mat)
        and (os.path.getsize(paths.cov_mat) == 0)
        and loaded_cov is None
    ):
        with open(paths.cov_mat, "w+b") as f:
            pprint.pp("Saving covariance matrix.")
            pickle.dump(obj=fs.X.cov_mat, file=f)
    # else:
    #    logger.warning(pprint.pformat('Covariance matrix could not be calculated.'))
    #

    # Eliminate Uninterpretable Descriptors
    # logger.info(names.uninterpretable)
    # bad_feats = pd.Index([c for c in std_cut_feats if any([u in c.lower() for u in names.uninterpretable]) and 'number'
    # not in c.lower() and 'count' not in c.lower()])
    bad_feats = pd.Index(
        [
            c
            for c in std_cut_feats
            if any([u.lower() in c.lower() for u in names.uninterpretable])
        ]
    )
    # bad_feats = pd.Index([c for c in std_cut_feats if c not in good_feats.tolist()])
    good_feats = pd.Index([c for c in std_cut_feats if c not in bad_feats.tolist()])
    """
    if not os.path.isdir(FEATS_DIR):
        os.makedirs(FEATS_DIR)
    with open('{}interpretable_features.csv'.format(FEATS_DIR), 'w', encoding='utf-8') as f:
        for x in good_feats.tolist():
            f.write('{}\n'.format(x))
    with open('{}uninterpretable_features.csv'.format(FEATS_DIR), 'w', encoding='utf-8') as f:
        for x in bad_feats.tolist():
            f.write('{}\n'.format(x))
    """

    # run_cv_search(fs=fs, use_feats=good_feats, cv_dir=PIPELINE_DIR, score_dict=score_dict)
    # bad_feats.to_csv('{}uninterpretable_features.csv'.format(os.environ.get('STATS_DIR')))
    # logger.info('Difficult to interpret features (left out of further selection): {}'.format(bad_feats.size))
    # logger.info(pprint.pformat(bad_feats, compact=True))

    # from ITMO_FS.filters.multivariate.measures import (CMIM, mutual_information, conditional_mutual_information,joint_mutual_information, matrix_mutual_information)

    def MRMR_proto(
        feats_df,
        sample_subset,
        n_feats_out,
        initial_feats=None,
        n_intial_feats=None,
        subsampler=RandomUnderSampler,
    ):
        if not intial_feats and n_intial_feats is not None:
            intial_feats = sample_subset.sample(n=n_intial_feats)
            pd.Series.sample()
        selected_feats = {initial_feats}
        while len(selected_feats) < n_feats_out:
            chunk_size = max(max_chunk, len(sample_subset))
            candidates = sample_subset.sample(n=chunk_size)
            sample_subset.drop(candidates)
            fight_pool = selected_feats.copy()
            fight_pool.update(candidates)
            MRMR(selected_feats.index)

    def mi_select(
        feats_df,
        target,
        feature_names_in,
        n_feats_out,
        save_dir,
        k=5,
        cv=30,
        n_pca_in=15,
        n_pca_out=15,
    ):

        # Mutual Info Calculation (or loading).
        mi_subset = feats_df.columns
        print(feats_df.head())
        # mi_ser = mutual_info_classif(X=fs.X.feat_frame[mi_subset], y=fs.y, n_neighbors=4, n_jobs=pfeat_numobs)
        # mi_ser = get_mi_feats(fs, subset=mi_subset.copy(), num_feats=None)
        if os.path.isfile("{}mutual_info.csv".format(save_dir)):
            mi_df = pd.read_csv("{}mutual_info.csv".format(save_dir), index_col="Index")
            mi_mean = mi_df["Mean"]
            mi_std = mi_df["Std"]
        else:
            mi_list = list()
            assert not feats_df.empty
            for rus_num in list(range(cv)):
                feats_under, label_under = RandomUnderSampler().fit_resample(
                    feats_df, target
                )
                feats_under = feats_df.loc[feats_under.index]
                new_mi = mutual_info_classif(
                    feats_under, label_under, n_neighbors=k, random_state=0, n_jobs=-1
                )
                mi_list.append(new_mi)
            # print('Length of mi_list: {}'.format(mi_list))
            mi_df = pd.concat(
                [pd.Series(data=mi, index=feats_df.columns) for mi in mi_list], axis=1
            )
            mi_mean = mi_df.mean(axis=1).squeeze()
            mi_std = mi_df.std(axis=1, skipna=True, ddof=2).squeeze()
        mi_ser = mi_mean[mi_mean > 0].sort_values(ascending=False)
        pd.concat(
            [mi_ser, mi_std[mi_ser.index]], axis=1, names=["Mean", "Std"]
        ).to_csv()
        logger.info("Mutual Info Based Feature Selection with MRMR-based measured.")
        # scaled_mi_feats = get_mi_feats(fs, subset=first_feats, num_feats=outputs['n_feats'][feat_num])
        vif_loops, n_vif_in = 25, 75
        candidates = mi_mean.copy()[:n_vif_in]
        vif_all_feats = feats_df.copy()
        vif_columns = vif_all_feats.columns
        # candidates.drop(vif_columns, inplace=True)
        for i in list(range(vif_loops)):
            vif_ser = calculate_vif(
                vif_all_feats, candidates, generalized=False, n_jobs=1
            )
            vif_ser.copy().transform(func=softmax, axis=0)
            mi_scores = (
                mi_ser[vif_ser.index].copy().transform(func=softmax)[vif_ser.index]
            )
            drop_chance = vif_ser.divide(mi_scores)
            print(drop_chance.head())
            vif_all_feats.drop(columns=drop_chance.sample(weights=drop_chance, axis=1))
            # top_vif = vif_ser.sort_values(ascending=False).columns[0]
            # print(vif_ser[top_vif])
            # vif_columns.(top_vif)

            new_feat = candidates.sample(n=1, weights=mi_mean)
            vif_columns.append(new_feat)
            candidates.drop(new_feat, inplace=True)
        pprint([c for c in vif_columns], compact=True)
        pprint(vif_score_df.to_dict(), width=60, compact=True)
        mi_sorted = mi_ser.sort_values(ascending=False).iloc[:n_feats_out]
        new_feats = feats_df[mi_sorted.index]
        logger.info(pprint.pformat(mi_sorted[:10], compact=True))
        logger.info(
            "Incremental PCA from MI output of top {} features, keeping:".format(
                n_pca_in, n_pca_out
            )
        )
        mi_pca = IncrementalPCA(n_components=15, whiten=True, batch_size=5000).fit(
            X=new_feats.iloc[:n_pca_in]
        )
        logger.info("PCA Results w/whitening: EV and EV ration")
        logger.info(pprint.pformat(pd.Series(mi_pca.explained_variance_)))
        logger.info(pprint.pformat(pd.Series(mi_pca.explained_variance_ratio_)))
        with open(
            "{}pca_{}_tp_{}.csv".format(save_dir, n_pca_in, n_pca_out),
            "w",
            encoding="utf-8",
        ) as f:
            pd.Series(mi_pca.explained_variance_).sort_values(ascending=False).to_csv(
                path_or_buf=f
            )
        plt.plot(mi_pca.explained_variance_)
        plt.show()
        return new_feats

    def vif_select(
        feats_df,
        target,
        feature_names_in,
        n_feats_out,
        save_dir,
        model_cv=100,
        model_size=20,
        model_type="ols",
        sample_wts=samp_wts,
        n_jobs=-1,
        verbose=0,
        **kwargs
    ):
        if uni_mi:
            uni_mi = mutual_info_classif(X=df, y=target, random_state=0, n_jobs=-1)
        vif_returns = repeated_stochastic_vif(
            feats_df[feature_names_in],
            importance_ser=uni_mi,
            sample_wts=sample_wts,
            model_size=model_size,
            rounds=model_cv,
            n_jobs=-1,
            verbose=verbose,
            **kwargs
        )
        # vif_criteria, vif_score_df, vif_stats, votes = vif_returns
        if save_dir is str:
            vif_paths = (
                "{}vif_mi_criteria.pkl".format(save_dir),
                "{}vif_score_df.pkl".format(save_dir),
                "{}vif_appearances.pkl".format(save_dir),
                "{}vif_summary.pkl".format(save_dir),
            )
        elif type(save_dir) is str or type(save_dir) is tuple:
            assert len(save_dir) == 4
        for vif_result, v_path in zip(vif_returns, vif_paths):
            with open(vif_paths, "w+b") as f:
                pickle.dump(vif_return, f)
        vif_feats = feats_df[vif_criteria.index[: int(n_feats_out)]]
        return vif_feats

    def extra_trees_select(feats_df, target, feature_names_in, n_feats_out, save_dir):
        feats_df = extra_trees_gini_importances(
            data_df=feats_df,
            labels=target,
            sample_wts=sample_wts,
            n_feats_out=n_feats_out,
            save_dir=save_dir,
        )
        with open(
            "{}extra_trees_feature_importance.csv".format(save_dir),
            "w",
            encoding="utf-8",
        ) as f:
            f.writelines([x for x in feats_df])
        return feats_df

    mi_select(
        feats_df=std_cut_feats[good_feats],
        target=fs.y,
        feature_names_in=good_feats.tolist(),
        n_feats_out=25,
        save_dir=os.environ.get("MODELS_DIR"),
    )

    # TODO Implement Select-Train-Score refactor.
    outputs_keys = ["n_feats", "feat_meth", "n_cv_splits", "results", "models"]
    first_run = {
        "initial_feats": good_feats,
        "n_feats_out": [250, 50],
        "feat_meth": ["disr", "disr"],
        "n_cv_splits": 5,
        "results": [],
        "models": [],
    }
    second_run = {
        "initial_feats": good_feats,
        "n_feats_out": [100, 50],
        "feat_meth": ["mi", "extra"],
        "n_cv_splits": 5,
        "results": [],
        "models": [],
    }
    all_feats_disr = {
        "initial_feats": std_cut_feats.columns,
        "n_feats_out": [250, 50],
        "feat_meth": ["disr", "disr"],
        "n_cv_splits": 5,
        "results": [],
        "models": [],
    }
    all_feats_mi_trees = {
        "initial_feats": std_cut_feats.columns,
        "n_feats_out": [250, 50],
        "feat_meth": ["mi", "extra"],
        "n_cv_splits": 5,
        "results": [],
        "models": [],
    }
    # output_list = [first_run, second_run, all_feats_mi_trees, all_feats_disr]
    output_list = [second_run]

    def feature_selection(split_data, target, fs_params, sel_dir):
        num_bins = 50
        new_feats = split_data[fs_params["initial_feats"]]
        for feat_num, feat_type in enumerate(fs_params["feat_meth"]):
            select_dir = "{}{}_{}_{}features/".format(
                sel_dir, feat_num, feat_type, fs_params["n_feats_out"][feat_num]
            )
            os.makedirs(
                select_dir
            )  # results = dict([(k, dict()) for k in fs_params['feat_meth'][feat_num]])
            if feat_type == "cmim":
                pass
                # ITMO_FS.CMIM()
            if feat_type == "disr":
                new_feats = ITMO_FS.filters.DISRWithMassive(
                    expected_size=fs_params["n_feats_out"][feat_num]
                ).fit_transform(X=new_feats, y=target, feature_names=new_feats.columns)
                with open("{}disr_features.csv", "w", encoding="utf-8") as f:
                    [f.write(c) for c in new_feats.columns.tolist()]

            if feat_type == "mi":
                logger.info("\nMutual Info Based Feature Selection")
                new_feats = mi_select(
                    new_feats,
                    target,
                    feature_names_in=new_feats.columns,
                    n_feats_out=fs_params["n_feats_out"][feat_num],
                    save_dir=select_dir,
                )
                """
                new_feats = pd.Series(data=mutual_info_classif(X=new_feats, y=target, random_state=0,
                                                               n_jobs=-1),
                                      index=new_feats.index)
                MODEL_DIR = '{}mutual_info/'.format(os.environ.get('MODELS_DIR'))
                if not os.path.isdir(MODEL_DIR):
                    os.makedirs(MODEL_DIR)
                with open('{}mutual_info.csv'.format(MODEL_DIR), 'w', encoding='utf-8') as f:
                    f.writelines(pprint.pformat(mi_ser))
                """
            if feat_type == "extra":
                new_feats = extra_trees_select(
                    new_feats,
                    target,
                    new_feats.columns,
                    n_feats_out=fs_params["n_feats_out"][feat_num],
                    save_dir=select_dir,
                )
            # Don't use until refactor to use new dictionary system.
            if feat_type == "fmim":
                from feature_selection.importance import fmim

                MODEL_DIR = "{}fmim/".format(select_dir)
                fmim_feats_in = good_feats.intersection(fs.X.dense)
                fmim_scores = fmim(
                    features=fs.X.feat_frame[fmim_feats_in],
                    label=fs.y,
                    num_feats=fs_params["n_feats_out"][feat_num],
                    n_jobs=run_params["n_jobs"],
                )
                first_feats = pd.Series(
                    data=fmim_scores, index=fmim_feats_in
                ).sort_values(ascending=False)
                logger.info("FMIM Features Selected:")
                # logger.info(first_feats.iloc[:15], compact=True)
                if not os.path.isdir(MODEL_DIR):
                    os.makedirs(MODEL_DIR)
                with open("{}fmim.csv".format(MODEL_DIR), "w", encoding="utf-8") as f:
                    f.writelines(
                        ["{:.6f}\n".format(x) for x in first_feats.index.tolist()]
                    )
            logger.info("Final features selected for model training:")
            logger.info(pprint.pformat(new_feats.columns.tolist()))
            return new_feats

    for run_num, out in enumerate(output_list):

        # Set Directories
        exp_num = 0
        EXP_DIR = "{}experiment_{}/".format(os.environ.get("MODELS_DIR"), exp_num)
        while exp_num >= 0 and os.path.isdir(EXP_DIR) and len(os.listdir(EXP_DIR)) > 0:
            exp_num += 1
            EXP_DIR = "{}experiment_{}/".format(os.environ.get("MODELS_DIR"), exp_num)
        if not os.path.isdir(EXP_DIR):
            os.makedirs(EXP_DIR)

        weights = {"dev": [], "eval": []}
        model_keys = ["true", "rf", "lr", "mlp"]
        split_keys = ["dev", "eval"]
        indices = dict([(k, list()) for k in split_keys])
        models = dict([(k, list()) for k in model_keys])
        results = dict(
            [(k, dict([(s, list()) for s in split_keys])) for k in model_keys]
        )
        out.update([("indices", indices), ("results", results), ("models", models)])

        feats_df_list = list()
        feature_dir = "{}run{}_features/".format(EXP_DIR, run_num)
        os.makedirs(feature_dir)
        # Pre-split indices
        cv = -1
        for d, e in StratifiedKFold().split(X=std_cut_feats, y=fs.y):
            cv += 1
            # print(results)
            # for cv in list(range(outputs[feat_num]['n_cv_splits'])):
            indices["dev"].append(d)
            indices["eval"].append(e)
            fs_dir = "{}/features_cv{}".format(feature_dir, cv)
            feats_df_list.append(
                feature_selection(
                    std_cut_feats.iloc[d], fs.y.iloc[d], fs_params=out, sel_dir=fs_dir
                )
            )

        for mod_name in model_keys:
            if mod_name == "true":
                continue
            model_dir = "{}run{}/{}/".format(EXP_DIR, run_num, mod_name)
            os.makedirs(model_dir)

            def run_balanced_tree(n_trees=1000):
                logger.info(
                    "Training Balanced Random Forest with {} trees".format(n_trees)
                )
                rf_path = "{}imb_rf_{}_cv_{}.pkl".format(
                    CV_DIR, "-".join([str(x) for x in out["n_feats_out"]]), cv
                )
                brfc = BalancedRandomForestClassifier(
                    n_estimators=n_trees,
                    max_depth=25,
                    n_jobs=-1,
                    random_state=0,
                    sampling_strategy="auto",
                    min_weight_fraction_leaf=0.05,
                    replacement=False,
                    verbose=1,
                    class_weight="balanced_subsample",
                    bootstrap=True,
                )
                brfc.fit(X=X_dev, y=y_dev)
                models["rf"].append(brfc)
                # TODO: Move to end.
                with open(rf_path, "w+b") as f:
                    pickle.dump(
                        (models["rf"][cv], (indices["dev"][cv], indices["eval"][cv])),
                        file=f,
                    )
                results["rf"]["dev"].append(models["rf"][cv].predict(X_dev))
                results["rf"]["eval"].append(models["rf"][cv].predict(X_eval))
                logger.info(
                    "Matthews Correlation Coefficient (Dev): {}:".format(
                        matthews_corrcoef(
                            y_true=results["true"]["dev"][cv],
                            y_pred=results["rf"]["dev"][cv],
                        )
                    )
                )
                logger.info(
                    "Matthews Correlation Coefficient (Eval): {}:".format(
                        matthews_corrcoef(
                            y_true=results["true"]["eval"][cv],
                            y_pred=results["rf"]["eval"][cv],
                        )
                    )
                )
                logger.info(
                    "Balanced Accuracy (Dev): {}:".format(
                        balanced_accuracy_score(
                            y_true=results["true"]["dev"][cv],
                            y_pred=results["rf"]["dev"][cv],
                        )
                    )
                )
                logger.info(
                    "Balanced Accuracy (Eval): {}:".format(
                        balanced_accuracy_score(
                            y_true=results["true"]["eval"][cv],
                            y_pred=results["rf"]["eval"][cv],
                        )
                    )
                )

            def logit():
                lr_path = "{}logit_{}_robust_cv_{}.pkl".format(
                    CV_DIR, "-".join([str(x) for x in out["n_feats_out"]]), cv
                )
                coeff_path = "{}logit_{}_robust_coeff.csv".format(
                    CV_DIR, "-".join([str(x) for x in out["n_feats_out"]]), cv
                )

                penalty, C = "l2", 16
                logger.info(
                    "Training Logistic Regression with {} penalty and C of {}".format(
                        penalty, C
                    )
                )
                robust_scale = RobustScaler(
                    with_centering=True, unit_variance=True
                ).fit(X_dev)
                X_dev_robust = robust_scale.transform(X_dev.astype(np.float64))
                X_eval_robust = robust_scale.transform(X_eval.astype(np.float64))
                models["lr"].append(
                    LogisticRegressionCV(
                        penalty=penalty,
                        tol=1e-4,
                        Cs=C,
                        max_iter=5000,
                        solver="newton-cg",
                        class_weight="balanced",
                        cv=3,
                        n_jobs=-1,
                        verbose=0,
                        scoring=make_scorer(matthews_corrcoef),
                        random_state=0,
                    ).fit(X=X_dev_robust, y=y_dev)
                )
                results["lr"]["dev"].append(models["lr"][cv].predict(X_dev_robust))
                results["lr"]["eval"].append(models["lr"][cv].predict(X_eval_robust))
                try:
                    pd.Series(models["lr"][-1].coef_, index=X_dev.columns).sort_values(
                        ascending=False
                    ).to_csv(coeff_path)
                except:
                    logger.info("No coef_ today.")
                with open(lr_path, "w+b") as f:
                    pickle.dump(
                        (models["lr"][0], (indices["dev"][0], indices["eval"][0])),
                        file=f,
                    )

            def run_perceptron(version=2):
                mlp_path = "{}deep_mlp_{}_normed_cv_{}.pkl".format(
                    CV_DIR, "-".join([str(x) for x in out["n_feats_out"]]), cv
                )

                norm_transformer = StandardScaler().fit(X_dev)
                X_dev_normed = norm_transformer.transform(X_dev)
                X_eval_normed = norm_transformer.transform(X_eval)
                X_dev_under, y_dev_under = NearMiss(version=2, n_jobs=-1).fit_resample(
                    X_dev_normed, y_dev
                )
                models["mlp"].append(
                    MLPClassifier(
                        random_state=0,
                        hidden_layer_sizes=(50, 75, 100, 75, 50),
                        learning_rate="invscaling",
                    ).fit(X_dev_under, y_dev_under)
                )
                if cv == 0:
                    logger.info(models["mlp"][cv].get_params())
                dev_under_score = models["mlp"][cv].predict(X_dev_under)
                dev_normed_score = models["mlp"][cv].predict(X_dev_normed)
                eval_normed_score = models["mlp"][cv].predict(X_eval_normed)
                results["mlp"]["dev"].append(dev_normed_score)
                results["mlp"]["eval"].append(eval_normed_score)
                # Score Input Tuples
                norm_under_dev_tup = (y_dev_under, dev_under_score)
                normed_all_dev_tup = (y_dev, dev_normed_score)
                normed_all_eval_tup = (y_eval, eval_normed_score)

                try:
                    logger.info(
                        "Results for (50, 75, 100, 75, 50) MLP: Normed NearMiss-{}".format(
                            version
                        )
                    )
                    logger.info(
                        "Matthews Correlation Coefficient (Dev Under-Sampled): {}:".format(
                            matthews_corrcoef(
                                y_true=norm_under_dev_tup[0],
                                y_pred=norm_under_dev_tup[1],
                            )
                        )
                    )
                    logger.info(
                        "Matthews Correlation Coefficient (Dev): {}:".format(
                            matthews_corrcoef(
                                y_true=normed_all_dev_tup[0],
                                y_pred=normed_all_dev_tup[1],
                            )
                        )
                    )
                    logger.info(
                        "Matthews Correlation Coefficient (Eval): {}:".format(
                            matthews_corrcoef(
                                y_true=normed_all_eval_tup[0],
                                y_pred=normed_all_eval_tup[1],
                            )
                        )
                    )
                    logger.info(
                        "Balanced Accuracy Score (Dev Under): {}:".format(
                            balanced_accuracy_score(
                                y_true=norm_under_dev_tup[0],
                                y_pred=norm_under_dev_tup[1],
                            )
                        )
                    )
                    logger.info(
                        "Balanced Accuracy Score (Dev): {}:".format(
                            balanced_accuracy_score(
                                y_true=normed_all_dev_tup[0],
                                y_pred=normed_all_dev_tup[1],
                            )
                        )
                    )
                    logger.info(
                        "Balanced Accuracy Score (Eval): {}:".format(
                            balanced_accuracy_score(
                                y_true=normed_all_eval_tup[0],
                                y_pred=normed_all_eval_tup[1],
                            )
                        )
                    )
                except:
                    print("MLP scoring error")
                with open(mlp_path, "w+b") as f:
                    pickle.dump(
                        (models["mlp"][cv], (indices["dev"][cv], indices["eval"][cv])),
                        file=f,
                    )

            """    results['lr']['dev'].append(models['lr'][cv].predict(X_dev))
            results['lr']['eval'].append(models['lr'][cv].predict(X_eval))
            logger.info(
                pprint.pformat('Balanced Accuracy Score for Dev: {:.4f}'.format(
                    balanced_accuracy_score(results['true']['dev'][cv], results['lr']['dev'][cv]))))
            logger.info(pprint.pformat(
                'Balanced Accuracy Score for Eval: {:.4f}'.format(
                    balanced_accuracy_score(results['true']['eval'][cv], results['lr']['eval'][cv]))))
            logger.info(
                pprint.pformat('Matthew Correlation Coefficient for Dev: {:.4f}'.format(
                    matthews_corrcoef(results['true']['dev'][cv], results['lr']['dev'][cv]))))
            logger.info(pprint.pformat(
                'Matthew Correlation Coefficient for Eval: {:.4f}'.format(
                    matthews_corrcoef(results['true']['eval'][cv], results['lr']['eval'][cv]))))"""
            # rocfig.savefig(roc_path, dpi='figure', format='png', transparent=True)

            for cv in list(range(5)):
                CV_DIR = "{}cv{}/".format(model_dir, cv)
                if not os.path.isdir(CV_DIR):
                    os.makedirs(CV_DIR)
                new_feats_df = feats_df_list[cv]
                # Prep current_data for modeling.
                X_dev_all_arr, y_dev_arr = check_X_y(
                    X=std_cut_feats.iloc[indices["dev"][cv]],
                    y=fs.y.iloc[indices["dev"][cv]],
                )
                X_eval_all_arr, y_eval_arr = check_X_y(
                    X=std_cut_feats.iloc[indices["eval"][cv]],
                    y=fs.y.iloc[indices["eval"][cv]],
                )
                X_eval_all = pd.DataFrame(
                    data=X_eval_all_arr,
                    index=std_cut_feats.index[indices["eval"][cv]],
                    columns=std_cut_feats.columns,
                )
                y_eval = pd.Series(
                    data=y_eval_arr, index=fs.y.iloc[indices["eval"][cv]]
                )
                X_dev_all = pd.DataFrame(
                    data=X_dev_all_arr,
                    index=std_cut_feats.index[indices["dev"][cv]],
                    columns=std_cut_feats.columns,
                    dtype=np.float32,
                )
                y_dev = pd.Series(data=y_dev_arr, index=fs.y.iloc[indices["dev"][cv]])
                # Feature Selection
                X_dev = X_dev_all[new_feats_df.columns]
                X_eval = X_eval_all[new_feats_df.columns]
                results["true"]["dev"].append(y_dev)
                results["true"]["eval"].append(y_eval)
                dev_counts = fs.y.iloc[indices["dev"][cv]].value_counts()
                eval_counts = fs.y.iloc[indices["eval"][cv]].value_counts()
                weights["dev"].append(
                    compute_sample_weight(class_weight="balanced", y=y_dev)
                )
                weights["eval"].append(
                    compute_sample_weight(class_weight="balanced", y=y_eval)
                )
                # sample_wts.iloc[indices['dev'][cv]], sample_wts.iloc[indices['eval'][cv]]
                if mod_name == "lr":
                    logit()
                elif mod_name == "rf":
                    run_balanced_tree()
                elif mod_name == "mlp":
                    run_perceptron()
        try:
            plot_results(
                results_dict=results,
                models=models,
                indices=indices,
                weights=weights,
                insol_label=insol_label,
                feats=feats_df_list,
                save_dir=RUN_DIR,
            )
        except:
            logger.warning("Plotting failed!")
