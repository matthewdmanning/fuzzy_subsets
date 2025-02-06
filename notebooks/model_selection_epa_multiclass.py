import itertools
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

import padel_categorization
import vapor_pressure_selection
from clean_new_epa_data import get_query_data


def get_confusion_weights():
    return np.array([[1.0, 0.0, -0.5], [0.25, 1.0, 0.0], [0.0, 0.25, 1.0]])


def three_class_solubility(y_true, y_pred, sample_weight=None, **kwargs):
    W = get_confusion_weights()
    try:
        C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight) * W
    except UserWarning:
        print("True, Predicted, and Confusion Weighting")
        pprint.pprint(y_true.unique())
        pprint.pprint(y_pred.unique())
        pprint.pprint(W)
        C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(C * W) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn("y_pred contains classes not in y_true")
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    return score


def optimize_tree(feature_df, labels, model, scoring, cv):
    param_grid = {
        "max_depth": [None, 25, 20, 15],
        "min_impurity_decrease": [0, 0.005, 0.01],
        "max_leaf_nodes": [100, 150, 200, 250],
    }
    tree_optimized = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=-1,
        cv=cv,
        return_train_score=True,
        error_score="raise",
    )
    tree_optimized.fit(feature_df, labels)
    print(tree_optimized.best_params_)
    return tree_optimized


def get_class_splits(labels):
    edges = [
        5.5,
        10.5,
        19,
    ]


def safe_mapper(x, map):
    if x in map.keys():
        return map[x]
    else:
        return x


def main():
    select_params = {
        "corr_method": "kendall",
        "xc_method": "kendall",
        "max_features_out": 15,
        "fails_min_vif": 0,
        "fails_min_perm": 6,
        "fails_min_sfs": 0,
        "features_min_vif": 5,
        "features_min_perm": 8,
        "features_min_sfs": 10,
        "thresh_reset": 0.025,
        "thresh_vif": 10,
        "thresh_perm": 0.0025,
        "thresh_sfs": -0.005,
        "thresh_xc": 0.95,
        "max_trials": 25,
        "cv": 5,
        "importance": False,
        "scoring": make_scorer(balanced_accuracy_score),
    }
    epa_scorer = make_scorer(balanced_accuracy_score)
    # Paths.
    parent_dir = "{}multiclass_trial/".format(os.environ.get("MODEL_DIR"))
    std_path = "{}stdizer_epa_query.csv".format(parent_dir)
    std_pkl = "{}stdizer_epa_query.pkl".format(parent_dir)
    lookup_path = "{}lookup.csv".format(parent_dir)
    desc_path = "{}padel_features_output_max_sol.pkl".format(parent_dir)
    transformer_path = "{}transformer.pkl".format(parent_dir)
    transformed_df_path = "{}transformed_epa_exact_sol_data.pkl".format(parent_dir)
    exp_dir = "{}model_selection/".format(parent_dir)
    epa_label_path = "{}epa_labels.csv".format(exp_dir)
    asinh_path = "{}epa_transformed.csv".format(parent_dir)
    os.makedirs(exp_dir, exist_ok=True)
    # Get data
    insol_labels, maxed_sol_labels, sol100_labels = get_query_data()
    combo_labels, convert_df = get_conversions(maxed_sol_labels, lookup_path)
    smiles_df, sid_to_key = standardize_smiles(
        convert_df, combo_labels, maxed_sol_labels, std_pkl
    )
    max_conc = combo_labels.set_index(keys="inchiKey", drop=True)[
        "sample_concentration"
    ].astype(np.float32)
    max_conc.index = max_conc.index.map(
        lambda x: safe_mapper(x, sid_to_key), na_action="ignore"
    )
    if os.path.isfile(transformed_df_path):
        transformed_df = pd.read_pickle(transformed_df_path)
        raw_df = pd.read_pickle(desc_path).loc[transformed_df.index]
        epa_labels = label_solubility_clusters(max_conc[transformed_df.index], exp_dir)

        train_raw_df, test_raw_df = train_test_split(
            raw_df, test_size=0.2, stratify=epa_labels.squeeze()
        )
        print(train_raw_df)
        train_labels = epa_labels[train_raw_df.index]
        test_labels = max_conc[test_raw_df.index]
        preprocessor, p = vapor_pressure_selection.get_standard_preprocessor(
            transform_func="asinh", corr_params=select_params
        )
        train_df = preprocessor.fit_transform(train_raw_df, train_labels)
        # epa_labels = pd.read_csv(epa_label_path, index_col="index").squeeze()
        # Save cleaned solubility data.
        # max_conc[train_df.index].to_csv("{}epa_max_conc_cleaned.csv".format(parent_dir))
        # assert epa_labels.shape[0] > 0
        # pd.concat([epa_labels, max_conc]).to_csv(epa_label_path, index_label="index")
        """
        tree_search = optimize_tree(
            train_df,
            epa_labels,
            model=RandomForestClassifier(
                bootstrap=False, class_weight="balanced", random_state=0
            ),
            scoring=epa_scorer,
            cv=RepeatedStratifiedKFold(n_repeats=3, random_state=0),
        )
        print(tree_search.best_params_)
        print(tree_search.best_score_)
        best_tree = tree_search.best_estimator_
        """
        cv_results = train_multilabel_models(
            train_df, train_labels, epa_scorer, select_params
        )
        print(cv_results)
        test_df = preprocessor.transform(test_raw_df, test_labels)
        return train_df, epa_labels, cv_results


def fetch_padel(smiles_df, desc_path, exp_dir):
    epa_df = pd.DataFrame([])
    if os.path.isfile("{}.pkl".format(desc_path)):
        epa_df = pd.read_pickle("{}.pkl".format(desc_path))
        # epa_df = epa_df[~epa_df.index.duplicated()]
    overage = epa_df.index.difference(smiles_df.index)
    epa_df.drop(overage, inplace=True)
    missing_smiles = smiles_df.index.difference(epa_df.index).dropna()
    new_smiles = smiles_df["smiles"].squeeze()[missing_smiles]
    """
            new_smiles = missing_smiles.intersection(
                smiles_df.index.map(sid_to_key)
            ).dropna()
            """
    # missing_smiles = different_smiles.difference(smiles_df.index)
    if new_smiles.size > 0:
        if not os.path.isfile("{}desc_api_out.pkl".format(exp_dir)):
            desc_df, info_df, failed_list = get_descriptors(
                smi_list=new_smiles.tolist()
            )
            assert not desc_df.empty or len(failed_list) > 0
            desc_df.to_pickle("{}desc_api_out.pkl".format(exp_dir))
        else:
            desc_df = pd.read_pickle("{}desc_api_out.pkl".format(exp_dir))
        if not desc_df.empty and not epa_df.empty:
            if desc_df.shape[1] == epa_df.shape[1]:
                epa_df = pd.concat([epa_df, desc_df])
            else:
                print(
                    "epa_df and desc_df are different shapes[1]:\n{}\n{}".format(
                        epa_df.shape, desc_df.shape
                    )
                )
                epa_df = desc_df
            epa_df = epa_df[~epa_df.index.duplicated()]
        elif epa_df.empty and not desc_df.empty:
            epa_df = desc_df
        elif not epa_df.empty:
            print("desc_df is empty!!!")
            pass
        else:
            raise ValueError
        epa_df.columns = padel_categorization.get_short_padel_names().astype(str)
        epa_df.dropna(how="all", axis=1, inplace=True)
        epa_df.dropna(how="any", inplace=True)
        epa_df.to_pickle("{}.pkl".format(desc_path))
    else:
        epa_df.dropna(how="all", axis=1, inplace=True)
        epa_df.dropna(how="any", inplace=True)
    return epa_df


def get_conversions(maxed_sol_labels, lookup_path):
    if os.path.isfile(lookup_path):
        convert_df = pd.read_csv(lookup_path)
        if True:
            convert_df.drop(
                columns=[x for x in convert_df.columns if "Unnamed" in str(x)],
                inplace=True,
            )
            convert_df.drop(columns="mol", inplace=True, errors="ignore")
            convert_df.to_csv(lookup_path)
    else:
        convert_df = lookup_chem(comlist=maxed_sol_labels.index.tolist())
        convert_df.drop(columns="mol", inplace=True, errors="ignore")
        convert_df.to_csv(lookup_path)
        convert_df.drop(
            columns=["bingoId", "name", "casrn", "formula", "cid"],
            inplace=True,
            errors="ignore",
        )
    combo_labels = pd.concat(
        [convert_df, maxed_sol_labels.reset_index(names="label_id")], axis=1
    )
    failed_lookup = combo_labels[combo_labels["id"] != combo_labels["label_id"]]
    combo_labels.drop(index=failed_lookup.index, inplace=True)
    nosmiles = combo_labels[
        (combo_labels["smiles"] == "nan") | combo_labels["smiles"].isna()
    ]
    combo_labels.drop(index=nosmiles.index, inplace=True)
    return combo_labels, convert_df


def train_multilabel_models(feature_df, labels, epa_scorer, select_params):
    if True:
        best_tree = RandomForestClassifier(
            bootstrap=False,
            max_leaf_nodes=100,
            min_impurity_decrease=0.005,
            class_weight="balanced",
            n_jobs=-1,
            random_state=0,
        )
        xtra_tree = ExtraTreesClassifier(
            max_leaf_nodes=100,
            min_impurity_decrease=0.005,
            class_weight="balanced",
            n_jobs=-1,
            random_state=0,
        )
        lrcv = LogisticRegressionCV(
            scoring=epa_scorer,
            class_weight="balanced",
            n_jobs=-1,
            max_iter=5000,
            random_state=0,
        )
        model_list = [best_tree, lrcv, xtra_tree]
        name_list = ["RandomForest", "Logistic", "ExtraTrees"]
    else:
        ovo_tree = OneVsOneClassifier(estimator=best_tree)
        ovo_lr = OneVsOneClassifier(estimator=lrcv)
        ovr_tree = OneVsRestClassifier(estimator=best_tree)
        ovr_lr = OneVsRestClassifier(estimator=lrcv)
        model_list = [ovo_tree, ovo_lr, ovr_tree, ovr_lr]
        name_list = ["ovo_tree", "ovo_logit", "ovr_tree", "ovr_logit"]

    # intersect_final = labels.index.intersection(feature_df.index)
    # labels = labels[intersect_final]
    # feature_df = feature_df.loc[intersect_final]
    # print(labels)
    print(feature_df.shape)
    print(labels)
    candidate_features = vapor_pressure_selection.get_search_features(feature_df)
    search_features = feature_df.columns[feature_df.columns.isin(candidate_features)].tolist()
    print("{} features to select from.".format(len(search_features)))
    feature_df = feature_df[search_features]
    dev_labels, eval_labels = train_test_split(
        labels, test_size=0.2, random_state=0, shuffle=True, stratify=labels
    )
    dev_df = feature_df.loc[dev_labels.index]
    eval_df = feature_df.loc[eval_labels.index]
    print("{} features to select from.".format(len(search_features)))
    best_corrs = dev_df[search_features].corrwith(dev_labels, method=select_params["xc_method"])
    cross_corr = dev_df[search_features].corr(method=select_params["corr_method"])
    cv_dict = dict()
    for m, n in zip(model_list, name_list):
        selection_models = {
            "predict": m,
            "permutation": m,
            "importance": m,
            "vif": linear_model.LinearRegression(n_jobs=-2),
        }
        model_dir = "{}feature_selection_metaestimator_trial/{}/".format(exp_dir, n)
        os.makedirs(model_dir, exist_ok=True)
        model_dict, score_dict, dropped_dict, best_features = (
            vapor_pressure_selection.grove_features_loop(
                dev_df[search_features],
                dev_labels,
                target_corr=best_corrs,
                cross_corr=cross_corr,
                select_params=select_params,
                selection_models=selection_models,
                hidden_test=(eval_df, eval_labels),
                save_dir=model_dir,
            )
        )
        print("Best features!")
        short_to_long = padel_categorization.padel_short_to_long()
        best_features_long = short_to_long[best_features].tolist()
        print("\n".join(best_features_long))
        pd.Series(best_features_long).to_csv(
            "{}best_features_{}.csv".format(model_dir, n)
        )
        """        
        with sklearn.config_context(enable_metadata_routing=False) as f:
            cv_dict[n] = cross_validate(
                m,
                dev_df[best_features],
                dev_labels,
                scoring=epa_scorer,
                cv=RepeatedStratifiedKFold(n_repeats=3, random_state=0),
                n_jobs=-1,
                return_train_score=True,
                return_indices=True,
            )["test_score"]
            """
        train_idx, test_idx = train_test_split(
            train_df, test_size=0.2, random_state=0, stratify=train_labels
        )
        cmd_train_df, cmd_train_labels = (
            train_df.loc[train_idx.index],
            train_labels[train_idx.index],
        )
        cmd_test_df, cmd_test_labels = (
            train_df.loc[test_idx.index],
            train_labels[test_idx.index],
        )
        m.fit(cmd_train_df, cmd_train_labels)
        cmd = ConfusionMatrixDisplay.from_estimator(
            m, X=cmd_test_df[best_features], y=cmd_test_labels
        )
        cmd.plot()
        cmd.figure_.savefig("{}confusion_matrix.png".format(model_dir))
    cv_results = pd.DataFrame.from_dict(cv_dict)
    print(cv_results, flush=True)
    ax = sns.catplot(cv_results)
    ax.savefig("{}test_scores.png".format(exp_dir))
    plt.show()
    return cv_results


def label_solubility_clusters(labels, exp_dir, algo=False):
    if algo:
        from sklearn.cluster import BisectingKMeans

        trimmed_labels = (
            labels.clip(upper=110.0, lower=4.5).multiply(100).apply(np.asinh)
        )
        print(
            "Asinh: {:.4f} {:.4f} {:.4f} {:.4f}".format(
                np.asinh(5), np.asinh(10), np.asinh(50), np.asinh(100)
            )
        )
        print(trimmed_labels.describe())
        # agg_clusters = AgglomerativeClustering(linkage="ward", metric="euclidean", n_clusters=3).fit_predict(trimmed_labels.to_frame())
        agg_clusters = BisectingKMeans(
            n_clusters=4, max_iter=5000, bisecting_strategy="largest_cluster"
        ).fit_predict(trimmed_labels.to_frame())
        agg_clusters = pd.Series(agg_clusters, index=trimmed_labels.index)
        for n in agg_clusters.sort_values().unique():
            cluster = labels.sort_values()[agg_clusters == n]
            print(labels[cluster.index].min(), labels[cluster.index].max())
        """
        label_clusterer = (
            KMeans(n_clusters=6, random_state=0)
            .set_output(transform="pandas")
            .fit(trimmed_labels.to_frame())
        )
        epa_labels = pd.Series(label_clusterer.predict(trimmed_labels.to_frame()), index=trimmed_labels.index)
        """
        epa_labels = agg_clusters
    else:
        epa_labels = labels.copy()
        epa_labels[epa_labels < 8.5] = 0
        epa_labels[(8.5 <= epa_labels) & (epa_labels < 45.0)] = 1
        epa_labels[(45.0 <= epa_labels) & (epa_labels < 90.0)] = 2
        epa_labels[90.0 <= epa_labels] = 3
    print("Cluster-assigned labels")
    print(np.sort(epa_labels.value_counts()), flush=True)
    return epa_labels


def standardize_smiles(feature_df, combo_labels, maxed_sol_labels, std_pkl):
    if os.path.isfile(std_pkl):
        smiles_df = pd.read_pickle(std_pkl)
        if False:
            clean_list = [x for x in smiles_df.tolist() if "sid" in x.keys()]
            unregistered_list = [
                x
                for x in smiles_df.to_dict()
                if "sid" not in x.keys() and "smiles" in x.keys()
            ]
            failed_list = [x for x in smiles_df.tolist() if "smiles" not in x.keys()]
            clean_df = pd.json_normalize(clean_list)
            unregistered_df = pd.json_normalize(unregistered_list)
            smiles_df = pd.concat([clean_df, unregistered_df], ignore_index=True)
            failed_df = pd.json_normalize(failed_list)
            if not failed_df.empty:
                failed_df.to_csv("{}failed_standardizer.csv".format(failed_df))
            assert not smiles_df.empty
            smiles_df.to_pickle(std_pkl)
            smiles_df.to_csv(std_path)
            if not os.path.isfile("{}failed_standardizer.csv".format(failed_df)):
                # TODO: Extract failed compounds by difference of inputs and outputs.
                pass

    else:
        smiles_df, failed_df = get_standardizer(comlist=feature_df["smiles"].tolist())
        if not failed_df.empty:
            failed_df.to_csv("{}failed_standardizer.csv".format(failed_df))
        assert not smiles_df.empty
        smiles_df.to_pickle(std_pkl)
    smiles_df.drop(
        columns=[
            "cid",
            "casrn",
            "name",
            "canonicalSmiles",
            "inchi",
            "mol",
            "formula",
            "id",
        ],
        inplace=True,
        errors="ignore",
    )
    assert not smiles_df.empty
    smiles_df = smiles_df[smiles_df["chemId"].isin(combo_labels["id"])]
    """
        if "smiles" in smiles_df.columns:
            smiles_df = smiles_df[~smiles_df.duplicated() & ~smiles_df["smiles"].isna()]
            # smiles_df.set_index(keys="sid", drop=True, inplace=True)
            print("Standardizer output:\n{}".format(smiles_df))
            print(smiles_df.columns)
            sid_to_smiles = smiles_df["smiles"].squeeze().to_dict()
            sid_to_key = smiles_df["inchiKey"].squeeze().to_dict()
            smiles_ind_df = smiles_df.reset_index().set_index(keys="smiles")
            smiles_to_sid = smiles_ind_df["sid"].squeeze().to_dict()
            smiles_fails = pd.Index(
                x for x in maxed_sol_labels.index if x not in sid_to_key.values()
            )
        else:
            smiles_fails = maxed_sol_labels.index
        # Get Descriptors.
        print("Raw label to InchiKey failures:\n{}".format(smiles_fails))
        raw_retry_df, retry_failures_df = get_standardizer(smiles_fails.tolist())
        if not raw_retry_df.empty and "smiles" in raw_retry_df.columns:
            smiles_df = pd.concat([smiles_df, raw_retry_df[~raw_retry_df["smiles"].isna()]])
            smiles_df.drop(index=smiles_df[~smiles_df["error"].isna()].index, inplace=True)
            smiles_df = smiles_df[~smiles_df.index.duplicated()]
            smiles_df.to_csv("{}updated_std_output.csv".format(exp_dir))
            sid_to_key = smiles_df["inchiKey"].squeeze().to_dict()
            smiles_fails = pd.Index(
                x for x in maxed_sol_labels.index if x not in sid_to_key.keys()
            )
            print("Retry label to InchiKey failures:\n{}".format(smiles_fails))
            # print(smiles_df.loc[smiles_fails])
        else:
            print("No return from retrying Standardizer on failed labels.")
            if "smiles" in smiles_df.columns:
                smiles_df.to_csv("{}updated_std_output.csv".format(exp_dir))
        smiles_fails.difference(smiles_df.index).to_series().to_csv(
            "{}failed_standardization.csv".format(exp_dir)
        )
        maxed_sol_labels.index = maxed_sol_labels.index.map(sid_to_key, na_action="ignore")
        """
    sid_to_key = (
        smiles_df[["inchiKey", "chemId"]]
        .set_index(keys="chemId", drop=True)
        .squeeze()
        .to_dict()
    )
    id_to_orig_smi = (
        smiles_df[["chemId", "smiles"]]
        .set_index(keys="chemId", drop=True)
        .squeeze()
        .to_dict()
    )
    smiles_df.set_index(keys="inchiKey", drop=True, inplace=True)
    return smiles_df, sid_to_key


def lookup_chem(comlist, batch_size=100, type="sid", prefix="DTXSID"):
    # stdizer = DescriptorRequestor.QsarStdizer(input_type="dtxsid")
    api_url = "https://ccte-cced-cheminformatics.epa.gov/api/search/download/properties"
    response_list = list()
    auth_header = {"x-api-key": os.environ.get("INTERNAL_KEY")}
    with requests.session() as r:
        req = requests.Request(method="POST", url=api_url)
        for c_batch in itertools.batched(comlist, n=batch_size):
            req_json = {"ids": [], "format": type}
            if prefix is not None:
                c_batch = [
                    c.removeprefix(prefix) for c in c_batch if prefix is not None
                ]
            for c in c_batch:
                req_json["ids"].append({"id": c, "sim": 0})
            response = r.request(
                method="POST", url=api_url, json=req_json, headers=auth_header
            )
            response_list.extend(response.json())
    response_df = pd.json_normalize(response_list)
    return response_df


def get_standardizer(comlist, batch_size=100, input_type="smiles"):
    api_url = "https://ccte-cced-cheminformatics.epa.gov/api/stdizer"
    response_list, failed_list = list(), list()
    auth_header = {"x-api-key": os.environ.get("INTERNAL_KEY")}
    with requests.session() as r:
        for c in comlist:
            params = {"workflow": "qsar-ready", input_type: c}
            response = r.request(
                method="GET", url=api_url, params=params, headers=auth_header
            )
            if isinstance(response.json(), list) and len(response.json()) > 0:
                response_list.append(response.json()[0])
            else:  # isinstance(response, (list, str)):
                failed_list.append(response.content)
    clean_list = [x for x in response_list if "sid" in x.keys()]
    unregistered_list = [
        x for x in response_list if "sid" not in x.keys() and "smiles" in x.keys()
    ]
    # failed_list = [x for x in response_list if "smiles" not in x.keys()]
    clean_df = pd.json_normalize(clean_list)
    unregistered_df = pd.json_normalize(unregistered_list)
    response_df = pd.concat([clean_df, unregistered_df], ignore_index=True)
    failed_df = pd.json_normalize(failed_list)
    return response_df.drop(columns="mol"), failed_df


def get_descriptors(smi_list, batch_size=100, desc_type="padel", input_type="smiles"):
    # TODO: Add original SMILES/identifier to info_df to link original data and descriptor data through info_df.
    # stdizer = DescriptorRequestor.QsarStdizer(input_type="dtxsid")
    api_url = "https://ccte-cced-cheminformatics.epa.gov/api/padel"
    """
    {
        "chemicals": ["string"],
        "options": {
            "headers": true,
            "timeout": 0,
            "compute2D": true,
            "compute3D": true,
            "computeFingerprints": true,
            "descriptorTypes": ["string"],
            "removeSalt": true,
            "standardizeNitro": true,
            "standardizeTautomers": true,
        },
    }
   """
    info_list, desc_dict, failed_list = list(), dict(), list()
    with requests.session() as r:
        auth_header = {"x-api-key": os.environ.get("INTERNAL_KEY")}
        req = requests.Request(method="GET", url=api_url, headers=auth_header).prepare()
        for c in smi_list:
            params = {"smiles": c, "type": "padel"}
            req.prepare_url(url=api_url, params=params)
            response = r.send(req)
            if response.status_code == 200 and len(response.json()) > 0:
                single_info_dict = dict(
                    [
                        (k, v)
                        for k, v in response.json()["chemicals"][0].items()
                        if k != "descriptors"
                    ]
                )
                single_info_dict.update([("SMILES_QSAR", c)])
                info_list.append(single_info_dict)
                desc_dict[response.json()["chemicals"][0]["inchiKey"]] = (
                    response.json()["chemicals"][0]["descriptors"]
                )
            else:
                failed_list.append(response.content)
    padel_names = padel_categorization.get_short_padel_names()
    info_df = pd.json_normalize(info_list)
    info_df.set_index(keys="inchiKey", inplace=True)
    info_df.drop(columns=padel_names, inplace=True, errors="ignore")
    desc_df = pd.DataFrame.from_dict(
        data=desc_dict, orient="index", columns=padel_names
    )
    return desc_df, info_df, failed_list


if __name__ == "__main__":
    logger = logging.getLogger(name="selection")
    main()
