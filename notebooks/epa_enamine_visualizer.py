import itertools
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    cross_val_predict,
    KFold,
    LearningCurveDisplay,
    StratifiedKFold,
)

import samples
import scoring


# from model_selection_epa_multiclass import _make_proba_residuals


def main():
    pass


if __name__ == "__main__":
    main()


def plot_model_scores(
    feature_df,
    train_labels,
    score_tups,
    estimator,
    subsets,
    cv=None,
    sample_weight=None,
    **kwargs,
):
    assert len(subsets) > 0
    results_list = list()
    score_func_dict = dict(score_tups)
    kwargs.update({"pos_label": 0})
    for i, best_features in enumerate(subsets):
        # name_scorer_tups = dict((k, make_scorer(v)) for k, v in score_tups)
        if len(best_features) == 0:
            continue
        else:
            print("Sample weights for model score plots:\n{}".format(sample_weight))
        # print([c for c in best_features if c not in feature_df.columns])
        if isinstance(sample_weight, dict):
            weights = list(sample_weight.values())[i]
        else:
            weights = sample_weight
        results_correct, test_idx_list = scoring.cv_model_generalized(
            estimator=estimator,
            feature_df=feature_df[pd.Index(best_features)],
            labels=train_labels,
            cv=cv,
            return_train=True,
            score_list=score_tups,
            sample_weight=sample_weight,
            **kwargs,
        )
        score_dict = scoring.score_cv_results(
            results_correct, score_func_dict, y_true=train_labels, **kwargs
        )
        exploded = _format_cv_generalized_score(score_dict, i, hue_category="Correct")
        # exploded.reset_index(level=0, names="Test/Train", inplace=True)
        # print("Exploded:\n{}".format(pprint.pformat(exploded)))
        results_list.append(exploded)
        rand_results = scoring.score_randomized_classes(
            estimator=estimator,
            feature_df=feature_df[pd.Index(best_features)],
            labels=train_labels,
            cv=cv,
            score_tups=score_tups,
            return_train=True,
            sample_weight=weights,
            **kwargs,
        )
        # print(rand_results)
        rand_score_dict = scoring.score_cv_results(
            rand_results,
            score_func_dict,
            y_true=train_labels,
            sample_weight=weights,
            pos_label=0,
        )
        rexploded = _format_cv_generalized_score(
            rand_score_dict, i, hue_category="Randomized"
        )
        # rexploded.reset_index(level=0, names="Test/Train", inplace=True)
        # print("Exploded:\n{}".format(pprint.pformat(rexploded)))
        results_list.append(rexploded)
    if len(results_list) == 0:
        print("Results list: {}".format(results_list), flush=True)
        raise ValueError
    else:
        all_results = pd.concat(results_list)
    all_results.reset_index(drop=True, inplace=True)
    print("Plot input columns:\n{}".format(all_results))
    sns.set_theme(
        "notebook", palette=sns.color_palette("colorblind"), style="whitegrid"
    )
    plot = sns.catplot(
        all_results,
        x="Split",
        y="score",
        hue="Labels",
        col="Subset",
        row="Metric",
        errorbar="se",
        aspect=0.5,
        margin_titles=True,
        kind="swarm",
        sharey="row",
    )
    plot.despine(left=True, bottom=False)
    plot.figure.subplots_adjust(wspace=0.0, hspace=0.15)
    return all_results, plot


def plot_dmso_model_displays():
    pass


def plot_clf_model_displays(
    estimator,
    estimator_name,
    train_df,
    train_labels,
    select_params,
    preds=None,
    probs=None,
    subset_dir=None,
    sample_weight=None,
    display_labels=None,
):
    """

    Parameters
    ----------
    select_params
    estimator : pd.BaseEstimator, ClassifierMixin
        Estimator to be used when constructing
    estimator_name : str
    train_df : pd.DataFrame
        Descriptors for training
    train_labels : pd.Series | pd.DataFrame
        Ground truth labels, DataFrame format for multioutput classifiers
    preds : pd.Series | pd.DataFrame
        Class predictions from estimator
    probs : pd.DataFrame
        Class-wise probability predictions from estimator
    subset_dir : str | Path-like
        Location to save figures
    sample_weight : pd.Series
        Previously calculated sample weights
    display_labels : Iterable[str]
    Returns
    -------

    """
    if display_labels is None:
        display_labels = [str(s) for s in train_labels.unique()]
    if is_regressor(estimator):
        cv = KFold(shuffle=True, random_state=0)
    else:
        cv = StratifiedKFold(shuffle=True, random_state=0)
    if probs is None and preds is None:
        cv_results = cross_val_predict(
            estimator=clone(estimator),
            X=train_df,
            y=train_labels,
            cv=cv,
            n_jobs=-2,
            method="predict_proba",
            params={"sample_weight": sample_weight},
        )
        probs = pd.DataFrame(
            cv_results, index=train_labels.index, columns=display_labels
        )[display_labels[0]]
        if subset_dir is not None:
            os.makedirs(subset_dir, exist_ok=True)
            probs.to_csv(
                "{}{}_{}.csv".format(
                    subset_dir, estimator_name, select_params["score_func"]
                )
            )
    # if sample_weight is not None and isinstance(sample_weight, pd.Series):

    rcd = RocCurveDisplay.from_predictions(
        train_labels,
        probs,
        pos_label=0,
        name="DMSO Insolubles",
        plot_chance_level=True,
        sample_weight=sample_weight,
    )
    rcd.ax_.set(ylim=[0, 1.0])
    rcd.ax_.set(xlim=[0, 1.0])
    rcd.figure_.set_dpi(300)
    if subset_dir is not None:
        rcd.figure_.set_dpi(300)
        rcd.figure_.savefig("{}RocCurve_{}.png".format(subset_dir, estimator_name))
    det = DetCurveDisplay.from_predictions(
        train_labels,
        probs,
        pos_label=0,
        name="DMSO Insolubles",
        sample_weight=sample_weight,
    )
    # det.ax_.set(ylim=[0, 1.0])
    # det.ax_.set(xlim=[0, 1.0])
    det.ax_.set(xlim=[0.0, 1.0], ylim=[0.0, 1.0])
    det.figure_.set_dpi(300)
    if subset_dir is not None:
        det.figure_.savefig("{}DetCurve_{}.png".format(subset_dir, estimator_name))
    if is_classifier(estimator):
        if preds is None:
            preds = cross_val_predict(
                estimator=clone(estimator),
                X=train_df,
                y=train_labels,
                cv=cv,
                n_jobs=-2,
                method="predict",
                params={"sample_weight": sample_weight},
            )
        cmd = ConfusionMatrixDisplay.from_predictions(
            y_true=train_labels,
            y_pred=preds,
            display_labels=display_labels,
            normalize="true",
            cmap="Blues",
        )
        cmd.figure_.set_dpi(300)
        if subset_dir is not None:
            cmd.figure_.savefig(
                "{}confusion_matrix_{}.png".format(subset_dir, estimator_name)
            )
    lcd = LearningCurveDisplay.from_estimator(
        estimator=clone(estimator),
        X=train_df,
        y=train_labels,
        train_sizes=np.linspace(0.05, 0.9, num=15),
        scoring=select_params["scorer"],
        n_jobs=-2,
        shuffle=True,
        random_state=0,
        score_name=select_params["score_name"],
    )
    lcd.figure_.set_dpi(300)
    lcd.ax_.set(ylim=[0, 1.0])
    if subset_dir is not None:
        lcd.figure_.savefig("{}lcd_{}.png".format(subset_dir, estimator_name))
    return rcd, det, lcd, cmd


def _plot_proba_pairs(labels, n, subset_predicts, select_params):
    marker_style = dict(
        color="tab:blue",
        linestyle=":",
        marker="o",
        #  markersize=15,
        markerfacecoloralt="tab:red",
    )
    size = (1 - labels.copy()).clip(lower=0.5) ** 4
    alpha = (1 - labels.copy()).clip(lower=0.1, upper=1.00)
    pred_df = labels.copy()
    pred_df.name = "Ground"
    data_list = list()
    for s_i, sers in enumerate(subset_predicts[n]):
        # print(pd.concat(sers["predict_proba"]["test"]).iloc[:, 0], flush=True)
        try:
            df = pd.concat(sers[select_params["score_func"]]["test"])
            if len(df.shape) > 1:
                print(df)
                df = df.iloc[:, 0]
            # df = pd.concat([s.iloc[:, 0] for s in sers["predict_proba"]["train"]])
            # df.columns = ["{}_{}".format(s_i, f_i) for f_i in np.arange(df.shape[1])]
            # print(df.head(), df.shape)
        except TypeError:
            try:
                print("Exception caught!\n\n\n")
                print(sers)
                df = pd.concat(
                    [
                        pd.concat(
                            [a[select_params["score_func"]]["train"] for a in sers]
                        )
                    ],
                    axis=1,
                )
            except TypeError:
                df = pd.concat([a for a in sers[select_params["score_func"]]["train"]])
        df.name = "Subset_{}".format(s_i)
        data_list.append(df.sort_index())
    data_list.append(pred_df.sort_index())
    pred_df = pd.concat(
        data_list, axis=1
    )  # , left_index=True, right_index=True, how="inner")
    ### Rework this. Made to avoid circular import.
    # resid_df = _make_proba_residuals(pred_df, labels=labels.loc[pred_df.index])
    resid = dict()
    for col_a in np.arange(pred_df.shape[1]):
        for col_b in np.arange(col_a + 1, pred_df.shape[1]):
            resid[(col_a, col_b)] = pred_df.iloc[:, col_a] - pred_df.iloc[:, col_b]
    resid_df = pd.DataFrame.from_dict(resid)

    # pprint.pp(resid_df)
    # full_kde = True
    # full_pred = pred_df.copy()
    sample_fraction = 0.5
    # pred_df = pred_df.groupby(["Ground"], observed=True).sample(frac=sample_fraction, replace=True)
    # print(pred_df["Ground"])
    flat_resids = resid_df.copy()
    flat_resids.columns = resid_df.columns.map(
        dict(enumerate(pred_df.drop(columns="Ground").columns))
    ).to_flat_index()
    together_df = pd.concat([resid_df, pred_df.drop(columns="Ground")], axis=1)
    # flat_resids = flat_resids.merge(pred_df["Ground"], right_index=True, left_index=True)
    # fig = plt.figure(figsize=(8, 8), dpi=500)
    # pg = sns.PairGrid(data=together_df, x_vars=pred_df.drop(columns="Ground").columns.tolist(), y_vars=resid_df.columns.tolist(), diag_sharey=False, corner=True)
    # pg = pg.map_lower(sns.kdeplot, data=pred_df, hue="Ground", common_norm=False, levels=5, legend=False)
    # pg.map_lower(sns.kdeplot, data=flat_resids[flat_resids["Ground"] == 1].drop(columns="Ground"), sizes=0.1, alpha=0.01)
    # pg.map_lower(sns.kdeplot, data=flat_resids[flat_resids["Ground"] == 0].drop(columns="Ground"), size=2.0, alpha=1.0)
    print(pred_df)
    pred_long = pred_df.melt(id_vars="Ground")
    print(pred_long)
    fg = sns.FacetGrid(
        pred_long, row="variable", col="variable", margin_titles=True, sharey=False
    )
    for i, j in itertools.combinations(pred_df.drop(columns="Ground").columns, r=2):
        for dg in pred_df["Ground"].unique():
            data_group = pred_df.loc[pred_df["Ground"] == dg]
            # print(col[0]*(pred_df.shape[1] - 1) + col[1] - 1)
            sns.scatterplot(
                x=data_group[i],
                y=data_group[i] - data_group[j],
                ax=fg.facet_axis(
                    data_group.columns.get_loc(i), data_group.columns.get_loc(j)
                ),
                size=0.05,
                legend=False,
            )
            sns.scatterplot(
                x=data_group[i],
                y=data_group[j],
                ax=fg.facet_axis(
                    data_group.columns.get_loc(j), data_group.columns.get_loc(i)
                ),
                size=0.05,
                legend=False,
            )
    for i in pred_df.drop(columns="Ground").columns:
        print(pred_long[pred_long["variable"] == i])
        sns.histplot(
            pred_long[pred_long["variable"] == i].drop(columns="variable"),
            x="value",
            hue="Ground",
            stat="density",
            common_norm=False,
            common_bins=False,
            ax=fg.facet_axis(pred_df.columns.get_loc(i), pred_df.columns.get_loc(i)),
            legend=False,
        )
        # fg.fig.axes[col[0]+ col[1] - 1].scatter(x=pred_df[col[0]], y=flat_resids[col])
        # pg = pg.fig.axes[col[0]*(resid_df.shape[1] + 1) + col[1]].scatter(x=pred_df[col[0]], y=flat_resids[col])
        # pg = pg.map_upper(sns.scatterplot, data=flat_resids, size=0.75, alpha=.6)
    plt.close()
    return fg


def plot_proba_distances(
    feature_df, labels, model_subsets_dict, name_model_dict, path_dict=None
):
    triple_tup = list()
    for i, name in enumerate(model_subsets_dict.keys()):
        triple_tup = [
            (name, name_model_dict[name], subset) for subset in model_subsets_dict[name]
        ]
    model_dist_plots = samples.cv_model_prediction_distance(
        feature_df, labels, triple_tup
    )
    if path_dict is not None:
        for i, (name, plot) in enumerate(model_dist_plots.items()):
            plot.savefig(
                "{}subset{}/{}_prediction_distance_plot.png".format(
                    path_dict["exp_dir"], i, name
                )
            )
    return model_dist_plots


def _format_cv_generalized_score(results_correct, i, hue_category=None):
    # results_dict_df = [pd.DataFrame.from_records(r) for r, v in results_correct.items()]
    # print(results_dict_df)
    # print(results_dict_df.explode().reset_index())
    # print([ser.explode().reset_index() for col, ser in results_dict_df.items()])
    # exploded_list = results_dict_df.melt()
    """
    exploded_list = [
            ser.explode()
            .reset_index()
            .melt(id_vars="index", value_name="Score", ignore_index=False)
            .reset_index(drop=True)
            for col, ser in results_dict_df.items()
        ]
    print(exploded_list)
    if len(exploded_list) > 1:
        exploded = pd.concat(exploded_list) # , keys=results_dict_df.columns)
    else:
        exploded = exploded_list[0]
    """
    exploded = results_correct.melt(
        id_vars=["Metric", "Split", "CV Fold"], value_name="score"
    )
    exploded.insert(loc=0, column="Subset", value=i)
    if hue_category is not None:
        exploded.insert(loc=0, column="Labels", value=hue_category)
    return exploded
