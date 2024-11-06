import inspect
import os.path
import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    get_scorer,
    get_scorer_names,
    RocCurveDisplay,
)

from data.constants import names_dict, score_dict
from data_handling.persistence import logging


def clf_score_report_from_fitted(results_dict, weights, metrics, displays):
    score_dict = dict([(s, get_scorer(s)) for s in get_scorer_names()])
    metrics_dict, display_dict = dict(), dict()
    for mod_name, mod in results_dict.items():
        [v.update((mod_name, dict())) for v in mod.values()]
        if mod_name != "true":
            for s, split_name in enumerate(mod.keys()):
                if split_name == "true":
                    continue
                [v[mod_name].update((split_name, list())) for v in mod.values()]
                for cv_num in range(len(mod[split_name])):
                    score_kwargs = {
                        "sample_weight": weights[split_name][cv_num],
                        "pos_label": 0,
                        "average": "weighted",
                        "adjusted": "False",
                    }
    rocfig, rocax = plt.subplots()
    rocfig.set_dpi(600)
    return rocfig, rocax


def get_confusion_display(
    estimator, X, y_true, class_names=None, axes=None, colors=None
):
    if colors is None:
        colors = plt.get_cmap("coolwarm").reversed(name="warmcool")
    cmd = ConfusionMatrixDisplay.from_estimator(
        X=X,
        y=y_true,
        estimator=estimator,
        ax=axes,
        display_labels=class_names,
        cmap=colors,
        xticks_rotation=0.3,
        normalize="true",
        values_format=".4f",
    )
    return cmd


def roc_bounds(
    model_dict, fig, ax, label_insol, results_dict, estimators, feat_df, plot_cvs=False
):
    from sklearn.metrics import auc

    mod_colors = ["b", "r", "g"]
    for j, mod_name in enumerate(model_dict.keys()):
        disp_list = model_dict[mod_name]
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        for i, disp in enumerate(disp_list):
            interp_tpr = np.interp(mean_fpr, disp.fpr, disp.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(disp.roc_auc)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        print(
            "{}: Mean AUC: {}. StdDev AUC: {}".format(
                names_dict[mod_name], mean_auc, std_auc
            )
        )
        ax.plot(
            mean_fpr,
            mean_tpr,
            color=mod_colors[j],
            lw=1.5,
            alpha=0.8,
            label=r"%s Mean ROC (AUC = %0.3f $\pm$ %0.3f)"
            % (names_dict[mod_name], mean_auc, std_auc),
        )
        if plot_cvs:
            for i, disp in enumerate(disp_list):
                disp_list.plot(
                    ax=ax,
                    name="{}: CV {}".format(names_dict[mod_name], i),
                    alpha=0.2,
                    lw=1,
                    color=mod_colors[j],
                )
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=mod_colors[j],
            alpha=0.2,
            label=r"$\pm$ 1 Std. Dev.",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    ax.set(
        xlim=[-0.01, 1.01],
        ylim=[0, 1.05],
        title="ROC Plot: DMSO Solubility Model",
    )
    ax.legend(loc="lower right")
    return fig, ax


def plot_results(
    results_dict, indices, models, feats, weights=None, insol_label=0, save_dir=None
):
    mod_colors = ["b", "r", "g"]
    # cmd_stub = '{}{}_scaled_confusion.png'.format(CV_DIR, outputs['params']['n_feats'], cv)
    # roc_stub = '{}{}_scaled_roc.png'.format(CV_DIR, outputs['params']['n_feats'], cv)
    roc_dict, det_dict = {}, {}
    config, conax = plt.subplots()
    rocfig, rocax = plt.subplots()
    detfig, detax = plt.subplots()
    config.set_dpi(600)
    rocfig.set_dpi(600)
    detfig.set_dpi(600)
    scores = dict([(metric_name, dict()) for metric_name in score_dict.keys()])
    model_dicts = list(scores[list(score_dict.keys())[1]])
    index_multi = pd.MultiIndex.from_product(
        iterables=[[k for k in results_dict.keys() if k != "true"], scores.keys()],
        names=["Model", "Metric"],
    )
    col_multi = pd.MultiIndex.from_product(
        iterables=[["Mean", "Std"], ["dev", "eval"]], names=["Stat", "Split"]
    )
    score_df = pd.DataFrame(data=pd.NA, index=index_multi, columns=col_multi)
    for m, (mod_name, mod) in enumerate(results_dict.items()):
        print(scores.items(), flush=True)
        for v in scores.values():
            v[mod_name] = dict()
        roc_dict[mod_name] = list()
        for s, split_name in enumerate(mod.keys()):
            if mod_name == "true":
                continue
            for v in scores.values():
                v[mod_name][split_name] = list()
            for metric, fnc in score_dict.items():
                score_list = list()
                for cv_num in range(len(mod[split_name])):
                    score_kwargs = {
                        "sample_weight": weights[split_name][cv_num],
                        "pos_label": insol_label,
                        "average": "weighted",
                        "adjusted": "False",
                    }
                    rel_kwargs = dict(
                        [
                            (k, v)
                            for k, v in score_kwargs.items()
                            if k in inspect.getfullargspec(fnc)
                        ]
                    )
                    score_list.append(
                        fnc(
                            y_true=results_dict["true"][split_name][cv_num],
                            y_pred=results_dict[mod_name][split_name][cv_num],
                            **rel_kwargs
                        )
                    )
                score_df.loc[(mod_name, metric), ("Mean", split_name)] = np.mean(
                    score_list
                )
                if len(score_list) > 1:
                    score_df.loc[(mod_name, metric), ("Std", split_name)] = np.std(
                        score_list
                    )
                else:
                    score_df.loc[(mod_name, metric), ("Std", split_name)] = "N/A"
            for cv_num in range(len(mod[split_name])):

                # output_name = '{} {} {}'.format(name_dict[mod_name], name_dict[split_name], cv_num)
                # conf_disp = get_confusion_display(estimator=models[mod_name][cv_num], X=X.iloc[indices[
                # split_name][cv_num]], y_true=results_dict['true'][split_name][cv_num], class_names=[
                # 'Insoluble', 'Soluble'], axes=conax, colors='Blues')
                if split_name == "eval":
                    roc_dict[mod_name] = [
                        RocCurveDisplay.from_estimator(
                            estimator=models[mod_name][cv_num],
                            X=feats.iloc[indices[split_name][cv_num], :],
                            y=results_dict["true"][split_name][cv_num],
                            sample_weight=weights[split_name][cv_num],
                            pos_label=insol_label,
                            name=names_dict[mod_name],
                        )
                        for cv_num in list(range(len(models[mod_name])))
                    ]
                    det_dict[mod_name] = [
                        DetCurveDisplay.from_estimator(
                            estimator=models[mod_name][cv_num],
                            X=feats.iloc[indices[split_name][cv_num], :],
                            y=results_dict["true"][split_name][cv_num],
                            sample_weight=weights[split_name][cv_num],
                            pos_label=insol_label,
                            name=names_dict[mod_name],
                        )
                        for cv_num in list(range(len(models[mod_name])))
                    ]

                    [new_roc.plot(ax=rocax) for new_roc in roc_dict[mod_name]]
                    [new_det.plot(ax=detax) for new_det in det_dict[mod_name]]

                    # roc_plt.plot(ax=rocax, alpha=0.2, name=output_name)
                    logging.info(
                        pprint.pformat(
                            "ROC-AUC for {} Eval Set: {:.4f}".format(
                                mod_name, roc_dict[mod_name][cv_num].roc_auc
                            )
                        )
                    )
    logging.info("Scores for CV Runs:")
    logging.info(pprint.pformat(score_df))
    i = 1
    score_path = "{}score_report_{}.csv".format(save_dir, i)
    while os.path.isfile(score_path):
        i += 1
    logging.info(
        pprint.pformat(score_df.to_string(float_format="{:.5f}"), compact=True)
    )
    score_df.to_csv(score_path, sep="\t", float_format="{:.5f}")
    null_rocs = [k for k, v in roc_dict.items() if len(v) < 2]
    [roc_dict.pop(k) for k in null_rocs]
    null_dets = [k for k, v in det_dict.items() if len(v) < 2]
    [det_dict.pop(k) for k in null_dets]
    try:
        if feats is not None:
            fig, ax = roc_bounds(
                fig=rocfig,
                ax=rocax,
                label_insol=insol_label,
                estimators=models["roc"],
                results_dict=results_dict,
                model_dict=models,
                feat_df=feats,
            )
    except:
        print("Failed to save ROC figure!")
    try:
        rocfig.savefig(
            "{}roc_{}.png".format(save_dir, "_".join(roc_dict.keys())),
            dpi=600,
            format="png",
            transparent=True,
        )
        detfig.savefig(
            "{}det_{}.png".format(save_dir, "_".join(det_dict.keys())),
            dpi=600,
            format="png",
            transparent=True,
        )
    except:
        print("Failed to save ROC figure!")
        # for score_name, scorer in score_dict.items():
        #     score_results[score_name] = scorer(y_true=results_dict['true'][split_name][cv_num], y_pred=results_dict[mod_name][split_name][cv_num], **score_kwargs)
