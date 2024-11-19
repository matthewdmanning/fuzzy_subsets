import os

import pandas as pd

from feature_selection.mutual_info_tools import jmi_homebrew


def mi_bivariate(feature_df, labels, save_dir=None):
    """
    cross_mutual_df = jmi_homebrew.mi_mixed_types(feature_df=feature_df)
    cross_mutual_df.to_csv('{}features_mi_unweighted.csv'.format(save_dir))
    logger.info(cross_mutual_df.head())
    """
    if os.path.isdir("{}balanced_mi_all_train_cv30.csv".format(save_dir)):
        mi_ser = pd.read_csv("{}balanced_mi_all_train_cv30.csv".format(save_dir))
    else:
        mi_ser = jmi_homebrew.balanced_mi_y(feature_df=feature_df, labels=labels)
        mi_ser.to_csv("{}balanced_mi_all_train_cv30.csv".format(save_dir))
    if len(mi_ser.squeeze().shape) == 2:
        mi_ser = mi_ser.mean(axis=1)
        mi_ser.sort_values(ascending=False, inplace=True)
        logger.info(mi_ser.iloc[:50])
    if (
        False
    ):  # os.path.isdir('{}conditional_mis_all_train_neg.csv'.format(save_dir)) and os.path.isdir('{}conditional_mis_all_train_pos.csv'.format(save_dir)):
        condition_list = list()
        condition_list.append(
            pd.read_csv("{}conditional_mis_all_train_neg.csv".format(save_dir))
        )
        condition_list.append(
            pd.read_csv("{}conditional_mis_all_train_pos.csv".format(save_dir))
        )
    else:
        condition_list = jmi_homebrew.condition_by_label(
            feature_df=feature_df, labels=labels
        )
        condition_list[0].to_pickle(
            "{}conditional_mis_all_train_neg.pkl".format(save_dir)
        )
        condition_list[0].to_csv("{}conditional_mis_all_train_neg.csv".format(save_dir))
        condition_list[1].to_pickle(
            "{}conditional_mis_all_train_pos.pkl".format(save_dir)
        )
        condition_list[1].to_csv("{}conditional_mis_all_train_pos.csv".format(save_dir))

    logger.info("Top bivariate measure pairs:")
    top_pairs = condition_list[0].add(condition_list[1]).stack().nlargest(60)
    print(top_pairs, flush=True)
    logger.info(top_pairs.tolist())
    bivariates = jmi_homebrew.bivariate_conditional(
        feature_df, labels, x_y_mi=mi_ser, conditional_dfs=condition_list
    )
    bivariates.to_csv("{}bivariates_all_train_cv30.csv".format(save_dir))
    bivariates.to_pickle("{}bivariates_all_train_cv30.pkl".format(save_dir))


loaded_X, loaded_y = get_interpretable()
mi_path = "C:/Users/mmanning/OneDrive - Environmental Protection Agency (EPA)/qsar-modeling-workflow/models/mutual_info_features/"
main(loaded_X, loaded_y, mi_path)
