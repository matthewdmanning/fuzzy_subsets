from feature_selection.bivariate_filters import filter_covariants
from feature_selection.mutual_info_tools.jmi_homebrew import balanced_mi_y


def pearson_filtered(
    feature_df,
    labels,
    n_features_out,
    var_thresh=0.1,
    covar_thresh=0.9,
    corr_thresh=None,
    cov_method="pearson",
):
    col_size = min(2 * n_features_out, feature_df.shape[1])
    feature_df = feature_df.iloc[:, :col_size].copy()
    if var_thresh is not None:
        feat_std = feature_df.std() / feature_df.mean(axis=0)
        feature_df.drop(
            columns=feat_std[feat_std.abs() < var_thresh].index, inplace=True
        )
    print(feature_df.shape)
    print(labels.shape)
    feat_corr = feature_df.corrwith(labels).abs().sort_values(ascending=False)
    # feat_corr = pointbiserialr(x=feature_df, y=labels.to_numpy().reshape(-1, 1)).pvalue.abs().sort_values(ascending=False)
    if corr_thresh is not None:
        feature_df.drop(columns=feat_corr[feat_corr < corr_thresh].index, inplace=True)
    if covar_thresh is not None:
        if (
            n_features_out is not None
            and int(1.5 * n_features_out) > feature_df.shape[1]
        ):
            feat_cov = feature_df.iloc[:, : int(1.5 * n_features_out)].corr(
                method=cov_method
            )
        else:
            feat_cov = feature_df.corr(method=cov_method)
        cov_drop, cov_matrix = filter_covariants(
            feature_df,
            labels,
            covar_thresh,
            n_features_out,
            sort_ser=feat_corr,
            cov_method=cov_method,
            cov_matrix=feat_cov,
        )
        feature_df.drop(columns=cov_drop, inplace=True)
    selected = feat_corr[feature_df.columns].iloc[:n_features_out].index
    if selected.empty:
        print("\n\n Selected features from PCC is empty!!!\n\n")
    return feature_df[selected]


def get_mi_features(
    feature_df,
    labels,
    n_features_out,
    test_feature_df=None,
    save_mi=False,
    n_neighbors=7,
    filter_interactions=True,
):
    univariate = balanced_mi_y(feature_df.copy(), labels, n_neighbors=7)
    # dropped_feats = filter_mi_interactions(feature_df, labels, n_features_out=n_features_out, univariate_mi=univariate)
    # insol_list, sol_list, label_list = list(), list(), list()
    # for dev_X, dev_y, eva_X, eva_y in utils.cv_tools.split_df(feature_df.copy(), labels, indices_list=index_list):
    # insol_hbond, sol_hbond = dict(), dict()
    # mean_insol_df = np.dstack(([d.to_numpy() for d in insol_list])).mean(axis=2)
    # mean_sol_df = np.dstack(([d.to_numpy() for d in sol_list])).mean(axis=2)
    # mean_mi_df = pd.concat(label_list, axis=1).mean(axis=1)
    # mean_mi_df.to_csv('{}hbond_30x.csv'.format(os.environ.get('MODELS_DIR')))
    # pd.DataFrame(mean_insol_df, index=feature_groups['HBond'], columns=feature_groups['HBond']).to_csv('{}insol_hbond_bivary.csv'.format(os.environ.get('MODELS_DIR')))
    # pd.DataFrame(mean_sol_df, index=feature_groups['HBond'], columns=feature_groups['HBond']).to_csv('{}sol_hbond_bivary.csv'.format(os.environ.get('MODELS_DIR')))
    # pprint.pp(pd.DataFrame(mean_mi_df), compact=True, width=140)
    return univariate.index[:n_features_out]  # .drop(dropped_feats).tolist()
