import copy
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


# Prototype functions to calculate and filter features that display a high level of interaction.

def filter_covariants(feature_df, labels, cov_thresh, n_features_out, sort_ser=None, cov_method='pearson',
                      cov_matrix=None, save_cov=True):
    feature_df = feature_df.copy()
    if cov_matrix is None:
        cov_matrix = feature_df.corr(method=cov_method)
    if sort_ser is not None:
        cov_matrix = cov_matrix[sort_ser[feature_df.columns].sort_values(ascending=False).index]
        cov_matrix = cov_matrix.loc[cov_matrix.columns]
        feature_df = feature_df[cov_matrix.columns]
    col_list = feature_df.columns.tolist()
    c = 0
    for feat in col_list:
        if feat not in feature_df.columns:
            continue
        over_thresh = cov_matrix[cov_matrix[feat] > cov_thresh].columns
        over_thresh = [c for c in over_thresh if c != feat]
        if len(over_thresh) > 0:
            feature_df.drop(columns=over_thresh, inplace=True)
            cov_matrix = cov_matrix.drop(columns=over_thresh).drop(index=over_thresh)
        c += 1
        if c >= n_features_out:
            break
    return feature_df.columns.symmetric_difference(pd.Index(col_list)).tolist(), cov_matrix


def filter_mi_interactions(feature_df, labels, n_features_out, univariate_mi, n_neighbors=15, univariate_quintile=0.75,
                           bivariate_mi=None):
    univariate_mi.index = feature_df.columns
    univariate_mi.sort_values(ascending=False, inplace=True)
    col_size = min(2 * n_features_out, feature_df.shape[1])
    feature_df = feature_df[univariate_mi.index].copy().iloc[:, :col_size]
    n = n_features_out
    '''     
    while True:
    vifs = calculate_vif(data=feature_df.iloc[:, :n_features_out])
    over_vif = vifs[vifs > 10]
    if not over_vif.empty:
        n = n + over_vif.size
        feature_df.drop(over_vif, inplace=True)

        continue'''
    feat_list = feature_df.columns.tolist()
    '''
    if bivariate_mi is None:
        bivariate_path = '{}bivariates_all_train_cv30.csv'.format(os.environ.get('MODELS_DIR'))
        bivariate_mi = pd.read_csv(bivariate_path, index_col=True)
    col_list = [c for c in bivariate_mi.columns if c in feat_list]
    bivariate_mi = utils.math.norm_df_by_trace(bivariate_mi.loc[col_list, col_list])
    '''
    col_list = copy.deepcopy(feat_list)
    drop_list = list()
    c = 0
    for i, feat in enumerate(col_list):
        if feat in drop_list:
            continue
        mi = mutual_info_regression(feature_df.iloc[:, i + 1:], feature_df[feat], n_neighbors=n_neighbors,
                                    random_state=0, n_jobs=-1) / univariate_mi.iloc[i]
        self_info = mutual_info_regression(feature_df[feat].to_frame(), feature_df[feat], n_neighbors=n_neighbors,
                                           random_state=0, n_jobs=-1)
        normed_mi = pd.Series(mi / self_info, index=feature_df.columns[i + 1:])
        if not normed_mi.empty:
            over_thresh = feature_df.columns[i + 1:][normed_mi > 2.5]
            [drop_list.append(c) for c in over_thresh if c not in drop_list]
            # feature_df.drop(columns=over_thresh, inplace=True)
            # bivariate_mi = bivariate_mi.drop(columns=over_thresh).drop(index=over_thresh)
        c += 1
        if c >= n_features_out:
            break
    print('Drop List for MI Interactions')
    print(drop_list)
    return drop_list
