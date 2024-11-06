import copy
import pprint

import numpy as np
import pandas as pd
import scipy
import sklearn.utils
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import RobustScaler

from feature_selection.importance import logger
from utils.features import compute_gram


# from features import compute_gram


def calculate_vif(feature_df, labels=None, model='ols', sample_wts=None, generalized=True,
                  n_jobs=-1, verbose=0):
    from sklearn.linear_model import ElasticNetCV
    ElasticNetCV().
    if model == "ols":
        model = LinearRegression(n_jobs=n_jobs)
    elif model == 'elasticnet' or model == "elastic-net":
        model = ElasticNetCV(tol=5e-2, )
    assert feature_df.shape[1] > 1
    vif_dict = dict()
    for col in feature_df.columns:
        training = feature_df.drop(columns=col)
        y = feature_df[col]
        .fit(X=training, y=y, sample_weight=sample_wts)
        r_squared = ols.score(X=training, y=y, sample_weight=sample_wts)
        print(r_squared)
        if r_squared != 1.:
            vif_i = 1. / (1. - r_squared)
        else:
            logger.error('Feature had OOB OLS R2 score of 0!!!! Feature name: {}'.format(col))
            vif_i = 1000000
        # print('Original VIF: {}'.format(vif_i))
        if generalized:
            vif_dict[col] = (vif_i, np.log(vif_i) / np.sqrt(2 * (feature_df.shape[1])))
            # print('Generalized VIF: {}'.format(vif_dict[col]))
        else:
            vif_dict[col] = vif_i
    # print('\n\nVIF length: {}\n\n'.format(len(vif_dict.keys())))
    if generalized:
        vif_ser = pd.DataFrame.from_dict(vif_dict, orient='index', columns=['VIF', 'GVIF'])
    else:
        vif_ser = pd.Series(vif_dict)
    if verbose > 0:
        logger.info(pprint.pformat(vif_ser, compact=True))
    assert not vif_ser.empty
    return vif_ser


def stochastic_vif(feature_df, mi_ser, num_select=100, vif_cut=5, scoring='logistic', sample_wts=None, feat_wts='auto',
                   chunk_size=3, n_jobs=-1, scale_data=True,
                   verbose=1):
    if scale_data:
        df = RobustScaler(with_centering=False, quantile_range=(15, 85), unit_variance=True).fit_transform(feature_df)
    df = pd.DataFrame(data=df, index=feature_df.index, columns=feature_df.columns)
    logger.info('Number of features for VIF selections: {}'.format(mi_ser.size))
    alpha = 0.3
    corr_ser = mi_ser.copy()
    select_set = list()
    if feat_wts == 'auto':
        feat_wts = corr_ser
    if num_select is None:
        num_select = 1
    chunk = chunk_size
    cut_list = list()
    while len(select_set) < num_select and corr_ser.shape[0] > chunk:
        # print('Selected: {}'.format(len(select_set)))
        feats = corr_ser.sample(weights=feat_wts[corr_ser.index], n=chunk_size).index.tolist()
        corr_ser.drop(feats)
        feat = copy.deepcopy(feats)
        feat.extend(select_set)
        # print(feat)
        if feat is None or len(feat) == 0:
            logger.warning('WARNING!!! No features left!')
            break
        vif_ser = calculate_vif(df[feat], scoring=scoring, sample_wts=sample_wts, n_jobs=n_jobs, verbose=verbose)
        # probs = max((alpha * vif / 100.), 1.0)
        # for np.random.(n=1, p=0.5)
        # cuts = vif_ser[(vif_ser.iloc[:, 1] > alpha) | (vif_ser.iloc[:, 0] > vif_cut)]
        cuts = vif_ser[(vif_ser.iloc[:, 1] * (vif_ser.iloc[:, 0]) > vif_cut)]
        logger.info(pprint.pformat(cuts, compact=True))
        if not cuts.empty:
            cut_list.append((tuple(cuts.index.tolist()), vif_ser))
            for c in cuts.index.to_list():
                if c not in feat:
                    select_set.remove(c)
        [select_set.append(f) for f in feat if f not in cuts.index]
        chunk = min((chunk_size, corr_ser.shape[0]))
        if verbose > 1:
            logger.info(pprint.pformat(vif_ser.sort_values(by='VIF', ascending=False), compact=True))
    if verbose > 0:
        # pprint.pp(vif_ser.sort_values(ascending=False), compact=True, width=80)
        # print('Features cut by VIF:')
        # [print('{:.50s}'.format(x[0])) for x in cut_list]
        logger.info('Top 25 selected features')
        logger.info(pprint.pformat(vif_ser.sort_values(by='VIF', ascending=False).iloc[:25]))
    return vif_ser, select_set, cut_list


def vif_bad_apples(feature_df, target, model='ols', sample_wts=None, class_wts=None, generalized=True, n_jobs=1,
                   verbose=0, kernel='laplacian'):
    # In progress function to remove features causing multicollinearity with several other functions.
    coefs_list, r_two_list = list(), list()
    if model == 'lasso':
        gram, normed_weights = compute_gram(feature_df, target, sample_weights=sample_wts)
        reg = LassoCV(n_jobs=n_jobs, tol=5e-2, max_iter=5000, random_state=0, selection='random')
        print(reg.coef_)
    weak_list = list()
    y = feature_df[target]
    while True:
        training = feature_df.drop(columns=target)
        if model == 'ols':
            reg = LinearRegression(n_jobs=n_jobs).fit(X=training, y=y, sample_weight=sample_wts)
            coefs = pd.Series(data=reg.coef_, index=training.index)
            weak_one = coefs.idxmin()
            r_squared = reg.score(X=training, y=y, sample_weight=sample_wts)
            if r_squared != 1.:
                vif_i = 1. / (1. - r_squared)
            else:
                logger.error('Feature had OOB OLS R2 score of 0!!!! Feature name: {}'.format(col))
                vif_i = 1000000


def repeated_stochastic_vif(feature_df, mi_ser, target=None, num_select=100, cut=5., sample_wts='balanced',
                            class_wts='balanced',
                            feat_wts='auto', scoring='logistic', model_size=25, cv=10,
                            n_jobs=1, verbose=1):
    df = RobustScaler(with_centering=False, quantile_range=(15, 85), unit_variance=True).fit_transform(feature_df)
    df = pd.DataFrame(data=df, index=feature_df.index, columns=feature_df.columns)
    print('Number of features for CV Stochastic VIF selections: {}'.format(mi_ser.size))
    print('Running batches of {} {} times'.format(model_size, cv))
    corr_ser = mi_ser.copy()
    if feat_wts == 'auto':
        feat_wts = mi_ser
    if num_select is None:
        num_select = 1
    if class_wts == 'balanced' and target is not None:
        class_wts = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(target),
                                                                    y=target)
    vif_list = list()
    # Start with 0.5 to avoid division by 0 later. Chosen to agree with no-info Bayesian prior.
    votes = pd.DataFrame(np.zeros((mi_ser.index.size, 2), dtype=np.float64), index=corr_ser.index,
                         columns=['Appearances', 'Cuts']).add(0.5)
    for i in list(range(cv)):
        # print('Selected: {}'.format(len(select_set)))
        feats = corr_ser.sample(weights=scipy.special.softmax(feat_wts), axis=0, n=model_size)
        votes.loc[feats.index, 'Appearances'].add(1)
        vif_ser = calculate_vif(feature_df=df[df.columns.intersection(feats.index)], labels=target, scoring=scoring,
                                sample_wts=sample_wts,
                                generalized=False, n_jobs=n_jobs,
                                verbose=verbose)
        votes.loc[vif_ser[vif_ser > 10.].index, 'Cuts'].add(1)
        vif_list.append(vif_ser)
        # crit_ser = vif_ser.divide(feats)
        # pprint.pp(crit_ser.sort_values(ascending=False).iloc[:5], compact=True)
        # votes['Cuts'][crit_ser[crit_ser < cut].index].add(1)
        # cuts = vif_ser[(vif_ser.iloc[:, 1] > alpha) | (vif_ser.iloc[:, 0] > vif_cut)]
        # cuts = vif_ser[(vif_ser.iloc[:, 1] * (vif_ser.iloc[:, 0]) > vif_cut)]
        if verbose > 2:
            logger.info(pprint.pformat(vif_ser.sort_values(ascending=False), compact=True))
        # pprint.pp(vif_ser.sort_values(ascending=False), compact=True, width=80)
        # print('Features cut by VIF:')
        # [print('{:.50s}'.format(x[0])) for x in cut_list]
    logger.info(pprint.pformat(votes.sort_values(by='Appearances', ascending=False).iloc[:15], compact=True))
    '''
    try:
        all_ndices = set([ser.index for ser in vif_list])
        [all_df.merge(right=df.reindex_like(mi_ser)['VIF_{}'.format(i)], left_index=True, how='outer') for i, df in
         enumerate(vif_list)]
        vif_stats = pd.DataFrame(current_data=[np.nanmean(all_df, axis=1), np.nanstd(all_df, axis=1)])
    except:
        print('Failure')
        '''
    crits = mi_ser.multiply(votes['Appearances'].squeeze().divide(votes['Cuts'].squeeze())).sort_values(ascending=False,
                                                                                                        inplace=True)
    logger.info(pprint.pformat(crits.iloc[:10], compact=True))

    vif_score_df = pd.concat(vif_list, axis=1, keys=['VIF_{}'.format(i) for i in list(range(cv))])
    # logger.info(pprint.pformat(all_df, compact=True))
    try:
        logger.info(
            pprint.pformat(pd.concat([vif_score_df.mean(axis=1), vif_score_df.std(axis=1)], axis=1), compact=True))
        vif_stats = pd.DataFrame(data=pd.concat((vif_score_df.mean(axis=1), vif_score_df.std(axis=1)), axis=1))
    except:
        print('Failed to concatenate on 1 axis', flush=True)
        vif_stats = pd.DataFrame(data=pd.concat((vif_score_df.mean(axis=1), vif_score_df.std(axis=1)), axis=0))
    vif_stats.columns = pd.Index(['Mean', 'StdDev'], tupleize_cols=False)
    vif_stats.sort_values(by='Mean', ascending=False, inplace=True)
    logger.info(pprint.pformat(vif_stats))
    return crits, vif_score_df, vif_stats, votes


def sequential_vif(feature_df, sort_ser, vif_cut=10, verbose=0):
    # Inputs: Features, Target correlations: Series, Covariance, correlation with target, VIF/Condition Num.
    # https://www.sciencedirect.com/science/article/abs/pii/S1386142521012294
    corr_ser = sort_ser.copy()
    corr_ser.sort_values(ascending=False, inplace=True)
    if type(corr_ser) is pd.DataFrame:
        corr_ser = corr_ser.squeeze()
    feats_df = feature_df[corr_ser.index].copy()
    selected_dex = pd.Index([])
    cut_dex = pd.Index([])
    for feat in corr_ser.index:
        vif_ser = calculate_vif(feats_df)
        '''
        if vif_ser[feat] < vif_cut:
            selected_dex.append(feat)
        cuts = vif_ser[vif_ser > vif_cut].index
        if cuts.size > 0:
            cut_dex.append(cuts)
            feats_df.drop(columns=cuts)
        '''
    # if verbose > 0:
    # pprint.pp(vif_ser.sort_values(ascending=False), compact=True, width=80)
    # print('Features cut by VIF:')
    # [print('{:.50s}'.format(x[0])) for x in cut_list]
    # print('Top 25 selected features')
    # [print('{:.50s}'.format(x[0])) for x in selected_list[:25]]
    return vif_ser, selected_dex, cut_dex
