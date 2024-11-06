import itertools
import sys
import pickle

import imblearn.metrics
import numpy as np
import pandas as pd
import plotly.express
import scipy.ndimage
import sklearn.feature_selection
import sklearn.linear_model

# from rdkit.ML.Cluster import Butina
# from rdkit.DataManip.Metric import GetTanimotoDistMat, GetTanimotoSimMat
from dmso_model_dev.constants import DISALLOWED
# from rdkit.DataStructs.cDataStructs import CreateFromBinaryText
from dmso_model_dev.DescriptorRequestor import DescriptorGrabber
# from dmso_model_dev.data_handling.descriptor_preprocessing import ks_stats, kde_grouped
import dmso_model_dev.data_handling.padel_categorization as padel_categorization
import dmso_model_dev.data_handling.descriptor_preprocessing as descriptor_preprocessing

sys.path.insert(0, 'C:/Users/mmanning/PycharmProjects/compLoiel/dmso_model_dev/data_handling')
sys.path.insert(0, 'C:/Users/mmanning/PycharmProjects/compLoiel/dmso_model_dev/')

FP_MAPPER = col_mapper = dict(zip((
    'info', 'name', 'version', 'options', 'type', 'radius', 'bits', 'chemicals', 'smiles',
    'inchi', 'inchiKey', 'descriptors'), (
    'FP_SOFTWARE', 'FP_NAME', 'FP_VERSION', 'FP_OPTIONS', 'FP_TYPE', 'FP_RADIUS',
    'FP_BITS', 'FP_COMPOUNDS', 'FP_SMILES', 'FP_INCHI', 'FP_INCHI_KEY',
    'FP_DESCRIPTORS')))

FP_MULTIINDEX = pd.MultiIndex.from_tuples(
    [('FP_SOFTWARE', 'FP_NAME', 'FP_VERSION'), ('FP_OPTIONS', 'FP_TYPE', 'FP_RADIUS', 'FP_BITS'),
     ('FP_COMPOUNDS', 'FP_INCHI', 'FP_INCHI_KEY',
      'FP_DESCRIPTORS')])  # "?smiles=CCCCCCC&headers=true&type=cfp&radius=4&bits=1024
cfp_radius = 4
fp_num_bits = 1024
fp_name = 'cpf'
COL_LABELS = padel_categorization.padel_names
DROP_STRINGS = DISALLOWED
DROP_STRINGS.append('SH')
DATA_KEYS = ['epa_sol', 'epa_in', 'en_sol', 'en_in']
final_dir = os.environ.get("FINAL_DIR")
cache_dir = "C:/Users/mmanning/PycharmProjects/data/tmp_cache.pkl"
DATA_DIR = "{}filtered/PADEL_EPA_ENAMINE_5mM.pkl".format(final_dir)
TRAIN_DIR = "{}filtered/MAXMIN_PADEL_TRAIN.pkl".format(final_dir)
TEST_DIR = "{}filtered/MAXMIN_PADEL_TEST.pkl".format(final_dir)
COMBO_DIR = "{}filtered/PADEL_CFP_COMBO_5mM.pkl".format(final_dir)
TUP_DIR = "{}padel/PADEL_EPA_ENAMINE_5mM_TUPLES.pkl".format(final_dir)
dist_path = "{}circ_fingerprints/TANIMOTO_R{}_{}.pkl".format(final_dir, cfp_radius, fp_num_bits)
DROPPED_DIR = "{}filtered/PADEL_EPA_ENAMINE_5mM_FILTERED.pkl".format(final_dir)

new_desc_list, cfp_df_list, padel_df_list, new_cfps, fp_list = list(), list(), list(), list(), list()
dropped_dict = dict()
dropped_dict['total'] = list()
'''
# TODO: Make INCHI_KEYS the index.
# TODO: Verify agreement with descriptor (ie. Padel) API output.
# TODO: Incorporate class weights into data structures.
# Recover cache.
with (open(TUP_DIR, 'rb') as f):
    tups = pickle.load(f)
with open(cache_dir, 'rb') as f:
    while True:
        try:
            fp_list.append(pickle.load(f))
        except EOFError:
            break
cfp_total = pd.DataFrame.from_records(data=fp_list)
# Change dtypes to categorical to save memory.
cfp_total['name'] = cfp_total['name'].astype('category')
cfp_total['version'] = cfp_total['version'].astype('category')
cfp_total['type'] = cfp_total['type'].astype('category')
cfp_total['radius'] = cfp_total['radius'].astype('category')
cfp_total['DATA_SOURCE'] = cfp_total['DATA_SOURCE'].astype('category')
cfp_total['DMSO_SOLUBILITY'] = cfp_total['DMSO_SOLUBILITY'].astype('category')
cfp_total.rename(mapper=FP_MAPPER, axis='columns', inplace=True)
cfp_total.set_index(keys='FP_INCHI_KEY', drop=False, inplace=True)
drop_comps = cfp_total[cfp_total['FP_SMILES'].isin(DROP_STRINGS)].index
if not drop_comps.size > 0:
    cfp_total.drop(index=drop_comps, inplace=True)
dropped_dict['total'].append(drop_comps)
cfp_dups = cfp_total.index.duplicated()
if not cfp_dups.size > 0:
    cfp_total.drop(index=cfp_dups, inplace=True)
dropped_dict['total'].append(cfp_dups)
if True:
    for key, value in tups.items():
        # Filter out problematic compounds and eliminate NaN while preserving info.
        dropped_indices = pd.concat([value[1][value[1]['SMILES_QSAR'].str.contains(s)] for s in DROP_STRINGS]).index
        [df.drop(index=[a for a in dropped_indices if a in df.index], inplace=True) for df in value]
        comp_na_dropped = value[2].dropna(axis=0, thresh=len(COL_LABELS) - 50).index.symmetric_difference(value[2].index)
        value[2].drop(index=comp_na_dropped, inplace=True)
        feat_na_dropped = value[2].dropna(axis=1, thresh=value[2].shape[0] * .9).columns.symmetric_difference(value[2].columns)
        value[2].drop(index=feat_na_dropped, inplace=True)
        final_na_dropped = value[2].dropna(axis=0).index.symmetric_difference(value[2].index)
        value[2].drop(index=final_na_dropped, inplace=True)
        value[1].drop(value[1].index.difference(value[2].index), inplace=True)
        value[0].drop(value[0].index.difference(value[2].index), inplace=True)
        if 'DESCRIPTORS' in value[1].columns:
            value[1].drop(columns='DESCRIPTORS', inplace=True)
        # print(value[2].columns[:5])
        # print(value[2].isna().sum(axis=0).sort_values(ascending=False)[:7])
        # print(value[2].isna().sum(axis=1).sort_values(ascending=False)[:7])
        # new_desc_df = pd.concat([value[0], value[1], value[2]], axis=1)
        new_desc_df = value[2]
        new_desc_df.attrs['type'] = 'PADEL'
        new_desc_df.attrs['version'] = '2.21'
        if 'epa' in key:
            new_desc_df.__setattr__('source', 'EPA')
            source = 'EPA'
        elif 'en' in key:
            new_desc_df.__setattr__('source', 'ENAMINE')
            source = 'ENAMINE'
        if 'sol' in key:
            new_desc_df.__setattr__('dmso_soluble', 1)
            dmso = 1
        elif 'in' in key:
            new_desc_df.__setattr__('dmso_soluble', 0)
            dmso = 0
        value[1].drop(columns=['PLACEONE', 'PLACETWO'], inplace=True)
        value[0].rename('Solubility', inplace=True)
        [a.rename(mapper=dict(zip(a.columns, padel_categorization.padel_names)), axis='columns', inplace=True) for a in
         new_desc_list if 1443 in a.columns]
        fp_padel_diff = cfp_total.index.symmetric_difference(new_desc_df.index)
        cfp_total.drop(index=[a for a in fp_padel_diff if a in cfp_total.columns], inplace=True)
        new_desc_df.drop(index=[a for a in fp_padel_diff if a in new_desc_df.columns], inplace=True)
        padel_df_list.append(value[1])
        new_desc_list.append(new_desc_df)
        dropped_dict[key] = [dropped_indices, comp_na_dropped, feat_na_dropped, final_na_dropped, fp_padel_diff]

    def splitter(dfs_dict):
        test_list = []
        train_list = []
        for key, df in dfs_dict.items():
            if 'in' in key:
                all_labels = pd.Series(index=df.index, data=0)
            elif 'sol' in key:
                all_labels = pd.Series(index=df.index, data=1)
            train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(df, all_labels, train_size=0.8, random_state=0)
            test_list.append((test_label, test_data))
            train_list.append((train_label, train_data))
        train_df = (pd.concat([x[0] for x in train_list]), pd.concat([x[1] for x in train_list]))
        test_df = (pd.concat([x[0] for x in test_list]), pd.concat([x[1] for x in test_list]))
        return train_df, test_df
    with open(cache_dir, 'ab', buffering=0) as f:
        new_smiles = value[1]['SMILES_QSAR'].to_list()
        new_inchikeys = value[1].index.tolist()
        cfp_header = {'headers': 'true', 'type': fp_name, 'radius': cfp_radius, 'bits': fp_num_bits}
        fingerprinter = DescriptorGrabber(desc_set='cfp', api_url="https://hazard-dev.sciencedataexperts.com/api/rdkit", payload=cfp_header)
        # TODO: Cache file for API calls.
        with requests.session() as r:
            for ik, smi in zip(new_inchikeys, new_smiles):
                fp_return, _ = fingerprinter.make_api_call(payload_input=smi, api_session=r)
                fp_return['INCHI_KEY'] = ik
                if 'descriptors' not in list(fp_return.keys()):
                    print(_)
                    fp_return['descriptors'] =  np.NaN
                    fp_return['FP_BIT_VECTOR'] = np.NaN
                else:
                    bitvec = ''.join([str(i) for i in fp_return['descriptors']]).strip('[').strip(']').replace(', ', '')
                    fp_return['FP_BIT_VECTOR'] = bitvec
                fp_return['DATA_SOURCE'] = source
                fp_return['DMSO_SOLUBILITY'] = dmso
                pickle.dump(fp_return, f)
                new_cfps.append(pd.Series(fp_return))
    # new_cfp_df = pd.concat(new_cfps)
    # new_cfp_df = pd.DataFrame.from_records(new_cfps, orient='columns').rename_axis(columns=FP_MAPPER).set_index(keys='INCHI_KEY', drop=True, inplace=True)
    # new_cfp_df = pd.concat([pd.Series(x[0]) for x in new_cfps], axis=1).T.rename_axis(columns={'inchiKey': 'INCHI_KEY'}).set_index('INCHI_KEY', drop=True, inplace=True)
    # cfp_df_list.append(new_cfp_df)
cfp_total = pd.concat(new_cfps)
cfp_df_path = "{}circ_fingerprints/PADEL_TANIMOTO_COMBINED_5mM.pkl".format(final_dir)
cfp_total.to_pickle(cfp_df_path)
''
# Combine DataFrames
total_padel_df = pd.concat(new_desc_list)
desc_duplicated = total_padel_df.index.duplicated()
if not desc_duplicated.size > 0:
    total_padel_df.drop(index=desc_duplicated.index, inplace=True)
total_padel_df.to_pickle(DATA_DIR)
dropped_dict['total'].append(desc_duplicated)
padel_meta_df = pd.concat(padel_df_list)
padel_dups = padel_meta_df.index.duplicated()
if not padel_dups.size > 0:
    padel_meta_df.drop(index=padel_dups, inplace=True)
dropped_dict['total'].append(padel_dups)
# Save list of dropped compounds
with open(DROPPED_DIR, 'wb') as f:
    pickle.dump(dropped_dict, f)
print(padel_meta_df.columns, cfp_total.columns)
total_meta_df = pd.merge(left=padel_meta_df, right=cfp_total, left_index=True, right_index=True)
#missing_cols = [(i, a) for i, a in enumerate(set(itertools.chain.from_iterable([df[2].columns for df in tups.values()]))) if not all([a in b[2].columns for b in tups.values()])]
del dropped_dict, padel_df_list, cfp_df_list
with open(COMBO_DIR, 'wb') as f:
    pickle.dump(total_meta_df, f)

del fp_list, padel_meta_df, cfp_total
# rdk_vecs = dict([(key, CreateFromBinaryText(bits)) for key, bits in cfp_total['FP_BIT_VECTOR'].items()])
# .drop(2366) for Fe[CO]4 in EPA Soluble set, which gave nan for cfp.

from scipy.spatial.distance import jaccard, rogerstanimoto
from scipy.spatial.distance import pdist
twod_bits = np.vstack(np.array(cfp_total['descriptors'].values))
rogers_arr = pdist(twod_bits['descriptors'].to_numpy, metric='rogerstanimoto')
tani_sim_arr = GetTanimotoSimMat(list(rdk_vecs.values()))
tani_dis_arr = -np.log2(tani_sim_arr)
dist_mat = rogers_arr
'''

# np.save(dist_path, dist_mat)
import deepchem.data
from rdkit.SimDivFilters import MaxMinPicker

with open(COMBO_DIR, 'rb') as f:
    total_meta_df = pickle.load(f)
with open(dist_path, 'rb') as f:
    dist_mat = pickle.load(f)
assert total_meta_df.shape[0] > 0
cweights = total_meta_df['DMSO_SOLUBILITY'].astype(int).replace({1: 1, 0: 10})
rog_set = deepchem.data.NumpyDataset(dist_mat, y=total_meta_df['DMSO_SOLUBILITY'].to_numpy(), w=cweights)
del dist_mat
mms = MaxMinPicker()
n_test = int(total_meta_df.shape[0] / 5)
n_train = int(total_meta_df.shape[0] - n_test)
mms_picks = list(mms.Pick(distMat=rog_set.X, poolSize=int(total_meta_df.shape[0]), pickSize=int(n_test), seed=0))
del rog_set
with open(DATA_DIR, 'rb') as f:
    total_padel_df = pickle.load(f)
# from rdkit.Chem.Fingerprints.ClusterMols import GetDistanceMatrix
# from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetEuclideanDistMat
# dist_arr = GetEuclideanDistMat(list(rdk_vecs.values()))
# from rdkit.DataStructs.cDataStructs import CreateFromBinaryText, SparseBitVect, ExplicitBitVect
# from deepchem.splits import Splitter
# from rdkit.DataStructs import BulkTanimotoSimilarity
assert total_padel_df.shape[0] == total_meta_df.shape[0]
print(total_padel_df.shape, total_meta_df.shape)
# Diversity Optimization

test = total_meta_df.index[mms_picks]
train = total_meta_df.index.symmetric_difference(test)
print(test.shape, '\n', train.shape)
assert test.shape[0] > 0
assert train.shape[0] > 0
train_meta, test_meta = total_meta_df.loc[train], total_meta_df.loc[test]
# train_missing, test_missing = train_meta.loc[train.difference(total_padel_df.index)], test_meta.loc[test.difference(total_padel_df.index)]
# print(train_missing['FP_SMILES'], test_missing['FP_SMILES'])
# TODO: Optimize overlap/intersection code.
# TODO: Store trimmed data and load from file instead of rerunning cleaning code.
# TODO: Implement assert methods at beginning to validate incoming data before any operations.
train_X = total_padel_df.loc[train]
test_X = total_padel_df.loc[test]
train_y = train_meta['DMSO_SOLUBILITY'].to_frame()
test_y = test_meta['DMSO_SOLUBILITY'].to_frame()
check_X_y(train_X, train_y)
check_X_y(test_X, test_y)

with open(TRAIN_DIR, 'wb') as f:
    pickle.dump(train_tup, f)
with open(TEST_DIR, 'wb') as f:
    pickle.dump(test_tup, f)
'''
# Graphing Options
from plotly.express import scatter
import plotly.io as pio
from plotly.express.colors import qualitative, sequential, diverging, colorscale_to_colors
vivid_colors = qualitative.Vivid
sol_colors = [qualitative.Light24[0], qualitative.Light24[13]]
pio.renderers.default = 'browser'
# Load Data
from sklearn.utils.validation import check_X_y
with open(COMBO_DIR, 'rb') as f:
    total_meta_df = pickle.load(f)
with open(TRAIN_DIR, 'rb') as f:
    train_tup = pickle.load(f)
train_X, train_y = train_tup
train_y = train_y.squeeze().astype(int)
sklearn.utils.check_X_y(train_X, train_y, y_numeric=True)
# Cluster and display dimensionality reduction of descriptors
# plotmarkers = pd.concat([total_meta_df['DATA_SOURCE'].loc[train], total_meta_df['DATA_SOURCE'].loc[test]])
# TODO: Implement splitting loop.
estate = [a for a in train_X.columns if 'e-state' in a.lower()]
bcut = [a for a in train_X.columns if 'bcut' in a.lower()]
matrix = [a for a in train_X.columns if 'matrix' in a.lower()]
from dmso_model_dev.models.feature_selector import pd_sparse_feats
sparse = pd_sparse_feats(train_X)
dense = train_X.columns.symmetric_difference(sparse)
# from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SequentialFeatureSelector
# from sklearn.neural_network import MLPClassifier
# from sklearn.decomposition import PCA, KernelPCA
# from sklearn.metrics import make_scorer, adjusted_mutual_info_score
# from functools import partial, partialmethod
print(train_X.shape)

# x_sample = train_X.sample(axis=1, frac=0.2, random_state=0)
# TODO: Optimize this sorting.
sparse_df = train_X[(train_X.nunique(dropna=False, axis=0) <= 2).index]
dense_df = train_X[train_X.columns.symmetric_difference(sparse_df.columns)]

serlist = [(col, ser.value_counts(normalize=True)) for col, ser in train_X.items()]
# sorted(serlist, key=lambda x: max(x[1].sort_values(ascending=False)))
freq_list = [(s[1].iloc[0], s[0]) for s in serlist if s[1].shape[0] > 1]
# sorted(freq_list, key=lambda x: x[0], reverse=True)
x_var = pd.Series(index=train_X.columns).value_counts()
print(x_var[:10])
for col, ser in train_X.items():
    if col in sparse:
        x_var[col] = 1
    else:
        x_var[col] = ser.var()
print(x_var.index.difference(train_X.columns))
x_var = train_X.var()
# plotly.express.scatter(y=x_var.add(1).sort_values(), log_y=True).show()
X_train = train_X[(x_var > 0).index]
n_feats = 100
# Remove correlated features.
not_corr = False
if not_corr:
    X_corr = dense_df.corr(method='kendall').abs()
    x_na_corr = X_corr.isna().astype(int)
    col_na = x_na_corr.sum(axis=0) < X_corr.shape[1]-1
    print(col_na)
    X_corr_val = X_corr.loc[col_na][col_na]
    print(X_corr_val, X_corr_val.shape)
    # X_corr.dropna(how='all', axis=0, inplace=True)
    # X_corr = X_corr[X_corr.index.tolist()]
    print(X_corr.shape)
    print(X_corr[:10])
    corr_sorted = X_corr.unstack().sort_values(kind='quicksort', ascending=False)
    print('Sorted Correlation values:\n{}'.format(corr_sorted[:10]))
    pairs_to_drop = set()
    for i in range(0, X_train.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((X_train.columns[i], X_train.columns[j]))
cweights = train_y.replace({1: 1, 0: 10})
X_train_robust = sklearn.preprocessing.RobustScaler(quantile_range=(0.05, 0.95), unit_variance=False).fit_transform(X_train)
X_train_robust_unit = sklearn.preprocessing.RobustScaler(quantile_range=(0.05, 0.95), unit_variance=True).fit_transform(X_train)
X_train_normal = sklearn.preprocessing.Normalizer().fit_transform(X_train)
linear_dim_reduction = False
if linear_dim_reduction:
    if dense_df.shape[1] < 50:
        score_f = ElasticNetCV(n_jobs=n_parallel, random_state=0, max_iter=5000, tol=1e-5, cv=3)
    else:
        score_f = LassoCV(fit_intercept=False, max_iter=2500, tol=1e-5, n_jobs=n_parallel, random_state=0, cv='balanced')
    selector = sklearn.feature_selection.SequentialFeatureSelector(estimator=score_f, scoring='balanced_accuracy', n_jobs=n_parallel)
# selector = sklearn.feature_selection.SelectFromModel(estimator=score_f, max_features=n_feats, prefit=True, threshold='median')
# Model running options.
plot_tsne = True
plot_logreg = False
forest = True
n_parallel = 4
feat_group_names = ['Sparse', 'Dense', 'Matrix-based', 'E-States', 'All Features']
feat_selector_bool = [False, False, True, True, True]
selector_dict = dict(zip(feat_group_names, feat_selector_bool))
feat_col_list = [sparse_df, dense_df, matrix, estate, X_train.columns]
for i, feat_cols in zip(feat_group_names, feat_col_list):
    dense_in = [c for c in feat_cols if c in dense_df.columns.to_list()]
    sparse_in = [c for c in feat_cols if c in sparse_df.columns.to_list()]
    if len(feat_cols) > 50 and not feat_cols.equals(dense_df.columns) and not feat_cols.equals(sparse_df.columns):
            best_feats = pd.concat([selector.fit_transform(X_train_robust_unit[dense_in], train_y), X_train[sparse_in]])
    elif len(feat_cols) > 50 and feat_cols.equals(dense_df.columns):
        best_feats = selector.fit_transform(X_train_robust_unit[dense_in])
    else:
        best_feats = X_train_robust_unit
    from sklearn.metrics import balanced_accuracy_score, RocCurveDisplay
    # TODO: Implement hovertext, opacity,
    if plot_tsne:
        from sklearn.manifold import TSNE, LocallyLinearEmbedding
        sne = TSNE(n_jobs=n_parallel, early_exaggeration=8.0, n_iter=1500, verbose=1, random_state=0)
        mapped_comps = sne.fit_transform(X=best_feats[dense])
        plotdata = total_meta_df
        hover_df = plotdata[['SMILES_QSAR', 'DATA_SOURCE', 'INCHI_KEY']]
        fig = scatter(data_frame=plotdata, x='X', y='y', color='DMSO_SOLUBILITY', symbol='DATA_SOURCE', opacity=0.2, marginal_x='violin', marginal_y='violin')
        fig.update_traces(
            fillcolor="rgb(0, 0, 0)",
            customdata=hover_df,
            hovertemplate =
                        "SMILES: <b>%{customdata[0]}</b><br>" +
                        "Data Source: %{customdata[1]}<br>" +
                        "INCHI Key: %{customdata[2]}" +
                        "<extra></extra>",
        )
        fig.show(validate=True)
        print('Saving t-SNE image...')
        fig.write_image(file='tsne_8_1500_{}_marginal.png'.format(i), format='png')
        fig.write_html(file='tsne_8_1500_{}_marginal.html'.format(i))
    if plot_logreg:
        from sklearn.linear_model import LassoCV, RidgeClassifierCV, ElasticNetCV
        logcv = RidgeClassifierCV(scoring='balanced_accuracy', max_iter=2500, class_weight='balanced', n_jobs=n_parallel, random_state=0)
        for lname, lin_clf in [('LogisticCV', logcv)]:
            lin_clf.fit(X=best_feats, y=train_y)
            print('{} CV Values: {}'.format(lname, lin_clf.cv_values_))
            print('{} Coefficients: {}'.format(lname, lin_clf.coef_))
            print('{} Alpha: {}'.format(cname, lin_clf.alpha_))
            rocd = RocCurveDisplay.from_predictions(y_true=eval_y, y_pred=eval_y_pred, ax=sub, plot_chance_level=True,
                                                    name=lname)
            rocd.plot()
    if forest:
        report_list = list()
        import matplotlib.pyplot as plt
        from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.metrics import ConfusionMatrixDisplay, pair_confusion_matrix
        for train, test in StratifiedKFold().split(best_feats, y=train_y):
            dev_X, eval_X = best_feats[train, :], best_feats[test, :]
            dev_y, eval_y = train_y[train], train_y[test]
            params = dict()
            rbc = RUSBoostClassifier(random_state=0)
            brf = BalancedRandomForestClassifier(n_estimators=50, max_depth=10000, random_state=0, verbose=0, bootstrap=False, sampling_strategy='majority', n_jobs=n_parallel)
            # cv_res = cross_validate(estimator=clf, X=best_feats, y=train_y.ravel(), scoring='balanced_accuracy', cv=5, n_jobs=n_parallel, params=params)
            # cv_res = cross_val_score(estimator=clf, X=best_feats, y=train_y, scoring='balanced_accuracy', n_jobs=n_parallel)
            sub, fig = plt.subplots()
            for cname, clf in [('Random Undersampling Boost', rbc), ('Balanced RF', 'brf')]:
                clf.fit(X=dev_X, y=dev_y)
                eval_y_pred = clf.predict(eval_X)
                report = imblearn.metrics.classification_report_imbalanced(y_true=eval_y, y_pred=eval_y_pred, target_names=('Insoluble', 'Soluble'))
                report_list.append(report)
                rocd = RocCurveDisplay.from_predictions(y_true=eval_y, y_pred=eval_y_pred, ax=sub, plot_chance_level=True, name=cname)
                print('ROC-AUC for {}: {}'.format(cname, rocd.roc_auc))
                rocd.plot(ax=sub, name=cname, plot_chance_level=True)
            fig.show()
        [print(a) for a in report_list]
'''
from rdkit.DataStructs import cDataStructs

butina_clusters = Butina.ClusterData(data=dist_mat, nPts=total_meta_df.shape[0], distThresh=0.6, isDistData=True,
                                     reordering=True)
[print(len(a) for a in butina_clusters)]
butina_path = "{}circ_fingerprints/BUTINA_CLUSTERS.pkl".format(final_dir)
with open(butina_path, 'wb') as f:
    pickle.dump(file=f, obj=(train, test))
'''

from sklearn.manifold import TSNE
'''
intercols = new_desc_list[0].columns.intersection(new_desc_list[1].columns)
for dft in new_desc_list[2:]:
    intercols = intercols.intersection(dft.columns)
if 1443 in intercols:
    raise KeyError
''
df_pairs, ks_list = list(), list()
# Calculate KS statistic and p-value for each column in each pair of DFs.

for df1_meta, df2_meta in itertools.combinations(new_desc_list, r=2):
    if df1_meta.source != df2_meta.source and df1_meta.dmso_soluble != df2_meta.dmso_soluble:
        print('Nonmatching pair found\n\n.')
        continue
    # df1 = pd.DataFrame(df1_meta['DESCRIPTORS'].to_list(), columns=COL_LABELS)
    df1 = df1_meta[intercols]
    df2 = df2_meta[intercols]
    df_pairs.append((df1, df2))
ks_pvals = descriptor_preprocessing.ks_stats(dfs=new_desc_list, df_pairs=df_pairs, use_cols=intercols)
ks_list.append(pd.DataFrame.from_dict(ks_pvals, orient='index'))
ks_results = pd.concat(ks_list, axis=1).T
# Nested dict col -> (pair, results)
from scipy.stats import pmean

pair_dict = dict(enumerate(df_pairs))
# ks_summary = pd.DataFrame(index=list(pair_dict.keys()), columns=intercols)
from scipy.stats import pmean

# ks_summary = pd.DataFrame(index=df_pairs, columns=COL_LABELS).reset_index(inplace=True, names='df_pairs')
for ks_col in ks_results.columns:
    ks_ser = ks_results[ks_col]
    # Optimization Possible by only checking one value/df_pairs pair
    for pair_key in pair_dict.keys():
        # pair_key = [key for key, value in pair_dict.items() if all([x.equals(y) for x, y in zip(value, df_pairs)])]
        if pair_key and ks_col and type(pair_key) is not list:
            ks_summary.loc[pair_key, ks_col] = ks_ser
        else:
            print('Pair_key is an empty list.')
            ks_summary[pair_key, ks_col] = 1.0

ks_summary = pd.DataFrame(columns=ks_results.columns)
ks_results.drop_duplicates(subset=ks_results.columns[:20], inplace=True)
for col_name, ser in ks_results.items():
    if type(ser[1]) is np.float64:
        ks_summary[col_name] = [pmean(x[1], p=2) for col_name, x in ks_results.items()]
    elif type(ser[1]) is pd.Series:
        ks_summary[col_name] = [pmean(x[1], p=2) for col_name, x in ks_results.items()]
# for cname, tcol in ks_results.items():
#    ks_summary[cname] = pmean(tcol.drop_duplicates().tolist(), p=2)
# sorted_cols = ks_summary.max(axis='rows')
# sorted_cols.dropna(inplace=True)
# print(sorted_cols.shape)
sorted_cols = ks_summary.sort_values().squeeze()
if len(sorted_cols.shape) < 2 or sorted_cols.shape[1] <= 0:
    raise ValueError
print(sorted_cols)
ks_results.var(axis='columns', numeric_only=True)
descriptor_preprocessing.kde_grouped(dfs=new_desc_list, cols=sorted_cols[:100])

# Calculate weighted geometric mean.
# Sort columns by WGM. Plot KDEs of columns.
# Add mutual info/F-score to KDEs.
#
pmeans = dict()
for col, pvals in ks_summary.items():
    try:
        pmeans[col] = hmean(pvals.to_numpy())
    except:
        print(pvals)
sorted_cols = pd.Series(pmeans)
print(sorted_cols.sort_values())
descriptor_preprocessing.kde_grouped(dfs=new_desc_list, cols=sorted_cols.sort_values()[:100])
'''
# pd.DataFrame(df['DESCRIPTORS'].to_list(), columns=COL_NAMES)
