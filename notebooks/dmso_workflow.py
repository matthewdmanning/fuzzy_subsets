# from chefboost import Chefboost
import pandas as pd
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
import utils
from data_handling.data_tools import load_training_data
from data_handling.persistence import logger
import cv_tools

train_X, train_y, total_meta_df = load_training_data()
assert not train_y.empty and not train_X.empty
# Sample Clustering
#


misc_cols = ["Number of hydrogen bond donors (using CDK HBondDonorCountDescriptor algorithm)",
             'Molecular weight', 'Number of rotatable bonds, '
                                 'excluding terminal '
                                 'bonds',
             "Weiner polarity number", 'Molecular path count of order 9']
lipinki_col = [c for c in train_X.columns if 'five' in c.lower() and 'rule' in c.lower()]
mol_wt_col = [c for c in train_X.columns if 'molecular weight' in c.lower()]
logp = [c for c in train_X.columns if 'logp' in c.lower()]
rotate = [c for c in train_X.columns if 'rotat' in c.lower()]
# print(mol_wt_col, logp, rotate)
# train_y = FramLabelBinarizer().fit_transform(y=train_y.astype(int))
fused = [c for c in train_X.columns if
         ('rings' in c.lower() and 'fused' in c.lower() and not 'hetero' in c.lower()) and 'membered' in c.lower()]
plain_hetero = [c for c in train_X.columns if (
        'rings' in c.lower() and not 'fused' in c.lower() and 'hetero' in c.lower()) and 'membered' in c.lower()]
all_rings = [c for c in train_X.columns if 'rings' in c.lower() and 'membered' in c.lower()]
path_cols = [c for c in train_X.columns if 'Molecular walk count' in c or 'Molecular path count' in c]
bo_cols = [c for c in train_X.columns if 'Conventional bond order' in c]
names = {'Fused rings':     fused, 'Hetero plain': plain_hetero, 'Hetero fused': all_rings,
         'Path_Counts_PCA': path_cols, 'Bond_Oorders_PCA': bo_cols}
cols = ['Molecular path count of order {}'.format(i) for i in
        range(5, 11)]  # .append('Total path count (up to order 10)')
mi_df = pd.DataFrame(index=list(range(5, 11)), columns=list(range(5, 11)))
# for i in (list(range(6))):
pca_replacers = ['Bond_Orders_PCA', 'Path_Counts_PCA']
cv_indices = cv_tools.get_split_ind(train_X, labels=train_y)

var_thresh = 0.9
for dev, eva in cv_indices:
    dev_X = train_X.iloc[dev]
    dev_y = train_y.iloc[dev]
    eva_X = train_X.iloc[eva]
    eva_y = train_y.iloc[eva]
    threshold = 0.025
    over_X, over_y = RandomOverSampler(random_state=0).fit_resample(dev_X, dev_y)
    for feats in pca_replacers:
        improvement = 1
        feats = utils.features.sort_ordinals(feats)
        subset = list()
        [subset.append(feats.pop(0)) for _ in range(2)]
        for feat in feats:
            subset.append(feat)
            # noinspection PyArgumentEqualDefault
            feature_pca = PCA(n_components=1, whiten=False).fit(X=over_X[subset], y=over_y)
            logger.info('{} PCA')
            pca_dev_X = feature_pca.transform(dev_X[subset])
            pca_eva_X = feature_pca.transform(eva_X[subset])
            logger.info('Explained Variance Ratio: {}'.format(feature_pca.explained_variance_ratio_))
            logger.info('Principal Components}: {}'.format(feature_pca.components_))
            if feature_pca.explained_variance_ratio_ < var_thresh:
                subset.pop()
    '''    
    mi_list = list()
    for p in range(10):
        new_X, y = RandomUnderSampler().fit_resample(train_X, y=train_y)
        mi = mutual_info_regression(X=new_X[cols[i]].to_frame(), y=new_X[cols[j]],
                                    discrete_features=False, random_state=0)
        mi_list.append(mi)
    pprint.pp('{:.5f} ({:.5f})    {}    {}'.format(np.mean(mi_list), np.std(mi_list), i, j))
    mi_df.iloc[i, j] = np.mean(mi_list)
    '''
print(mi_df.to_string(max_colwidth=10))

exit()
print(condition_X.columns)
cats = condition_X.columns
alpha = 1 - (train_y.to_numpy() / 1.25)
print(alpha)
# condition_X = StandardScaler().fit_transform(condition_X.map(asinh))
# condition_X = RobustScaler().fit_transform(condition_X)
condition_X['sol'] = train_y
# fig = px.histogram(condition_X, x='Molecular path count of order 9', color='sol', marginal='violin')
fig = px.scatter(condition_X.sample(n=1000), x='Molecular path count of order 9', y='Weiner polarity number',
                 color='sol',
                 opacity=0.1, size_max=0.5)
'''fig = px.scatter_matrix(condition_X, dimensions=cats.tolist(), color=train_y.divide(2),
                        color_continuous_scale='Tropic',
                        opacity=alpha, size_max=0.25)'''
fig.show()
plt.savefig(filename='{}tsne_simple_est.png'.format(os.environ.get('MODELS_DIR')))
plt.show()
exit()
["Number of rings (includes counts from fused rings)", "Crippen's LogP"]
sne = TSNE(n_jobs=-1, perplexity=50)
sne_X = sne.fit_transform(X=condition_X)
sne_X.to_pickle('{}tsne_simple_est.pkl'.format(os.environ.get('MODELS_DIR')))

sneplot = plt.scatter(x=sne_X.iloc[:, 0].to_numpy(), y=sne_X.iloc[:, 1].to_numpy(), alpha=alpha, c=train_y.to_numpy(),
                      s=2)
fig.show()
plt.show()
print(sne.kl_divergence_)
exit()
working_notebook.train_models(train_X, train_y, total_meta_df)

# Calculate performance metrics for CV rounds of Dev and eva sets for all models.
'''
score_results = dict([(s, dict()) for s in score_dict.keys()])
for mod_name, mod in results.items():
    score_results[mod_name] = dict()
    print(mod)
    if mod_name != 'true':
        for split_name, predicts in mod.items():
            score_list = list()
            print(predicts)
            for score_name, scorer in score_dict.items():
                for cv_num in range(len(predicts)):
                    score_kwargs = {'sample_weight': weights['eva'][cv_num], 'labels': [insol_label, sol_label],
                                    'pos_label':     insol_label}
                    print(scorer(y_true=results['true'][split_name][cv_num],
                                 y_pred=results[mod_name][split_name][cv_num]))
                    score_list.append(scorer(y_true=results['true'][split_name][cv_num],
                                             y_pred=results[mod_name][split_name][cv_num]))
                if len(score_list) > 0:
                    score_results[mod_name][split_name] = (np.mean(score_list), np.std(score_list))
                    logger.info(
                        '{} scores for {} {}: Mean: {:.4}, St Dev: {:.4}'.format(score_name, mod_name, split_name,
                                                                                 np.mean(score_list),
                                                                                 np.std(score_list)))
                else:
                    print(score_name)
print('Failed to calculate scores post-hoc.')

cd_df, axs = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
cd_df.set_layout_engine(layout='constrained')
cd_df.set_dpi(600)
for mod_name, mod in results.items():
for score_name, scorer in score_dict.items():
    for split_name, cv_list in mod.items():
        for cv_num in cv_list:
            conf_disp = get_confusion_display(y_true=results['true'][split_name][int(cv_num)], estimator=models[mod_name][split_name][cv_num], class_names=['Insoluble', 'Soluble'])
try:
    rocax.set_xticks(np.array([0, 1]), labels=['Insoluble', 'Soluble'])
    rocax.set_yticks(np.array([0, 1]), labels=['Insoluble', 'Soluble'])

    # Create colorbar
    cbar = rocax.figure.colorbar(im, ax=rocax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=0, va="top")
    rocax.tick_params(top=False, bottom=False,
                      labeltop=True, labelbottom=False)
    plt.setp(rocax.get_xticklabels(), rotation=30, ha="left",
             rotation_mode="anchor")
    try:
        roc_plt.figure_.savefig('{}_.png'.format(cmd_stub), dpi=600, format='png', transparent=True)
    except:
        print('Failed to save confusion matrix!')
except:
    print('Failed to modify confusion matrix!')
'''

# Highest VIF/Condition_Num -> Eliminate n feats -> Reevauate models -> Continue until {stop} features.
