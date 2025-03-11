import itertools

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from plotly import subplots as sp
from sklearn.feature_selection import mutual_info_classif


def mutual_info_rus_(feature_df, labels):
    print(labels)
    for name, cols in names.items():
        if len(cols) == 0:
            print("{} has no columns!".format(name))
            continue
        ncols = int(np.floor(len(cols) / 2))
        nrows = int(np.ceil(len(cols) / ncols))
        fig = sp.make_subplots(rows=nrows, cols=ncols, shared_xaxes="columns")
        print(fig)
        labeled_cols = pd.concat(
            [feature_df[cols], labels], axis=1
        )  # .rename(columns=labels)'
        sol_labeled = labeled_cols[labeled_cols["Solubility"] == 1]
        insol_labeled = labeled_cols[labeled_cols["Solubility"] == 0]
        vi = sns.boxplot(
            data=labeled_cols[cols]
        )  # , color='Solubility')  # , row=i, col=j)
        vi.imshow()
        for i, j in itertools.product(range(nrows), range(ncols)):
            if i + j >= len(cols):
                continue
            print(labeled_cols[cols[i + j]])
        labels = cols.copy()
        # plt.savefig(fig.to'{}{}_dist.svg'.format(os.environ.get('MODELS_DIR'), name))
        # fig.write_image('{}{}_dist.svg'.format(os.environ.get('MODELS_DIR'), name))
        continue
        print(labeled_cols)
    exit()
    if True:
        px.histogram(data_frame=labeled_cols, x=cols, color="Solubility")

        selected_X = feature_df[cols]
        evr_list, mi_list = list(), list()
        for i in range(1):
            new_X, new_y = RandomUnderSampler().fit_resample(X=selected_X, y=labels)
            print("Total Sample: {}".format(new_y.size))
            print("Soluble Sample: {}".format(new_y[new_y == 1].size))
            print("Inoluble Sample: {}".format(new_y[new_y == 0].size))
            mi_list.append(
                pd.Series(
                    mutual_info_classif(
                        X=new_X[cols], y=new_y, discrete_features=True, n_jobs=-1
                    ),
                    index=cols,
                )
            )
            # pc = PCA(n_components=None, whiten=True)
            # X_pc = pc.fit_transform(new_X)
            # evr_list.append(pd.Series(pc.explained_variance_ratio_, index=cols))
        # evrdf = pd.concat(evr_list, axis=1)
        mi_df = pd.concat(mi_list, axis=1)
        mi_mean = mi_df.mean(axis=1)
        mi_std = mi_df.std(axis=1)
        # evmean = evrdf.mean(axis=1)
        # evstd = evrdf.std(axis=1)
        # print('Explained Variance Ratios (Mean, Std) from PCA = 10x RUS')
        # [print('{:.6f}, {:.6f}'.format(m, s)) for m, s in zip(evmean.tolist(), evstd.tolist())]
        print(name, "Mutual Information: Mean and Std from 10x RUS")
        [
            print("{}, {:.6f}, {:.6f}".format(col, mi, mistd))
            for col, mi, mistd in zip(cols, mi_mean.tolist(), mi_std.tolist())
        ]

        labeled_cols = pd.concat([feature_df[names[name]], labels]).columns.set_names(
            names=list(names.keys()).append("Solubility")
        )
        print(labeled_cols.head())
        # px.bar(data_frame=labeled_cols, x=)
    """g = Plot()
    sns.set_theme(style="ticks", font_scale=1, rc={"figure.dpi": 300, })
    ax = sns.scatterplot(data=evmean, s=25)
    ax.set(ylabel='Explained Variance Ratio')
    # ax.set(xlabel='n-th Component')
    # ax.set(title='Conventional Bond Order Features')
    ax.set(title='Conventional Bond Order Features')
    g.label(title='Conventional Bond Order Features')
    ax.set(xticks=[0, 5, 10, 15])
    # ax.despine(trim=True)
    sns.despine(top=True, right=True)
    g.plot()
    g.save(loc='{}pca_bond_order_rus10.png'.format(os.environ.get('MODELS_DIR')))
    # ax.set_title(label='PCA of Conventional Bond Order Features')
    exit()
    plt.errorbar(yerr=evstd[:10], capsize=2, barsabove=True)
    plt.title('PCA of Molecular Walk/Path Counts')
    plt.ylabel('Explained Variance Ratio')"""


def cv_boxplots(cv_folds, feature_subset="auto", max_features=10):
    import matplotlib.pyplot as plt
    import plotly.express as px
    from utils import distributions

    if feature_subset == "auto":
        cv_means = distributions.ks_feature_tests(cv_folds).index
    elif feature_subset == "all":
        cv_means = cv_means = distributions.ks_feature_tests(cv_folds).index
    else:
        cv_means = feature_subset
    if type(max_features) is int:
        cv_means = cv_means[:max_features]
    for col in cv_means:
        cv_data = pd.concat([df[col] for df in cv_folds], axis=1)
        px.box(cv_data)
        plt.show()
